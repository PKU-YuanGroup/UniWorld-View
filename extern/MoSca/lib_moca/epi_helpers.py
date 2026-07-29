# Copyright (c) Meta Platforms, Inc. and affiliates.
import cv2
import imageio
import numpy as np
import torch
from tqdm import tqdm
from multiprocessing.dummy import Pool
import multiprocessing
import os, os.path as osp
import json, logging, time
from matplotlib import cm


# cpu multi thread with return values
def compute_sampson_error(x1, x2, F):
    """
    :param x1 (*, N, 2)
    :param x2 (*, N, 2)
    :param F (*, 3, 3)
    """
    h1 = torch.cat([x1, torch.ones_like(x1[..., :1])], dim=-1)
    h2 = torch.cat([x2, torch.ones_like(x2[..., :1])], dim=-1)
    d1 = torch.matmul(h1, F.transpose(-1, -2))  # (B, N, 3)
    d2 = torch.matmul(h2, F)  # (B, N, 3)
    z = (h2 * d1).sum(dim=-1)  # (B, N)
    err = z**2 / (d1[..., 0] ** 2 + d1[..., 1] ** 2 + d2[..., 0] ** 2 + d2[..., 1] ** 2)
    return err


def epi_thread(send_end, i, j, uv1_normalzied, uv2_normalized, covis_mask, F=None):
    x1_raw = uv1_normalzied[covis_mask][..., :2]
    x2_raw = uv2_normalized[covis_mask][..., :2]
    
    x1_np = np.ascontiguousarray(x1_raw.detach().cpu().numpy(), dtype=np.float32)
    x2_np = np.ascontiguousarray(x2_raw.detach().cpu().numpy(), dtype=np.float32)

    F_np, inlier_np = None, None
    is_fallback = False

    if F is None:
        if len(x1_np) >= 8:
            try:
                F_np, inlier_np = cv2.findFundamentalMat(x1_np, x2_np, cv2.FM_RANSAC, 0.002)
            except:
                F_np, inlier_np = None, None
    else:
        F_np = F.detach().cpu().numpy() if isinstance(F, torch.Tensor) else F
        inlier_np = np.ones(len(x1_np), dtype=bool)

    if F_np is None or inlier_np is None or inlier_np.sum() < 8:
        dist = np.linalg.norm(x1_np - x2_np, axis=-1)
        if np.median(dist) < 0.01:
            is_fallback = True
            F_torch = torch.zeros(3, 3)
            inlier_torch = torch.from_numpy(dist < 0.005).to(torch.bool)
            epi_error = torch.from_numpy(dist).float()
        else:
            F_torch = torch.zeros(3, 3)
            inlier_torch = torch.zeros(len(x1_raw), dtype=torch.bool)
            epi_error = torch.ones(len(x1_raw), dtype=torch.float32) * 1e10
    else:
        F_torch = torch.from_numpy(F_np.astype(np.float32))
        inlier_torch = torch.from_numpy(inlier_np.flatten()).to(torch.bool)
        epi_error = compute_sampson_error(x1_raw, x2_raw, F_torch)

    ret_epi = torch.full_like(covis_mask, -1.0, dtype=torch.float32)
    ret_epi[covis_mask] = epi_error.to(ret_epi.device)
    
    ret_inlier = torch.zeros_like(covis_mask, dtype=torch.bool)
    if len(inlier_torch) == covis_mask.sum():
        ret_inlier[covis_mask] = inlier_torch.to(ret_inlier.device)

    send_end.send({
        "i": i,
        "j": j,
        "F": F_torch,
        "padded_inlier": ret_inlier,
        "inlier_ratio": inlier_torch.float().mean() if len(inlier_torch) > 0 else torch.tensor(0.0),
        "inlier_count": inlier_torch.sum() if len(inlier_torch) > 0 else torch.tensor(0),
        "epi_error": ret_epi,
    })

@torch.no_grad()
def analyze_track_epi(pair_list, track, track_mask, H, W, F_list=None):
    logging.info(f"Start analyzing {len(pair_list)} pairs")
    start_t = time.time()
    # track: tensor, (T, N, 2)
    # track_mask: tensor, (T, N)
    # H: int, height, W: int, width
    track_noramlized = track.clone().cpu().float()
    track_noramlized[..., 0] = 2.0 * track_noramlized[..., 0] / W - 1.0
    track_noramlized[..., 1] = 2.0 * track_noramlized[..., 1] / H - 1.0

    # * parallel solve
    jobs = []
    pipe_list = []
    for _ind in range(len(pair_list)):
        i, j = pair_list[_ind]
        recv_end, send_end = multiprocessing.Pipe(False)
        if F_list is not None:
            F = F_list[_ind]
        else:
            F = None
        args = (
            send_end,
            i,
            j,
            track_noramlized[i].detach().cpu().clone(),
            track_noramlized[j].detach().cpu().clone(),
            (track_mask[i] & track_mask[j]).cpu().clone(),
            F,
        )
        p = multiprocessing.dummy.Process(target=epi_thread, args=args)
        jobs.append(p)
        pipe_list.append(recv_end)
        p.start()
    for proc in jobs:
        proc.join()

    # * collect results
    result_list = [x.recv() for x in pipe_list]
    result_dict = {(it["i"], it["j"]): it for it in result_list}
    F_list, epi_error_list, inlier_list = [], [], []
    # todo: here may also use the inlier count to improve the robustness
    for pair in pair_list:
        sol = result_dict[pair]
        inlier_count = sol["inlier_count"]
        F = sol["F"]
        epi_error = sol["epi_error"]
        inlier = sol["padded_inlier"]
        F_list.append(F)
        epi_error_list.append(epi_error)
        inlier_list.append(inlier)

    F_list = torch.stack(F_list, dim=0)
    epi_error_list = torch.stack(epi_error_list, dim=0)
    inlier_list = torch.stack(inlier_list, dim=0)
    run_time = time.time() - start_t
    logging.info(f"Finished epi for {len(pair_list)} pairs, time: {run_time:.2f}s")
    return F_list, epi_error_list, inlier_list


def compute_track_epi(pair_list, track, track_mask, F_list, H, W):
    logging.info(f"Start analyzing {len(pair_list)} pairs")
    start_t = time.time()
    # track: tensor, (T, N, 2)
    # track_mask: tensor, (T, N)
    # H: int, height, W: int, width
    track_noramlized = track.clone().cpu()
    track_noramlized[..., 0] = 2.0 * track_noramlized[..., 0] / W - 1.0
    track_noramlized[..., 1] = 2.0 * track_noramlized[..., 1] / H - 1.0

    # * parallel solve
    jobs = []
    pipe_list = []
    for i, j in pair_list:
        recv_end, send_end = multiprocessing.Pipe(False)
        args = (
            send_end,
            i,
            j,
            track_noramlized[i],
            track_noramlized[j],
            (track_mask[i] & track_mask[j]).cpu(),
        )
        p = multiprocessing.dummy.Process(target=epi_thread, args=args)
        jobs.append(p)
        pipe_list.append(recv_end)
        p.start()
    for proc in jobs:
        proc.join()

    # * collect results
    result_list = [x.recv() for x in pipe_list]
    result_dict = {(it["i"], it["j"]): it for it in result_list}
    F_list, epi_error_list, inlier_list = [], [], []
    # todo: here may also use the inlier count to improve the robustness
    for pair in pair_list:
        sol = result_dict[pair]
        inlier_count = sol["inlier_count"]
        F = sol["F"]
        epi_error = sol["epi_error"]
        inlier = sol["padded_inlier"]
        F_list.append(F)
        epi_error_list.append(epi_error)
        inlier_list.append(inlier)

    F_list = torch.stack(F_list, dim=0)
    epi_error_list = torch.stack(epi_error_list, dim=0)
    inlier_list = torch.stack(inlier_list, dim=0)
    run_time = time.time() - start_t
    logging.info(f"Finished epi for {len(pair_list)} pairs, time: {run_time:.2f}s")
    return F_list, epi_error_list, inlier_list

 
def identify_tracks(epierr_list, epi_th, static_cnt=1, dynamic_cnt=1):
    over_th_cnt = (epierr_list > epi_th).sum(0)
    dym_mask = over_th_cnt >= dynamic_cnt
    sta_mask = over_th_cnt < static_cnt
    logging.warning(
        f"Identify {sta_mask.sum()} static, {dym_mask.sum()} dynamic outof {len(sta_mask)} tracks"
    )
    return sta_mask, dym_mask
