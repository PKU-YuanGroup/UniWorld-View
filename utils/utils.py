import logging
import os

import imageio
import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation
import random


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_world_size():
    if dist.is_initialized():
        return dist.get_world_size()
    else:
        return int(os.environ.get("WORLD_SIZE", 1))


def get_rank():
    if dist.is_initialized():
        return dist.get_rank()
    else:
        return int(os.environ.get("RANK", 0))


def get_local_rank():
    if torch.cuda.device_count() == 0:
        print("WARNING: No available GPU.")
        return 0
    return get_rank() % torch.cuda.device_count()


def is_distributed():
    return os.environ.get("WORLD_SIZE", None) is not None


def is_main_process():
    return not is_distributed() or dist.get_rank() == 0


def create_logger(logging_dir=None):
    """
    Create a logger that writes to a log file and stdout.
    """
    if is_main_process():  # real logger
        additional_args = dict()
        if logging_dir is not None:
            additional_args["handlers"] = [
                logging.StreamHandler(),
                logging.FileHandler(f"{logging_dir}/log.txt"),
            ]
        logging.basicConfig(
            level=logging.INFO,
            format="[\033[34m%(asctime)s\033[0m] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            **additional_args,
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def load_video(video_path):
    reader = imageio.get_reader(video_path)
    total_frames = reader.count_frames()
    frames = []
    for i in range(total_frames):
        frame = reader.get_data(i)
        frames.append(Image.fromarray(frame))

    reader.close()

    return frames


def points_padding(points):
    padding = torch.ones_like(points)[..., 0:1]
    points = torch.cat([points, padding], dim=-1)
    return points


def np_points_padding(points):
    padding = np.ones_like(points)[..., 0:1]
    points = np.concatenate([points, padding], axis=-1)
    return points


def txt_interpolation(input_list, n, mode='smooth'):
    x = np.linspace(0, 1, len(input_list))
    if mode == 'smooth':
        f = UnivariateSpline(x, input_list, k=3)
    elif mode == 'linear':
        f = interp1d(x, input_list)
    else:
        raise KeyError(f"Invalid txt interpolation mode: {mode}")
    xnew = np.linspace(0, 1, n)
    ynew = f(xnew)
    return ynew


def traj_map(traj_type):
    # pre-defined trajectories
    if traj_type == "free1":  # Zoom out and rotate to the upper left
        cam_traj = "free"
        x_offset = 0.0
        y_offset = 0.0
        z_offset = 0.0
        d_theta = -15.0
        d_phi = 45.0
        d_r = 1.6
    elif traj_type == "free2":  # Rotate to the right horizontally
        cam_traj = "free"
        x_offset = -0.05
        y_offset = 0.0
        z_offset = 0.0
        d_theta = 0.0
        d_phi = -60.0
        d_r = 1.0
    elif traj_type == "free3":  # Move back to the left
        cam_traj = "free"
        x_offset = -0.25
        y_offset = 0.0
        z_offset = 0.0
        d_theta = 0.0
        d_phi = 0.0
        d_r = 1.7
    elif traj_type == "free4":  # Rotate and approach to the upper right
        cam_traj = "free"
        x_offset = 0.0
        y_offset = 0.0
        z_offset = 0.0
        d_theta = -15.0
        d_phi = -60.0
        d_r = 0.75
    elif traj_type == "free5":  # Large-angle camera movement to the upper right
        cam_traj = "free"
        x_offset = 0.0
        y_offset = 0.0
        z_offset = 0.0
        d_theta = -15.0
        d_phi = -120.0
        d_r = 1.6
    elif traj_type == "swing1":  # Swing shot 1
        cam_traj = "swing1"
        x_offset = 0.0
        y_offset = 0.0
        z_offset = 0.0
        d_theta = 0.0
        d_phi = 0.0
        d_r = 1.0
    elif traj_type == "swing2":  # Swing shot 2
        cam_traj = "swing2"
        x_offset = 0.0
        y_offset = 0.0
        z_offset = 0.0
        d_theta = 0.0
        d_phi = 0.0
        d_r = 1.0
    elif traj_type == "orbit":  # 360-degree counterclockwise rotation
        cam_traj = "free"
        x_offset = 0.0
        y_offset = 0.0
        z_offset = 0.0
        d_theta = 0.0
        d_phi = -360.0
        d_r = 1.0
    else:
        raise NotImplementedError
    return cam_traj, x_offset, y_offset, z_offset, d_theta, d_phi, d_r


def set_initial_camera(start_elevation, radius):
    c2w_0 = torch.tensor([[1, 0, 0, 0],
                          [0, 1, 0, 0],
                          [0, 0, 1, -radius],
                          [0, 0, 0, 1]], dtype=torch.float32)
    elevation_rad = np.deg2rad(start_elevation)
    R_elevation = torch.tensor([[1, 0, 0, 0],
                                [0, np.cos(-elevation_rad), -np.sin(-elevation_rad), 0],
                                [0, np.sin(-elevation_rad), np.cos(-elevation_rad), 0],
                                [0, 0, 0, 1]], dtype=torch.float32)
    c2w_0 = R_elevation @ c2w_0
    w2c_0 = c2w_0.inverse()

    return w2c_0, c2w_0


def build_cameras(cam_traj, w2c_0, c2w_0, intrinsic, nframe, focal_length,
                  d_theta, d_phi, d_r, radius, x_offset, y_offset, z_offset):
    # build camera viewpoints according to d_theta，d_phi, d_r
    # return: w2cs:[V,4,4], c2ws:[V,4,4], intrinsic:[V,3,3]
    if intrinsic.ndim == 2:
        intrinsic = intrinsic[None].repeat(nframe, 1, 1)

    c2ws = [c2w_0]
    w2cs = [w2c_0]
    d_thetas, d_phis, d_rs = [], [], []
    x_offsets, y_offsets, z_offsets = [], [], []
    if cam_traj == "free":
        for i in range(nframe - 1):
            coef = (i + 1) / (nframe - 1)
            d_thetas.append(d_theta * coef)
            d_phis.append(d_phi * coef)
            d_rs.append(coef * d_r + (1 - coef) * 1.0)
            x_offsets.append(radius * x_offset * ((i + 1) / nframe))
            y_offsets.append(radius * y_offset * ((i + 1) / nframe))
            z_offsets.append(radius * z_offset * ((i + 1) / nframe))
    elif cam_traj == "swing1":
        phis__ = [0, -5, -25, -30, -20, -8, 0]
        thetas__ = [0, -8, -12, -20, -17, -12, -5, -2, 1, 5, 3, 1, 0]
        rs__ = [0, 0.2]
        d_phis = txt_interpolation(phis__, nframe, mode='smooth')
        d_phis[0] = phis__[0]
        d_phis[-1] = phis__[-1]
        d_thetas = txt_interpolation(thetas__, nframe, mode='smooth')
        d_thetas[0] = thetas__[0]
        d_thetas[-1] = thetas__[-1]
        d_rs = txt_interpolation(rs__, nframe, mode='linear')
        d_rs = 1.0 + d_rs
    elif cam_traj == "swing2":
        phis__ = [0, 5, 25, 30, 20, 10, 0]
        thetas__ = [0, -5, -14, -11, 0, 1, 5, 3, 0]
        rs__ = [0, -0.03, -0.1, -0.2, -0.17, -0.1, 0]
        d_phis = txt_interpolation(phis__, nframe, mode='smooth')
        d_phis[0] = phis__[0]
        d_phis[-1] = phis__[-1]
        d_thetas = txt_interpolation(thetas__, nframe, mode='smooth')
        d_thetas[0] = thetas__[0]
        d_thetas[-1] = thetas__[-1]
        d_rs = txt_interpolation(rs__, nframe, mode='smooth')
        d_rs = 1.0 + d_rs
    else:
        raise NotImplementedError("Unknown trajectory type...")

    for i in range(nframe - 1):
        d_theta_rad = np.deg2rad(d_thetas[i])
        R_theta = torch.tensor([[1, 0, 0, 0],
                                [0, np.cos(d_theta_rad), -np.sin(d_theta_rad), 0],
                                [0, np.sin(d_theta_rad), np.cos(d_theta_rad), 0],
                                [0, 0, 0, 1]], dtype=torch.float32)
        d_phi_rad = np.deg2rad(d_phis[i])
        R_phi = torch.tensor([[np.cos(d_phi_rad), 0, np.sin(d_phi_rad), 0],
                              [0, 1, 0, 0],
                              [-np.sin(d_phi_rad), 0, np.cos(d_phi_rad), 0],
                              [0, 0, 0, 1]], dtype=torch.float32)
        c2w_1 = R_phi @ R_theta @ c2w_0
        if i < len(x_offsets) and i < len(y_offsets) and i < len(z_offsets):
            c2w_1[:3, -1] += torch.tensor([x_offsets[i], y_offsets[i], z_offsets[i]])
        c2w_1[:3, -1] *= d_rs[i]
        w2c_1 = c2w_1.inverse()
        c2ws.append(c2w_1)
        w2cs.append(w2c_1)

        intrinsic[i + 1, :2, :2] = intrinsic[i + 1, :2, :2] * focal_length * ((i + 1) / nframe) + \
                                   intrinsic[i + 1, :2, :2] * ((nframe - (i + 1)) / nframe)

    w2cs = torch.stack(w2cs, dim=0)
    c2ws = torch.stack(c2ws, dim=0)

    return w2cs, c2ws, intrinsic


def rotation_matrix_from_vectors(v1, v2):
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)

    cross_product = np.cross(v1, v2)
    dot_product = np.dot(v1, v2)
    angle = np.arccos(dot_product)

    # special issue
    if np.linalg.norm(cross_product) < 1e-10:  # if is parallel
        if dot_product > 0:
            return np.eye(3)
        else:
            arbitrary_axis = np.array([1, 0, 0]) if np.all(v1 != np.array([1, 0, 0])) else np.array([0, 1, 0])
            return Rotation.from_rotvec(np.pi * arbitrary_axis).as_matrix()

    # Rodrigues formula to get rotation
    r = Rotation.from_rotvec(angle * cross_product / np.linalg.norm(cross_product))
    return r.as_matrix()
