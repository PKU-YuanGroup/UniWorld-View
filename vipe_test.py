from pathlib import Path
import sys
import hydra
import torch
import cv2

# 将 extern 作为优先导入路径，这样 `import vipe` 会使用 extern/vipe 拷贝
sys.path.insert(0, str(Path(__file__).parent / "extern/vipe"))

from vipe import get_config_path, make_pipeline  # noqa: E402
from vipe.streams.tensor_stream import TensorVideoStream  # noqa: E402


def load_video_as_tensor(path: Path) -> tuple[torch.Tensor, float]:
    vcap = cv2.VideoCapture(str(path))
    if not vcap.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")

    fps = float(vcap.get(cv2.CAP_PROP_FPS))
    frames: list[torch.Tensor] = []

    while True:
        ret, frame = vcap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(torch.from_numpy(frame))  # uint8, (H,W,3)

    vcap.release()

    if len(frames) == 0:
        raise RuntimeError(f"No frames decoded from: {path}")

    video = torch.stack(frames, dim=0)  # (T,H,W,3), uint8
    return video, fps


with hydra.initialize_config_dir(config_dir=str(get_config_path()), version_base=None):
    cfg = hydra.compose(
        "default",
        overrides=[
            "pipeline=default",
            "pipeline.post.depth_align_model=adaptive_unidepth-l_vda",
            "pipeline.output.save_viz=true",
            "pipeline.output.save_artifacts=false",
            "pipeline.init.instance=null",
            "pipeline.output.path=output",
        ],
    )

p = make_pipeline(cfg.pipeline)
p.return_output_streams = True

video_path = Path("./test/videos/1.mp4")
video_tensor, fps = load_video_as_tensor(video_path)

s = p.run(TensorVideoStream(video_tensor, fps=fps, name="dog_example")).output_streams[0]

depths = torch.stack([f.metric_depth for f in s])  # [T,H,W]
Ks = torch.stack([f.intrinsics for f in s])        # [T,4+D]
Twcs = torch.stack([f.pose.matrix() for f in s])   # [T,4,4]

print("Depths shape:", tuple(depths.shape))
print("Intrinsics shape:", tuple(Ks.shape))
print("Extrinsics(4x4) shape:", tuple(Twcs.shape))
