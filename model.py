from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import torch
from torch import nn

from dataset import BrightSpotVideoDataset, VideoFromImagesConfig


@dataclass
class TemporalCompressorConfig:
    """Configuration for :class:`TemporalCompressionModel`."""

    temporal_kernel: int = 5
    softmax_temperature: float = 1.0
    eps: float = 1e-8


class TemporalSoftmaxPool(nn.Module):
    """Produces a softmax distribution across time steps."""

    def __init__(self, temporal_kernel: int, temperature: float = 1.0) -> None:
        super().__init__()
        if temporal_kernel % 2 == 0:
            raise ValueError("Temporal kernel must be odd for same-length conv1d.")
        self.temperature = temperature
        self.conv = nn.Conv1d(1, 1, temporal_kernel, padding=temporal_kernel // 2, bias=False)
        with torch.no_grad():
            self.conv.weight.zero_()
            self.conv.weight[..., temporal_kernel // 2] = 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(f"Expected 2D tensor [B,T], got {tuple(x.shape)}")
        scores = self.conv(x.unsqueeze(1)).squeeze(1)
        if self.temperature != 1.0:
            scores = scores / self.temperature
        return torch.softmax(scores, dim=-1)


class TemporalCompressionModel(nn.Module):
    """Compress (B, T, H, W) -> (B, 1, H, W) while prioritising temporal detail."""

    def __init__(self, cfg: TemporalCompressorConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or TemporalCompressorConfig()
        self.temporal_pool = TemporalSoftmaxPool(self.cfg.temporal_kernel, self.cfg.softmax_temperature)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected 4D tensor [B,T,H,W], got {tuple(x.shape)}")
        b, t, h, w = x.shape
        if t == 1:
            return x[:, :1]

        weights = self.temporal_pool(x.mean(dim=(2, 3)))
        weights = weights + self.cfg.eps
        weights = weights / weights.sum(dim=1, keepdim=True)
        weights = weights.view(b, t, 1, 1)
        out = (x * weights).sum(dim=1, keepdim=True)
        return out


class MosaicLayer(nn.Module):
    """PyTorch port of the MATLAB MosaicLayer for reshaping sequences."""

    def __init__(self, mosaic_size: Tuple[int, int, int], name: str | None = None) -> None:
        super().__init__()
        if len(mosaic_size) != 3:
            raise ValueError("mosaic_size must have three elements")
        self.mosaic_size = tuple(int(m) for m in mosaic_size)
        self.name = name or "mosaic"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected 4D tensor [B,T,H,W], got {tuple(x.shape)}")
        b, t, h, w = x.shape
        mt, mh, mw = self.mosaic_size
        if t % mt != 0 or h % mh != 0 or w % mw != 0:
            raise ValueError("Input dimensions must be divisible by mosaic_size")
        m_t, m_h, m_w = t // mt, h // mh, w // mw
        temp = x.view(b, mt, m_t, mh, m_h, mw, m_w)
        out = temp.permute(0, 2, 4, 1, 3, 5, 6).contiguous()
        return out.view(b, m_t * m_h, mt * mh * mw, m_w)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run temporal compression on a sample video and visualise the result.")
    parser.add_argument("--root", type=Path, required=True, help="Directory with source images for the synthetic dataset.")
    parser.add_argument("--frames", type=int, default=16, help="Number of frames to synthesise per clip.")
    parser.add_argument("--height", type=int, default=64, help="Frame height.")
    parser.add_argument("--width", type=int, default=64, help="Frame width.")
    parser.add_argument("--index", type=int, default=0, help="Dataset index to sample for visualisation.")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device for the model (e.g. cpu or cuda).")
    parser.add_argument("--save", type=Path, help="Optional path to save the comparison output (supports .gif for animation).")
    parser.add_argument("--no-show", action="store_true", help="Skip calling plt.show().")
    parser.add_argument("--interval", type=int, default=100, help="Animation frame interval in milliseconds.")
    args = parser.parse_args()

    cfg = VideoFromImagesConfig(
        root=args.root,
        frames=args.frames,
        height=args.height,
        width=args.width,
        return_ch_first=False,
        normalize=True,
    )

    dataset = BrightSpotVideoDataset(cfg)
    if len(dataset) == 0:
        raise RuntimeError(f"No images found under {args.root} to build a video from.")

    sample = dataset[args.index % len(dataset)]
    video = sample["video"].unsqueeze(0)

    device = torch.device(args.device)
    model = TemporalCompressionModel().to(device)

    with torch.no_grad():
        output = model(video.to(device)).cpu()[0, 0]

    video_np = video[0].cpu().numpy()
    output_np = output.numpy()
    vmin = float(min(video_np.min(), output_np.min()))
    vmax = float(max(video_np.max(), output_np.max()))

    import matplotlib.pyplot as plt
    from matplotlib import animation

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    input_ax, output_ax = axes

    input_ax.set_title("Input Video (animated)")
    input_ax.axis("off")
    output_ax.set_title("Compressed Output")
    output_ax.axis("off")

    im = input_ax.imshow(video_np[0], cmap="gray", vmin=vmin, vmax=vmax, animated=True)
    output_ax.imshow(output_np, cmap="gray", vmin=vmin, vmax=vmax)

    def update(frame_idx: int):
        im.set_data(video_np[frame_idx])
        return (im,)

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=video_np.shape[0],
        interval=max(1, args.interval),
        blit=True,
        repeat=True,
    )

    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        suffix = args.save.suffix.lower()
        if suffix == ".gif":
            writer = animation.PillowWriter(fps=max(1, int(1000 / max(1, args.interval))))
            anim.save(args.save, writer=writer)
        else:
            fig.savefig(args.save, dpi=200)

    if args.no_show:
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    main()
