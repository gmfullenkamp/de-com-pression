from __future__ import annotations

import argparse
import math
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
    components: int = 2
    include_max_map: bool = True
    include_motion_map: bool = True
    eps: float = 1e-8


class TemporalScoring(nn.Module):
    """Learn attention weights over time using per-frame statistics."""

    def __init__(self, feature_dim: int, components: int, kernel_size: int, temperature: float = 1.0) -> None:
        super().__init__()
        if components < 1:
            raise ValueError("components must be >= 1")
        if kernel_size % 2 == 0:
            raise ValueError("Temporal kernel must be odd for same-length conv1d.")
        self.temperature = temperature
        padding = kernel_size // 2
        self.proj = nn.Conv1d(feature_dim, components, kernel_size, padding=padding, bias=True)
        nn.init.zeros_(self.proj.weight)
        center = kernel_size // 2
        with torch.no_grad():
            for i in range(components):
                channel = min(i, feature_dim - 1)
                self.proj.weight[i, channel, center] = 1.0

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3:
            raise ValueError(f"Expected [B,T,F], got {tuple(features.shape)}")
        feats = features.transpose(1, 2)  # [B,F,T]
        scores = self.proj(feats)  # [B,components,T]
        if self.temperature != 1.0:
            scores = scores / self.temperature
        weights = torch.softmax(scores, dim=-1)
        return weights


def _temporal_features(x: torch.Tensor) -> torch.Tensor:
    """Collect frame-level statistics that highlight transient bright spots."""
    if x.ndim != 4:
        raise ValueError(f"Expected 4D tensor [B,T,H,W], got {tuple(x.shape)}")
    b, t, h, w = x.shape
    frame_mean = x.mean(dim=(2, 3))
    frame_max = x.amax(dim=(2, 3))
    frame_std = x.var(dim=(2, 3), unbiased=False).sqrt()
    motion = torch.zeros(b, t, device=x.device, dtype=x.dtype)
    if t > 1:
        diffs = (x[:, 1:] - x[:, :-1]).abs().mean(dim=(2, 3))
        motion[:, 1:] = diffs
        motion[:, 0] = diffs[:, 0]
    features = torch.stack((frame_mean, frame_max, frame_std, motion), dim=-1)
    return features


class TemporalCompressionModel(nn.Module):
    """Compress (B, T, H, W) -> (B, 1, H, W) with transient-aware focus maps."""

    def __init__(self, cfg: TemporalCompressorConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or TemporalCompressorConfig()
        feature_dim = 4
        self.attention = TemporalScoring(
            feature_dim=feature_dim,
            components=self.cfg.components,
            kernel_size=self.cfg.temporal_kernel,
            temperature=self.cfg.softmax_temperature,
        )

    def forward(self, x: torch.Tensor, return_components: bool = False):
        if x.ndim != 4:
            raise ValueError(f"Expected 4D tensor [B,T,H,W], got {tuple(x.shape)}")
        b, t, h, w = x.shape

        if t == 1:
            base = x[:, :1].expand(-1, self.cfg.components, -1, -1).contiguous()
            weights = None
        else:
            feats = _temporal_features(x)
            weights = self.attention(feats)  # [B,components,T]
            weights = weights.unsqueeze(-1).unsqueeze(-1)  # [B,components,T,1,1]
            expanded = x.unsqueeze(1)  # [B,1,T,H,W]
            base = (expanded * weights).sum(dim=2)  # [B,components,H,W]

        base_mean = base.mean(dim=1, keepdim=True)
        base_peak = base.amax(dim=1, keepdim=True)

        candidate_maps = [base_mean, base_peak]
        viz_maps = [base]

        max_map = None
        motion_map = None
        if self.cfg.include_max_map:
            max_map = x.amax(dim=1, keepdim=True)
            candidate_maps.append(max_map)
            viz_maps.append(max_map)

        if self.cfg.include_motion_map:
            if t > 1:
                motion_map = (x[:, 1:] - x[:, :-1]).abs().amax(dim=1, keepdim=True)
            else:
                motion_map = torch.zeros(b, 1, h, w, dtype=x.dtype, device=x.device)
            candidate_maps.append(motion_map)
            viz_maps.append(motion_map)

        candidates = torch.cat(candidate_maps, dim=1)  # [B,num_candidates,H,W]
        final_map = candidates.amax(dim=1, keepdim=True)

        if not return_components:
            return final_map

        viz_tensor = torch.cat([final_map] + viz_maps, dim=1)
        aux = {
            "base_components": base,
            "weights": weights,
            "max_map": max_map,
            "motion_map": motion_map,
            "viz_tensor": viz_tensor,
        }
        return final_map, aux


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


def _channel_grid(tensor: torch.Tensor) -> torch.Tensor:
    """Arrange C feature maps into a square-ish grid for visualisation."""
    if tensor.ndim != 3:
        raise ValueError(f"Expected 3D tensor [C,H,W], got {tuple(tensor.shape)}")
    c, h, w = tensor.shape
    cols = max(1, math.ceil(math.sqrt(c)))
    rows = math.ceil(c / cols)
    grid = torch.zeros(rows * h, cols * w, dtype=tensor.dtype, device=tensor.device)
    for idx in range(c):
        r, col = divmod(idx, cols)
        grid[r * h:(r + 1) * h, col * w:(col + 1) * w] = tensor[idx]
    return grid


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
        compressed, aux = model(video.to(device), return_components=True)

    print("video:", video.shape, "compressed:", compressed.shape)

    compressed_map = compressed[0, 0].cpu()
    viz_tensor = aux["viz_tensor"][0].cpu()

    video_np = video[0].cpu().numpy()
    compressed_np = compressed_map.numpy()
    viz_np = viz_tensor.numpy()
    vmin = float(min(video_np.min(), compressed_np.min(), viz_np.min()))
    vmax = float(max(video_np.max(), compressed_np.max(), viz_np.max()))

    import matplotlib.pyplot as plt
    from matplotlib import animation

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    input_ax, output_ax = axes

    input_ax.set_title("Input Video (animated)")
    input_ax.axis("off")
    output_ax.set_title("Compressed Map + Aux Channels")
    output_ax.axis("off")

    im = input_ax.imshow(video_np[0], cmap="gray", vmin=vmin, vmax=vmax, animated=True)
    output_ax.imshow(_channel_grid(viz_tensor).numpy(), cmap="gray", vmin=vmin, vmax=vmax)

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
