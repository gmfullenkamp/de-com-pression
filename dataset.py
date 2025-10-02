from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union
import math
import random
import os

import numpy as np
from PIL import Image, ImageOps
import torch
from torch.utils.data import Dataset, DataLoader

# ============================================================
# IO helpers
# ============================================================

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def list_images(root: Union[str, Path]) -> List[Path]:
    """Recursively list image paths under root."""
    root = Path(root)
    imgs = [p for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS]
    if not imgs:
        raise FileNotFoundError(f"No images found under {root}")
    return sorted(imgs)


# ============================================================
# Schedules
# ============================================================

def triangular_radius_schedule(n_frames: int, peak: int = 5) -> List[int]:
    """Triangular 1..peak..0 padded with zeros."""
    if n_frames <= 0:
        return []
    if n_frames == 1:
        return [1]
    if n_frames == 2:
        return [1, 0]

    up = list(range(1, peak + 1))
    down = list(range(peak - 1, -1, -1))
    seq = up + down
    return (seq[:n_frames]) if len(seq) >= n_frames else (seq + [0] * (n_frames - len(seq)))


# ============================================================
# Image ops
# ============================================================

def _center_crop_to_aspect(pil_img: Image.Image | np.ndarray, target_hw: Tuple[int, int]) -> Image.Image:
    """Center-crop to match target aspect ratio (accepts PIL or ndarray)."""
    # be robust to numpy input
    if isinstance(pil_img, np.ndarray):
        if pil_img.dtype != np.uint8:
            pil_img = np.clip(pil_img, 0, 255).astype(np.uint8)
        pil = Image.fromarray(pil_img)
    else:
        pil = pil_img

    th, tw = target_hw
    target_ratio = tw / float(th)
    w, h = pil.size
    ratio = w / float(h)
    if math.isclose(ratio, target_ratio, rel_tol=1e-3):
        return pil
    if ratio > target_ratio:  # too wide
        new_w = int(round(h * target_ratio))
        left = (w - new_w) // 2
        return pil.crop((left, 0, left + new_w, h))
    # too tall
    new_h = int(round(w / target_ratio))
    top = (h - new_h) // 2
    return pil.crop((0, top, w, top + new_h))


# robust grayscale for 8/16-bit/RGB/float images → PIL "L"
def to_u8_gray(im: Image.Image) -> Image.Image:
    """Convert to uint8 grayscale with percentile normalization if needed."""
    im = ImageOps.exif_transpose(im)
    if im.mode in ("I;16", "I", "F"):
        arr = np.array(im, dtype=np.float32)
        # robust min/max using percentiles to avoid hot pixels
        lo, hi = np.percentile(arr, (1.0, 99.0))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = float(arr.min()), float(arr.max() + 1e-6)
        arr = np.clip((arr - lo) / (hi - lo + 1e-6), 0.0, 1.0)
        return Image.fromarray((arr * 255.0).astype(np.uint8))
    if im.mode == "L":
        return im
    # RGB/RGBA/… → luminance
    return im.convert("L")


def load_grayscale_resized(path: Path, out_hw: Tuple[int, int], interpolation=Image.BICUBIC) -> np.ndarray:
    """Load image -> grayscale -> aspect crop -> resize -> uint8 [H,W]."""
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        im = to_u8_gray(im)
        im = _center_crop_to_aspect(im, target_hw=(out_hw[0], out_hw[1]))
        im = im.resize((out_hw[1], out_hw[0]), interpolation)  # (W,H)
        arr = np.asarray(im)
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return arr

# ============================================================
# Synthetic generator
# ============================================================

class SyntheticVideoGenerator:
    """Make grayscale clips with fluid bg drift + non-circular bright spots."""

    def __init__(self, cfg: 'VideoFromImagesConfig') -> None:
        self.cfg = cfg

    # ---------- subpixel shift (bilinear, edge pad) ----------
    @staticmethod
    def _shift_subpx(img_u8: np.ndarray, tx: float, ty: float) -> np.ndarray:
        """Subpixel translate using torch.grid_sample."""
        h, w = img_u8.shape
        img = torch.from_numpy(np.array(img_u8, copy=True)).float().unsqueeze(0).unsqueeze(0) / 255.0  # [1,1,H,W]
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, h), torch.linspace(-1, 1, w), indexing="ij"
        )
        gx = xx - 2.0 * tx / max(1, w - 1)
        gy = yy - 2.0 * ty / max(1, h - 1)
        grid = torch.stack([gx, gy], dim=-1).unsqueeze(0)
        warped = torch.nn.functional.grid_sample(
            img, grid, mode="bilinear", padding_mode="border", align_corners=True
        )
        out = (warped.squeeze().clamp(0, 1) * 255.0).numpy().astype(np.uint8)
        return out

    # ---------- spot kernels ----------
    @staticmethod
    def _gaussian_mask(h: int, w: int, cx: float, cy: float, sx: float, sy: float, theta: float) -> np.ndarray:
        """Rotated anisotropic Gaussian mask in [0,1]."""
        y, x = np.mgrid[0:h, 0:w].astype(np.float32)
        x0, y0 = x - cx, y - cy
        ct, st = np.cos(theta), np.sin(theta)
        xr =  ct * x0 + st * y0
        yr = -st * x0 + ct * y0
        m = np.exp(-0.5 * ((xr / (sx + 1e-6)) ** 2 + (yr / (sy + 1e-6)) ** 2))
        return m

    @staticmethod
    def _superellipse_mask(h: int, w: int, cx: float, cy: float, ax: float, ay: float, p: float, theta: float) -> np.ndarray:
        """Superellipse |x/ax|^p + |y/ay|^p <= 1 -> soft mask."""
        y, x = np.mgrid[0:h, 0:w].astype(np.float32)
        x0, y0 = x - cx, y - cy
        ct, st = np.cos(theta), np.sin(theta)
        xr =  ct * x0 + st * y0
        yr = -st * x0 + ct * y0
        val = (np.abs(xr) / (ax + 1e-6)) ** p + (np.abs(yr) / (ay + 1e-6)) ** p
        m = (1.0 - np.clip(val, 0, 1))
        m = np.clip(m, 0, 1)
        return np.power(m, 0.5)

    @staticmethod
    def _multi_gaussian_mask(h: int, w: int, cx: float, cy: float, base_r: float, rng: random.Random) -> np.ndarray:
        """Sum of a few small Gaussians for irregular blobs."""
        K = rng.randint(2, 3)
        acc = np.zeros((h, w), np.float32)
        for _ in range(K):
            dx = rng.uniform(-0.3, 0.3) * base_r
            dy = rng.uniform(-0.3, 0.3) * base_r
            sx = rng.uniform(0.5, 0.9) * base_r
            sy = rng.uniform(0.5, 0.9) * base_r
            th = rng.uniform(0, np.pi)
            acc += SyntheticVideoGenerator._gaussian_mask(h, w, cx + dx, cy + dy, sx, sy, th)
        acc /= (acc.max() + 1e-6)
        return acc

    # ---------- spot render ----------
    @staticmethod
    def _blend_add_u8(img_u8: np.ndarray, mask01: np.ndarray, intensity: float) -> np.ndarray:
        """Additive blend with given intensity (0..255)."""
        out = img_u8.astype(np.float32) + mask01.astype(np.float32) * float(intensity)
        return np.clip(out, 0, 255).astype(np.uint8)

    # ---------- sequence synthesis ----------
    def build_video(self, base_u8: np.ndarray, rng: random.Random) -> Tuple[np.ndarray, dict]:
        """Create [T,H,W] with drift + 0..N bright blobs that fade in/out."""
        H, W, T = self.cfg.height, self.cfg.width, self.cfg.frames

        # background drift as smooth random walk (tunable)
        drift_x = 0.0
        drift_y = 0.0
        drift_std = float(getattr(self.cfg, 'drift_std', 0.10))   # px per frame
        drift_decay = float(getattr(self.cfg, 'drift_decay', 0.90))

        # how many spots?
        spot_prob = float(getattr(self.cfg, 'spot_probability', 0.85))
        choices = tuple(getattr(self.cfg, 'spot_count_choices', (0, 1, 2, 3)))
        if rng.random() < spot_prob:
            positive = [c for c in choices if c > 0]
            n_spots = (rng.choice(positive) if positive else 1)
        else:
            n_spots = 0

        # per-spot state
        max_r = self.cfg.max_radius
        spots = []
        for _ in range(n_spots):
            cx = rng.randint(max_r, max( max_r, W - 1 - max_r ))
            cy = rng.randint(max_r, max( max_r, H - 1 - max_r ))
            x, y = float(cx), float(cy)

            # temporal envelope (Hann burst) with random start/duration
            Tint = int(T)
            env = np.zeros((Tint,), dtype=np.float32)
            dmin = int(max(1, getattr(self.cfg, 'spot_min_frames', 3)))
            dmax_cfg = getattr(self.cfg, 'spot_max_frames', None)
            dmax = int(T if dmax_cfg is None else min(max(dmin, dmax_cfg), T))
            dur = rng.randint(dmin, dmax)
            start = rng.randint(0, max(0, T - dur))
            end = start + dur
            if dur >= 3:
                burst = np.hanning(dur).astype(np.float32)
                if burst.max() > 0:
                    burst = burst / float(burst.max())
            else:
                burst = np.ones((dur,), dtype=np.float32)
            env[start:end] = burst

            spots.append({"x": x, "y": y, "cx0": cx, "cy0": cy, "env": env})

        # strength/blur settings
        intensity_max = float(getattr(self.cfg, 'spot_intensity', 160.0))
        blur_alpha = float(getattr(self.cfg, 'motion_blur_strength', 0.35))  # 0..1
        gmin = float(getattr(self.cfg, 'gauss_sigma_min', 0.4))
        gmax = float(getattr(self.cfg, 'gauss_sigma_max', 0.8))

        frames: list[np.ndarray] = []
        for ti in range(T):
            # background drift
            drift_x = drift_decay * drift_x + rng.gauss(0, drift_std)
            drift_y = drift_decay * drift_y + rng.gauss(0, drift_std)
            bg = self._shift_subpx(base_u8, drift_x, drift_y)

            # light motion blur via neighboring subpixel offsets
            if blur_alpha > 0:
                bx = self._shift_subpx(base_u8, drift_x + 0.25, drift_y)
                by = self._shift_subpx(base_u8, drift_x, drift_y + 0.25)
                bg = ((1 - blur_alpha) * bg.astype(np.float32)
                    + (blur_alpha / 2.0) * (bx.astype(np.float32) + by.astype(np.float32))).astype(np.float32)
                bg = np.clip(bg, 0, 255).astype(np.uint8)

            # accumulate all spot contributions in float, then add once
            total_add = np.zeros((H, W), dtype=np.float32)

            for s in spots:
                # tiny random walk per spot
                s["x"] += rng.uniform(-0.3, 0.3)
                s["y"] += rng.uniform(-0.3, 0.3)
                s["x"] = float(np.clip(s["x"], max_r, W - 1 - max_r))
                s["y"] = float(np.clip(s["y"], max_r, H - 1 - max_r))

                e = float(s["env"][ti]) if ti < len(s["env"]) else 0.0
                if e <= 0.0:
                    continue

                # choose spot style each frame
                style = rng.choice(["gauss", "aniso", "superellipse", "multi_gauss"])
                if style == "gauss":
                    r = rng.uniform(gmin, gmax) * max_r
                    mask = self._gaussian_mask(H, W, s["x"], s["y"], r, r, 0.0)
                elif style == "aniso":
                    rx = rng.uniform(gmin, gmax) * max_r
                    ry = rng.uniform(gmin, gmax) * max_r
                    th = rng.uniform(0, np.pi)
                    mask = self._gaussian_mask(H, W, s["x"], s["y"], rx, ry, th)
                elif style == "superellipse":
                    ax = rng.uniform(gmin, gmax) * max_r
                    ay = rng.uniform(gmin, gmax) * max_r
                    p = rng.uniform(1.6, 2.6)
                    th = rng.uniform(0, np.pi)
                    mask = self._superellipse_mask(H, W, s["x"], s["y"], ax, ay, p, th)
                else:
                    mask = self._multi_gaussian_mask(H, W, s["x"], s["y"], base_r=max_r * 0.8, rng=rng)

                total_add += mask.astype(np.float32) * (intensity_max * e)

            frame = np.clip(bg.astype(np.float32) + total_add, 0, 255).astype(np.uint8)
            frames.append(frame)

        # metadata: keep legacy keys for first spot; add per-spot list
        meta_spots = []
        for s in spots:
            meta_spots.append({
                "cx0": int(s["cx0"]), "cy0": int(s["cy0"]),
                "cxT": int(round(s["x"])), "cyT": int(round(s["y"]))
            })
        if meta_spots:
            meta = {**meta_spots[0], "n_spots": n_spots, "spots": meta_spots}
        else:
            meta = {"cx0": -1, "cy0": -1, "cxT": -1, "cyT": -1, "n_spots": 0, "spots": []}

        video_u8 = np.stack(frames, axis=0)
        return video_u8, meta

# ============================================================
# Dataset
# ============================================================

@dataclass
class VideoFromImagesConfig:
    """Config for synthetic grayscale videos."""

    root: Union[str, Path]
    frames: int = 16
    height: int = 64
    width: int = 64
    fixed_spot: bool = True
    max_radius: int = 5
    schedule_fn: Callable[[int], List[int]] = triangular_radius_schedule
    return_ch_first: bool = False
    normalize: bool = True
    seed: Optional[int] = None
    # jitter controls
    bg_jitter_px: int = 1
    bg_jitter_every: int = 3
    spot_jitter_px: int = 0
    spot_jitter_every: int = 100
    # noise/contrast
    gaussian_noise_std: float = 0.0
    contrast_jitter: bool = False
    # new realism controls (defaults tuned DOWN)
    drift_std: float = 0.10
    drift_decay: float = 0.90
    motion_blur_strength: float = 0.35
    spot_intensity: float = 160.0
    gauss_sigma_min: float = 0.4
    gauss_sigma_max: float = 0.8
    # new controls for spot presence/timing
    spot_probability: float = 1.0
    spot_min_frames: int = 1
    spot_max_frames: Optional[int] = 10
    spot_count_choices: Tuple[int, ...] = (4, 8, 12)


class BrightSpotVideoDataset(Dataset):
    """Build synthetic grayscale video pairs with optional jitter/noise."""

    def __init__(self, cfg: VideoFromImagesConfig, paths: Optional[list[Path]] = None) -> None:
        super().__init__()
        self.cfg = cfg
        self.paths = paths if paths is not None else list_images(cfg.root)
        self.base_rng = random.Random(cfg.seed)
        self.gen = SyntheticVideoGenerator(cfg)
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.paths)
    
    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _rng_for_index(self, idx: int) -> random.Random:
        seed_base = 0 if self.cfg.seed is None else int(self.cfg.seed)
        mix = (idx + 0x9E3779B97F4A7C15) ^ (self.epoch * 0xBF58476D1CE4E5B9)
        seed = (seed_base ^ (mix & 0xFFFFFFFFFFFFFFFF)) & 0xFFFFFFFFFFFFFFFF
        return random.Random(seed)

    def __getitem__(self, idx: int):
        rng = self._rng_for_index(idx)
        path = self.paths[rng.randrange(len(self.paths))]
        base_u8 = load_grayscale_resized(path, out_hw=(self.cfg.height, self.cfg.width))

        # Prefer the new generator pathway for realism
        video_u8, meta = self.gen.build_video(base_u8, rng)

        vid = torch.from_numpy(video_u8).float()
        if self.cfg.normalize:
            vid = vid / 255.0
        if self.cfg.return_ch_first:
            vid = vid.unsqueeze(1)

        meta.pop("spots", None)  # <-- remove variable-length field so default_collate is happy
        return {"video": vid, "target": vid.clone(), "path": str(path), **meta}

# ============================================================
# Dataloader helpers
# ============================================================

def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_loader(
    cfg: VideoFromImagesConfig,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> DataLoader:
    ds = BrightSpotVideoDataset(cfg)
    g = torch.Generator()
    if cfg.seed is not None:
        g.manual_seed(int(cfg.seed))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory, worker_init_fn=seed_worker, generator=g)


def split_paths(root: str | Path, val_ratio: float = 0.1, seed: int | None = 123) -> Tuple[list[Path], list[Path]]:
    all_paths = list_images(root)
    rng = random.Random(seed)
    rng.shuffle(all_paths)
    n_total = len(all_paths)
    n_val = max(1, int(round(n_total * val_ratio)))
    val_paths = all_paths[:n_val]
    train_paths = all_paths[n_val:]
    if len(train_paths) == 0 and n_total > 0:
        train_paths, val_paths = all_paths[:1], all_paths[1:]
    return train_paths, val_paths


def make_loaders_from_single_root(
    root: str | Path,
    frames: int,
    height: int,
    width: int,
    batch_size: int = 16,
    num_workers: int = 4,
    val_ratio: float = 0.1,
    seed: int | None = 123,
):
    train_paths, val_paths = split_paths(root, val_ratio=val_ratio, seed=seed)

    train_cfg = VideoFromImagesConfig(root=root, frames=frames, height=height, width=width, fixed_spot=True, seed=seed)
    val_cfg = VideoFromImagesConfig(root=root, frames=frames, height=height, width=width, fixed_spot=True, seed=(None if seed is None else seed + 1))

    train_ds = BrightSpotVideoDataset(train_cfg, paths=train_paths)
    val_ds = BrightSpotVideoDataset(val_cfg, paths=val_paths)

    g = torch.Generator()
    if seed is not None:
        g.manual_seed(int(seed))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, worker_init_fn=seed_worker, generator=g)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, worker_init_fn=seed_worker, generator=g)
    return train_loader, val_loader


# ============================================================
# GIF utilities
# ============================================================

def save_video_gif(video, path, interval_ms: int = 100) -> None:
    import matplotlib.pyplot as plt
    from matplotlib import animation

    if isinstance(video, torch.Tensor):
        video = video.detach().cpu().numpy()

    if video.ndim == 4 and video.shape[1] == 1:
        video = video[:, 0, :, :]
    assert video.ndim == 3, f"Expected [T,H,W], got {video.shape}"
    T, H, W = video.shape

    if video.dtype != np.uint8:
        vmin = float(video.min())
        vmax = float(video.max()) if float(video.max()) > vmin else vmin + 1.0
        video_u8 = ((video - vmin) / (vmax - vmin) * 255.0).clip(0, 255).astype(np.uint8)
    else:
        video_u8 = video

    fig, ax = plt.subplots()
    ax.set_axis_off()
    im = ax.imshow(video_u8[0], animated=True, cmap="gray")

    def init():
        im.set_data(video_u8[0])
        return (im,)

    def update(i):
        im.set_data(video_u8[i])
        return (im,)

    anim = animation.FuncAnimation(fig, update, init_func=init, frames=T, interval=interval_ms, blit=True, repeat=True)

    writer = animation.PillowWriter(fps=max(1, int(1000 / interval_ms)))
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    anim.save(path, writer=writer)
    plt.close(fig)


def save_batch_as_gifs(batch, out_dir: str = "gifs", prefix: str = "vid", interval_ms: int = 100, max_videos: int | None = None) -> list[str]:
    """Save every video in a dataloader batch to GIFs."""
    vids = batch["video"]
    if isinstance(vids, torch.Tensor):
        vids = vids.detach().cpu()

    if vids.ndim == 5 and vids.shape[2] == 1:
        vids = vids[:, :, 0, :, :]
    assert vids.ndim == 4, f"Expected [B,T,H,W], got {tuple(vids.shape)}"

    os.makedirs(out_dir, exist_ok=True)
    paths: list[str] = []
    B = vids.shape[0]
    N = min(B, max_videos) if max_videos is not None else B
    for i in range(N):
        path = os.path.join(out_dir, f"{prefix}_{i:03d}.gif")
        save_video_gif(vids[i], path, interval_ms=interval_ms)
        paths.append(path)
    return paths


if __name__ == "__main__":
    cfg = VideoFromImagesConfig(
        root="sub_band8",
        frames=16,
        height=64,
        width=64,
        fixed_spot=True,
        seed=123,
        bg_jitter_px=1,
        bg_jitter_every=3,
        spot_jitter_px=1,
        spot_jitter_every=3,
        gaussian_noise_std=5.0,
        contrast_jitter=True,
    )

    loader = make_loader(cfg, batch_size=16)

    for batch in loader:
        paths = save_batch_as_gifs(batch, out_dir="gifs", prefix="band8", interval_ms=100)
        print("saved:", paths)
        break
