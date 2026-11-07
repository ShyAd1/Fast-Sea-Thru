#!/usr/bin/env python3
"""
Sea-Thru Underwater Image Restoration
Based on Sea-Thru-Impl by Zeyuan HE (Teragion)
https://github.com/Teragion/Sea-Thru-Impl

Depth estimation with MiDaS and Sea-Thru illuminant via CUDA C++ DLL.
"""

import argparse
import ctypes
import glob
import math
import os
import sys
import traceback
from collections import deque
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
import scipy.optimize
import torch
import torch.nn.functional as F

try:
    from skimage import exposure as sk_exposure
except Exception:
    sk_exposure = None

try:
    import rawpy  # type: ignore[import-not-found]
except Exception:
    rawpy = None

NUM_BINS = 10
MAX_SIDE = 1280  # default longest-side resize target
DEFAULT_ILLUMINANT_ITERS = 200
DEFAULT_BS_ATTEMPTS = 40
DEFAULT_ATT_ATTEMPTS = 12
DEFAULT_CUDA_WINDOW_SIZE = 7
DEFAULT_CUDA_BLOCK_X = 32
DEFAULT_CUDA_BLOCK_Y = 16
DEFAULT_GPU_FIT_STEPS = 120
MAX_FIT_POINTS = 200_000  # cap on points used by the GPU/CPU curve fits (avoids OOM)
MIN_FAR = 1.0  # lower bound for the estimated far distance
DEFAULT_DEPTH_SHIFT = 2.0  # disparity offset in depth = 10/(9.5*disp + shift); see merge notes
ATT_NUM_BINS = 48  # equal-count depth bins used by the robust attenuation fit
ATT_MIN_BIN_COUNT = 20
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


_ILLU_DLL = None


def _collect_input_images(inputs: list[str], recursive: bool = False) -> list[Path]:
    """Expand input args into a deduplicated list of image files."""
    images: list[Path] = []
    seen: set[str] = set()

    for item in inputs:
        p = Path(item)
        candidates: list[Path] = []

        if p.is_dir():
            it = p.rglob("*") if recursive else p.glob("*")
            candidates = [
                x for x in it if x.is_file() and x.suffix.lower() in IMAGE_EXTENSIONS
            ]
        elif p.is_file():
            candidates = [p]
        else:
            # Preserve original behavior: unresolved path is passed and handled at read time.
            candidates = [p]

        for c in candidates:
            key = str(c.resolve()) if c.exists() else str(c)
            if key not in seen:
                seen.add(key)
                images.append(c)

    return images


def _iter_batches(items: list[Path], batch_size: int) -> Iterator[list[Path]]:
    n = max(1, int(batch_size))
    for i in range(0, len(items), n):
        yield items[i : i + n]


def _register_dll_dirs():
    if sys.platform != "win32":
        return
    dirs = [r"C:\msys64\ucrt64\bin", r"C:\msys64\mingw64\bin"]
    for env in ("CUDA_PATH", "CUDA_HOME"):
        if os.environ.get(env):
            dirs.append(os.path.join(os.environ[env], "bin"))
            dirs.append(os.path.join(os.environ[env], "bin", "x64"))
    dirs += sorted(
        glob.glob(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v*\bin"),
        reverse=True,
    )
    for p in dirs:
        if os.path.isdir(p):
            try:
                os.add_dll_directory(p)
            except Exception:
                pass


def _load_dll():
    global _ILLU_DLL
    if _ILLU_DLL is not None:
        return _ILLU_DLL
    _register_dll_dirs()
    here = Path(__file__).resolve().parent
    candidates: list[Path] = []
    if os.environ.get("SEATHRU_DLL"):
        candidates.append(Path(os.environ["SEATHRU_DLL"]))
    for base in (here, Path.cwd()):
        for name in (
            "illuminant_cuda.dll",
            "illuminant_cuda.so",
            "sillu.dll",
            "sillu.so",
        ):
            candidates.append(base / name)
    for p in candidates:
        if p.exists():
            try:
                _ILLU_DLL = ctypes.CDLL(str(p))
                print(f"[dll] Loaded: {p.name}")
                return _ILLU_DLL
            except Exception as e:
                print(f"[dll] Failed to load {p.name}: {e}", file=sys.stderr)
    return None


def merge_disparity_scales(
    disp_lo: np.ndarray, disp_hi: np.ndarray, sigma_frac: float = 0.03
) -> np.ndarray:
    """Combine a coarse (globally consistent) and a fine (detailed) disparity map.

    Low frequencies come from `disp_lo`; high frequencies (detail) from `disp_hi`,
    after an affine alignment of `disp_hi` to `disp_lo` (relative disparity has an
    arbitrary scale and shift per inference). Falls back to `disp_lo` if the two
    maps are not positively correlated.
    """
    H, W = disp_lo.shape
    s = max(2.0, float(sigma_frac) * min(H, W))
    lo32 = disp_lo.astype(np.float32)
    hi32 = disp_hi.astype(np.float32)
    lo_l = cv2.GaussianBlur(lo32, (0, 0), s)
    hi_l = cv2.GaussianBlur(hi32, (0, 0), s)
    step = max(1, int(math.sqrt(lo_l.size / 100_000.0)))
    x = hi_l[::step, ::step].ravel().astype(np.float64)
    y = lo_l[::step, ::step].ravel().astype(np.float64)
    var = float(np.var(x))
    if var < 1e-12:
        return disp_lo
    a = float(np.mean((x - x.mean()) * (y - y.mean())) / var)
    if not np.isfinite(a) or a <= 0:
        return disp_lo
    b = float(y.mean() - a * x.mean())
    hi_aligned = a * hi32 + b
    detail = hi_aligned - cv2.GaussianBlur(hi_aligned, (0, 0), s)
    return (lo_l + detail).astype(np.float64)


class MiDaS:
    def __init__(self, model_type: str = "MiDaS_small"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_type = model_type
        print(f"[midas] Loading {model_type} on {self.device}...")
        self.model = (
            torch.hub.load("intel-isl/MiDaS", model_type, trust_repo=True)
            .to(self.device)
            .eval()
        )
        hub_tfm = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
        self.tfm = (
            hub_tfm.dpt_transform
            if model_type in ("DPT_Large", "DPT_Hybrid")
            else hub_tfm.small_transform
        )
        print("[midas] Ready.")

    def _run_model(self, inp: torch.Tensor, H: int, W: int) -> np.ndarray:
        """Run the network on a prepared input and return disparity at (H, W)."""
        inp = inp.to(self.device)
        use_amp = self.device.type == "cuda"
        with torch.autocast(self.device.type, dtype=torch.float16, enabled=use_amp):
            disp = self.model(inp)
        if use_amp and not bool(torch.isfinite(disp).all()):
            # fp16 overflow -> NaN/inf; redo in full precision.
            print("  [midas] non-finite fp16 output, retrying in fp32")
            with torch.autocast(self.device.type, enabled=False):
                disp = self.model(inp.float())
        return (
            F.interpolate(
                disp.unsqueeze(1), size=(H, W), mode="bicubic", align_corners=False
            )
            .squeeze()
            .cpu()
            .float()
            .numpy()
        )

    def _detail_input(self, rgb_f32: np.ndarray, long_side: int) -> torch.Tensor:
        """Own resize + normalisation so the network can run above its default 384 px."""
        H, W = rgb_f32.shape[:2]
        scale = float(long_side) / max(H, W)
        th = max(32, int(round(H * scale / 32.0)) * 32)
        tw = max(32, int(round(W * scale / 32.0)) * 32)
        img = cv2.resize(rgb_f32, (tw, th), interpolation=cv2.INTER_CUBIC)
        if self.model_type in ("DPT_Large", "DPT_Hybrid"):
            mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
            std = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        else:
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        x = (img.astype(np.float32) - mean) / std
        return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1))).unsqueeze(0)

    @torch.no_grad()
    def predict_depth(
        self,
        rgb_f32: np.ndarray,
        debias_sigma_frac: float = 0.0,
        depth_shift: float = DEFAULT_DEPTH_SHIFT,
        detail_size: int = 0,
    ) -> np.ndarray:
        """
        rgb_f32 : HxWx3 float32 [0,1]
        Returns : HxW float64 relative depth (larger = farther)

        depth_shift : offset in depth = 10 / (9.5 * disp_norm + shift). Small values
            (the old 0.5) blow up the far tail: a small dark patch of low disparity
            became ~10x farther than its surroundings, which showed up as strong
            color blotches after recovery. Larger values compress that tail.
        detail_size : if > 0, also run the network at this long side (e.g. 768) and
            merge its fine detail onto the standard-resolution result (experimental).
        """
        H, W = rgb_f32.shape[:2]
        disp = self._run_model(self.tfm(rgb_f32), H, W)
        if detail_size and detail_size > 0:
            try:
                disp_hi = self._run_model(self._detail_input(rgb_f32, int(detail_size)), H, W)
                if np.isfinite(disp_hi).all() and np.isfinite(disp).all():
                    disp = merge_disparity_scales(disp, disp_hi)
                else:
                    print("  [midas] detail pass non-finite, using single scale")
            except Exception as e:  # never let the optional pass break the run
                print(f"  [midas] detail pass failed ({e}); using single scale", file=sys.stderr)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        if debias_sigma_frac > 0:
            # Remove very low-frequency disparity bias (common underwater spotlight artifact).
            sigma = max(1.0, float(min(H, W)) * float(debias_sigma_frac))
            low = cv2.GaussianBlur(disp.astype(np.float32), (0, 0), sigmaX=sigma)
            disp = np.maximum(disp - low + float(np.median(low)), 1e-6)
        disp = np.nan_to_num(disp, nan=0.0, posinf=0.0, neginf=0.0)
        lo, hi = np.percentile(disp, 1), np.percentile(disp, 99)
        if hi <= lo:
            hi = lo + 1e-6
        # Invert disparity -> depth. disp_n in [0,1] => depth in [10/(9.5+s), 10/s].
        disp_n = np.clip((disp - lo) / (hi - lo), 0.0, 1.0)
        depth = 10.0 / (disp_n * 9.5 + max(float(depth_shift), 0.05))
        return depth.astype(np.float64)


def read_image(path: str, max_side: int = MAX_SIDE) -> np.ndarray:
    """Returns HxWx3 float64 RGB [0,1], longest side ≤ max_side."""
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is not None:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    elif rawpy is not None:
        try:
            rgb = rawpy.imread(path).postprocess()
        except Exception as e:
            raise FileNotFoundError(f"Cannot read image (OpenCV/RAW): {path}: {e}")
    else:
        raise FileNotFoundError(f"Cannot read image: {path}")
    h, w = rgb.shape[:2]
    if max_side and max(h, w) > max_side:
        scale = max_side / max(h, w)
        rgb = cv2.resize(
            rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
        )
    return rgb.astype(np.float64) / 255.0


def read_depthmap(path: str, target_hw: tuple[int, int]) -> np.ndarray:
    """Read a depth map and resize to the RGB image size (H, W)."""
    dep = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if dep is None:
        raise FileNotFoundError(f"Cannot read depth map: {path}")
    if dep.ndim == 3:
        dep = cv2.cvtColor(dep, cv2.COLOR_BGR2GRAY)
    dep = dep.astype(np.float64)
    h, w = target_hw
    if dep.shape[:2] != (h, w):
        dep = cv2.resize(dep, (w, h), interpolation=cv2.INTER_CUBIC)
    return dep


def save_image(path: str, img: np.ndarray):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    u8 = np.clip(img * 255, 0, 255).astype(np.uint8) if img.dtype != np.uint8 else img
    cv2.imwrite(path, cv2.cvtColor(u8, cv2.COLOR_RGB2BGR))


DEPTH_COLORMAPS = {
    "magma": cv2.COLORMAP_MAGMA,
    "inferno": cv2.COLORMAP_INFERNO,
    "viridis": cv2.COLORMAP_VIRIDIS,
    "turbo": cv2.COLORMAP_TURBO,
}


def depth_to_vis(
    depths: np.ndarray,
    depth_vis_min: float | None = None,
    depth_vis_max: float | None = None,
    colormap: str = "magma",
) -> np.ndarray:
    """Render a depth map with optional fixed display range to avoid per-image look-alike maps."""
    if depth_vis_min is None:
        vmin = float(np.min(depths))
    else:
        vmin = float(depth_vis_min)

    if depth_vis_max is None:
        vmax = float(np.max(depths))
    else:
        vmax = float(depth_vis_max)

    if not np.isfinite(vmin):
        vmin = 0.0
    if not np.isfinite(vmax):
        vmax = vmin + 1.0
    if vmax <= vmin:
        vmax = vmin + 1e-6

    d_n = np.clip((depths - vmin) / (vmax - vmin), 0.0, 1.0)
    d_u8 = (d_n * 255).astype(np.uint8)
    if colormap.lower() == "gray":
        return d_u8  # single-channel 8-bit, brighter = farther
    return cv2.applyColorMap(d_u8, DEPTH_COLORMAPS.get(colormap.lower(), cv2.COLORMAP_MAGMA))


def depth_to_gray(
    depths: np.ndarray,
    depth_vis_min: float | None = None,
    depth_vis_max: float | None = None,
    bits: int = 16,
) -> np.ndarray:
    """Render a normalized grayscale depth image for diagnostics (like MiDaS PNG export)."""
    if depth_vis_min is None:
        vmin = float(np.min(depths))
    else:
        vmin = float(depth_vis_min)

    if depth_vis_max is None:
        vmax = float(np.max(depths))
    else:
        vmax = float(depth_vis_max)

    if not np.isfinite(vmin):
        vmin = 0.0
    if not np.isfinite(vmax):
        vmax = vmin + 1.0
    if vmax <= vmin:
        vmax = vmin + 1e-6

    d_n = np.clip((depths - vmin) / (vmax - vmin), 0.0, 1.0)
    if bits >= 16:
        return (d_n * 65535.0).astype(np.uint16)
    return (d_n * 255.0).astype(np.uint8)


def normalize_depth_map(
    depths: np.ndarray, z_min: float = 0.1, z_inf: float = 6.0
) -> np.ndarray:
    depths = depths.copy()
    z_max = max(float(np.max(depths)), float(z_inf))
    depths[depths == 0] = z_max
    depths[depths < z_min] = z_min
    return depths


def estimate_far(image: np.ndarray, frac: float = 0.2) -> float:
    """Estimate scene far distance using luminance contrast heuristic from the reference impl."""
    r = image[:, :, 0] * 0.2126
    g = image[:, :, 1] * 0.7152
    b = image[:, :, 2] * 0.0722
    lum = r + g + b
    lum.sort(axis=0)

    rows = max(1, int(frac * lum.shape[0]))
    darkest = np.mean(lum[rows : (2 * rows), :], axis=0)
    brightest = np.mean(lum[-(2 * rows) : (-rows), :], axis=0)

    ratio = np.mean(brightest / np.clip(darkest, 1e-6, None))
    far = float(np.log2(max(ratio, 1e-6)) * 10.0)
    if not np.isfinite(far) or far < MIN_FAR:
        print(f"  [depth] estimated far={far:.3f} too small, using {MIN_FAR}")
        far = MIN_FAR
    return far


def refine_depths_from_hint(depths: np.ndarray, avg: float) -> np.ndarray:
    oavg = float(np.mean(depths))
    if oavg <= 1e-8:
        return depths
    scaled = depths * (avg / oavg)
    print(f"  [depth] hint-scaled max={np.max(scaled):.3f}, mean={np.mean(scaled):.3f}")
    return scaled


def preprocess_predicted_depths(
    original: np.ndarray, depths: np.ndarray, far_override: float | None = None
) -> np.ndarray:
    if far_override is not None and far_override > 0:
        far = float(far_override)
        print(f"  [depth] using manual far={far:.3f}")
    else:
        far = estimate_far(original)
    ratio = far / max(float(np.max(depths)), 1e-6)
    scaled = depths * ratio
    print(f"  [depth] estimated far={far:.3f}, scaled mean={np.mean(scaled):.3f}")
    return scaled


def combine_map_predict(
    depth_map: np.ndarray, predicted_depth: np.ndarray
) -> np.ndarray:
    """Fill invalid depth-map holes with MiDaS prediction and smooth softly."""
    depths = depth_map.copy()
    idxs = np.where(depths == 0)
    if idxs[0].size == 0:
        return depths

    mfar = (
        float(np.percentile(depths[depths > 0], 95))
        if np.any(depths > 0)
        else float(np.percentile(predicted_depth, 95))
    )
    far_idxs = np.where(depths > mfar)
    if far_idxs[0].size > 0:
        pfar = float(np.min(predicted_depth[far_idxs]))
    else:
        pfar = float(np.min(predicted_depth))
    ratio = mfar / max(pfar, 1e-6)
    depths[idxs] = predicted_depth[idxs] * ratio

    # OpenCV bilateral keeps depth boundaries better than gaussian blur.
    depths32 = depths.astype(np.float32)
    depths = cv2.bilateralFilter(depths32, d=5, sigmaColor=0.08, sigmaSpace=7).astype(
        np.float64
    )
    return depths


def _resolve_depth_path_for_image(depth_source: str, image_path: Path) -> Path:
    p = Path(depth_source)
    if p.is_file():
        return p
    if not p.is_dir():
        raise FileNotFoundError(f"Depth source not found: {depth_source}")

    candidates = list(p.glob(f"{image_path.stem}.*"))
    candidates = [c for c in candidates if c.is_file()]
    if not candidates:
        raise FileNotFoundError(
            f"No depth map matching '{image_path.stem}.*' inside: {depth_source}"
        )
    candidates.sort()
    return candidates[0]


def _rgb2ycbcr(im_u8: np.ndarray) -> np.ndarray:
    xform = np.array(
        [[0.299, 0.587, 0.114], [-0.1687, -0.3313, 0.5], [0.5, -0.4187, -0.0813]],
        dtype=np.float64,
    )
    ycbcr = im_u8.astype(np.float64).dot(xform.T)
    ycbcr[:, :, [1, 2]] += 128.0
    return ycbcr


def _ycbcr2rgb(im: np.ndarray) -> np.ndarray:
    xform = np.array(
        [[1.0, 0.0, 1.402], [1.0, -0.34414, -0.71414], [1.0, 1.772, 0.0]],
        dtype=np.float64,
    )
    rgb = im.astype(np.float64)
    rgb[:, :, [1, 2]] -= 128.0
    rgb = rgb.dot(xform.T)
    return np.clip(rgb, 0.0, 255.0).astype(np.uint8)


def wb_ycbcr_mean(data_u8: np.ndarray) -> np.ndarray:
    ycbcr = _rgb2ycbcr(data_u8)
    yc = [float(np.mean(ycbcr[:, :, i])) for i in range(3)]
    for i in (1, 2):
        ycbcr[:, :, i] = np.clip(ycbcr[:, :, i] + (128.0 - yc[i]), 0.0, 255.0)
    return _ycbcr2rgb(ycbcr)


def postprocess_image(ja: np.ndarray, mode: str = "original") -> np.ndarray:
    """Optional visual postprocess used in the original implementation."""
    x = np.clip(ja, 0.0, 1.0)
    maxv = float(np.max(x))
    if maxv > 1e-8:
        x = x / maxv

    mode_l = mode.lower()
    if mode_l == "original" and sk_exposure is not None:
        # Original Sea-Thru order: normalize -> equalize_adapthist -> wb_ycbcr_mean.
        x = sk_exposure.equalize_adapthist(x)
        x_u8 = np.clip(x * 255.0, 0, 255).astype(np.uint8)
        x_u8 = wb_ycbcr_mean(x_u8)
        return x_u8.astype(np.float64) / 255.0

    # Fast OpenCV fallback when skimage is unavailable or explicitly requested.
    x_u8 = (x * 255.0).astype(np.uint8)
    ycrcb = cv2.cvtColor(x_u8, cv2.COLOR_RGB2YCrCb)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    ycrcb[:, :, 0] = clahe.apply(ycrcb[:, :, 0])
    x_u8 = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
    x_u8 = wb_ycbcr_mean(x_u8)
    return x_u8.astype(np.float64) / 255.0


def _bs_model(z, veil, beta_b, recover, beta_d):
    return veil * (1 - np.exp(-beta_b * z)) + recover * np.exp(-beta_d * z)


def _find_reference_points(
    image: np.ndarray, depths: np.ndarray, frac: float = 0.02
) -> np.ndarray:
    z_min, z_max = np.min(depths), np.max(depths)
    bins = np.linspace(z_min, z_max, NUM_BINS + 1)
    rgb_norm = np.linalg.norm(image, axis=2)
    pts = []
    for i in range(NUM_BINS):
        lo, hi = bins[i], bins[i + 1]
        if i == NUM_BINS - 1:  # include the farthest pixels
            idx = np.where((depths >= lo) & (depths <= hi))
        else:
            idx = np.where((depths >= lo) & (depths < hi))
        if idx[0].size == 0:
            continue
        nrm, z_b, col = rgb_norm[idx], depths[idx], image[idx]
        n = max(1, math.ceil(len(z_b) * frac))
        order = np.argpartition(nrm, min(n, len(nrm) - 1))[:n]
        for j in order:
            pts.append((nrm[j], z_b[j], col[j, 0], col[j, 1], col[j, 2]))
    if not pts:
        return np.zeros((0, 5), dtype=np.float64)
    return np.array(pts)


def _subsample(*arrays: np.ndarray, max_points: int = MAX_FIT_POINTS):
    """Random (seeded by global numpy RNG) subsample of aligned 1-D arrays."""
    n = arrays[0].shape[0]
    if n <= max_points:
        return arrays
    idx = np.random.choice(n, size=max_points, replace=False)
    return tuple(a[idx] for a in arrays)


def estimate_backscatter(
    image: np.ndarray,
    depths: np.ndarray,
    attempts: int = DEFAULT_BS_ATTEMPTS,
    ref_frac: float = 0.03,
) -> np.ndarray:
    """Returns HxWx3 float64 backscatter map."""
    pts = _find_reference_points(image, depths, frac=ref_frac)
    if pts.shape[0] < 8:
        print("  [bs] not enough reference points, skipping backscatter removal")
        return np.zeros_like(image, dtype=np.float64)
    lo_b, hi_b = np.array([0.0, 0, 0, 0]), np.array([1.0, 5, 1, 5])
    channels = []
    for ch in range(3):
        best_loss, best_p = np.inf, np.zeros(4)
        for _ in range(attempts):
            p0 = np.random.rand(4) * (hi_b - lo_b) + lo_b
            try:
                popt, _ = scipy.optimize.curve_fit(
                    _bs_model,
                    pts[:, 1],
                    pts[:, ch + 2],
                    p0,
                    bounds=(lo_b, hi_b),
                    maxfev=2000,
                )
                loss = float(
                    np.mean((_bs_model(pts[:, 1], *popt) - pts[:, ch + 2]) ** 2)
                )
                if loss < best_loss:
                    best_loss, best_p = loss, popt
            except (RuntimeError, ValueError):
                pass
        print(f"  [bs] ch={ch}  mse={best_loss:.6f}")
        channels.append(_bs_model(depths, *best_p) if np.isfinite(best_loss) else np.zeros_like(depths))
    return np.stack(channels, axis=2)


def _estimate_backscatter_torch(
    image: np.ndarray,
    depths: np.ndarray,
    attempts: int = DEFAULT_BS_ATTEMPTS,
    ref_frac: float = 0.03,
    fit_steps: int = DEFAULT_GPU_FIT_STEPS,
) -> np.ndarray:
    """GPU backscatter fit via vectorized multi-restart Adam optimization."""
    if not torch.cuda.is_available():
        return estimate_backscatter(image, depths, attempts=attempts, ref_frac=ref_frac)

    pts = _find_reference_points(image, depths, frac=ref_frac)
    if pts.shape[0] < 8:
        print("  [bs-gpu] not enough reference points, skipping backscatter removal")
        return np.zeros_like(image, dtype=np.float64)

    device = torch.device("cuda")
    z_pts = torch.from_numpy(pts[:, 1].astype(np.float32)).to(device)
    z_map = torch.from_numpy(depths.astype(np.float32)).to(device)

    channels = []
    A = max(8, int(attempts))
    steps = max(30, int(fit_steps))
    for ch in range(3):
        y_pts = torch.from_numpy(pts[:, ch + 2].astype(np.float32)).to(device)
        raw = torch.randn(
            (A, 4), device=device, dtype=torch.float32, requires_grad=True
        )
        opt = torch.optim.Adam([raw], lr=0.08)

        for _ in range(steps):
            veil = torch.sigmoid(raw[:, 0:1])
            beta_b = 5.0 * torch.sigmoid(raw[:, 1:2])
            recover = torch.sigmoid(raw[:, 2:3])
            beta_d = 5.0 * torch.sigmoid(raw[:, 3:4])

            z = z_pts.unsqueeze(0)
            pred = veil * (1.0 - torch.exp(-beta_b * z)) + recover * torch.exp(
                -beta_d * z
            )
            mse = ((pred - y_pts.unsqueeze(0)) ** 2).mean(dim=1)
            loss = mse.mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        with torch.no_grad():
            veil = torch.sigmoid(raw[:, 0:1])
            beta_b = 5.0 * torch.sigmoid(raw[:, 1:2])
            recover = torch.sigmoid(raw[:, 2:3])
            beta_d = 5.0 * torch.sigmoid(raw[:, 3:4])

            z = z_pts.unsqueeze(0)
            pred = veil * (1.0 - torch.exp(-beta_b * z)) + recover * torch.exp(
                -beta_d * z
            )
            mse = ((pred - y_pts.unsqueeze(0)) ** 2).mean(dim=1)
            mse = torch.nan_to_num(mse, nan=float("inf"))
            best = int(torch.argmin(mse).item())

            v = veil[best, 0]
            bb = beta_b[best, 0]
            r = recover[best, 0]
            bd = beta_d[best, 0]
            full = v * (1.0 - torch.exp(-bb * z_map)) + r * torch.exp(-bd * z_map)
            print(f"  [bs-gpu] ch={ch}  mse={float(mse[best].item()):.6f}")
            channels.append(full.detach().cpu().numpy().astype(np.float64))

    return np.stack(channels, axis=2)


def _release_dll_buffers(lib) -> None:
    try:
        fn = lib.release_illuminant_cuda_buffers
        fn.restype = None
        fn.argtypes = []
        fn()
    except Exception:
        pass


def _sanitize_launch(window_size: int, block_x: int, block_y: int) -> tuple[int, int, int]:
    """Clamp CUDA launch parameters to values the hardware accepts."""
    window_size = max(1, int(window_size))
    if window_size % 2 == 0:
        window_size += 1  # kernel uses a symmetric window
    block_x, block_y = int(block_x), int(block_y)
    if block_x <= 0 or block_y <= 0 or block_x * block_y > 1024:
        print(
            f"[dll] invalid block {block_x}x{block_y} (max 1024 threads); using 32x8",
            file=sys.stderr,
        )
        block_x, block_y = 32, 8
    return window_size, block_x, block_y


def _illuminant_dll(
    D: np.ndarray,
    depths: np.ndarray,
    p: float = 0.4,
    f: float = 2.0,
    iterations: int = 500,
    window_size: int = DEFAULT_CUDA_WINDOW_SIZE,
    block_x: int = DEFAULT_CUDA_BLOCK_X,
    block_y: int = DEFAULT_CUDA_BLOCK_Y,
) -> np.ndarray | None:
    """Call the optional CUDA C++ DLL for illuminant computation.

    Returns None whenever the DLL is missing, a CUDA call fails or the result
    is not finite, so the caller can fall back to the CPU backend.
    """
    lib = _load_dll()
    if lib is None:
        return None
    window_size, block_x, block_y = _sanitize_launch(window_size, block_x, block_y)
    eps = max(float(np.mean(depths) / 50.0), 1e-6)
    z = np.require(np.nan_to_num(depths, nan=0.0), np.float64, ["C", "ALIGNED"])
    Dc_rgb = np.require(
        np.nan_to_num(np.ascontiguousarray(D), nan=0.0), np.float64, ["C", "ALIGNED"]
    )
    H, W = z.shape

    # The DLL runs in its own CUDA runtime: hand back PyTorch's cached VRAM first.
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        except Exception:
            pass

    # Preferred path: single RGB call with reusable GPU buffers.
    if hasattr(lib, "compute_illuminant_cuda_rgb"):
        func_rgb = lib.compute_illuminant_cuda_rgb
        func_rgb.restype = ctypes.c_int
        func_rgb.argtypes = [
            np.ctypeslib.ndpointer(np.float64, ndim=3, flags="C,A"),
            np.ctypeslib.ndpointer(np.float64, ndim=2, flags="C,A"),
            np.ctypeslib.ndpointer(np.float64, ndim=3, flags="C,A,W"),
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        for attempt in (1, 2):
            out_rgb = np.zeros_like(Dc_rgb)
            rc = func_rgb(
                Dc_rgb, z, out_rgb, p, f, eps, H, W,
                int(iterations), window_size, block_x, block_y,
            )
            if rc == 0 and np.isfinite(out_rgb).all():
                return out_rgb
            print(
                f"[dll] compute_illuminant_cuda_rgb failed (code={rc}, attempt {attempt}/2)",
                file=sys.stderr,
            )
            _release_dll_buffers(lib)  # drop stale buffers / broken stream and retry once
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return None  # do NOT fall through to the legacy kernels of the same DLL

    # Older DLL: per-channel CUDA entry point.
    if hasattr(lib, "compute_illuminant_cuda"):
        func = lib.compute_illuminant_cuda
        func.restype = None
        func.argtypes = [
            np.ctypeslib.ndpointer(np.float64, ndim=2, flags="C,A"),
            np.ctypeslib.ndpointer(np.float64, ndim=2, flags="C,A"),
            np.ctypeslib.ndpointer(np.float64, ndim=2, flags="C,A,W"),
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        channels = []
        for ch in range(3):
            Dc = np.require(np.ascontiguousarray(Dc_rgb[:, :, ch]), np.float64, ["C", "ALIGNED"])
            out = np.zeros_like(Dc)
            func(Dc, z, out, p, f, eps, H, W, int(iterations), window_size)
            channels.append(out)
        res = np.stack(channels, axis=2)
        return res if np.isfinite(res).all() else None

    # Legacy CPU C++ DLL (compute_illuminant_map, no window param).
    if hasattr(lib, "compute_illuminant_map"):
        func = lib.compute_illuminant_map
        func.restype = None
        func.argtypes = [
            np.ctypeslib.ndpointer(np.float64, ndim=2, flags="C,A"),
            np.ctypeslib.ndpointer(np.float64, ndim=2, flags="C,A"),
            np.ctypeslib.ndpointer(np.float64, ndim=2, flags="C,A,W"),
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        channels = []
        for ch in range(3):
            Dc = np.require(np.ascontiguousarray(Dc_rgb[:, :, ch]), np.float64, ["C", "ALIGNED"])
            out = np.zeros_like(Dc)
            func(Dc, z, out, p, f, eps, H, W, int(iterations))
            channels.append(out)
        res = np.stack(channels, axis=2)
        return res if np.isfinite(res).all() else None

    print("[dll] no known illuminant entry point found in the library", file=sys.stderr)
    return None


def _illuminant_cpu_window(
    D: np.ndarray,
    depths: np.ndarray,
    p: float,
    f: float,
    eps: float,
    iterations: int,
    window_size: int,
) -> np.ndarray:
    """NumPy CPU fallback that mirrors the CUDA window + depth-threshold update.

    Same maths as the old version but ~10x faster: the (iteration-independent)
    neighbour masks/counts are computed once, shifts use padded slicing instead
    of np.roll, and the three channels are processed together in float32.
    """
    h, w = depths.shape
    half = max(0, int(window_size) // 2)
    z = np.nan_to_num(depths, nan=0.0).astype(np.float32)
    zp = np.pad(z, half, mode="constant", constant_values=np.nan)  # NaN => never a neighbour
    dc = np.nan_to_num(D, nan=0.0).astype(np.float32)

    offsets = [(dy, dx) for dy in range(2 * half + 1) for dx in range(2 * half + 1)]
    cache_masks = h * w * len(offsets) <= 600_000_000  # ~600 MB of bool masks max

    def make_mask(dy: int, dx: int) -> np.ndarray:
        return np.abs(zp[dy : dy + h, dx : dx + w] - z) < eps

    masks: list[np.ndarray] = []
    cnt = np.zeros((h, w), dtype=np.float32)
    for dy, dx in offsets:
        m = make_mask(dy, dx)
        cnt += m
        if cache_masks:
            masks.append(m)
    inv_cnt = np.where(cnt > 0, 1.0 / np.maximum(cnt, 1.0), 0.0).astype(np.float32)[
        :, :, None
    ]

    ac = np.zeros_like(dc)
    pf = np.float32(p)
    for _ in range(max(1, int(iterations))):
        acp = np.pad(ac, ((half, half), (half, half), (0, 0)), mode="constant")
        acc = np.zeros_like(ac)
        for k, (dy, dx) in enumerate(offsets):
            m = masks[k] if cache_masks else make_mask(dy, dx)
            acc += acp[dy : dy + h, dx : dx + w] * m[:, :, None]
        ac = dc * pf + acc * inv_cnt * (1.0 - pf)
    return (ac * np.float32(f)).astype(np.float64)


def _depth_components_bfs(depths: np.ndarray, eps: float) -> tuple[np.ndarray, int]:
    """Connected components where neighbor links satisfy |z_n - z_cur| < eps."""
    h, w = depths.shape
    labels = np.zeros((h, w), dtype=np.int32)
    comp_id = 0

    for y in range(h):
        for x in range(w):
            if labels[y, x] != 0:
                continue
            comp_id += 1
            labels[y, x] = comp_id
            q: deque[tuple[int, int]] = deque()
            q.append((y, x))

            while q:
                cy, cx = q.popleft()
                zc = depths[cy, cx]

                if (
                    cy > 0
                    and labels[cy - 1, cx] == 0
                    and abs(depths[cy - 1, cx] - zc) < eps
                ):
                    labels[cy - 1, cx] = comp_id
                    q.append((cy - 1, cx))
                if (
                    cx > 0
                    and labels[cy, cx - 1] == 0
                    and abs(depths[cy, cx - 1] - zc) < eps
                ):
                    labels[cy, cx - 1] = comp_id
                    q.append((cy, cx - 1))
                if (
                    cy < h - 1
                    and labels[cy + 1, cx] == 0
                    and abs(depths[cy + 1, cx] - zc) < eps
                ):
                    labels[cy + 1, cx] = comp_id
                    q.append((cy + 1, cx))
                if (
                    cx < w - 1
                    and labels[cy, cx + 1] == 0
                    and abs(depths[cy, cx + 1] - zc) < eps
                ):
                    labels[cy, cx + 1] = comp_id
                    q.append((cy, cx + 1))

    return labels, comp_id


def _illuminant_cpu_bfs(
    D: np.ndarray,
    depths: np.ndarray,
    p: float,
    f: float,
    eps: float,
    iterations: int,
) -> np.ndarray:
    """CPU reference-like illuminant using connected depth neighborhoods."""
    labels, n_comp = _depth_components_bfs(depths, eps)
    if n_comp <= 0:
        return np.zeros_like(D, dtype=np.float64)

    flat_labels = labels.ravel()
    counts = np.bincount(flat_labels, minlength=n_comp + 1).astype(np.float64)
    counts = np.clip(counts, 1.0, None)

    channels: list[np.ndarray] = []
    for ch in range(3):
        dc = D[:, :, ch]
        ac = np.zeros_like(dc)
        for _ in range(max(1, int(iterations))):
            sums = np.bincount(flat_labels, weights=ac.ravel(), minlength=n_comp + 1)
            means = sums / counts
            ac_p = means[labels]
            ac = dc * p + ac_p * (1.0 - p)
        channels.append(ac * f)

    return np.stack(channels, axis=2)


def compute_illuminant(
    D: np.ndarray,
    depths: np.ndarray,
    p: float = 0.4,
    f: float = 2.0,
    iterations: int = 100,
    window_size: int = DEFAULT_CUDA_WINDOW_SIZE,
    block_x: int = DEFAULT_CUDA_BLOCK_X,
    block_y: int = DEFAULT_CUDA_BLOCK_Y,
    backend: str = "auto",
) -> np.ndarray:
    """Compute illuminant with selectable backend: auto/cuda/window-cpu/bfs-cpu."""
    backend_l = backend.lower()
    eps = max(float(np.mean(depths) / 50.0), 1e-6)

    if backend_l in {"auto", "cuda"}:
        result = _illuminant_dll(
            D,
            depths,
            p,
            f,
            iterations,
            window_size=window_size,
            block_x=block_x,
            block_y=block_y,
        )
        if result is not None:
            return result
        if backend_l == "cuda":
            raise RuntimeError(
                "[illu] CUDA backend requested but the DLL is missing or its CUDA call "
                "failed (see [dll]/[cuda] messages above)"
            )
        print("[illu] CUDA DLL unavailable, falling back to CPU.", file=sys.stderr)

    if backend_l in {"auto", "window-cpu"}:
        print("[illu] Using CPU window backend (slower).")
        return _illuminant_cpu_window(
            D,
            depths,
            p,
            f,
            eps,
            iterations=max(1, int(iterations)),
            window_size=window_size,
        )

    if backend_l == "bfs-cpu":
        print("[illu] Using CPU BFS backend (reference-like, slowest).")
        return _illuminant_cpu_bfs(
            D,
            depths,
            p,
            f,
            eps,
            iterations=max(1, int(iterations)),
        )

    raise ValueError(f"Unsupported illuminant backend: {backend}")


def _att_model(depths, a, b, c, d):
    return a * np.exp(b * depths) + c * np.exp(d * depths)


def _z_model(x, a, b, c, d):
    Ec, depth = x
    beta = np.clip(_att_model(depth, a, b, c, d), 1e-10, None)
    return -np.log(np.clip(Ec, 1e-10, None)) / beta


def estimate_attenuation(
    illuminant: np.ndarray,
    depths: np.ndarray,
    attempts: int = DEFAULT_ATT_ATTEMPTS,
) -> np.ndarray:
    """Returns HxWx3 float64 per-channel attenuation coefficient map."""
    b_lo = np.array([0.0, -np.inf, 0.0, -np.inf])
    b_hi = np.array([np.inf, 0.0, np.inf, 0.0])
    lo = np.array([0.0, -10.0, 0.0, -10.0])
    hi = np.array([100.0, 0.0, 100.0, 0.0])

    channels = []
    for ch in range(3):
        Ec = illuminant[:, :, ch].ravel()
        z = depths.ravel()
        good = np.isfinite(Ec) & np.isfinite(z) & (Ec > 1e-5)
        Ec_g, z_g = Ec[good], z[good]
        if Ec_g.size < 16:
            print(f"  [att] ch={ch}  insufficient valid points")
            channels.append(np.zeros_like(depths, dtype=np.float64))
            continue
        Ec_g, z_g = _subsample(Ec_g, z_g)
        beta_hat = -np.log(Ec_g) / np.clip(z_g, 1e-6, None)

        best_loss, best_p = np.inf, np.array([0.5, -0.5, 0.5, -0.5])
        for _ in range(attempts):
            p0 = np.random.rand(4) * (hi - lo) + lo
            try:
                popt, _ = scipy.optimize.curve_fit(
                    _att_model, z_g, beta_hat, p0, bounds=(b_lo, b_hi), maxfev=2000
                )
                # Refinement step using z-model
                locs = (
                    (Ec_g > 1e-5)
                    & (z_g < np.percentile(z_g, 95))
                    & (z_g > np.percentile(z_g, 10))
                )
                if locs.sum() > 4:
                    try:
                        popt2, _ = scipy.optimize.curve_fit(
                            _z_model,
                            (Ec_g[locs], z_g[locs]),
                            z_g[locs],
                            popt,
                            bounds=(b_lo, b_hi),
                            maxfev=2000,
                        )
                        popt = popt2
                    except (RuntimeError, ValueError):
                        pass
                loss = float(np.mean((_z_model((Ec_g, z_g), *popt) - z_g) ** 2))
                if loss < best_loss:
                    best_loss, best_p = loss, popt
            except (RuntimeError, ValueError):
                pass
        channels.append(_att_model(depths, *best_p))
        print(f"  [att] ch={ch}  mse={best_loss:.6f}")
    return np.stack(channels, axis=2)


def _estimate_attenuation_torch(
    illuminant: np.ndarray,
    depths: np.ndarray,
    attempts: int = DEFAULT_ATT_ATTEMPTS,
    fit_steps: int = DEFAULT_GPU_FIT_STEPS,
) -> np.ndarray:
    """GPU attenuation fit via vectorized multi-restart Adam optimization."""
    if not torch.cuda.is_available():
        return estimate_attenuation(illuminant, depths, attempts=attempts)

    device = torch.device("cuda")
    z_map = torch.from_numpy(depths.astype(np.float32)).to(device)

    channels = []
    A = max(6, int(attempts))
    steps = max(30, int(fit_steps))
    for ch in range(3):
        Ec = illuminant[:, :, ch].ravel()
        z = depths.ravel()
        good = np.isfinite(Ec) & np.isfinite(z) & (Ec > 1e-5)
        Ec_g, z_g = Ec[good], z[good]
        if Ec_g.size < 16:
            channels.append(np.zeros_like(depths, dtype=np.float64))
            print(f"  [att-gpu] ch={ch}  insufficient valid points")
            continue
        Ec_g, z_g = _subsample(Ec_g, z_g)

        beta_hat = -np.log(np.clip(Ec_g, 1e-10, None)) / np.clip(z_g, 1e-6, None)
        beta_hat = np.clip(beta_hat, 0.0, 120.0)

        z_t = torch.from_numpy(z_g.astype(np.float32)).to(device)
        b_t = torch.from_numpy(beta_hat.astype(np.float32)).to(device)

        raw = torch.randn(
            (A, 4), device=device, dtype=torch.float32, requires_grad=True
        )
        opt = torch.optim.Adam([raw], lr=0.08)
        for _ in range(steps):
            a = 100.0 * torch.sigmoid(raw[:, 0:1])
            b = -10.0 * torch.sigmoid(raw[:, 1:2])
            c = 100.0 * torch.sigmoid(raw[:, 2:3])
            d = -10.0 * torch.sigmoid(raw[:, 3:4])

            z_in = z_t.unsqueeze(0)
            pred = a * torch.exp(b * z_in) + c * torch.exp(d * z_in)
            mse = ((pred - b_t.unsqueeze(0)) ** 2).mean(dim=1)
            loss = mse.mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        with torch.no_grad():
            a = 100.0 * torch.sigmoid(raw[:, 0:1])
            b = -10.0 * torch.sigmoid(raw[:, 1:2])
            c = 100.0 * torch.sigmoid(raw[:, 2:3])
            d = -10.0 * torch.sigmoid(raw[:, 3:4])

            pred = a * torch.exp(b * z_t.unsqueeze(0)) + c * torch.exp(
                d * z_t.unsqueeze(0)
            )
            mse = ((pred - b_t.unsqueeze(0)) ** 2).mean(dim=1)
            mse = torch.nan_to_num(mse, nan=float("inf"))
            best = int(torch.argmin(mse).item())

            aa = a[best, 0]
            bb = b[best, 0]
            cc = c[best, 0]
            dd = d[best, 0]
            full = aa * torch.exp(bb * z_map) + cc * torch.exp(dd * z_map)
            print(f"  [att-gpu] ch={ch}  mse={float(mse[best].item()):.6f}")
            channels.append(full.detach().cpu().numpy().astype(np.float64))

    return np.stack(channels, axis=2)


def _binned_beta_samples(
    illu_ch: np.ndarray, depths: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Per-depth-bin attenuation samples: beta_bin = -ln(mean(E)) / median(z).

    Averaging E *in the linear domain* inside each equal-count depth bin is robust
    to dark (E ~ 0) pixels, whose per-pixel beta_hat = -ln(E)/z is huge and used to
    dominate the least-squares fit (a per-pixel median of beta_hat was clearly worse
    in synthetic tests). Weights are sqrt(pixel count) per bin.
    """
    Ec = illu_ch.ravel()
    z = depths.ravel()
    good = np.isfinite(Ec) & np.isfinite(z) & (z > 1e-3)
    Ec, z = Ec[good], z[good]
    if Ec.size < ATT_NUM_BINS * ATT_MIN_BIN_COUNT // 4:
        return None
    order = np.argsort(z, kind="stable")
    zs, es = z[order], Ec[order]
    nb = int(min(ATT_NUM_BINS, max(4, Ec.size // 200)))
    zc, bc, wc = [], [], []
    for zz, ee in zip(np.array_split(zs, nb), np.array_split(es, nb)):
        if zz.size < ATT_MIN_BIN_COUNT:
            continue
        z_med = float(np.median(zz))
        e_mean = float(np.mean(ee))
        if z_med <= 1e-3 or e_mean <= 1e-6:
            continue  # fully black bin: no information about attenuation
        zc.append(z_med)
        bc.append(-math.log(e_mean) / z_med)
        wc.append(math.sqrt(zz.size))
    if len(zc) < 4:
        return None
    return np.array(zc), np.array(bc), np.array(wc)


def _fit_att_curve(
    zc: np.ndarray, bc: np.ndarray, wc: np.ndarray, attempts: int
) -> tuple[np.ndarray, float]:
    """Weighted multi-restart fit of beta(z)=a*exp(b*z)+c*exp(d*z) on binned data."""
    lo = np.array([0.0, -50.0, 0.0, -50.0])
    hi = np.array([100.0, 0.0, 100.0, 0.0])
    w2 = wc**2

    def wmse(p: np.ndarray) -> float:
        return float(np.sum(w2 * (_att_model(zc, *p) - bc) ** 2) / np.sum(w2))

    # Safe fallback: constant beta (always valid, bounds allow b=d=0).
    const = float(np.average(bc, weights=wc))
    best_p = np.array([max(const, 0.0), 0.0, 0.0, 0.0])
    best_loss = wmse(best_p)

    amax = max(2.0 * float(np.max(bc)), 0.5)
    for _ in range(max(1, int(attempts))):
        p0 = np.array(
            [
                np.random.rand() * amax,
                -np.random.rand() * 3.0,
                np.random.rand() * amax,
                -np.random.rand() * 3.0,
            ]
        )
        try:
            popt, _ = scipy.optimize.curve_fit(
                _att_model, zc, bc, p0, bounds=(lo, hi), sigma=1.0 / wc, maxfev=4000
            )
        except (RuntimeError, ValueError):
            continue
        loss = wmse(popt)
        if np.isfinite(loss) and loss < best_loss:
            best_loss, best_p = loss, popt
    return best_p, best_loss


def estimate_attenuation_robust(
    illuminant: np.ndarray,
    depths: np.ndarray,
    attempts: int = DEFAULT_ATT_ATTEMPTS,
) -> np.ndarray:
    """Robust per-channel attenuation map (per-depth-bin statistics + weighted curve fit).

    Fits ~48 depth-bin samples instead of millions of noisy pixels, so it is fast,
    needs no GPU memory and is not dominated by dark/saturated outliers.
    """
    channels = []
    zref = float(np.median(depths))
    for ch in range(3):
        samples = _binned_beta_samples(illuminant[:, :, ch], depths)
        if samples is None:
            print(f"  [att] ch={ch}  insufficient valid points")
            channels.append(np.zeros_like(depths, dtype=np.float64))
            continue
        zc, bc, wc = samples
        p, loss = _fit_att_curve(zc, bc, wc, attempts)
        beta_ref = float(_att_model(np.array([zref]), *p)[0])
        print(
            f"  [att] ch={ch}  wmse={loss:.6f}  beta(z={zref:.2f})={beta_ref:.3f}  bins={len(zc)}"
        )
        channels.append(_att_model(depths, *p))
    return np.stack(channels, axis=2)


def enforce_attenuation_order(att: np.ndarray) -> np.ndarray:
    """Physical prior for water: attenuation grows toward long wavelengths (R >= G >= B).

    Removes the channel inversions caused by noisy fits, which show up as
    magenta/green color casts after recovery.
    """
    out = att.copy()
    g = np.maximum(att[:, :, 1], att[:, :, 2])
    out[:, :, 1] = g
    out[:, :, 0] = np.maximum(att[:, :, 0], g)
    return out


def recover_image(
    Da: np.ndarray,
    attenuation: np.ndarray,
    depths: np.ndarray,
    beta_depth_cap: float = 1.2,
) -> np.ndarray:
    """Undo depth-dependent attenuation to recover scene radiance (Eq. in paper)."""
    beta_D = attenuation * depths[:, :, np.newaxis]
    # Optional cap; set <= 0 to disable clipping and match original recovery behavior.
    if beta_depth_cap > 0:
        beta_D = np.clip(beta_D, 0.0, beta_depth_cap)
    Ja = Da * np.exp(beta_D)
    return np.clip(Ja, 0, 1)


def sea_thru(
    image: np.ndarray,
    depths: np.ndarray,
    illuminant_iters: int = DEFAULT_ILLUMINANT_ITERS,
    illuminant_window_size: int = DEFAULT_CUDA_WINDOW_SIZE,
    illuminant_block_x: int = DEFAULT_CUDA_BLOCK_X,
    illuminant_block_y: int = DEFAULT_CUDA_BLOCK_Y,
    illuminant_backend: str = "auto",
    bs_attempts: int = DEFAULT_BS_ATTEMPTS,
    att_attempts: int = DEFAULT_ATT_ATTEMPTS,
    use_gpu_fits: bool = True,
    fit_steps: int = DEFAULT_GPU_FIT_STEPS,
    ref_frac: float = 0.03,
    beta_depth_cap: float = 1.2,
    attenuation_cap: float = 1.2,
    return_intermediates: bool = False,
    illuminant_p: float = 0.4,
    illuminant_f: float = 2.0,
    att_method: str = "robust",
    att_order: str = "rgb",
) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Full Sea-Thru pipeline.
      image  : HxWx3 float64 [0,1]
      depths : HxW  float64 (larger = farther)
    Returns    HxWx3 float64 [0,1] restored image.
    """
    depths = np.nan_to_num(depths, nan=float(np.nanmean(depths)), posinf=None, neginf=None)
    gpu_ok = use_gpu_fits and torch.cuda.is_available()

    print("[sea-thru] Estimating backscatter...")
    Ba = None
    if gpu_ok:
        try:
            Ba = _estimate_backscatter_torch(
                image,
                depths,
                attempts=bs_attempts,
                ref_frac=ref_frac,
                fit_steps=fit_steps,
            )
        except RuntimeError as e:  # includes torch.cuda.OutOfMemoryError
            print(f"  [bs-gpu] failed ({e}); falling back to CPU fit", file=sys.stderr)
            torch.cuda.empty_cache()
    if Ba is None:
        Ba = estimate_backscatter(
            image, depths, attempts=bs_attempts, ref_frac=ref_frac
        )
    Ba = np.nan_to_num(Ba, nan=0.0)
    Da = np.clip(image - Ba, 0, 1)

    print("[sea-thru] Computing illuminant map...")
    illu = compute_illuminant(
        Da,
        depths,
        p=illuminant_p,
        f=illuminant_f,
        iterations=illuminant_iters,
        window_size=illuminant_window_size,
        block_x=illuminant_block_x,
        block_y=illuminant_block_y,
        backend=illuminant_backend,
    )
    illu = np.clip(np.nan_to_num(illu, nan=0.0), 0, 1)

    print("[sea-thru] Estimating wideband attenuation...")
    att = None
    if att_method == "robust":
        att = estimate_attenuation_robust(illu, depths, attempts=att_attempts)
    elif gpu_ok:
        try:
            att = _estimate_attenuation_torch(
                illu,
                depths,
                attempts=att_attempts,
                fit_steps=fit_steps,
            )
        except RuntimeError as e:  # includes torch.cuda.OutOfMemoryError
            print(f"  [att-gpu] failed ({e}); falling back to CPU fit", file=sys.stderr)
            torch.cuda.empty_cache()
    if att is None:
        att = estimate_attenuation(illu, depths, attempts=att_attempts)
    att = np.nan_to_num(att, nan=0.0)
    if att_order == "rgb":
        att = enforce_attenuation_order(att)
    # Optional cap; set <= 0 to disable and match original attenuation behavior.
    if attenuation_cap > 0:
        att = np.clip(att, 0.0, attenuation_cap)

    if beta_depth_cap > 0:
        hit = (att * depths[:, :, None] > beta_depth_cap).mean(axis=(0, 1)) * 100.0
        print(
            f"  [cap] pixels at the beta*z cap  R={hit[0]:.0f}%  G={hit[1]:.0f}%  B={hit[2]:.0f}%"
            "  (high values => raise --beta-depth-cap or lower --far)"
        )

    print("[sea-thru] Recovering scene radiance...")
    Ja = recover_image(Da, att, depths, beta_depth_cap=beta_depth_cap)
    if return_intermediates:
        return Ja, Da, illu, att, Ba
    return Ja


def _resolve_depths_for_image(
    img: np.ndarray,
    img_path: Path,
    mode: str,
    estimator: MiDaS | None,
    depth_source: str | None,
    hint_source: str | None,
    z_min: float,
    z_inf: float,
    depth_debias_sigma_frac: float,
    far_override: float | None = None,
    depth_shift: float = DEFAULT_DEPTH_SHIFT,
    depth_detail_size: int = 0,
) -> np.ndarray:
    mode_l = mode.lower()
    if mode_l not in {"predict", "map", "hybrid"}:
        raise ValueError(f"Unsupported mode: {mode}")

    def _predict_depth() -> np.ndarray:
        if estimator is None:
            raise RuntimeError(
                f"{mode} mode requested but MiDaS estimator is not initialized"
            )
        depths = estimator.predict_depth(
            img.astype(np.float32),
            debias_sigma_frac=depth_debias_sigma_frac,
            depth_shift=depth_shift,
            detail_size=depth_detail_size,
        )
        if hint_source:
            hp = _resolve_depth_path_for_image(hint_source, img_path)
            hint_depths = read_depthmap(str(hp), img.shape[:2])
            depths = refine_depths_from_hint(depths, float(np.mean(hint_depths)))
        else:
            depths = preprocess_predicted_depths(img, depths, far_override)
        return depths

    if mode_l == "predict":
        return _predict_depth()

    depth_map = None
    if depth_source:
        try:
            dp = _resolve_depth_path_for_image(depth_source, img_path)
            depth_map = read_depthmap(str(dp), img.shape[:2])
            depth_map = normalize_depth_map(depth_map, z_min=z_min, z_inf=z_inf)
        except FileNotFoundError as e:
            if mode_l == "map":
                raise
            print(
                f"[depth] {e}. Falling back to MiDaS prediction for {img_path.name}",
                file=sys.stderr,
            )

    if mode_l == "map":
        if depth_map is None:
            raise ValueError(f"Mode '{mode}' requires --depth")
        return depth_map

    predicted = _predict_depth()
    if depth_map is None:
        return predicted
    return combine_map_predict(depth_map, predicted)


def main():
    ap = argparse.ArgumentParser(
        description="Sea-Thru Underwater Image Restoration (GPU-accelerated)"
    )
    ap.add_argument("input", nargs="+", help="Input image path(s)")
    ap.add_argument(
        "--output", "-o", default="output", help="Output directory (default: output)"
    )
    ap.add_argument(
        "--prefix",
        default="",
        help="Optional filename prefix for all generated outputs",
    )
    ap.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of input images processed per batch report (default: 1)",
    )
    ap.add_argument(
        "--recursive",
        action="store_true",
        help="When an input is a directory, include images in subdirectories",
    )
    ap.add_argument(
        "--max-side",
        type=int,
        default=MAX_SIDE,
        help=f"Resize longest side to N pixels (default: {MAX_SIDE})",
    )
    ap.add_argument(
        "--model",
        default="MiDaS_small",
        choices=["MiDaS_small", "DPT_Hybrid", "DPT_Large"],
        help="MiDaS depth model (default: MiDaS_small)",
    )
    ap.add_argument(
        "--mode",
        default="Predict",
        choices=["Predict", "Map", "Hybrid", "predict", "map", "hybrid"],
        help="Depth mode: Predict (MiDaS), Map (provided depth), Hybrid (map + MiDaS, with MiDaS fallback if map is missing)",
    )
    ap.add_argument(
        "--depth",
        default=None,
        help="Depth map file or directory (required for Map/Hybrid)",
    )
    ap.add_argument(
        "--hint",
        default=None,
        help="Optional depth hint file or directory used to rescale predicted depth",
    )
    ap.add_argument(
        "--depth-debias-sigma-frac",
        type=float,
        default=0.0,
        help="If >0, removes low-frequency disparity bias before depth conversion. Recommended 0.03-0.08 underwater.",
    )
    ap.add_argument(
        "--depth-shift",
        type=float,
        default=DEFAULT_DEPTH_SHIFT,
        help="Predict mode: disparity offset in depth=10/(9.5*disp+shift) (default: 2.0; "
        "old behaviour was 0.5). Larger values compress the far tail of the depth map, "
        "which avoids color blotches where dark regions are mistaken for distant ones",
    )
    ap.add_argument(
        "--depth-detail-size",
        type=int,
        default=0,
        help="EXPERIMENTAL, Predict mode: also run MiDaS at this long side (e.g. 768) and "
        "merge its fine detail into the standard-resolution depth (0 = off, default). "
        "Falls back to single scale if it fails",
    )
    ap.add_argument(
        "--far",
        type=float,
        default=None,
        help="Manual scene far distance (max depth, in meters) for Predict mode; "
        "overrides the automatic luminance-based estimate (which can be far too large "
        "for close-ups lit by ROV lights)",
    )
    ap.add_argument(
        "--z-min",
        type=float,
        default=0.1,
        help="Minimum valid depth when loading a depth map (default: 0.1)",
    )
    ap.add_argument(
        "--z-inf",
        type=float,
        default=6.0,
        help="Fallback depth value used for zero-valued map pixels (default: 6.0)",
    )
    ap.add_argument(
        "--illuminant-iters",
        type=int,
        default=DEFAULT_ILLUMINANT_ITERS,
        help=f"Illuminant iterations for CUDA DLL backend (default: {DEFAULT_ILLUMINANT_ITERS})",
    )
    ap.add_argument(
        "--illuminant-window-size",
        type=int,
        default=DEFAULT_CUDA_WINDOW_SIZE,
        help=f"CUDA illuminant neighborhood window size (default: {DEFAULT_CUDA_WINDOW_SIZE})",
    )
    ap.add_argument(
        "--illuminant-block-x",
        type=int,
        default=DEFAULT_CUDA_BLOCK_X,
        help=f"CUDA block X dimension for illuminant kernel (default: {DEFAULT_CUDA_BLOCK_X})",
    )
    ap.add_argument(
        "--illuminant-block-y",
        type=int,
        default=DEFAULT_CUDA_BLOCK_Y,
        help=f"CUDA block Y dimension for illuminant kernel (default: {DEFAULT_CUDA_BLOCK_Y})",
    )
    ap.add_argument(
        "--illuminant-backend",
        default="auto",
        choices=["auto", "cuda", "window-cpu", "bfs-cpu"],
        help="Illuminant backend: auto (prefer CUDA), cuda, window-cpu, bfs-cpu",
    )
    ap.add_argument(
        "--save-depth",
        dest="save_depth",
        action="store_true",
        help="Save depth map output (PNG)",
    )
    ap.add_argument(
        "--no-save-depth",
        dest="save_depth",
        action="store_false",
        help="Disable saving depth map outputs",
    )
    ap.set_defaults(save_depth=True)
    ap.add_argument(
        "--save-depth-npy",
        action="store_true",
        help="Also save raw depth arrays as .npy files",
    )
    ap.add_argument(
        "--save-depth-gray",
        action="store_true",
        help="Also save depth as normalized grayscale PNG (16-bit)",
    )
    ap.add_argument(
        "--depth-colormap",
        default="magma",
        choices=["magma", "gray", "inferno", "viridis", "turbo"],
        help="Colormap of the *_depth.png visualization (default: magma). "
        "Visualization only: it never affects the restoration.",
    )
    ap.add_argument(
        "--depth-vis-min",
        type=float,
        default=None,
        help="Fixed minimum depth for visualization; defaults to per-image min",
    )
    ap.add_argument(
        "--depth-vis-max",
        type=float,
        default=None,
        help="Fixed maximum depth for visualization; defaults to per-image max",
    )
    ap.add_argument(
        "--bs-attempts",
        type=int,
        default=DEFAULT_BS_ATTEMPTS,
        help=f"Random-restart attempts for backscatter fit (default: {DEFAULT_BS_ATTEMPTS})",
    )
    ap.add_argument(
        "--att-attempts",
        type=int,
        default=DEFAULT_ATT_ATTEMPTS,
        help=f"Random-restart attempts for attenuation fit (default: {DEFAULT_ATT_ATTEMPTS})",
    )
    ap.add_argument(
        "--fit-steps",
        type=int,
        default=DEFAULT_GPU_FIT_STEPS,
        help=f"Optimization steps for GPU fitting (default: {DEFAULT_GPU_FIT_STEPS})",
    )
    ap.add_argument(
        "--cpu-fits",
        action="store_true",
        help="Force scipy CPU fitting for backscatter/attenuation",
    )
    ap.add_argument(
        "--ref-frac",
        type=float,
        default=0.03,
        help="Fraction of darkest pixels per depth bin used as references (default: 0.03)",
    )
    ap.add_argument(
        "--beta-depth-cap",
        type=float,
        default=1.2,
        help="Cap for attenuation*depth exponent term; set <=0 to disable (default: 1.2)",
    )
    ap.add_argument(
        "--att-cap",
        type=float,
        default=1.2,
        help="Upper bound for attenuation coefficients; set <=0 to disable (default: 1.2)",
    )
    ap.add_argument(
        "--postprocess",
        action="store_true",
        help="Apply optional CLAHE + white balance postprocess (original Sea-Thru style)",
    )
    ap.add_argument(
        "--postprocess-mode",
        default="original",
        choices=["original", "fast"],
        help="Postprocess flavor: original (equalize_adapthist) or fast (OpenCV CLAHE)",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for the multi-restart fits (default: 0; makes runs reproducible)",
    )
    ap.add_argument(
        "--illuminant-p",
        type=float,
        default=0.4,
        help="Illuminant: weight of the direct signal vs. neighbourhood average (default: 0.4)",
    )
    ap.add_argument(
        "--illuminant-f",
        type=float,
        default=2.0,
        help="Illuminant: output scale factor (default: 2.0)",
    )
    ap.add_argument(
        "--att-method",
        default="robust",
        choices=["robust", "legacy"],
        help="Attenuation fit: robust (binned medians, default) or legacy (per-pixel fit)",
    )
    ap.add_argument(
        "--att-order",
        default="rgb",
        choices=["rgb", "none"],
        help="rgb: force attenuation R>=G>=B (physical prior, default); none: keep raw fits",
    )
    ap.add_argument(
        "--save-intermediates",
        action="store_true",
        help="Save direct signal and illuminant map images",
    )
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    image_paths = _collect_input_images(args.input, recursive=args.recursive)
    if not image_paths:
        print("No input images found.", file=sys.stderr)
        sys.exit(1)

    mode_l = args.mode.lower()
    needs_midas = mode_l in {"predict", "hybrid"}
    if mode_l == "map" and not args.depth:
        print("--depth is required for mode Map", file=sys.stderr)
        sys.exit(2)

    estimator = MiDaS(args.model) if needs_midas else None

    total = len(image_paths)
    batch_count = math.ceil(total / max(1, args.batch_size))
    done = 0
    for bi, batch in enumerate(_iter_batches(image_paths, args.batch_size), start=1):
        print(
            f"\n{'#'*60}\nBatch {bi}/{batch_count} | size={len(batch)} | processed={done}/{total}\n{'#'*60}"
        )
        for p in batch:
            print(f"\n{'='*60}\nProcessing: {p.name}\n{'='*60}")
            try:
                img = read_image(str(p), args.max_side)
            except FileNotFoundError as e:
                print(e, file=sys.stderr)
                continue

            print(f"  Size: {img.shape[1]}x{img.shape[0]}")
            try:
                depths = _resolve_depths_for_image(
                    img,
                    p,
                    mode=args.mode,
                    estimator=estimator,
                    depth_source=args.depth,
                    hint_source=args.hint,
                    z_min=args.z_min,
                    z_inf=args.z_inf,
                    depth_debias_sigma_frac=args.depth_debias_sigma_frac,
                    far_override=args.far,
                    depth_shift=args.depth_shift,
                    depth_detail_size=args.depth_detail_size,
                )
            except Exception as e:
                print(f"Depth preparation failed for {p.name}: {e}", file=sys.stderr)
                continue
            print(
                "  Depth stats: "
                f"min={depths.min():.2f}, p5={np.percentile(depths, 5):.2f}, "
                f"p50={np.percentile(depths, 50):.2f}, p95={np.percentile(depths, 95):.2f}, "
                f"max={depths.max():.2f} m"
            )

            if args.save_depth:
                d_vis = depth_to_vis(
                    depths,
                    depth_vis_min=args.depth_vis_min,
                    depth_vis_max=args.depth_vis_max,
                    colormap=args.depth_colormap,
                )
                d_png = out_dir / (args.prefix + p.stem + "_depth.png")
                if not cv2.imwrite(str(d_png), d_vis):
                    print(f"  [warn] could not write {d_png}", file=sys.stderr)
                if args.save_depth_gray:
                    d_gray = depth_to_gray(
                        depths,
                        depth_vis_min=args.depth_vis_min,
                        depth_vis_max=args.depth_vis_max,
                        bits=16,
                    )
                    d_gray_png = out_dir / (args.prefix + p.stem + "_depth_gray.png")
                    cv2.imwrite(str(d_gray_png), d_gray)
            if args.save_depth_npy:
                d_npy = out_dir / (args.prefix + p.stem + "_depth.npy")
                np.save(str(d_npy), depths.astype(np.float32))

            try:
                sea_out = sea_thru(
                    img,
                    depths,
                    illuminant_iters=args.illuminant_iters,
                    illuminant_window_size=args.illuminant_window_size,
                    illuminant_block_x=args.illuminant_block_x,
                    illuminant_block_y=args.illuminant_block_y,
                    illuminant_backend=args.illuminant_backend,
                    bs_attempts=args.bs_attempts,
                    att_attempts=args.att_attempts,
                    use_gpu_fits=not args.cpu_fits,
                    fit_steps=args.fit_steps,
                    ref_frac=args.ref_frac,
                    beta_depth_cap=args.beta_depth_cap,
                    attenuation_cap=args.att_cap,
                    return_intermediates=args.save_intermediates,
                    illuminant_p=args.illuminant_p,
                    illuminant_f=args.illuminant_f,
                    att_method=args.att_method,
                    att_order=args.att_order,
                )
            except Exception as e:
                print(f"Sea-Thru failed for {p.name}: {e}", file=sys.stderr)
                traceback.print_exc()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue

            if args.save_intermediates:
                restored, direct_signal, illuminant, _, _ = sea_out
                save_image(
                    str(out_dir / (args.prefix + p.stem + "_direct_signal.png")),
                    direct_signal,
                )
                save_image(
                    str(out_dir / (args.prefix + p.stem + "_illuminant_map.png")),
                    np.clip(illuminant, 0, 1),
                )
            else:
                restored = sea_out

            if args.postprocess:
                restored = postprocess_image(restored, mode=args.postprocess_mode)

            out_path = out_dir / (args.prefix + p.stem + "_restored.png")
            save_image(str(out_path), restored)
            print(f"  Saved -> {out_path}")
            done += 1
            if torch.cuda.is_available():
                torch.cuda.empty_cache()  # keep VRAM free for the next image / CUDA DLL


if __name__ == "__main__":
    main()
