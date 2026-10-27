"""
Degradation simulation pipeline — Section 3.5, Table 3.1.

Fix #2 — Low-light convention (explicit):
  The config uses gamma values {0.7, 0.5, 0.3} with the convention
      out = in^(1/gamma)
  Since gamma < 1, the exponent 1/gamma > 1 and the image darkens
  (e.g. in=0.5, gamma=0.5 -> 0.5^2 = 0.25).  The docstring and the
  function below state this explicitly so Table 3.1 is unambiguous.
  If someone mistakenly used out = in^gamma with these numbers, the
  image would brighten — the opposite of "low light".

All functions operate on float32 images in [0, 1], shape HxWx3, RGB.
They are deterministic given the numpy RNG state so that training can
be reproducible per seed.
"""
from __future__ import annotations

import cv2
import numpy as np


# ------------------------------------------------------------------ helpers

def _ensure_float01(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0
    return np.clip(img.astype(np.float32), 0.0, 1.0)


def _to_uint8(img01: np.ndarray) -> np.ndarray:
    return np.clip(img01 * 255.0, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------- degradations

def fog(img: np.ndarray, severity: str = "moderate") -> np.ndarray:
    """
    Atmospheric scattering: blend with a uniform fog layer.
    Light / moderate / heavy map to alpha 0.30 / 0.55 / 0.75.
    """
    alpha_map = {"light": 0.30, "moderate": 0.55, "heavy": 0.75}
    alpha = alpha_map.get(severity, 0.55)
    # Fog colour: slight desaturation toward light grey-blue
    fog_colour = np.array([0.82, 0.82, 0.85], dtype=np.float32)
    x = _ensure_float01(img)
    # Per-pixel fog that is slightly stronger in the distance (top of image
    # is farther when camera looks forward). Simple vertical gradient.
    h = x.shape[0]
    gradient = np.linspace(0.85, 1.15, h, dtype=np.float32)[:, None, None]
    a = np.clip(alpha * gradient, 0, 0.95)
    # Broadcast fog colour
    out = (1 - a) * x + a * fog_colour
    return np.clip(out, 0, 1)


def motion_blur(img: np.ndarray, kernel_size: int = 9, angle_deg: float | None = None) -> np.ndarray:
    """
    Linear motion blur with random angle if not specified.
    Kernel sizes: 5 / 9 / 15 px.
    """
    x = _ensure_float01(img)
    if angle_deg is None:
        angle_deg = float(np.random.uniform(0, 180))
    k = int(kernel_size)
    if k % 2 == 0:
        k += 1
    # Build horizontal kernel then rotate
    kernel = np.zeros((k, k), dtype=np.float32)
    kernel[k // 2, :] = 1.0 / k
    M = cv2.getRotationMatrix2D((k / 2, k / 2), angle_deg, 1)
    kernel = cv2.warpAffine(kernel, M, (k, k), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    kernel = kernel / (kernel.sum() + 1e-8)
    out = cv2.filter2D(x, -1, kernel)
    return np.clip(out, 0, 1)


def low_light(
    img: np.ndarray,
    gamma: float = 0.5,
    brightness_scale: float = 0.75,
) -> np.ndarray:
    """
    Low-light via gamma darkening.

    Convention (Fix #2):  out = in^(1/gamma)  with gamma in {0.7, 0.5, 0.3}.
    Because gamma < 1, exponent 1/gamma > 1 -> image darkens.
    Equivalently, out = in^gamma' with gamma' in {1.43, 2.0, 3.33}.

    A linear brightness_scale is applied after the gamma step to mimic
    reduced illumination (see config: 0.90 / 0.75 / 0.60).
    """
    x = _ensure_float01(img)
    if gamma <= 0:
        raise ValueError("gamma must be > 0")
    exponent = 1.0 / gamma  # >1 when gamma<1 -> darkens
    out = np.power(np.clip(x, 0, 1), exponent)
    out = out * float(brightness_scale)
    return np.clip(out, 0, 1)


def gaussian_noise(img: np.ndarray, sigma: float = 0.05) -> np.ndarray:
    """Additive Gaussian noise, sigma in {0.02, 0.05, 0.10}."""
    x = _ensure_float01(img)
    noise = np.random.normal(0, sigma, size=x.shape).astype(np.float32)
    return np.clip(x + noise, 0, 1)


def rain(img: np.ndarray, severity: str = "moderate") -> np.ndarray:
    """
    Synthetic rain streaks: directional line noise overlay.
    """
    count_map = {"light": 40, "moderate": 90, "heavy": 160}
    n = count_map.get(severity, 90)
    x = _ensure_float01(img)
    h, w = x.shape[:2]
    overlay = x.copy()
    # Rain colour: slightly bright streaks with small blue tint
    for _ in range(n):
        x0 = int(np.random.uniform(0, w))
        y0 = int(np.random.uniform(0, h * 0.3))
        length = int(np.random.uniform(h * 0.15, h * 0.35))
        thickness = int(np.random.choice([1, 1, 2]))
        angle = float(np.random.uniform(-15, 15))  # mostly vertical
        x1 = int(x0 + length * np.sin(np.deg2rad(angle)))
        y1 = int(y0 + length * np.cos(np.deg2rad(angle)))
        alpha = float(np.random.uniform(0.12, 0.32))
        colour = np.array([0.75, 0.78, 0.88], dtype=np.float32) * alpha
        # Draw on overlay by alpha-blending a line (approximate with cv2 on uint8)
        tmp = _to_uint8(overlay)
        cv2.line(tmp, (x0, y0), (x1, y1), tuple((colour * 255).astype(int).tolist()), thickness, cv2.LINE_AA)
        overlay = tmp.astype(np.float32) / 255.0
        # Blend lightly so streaks are translucent
        overlay = np.clip(overlay, 0, 1)
    # Final blend to keep rain subtle
    out = 0.88 * x + 0.12 * overlay + 0.0  # overlay already blended
    # Instead do proper alpha blend: overlay already contains blended lines
    out = overlay  # overlay was built incrementally from x
    return np.clip(out, 0, 1)


def occlusion(img: np.ndarray, fraction: float = 0.25) -> np.ndarray:
    """
    Random rectangular patch masking — fraction of image area (0.10 / 0.25 / 0.40).
    Fill colour is random grey to mimic an occluding object, not black.
    """
    x = _ensure_float01(img).copy()
    h, w = x.shape[:2]
    area = h * w * float(fraction)
    # Random aspect ratio
    aspect = float(np.random.uniform(0.5, 2.0))
    rh = int(np.sqrt(area * aspect))
    rw = int(np.sqrt(area / aspect))
    rh = min(rh, h - 1)
    rw = min(rw, w - 1)
    y0 = int(np.random.uniform(0, h - rh))
    x0 = int(np.random.uniform(0, w - rw))
    # Occluder colour: random mid-grey with slight variation
    grey = float(np.random.uniform(0.35, 0.65))
    x[y0 : y0 + rh, x0 : x0 + rw] = grey
    return x


# ---------------------------------------------------------------- registry

DEGRADATION_FNS: dict[str, callable] = {
    "fog": fog,
    "motion_blur": motion_blur,
    "low_light": low_light,
    "gaussian_noise": gaussian_noise,
    "rain": rain,
    "occlusion": occlusion,
}

# Severity mappings for random choice during training
FOG_LEVELS = ["light", "moderate", "heavy"]
BLUR_KERNELS = [5, 9, 15]
LOW_LIGHT_GAMMAS = [0.7, 0.5, 0.3]
LOW_LIGHT_BRIGHTNESS = {0.7: 0.90, 0.5: 0.75, 0.3: 0.60}
NOISE_SIGMAS = [0.02, 0.05, 0.10]
RAIN_LEVELS = ["light", "moderate", "heavy"]
OCCLUSION_FRACS = [0.10, 0.25, 0.40]


def apply_random_degradation(img: np.ndarray, rng: np.random.Generator | None = None) -> tuple[np.ndarray, str]:
    """
    Pick a random degradation type + severity and apply it.
    Returns (degraded_image, label_str) e.g. "low_light/gamma0.5".
    """
    if rng is None:
        rng = np.random.default_rng()

    choice = rng.choice(list(DEGRADATION_FNS.keys()))

    if choice == "fog":
        sev = rng.choice(FOG_LEVELS)
        return fog(img, severity=str(sev)), f"fog/{sev}"
    elif choice == "motion_blur":
        k = int(rng.choice(BLUR_KERNELS))
        ang = float(rng.uniform(0, 180))
        return motion_blur(img, kernel_size=k, angle_deg=ang), f"motion_blur/k{k}"
    elif choice == "low_light":
        g = float(rng.choice(LOW_LIGHT_GAMMAS))
        b = LOW_LIGHT_BRIGHTNESS[g]
        return low_light(img, gamma=g, brightness_scale=b), f"low_light/gamma{g}"
    elif choice == "gaussian_noise":
        s = float(rng.choice(NOISE_SIGMAS))
        return gaussian_noise(img, sigma=s), f"gaussian_noise/sigma{s}"
    elif choice == "rain":
        sev = rng.choice(RAIN_LEVELS)
        return rain(img, severity=str(sev)), f"rain/{sev}"
    else:  # occlusion
        frac = float(rng.choice(OCCLUSION_FRACS))
        return occlusion(img, fraction=frac), f"occlusion/{frac:.2f}"


def apply_degradation_by_name(
    img: np.ndarray,
    name: str,
    severity: str | float | int = "moderate",
) -> np.ndarray:
    """
    Deterministic single-degradation apply for evaluation grids.

    name: one of DEGRADATION_FNS keys
    severity: string or numeric level (interpreted per type)
    """
    if name == "fog":
        return fog(img, severity=str(severity))
    elif name == "motion_blur":
        return motion_blur(img, kernel_size=int(severity))
    elif name == "low_light":
        g = float(severity)
        return low_light(img, gamma=g, brightness_scale=LOW_LIGHT_BRIGHTNESS.get(g, 0.75))
    elif name == "gaussian_noise":
        return gaussian_noise(img, sigma=float(severity))
    elif name == "rain":
        return rain(img, severity=str(severity))
    elif name == "occlusion":
        return occlusion(img, fraction=float(severity))
    else:
        raise ValueError(f"Unknown degradation: {name}")
