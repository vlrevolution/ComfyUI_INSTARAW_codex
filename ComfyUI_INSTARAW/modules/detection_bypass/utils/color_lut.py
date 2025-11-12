import numpy as np
import re, os
from PIL import Image

def apply_1d_lut(img_arr: np.ndarray, lut: np.ndarray, strength: float = 1.0) -> np.ndarray:
    if img_arr.ndim != 3 or img_arr.shape[2] != 3:
        raise ValueError("apply_1d_lut expects an HxWx3 image array")
    arr = img_arr.astype(np.float32)
    lut_arr = np.array(lut, dtype=np.float32)
    if lut_arr.ndim == 1:
        lut_arr = np.stack([lut_arr, lut_arr, lut_arr], axis=1)
    if lut_arr.shape[1] != 3:
        raise ValueError("1D LUT must have shape (N,) or (N,3)")
    N = lut_arr.shape[0]
    src_positions = np.linspace(0, 255, N)
    out = np.empty_like(arr)
    for c in range(3):
        channel = arr[..., c].ravel()
        mapped = np.interp(channel, src_positions, lut_arr[:, c])
        out[..., c] = mapped.reshape(arr.shape[0], arr.shape[1])
    out = np.clip(out, 0, 255).astype(np.uint8)
    if strength >= 1.0:
        return out
    else:
        blended = ((1.0 - strength) * img_arr.astype(np.float32) + strength * out.astype(np.float32))
        return np.clip(blended, 0, 255).astype(np.uint8)

def _trilinear_sample_lut(img_float: np.ndarray, lut: np.ndarray) -> np.ndarray:
    S = lut.shape[0]
    if lut.shape[0] != lut.shape[1] or lut.shape[1] != lut.shape[2]:
        raise ValueError("3D LUT must be cubic (SxSxSx3)")
    idx = img_float * (S - 1)
    r_idx, g_idx, b_idx = idx[..., 0], idx[..., 1], idx[..., 2]
    r0, g0, b0 = np.floor(r_idx).astype(np.int32), np.floor(g_idx).astype(np.int32), np.floor(b_idx).astype(np.int32)
    r1, g1, b1 = np.clip(r0 + 1, 0, S - 1), np.clip(g0 + 1, 0, S - 1), np.clip(b0 + 1, 0, S - 1)
    dr, dg, db = (r_idx - r0)[..., None], (g_idx - g0)[..., None], (b_idx - b0)[..., None]
    c000, c001 = lut[r0, g0, b0], lut[r0, g0, b1]
    c010, c011 = lut[r0, g1, b0], lut[r0, g1, b1]
    c100, c101 = lut[r1, g0, b0], lut[r1, g0, b1]
    c110, c111 = lut[r1, g1, b0], lut[r1, g1, b1]
    c00 = c000 * (1 - db) + c001 * db
    c01 = c010 * (1 - db) + c011 * db
    c10 = c100 * (1 - db) + c101 * db
    c11 = c110 * (1 - db) + c111 * db
    c0 = c00 * (1 - dg) + c01 * dg
    c1 = c10 * (1 - dg) + c11 * dg
    c = c0 * (1 - dr) + c1 * dr
    return c

def apply_3d_lut(img_arr: np.ndarray, lut3d: np.ndarray, strength: float = 1.0) -> np.ndarray:
    if img_arr.ndim != 3 or img_arr.shape[2] != 3:
        raise ValueError("apply_3d_lut expects an HxWx3 image array")
    img_float = img_arr.astype(np.float32) / 255.0
    sampled = _trilinear_sample_lut(img_float, lut3d)
    out = np.clip(sampled * 255.0, 0, 255).astype(np.uint8)
    if strength >= 1.0:
        return out
    else:
        blended = ((1.0 - strength) * img_arr.astype(np.float32) + strength * out.astype(np.float32))
        return np.clip(blended, 0, 255).astype(np.uint8)

def apply_lut(img_arr: np.ndarray, lut: np.ndarray, strength: float = 1.0) -> np.ndarray:
    lut = np.array(lut)
    if lut.ndim == 4 and lut.shape[3] == 3:
        if lut.dtype != np.float32 and lut.max() > 1.0:
            lut = lut.astype(np.float32) / 255.0
        return apply_3d_lut(img_arr, lut, strength=strength)
    elif lut.ndim in (1, 2):
        return apply_1d_lut(img_arr, lut, strength=strength)
    else:
        raise ValueError("Unsupported LUT shape: {}".format(lut.shape))

# --- THE FIX IS HERE: A NEW, MORE ROBUST .CUBE PARSER ---
def load_cube_lut(path: str) -> np.ndarray:
    """
    A robust parser for .cube files that handles common format variations.
    It ignores comments, blank lines, and unknown metadata, focusing only on
    LUT_3D_SIZE and the color data itself.
    """
    size = None
    data = []
    
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            if line.upper().startswith('LUT_3D_SIZE'):
                try:
                    size = int(line.split()[-1])
                except (ValueError, IndexError):
                    raise ValueError(f"Invalid LUT_3D_SIZE format in {path}: {line}")
                continue
            
            # This regex is more forgiving and just looks for three numbers.
            # It will correctly grab data lines and ignore other metadata like TITLE.
            match = re.match(r'^(-?\d+(\.\d+)?(e-?\d+)?)\s+(-?\d+(\.\d+)?(e-?\d+)?)\s+(-?\d+(\.\d+)?(e-?\d+)?)$', line)
            if match:
                try:
                    data.append([float(c) for c in line.split()])
                except ValueError:
                    # Ignore lines that look like data but aren't
                    print(f"Warning: Skipping malformed data line in {path}: {line}")
                    continue

    if size is None:
        raise ValueError(f"LUT_3D_SIZE not found in .cube file: {path}")

    expected_size = size ** 3
    if len(data) != expected_size:
        raise ValueError(f"Cube LUT data length does not match size^3 (got {len(data)}, expected {expected_size})")

    lut = np.array(data, dtype=np.float32).reshape((size, size, size, 3))
    
    # Ensure LUT values are normalized to [0, 1]
    if lut.max() > 1.0 + 1e-6:
        lut = lut / 255.0
        
    return np.clip(lut, 0.0, 1.0)

def load_lut(path: str) -> np.ndarray:
    """
    Load a LUT from:
     - .npy (numpy saved array)
     - .cube (3D LUT)
     - image (PNG/JPG) that is a 1D LUT strip (common 256x1 or 1x256)
    Returns numpy array (1D, 2D, or 4D LUT).
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == '.npy':
        return np.load(path)
    elif ext == '.cube':
        return load_cube_lut(path)
    else:
        try:
            with Image.open(path) as im:
                im = im.convert('RGB')
                arr = np.array(im)
            h, w = arr.shape[:2]
            if (w == 256 and h == 1) or (h == 256 and w == 1):
                lut = arr[0, :, :] if h == 1 else arr[:, 0, :]
                return lut.astype(np.float32)
            flat = arr.reshape(-1, 3).astype(np.float32)
            if flat.shape[0] <= 4096:
                return flat
            raise ValueError("Image LUT not a recognized size")
        except Exception as e:
            raise ValueError(f"Unsupported LUT file or parse error for {path}: {e}")