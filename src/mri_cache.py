"""Disk cache for the deterministic part of the MRI preprocessing.

Copied from UnbiasedBrainAge (lib/data/cache.py) unchanged in behavior: the cache keys, CACHE_VERSION,
storage dtype and BRAINAGE_CACHE_DIR are the same, so both repositories can share one cache folder.

Every MRI transform in src/brain_datasets.py starts with the same fixed steps:

    LoadImaged -> EnsureChannelFirstd -> Resized(image_size, trilinear) -> NormalizeIntensityd(nonzero, channel_wise)

Loading and resampling a NIfTI volume is the slow part of each training step. CachedPreprocessd runs
those steps once per image and saves the result as a small .npy file; later reads (next epochs,
other folds, other scripts) load that file instead. Random augmentation runs after it, so it still
changes every epoch.

Where the cache lives:
    - cache_dir argument / `dataset.cache_dir` in a config, else
    - the BRAINAGE_CACHE_DIR environment variable, else
    - dataset/brainage/cache (relative to the working directory).
Set it to "" or "none" to turn caching off (the same preprocessing then runs every time).

Cache entries are keyed on the image's absolute path, size and modification time, the image size and
CACHE_VERSION, so a changed file or a different image_size never reuses a stale entry. Files are
written to a temporary name and renamed, so parallel DataLoader workers and parallel jobs can share
one cache folder safely. Volumes are stored as float16 by default (half the disk space; the
z-scored intensities lose < 1e-3 relative precision).

Fill the cache once before training (optional, otherwise the first epoch fills it):
    python scripts/build_mri_cache.py --csv 'dataset/brainage/UKB&ADNI_fold.csv'
"""
import hashlib
import os

import numpy as np

CACHE_VERSION = "v1"  # bump if the preprocessing steps change
DEFAULT_CACHE_DIR = "dataset/brainage/cache"
_OFF = ("", "none", "off", "false", "0")


def resolve_cache_dir(cache_dir=None):
    """cache_dir argument > BRAINAGE_CACHE_DIR > DEFAULT_CACHE_DIR; returns None when caching is off."""
    if cache_dir is None:
        cache_dir = os.environ.get("BRAINAGE_CACHE_DIR", DEFAULT_CACHE_DIR)
    if cache_dir is False or str(cache_dir).strip().lower() in _OFF:
        return None
    return str(cache_dir)


class CachedPreprocessd:
    """Dictionary transform: replaces each key's file path with its preprocessed volume [C, D, H, W].

    Equivalent to Compose([LoadImaged, EnsureChannelFirstd, Resized, NormalizeIntensityd]) on `keys`,
    with the output cached on disk. Returns float32 torch tensors without MONAI metadata (nothing
    downstream uses it), the same whether the volume came from the cache or was just computed.
    """

    def __init__(self, keys=("image",), image_size=(96, 96, 96), cache_dir=None, dtype="float16"):
        self.keys = [keys] if isinstance(keys, str) else list(keys)
        self.image_size = tuple(int(s) for s in image_size)
        self.cache_dir = resolve_cache_dir(cache_dir)
        self.dtype = np.dtype(dtype)
        self._preprocess = None  # built lazily, so the object pickles cheaply into DataLoader workers

    # ------------------------------------------------------------------ preprocessing
    def _build_preprocess(self):
        from monai.transforms import Compose, EnsureChannelFirstd, LoadImaged, NormalizeIntensityd, Resized
        return Compose([
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            Resized(keys=["image"], spatial_size=self.image_size, mode="trilinear"),
            NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        ])

    def compute(self, path):
        """Run the preprocessing on one file; returns a numpy array in the storage dtype."""
        if self._preprocess is None:
            self._preprocess = self._build_preprocess()
        image = self._preprocess({"image": path})["image"]
        if hasattr(image, "as_tensor"):  # MONAI MetaTensor -> plain tensor
            image = image.as_tensor()
        if hasattr(image, "detach"):
            image = image.detach().cpu().numpy()
        return np.asarray(image).astype(self.dtype)

    # ------------------------------------------------------------------ cache
    def cache_path(self, path):
        st = os.stat(path)
        key = f"{os.path.abspath(path)}|{st.st_size}|{int(st.st_mtime)}|{self.image_size}|{self.dtype}|{CACHE_VERSION}"
        digest = hashlib.sha1(key.encode()).hexdigest()
        return os.path.join(self.cache_dir, digest[:2], digest + ".npy")

    def load_array(self, path):
        """Preprocessed volume as a numpy array (storage dtype), from the cache when possible."""
        if self.cache_dir is None:
            return self.compute(path)
        target = self.cache_path(path)
        try:
            return np.load(target)
        except (FileNotFoundError, ValueError, EOFError, OSError):
            pass  # not cached yet (or a truncated file from a killed job): recompute and overwrite
        array = self.compute(path)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        tmp = f"{target}.{os.getpid()}.tmp"
        with open(tmp, "wb") as f:
            np.save(f, array)
        os.replace(tmp, target)  # atomic: readers see either no file or the complete one
        return array

    def __call__(self, data):
        import torch
        out = dict(data)
        for key in self.keys:
            out[key] = torch.from_numpy(self.load_array(out[key]).astype(np.float32))
        return out
