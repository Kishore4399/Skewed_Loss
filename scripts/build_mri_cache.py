"""Fill the preprocessed-MRI cache for every scan in a fold CSV (see src/mri_cache.py).

Optional: without it, the first epoch of each run fills the cache as it goes. Running it once up
front (on a CPU node, many workers) makes every later run and fold read from cache. The cache format
is the same as UnbiasedBrainAge's build_cache.py, so a cache built there is reused here.

    python scripts/build_mri_cache.py --csv 'dataset/brainage/UKB&ADNI_fold.csv' --workers 32
    python scripts/build_mri_cache.py --csv ... --cache-dir /scratch/$USER/brainage_cache_96

Use the same --image-size and cache location as training (--image-size / --cache-dir of main.py,
or BRAINAGE_CACHE_DIR), otherwise training will not find these entries.
"""
import argparse
import os
import sys
from pathlib import Path

import pandas as pd
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.mri_cache import CachedPreprocessd, resolve_cache_dir  # noqa: E402


class _Paths(Dataset):
    def __init__(self, paths, cacher):
        self.paths, self.cacher = paths, cacher

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        path = self.paths[i]
        try:
            self.cacher.load_array(path)
            return path, ''
        except Exception as e:  # report and keep going; every failure is listed at the end
            return path, f'{type(e).__name__}: {e}'


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--csv', default='dataset/brainage/UKB&ADNI_fold.csv', help='CSV with an image_path column')
    p.add_argument('--data-folder', default='', help='root for relative image_path values (as in main.py)')
    p.add_argument('--image-size', type=int, nargs=3, default=[96, 96, 96])
    p.add_argument('--cache-dir', default=None, help='default: BRAINAGE_CACHE_DIR, else dataset/brainage/cache')
    p.add_argument('--workers', type=int, default=16)
    args = p.parse_args()

    cache_dir = resolve_cache_dir(args.cache_dir)
    if cache_dir is None:
        sys.exit('Caching is turned off (cache dir is empty/none); nothing to build.')
    cacher = CachedPreprocessd(keys=['image'], image_size=args.image_size, cache_dir=cache_dir)

    paths = pd.read_csv(args.csv)['image_path'].dropna().drop_duplicates().astype(str).tolist()
    paths = [q if os.path.isabs(q) else os.path.join(args.data_folder, q) for q in paths]
    print(f'{len(paths)} scans -> {os.path.abspath(cache_dir)} (image size {tuple(args.image_size)})')

    loader = DataLoader(_Paths(paths, cacher), batch_size=1, num_workers=args.workers,
                        collate_fn=lambda batch: batch[0])
    failed = []
    for i, (path, error) in enumerate(loader, 1):
        if error:
            failed.append((path, error))
        if i % 500 == 0:
            print(f'  {i}/{len(paths)}', flush=True)

    size = sum(f.stat().st_size for f in Path(cache_dir).rglob('*.npy'))
    print(f'done: {len(paths) - len(failed)} cached, {len(failed)} failed; cache folder now {size / 1e9:.1f} GB')
    for path, error in failed[:20]:
        print(f'  FAILED {path}: {error}')
    if len(failed) > 20:
        print(f'  ... and {len(failed) - 20} more')


if __name__ == '__main__':
    main()
