"""Brain-age datasets: 3D T1 MRI with a fold CSV, as in UnbiasedBrainAge.

The CSV has one row per scan with the columns
    image_path   path to the NIfTI volume (absolute, or relative to --data-folder)
    age          the target (another column can be chosen with --label-name)
    fold         cross-validation fold, e.g. 0-9

For --val-fold v and --test-fold t (default t = (v + 9) % 10, as in UnbiasedBrainAge's job scripts):
    train = every fold except v and t,  val = fold v,  test = fold t.

Preprocessing and augmentation match UnbiasedBrainAge (lib/data/dataset.py):
    load -> channel first -> resize to --image-size (trilinear) -> z-score over non-zero voxels,
cached on disk (src/mri_cache.py), then for training only random affine, left-right flip,
Gaussian smoothing, Gaussian noise and contrast, each with probability --aug-prob.

MONAI (and nibabel to read NIfTI) are only imported when this module is used.
"""
import os

import numpy as np
import pandas as pd
import torch
from torch.utils import data

from src.mri_cache import CachedPreprocessd

REQUIRED_COLUMNS = ('image_path', 'fold')


def default_test_fold(val_fold, n_folds=10):
    return (val_fold + n_folds - 1) % n_folds


def build_mri_transform(image_size, train, aug_prob=0.2, cache_dir=None):
    from monai.transforms import (Compose, RandAdjustContrastd, RandAffined, RandFlipd,
                                  RandGaussianNoised, RandGaussianSmoothd, ToTensord)
    steps = [CachedPreprocessd(keys=['image'], image_size=image_size, cache_dir=cache_dir)]
    if train and aug_prob > 0:
        steps += [
            RandAffined(keys=['image'], prob=aug_prob, translate_range=(2, 2, 2),
                        rotate_range=(0.087, 0.087, 0.087), scale_range=(0.05, 0.05, 0.05),
                        padding_mode='border'),
            RandFlipd(keys=['image'], spatial_axis=[0], prob=aug_prob),   # left-right only
            RandGaussianSmoothd(keys=['image'], prob=aug_prob),
            RandGaussianNoised(keys=['image'], prob=aug_prob, std=0.05),
            RandAdjustContrastd(keys=['image'], prob=aug_prob, gamma=(0.7, 1.3)),
        ]
    steps.append(ToTensord(keys=['image']))
    return Compose(steps)


class BrainAge(data.Dataset):
    """One split of a fold CSV. Items are (volume [1, D, H, W] float32, label [1] float32)."""

    def __init__(self, df, data_folder='', transform=None, label_name='age'):
        self.df = df.reset_index(drop=True)
        self.data_folder = data_folder or ''
        self.transform = transform
        self.label_name = label_name

    def __len__(self):
        return len(self.df)

    def path(self, index):
        p = str(self.df.iloc[index]['image_path'])
        return p if os.path.isabs(p) else os.path.join(self.data_folder, p)

    def __getitem__(self, index):
        label = np.asarray([self.df.iloc[index][self.label_name]]).astype(np.float32)
        image = self.transform({'image': self.path(index)})['image']
        if not isinstance(image, torch.Tensor):
            image = torch.as_tensor(np.asarray(image))
        if hasattr(image, 'as_tensor'):           # MONAI MetaTensor -> plain tensor
            image = image.as_tensor()
        return image.float(), label

    @property
    def ages(self):
        return self.df[self.label_name].to_numpy(dtype=np.float32)


def read_fold_csv(csv_path, label_name):
    df = pd.read_csv(csv_path)
    missing = [c for c in (*REQUIRED_COLUMNS, label_name) if c not in df.columns]
    if missing:
        raise KeyError(f'{csv_path} has no column(s) {missing}; columns are {list(df.columns)}. '
                       f'MRI mode needs image_path, fold and the target column (--label-name).')
    return df


def split_folds(df, val_fold, test_fold):
    if val_fold == test_fold:
        raise ValueError(f'--val-fold and --test-fold must differ (both {val_fold})')
    folds = set(df['fold'].unique().tolist())
    for name, f in (('--val-fold', val_fold), ('--test-fold', test_fold)):
        if f not in folds:
            raise ValueError(f'{name} {f} is not in the CSV; folds are {sorted(folds)}')
    return {
        'train': df[~df['fold'].isin([val_fold, test_fold])],
        'val': df[df['fold'] == val_fold],
        'test': df[df['fold'] == test_fold],
    }


def _splits(args):
    return split_folds(read_fold_csv(args.csv, args.label_name), args.val_fold, args.test_fold)


def build_mri_eval_dataset(args, split):
    """A split with the evaluation transform (no augmentation), e.g. to predict on the training set."""
    tf = build_mri_transform(args.image_size, train=False, cache_dir=args.cache_dir)
    return BrainAge(_splits(args)[split], args.data_folder, tf, args.label_name)


def build_mri_datasets(args):
    """Train/val/test datasets plus the training-split age statistics the skewed loss needs."""
    splits = _splits(args)
    ds_train = BrainAge(splits['train'], args.data_folder,
                        build_mri_transform(args.image_size, True, args.aug_prob, args.cache_dir), args.label_name)
    eval_tf = build_mri_transform(args.image_size, False, cache_dir=args.cache_dir)
    ds_val = BrainAge(splits['val'], args.data_folder, eval_tf, args.label_name)
    ds_test = BrainAge(splits['test'], args.data_folder, eval_tf, args.label_name)

    ages = ds_train.ages
    age_stats = {'min': float(ages.min()), 'max': float(ages.max()), 'median': float(np.median(ages))}
    return ds_train, ds_val, ds_test, age_stats
