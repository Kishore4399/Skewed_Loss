"""AgeDB dataset and train/val/test construction."""
import os

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils import data
from torchvision import transforms


class AgeDB(data.Dataset):
    """Face images with a CSV of (path, <label column>, split), e.g. the AgeDB-DIR benchmark file."""

    def __init__(self, data_folder, csv_path, transform=None, split='train', label_name='age'):
        df = pd.read_csv(csv_path)
        missing = [c for c in ('path', label_name, 'split') if c not in df.columns]
        if missing:
            raise KeyError(f'{csv_path} has no column(s) {missing}; columns are {list(df.columns)}. '
                           f'Set --label-name to the column that holds the target.')
        self.label_name = label_name
        self.df = df[df['split'] == split].reset_index(drop=True)
        self.split = split
        self.data_folder = data_folder
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        # shape (1,) so a batch is (B, 1), matching the model output
        label = np.asarray([row[self.label_name]]).astype(np.float32)
        img = Image.open(os.path.join(self.data_folder, row['path'])).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    @property
    def ages(self):
        return self.df[self.label_name].to_numpy(dtype=np.float32)


def build_transforms(img_size, train):
    # ImageNet statistics, since the default backbone is ImageNet-pretrained
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if train:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomCrop(img_size, padding=16),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        normalize,
    ])


def build_eval_dataset(args, split):
    """A split with the evaluation transform (no augmentation), e.g. to predict on the training set."""
    return AgeDB(args.data_folder, args.csv, build_transforms(args.img_size, False), split,
                 label_name=args.label_name)


def build_datasets(args):
    """Returns train/val/test datasets plus the age statistics the skewed loss needs.

    lim (min, max) and the median come from the TRAINING split only, because they
    define where lambda(y) is -lambda_max, 0 and +lambda_max.
    """
    ds_train = AgeDB(args.data_folder, args.csv, build_transforms(args.img_size, True), 'train',
                     label_name=args.label_name)
    ds_val = AgeDB(args.data_folder, args.csv, build_transforms(args.img_size, False), 'val',
                   label_name=args.label_name)
    ds_test = AgeDB(args.data_folder, args.csv, build_transforms(args.img_size, False), 'test',
                    label_name=args.label_name)

    train_ages = ds_train.ages
    age_stats = {
        'min': float(train_ages.min()),
        'max': float(train_ages.max()),
        'median': float(np.median(train_ages)),
    }
    return ds_train, ds_val, ds_test, age_stats
