"""Brain-age (MRI) mode: fold splitting, 3D models, and an end-to-end run on synthetic NIfTI volumes."""
import json
import os
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip('torch')

from src.brain_datasets import default_test_fold, split_folds  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def fold_df(n_folds=10, per_fold=3):
    rows = [{'image_path': f'{f}_{i}.nii.gz', 'age': 20 + f + i, 'fold': f}
            for f in range(n_folds) for i in range(per_fold)]
    return pd.DataFrame(rows)


def test_default_test_fold_matches_unbiasedbrainage_jobs():
    assert [default_test_fold(v) for v in range(10)] == [9, 0, 1, 2, 3, 4, 5, 6, 7, 8]


def test_split_folds():
    s = split_folds(fold_df(), val_fold=0, test_fold=9)
    assert set(s['val'].fold) == {0} and set(s['test'].fold) == {9}
    assert set(s['train'].fold) == set(range(1, 9))
    assert len(s['train']) + len(s['val']) + len(s['test']) == 30


def test_split_folds_rejects_bad_folds():
    with pytest.raises(ValueError):
        split_folds(fold_df(), 2, 2)
    with pytest.raises(ValueError):
        split_folds(fold_df(), 0, 42)


def test_small_resnet3d_shape_and_modality_check():
    from src.models import build_model
    net = build_model(SimpleNamespace(model='small_resnet3d', modality='mri', pretrained=False))
    assert net(torch.zeros(2, 1, 64, 64, 64)).shape == (2, 1)
    with pytest.raises(ValueError):
        build_model(SimpleNamespace(model='resnet18', modality='mri', pretrained=False))
    with pytest.raises(ValueError):
        build_model(SimpleNamespace(model='resnet18_3d', modality='image', pretrained=False))


def test_monai_resnet18_3d_shape():
    pytest.importorskip('monai')
    from src.models import build_model
    net = build_model(SimpleNamespace(model='resnet18_3d', modality='mri', pretrained=False))
    assert net(torch.zeros(2, 1, 32, 32, 32)).shape == (2, 1)


@pytest.fixture(scope='module')
def fake_brains(tmp_path_factory):
    pytest.importorskip('monai')
    nib = pytest.importorskip('nibabel')
    root = tmp_path_factory.mktemp('brains')
    rng = np.random.default_rng(0)
    rows = []
    for fold in range(4):
        for i in range(8):
            age = int(rng.integers(20, 90))
            # a sphere whose brightness encodes age, so there is something to learn
            g = np.linspace(-1, 1, 24)
            mask = (g[:, None, None] ** 2 + g[None, :, None] ** 2 + g[None, None, :] ** 2) < 0.6
            vol = (mask * (age / 10.0 + rng.normal(0, 0.3, mask.shape))).astype(np.float32)
            rel = f'scans/f{fold}_{i}.nii.gz'
            os.makedirs(root / 'scans', exist_ok=True)
            nib.save(nib.Nifti1Image(vol, np.eye(4)), str(root / rel))
            rows.append({'image_path': rel, 'age': age, 'fold': fold})
    pd.DataFrame(rows).to_csv(root / 'folds.csv', index=False)
    return root


def run(mode, data, out):
    cmd = [sys.executable, os.path.join(ROOT, 'main.py'), '--modality', 'mri', '--mode', mode,
           '--model', 'small_resnet3d', '--image-size', '64', '64', '64',
           '--data-folder', str(data), '--csv', str(data / 'folds.csv'), '--out-dir', str(out),
           '--cache-dir', str(data / 'cache'), '--val-fold', '1', '--test-fold', '0',
           '--workers', '0', '--batch-size', '8', '--eval-batch-size', '8', '--epochs', '4',
           '--lr', '1e-2', '--save-start', '1', '--lambda-stage2-start', '2', '--lambda-stage3-start', '3',
           '--lambda-interval', '1', '--adc-tol', '1.0', '--dataset', 'fake_brain']
    subprocess.run(cmd, check=True, cwd=ROOT)


def test_mri_normal_and_skewed_runs(fake_brains, tmp_path):
    run('normal', fake_brains, tmp_path)
    run('skewed', fake_brains, tmp_path)
    runs = [tmp_path / 'fake_brain' / d for d in os.listdir(tmp_path / 'fake_brain')]
    summaries = [json.load(open(r / 'summary.json')) for r in runs]
    assert {m for s in summaries for m in s['test']} == {'normal', 'two_stage', 'skewed'}
    assert all(s['modality'] == 'mri' and s['val_fold'] == 1 and s['test_fold'] == 0 for s in summaries)
    test_df = pd.read_csv(next(r / 'predictions_skewed.csv' for r in runs if (r / 'predictions_skewed.csv').exists()))
    train_df = pd.read_csv(next(r / 'predictions_normal_train.csv' for r in runs
                                if (r / 'predictions_normal_train.csv').exists()))
    assert len(test_df) == 8 and len(train_df) == 16          # test = fold 0, train = folds 2 and 3
    assert any((fake_brains / 'cache').rglob('*.npy'))       # volumes were cached
    subprocess.run([sys.executable, os.path.join(ROOT, 'scripts', 'summarize.py'), '--runs', str(tmp_path)],
                   check=True, cwd=ROOT)
