"""End-to-end: tiny synthetic AgeDB, a few epochs of each mode, through main.py."""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

pytest.importorskip('torch')
pytest.importorskip('torchvision')
from PIL import Image  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope='module')
def fake_agedb(tmp_path_factory):
    root = tmp_path_factory.mktemp('agedb')
    rng = np.random.default_rng(0)
    rows = ['path,true_age,split']  # non-default label column, read via --label-name
    for split, n in [('train', 96), ('val', 32), ('test', 32)]:
        os.makedirs(root / split, exist_ok=True)
        for i in range(n):
            age = int(rng.integers(1, 100))
            # brightness encodes age, so the model has something to learn
            img = np.clip(rng.normal(age * 2.5, 20, (64, 64, 3)), 0, 255).astype(np.uint8)
            rel = f'{split}/{i}.png'
            Image.fromarray(img).save(root / rel)
            rows.append(f'{rel},{age},{split}')
    (root / 'agedb.csv').write_text('\n'.join(rows))
    return root


def run(mode, data, out, extra=()):
    cmd = [sys.executable, os.path.join(ROOT, 'main.py'), '--mode', mode,
           '--data-folder', str(data), '--csv', str(data / 'agedb.csv'), '--out-dir', str(out),
           '--model', 'small_resnet', '--img-size', '64', '--workers', '0',
           '--batch-size', '16', '--epochs', '8', '--lr', '1e-2', '--save-start', '1',
           '--lambda-stage2-start', '3', '--lambda-stage3-start', '6', '--lambda-interval', '1',
           '--adc-tol', '1.0', '--dataset', 'fake', '--label-name', 'true_age', *extra]
    subprocess.run(cmd, check=True, cwd=ROOT)


def test_normal_and_skewed_runs(fake_agedb, tmp_path):
    run('normal', fake_agedb, tmp_path)
    run('skewed', fake_agedb, tmp_path)
    runs = [tmp_path / 'fake' / d for d in os.listdir(tmp_path / 'fake')]
    summaries = [json.load(open(r / 'summary.json')) for r in runs]
    methods = {m for s in summaries for m in s['test']}
    assert methods == {'normal', 'two_stage', 'skewed'}
    skewed = next(s for s in summaries if s['mode'] == 'skewed')
    assert len(skewed['lambda_pairs']) == 6  # updates at epochs 3..8
    import pandas as pd
    files = sorted(f.name for r in runs for f in r.glob('predictions_*.csv'))
    assert files == ['predictions_normal.csv', 'predictions_normal_train.csv',
                     'predictions_skewed.csv', 'predictions_skewed_train.csv',
                     'predictions_two_stage.csv', 'predictions_two_stage_train.csv']
    train_df = pd.read_csv(runs[0] / next(f.name for f in runs[0].glob('predictions_*_train.csv')))
    assert list(train_df.columns) == ['y_pred', 'y_true'] and len(train_df) == 96
    test_file = next(f for f in runs[0].glob('predictions_*.csv') if not f.name.endswith('_train.csv'))
    df = pd.read_csv(test_file)
    assert list(df.columns) == ['y_pred', 'y_true'] and len(df) == 32
    subprocess.run([sys.executable, os.path.join(ROOT, 'scripts', 'summarize.py'), '--runs', str(tmp_path)],
                   check=True, cwd=ROOT)
