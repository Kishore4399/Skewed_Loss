# Skewed loss for age prediction (AgeDB faces and brain MRI)

An adaptation of the skewed loss and dynamic lambda strategy from

> H. Wang, M. S. Treder, D. Marshall, D. K. Jones, Y. Li. *A Skewed Loss Function for Correcting Predictive Bias in Brain Age Prediction.* IEEE Transactions on Medical Imaging 42(6):1577–1589, 2023. doi:10.1109/TMI.2022.3231730 ([original code](https://github.com/hanzhiwangchn/MRI_Age_Prediction))

to 2D face images (AgeDB), and back to its original task, brain age from 3D T1 MRI (see [Brain age](#brain-age-3d-mri)). It compares three ways of handling regression to the mean:

| Method | What it does |
|---|---|
| `normal` | Plain L1 (or L2) regression. Usually shows a negative age-delta correlation (ADC). |
| `two_stage` | The `normal` model, then `pred_corrected = (pred − b0) / b1`, with `pred = b1·age + b0` fitted on the validation set. Reported automatically for every `normal` run. |
| `skewed` | Trains with `L(y, ŷ) · exp(sign(y − ŷ) · λ(y))`. λ(y) goes linearly from −λmax at the youngest training age, through 0 at the median, to +λmax at the oldest. λmax is tuned during training so validation ADC goes to zero. |

## Install

```bash
pip install -r requirements.txt
```

## Data

The CSV needs the columns `path`, `age` and `split` (`train` / `val` / `test`). If the target column has another name, pass it with `--label-name` (e.g. `--label-name real_age`). `path` is relative to `--data-folder`. This is the layout of the AgeDB-DIR benchmark file (`agedb.csv` from Yang et al., *Delving into Deep Imbalanced Regression*).

```
data/
├── agedb.csv
└── AgeDB/...   # images, as referenced by the "path" column
```

The skewed loss takes its min, median and max ages from the **training** split.

In AgeDB-DIR the training set is long-tailed, while val and test are balanced across ages. λmax is tuned on the validation set, so it is tuned for a distribution that matches the test set. The median that anchors λ(y), however, comes from the long-tailed training set.

## Run

```bash
# baseline: normal L1 model; summary.json also contains the two-stage result
python main.py --mode normal --seed 0

# skewed L1 loss with dynamic lambda
python main.py --mode skewed --seed 0

# several seeds of each, then compare
for s in 0 1 2 3 4; do
  python main.py --mode normal --seed $s
  python main.py --mode skewed --seed $s
done
python scripts/summarize.py --runs ./runs
```

`summarize.py` searches the runs folder recursively and groups by dataset (`--dataset agedb` to filter). It prints test MAE and ADC (mean ± std over seeds) for normal, two-stage and skewed. It then runs paired Wilcoxon and paired t-tests between two-stage and skewed, matched by seed.

Useful flags:
- `--model resnet18|resnet34|resnet50|small_resnet`. The torchvision models start from ImageNet weights (`--no-pretrained` to disable). `small_resnet` is a 2D version of the paper's network.
- `--correlation spearman` drives Spearman's ρ to zero instead of Pearson's r (paper Section VI-C).
- `--no-dynamic --lambda-init 1.5` uses a fixed λmax.
- `--amp` turns on mixed precision on CUDA.

## Brain age (3D MRI)

`--modality mri` trains on 3D T1 volumes. The data handling follows [UnbiasedBrainAge](https://github.com/Kishore4399/UnbiasedBrainAge), so the same CSV, folds and cache work in both repositories.

### Data

One CSV with one row per scan (UnbiasedBrainAge's `dataset/brainage/UKB&ADNI_fold.csv`):

| Column | Content |
|---|---|
| `image_path` | NIfTI volume; absolute, or relative to `--data-folder` |
| `age` | Chronological age (another column with `--label-name`) |
| `fold` | Cross-validation fold, 0–9 |

For `--val-fold v` and `--test-fold t`:
- training = every fold except `v` and `t`,
- validation = fold `v` (λmax is tuned and the checkpoint is selected on it),
- test = fold `t`. The default is `t = (v + 9) % 10`, as in UnbiasedBrainAge's job scripts.

The skewed loss takes its min, median and max ages from the training folds.

Preprocessing, also as in UnbiasedBrainAge:
1. Load the volume, make it channel-first, resize to `--image-size` (default 96³, trilinear), z-score the non-zero voxels.
2. Training only: random affine, left-right flip, Gaussian smoothing, Gaussian noise and contrast, each with probability `--aug-prob` (default 0.2).

Step 1 is cached on disk as float16 (`src/mri_cache.py`, copied from UnbiasedBrainAge). The cache location is `--cache-dir`, else `BRAINAGE_CACHE_DIR`, else `dataset/brainage/cache`; `--cache-dir none` turns it off. The cache keys are the same as in UnbiasedBrainAge, so pointing `BRAINAGE_CACHE_DIR` at the cache you already built there reuses it. To fill a new cache once on a CPU node:

```bash
export BRAINAGE_CACHE_DIR=/path/with/space/brainage_cache_96
python scripts/build_mri_cache.py --csv 'dataset/brainage/UKB&ADNI_fold.csv' --workers 32
```

Extra requirements: `monai` and `nibabel` (in `requirements.txt`; only needed for MRI).

### Models

- `resnet18_3d` (default for MRI): MONAI 3D ResNet-18 (`shortcut_type A`, 512 features) with a linear head, the network of UnbiasedBrainAge. Trained from scratch.
- `small_resnet3d`: the paper's 3D network (5 residual blocks, 8→128 filters). Needs at least 64³ input.

### Train

```bash
export BRAINAGE_CACHE_DIR=/path/with/space/brainage_cache_96
CSV='/path/to/UnbiasedBrainAge/dataset/brainage/UKB&ADNI_fold.csv'

# one fold: validation fold 0, test fold 9
python main.py --modality mri --mode skewed --model resnet18_3d --dataset brainage \
    --csv "$CSV" --data-folder /path/to/UnbiasedBrainAge --val-fold 0 \
    --epochs 200 --lr 1e-3 --lr-step 50 --batch-size 64 --eval-batch-size 64 \
    --lambda-stage2-start 60 --lambda-stage3-start 120 --lambda-interval 5 --save-start 50 --amp

# baseline on the same fold (also reports two-stage): same command with --mode normal

# all 10 folds, skewed and normal (edit the paths at the top first)
sbatch job_sub_brainage.sh
# or one fold only
sbatch job_sub_brainage.sh 3

# compare over folds
python scripts/summarize.py --runs ./runs --dataset brainage
```

**GPU memory.** MONAI's ResNet-18 keeps its first layer at full resolution, so memory grows quickly with batch size and `--image-size`. For MRI, `--eval-batch-size` defaults to `--batch-size` (the 256 used for images would run out of memory at 96³). Use `--amp` (half the activation memory); if a run still runs out of memory, lower `--batch-size`.

The schedule in these commands is a starting point, not tuned: stage 2 has to start once the model fits reasonably, and stage 3 needs enough (λmax, ADC) pairs. Check `history.csv` for how validation ADC responds to λmax.

Runs go to `runs/brainage/<model>_<mode>_<loss>_<corr>_seed<s>_val<v>_test<t>/` with the same files as above. `summary.json` also records `modality`, `val_fold` and `test_fold`. `summarize.py` pairs two-stage and skewed runs by seed **and test fold**, so 10 folds with one seed give 10 pairs.

## Outputs (`runs/<dataset>/<run-name>/`)

`--dataset` (default `agedb`) names the dataset folder, so runs on different datasets stay apart, e.g. `python main.py --dataset agedb_dir --mode skewed`.

| File | Contents |
|---|---|
| `config.json` | All arguments |
| `history.csv` | Per epoch: LR, λmax in use, stage, training loss, validation MAE and ADC |
| `best.pt` | Selected checkpoint |
| `predictions_<method>.csv` | Final test-set predictions, columns `y_pred`, `y_true`. A normal run writes `predictions_normal.csv` and `predictions_two_stage.csv`; a skewed run writes `predictions_skewed.csv`. |
| `predictions_<method>_train.csv` | The same for the training set: selected checkpoint, no augmentation, same columns. Two-stage uses the correction fitted on validation. |
| `summary.json` | Test MAE/ADC per method, selected epoch, final λmax, and every recorded (λmax, ADC) pair |

## How λmax is tuned (`src/dynamic_lambda.py`)

| Stage | Epochs (default) | Update every `--lambda-interval` epochs |
|---|---|---|
| 1 | 1–29 | None: train with `--lambda-init` |
| 2 | 30–59 (`--lambda-stage2-start`) | If ADC < −θ, λmax ×= η; if ADC > +θ, λmax ÷= η |
| 3 | 60–90 (`--lambda-stage3-start`) | Fit ADC ≈ a·λmax + b on recorded pairs; set λmax = −b/a |

Paper symbols map to flags as follows:
- α → `--lambda-stage2-start`
- β → `--lambda-stage3-start`
- ε → `--lambda-interval`
- η → `--lambda-eta`
- θ → `--lambda-theta`

The schedule is shorter than the paper's 400 epochs because AgeDB is larger and the backbone is pretrained. The learning rate is halved every 30 epochs rather than divided by 10. A slower decay keeps the network able to respond to λmax changes in stage 3.

**Checkpoint selection** follows the authors' code. From `--save-start` on, the epoch with the lowest validation MAE is kept. In skewed mode it must also have |validation ADC| ≤ `--adc-tol`. If no epoch qualifies, the least-biased epoch is used and the log says so.

## Differences from the paper and the original code

- **Smoothed ADC** (`--no-lambda-smooth` to disable). Updates use the median of a 3-epoch moving average of validation ADC, as in the authors' code. The paper uses the single latest value.
- **Limited regression window** (`--lambda-history 0` for all pairs). Stage 3 fits only the last 10 (λmax, ADC) pairs, as in the authors' code. The paper uses all pairs.
- **Fixed η.** The authors' code multiplies by a random draw from N(1.4, 0.05); this uses a fixed η.
- **Stage-3 step limit** (new; `--lambda-max-step 0` to disable). Each stage-3 update can change λmax by at most ×1.5. Late in training the recorded λmax values cluster together, the fitted slope gets noisy, and the zero crossing can jump far away.
- **Vectorized loss.** The skewed loss is vectorized. `tests/test_losses.py` checks that values and gradients match the per-sample loop of the original implementation.
- **Smaller fixes.** Seeds are set for Python, NumPy and PyTorch. Validation and test use large batches with correct averaging. Test-set metrics are computed only once, at the end.

## Tests

```bash
pytest -q tests
```

- `test_losses.py`: matches the reference loop; checks the direction of the skew.
- `test_dynamic_lambda.py`: stage logic, zero-crossing fit, and convergence on a noisy toy model.
- `test_two_stage.py`: the correction removes a linear bias.
- `test_smoke.py`: trains `small_resnet` for a few epochs on a synthetic AgeDB in both modes, then runs `summarize.py`.
- `test_brain_age.py`: fold splitting, 3D models, and an end-to-end MRI run on synthetic NIfTI volumes (skipped without `monai`/`nibabel`).

## What to expect

On face apparent-age prediction (paper Table IX), removing the bias raised MAE: 8.32 (normal) vs 13.64 (skewed) vs 15.42 (two-stage). Expect the corrected methods to trade some MAE for an ADC near zero. The comparison of interest is skewed vs two-stage, not either vs normal.
