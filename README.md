# Skewed loss for age prediction on AgeDB

An adaptation of the skewed loss and dynamic lambda strategy from

> H. Wang, M. S. Treder, D. Marshall, D. K. Jones, Y. Li. *A Skewed Loss Function for Correcting Predictive Bias in Brain Age Prediction.* IEEE Transactions on Medical Imaging 42(6):1577–1589, 2023. doi:10.1109/TMI.2022.3231730 ([original code](https://github.com/hanzhiwangchn/MRI_Age_Prediction))

from 3D brain MRI to 2D face images (AgeDB). It compares three ways of handling regression to the mean:

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

The CSV needs the columns `path`, `age` and `split` (`train` / `val` / `test`). `path` is relative to `--data-folder`. This is the layout of the AgeDB-DIR benchmark file (`agedb.csv` from Yang et al., *Delving into Deep Imbalanced Regression*).

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

`summarize.py` prints test MAE and ADC (mean ± std over seeds) for normal, two-stage and skewed. It then runs paired Wilcoxon and paired t-tests between two-stage and skewed, matched by seed.

Useful flags:
- `--model resnet18|resnet34|resnet50|small_resnet`. The torchvision models start from ImageNet weights (`--no-pretrained` to disable). `small_resnet` is a 2D version of the paper's network.
- `--correlation spearman` drives Spearman's ρ to zero instead of Pearson's r (paper Section VI-C).
- `--no-dynamic --lambda-init 1.5` uses a fixed λmax.
- `--amp` turns on mixed precision on CUDA.

## Outputs (`runs/<run-name>/`)

| File | Contents |
|---|---|
| `config.json` | All arguments |
| `history.csv` | Per epoch: LR, λmax in use, stage, training loss, validation MAE and ADC |
| `best.pt` | Selected checkpoint |
| `test_predictions.csv` | `age`, `pred`, plus `pred_two_stage` for normal runs |
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

## What to expect

On face apparent-age prediction (paper Table IX), removing the bias raised MAE: 8.32 (normal) vs 13.64 (skewed) vs 15.42 (two-stage). Expect the corrected methods to trade some MAE for an ADC near zero. The comparison of interest is skewed vs two-stage, not either vs normal.
