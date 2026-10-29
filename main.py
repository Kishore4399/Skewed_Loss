"""Skewed loss vs. two-stage bias correction for age prediction on AgeDB.

Adapted from the method in:
    H. Wang, M. S. Treder, D. Marshall, D. K. Jones, Y. Li,
    "A Skewed Loss Function for Correcting Predictive Bias in Brain Age Prediction,"
    IEEE Transactions on Medical Imaging 42(6), 2023. doi:10.1109/TMI.2022.3231730

Examples:
    # plain L1 model; also reports the two-stage-corrected result from the same model
    python main.py --mode normal --seed 0
    # skewed L1 loss with the dynamic lambda strategy
    python main.py --mode skewed --seed 0
"""
import argparse
import json
import logging
import os
import random
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.datasets import build_datasets, build_eval_dataset
from src.dynamic_lambda import DynamicLambda
from src.engine import predict, train_one_epoch
from src.losses import build_train_loss
from src.metrics import adc, mae
from src.models import build_model
from src.two_stage import apply_two_stage, fit_two_stage

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
log = logging.getLogger('skewed')


def build_parser():
    p = argparse.ArgumentParser(description='Skewed loss for age prediction on AgeDB')
    # data
    p.add_argument('--data-folder', default='./data', help='root that the CSV "path" column is relative to')
    p.add_argument('--csv', default='./data/agedb.csv',
                   help='CSV with columns path, <--label-name>, split (train/val/test)')
    p.add_argument('--label-name', default='age',
                   help='CSV column holding the regression target (age)')
    p.add_argument('--img-size', type=int, default=224)
    p.add_argument('--workers', type=int, default=8)
    # method
    p.add_argument('--mode', choices=['normal', 'skewed'], default='skewed',
                   help='normal: symmetric loss (+ two-stage correction reported); skewed: skewed loss')
    p.add_argument('--base-loss', choices=['L1', 'L2'], default='L1')
    p.add_argument('--correlation', choices=['pearson', 'spearman'], default='pearson',
                   help='ADC metric driven to zero by the dynamic lambda strategy')
    # model and optimization
    p.add_argument('--model', choices=['resnet18', 'resnet34', 'resnet50', 'small_resnet'], default='resnet50')
    p.add_argument('--no-pretrained', dest='pretrained', action='store_false',
                   help='train torchvision backbones from scratch instead of ImageNet weights')
    p.add_argument('--epochs', type=int, default=90)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--eval-batch-size', type=int, default=256)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--lr-step', type=int, default=30, help='multiply LR by --lr-gamma every this many epochs')
    p.add_argument('--lr-gamma', type=float, default=0.5,
                   help='mild decay on purpose: the model must still react to lambda changes late in training')
    p.add_argument('--amp', action='store_true', help='mixed precision on CUDA')
    p.add_argument('--seed', type=int, default=0)
    # dynamic lambda strategy (Algorithm 1)
    p.add_argument('--lambda-init', type=float, default=1.0)
    p.add_argument('--no-dynamic', dest='dynamic', action='store_false',
                   help='keep lambda_max fixed at --lambda-init')
    p.add_argument('--lambda-stage2-start', type=int, default=30, help='alpha in the paper')
    p.add_argument('--lambda-stage3-start', type=int, default=60, help='beta in the paper')
    p.add_argument('--lambda-interval', type=int, default=3, help='epsilon in the paper')
    p.add_argument('--lambda-eta', type=float, default=1.5, help='stage-2 multiplier (eta)')
    p.add_argument('--lambda-theta', type=float, default=0.1, help='stage-2 ADC tolerance (theta)')
    p.add_argument('--no-lambda-smooth', dest='lambda_smooth', action='store_false',
                   help='update on the latest validation ADC only (paper) instead of a smoothed value (authors\' code)')
    p.add_argument('--lambda-history', type=int, default=10,
                   help='stage-3 regression uses this many most recent pairs; 0 = all (paper)')
    p.add_argument('--lambda-cap', type=float, default=20.0)
    p.add_argument('--lambda-max-step', type=float, default=1.5,
                   help='stage-3 updates change lambda_max by at most this factor; 0 = unlimited (paper/authors)')
    # checkpoint selection
    p.add_argument('--save-start', type=int, default=20, help='first epoch eligible as the final model')
    p.add_argument('--adc-tol', type=float, default=0.05,
                   help='skewed mode: only epochs with |val ADC| <= this can be selected (as in the authors\' code)')
    # output
    p.add_argument('--dataset', default='agedb',
                   help='dataset name; runs are saved under <out-dir>/<dataset>/<run-name>')
    p.add_argument('--out-dir', default='./runs')
    p.add_argument('--run-name', default=None)
    return p


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def make_loaders(args, ds_train, ds_val, ds_test, device):
    common = {'num_workers': args.workers, 'pin_memory': device.type == 'cuda',
              'persistent_workers': args.workers > 0}
    g = torch.Generator()
    g.manual_seed(args.seed)
    train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, drop_last=False,
                       generator=g, **common)
    val = DataLoader(ds_val, batch_size=args.eval_batch_size, shuffle=False, **common)
    test = DataLoader(ds_test, batch_size=args.eval_batch_size, shuffle=False, **common)
    return train, val, test


def main():
    args = build_parser().parse_args()
    if args.run_name is None:
        args.run_name = f'{args.model}_{args.mode}_{args.base_loss}_{args.correlation}_seed{args.seed}'
    run_dir = os.path.join(args.out_dir, args.dataset, args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    set_seed(args.seed)
    device = pick_device()
    log.info(f'device={device} run_dir={run_dir}')

    ds_train, ds_val, ds_test, age_stats = build_datasets(args)
    log.info(f'train/val/test = {len(ds_train)}/{len(ds_val)}/{len(ds_test)}; training ages {age_stats}')
    train_loader, val_loader, test_loader = make_loaders(args, ds_train, ds_val, ds_test, device)

    net = build_model(args).to(device)
    loss_fn = build_train_loss(args, age_stats)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)
    use_amp = args.amp and device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    lam = None
    if args.mode == 'skewed' and args.dynamic:
        lam = DynamicLambda(init=args.lambda_init, stage2_start=args.lambda_stage2_start,
                            stage3_start=args.lambda_stage3_start, interval=args.lambda_interval,
                            eta=args.lambda_eta, theta=args.lambda_theta, smooth=args.lambda_smooth,
                            history=args.lambda_history, cap=args.lambda_cap,
                            max_step=args.lambda_max_step)

    best_path = os.path.join(run_dir, 'best.pt')
    fallback_path = os.path.join(run_dir, 'fallback.pt')
    best_mae, best_epoch = float('inf'), None
    fallback_adc, fallback_epoch = float('inf'), None
    history, val_adcs = [], []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        lambda_used = loss_fn.lambda_max if args.mode == 'skewed' else None
        train_loss = train_one_epoch(net, train_loader, loss_fn, optimizer, device, scaler)
        val_preds, val_labels = predict(net, val_loader, device, use_amp)
        val_mae, val_adc = mae(val_preds, val_labels), adc(val_preds, val_labels, args.correlation)
        val_adcs.append(val_adc)
        lr = optimizer.param_groups[0]['lr']
        scheduler.step()

        new_lambda = lam.step(epoch, val_adcs) if lam is not None else None
        if new_lambda is not None:
            loss_fn.set_lambda_max(new_lambda)

        # checkpoint selection on the validation set
        if epoch >= args.save_start:
            eligible = args.mode != 'skewed' or abs(val_adc) <= args.adc_tol
            if eligible and val_mae < best_mae:
                best_mae, best_epoch = val_mae, epoch
                torch.save(net.state_dict(), best_path)
            # skewed mode: remember the least-biased epoch in case none meets --adc-tol
            if args.mode == 'skewed' and abs(val_adc) < fallback_adc:
                fallback_adc, fallback_epoch = abs(val_adc), epoch
                torch.save(net.state_dict(), fallback_path)

        row = {'epoch': epoch, 'lr': lr, 'lambda_max': lambda_used,
               'stage': lam.stage(epoch) if lam is not None else None,
               'train_loss': train_loss, 'val_mae': val_mae, 'val_adc': val_adc,
               'seconds': time.time() - t0}
        history.append(row)
        pd.DataFrame(history).to_csv(os.path.join(run_dir, 'history.csv'), index=False)
        msg = f'epoch {epoch:3d} train_loss {train_loss:.3f} val_MAE {val_mae:.3f} val_ADC {val_adc:+.3f}'
        if lambda_used is not None:
            msg += f' lambda_max {lambda_used:.3f}'
        if new_lambda is not None:
            msg += f' -> {new_lambda:.3f}'
        log.info(msg)

    # ---- final evaluation on the selected checkpoint ----
    if best_epoch is not None:
        selected, selected_epoch, selection = best_path, best_epoch, 'lowest val MAE'
        if args.mode == 'skewed':
            selection += f' with |val ADC| <= {args.adc_tol}'
    elif fallback_epoch is not None:
        selected, selected_epoch, selection = fallback_path, fallback_epoch, 'lowest |val ADC| (fallback)'
        log.warning(f'no epoch reached |val ADC| <= {args.adc_tol}; using the least-biased epoch {fallback_epoch}')
    else:
        selected, selected_epoch, selection = None, args.epochs, 'last epoch'
        log.warning('--save-start is beyond the last epoch; evaluating the final weights')
    if selected is not None:
        net.load_state_dict(torch.load(selected, map_location=device))

    val_preds, val_labels = predict(net, val_loader, device, use_amp)
    test_preds, test_labels = predict(net, test_loader, device, use_amp)
    # training set again, without augmentation, in a fixed order
    train_eval_loader = DataLoader(build_eval_dataset(args, 'train'), batch_size=args.eval_batch_size,
                                   shuffle=False, num_workers=args.workers, pin_memory=device.type == 'cuda')
    train_preds, train_labels = predict(net, train_eval_loader, device, use_amp)
    results, final_preds, train_final_preds = {}, {}, {}
    if args.mode == 'normal':
        results['normal'] = {'mae': mae(test_preds, test_labels), 'adc': adc(test_preds, test_labels, args.correlation)}
        b1, b0 = fit_two_stage(val_preds, val_labels)
        corrected = apply_two_stage(test_preds, b1, b0)
        results['two_stage'] = {'mae': mae(corrected, test_labels), 'adc': adc(corrected, test_labels, args.correlation),
                                'slope': b1, 'intercept': b0}
        final_preds['normal'], final_preds['two_stage'] = test_preds, corrected
        # same correction (fitted on validation) applied to the training predictions
        train_final_preds['normal'] = train_preds
        train_final_preds['two_stage'] = apply_two_stage(train_preds, b1, b0)
    else:
        results['skewed'] = {'mae': mae(test_preds, test_labels), 'adc': adc(test_preds, test_labels, args.correlation)}
        final_preds['skewed'] = test_preds
        train_final_preds['skewed'] = train_preds
    # one file per method: predictions_<method>.csv (test) and predictions_<method>_train.csv
    for name, preds in final_preds.items():
        pd.DataFrame({'y_pred': preds, 'y_true': test_labels}).to_csv(
            os.path.join(run_dir, f'predictions_{name}.csv'), index=False)
    for name, preds in train_final_preds.items():
        pd.DataFrame({'y_pred': preds, 'y_true': train_labels}).to_csv(
            os.path.join(run_dir, f'predictions_{name}_train.csv'), index=False)
    train_results = {name: {'mae': mae(p, train_labels), 'adc': adc(p, train_labels, args.correlation)}
                     for name, p in train_final_preds.items()}

    summary = {
        'run_name': args.run_name, 'dataset': args.dataset, 'label_name': args.label_name, 'mode': args.mode, 'model': args.model, 'seed': args.seed,
        'correlation': args.correlation, 'selected_epoch': selected_epoch, 'selection': selection,
        'final_lambda_max': loss_fn.lambda_max if args.mode == 'skewed' else None,
        'lambda_pairs': lam.pairs if lam is not None else None,
        'val': {'mae': mae(val_preds, val_labels), 'adc': adc(val_preds, val_labels, args.correlation)},
        'train': train_results,
        'test': results,
    }
    with open(os.path.join(run_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    for name, r in results.items():
        log.info(f'TEST {name:9s} MAE {r["mae"]:.3f}  ADC {r["adc"]:+.3f}  (epoch {selected_epoch}, {selection})')


if __name__ == '__main__':
    main()
