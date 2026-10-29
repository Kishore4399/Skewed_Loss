"""One training epoch, inference, and the two-stage correction."""
import contextlib

import numpy as np
import torch


def _autocast(enabled):
    return torch.autocast(device_type='cuda') if enabled else contextlib.nullcontext()


def train_one_epoch(net, loader, loss_fn, optimizer, device, scaler=None):
    net.train()
    total, n = 0.0, 0
    use_amp = scaler is not None
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with _autocast(use_amp):
            preds = net(images)
        # loss in float32: exp() of the skew weight is sensitive to half precision
        loss = loss_fn(preds.float(), labels)
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        # weight by the real batch size so the last, smaller batch is counted correctly
        total += loss.item() * len(labels)
        n += len(labels)
    return total / max(n, 1)


@torch.no_grad()
def predict(net, loader, device, amp=False):
    net.eval()
    preds, labels = [], []
    for images, y in loader:
        with _autocast(amp):
            out = net(images.to(device, non_blocking=True))
        preds.append(out.float().cpu().numpy().ravel())
        labels.append(np.asarray(y).ravel())
    return np.concatenate(preds), np.concatenate(labels)
