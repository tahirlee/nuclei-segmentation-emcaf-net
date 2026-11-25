"""
Training script for the dual-decoder U-Net baseline (model/dual_decoder_unet.py).

Same data, augmentation, optimiser, schedule, early stopping and checkpoint rule as train.py.
The only differences:
  - the model has two outputs in training mode: segmentation logit and distance map
  - loss = BCE + soft Dice (segmentation)  +  dist_weight * MSE (distance map)
  - the distance target is computed on the fly from the binary mask: Euclidean distance transform,
    divided by its maximum inside every connected component (values 0..1)
  - validation and the checkpoint rule use the segmentation output only (same as for EMCAF-Net)

Usage (fold 1):
    python train_dual.py --fold 1 --model_name dual_decoder_unet
Then evaluate with the unchanged test_instance.py:
    python test_instance.py --fold 1 --arch dual_decoder_unet --model_path trained_models\\dual_decoder_unet\\dual_decoder_unet_fold1_lr3.pth
"""
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
import os
import random
import argparse
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import importlib

from scipy import ndimage as ndi
from torch.utils.data import DataLoader

from data import get_loader, SalObjDataset, _seed_worker

DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def distance_target(mask):
    """distance-to-background of every nucleus region, normalised to 0..1 per connected component"""
    mask = mask > 0.5
    out = np.zeros(mask.shape, np.float32)
    if not mask.any():
        return out
    lab, n = ndi.label(mask)
    edt = ndi.distance_transform_edt(mask)
    mx = np.zeros(n + 1, np.float32)
    mx[1:] = ndi.maximum(edt, lab, index=np.arange(1, n + 1))
    out[mask] = (edt[mask] / np.maximum(mx[lab[mask]], 1e-6)).astype(np.float32)
    return out


def collate_with_dist(batch):
    images = torch.stack([b[0] for b in batch])
    gts = torch.stack([b[1] for b in batch])
    dist = torch.stack([torch.from_numpy(distance_target(g[0].numpy())).unsqueeze(0) for g in gts])
    return images, gts, dist


def parse_args():
    p = argparse.ArgumentParser(description='Train the dual-decoder U-Net baseline')
    p.add_argument('--epoch',         type=int,   default=100)
    p.add_argument('--min_epochs',    type=int,   default=15,
                   help='early stopping cannot trigger before this epoch')
    p.add_argument('--lr',            type=float, default=1e-4)
    p.add_argument('--weight_decay',  type=float, default=1e-4)
    p.add_argument('--batchsize',     type=int,   default=4)
    p.add_argument('--accum_steps',   type=int,   default=2,
                   help='effective batch = batchsize x accum_steps')
    p.add_argument('--trainsize',     type=int,   default=512)
    p.add_argument('--clip_norm',     type=float, default=1.0,
                   help='max global gradient norm; 0 disables')
    p.add_argument('--plateau_factor',   type=float, default=0.5)
    p.add_argument('--plateau_patience', type=int,   default=2)
    p.add_argument('--patience',      type=int,   default=5)
    p.add_argument('--dice_weight',   type=float, default=1.0)
    p.add_argument('--seed',          type=int,   default=12393)
    p.add_argument('--deterministic', type=int,   default=1)
    p.add_argument('--fold',          type=int,   default=1)
    p.add_argument('--dataset_path',  type=str,   default='./dataset')
    p.add_argument('--train_split',   type=str,   default='train_aug',
                   help="train_aug = offline-augmented set; train = original images")
    p.add_argument('--online_aug',    type=int,   default=1,
                   help='1 = random flip/crop/rotation/colour jitter on the fly, 0 = off')
    p.add_argument('--val_split',     type=str,   default='val')
    p.add_argument('--model_name',    type=str,   default='dual_decoder_unet')
    p.add_argument('--dist_weight',   type=float, default=1.0, help='weight of the distance-map MSE loss')
    p.add_argument('--num_workers',   type=int,   default=4)
    p.add_argument('--arch',          type=str,   default='dual_decoder_unet',
                   help='model file in model/ to use, e.g. nucleinet_no_ca (no file swapping)')
    return p.parse_args()


def set_seed(seed, deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = not deterministic
    cudnn.deterministic = deterministic


# ── Loss: BCE + soft Dice ─────────────────────────────────────────────────────
bce = nn.BCEWithLogitsLoss()


def soft_dice_loss(logits, target, smooth=1.0):
    prob = torch.sigmoid(logits)
    inter = (prob * target).sum(dim=(2, 3))
    union = prob.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    return 1.0 - ((2.0 * inter + smooth) / (union + smooth)).mean()


def criterion(logits, target, dice_weight=1.0):
    return bce(logits, target) + dice_weight * soft_dice_loss(logits, target)


# ── One epoch ─────────────────────────────────────────────────────────────────
def train_epoch(loader, model, optimizer, epoch, args):
    model.train()
    total_loss, n = 0.0, len(loader)
    optimizer.zero_grad()
    for step, (images, targets, dist_t) in enumerate(loader, start=1):
        images, targets, dist_t = images.to(DEV), targets.to(DEV), dist_t.to(DEV)
        seg, dist_p = model(images)
        loss = criterion(seg, targets, args.dice_weight) + args.dist_weight * F.mse_loss(dist_p, dist_t)
        (loss / args.accum_steps).backward()

        if step % args.accum_steps == 0 or step == n:
            if args.clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item()
        if step % 50 == 0 or step == n:
            print('{} Epoch [{:03d}/{:03d}] Step [{:04d}/{:04d}] LR: {:.2e} Loss: {:.4f}'.format(
                datetime.now().strftime('%H:%M:%S'), epoch, args.epoch, step, n,
                optimizer.param_groups[0]['lr'], loss.item()))
    return total_loss / n


@torch.no_grad()
def validate(loader, model, args):
    model.eval()
    total_loss, inter, denom = 0.0, 0.0, 0.0
    for images, targets in loader:
        images, targets = images.to(DEV), targets.to(DEV)
        logits = model(images)
        total_loss += criterion(logits, targets, args.dice_weight).item()
        pred = (torch.sigmoid(logits) > 0.5).float()
        inter += (pred * targets).sum().item()
        denom += pred.sum().item() + targets.sum().item()
    return total_loss / len(loader), 2.0 * inter / (denom + 1e-8)


def main():
    args = parse_args()
    if args.val_split == 'test' or args.train_split == 'test':
        raise ValueError('The test split must never be used for training or validation.')
    set_seed(args.seed, deterministic=bool(args.deterministic))

    fold_tag = f'fold{args.fold}'
    tr_img = os.path.join(args.dataset_path, args.train_split, fold_tag, 'images') + '/'
    tr_gt  = os.path.join(args.dataset_path, args.train_split, fold_tag, 'GT') + '/'
    va_img = os.path.join(args.dataset_path, args.val_split,   fold_tag, 'images') + '/'
    va_gt  = os.path.join(args.dataset_path, args.val_split,   fold_tag, 'GT') + '/'

    train_loader = DataLoader(SalObjDataset(tr_img, tr_gt, args.trainsize, augment=bool(args.online_aug)),
                              batch_size=args.batchsize, shuffle=True, num_workers=args.num_workers,
                              pin_memory=True, worker_init_fn=_seed_worker,
                              persistent_workers=args.num_workers > 0, collate_fn=collate_with_dist)
    val_loader = get_loader(va_img, va_gt, batchsize=args.batchsize,
                            trainsize=args.trainsize, augment=False, shuffle=False)

    Net = importlib.import_module(f'model.{args.arch}').nucleinet
    model = Net().to(DEV)
    print(f'Architecture: model/{args.arch}.py')
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=args.plateau_factor,
        patience=args.plateau_patience)

    save_dir = os.path.join('trained_models', args.model_name)
    os.makedirs(save_dir, exist_ok=True)
    ckpt = os.path.join(save_dir, f'{args.model_name}_{fold_tag}_lr3.pth')
    if os.path.exists(ckpt):
        raise FileExistsError(f'{ckpt} exists. Use a new --model_name for a new run.')

    # save the exact settings next to the checkpoint
    with open(os.path.join(save_dir, f'{fold_tag}_settings.txt'), 'w') as f:
        for k, v in sorted(vars(args).items()):
            f.write(f'{k}: {v}\n')

    print(f'Training {fold_tag} | train: {tr_img} ({len(train_loader.dataset)} images, '
          f'online_aug={args.online_aug}) | val: {va_img} ({len(val_loader.dataset)} images)')
    print(f'Checkpoint: {ckpt}')

    best_loss, best_epoch, wait = float('inf'), 0, 0
    for epoch in range(1, args.epoch + 1):
        tr_loss = train_epoch(train_loader, model, optimizer, epoch, args)
        va_loss, va_dice = validate(val_loader, model, args)
        scheduler.step(va_loss)
        print(f'Epoch [{epoch}/{args.epoch}]  Train: {tr_loss:.4f}  '
              f'Val loss: {va_loss:.4f}  Val Dice: {va_dice:.4f}  '
              f'LR: {optimizer.param_groups[0]["lr"]:.2e}')

        if va_loss < best_loss:
            best_loss, best_epoch, wait = va_loss, epoch, 0
            torch.save(model.state_dict(), ckpt)
            print('  -> Best model saved.')
        else:
            wait += 1
            print(f'  No improvement ({wait}/{args.patience})')

        if epoch >= args.min_epochs and wait >= args.patience:
            print(f'Early stopping at epoch {epoch}. Best epoch: {best_epoch}.')
            break

    print(f'Done. Best val loss {best_loss:.4f} at epoch {best_epoch}.')
    with open(os.path.join(save_dir, f'{fold_tag}_settings.txt'), 'a') as f:
        f.write(f'best_epoch: {best_epoch}\nbest_val_loss: {best_loss:.6f}\n'
                f'stopped_at: {epoch}\n')


if __name__ == '__main__':
    main()