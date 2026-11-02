"""
Training script for EMCAF-Net.

The dataset folders are prepared beforehand (see README):
    <dataset_path>/train_aug/fold<N>/{images,GT}   offline-augmented training set
    <dataset_path>/train/fold<N>/{images,GT}       the same images, not augmented
    <dataset_path>/val/fold<N>/{images,GT}         validation set, never augmented
    <dataset_path>/test/fold<N>/...                test set, never read by this script

Any dataset in this layout works; --fold selects the fold and --dataset_path the root.

Training configuration (Section 4.3 of the paper):
    AdamW, lr 1e-4, weight decay 1e-4 (beta1 0.9, beta2 0.999)
    loss = BCE (on logits) + soft Dice
    batch size 4 with gradient accumulation over 2 steps (effective batch 8)
    gradient clipping by global norm 1.0
    ReduceLROnPlateau on the validation loss (factor 0.5, patience 2)
    at most 100 epochs; early stopping after 5 epochs without improvement,
      but never before --min_epochs (15)
    the checkpoint with the lowest validation loss is kept

Two guards: the script refuses to run if 'test' is passed as the training or the
validation split, and it refuses to overwrite an existing checkpoint.

Main configuration:
    python train.py --fold 1 --model_name nucleinet
    python train.py --fold 3 --dataset_path ./dataset/cryonuseg --model_name cryo

Training without data augmentation (original images, online augmentation off):
    python train.py --fold 1 --train_split train --online_aug 0 --model_name noaug
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

from data import get_loader

# The reported results were produced on a single CUDA GPU. CPU is supported
# only so that the scripts can be smoke-tested without one.
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def parse_args():
    p = argparse.ArgumentParser(description='Train EMCAF-Net')
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
    p.add_argument('--model_name',    type=str,   default='nucleinet')
    p.add_argument('--arch',          type=str,   default='nucleinet',
                   help='model file in model/ to use, e.g. nucleinet_no_ca (no file swapping)')
    return p.parse_args()


def set_seed(seed, deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
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
    for step, (images, targets) in enumerate(loader, start=1):
        images, targets = images.to(DEVICE), targets.to(DEVICE)
        loss = criterion(model(images), targets, args.dice_weight)
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
        images, targets = images.to(DEVICE), targets.to(DEVICE)
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

    train_loader = get_loader(tr_img, tr_gt, batchsize=args.batchsize,
                              trainsize=args.trainsize, augment=bool(args.online_aug),
                              shuffle=True)
    val_loader = get_loader(va_img, va_gt, batchsize=args.batchsize,
                            trainsize=args.trainsize, augment=False, shuffle=False)

    Net = importlib.import_module(f'model.{args.arch}').nucleinet
    model = Net().to(DEVICE)
    print(f'Architecture: model/{args.arch}.py | device: {DEVICE}')
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