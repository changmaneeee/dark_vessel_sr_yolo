"""
=============================================================================
train_sr_only.py - SR 모델 단독 학습 스크립트
=============================================================================

[목적]
RFDN 또는 MambaSR을 단독으로 학습
Detection 없이 순수 SR 성능만 검증

[사용법]
python scripts/train_sr_only.py \
    --sr_model rfdn \
    --hr_dir data/hr/images/train \
    --lr_dir data/lr/images/train \
    --val_hr_dir data/hr/images/val \
    --val_lr_dir data/lr/images/val \
    --epochs 100 \
    --batch_size 16 \
    --lr 1e-4

[출력]
- checkpoints/sr_only/best.pth
- logs/sr_only/events.out.tfevents...
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm import tqdm
import cv2
import numpy as np
from datetime import datetime

# 프로젝트 import
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.sr_models.rfdn import RFDN
from src.models.sr_models.mamba_sr import MambaSR
from src.losses.sr_loss import SRLoss, CharbonnierLoss


# =============================================================================
# Dataset
# =============================================================================

class SRDataset(Dataset):
    """
    SR 학습용 데이터셋
    LR-HR 이미지 쌍 로드
    """
    def __init__(self, lr_dir, hr_dir, patch_size=192, augment=True):
        self.lr_dir = Path(lr_dir)
        self.hr_dir = Path(hr_dir)
        self.patch_size = patch_size
        self.augment = augment
        
        # 이미지 파일 목록
        self.lr_files = sorted(list(self.lr_dir.glob("*.jpg")) + 
                               list(self.lr_dir.glob("*.png")))
        
        print(f"[SRDataset] Found {len(self.lr_files)} images")
    
    def __len__(self):
        return len(self.lr_files)
    
    def __getitem__(self, idx):
        # LR 이미지 로드
        lr_path = self.lr_files[idx]
        hr_path = self.hr_dir / lr_path.name
        
        lr_img = cv2.imread(str(lr_path))
        hr_img = cv2.imread(str(hr_path))
        
        if lr_img is None or hr_img is None:
            # 에러 시 다른 이미지 반환
            return self.__getitem__((idx + 1) % len(self))
        
        # BGR -> RGB
        lr_img = cv2.cvtColor(lr_img, cv2.COLOR_BGR2RGB)
        hr_img = cv2.cvtColor(hr_img, cv2.COLOR_BGR2RGB)
        
        # 패치 추출 (랜덤 크롭)
        lr_h, lr_w = lr_img.shape[:2]
        hr_h, hr_w = hr_img.shape[:2]
        scale = hr_h // lr_h  # 보통 4
        
        # LR 패치 크기
        lr_patch = self.patch_size // scale
        
        if lr_h >= lr_patch and lr_w >= lr_patch:
            # 랜덤 위치
            y = np.random.randint(0, lr_h - lr_patch + 1)
            x = np.random.randint(0, lr_w - lr_patch + 1)
            
            lr_img = lr_img[y:y+lr_patch, x:x+lr_patch]
            hr_img = hr_img[y*scale:(y+lr_patch)*scale, x*scale:(x+lr_patch)*scale]
        
        # Augmentation
        if self.augment:
            # 랜덤 수평 플립
            if np.random.rand() > 0.5:
                lr_img = np.fliplr(lr_img).copy()
                hr_img = np.fliplr(hr_img).copy()
            
            # 랜덤 수직 플립
            if np.random.rand() > 0.5:
                lr_img = np.flipud(lr_img).copy()
                hr_img = np.flipud(hr_img).copy()
            
            # 랜덤 90도 회전
            k = np.random.randint(0, 4)
            lr_img = np.rot90(lr_img, k).copy()
            hr_img = np.rot90(hr_img, k).copy()
        
        # numpy -> tensor, [0,255] -> [0,1]
        lr_tensor = torch.from_numpy(lr_img).permute(2, 0, 1).float() / 255.0
        hr_tensor = torch.from_numpy(hr_img).permute(2, 0, 1).float() / 255.0
        
        return lr_tensor, hr_tensor


# =============================================================================
# Metrics
# =============================================================================

def calculate_psnr(pred, target, max_val=1.0):
    """PSNR 계산"""
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 20 * torch.log10(max_val / torch.sqrt(mse))
    return psnr.item()


def calculate_ssim(pred, target, window_size=11):
    """간단한 SSIM 계산 (채널별 평균)"""
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    mu1 = torch.nn.functional.avg_pool2d(pred, window_size, stride=1, padding=window_size//2)
    mu2 = torch.nn.functional.avg_pool2d(target, window_size, stride=1, padding=window_size//2)
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = torch.nn.functional.avg_pool2d(pred ** 2, window_size, stride=1, padding=window_size//2) - mu1_sq
    sigma2_sq = torch.nn.functional.avg_pool2d(target ** 2, window_size, stride=1, padding=window_size//2) - mu2_sq
    sigma12 = torch.nn.functional.avg_pool2d(pred * target, window_size, stride=1, padding=window_size//2) - mu1_mu2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return ssim_map.mean().item()


# =============================================================================
# Training Loop
# =============================================================================

def train_one_epoch(model, dataloader, optimizer, loss_fn, device, epoch):
    model.train()
    total_loss = 0
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for lr_img, hr_gt in pbar:
        lr_img = lr_img.to(device)
        hr_gt = hr_gt.to(device)
        
        # Forward
        sr_img = model(lr_img)
        
        # Loss
        loss_dict = loss_fn(sr_img, hr_gt)
        loss = loss_dict['total']
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})
    
    return total_loss / len(dataloader)


@torch.no_grad()
def validate(model, dataloader, device):
    model.eval()
    total_psnr = 0
    total_ssim = 0
    count = 0
    
    for lr_img, hr_gt in tqdm(dataloader, desc="Validation"):
        lr_img = lr_img.to(device)
        hr_gt = hr_gt.to(device)
        
        sr_img = model(lr_img)
        
        # Clamp to [0, 1]
        sr_img = torch.clamp(sr_img, 0, 1)
        
        # Metrics
        for i in range(sr_img.size(0)):
            total_psnr += calculate_psnr(sr_img[i], hr_gt[i])
            total_ssim += calculate_ssim(sr_img[i:i+1], hr_gt[i:i+1])
            count += 1
    
    return total_psnr / count, total_ssim / count


# =============================================================================
# Main
# =============================================================================

def main(args):
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # 출력 디렉토리
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{args.sr_model}_{timestamp}"
    ckpt_dir = Path(args.output_dir) / "checkpoints" / exp_name
    log_dir = Path(args.output_dir) / "logs" / exp_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # TensorBoard
    writer = SummaryWriter(log_dir)
    
    # =========================================================================
    # Model
    # =========================================================================
    print(f"\n[Model] Creating {args.sr_model}...")
    
    if args.sr_model == 'rfdn':
        model = RFDN(
            in_channels=3,
            out_channels=3,
            nf=args.nf,
            num_modules=args.num_modules,
            upscale=args.scale
        )
    elif args.sr_model == 'mamba':
        model = MambaSR(
            in_channels=3,
            out_channels=3,
            dim=args.nf,
            n_blocks=args.num_modules,
            upscale=args.scale
        )
    else:
        raise ValueError(f"Unknown SR model: {args.sr_model}")
    
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")
    
    # =========================================================================
    # Dataset
    # =========================================================================
    print(f"\n[Dataset] Loading...")
    
    train_dataset = SRDataset(
        lr_dir=args.lr_dir,
        hr_dir=args.hr_dir,
        patch_size=args.patch_size,
        augment=True
    )
    
    val_dataset = SRDataset(
        lr_dir=args.val_lr_dir,
        hr_dir=args.val_hr_dir,
        patch_size=args.patch_size * 2,  # Validation은 더 큰 패치
        augment=False
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    print(f"  Train: {len(train_dataset)} images")
    print(f"  Val: {len(val_dataset)} images")
    
    # =========================================================================
    # Loss & Optimizer
    # =========================================================================
    loss_fn = SRLoss(
        l1_weight=1.0,
        perceptual_weight=args.perceptual_weight,
        ssim_weight=args.ssim_weight,
        use_charbonnier=True
    )
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.01
    )
    
    # =========================================================================
    # Training Loop
    # =========================================================================
    best_psnr = 0
    
    print(f"\n[Training] Starting for {args.epochs} epochs...")
    
    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device, epoch)
        
        # Validate
        val_psnr, val_ssim = validate(model, val_loader, device)
        
        # Scheduler step
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # Logging
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('PSNR/val', val_psnr, epoch)
        writer.add_scalar('SSIM/val', val_ssim, epoch)
        writer.add_scalar('LR', current_lr, epoch)
        
        print(f"Epoch {epoch}/{args.epochs} | Loss: {train_loss:.4f} | "
              f"PSNR: {val_psnr:.2f} | SSIM: {val_ssim:.4f} | LR: {current_lr:.6f}")
        
        # Save checkpoint
        is_best = val_psnr > best_psnr
        if is_best:
            best_psnr = val_psnr
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_psnr': best_psnr,
            'args': vars(args)
        }
        
        # 마지막 체크포인트
        torch.save(checkpoint, ckpt_dir / 'last.pth')
        
        # Best 체크포인트
        if is_best:
            torch.save(checkpoint, ckpt_dir / 'best.pth')
            print(f"  ✓ New best PSNR: {best_psnr:.2f}")
        
        # 주기적 저장
        if epoch % args.save_freq == 0:
            torch.save(checkpoint, ckpt_dir / f'epoch_{epoch}.pth')
    
    writer.close()
    print(f"\n✓ Training completed! Best PSNR: {best_psnr:.2f}")
    print(f"  Checkpoints: {ckpt_dir}")
    print(f"  Logs: {log_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SR Model")
    
    # Model
    parser.add_argument('--sr_model', type=str, default='rfdn', choices=['rfdn', 'mamba'])
    parser.add_argument('--nf', type=int, default=50, help='Feature channels')
    parser.add_argument('--num_modules', type=int, default=4, help='Number of blocks')
    parser.add_argument('--scale', type=int, default=4, help='Upscale factor')
    
    # Data
    parser.add_argument('--hr_dir', type=str, required=True, help='HR images directory')
    parser.add_argument('--lr_dir', type=str, required=True, help='LR images directory')
    parser.add_argument('--val_hr_dir', type=str, required=True, help='Validation HR directory')
    parser.add_argument('--val_lr_dir', type=str, required=True, help='Validation LR directory')
    parser.add_argument('--patch_size', type=int, default=192, help='Training patch size')
    
    # Training
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, default=4)
    
    # Loss weights
    parser.add_argument('--perceptual_weight', type=float, default=0.0)
    parser.add_argument('--ssim_weight', type=float, default=0.0)
    
    # Output
    parser.add_argument('--output_dir', type=str, default='runs/sr_only')
    parser.add_argument('--save_freq', type=int, default=10, help='Save checkpoint every N epochs')
    
    args = parser.parse_args()
    main(args)