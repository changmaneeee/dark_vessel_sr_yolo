"""
=============================================================================
train_arch0.py - Arch0 Sequential Pipeline 학습
=============================================================================

[사용법]
python scripts/train_arch0.py \
    --hr_dir data/hr/images/train \
    --lr_dir data/lr/images/train \
    --label_dir data/hr/labels/train \
    --pretrained_sr checkpoints/sr_only/best.pth \
    --epochs 100
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
from types import SimpleNamespace
import torch.nn.functional as F
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.pipelines.arch0_sequential import Arch0Sequential


# =============================================================================
# Dataset (SR + Detection)
# =============================================================================

class SRDetectionDataset(Dataset):
    """
    SR-Detection 파이프라인 학습용 데이터셋
    LR 이미지, HR GT, YOLO 형식 라벨 반환
    """
    def __init__(self, lr_dir, hr_dir, label_dir, img_size=192, augment=True):
        self.lr_dir = Path(lr_dir)
        self.hr_dir = Path(hr_dir)
        self.label_dir = Path(label_dir)
        self.img_size = img_size
        self.augment = augment
        
        # 이미지 파일 목록 (라벨이 있는 것만)
        self.samples = []
        
        for lr_path in sorted(self.lr_dir.glob("*.jpg")):
            hr_path = self.hr_dir / lr_path.name
            label_path = self.label_dir / (lr_path.stem + ".txt")
            
            if hr_path.exists():
                self.samples.append({
                    'lr': lr_path,
                    'hr': hr_path,
                    'label': label_path if label_path.exists() else None
                })
        
        print(f"[SRDetectionDataset] Found {len(self.samples)} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def load_labels(self, label_path, batch_idx):
        """YOLO 형식 라벨 로드: [batch_idx, class, x, y, w, h]"""
        labels = []
        
        if label_path and label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls = int(parts[0])
                        x, y, w, h = map(float, parts[1:5])
                        labels.append([batch_idx, cls, x, y, w, h])
        
        return labels
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 이미지 로드
        lr_img = cv2.imread(str(sample['lr']))
        hr_img = cv2.imread(str(sample['hr']))
        
        if lr_img is None or hr_img is None:
            return self.__getitem__((idx + 1) % len(self))
        
        # BGR -> RGB
        lr_img = cv2.cvtColor(lr_img, cv2.COLOR_BGR2RGB)
        hr_img = cv2.cvtColor(hr_img, cv2.COLOR_BGR2RGB)
        
        # Resize to fixed size (간단한 버전)
        lr_img = cv2.resize(lr_img, (self.img_size, self.img_size))
        hr_img = cv2.resize(hr_img, (self.img_size * 4, self.img_size * 4))
        
        # 라벨 로드
        labels = self.load_labels(sample['label'], 0)  # batch_idx는 collate에서 수정
        
        # Tensor 변환
        lr_tensor = torch.from_numpy(lr_img).permute(2, 0, 1).float() / 255.0
        hr_tensor = torch.from_numpy(hr_img).permute(2, 0, 1).float() / 255.0
        
        return lr_tensor, hr_tensor, labels


def collate_fn(batch):
    """Custom collate function for variable-length labels"""
    lr_imgs = torch.stack([item[0] for item in batch])
    hr_imgs = torch.stack([item[1] for item in batch])
    
    # 라벨 합치기 (batch_idx 업데이트)
    all_labels = []
    for batch_idx, item in enumerate(batch):
        for label in item[2]:
            label[0] = batch_idx  # batch_idx 수정
            all_labels.append(label)
    
    if all_labels:
        targets = torch.tensor(all_labels, dtype=torch.float32)
    else:
        targets = torch.zeros((0, 6), dtype=torch.float32)
    
    return lr_imgs, hr_imgs, targets


# =============================================================================
# Training
# =============================================================================

def train_one_epoch(model, dataloader, optimizer, device, epoch, sr_weight, det_weight):
    model.train()
    total_loss = 0
    total_sr_loss = 0
    total_det_loss = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for lr_imgs, hr_gts, targets in pbar:
        lr_imgs = lr_imgs.to(device)
        hr_gts = hr_gts.to(device)
        targets = targets.to(device)
        
        # Forward
        sr_imgs, detections = model(lr_imgs)
        
        # Loss (SR만 계산, Detection은 더미)
        
        sr_loss = F.l1_loss(sr_imgs, hr_gts)
        
        # TODO: Detection loss 추가 (실제 데이터 필요)
        det_loss = torch.tensor(0.0, device=device)
        
        loss = sr_weight * sr_loss + det_weight * det_loss
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_sr_loss += sr_loss.item()
        
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'sr': f"{sr_loss.item():.4f}"
        })
    
    n = len(dataloader)
    return total_loss/n, total_sr_loss/n, total_det_loss/n


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # 출력 디렉토리
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"arch0_{timestamp}"
    ckpt_dir = Path(args.output_dir) / "checkpoints" / exp_name
    log_dir = Path(args.output_dir) / "logs" / exp_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    writer = SummaryWriter(log_dir)
    
    # =========================================================================
    # Config & Model
    # =========================================================================
    config = SimpleNamespace(
        model=SimpleNamespace(
            rfdn=SimpleNamespace(nf=args.nf, num_modules=args.num_modules),
            yolo=SimpleNamespace(weights_path=args.yolo_weights, num_classes=args.num_classes)
        ),
        data=SimpleNamespace(upscale_factor=4),
        training=SimpleNamespace(
            sr_weight=args.sr_weight,
            det_weight=args.det_weight,
            freeze_detector=True
        ),
        device=str(device)
    )
    
    print("\n[Model] Creating Arch0Sequential...")
    model = Arch0Sequential(config)
    model = model.to(device)
    
    # Pretrained SR 로드
    if args.pretrained_sr and Path(args.pretrained_sr).exists():
        print(f"[Model] Loading pretrained SR: {args.pretrained_sr}")
        checkpoint = torch.load(args.pretrained_sr, map_location=device)
        model.sr_model.load_state_dict(checkpoint['model_state_dict'])
    
    # =========================================================================
    # Dataset
    # =========================================================================
    print("\n[Dataset] Loading...")
    
    train_dataset = SRDetectionDataset(
        lr_dir=args.lr_dir,
        hr_dir=args.hr_dir,
        label_dir=args.label_dir,
        img_size=args.img_size
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    # =========================================================================
    # Optimizer
    # =========================================================================
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=1e-4
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # =========================================================================
    # Training
    # =========================================================================
    print(f"\n[Training] Starting for {args.epochs} epochs...")
    
    best_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        train_loss, sr_loss, det_loss = train_one_epoch(
            model, train_loader, optimizer, device, epoch,
            args.sr_weight, args.det_weight
        )
        
        scheduler.step()
        
        # Logging
        writer.add_scalar('Loss/total', train_loss, epoch)
        writer.add_scalar('Loss/sr', sr_loss, epoch)
        writer.add_scalar('Loss/det', det_loss, epoch)
        
        print(f"Epoch {epoch} | Total: {train_loss:.4f} | SR: {sr_loss:.4f} | Det: {det_loss:.4f}")
        
        # Save
        is_best = train_loss < best_loss
        if is_best:
            best_loss = train_loss
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_loss,
            'args': vars(args)
        }
        
        torch.save(checkpoint, ckpt_dir / 'last.pth')
        if is_best:
            torch.save(checkpoint, ckpt_dir / 'best.pth')
    
    writer.close()
    print(f"\n✓ Training completed!")
    print(f"  Checkpoints: {ckpt_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Data
    parser.add_argument('--hr_dir', type=str, required=True)
    parser.add_argument('--lr_dir', type=str, required=True)
    parser.add_argument('--label_dir', type=str, required=True)
    parser.add_argument('--img_size', type=int, default=192)
    
    # Model
    parser.add_argument('--nf', type=int, default=50)
    parser.add_argument('--num_modules', type=int, default=4)
    parser.add_argument('--yolo_weights', type=str, default='yolov8n.pt')
    parser.add_argument('--num_classes', type=int, default=1)
    parser.add_argument('--pretrained_sr', type=str, default=None)
    
    # Training
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--sr_weight', type=float, default=1.0)
    parser.add_argument('--det_weight', type=float, default=0.0)
    parser.add_argument('--num_workers', type=int, default=4)
    
    # Output
    parser.add_argument('--output_dir', type=str, default='runs/arch0')
    
    args = parser.parse_args()
    main(args)