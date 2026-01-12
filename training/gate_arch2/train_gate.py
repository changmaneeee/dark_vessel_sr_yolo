#!/usr/bin/env python
"""
=============================================================================
train_gate.py - Lightweight Gate Network Training (Arch2용)
=============================================================================
LR 이미지를 보고 "SR이 필요한지" 예측하는 경량 Gate 학습

[모델]
- LightweightGate: ~50K params, Binary classifier
- Input: LR 이미지 (160x160)
- Output: SR 필요 확률 (0~1)

[라벨]
- 0: Bypass OK (SR 불필요)
- 1: SR needed (SR 필요)

사용법:
    python train_gate.py \
        --lr_root /path/to/lr_dataset \
        --labels_dir ./gate_labels_gt \
        --output ./checkpoints/gate \
        --epochs 30 \
        --batch_size 32
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import cv2

# Tensorboard
try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TB = True
except ImportError:
    HAS_TB = False


def set_seed(seed: int = 42):
    """재현성을 위한 시드 설정"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# Model: LightweightGate
# =============================================================================

class LightweightGate(nn.Module):
    """
    경량 Gate Network (~50K params)
    
    LR 이미지 → SR 필요 확률 (0~1)
    """
    
    def __init__(self, in_channels: int = 3, base_channels: int = 32):
        super().__init__()
        
        # Feature extractor (3 conv blocks with stride 2)
        self.features = nn.Sequential(
            # Block 1: 160 → 80
            nn.Conv2d(in_channels, base_channels, 3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            
            # Block 2: 80 → 40
            nn.Conv2d(base_channels, base_channels * 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            
            # Block 3: 40 → 20
            nn.Conv2d(base_channels * 2, base_channels * 4, 3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            
            # Block 4: 20 → 10
            nn.Conv2d(base_channels * 4, base_channels * 4, 3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
        )
        
        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(base_channels * 4, base_channels * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(base_channels * 2, 1),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: LR 이미지 [B, 3, H, W]
        
        Returns:
            SR 필요 확률 [B, 1] (sigmoid 적용됨)
        """
        feat = self.features(x)
        feat = self.gap(feat)
        feat = feat.view(feat.size(0), -1)
        out = self.classifier(feat)
        return torch.sigmoid(out)
    
    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Binary prediction"""
        prob = self.forward(x)
        return (prob >= threshold).float()


# =============================================================================
# Dataset: GateDataset
# =============================================================================

class GateDataset(Dataset):
    """
    Gate 학습용 데이터셋
    
    LR 이미지 + Gate 라벨 (0 or 1)
    """
    
    def __init__(
        self,
        lr_root: str,
        labels_path: str,
        split: str = 'train',
        img_size: int = 160,
        augment: bool = True
    ):
        self.lr_root = Path(lr_root)
        self.split = split
        self.img_size = img_size
        self.augment = augment and (split == 'train')
        
        # 이미지 디렉토리
        self.img_dir = self.lr_root / 'images' / split
        
        # 라벨 로드
        with open(labels_path, 'r') as f:
            self.labels = json.load(f)
        
        # 이미지 파일 목록 (라벨이 있는 것만)
        self.image_files = []
        for img_name, label in self.labels.items():
            # 확장자 찾기
            for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                img_path = self.img_dir / f"{img_name}{ext}"
                if img_path.exists():
                    self.image_files.append((img_path, label))
                    break
        
        # 클래스 분포 계산
        labels_list = [l for _, l in self.image_files]
        self.num_positive = sum(labels_list)
        self.num_negative = len(labels_list) - self.num_positive
        
        print(f"[GateDataset] {split}: {len(self.image_files)} images")
        print(f"  ├─ SR needed (1): {self.num_positive} ({self.num_positive/len(self.image_files)*100:.1f}%)")
        print(f"  └─ Bypass OK (0): {self.num_negative} ({self.num_negative/len(self.image_files)*100:.1f}%)")
    
    def __len__(self) -> int:
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_path, label = self.image_files[idx]
        
        # 이미지 로드
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize
        img = cv2.resize(img, (self.img_size, self.img_size))
        
        # Augmentation
        if self.augment:
            # Horizontal flip
            if random.random() < 0.5:
                img = np.fliplr(img).copy()
            
            # Brightness/Contrast
            if random.random() < 0.3:
                alpha = random.uniform(0.8, 1.2)  # contrast
                beta = random.uniform(-20, 20)    # brightness
                img = np.clip(alpha * img + beta, 0, 255).astype(np.uint8)
        
        # To tensor
        img = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)
        label = torch.tensor(label, dtype=torch.float32)
        
        return {
            'image': img,
            'label': label,
            'image_name': img_path.stem
        }


def create_dataloader(
    lr_root: str,
    labels_path: str,
    split: str,
    batch_size: int = 32,
    img_size: int = 160,
    num_workers: int = 4,
    augment: bool = True
) -> DataLoader:
    """DataLoader 생성"""
    dataset = GateDataset(lr_root, labels_path, split, img_size, augment)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == 'train')
    )


# =============================================================================
# Trainer
# =============================================================================

class GateTrainer:
    """Gate Network 학습 관리자"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        args
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.args = args
        
        self.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Class weights (imbalanced data 대응)
        if hasattr(train_loader.dataset, 'num_positive'):
            num_pos = train_loader.dataset.num_positive
            num_neg = train_loader.dataset.num_negative
            pos_weight = num_neg / (num_pos + 1e-6)
            self.pos_weight = torch.tensor([pos_weight], device=self.device)
            print(f"[Trainer] Class weights - pos_weight: {pos_weight:.2f}")
        else:
            self.pos_weight = None
        
        # Loss
        self.criterion = nn.BCELoss()
        
        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=0.01
        )
        
        # Scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=args.epochs,
            eta_min=args.lr * 0.01
        )
        
        # Logging
        self.log_dir = Path(args.output) / 'logs'
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(self.log_dir) if HAS_TB else None
        
        # Checkpoints
        self.ckpt_dir = Path(args.output)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        
        self.best_f1 = 0.0
        self.global_step = 0
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """한 에폭 학습"""
        self.model.train()
        
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for batch in pbar:
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device).unsqueeze(1)
            
            # Forward
            outputs = self.model(images)
            
            # Loss (with class weights)
            if self.pos_weight is not None:
                weights = torch.where(labels == 1, self.pos_weight, torch.ones_like(labels))
                loss = F.binary_cross_entropy(outputs, labels, weight=weights)
            else:
                loss = self.criterion(outputs, labels)
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Metrics
            total_loss += loss.item()
            all_preds.extend((outputs >= 0.5).cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())
            
            # Progress
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
            # Log
            self.global_step += 1
            if self.writer and self.global_step % 50 == 0:
                self.writer.add_scalar('train/loss', loss.item(), self.global_step)
        
        # Metrics 계산
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        accuracy = (all_preds == all_labels).mean()
        
        return {
            'loss': total_loss / len(self.train_loader),
            'accuracy': accuracy
        }
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validation"""
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        
        total_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []
        
        for batch in tqdm(self.val_loader, desc="Validation"):
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device).unsqueeze(1)
            
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            total_loss += loss.item()
            all_probs.extend(outputs.cpu().numpy().flatten())
            all_preds.extend((outputs >= 0.5).cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())
        
        # Metrics 계산
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        accuracy = (all_preds == all_labels).mean()
        
        # Precision, Recall, F1 (for class 1 = SR needed)
        tp = ((all_preds == 1) & (all_labels == 1)).sum()
        fp = ((all_preds == 1) & (all_labels == 0)).sum()
        fn = ((all_preds == 0) & (all_labels == 1)).sum()
        
        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)
        
        return {
            'loss': total_loss / len(self.val_loader),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def save_checkpoint(self, epoch: int, metrics: Dict, is_best: bool = False):
        """체크포인트 저장"""
        ckpt = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'args': vars(self.args)
        }
        
        # Last
        torch.save(ckpt, self.ckpt_dir / 'gate_last.pt')
        
        # Best
        if is_best:
            torch.save(ckpt, self.ckpt_dir / 'gate_best.pt')
            print(f"  ✓ Best model saved (F1: {metrics.get('f1', 0):.4f})")
    
    def train(self):
        """전체 학습"""
        print("\n" + "=" * 60)
        print("🚀 Gate Network Training")
        print("=" * 60)
        print(f"  Epochs: {self.args.epochs}")
        print(f"  Batch size: {self.args.batch_size}")
        print(f"  Learning rate: {self.args.lr}")
        print(f"  Device: {self.device}")
        print("=" * 60 + "\n")
        
        for epoch in range(1, self.args.epochs + 1):
            # Train
            train_metrics = self.train_epoch(epoch)
            
            # Scheduler
            self.scheduler.step()
            
            # Validate
            val_metrics = self.validate() if self.val_loader else {}
            
            # Best model 체크
            current_f1 = val_metrics.get('f1', 0)
            is_best = current_f1 > self.best_f1
            if is_best:
                self.best_f1 = current_f1
            
            # Save
            self.save_checkpoint(epoch, {**train_metrics, **val_metrics}, is_best)
            
            # Print
            print(f"[Epoch {epoch}/{self.args.epochs}] "
                  f"Train Loss: {train_metrics['loss']:.4f} | "
                  f"Val Loss: {val_metrics.get('loss', 0):.4f} | "
                  f"Acc: {val_metrics.get('accuracy', 0)*100:.1f}% | "
                  f"F1: {val_metrics.get('f1', 0):.4f}")
        
        print("\n" + "=" * 60)
        print("✓ Training completed!")
        print(f"  Best F1: {self.best_f1:.4f}")
        print(f"  Checkpoints: {self.ckpt_dir}")
        print("=" * 60)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train Gate Network')
    
    # Paths
    parser.add_argument('--lr_root', type=str, required=True,
                        help='LR 데이터셋 루트 경로')
    parser.add_argument('--labels_dir', type=str, required=True,
                        help='Gate 라벨 디렉토리')
    parser.add_argument('--output', type=str, default='./checkpoints/gate',
                        help='출력 디렉토리')
    
    # Label file prefix (gt 기반 vs 기존)
    parser.add_argument('--label_prefix', type=str, default='gate_labels_gt',
                        help='라벨 파일 prefix (gate_labels_gt or gate_labels)')
    
    # Training
    parser.add_argument('--epochs', type=int, default=30, help='에폭 수')
    parser.add_argument('--batch_size', type=int, default=32, help='배치 크기')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--img_size', type=int, default=160, help='입력 이미지 크기')
    parser.add_argument('--workers', type=int, default=4, help='DataLoader workers')
    
    # Hardware
    parser.add_argument('--device', type=str, default='cuda', help='디바이스')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Seed
    set_seed(args.seed)
    
    # 경로 확인
    lr_root = Path(args.lr_root)
    labels_dir = Path(args.labels_dir)
    
    if not lr_root.exists():
        print(f"[Error] LR root not found: {lr_root}")
        return
    
    if not labels_dir.exists():
        print(f"[Error] Labels dir not found: {labels_dir}")
        return
    
    # 라벨 파일 경로
    train_labels = labels_dir / f'{args.label_prefix}_train.json'
    val_labels = labels_dir / f'{args.label_prefix}_val.json'
    
    if not train_labels.exists():
        print(f"[Error] Train labels not found: {train_labels}")
        return
    
    print(f"[Labels] Train: {train_labels}")
    print(f"[Labels] Val: {val_labels}")
    
    # DataLoader
    train_loader = create_dataloader(
        str(lr_root), str(train_labels), 'train',
        args.batch_size, args.img_size, args.workers, augment=True
    )
    
    val_loader = None
    if val_labels.exists():
        val_loader = create_dataloader(
            str(lr_root), str(val_labels), 'val',
            args.batch_size, args.img_size, args.workers, augment=False
        )
    
    # Model
    print(f"\n[Model] Creating LightweightGate...")
    model = LightweightGate(in_channels=3, base_channels=32)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {num_params:,} ({num_params/1000:.1f}K)")
    
    # Trainer
    trainer = GateTrainer(model, train_loader, val_loader, args)
    
    # Train
    trainer.train()


if __name__ == '__main__':
    main()