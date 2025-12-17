"""
=============================================================================
train_arch5b.py - Architecture 5-B: Feature Fusion Training
=============================================================================

[Arch5-B 학습 전략]

Phase 1: 개별 Pretrain (이 스크립트 외부에서)
├── RFDN: train_sr_only.py로 학습 → rfdn_best.pth
└── YOLO: ultralytics CLI로 학습 → yolo_ship.pt

Phase 2: Fusion만 학습 (이 스크립트)
├── RFDN: ❄️ Frozen (pretrained 가중치 로드)
├── YOLO: ❄️ Frozen (pretrained 가중치 로드)
└── Fusion: 🔥 학습!
    └── Detection Loss → Fusion 모듈로 역전파
    └── "SR Feature가 Detection에 어떻게 기여하는지" 학습

Phase 3: 전체 Fine-tune (이 스크립트)
├── RFDN: 🔥 낮은 LR (0.1x)
├── YOLO: 🔥 낮은 LR (0.1x)
└── Fusion: 🔥 기본 LR

[사용법]
# Phase 2: Fusion만 학습
python scripts/train_arch5b.py \
    --lr_dir data/lr/images/train \
    --label_dir data/labels/train \
    --rfdn_weights checkpoints/rfdn_best.pth \
    --yolo_weights checkpoints/yolo_ship.pt \
    --phase 2 \
    --epochs 50

# Phase 3: 전체 Fine-tune
python scripts/train_arch5b.py \
    --lr_dir data/lr/images/train \
    --label_dir data/labels/train \
    --arch5b_weights checkpoints/arch5b_phase2.pth \
    --phase 3 \
    --epochs 30 \
    --lr 1e-5

[출력]
- checkpoints/arch5b_phase{N}/best.pth
- logs/arch5b_phase{N}/tensorboard
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

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.pipelines.arch5b_fusion import Arch5BFusion


# =============================================================================
# Dataset
# =============================================================================

class Arch5BDataset(Dataset):
    """
    Arch5-B 학습용 데이터셋
    
    [반환]
    - lr_image: LR 이미지 [3, H, W]
    - hr_image: HR GT (선택, SR Loss용) [3, H*4, W*4]
    - targets: YOLO 형식 라벨 [N, 6]
    """
    
    def __init__(
        self,
        lr_dir: str,
        label_dir: str,
        hr_dir: str = None,  # SR Loss용 (Phase 1에서만 사용)
        img_size: int = 640,  # YOLO 입력 크기
        augment: bool = True
    ):
        self.lr_dir = Path(lr_dir)
        self.label_dir = Path(label_dir)
        self.hr_dir = Path(hr_dir) if hr_dir else None
        self.img_size = img_size
        self.augment = augment
        
        # 이미지 목록
        self.samples = []
        for lr_path in sorted(self.lr_dir.glob("*.jpg")):
            label_path = self.label_dir / (lr_path.stem + ".txt")
            hr_path = (self.hr_dir / lr_path.name) if self.hr_dir else None
            
            self.samples.append({
                'lr': lr_path,
                'hr': hr_path,
                'label': label_path if label_path.exists() else None
            })
        
        print(f"[Arch5BDataset] Found {len(self.samples)} samples")
        print(f"  - LR dir: {lr_dir}")
        print(f"  - Label dir: {label_dir}")
        print(f"  - HR dir: {hr_dir if hr_dir else 'None (no SR loss)'}")
    
    def __len__(self):
        return len(self.samples)
    
    def load_image(self, path: Path, target_size: int = None) -> torch.Tensor:
        """이미지 로드 및 전처리"""
        img = cv2.imread(str(path))
        if img is None:
            return None
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if target_size:
            img = cv2.resize(img, (target_size, target_size))
        
        # HWC → CHW, 0-255 → 0-1
        tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        return tensor
    
    def load_labels(self, label_path: Path, batch_idx: int = 0) -> list:
        """YOLO 형식 라벨 로드"""
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
        
        # LR 이미지 로드 (YOLO 입력 크기로)
        lr_img = self.load_image(sample['lr'], self.img_size)
        if lr_img is None:
            return self.__getitem__((idx + 1) % len(self))
        
        # HR 이미지 로드 (있으면)
        hr_img = None
        if sample['hr'] and sample['hr'].exists():
            hr_img = self.load_image(sample['hr'], self.img_size * 4)
        
        # 라벨 로드
        labels = self.load_labels(sample['label'], 0)
        
        # Augmentation (간단한 버전)
        if self.augment:
            # 수평 플립
            if np.random.rand() > 0.5:
                lr_img = torch.flip(lr_img, dims=[2])  # W 축
                if hr_img is not None:
                    hr_img = torch.flip(hr_img, dims=[2])
                # 라벨 x 좌표 반전
                for label in labels:
                    label[2] = 1.0 - label[2]
        
        return lr_img, hr_img, labels


def collate_fn(batch):
    """Custom collate function"""
    lr_imgs = torch.stack([item[0] for item in batch])
    
    # HR 이미지 (None일 수 있음)
    hr_imgs = None
    if batch[0][1] is not None:
        hr_imgs = torch.stack([item[1] for item in batch if item[1] is not None])
    
    # 라벨 합치기 (batch_idx 업데이트)
    all_labels = []
    for batch_idx, item in enumerate(batch):
        for label in item[2]:
            label[0] = batch_idx
            all_labels.append(label)
    
    if all_labels:
        targets = torch.tensor(all_labels, dtype=torch.float32)
    else:
        targets = torch.zeros((0, 6), dtype=torch.float32)
    
    return lr_imgs, hr_imgs, targets


# =============================================================================
# Training Functions
# =============================================================================

def train_one_epoch(
    model: Arch5BFusion,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    phase: int,
    use_sr_loss: bool = False
) -> dict:
    """
    한 에폭 학습
    
    Args:
        model: Arch5BFusion 모델
        dataloader: 데이터로더
        optimizer: 옵티마이저
        device: 디바이스
        epoch: 현재 에폭
        phase: 학습 페이즈 (2 또는 3)
        use_sr_loss: SR Loss 사용 여부
    
    Returns:
        에폭 통계
    """
    model.train()
    
    # Phase별 freeze 설정
    if phase == 2:
        model.freeze_for_phase2()
    
    total_loss = 0
    total_det_loss = 0
    total_sr_loss = 0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} (Phase {phase})")
    
    for lr_imgs, hr_imgs, targets in pbar:
        lr_imgs = lr_imgs.to(device)
        targets = targets.to(device)
        hr_gt = hr_imgs.to(device) if hr_imgs is not None and use_sr_loss else None
        
        # Forward
        outputs, features = model(lr_imgs, return_features=True)
        
        # Loss 계산
        loss_dict = model.compute_loss(
            outputs=(outputs, features),
            targets=targets,
            lr_image=lr_imgs,
            hr_gt=hr_gt
        )
        
        loss = loss_dict['total']
        
        # Backward
        optimizer.zero_grad()
        
        if loss.requires_grad:
            loss.backward()
            
            # Gradient clipping (안정성)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            
            optimizer.step()
        
        # 통계
        total_loss += loss.item()
        total_det_loss += loss_dict['det_loss'].item()
        total_sr_loss += loss_dict['sr_loss'].item()
        num_batches += 1
        
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'det': f"{loss_dict['det_loss'].item():.4f}",
            'sr': f"{loss_dict['sr_loss'].item():.4f}"
        })
    
    return {
        'total_loss': total_loss / num_batches,
        'det_loss': total_det_loss / num_batches,
        'sr_loss': total_sr_loss / num_batches
    }


@torch.no_grad()
def validate(
    model: Arch5BFusion,
    dataloader: DataLoader,
    device: torch.device
) -> dict:
    """검증"""
    model.eval()
    
    total_loss = 0
    total_det_loss = 0
    num_batches = 0
    
    for lr_imgs, hr_imgs, targets in tqdm(dataloader, desc="Validation"):
        lr_imgs = lr_imgs.to(device)
        targets = targets.to(device)
        
        outputs, features = model(lr_imgs, return_features=True)
        
        loss_dict = model.compute_loss(
            outputs=(outputs, features),
            targets=targets,
            lr_image=lr_imgs
        )
        
        total_loss += loss_dict['total'].item()
        total_det_loss += loss_dict['det_loss'].item()
        num_batches += 1
    
    return {
        'val_loss': total_loss / num_batches,
        'val_det_loss': total_det_loss / num_batches
    }


# =============================================================================
# Main Training
# =============================================================================

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Phase: {args.phase}")
    
    # 출력 디렉토리
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"arch5b_phase{args.phase}_{timestamp}"
    ckpt_dir = Path(args.output_dir) / "checkpoints" / exp_name
    log_dir = Path(args.output_dir) / "logs" / exp_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    writer = SummaryWriter(log_dir)
    
    # =========================================================================
    # Model
    # =========================================================================
    config = SimpleNamespace(
        model=SimpleNamespace(
            rfdn=SimpleNamespace(nf=args.nf, num_modules=args.num_modules),
            yolo=SimpleNamespace(weights_path=args.yolo_weights, num_classes=args.num_classes),
            fusion=SimpleNamespace(
                use_cross_attention=args.use_cross_attention,
                use_cbam=args.use_cbam,
                num_heads=args.num_heads
            )
        ),
        data=SimpleNamespace(upscale_factor=4),
        training=SimpleNamespace(sr_weight=args.sr_weight, det_weight=args.det_weight),
        device=str(device)
    )
    
    print("\n[Model] Creating Arch5BFusion...")
    model = Arch5BFusion(config)
    model = model.to(device)
    
    # =========================================================================
    # Pretrained 가중치 로드
    # =========================================================================
    
    if args.phase == 2:
        # Phase 2: RFDN, YOLO 개별 가중치 로드
        if args.rfdn_weights and Path(args.rfdn_weights).exists():
            print(f"[Model] Loading RFDN: {args.rfdn_weights}")
            ckpt = torch.load(args.rfdn_weights, map_location=device)
            state = ckpt.get('model_state_dict', ckpt)
            model.sr_model.load_state_dict(state)
        
        # YOLO는 config에서 이미 로드됨
        print(f"[Model] YOLO loaded from: {args.yolo_weights}")
        
    elif args.phase == 3:
        # Phase 3: 전체 모델 가중치 로드 (Phase 2 결과)
        if args.arch5b_weights and Path(args.arch5b_weights).exists():
            print(f"[Model] Loading Arch5B: {args.arch5b_weights}")
            ckpt = torch.load(args.arch5b_weights, map_location=device)
            model.load_state_dict(ckpt['model_state_dict'])
        else:
            print("[Warning] Phase 3 without Phase 2 weights!")
    
    # =========================================================================
    # Dataset
    # =========================================================================
    print("\n[Dataset] Loading...")
    
    train_dataset = Arch5BDataset(
        lr_dir=args.lr_dir,
        label_dir=args.label_dir,
        hr_dir=args.hr_dir if args.use_sr_loss else None,
        img_size=args.img_size,
        augment=True
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    # Validation (선택)
    val_loader = None
    if args.val_lr_dir:
        val_dataset = Arch5BDataset(
            lr_dir=args.val_lr_dir,
            label_dir=args.val_label_dir,
            img_size=args.img_size,
            augment=False
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_fn
        )
    
    # =========================================================================
    # Optimizer (Phase별 다른 설정)
    # =========================================================================
    
    if args.phase == 2:
        # Phase 2: Fusion만 학습
        model.freeze_for_phase2()
        params = list(model.fusion.parameters())
        optimizer = optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
        
        print(f"\n[Optimizer] Phase 2: Fusion only")
        print(f"  - Parameters: {sum(p.numel() for p in params):,}")
        print(f"  - LR: {args.lr}")
        
    elif args.phase == 3:
        # Phase 3: 전체 학습 (다른 LR)
        param_groups = model.unfreeze_for_phase3()
        
        optimizer = optim.AdamW([
            {'params': param_groups['fusion'], 'lr': args.lr, 'name': 'fusion'},
            {'params': param_groups['sr'], 'lr': args.lr * 0.1, 'name': 'sr'},
            {'params': param_groups['detector'], 'lr': args.lr * 0.1, 'name': 'detector'}
        ], weight_decay=1e-4)
        
        print(f"\n[Optimizer] Phase 3: Full fine-tune")
        print(f"  - Fusion LR: {args.lr}")
        print(f"  - SR/YOLO LR: {args.lr * 0.1}")
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # =========================================================================
    # Training Loop
    # =========================================================================
    print(f"\n[Training] Starting Phase {args.phase} for {args.epochs} epochs...")
    
    best_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        # Train
        train_stats = train_one_epoch(
            model, train_loader, optimizer, device, epoch,
            phase=args.phase,
            use_sr_loss=args.use_sr_loss
        )
        
        # Validate
        val_stats = {}
        if val_loader:
            val_stats = validate(model, val_loader, device)
        
        # Scheduler
        scheduler.step()
        
        # Logging
        writer.add_scalar('Loss/train', train_stats['total_loss'], epoch)
        writer.add_scalar('Loss/det', train_stats['det_loss'], epoch)
        writer.add_scalar('Loss/sr', train_stats['sr_loss'], epoch)
        if val_stats:
            writer.add_scalar('Loss/val', val_stats['val_loss'], epoch)
        
        # Print
        print(f"\nEpoch {epoch}/{args.epochs}")
        print(f"  Train Loss: {train_stats['total_loss']:.4f} "
              f"(Det: {train_stats['det_loss']:.4f}, SR: {train_stats['sr_loss']:.4f})")
        if val_stats:
            print(f"  Val Loss: {val_stats['val_loss']:.4f}")
        
        # Save
        current_loss = val_stats.get('val_loss', train_stats['total_loss'])
        is_best = current_loss < best_loss
        if is_best:
            best_loss = current_loss
        
        checkpoint = {
            'epoch': epoch,
            'phase': args.phase,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': current_loss,
            'args': vars(args)
        }
        
        torch.save(checkpoint, ckpt_dir / 'last.pth')
        if is_best:
            torch.save(checkpoint, ckpt_dir / 'best.pth')
            print(f"  ✓ New best! Loss: {best_loss:.4f}")
        
        if epoch % args.save_freq == 0:
            torch.save(checkpoint, ckpt_dir / f'epoch_{epoch}.pth')
    
    writer.close()
    print(f"\n✓ Phase {args.phase} training completed!")
    print(f"  Best loss: {best_loss:.4f}")
    print(f"  Checkpoints: {ckpt_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arch5-B Feature Fusion Training")
    
    # Phase
    parser.add_argument('--phase', type=int, required=True, choices=[2, 3],
                       help='Training phase (2: Fusion only, 3: Full fine-tune)')
    
    # Data
    parser.add_argument('--lr_dir', type=str, required=True, help='LR images directory')
    parser.add_argument('--label_dir', type=str, required=True, help='Labels directory')
    parser.add_argument('--hr_dir', type=str, default=None, help='HR images for SR loss')
    parser.add_argument('--val_lr_dir', type=str, default=None, help='Validation LR dir')
    parser.add_argument('--val_label_dir', type=str, default=None, help='Validation labels')
    parser.add_argument('--img_size', type=int, default=640, help='Input image size')
    
    # Model weights
    parser.add_argument('--rfdn_weights', type=str, default=None, help='RFDN pretrained weights')
    parser.add_argument('--yolo_weights', type=str, default='yolov8n.pt', help='YOLO weights')
    parser.add_argument('--arch5b_weights', type=str, default=None, help='Arch5B weights (for Phase 3)')
    
    # Model config
    parser.add_argument('--nf', type=int, default=50, help='RFDN feature channels')
    parser.add_argument('--num_modules', type=int, default=4, help='RFDN blocks')
    parser.add_argument('--num_classes', type=int, default=1, help='Number of classes')
    parser.add_argument('--use_cross_attention', type=bool, default=True)
    parser.add_argument('--use_cbam', type=bool, default=True)
    parser.add_argument('--num_heads', type=int, default=4)
    
    # Loss weights
    parser.add_argument('--sr_weight', type=float, default=0.0, help='SR loss weight')
    parser.add_argument('--det_weight', type=float, default=1.0, help='Detection loss weight')
    parser.add_argument('--use_sr_loss', action='store_true', help='Use SR loss (requires hr_dir)')
    
    # Training
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=4)
    
    # Output
    parser.add_argument('--output_dir', type=str, default='runs/arch5b')
    parser.add_argument('--save_freq', type=int, default=10)
    
    args = parser.parse_args()
    main(args)