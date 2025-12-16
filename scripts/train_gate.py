"""
=============================================================================
train_gate.py - Arch2 Gate Network 학습
=============================================================================

[역할]
RFDN, YOLO는 pretrained 상태로 Freeze
Gate Network만 학습하여 "최적의 SR 적용 판단" 학습

[학습 원리]
- Detection Loss가 Gate까지 역전파
- Gate가 "탐지 성능 향상에 유리한 방향"으로 학습
- SR 필요한 이미지 → gate ↑
- SR 불필요한 이미지 → gate ↓

[사용법]
python scripts/train_gate.py \
    --lr_dir data/lr/images/train \
    --label_dir data/labels/train \
    --rfdn_weights checkpoints/rfdn_best.pth \
    --yolo_weights checkpoints/yolo_ship.pt \
    --epochs 50 \
    --batch_size 8
"""

import os
import argparse
import torch
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

from src.models.pipelines.arch2_softgate import Arch2SoftGate


# =============================================================================
# Dataset
# =============================================================================

class GateTrainingDataset(Dataset):
    """Gate 학습용 데이터셋 (Detection 타겟 필요)"""
    
    def __init__(self, lr_dir, label_dir, img_size=160):
        self.lr_dir = Path(lr_dir)
        self.label_dir = Path(label_dir)
        self.img_size = img_size
        
        # 이미지 목록
        self.samples = []
        for lr_path in sorted(self.lr_dir.glob("*.jpg")):
            label_path = self.label_dir / (lr_path.stem + ".txt")
            self.samples.append({
                'lr': lr_path,
                'label': label_path if label_path.exists() else None
            })
        
        print(f"[GateTrainingDataset] Found {len(self.samples)} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def load_labels(self, label_path, batch_idx):
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
        
        # LR 이미지 로드
        lr_img = cv2.imread(str(sample['lr']))
        if lr_img is None:
            return self.__getitem__((idx + 1) % len(self))
        
        lr_img = cv2.cvtColor(lr_img, cv2.COLOR_BGR2RGB)
        lr_img = cv2.resize(lr_img, (self.img_size, self.img_size))
        
        # 라벨 로드
        labels = self.load_labels(sample['label'], 0)
        
        # Tensor 변환
        lr_tensor = torch.from_numpy(lr_img).permute(2, 0, 1).float() / 255.0
        
        return lr_tensor, labels


def collate_fn(batch):
    """Custom collate"""
    lr_imgs = torch.stack([item[0] for item in batch])
    
    all_labels = []
    for batch_idx, item in enumerate(batch):
        for label in item[1]:
            label[0] = batch_idx
            all_labels.append(label)
    
    if all_labels:
        targets = torch.tensor(all_labels, dtype=torch.float32)
    else:
        targets = torch.zeros((0, 6), dtype=torch.float32)
    
    return lr_imgs, targets


# =============================================================================
# Training
# =============================================================================

def train_one_epoch(model, dataloader, optimizer, device, epoch):
    """Gate 학습 (RFDN, YOLO frozen)"""
    
    model.train()
    model.freeze_sr_and_yolo()  # Gate만 학습!
    
    total_loss = 0
    gate_values = []
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for lr_imgs, targets in pbar:
        lr_imgs = lr_imgs.to(device)
        targets = targets.to(device)
        
        # Forward (return_intermediates로 sr_image 포함)
        outputs = model(lr_imgs, return_intermediates=True)
        
        # Loss 계산 (Detection loss → Gate로 역전파)
        # hr_gt 없으므로 SR loss는 0
        loss_dict = model.compute_loss(outputs, targets, hr_gt=None)
        loss = loss_dict['total']
        
        # Backward (Gate만 업데이트됨!)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        gate_values.extend(outputs['gate'].detach().cpu().squeeze().tolist())
        
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'gate_mean': f"{outputs['gate'].mean().item():.3f}"
        })
    
    return {
        'loss': total_loss / len(dataloader),
        'gate_mean': np.mean(gate_values),
        'gate_std': np.std(gate_values)
    }


@torch.no_grad()
def validate(model, dataloader, device):
    """Gate 동작 검증"""
    model.eval()
    
    gate_values = []
    det_counts = []
    
    for lr_imgs, targets in tqdm(dataloader, desc="Validation"):
        lr_imgs = lr_imgs.to(device)
        
        outputs = model(lr_imgs, return_intermediates=True)
        
        gate_values.extend(outputs['gate'].cpu().squeeze().tolist())
        
        for det in outputs['detections']:
            if isinstance(det, dict):
                det_counts.append(len(det.get('boxes', [])))
            else:
                det_counts.append(0)
    
    return {
        'gate_mean': np.mean(gate_values),
        'gate_std': np.std(gate_values),
        'sr_ratio': np.mean([g > 0.5 for g in gate_values]),  # SR 적용 비율
        'avg_detections': np.mean(det_counts)
    }


# =============================================================================
# Main
# =============================================================================

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # 출력 디렉토리
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"gate_{timestamp}"
    ckpt_dir = Path(args.output_dir) / "checkpoints" / exp_name
    log_dir = Path(args.output_dir) / "logs" / exp_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    writer = SummaryWriter(log_dir)
    
    # =========================================================================
    # Model (Arch2)
    # =========================================================================
    config = SimpleNamespace(
        model=SimpleNamespace(
            rfdn=SimpleNamespace(nf=50, num_modules=4),
            yolo=SimpleNamespace(weights_path=args.yolo_weights, num_classes=args.num_classes),
            gate=SimpleNamespace(base_channels=32, num_layers=4)
        ),
        data=SimpleNamespace(upscale_factor=4),
        training=SimpleNamespace(sr_weight=0.0, det_weight=1.0),  # Detection만!
        device=str(device)
    )
    
    print("\n[Model] Creating Arch2SoftGate...")
    model = Arch2SoftGate(config)
    model = model.to(device)
    
    # Pretrained 가중치 로드
    if args.rfdn_weights and Path(args.rfdn_weights).exists():
        print(f"[Model] Loading RFDN: {args.rfdn_weights}")
        ckpt = torch.load(args.rfdn_weights, map_location=device)
        if 'model_state_dict' in ckpt:
            model.sr_model.load_state_dict(ckpt['model_state_dict'])
        else:
            model.sr_model.load_state_dict(ckpt)
    
    # YOLO는 config에서 이미 로드됨
    print(f"[Model] YOLO loaded from config: {args.yolo_weights}")
    
    # Gate만 학습 모드
    model.freeze_sr_and_yolo()
    
    # =========================================================================
    # Dataset
    # =========================================================================
    print("\n[Dataset] Loading...")
    
    train_dataset = GateTrainingDataset(
        lr_dir=args.lr_dir,
        label_dir=args.label_dir,
        img_size=args.img_size
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )
    
    # =========================================================================
    # Optimizer (Gate만!)
    # =========================================================================
    gate_params = list(model.gate_network.parameters())
    optimizer = optim.AdamW(gate_params, lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    print(f"\n[Optimizer] Gate parameters: {sum(p.numel() for p in gate_params):,}")
    
    # =========================================================================
    # Training
    # =========================================================================
    print(f"\n[Training] Starting Gate training for {args.epochs} epochs...")
    
    best_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        # Train
        train_stats = train_one_epoch(model, train_loader, optimizer, device, epoch)
        
        # Validate
        val_stats = validate(model, train_loader, device)  # 같은 데이터로 검증
        
        scheduler.step()
        
        # Logging
        writer.add_scalar('Loss/train', train_stats['loss'], epoch)
        writer.add_scalar('Gate/mean', train_stats['gate_mean'], epoch)
        writer.add_scalar('Gate/std', train_stats['gate_std'], epoch)
        writer.add_scalar('Gate/sr_ratio', val_stats['sr_ratio'], epoch)
        
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"  Loss: {train_stats['loss']:.4f}")
        print(f"  Gate: mean={train_stats['gate_mean']:.3f}, std={train_stats['gate_std']:.3f}")
        print(f"  SR ratio: {val_stats['sr_ratio']:.2%}")
        
        # Save
        is_best = train_stats['loss'] < best_loss
        if is_best:
            best_loss = train_stats['loss']
        
        # Gate만 저장
        gate_state = {
            'epoch': epoch,
            'gate_state_dict': model.gate_network.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'gate_mean': train_stats['gate_mean'],
            'gate_std': train_stats['gate_std'],
            'args': vars(args)
        }
        
        torch.save(gate_state, ckpt_dir / 'gate_last.pth')
        if is_best:
            torch.save(gate_state, ckpt_dir / 'gate_best.pth')
            print(f"  ✓ New best!")
        
        # 전체 모델도 저장
        if epoch % args.save_freq == 0:
            full_state = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'args': vars(args)
            }
            torch.save(full_state, ckpt_dir / f'arch2_epoch_{epoch}.pth')
    
    writer.close()
    print(f"\n✓ Gate training completed!")
    print(f"  Checkpoints: {ckpt_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Data
    parser.add_argument('--lr_dir', type=str, required=True)
    parser.add_argument('--label_dir', type=str, required=True)
    parser.add_argument('--img_size', type=int, default=160)
    
    # Pretrained weights
    parser.add_argument('--rfdn_weights', type=str, required=True)
    parser.add_argument('--yolo_weights', type=str, default='yolov8n.pt')
    parser.add_argument('--num_classes', type=int, default=1)
    
    # Training
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-3)  # Gate는 높은 LR 가능
    parser.add_argument('--num_workers', type=int, default=4)
    
    # Output
    parser.add_argument('--output_dir', type=str, default='runs/gate')
    parser.add_argument('--save_freq', type=int, default=10)
    
    args = parser.parse_args()
    main(args)