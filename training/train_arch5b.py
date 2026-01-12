#!/usr/bin/env python
"""
=============================================================================
train_arch5b.py - Arch5B Fusion Training Script
=============================================================================
Arch5B (Feature-Level Fusion) 전용 학습 스크립트

[학습 모드]
Mode 1 (--mode scratch): 기본 pretrained 가중치로 Fusion 학습
    - SR: ImageNet pretrained 또는 기본
    - YOLO: COCO pretrained (yolov8n.pt)
    
Mode 2 (--mode finetune): 선박 특화 가중치로 Fine-tuning
    - SR: 선박 데이터로 학습된 가중치
    - YOLO: 선박 데이터로 학습된 가중치

[가정]
- SR 모델 (MambaSR/RFDN): 개별 학습 완료, 가중치 파일 있음
- YOLO: 개별 학습 완료, 가중치 파일 있음

사용법:
    # Mode 1: 기본 가중치로 처음부터 학습
    python train_arch5b.py --mode scratch \
        --hr_root /path/to/hr_dataset \
        --lr_root /path/to/lr_dataset
    
    # Mode 2: 선박 특화 가중치로 fine-tuning
    python train_arch5b.py --mode finetune \
        --hr_root /path/to/hr_dataset \
        --lr_root /path/to/lr_dataset \
        --sr_weights /path/to/sr_ship.pth \
        --yolo_weights /path/to/yolo_ship.pt
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import json
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

# Tensorboard
try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TB = True
except ImportError:
    HAS_TB = False

# 프로젝트 루트 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.dataset import create_dataloader


def set_seed(seed: int = 42):
    """재현성을 위한 시드 설정"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_arch5b_model(
    sr_type: str = 'mamba',
    sr_weights: Optional[str] = None,
    yolo_weights: str = 'yolov8n.pt',
    num_classes: int = 1,
    device: str = 'cuda'
) -> nn.Module:
    """
    Arch5B 모델 생성
    
    Args:
        sr_type: 'rfdn' or 'mamba'
        sr_weights: SR 가중치 경로 (None이면 기본값)
        yolo_weights: YOLO 가중치 경로
        num_classes: 클래스 수
        device: 디바이스
    """
    from types import SimpleNamespace
    
    # Config 구성
    config = SimpleNamespace(
        device=device,
        model=SimpleNamespace(
            sr_type=sr_type,
            yolo=SimpleNamespace(
                weights_path=yolo_weights,
                num_classes=num_classes
            ),
            rfdn=SimpleNamespace(nf=50, num_modules=4),
            mamba=SimpleNamespace(
                embed_dim=48,
                depths=[5, 5, 5, 5],
                pretrain_path=sr_weights if sr_type == 'mamba' else None
            )
        ),
        data=SimpleNamespace(upscale_factor=4),
        training=SimpleNamespace(
            sr_weight=0.3,
            det_weight=0.7,
            freeze_detector=True
        )
    )
    
    # Arch5B 모델 생성
    from src.models.pipelines.arch5b_fusion import Arch5BFusion
    model = Arch5BFusion(config)
    
    # SR 가중치 로드 (RFDN인 경우)
    if sr_type == 'rfdn' and sr_weights:
        print(f"[Model] Loading SR weights: {sr_weights}")
        state_dict = torch.load(sr_weights, map_location='cpu')
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        model.sr_model.load_state_dict(state_dict, strict=False)
    
    return model.to(device)


class Arch5BTrainer:
    """Arch5B Fusion 학습 관리자"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        args
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.args = args
        
        self.device = torch.device(args.device)
        self.model.to(self.device)
        
        # 학습 대상 파라미터 설정
        self._setup_trainable_params()
        
        # Optimizer
        self.optimizer = AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=args.lr,
            weight_decay=0.01
        )
        
        # Scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=args.epochs,
            eta_min=args.lr * 0.01
        )
        
        # AMP
        self.use_amp = args.amp and torch.cuda.is_available()
        self.scaler = GradScaler() if self.use_amp else None
        
        # Logging
        self.log_dir = Path(args.log_dir) / args.exp_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(self.log_dir) if HAS_TB else None
        
        # Checkpoints
        self.ckpt_dir = Path(args.ckpt_dir) / args.exp_name
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        
        self.best_loss = float('inf')
        self.global_step = 0
    
    def _setup_trainable_params(self):
        """학습 대상 파라미터 설정"""
        mode = self.args.mode
        
        # 전체 freeze 먼저
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Mode에 따라 unfreeze
        if mode == 'scratch':
            # Fusion 모듈 + SR 모델 학습
            print("[Training] Mode: scratch")
            print("  - Fusion module: trainable")
            print("  - SR model: trainable")
            print("  - YOLO: frozen")
            
            # Fusion 모듈
            if hasattr(self.model, 'fusion'):
                for param in self.model.fusion.parameters():
                    param.requires_grad = True
            
            # SR 모델
            if hasattr(self.model, 'sr_model'):
                for param in self.model.sr_model.parameters():
                    param.requires_grad = True
        
        elif mode == 'finetune':
            # Fusion 모듈만 학습 (SR, YOLO는 이미 학습됨)
            print("[Training] Mode: finetune")
            print("  - Fusion module: trainable")
            print("  - SR model: frozen (pretrained)")
            print("  - YOLO: frozen (pretrained)")
            
            # Fusion 모듈만
            if hasattr(self.model, 'fusion'):
                for param in self.model.fusion.parameters():
                    param.requires_grad = True
        
        # 파라미터 수 출력
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"  - Total params: {total:,}")
        print(f"  - Trainable params: {trainable:,}")
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """한 에폭 학습"""
        self.model.train()
        
        total_loss = 0.0
        sr_loss_sum = 0.0
        det_loss_sum = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for batch in pbar:
            lr_images = batch['lr_images'].to(self.device)
            hr_images = batch['hr_images'].to(self.device)
            targets = batch['targets'].to(self.device)
            
            # Forward + Loss
            with autocast(enabled=self.use_amp):
                outputs = self.model(lr_images, return_features=True)
                loss_dict = self.model.compute_loss(outputs, targets, hr_gt=hr_images, lr_image=lr_images)
                loss = loss_dict['total']
            
            # Backward
            self.optimizer.zero_grad()
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
                self.optimizer.step()
            
            # Metrics
            total_loss += loss.item()
            sr_loss_sum += loss_dict.get('sr_loss', torch.tensor(0)).item() if isinstance(loss_dict.get('sr_loss', 0), torch.Tensor) else loss_dict.get('sr_loss', 0)
            det_loss_sum += loss_dict.get('det_loss', torch.tensor(0)).item() if isinstance(loss_dict.get('det_loss', 0), torch.Tensor) else loss_dict.get('det_loss', 0)
            num_batches += 1
            
            # Progress
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })
            
            # Log
            self.global_step += 1
            if self.writer and self.global_step % 50 == 0:
                self.writer.add_scalar('train/loss', loss.item(), self.global_step)
        
        return {
            'loss': total_loss / num_batches,
            'sr_loss': sr_loss_sum / num_batches,
            'det_loss': det_loss_sum / num_batches
        }
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validation - Detection Loss only"""
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        for batch in tqdm(self.val_loader, desc="Validation"):
            lr_images = batch['lr_images'].to(self.device)
            hr_images = batch['hr_images'].to(self.device)
            targets = batch['targets'].to(self.device)
            
            with autocast(enabled=self.use_amp):
                outputs = self.model(lr_images, return_features=True)
                loss_dict = self.model.compute_loss(outputs, targets, hr_gt=hr_images, lr_image=lr_images)
            
            total_loss += loss_dict['total'].item()
            num_batches += 1
        
        return {'loss': total_loss / num_batches}
    
    def save_checkpoint(self, epoch: int, metrics: Dict, is_best: bool = False):
        """체크포인트 저장"""
        ckpt = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'args': vars(self.args)
        }
        
        # Last
        torch.save(ckpt, self.ckpt_dir / 'last.pt')
        
        # Best
        if is_best:
            torch.save(ckpt, self.ckpt_dir / 'best.pt')
            print(f"  ✓ Best model saved (loss: {metrics['loss']:.4f})")
        
        # Periodic
        if epoch % 10 == 0:
            torch.save(ckpt, self.ckpt_dir / f'epoch_{epoch:03d}.pt')
    
    def train(self):
        """전체 학습 실행"""
        print("\n" + "=" * 60)
        print("🚀 Arch5B Fusion Training")
        print("=" * 60)
        print(f"  Mode: {self.args.mode}")
        print(f"  SR type: {self.args.sr_type}")
        print(f"  Epochs: {self.args.epochs}")
        print(f"  Batch size: {self.args.batch_size}")
        print(f"  Device: {self.device}")
        print(f"  AMP: {self.use_amp}")
        print("=" * 60 + "\n")
        
        for epoch in range(1, self.args.epochs + 1):
            # Train
            train_metrics = self.train_epoch(epoch)
            
            # Scheduler
            self.scheduler.step()
            
            # Validate
            val_metrics = self.validate() if self.val_loader else {}
            
            # Log
            current_loss = val_metrics.get('loss', train_metrics['loss'])
            is_best = current_loss < self.best_loss
            if is_best:
                self.best_loss = current_loss
            
            # Save
            self.save_checkpoint(epoch, {**train_metrics, **val_metrics}, is_best)
            
            # Print
            print(f"[Epoch {epoch}/{self.args.epochs}] "
                  f"Train Loss: {train_metrics['loss']:.4f} | "
                  f"Val Loss: {val_metrics.get('loss', 0):.4f} | "
                  f"PSNR: {val_metrics.get('psnr', 0):.2f} dB")
        
        print("\n" + "=" * 60)
        print("✓ Training completed!")
        print(f"  Best Loss: {self.best_loss:.4f}")
        print(f"  Checkpoints: {self.ckpt_dir}")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Arch5B Fusion Training')
    
    # Mode
    parser.add_argument('--mode', type=str, default='scratch',
                        choices=['scratch', 'finetune'],
                        help='Training mode: scratch (기본 가중치) or finetune (선박 특화 가중치)')
    
    # Model
    parser.add_argument('--sr_type', type=str, default='mamba',
                        choices=['rfdn', 'mamba'], help='SR model type')
    parser.add_argument('--sr_weights', type=str, default=None,
                        help='SR 가중치 경로 (finetune 모드)')
    parser.add_argument('--yolo_weights', type=str, default='yolov8n.pt',
                        help='YOLO 가중치 경로')
    parser.add_argument('--num_classes', type=int, default=1, help='클래스 수')
    
    # Data
    parser.add_argument('--hr_root', type=str, required=True, help='HR 데이터셋 경로')
    parser.add_argument('--lr_root', type=str, required=True, help='LR 데이터셋 경로')
    parser.add_argument('--hr_size', type=int, default=640, help='HR 이미지 크기')
    parser.add_argument('--lr_size', type=int, default=160, help='LR 이미지 크기')
    
    # Training
    parser.add_argument('--batch_size', type=int, default=8, help='배치 크기')
    parser.add_argument('--epochs', type=int, default=100, help='에폭 수')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--workers', type=int, default=4, help='DataLoader workers')
    
    # Logging
    parser.add_argument('--exp_name', type=str, default='arch5b_exp', help='실험 이름')
    parser.add_argument('--log_dir', type=str, default='./logs', help='로그 디렉토리')
    parser.add_argument('--ckpt_dir', type=str, default='./checkpoints', help='체크포인트 디렉토리')
    
    # Hardware
    parser.add_argument('--device', type=str, default='cuda', help='디바이스')
    parser.add_argument('--amp', action='store_true', default=True, help='Mixed precision')
    parser.add_argument('--no_amp', action='store_true', help='Disable AMP')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    if args.no_amp:
        args.amp = False
    
    # Seed
    set_seed(args.seed)
    
    # Auto experiment name
    if args.exp_name == 'arch5b_exp':
        args.exp_name = f"arch5b_{args.sr_type}_{args.mode}"
    
    # 경로 확인
    if not Path(args.hr_root).exists():
        print(f"[Error] HR root not found: {args.hr_root}")
        return
    if not Path(args.lr_root).exists():
        print(f"[Error] LR root not found: {args.lr_root}")
        return
    
    print(f"[Data] HR: {args.hr_root}")
    print(f"[Data] LR: {args.lr_root}")
    
    # DataLoader
    train_loader = create_dataloader(
        args.hr_root, args.lr_root, 'train',
        args.batch_size, args.hr_size, args.lr_size, args.workers
    )
    val_loader = create_dataloader(
        args.hr_root, args.lr_root, 'val',
        args.batch_size, args.hr_size, args.lr_size, args.workers, augment=False
    )
    
    # Model
    print(f"\n[Model] Creating Arch5B + {args.sr_type.upper()}...")
    
    # Mode에 따른 가중치 설정
    if args.mode == 'finetune' and args.sr_weights is None:
        print("[Warning] Finetune mode but no SR weights specified!")
        print("  Use --sr_weights /path/to/sr_ship.pth")
    
    model = create_arch5b_model(
        sr_type=args.sr_type,
        sr_weights=args.sr_weights,
        yolo_weights=args.yolo_weights,
        num_classes=args.num_classes,
        device=args.device
    )
    
    # Trainer
    trainer = Arch5BTrainer(model, train_loader, val_loader, args)
    
    # Train
    trainer.train()


if __name__ == '__main__':
    main()