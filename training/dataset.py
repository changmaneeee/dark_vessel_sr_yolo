"""
=============================================================================
dataset.py - SR-Detection Dataset (Arch5B Training)
=============================================================================
HR/LR 이미지 쌍과 YOLO format label 로드

[가정]
- SR, YOLO는 개별 학습 완료
- 이 DataLoader는 Arch5B Fusion 학습용
"""

import os
from pathlib import Path
from typing import Dict, List, Any, Optional
import random

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2


class SRDetectionDataset(Dataset):
    """
    Arch5B Fusion 학습용 데이터셋
    
    Args:
        hr_root: HR 데이터셋 루트 (hr_dataset/)
        lr_root: LR 데이터셋 루트 (lr_dataset/)
        split: 'train' or 'val'
        hr_size: HR 이미지 크기
        lr_size: LR 이미지 크기
        augment: Augmentation 여부
    """
    
    def __init__(
        self,
        hr_root: str,
        lr_root: str,
        split: str = 'train',
        hr_size: int = 640,
        lr_size: int = 160,
        augment: bool = True
    ):
        self.hr_root = Path(hr_root)
        self.lr_root = Path(lr_root)
        self.split = split
        self.hr_size = hr_size
        self.lr_size = lr_size
        self.augment = augment and (split == 'train')
        
        # 경로 (YOLO format: images/, labels/)
        self.hr_image_dir = self.hr_root / 'images' / split
        self.lr_image_dir = self.lr_root / 'images' / split
        self.label_dir = self.hr_root / 'labels' / split
        
        # 파일 목록
        self.image_files = self._get_valid_files()
        print(f"[Dataset] {split}: {len(self.image_files)} images")
    
    def _get_valid_files(self) -> List[str]:
        """HR, LR 모두 존재하는 파일만"""
        hr_files = {f.stem for f in self.hr_image_dir.glob('*.[jJ][pP][gG]')}
        hr_files.update(f.stem for f in self.hr_image_dir.glob('*.[pP][nN][gG]'))
        
        lr_files = {f.stem for f in self.lr_image_dir.glob('*.[jJ][pP][gG]')}
        lr_files.update(f.stem for f in self.lr_image_dir.glob('*.[pP][nN][gG]'))
        
        return sorted(hr_files & lr_files)
    
    def __len__(self) -> int:
        return len(self.image_files)
    
    def _find_image(self, directory: Path, stem: str) -> Optional[Path]:
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
            path = directory / f"{stem}{ext}"
            if path.exists():
                return path
        return None
    
    def _load_image(self, path: Path) -> np.ndarray:
        img = cv2.imread(str(path))
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    def _load_label(self, path: Path) -> np.ndarray:
        """YOLO format: class x_center y_center width height"""
        if not path.exists():
            return np.zeros((0, 5), dtype=np.float32)
        
        labels = []
        with open(path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    labels.append([int(parts[0])] + [float(x) for x in parts[1:5]])
        
        return np.array(labels, dtype=np.float32) if labels else np.zeros((0, 5), dtype=np.float32)
    
    def _resize(self, img: np.ndarray, size: int) -> np.ndarray:
        return cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    
    def _augment(self, hr: np.ndarray, lr: np.ndarray, labels: np.ndarray):
        """간단한 augmentation (HR, LR 동기화)"""
        # Horizontal flip
        if random.random() < 0.5:
            hr = np.fliplr(hr).copy()
            lr = np.fliplr(lr).copy()
            if len(labels) > 0:
                labels[:, 1] = 1.0 - labels[:, 1]
        return hr, lr, labels
    
    def _to_tensor(self, img: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        stem = self.image_files[idx]
        
        # Load
        hr = self._load_image(self._find_image(self.hr_image_dir, stem))
        lr = self._load_image(self._find_image(self.lr_image_dir, stem))
        labels = self._load_label(self.label_dir / f"{stem}.txt")
        
        # Resize
        hr = self._resize(hr, self.hr_size)
        lr = self._resize(lr, self.lr_size)
        
        # Augment
        if self.augment:
            hr, lr, labels = self._augment(hr, lr, labels)
        
        return {
            'lr_image': self._to_tensor(lr),
            'hr_image': self._to_tensor(hr),
            'labels': torch.from_numpy(labels),
            'image_id': stem
        }


def collate_fn(batch: List[Dict]) -> Dict[str, Any]:
    """Labels에 batch_idx 추가"""
    lr_images = torch.stack([b['lr_image'] for b in batch])
    hr_images = torch.stack([b['hr_image'] for b in batch])
    image_ids = [b['image_id'] for b in batch]
    
    # [batch_idx, class, x, y, w, h]
    targets = []
    for i, b in enumerate(batch):
        if len(b['labels']) > 0:
            batch_idx = torch.full((len(b['labels']), 1), i, dtype=torch.float32)
            targets.append(torch.cat([batch_idx, b['labels']], dim=1))
    
    targets = torch.cat(targets, 0) if targets else torch.zeros((0, 6))
    
    return {
        'lr_images': lr_images,
        'hr_images': hr_images,
        'targets': targets,
        'image_ids': image_ids
    }


def create_dataloader(
    hr_root: str,
    lr_root: str,
    split: str = 'train',
    batch_size: int = 8,
    hr_size: int = 640,
    lr_size: int = 160,
    num_workers: int = 4,
    augment: bool = True
) -> DataLoader:
    """DataLoader 생성"""
    dataset = SRDetectionDataset(
        hr_root, lr_root, split, hr_size, lr_size, augment
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=(split == 'train')
    )