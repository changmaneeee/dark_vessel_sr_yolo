#!/usr/bin/env python
"""
=============================================================================
inference.py - Arch 0, 2, 4 Inference Script
=============================================================================
개별 학습된 SR + YOLO 가중치를 조합하여 inference 수행

[지원 아키텍처]
- Arch0: Sequential (LR → SR → YOLO)
- Arch2: Soft Gate (LR → Gate → SR/Bypass → YOLO)
- Arch4: Adaptive 2-Pass (LR → YOLO → [조건부 SR] → YOLO)

[가정]
- SR 모델: 개별 학습 완료 (mamba_ship.pth 또는 rfdn_ship.pth)
- YOLO 모델: 개별 학습 완료 (yolo_ship.pt)
- Gate 모델 (Arch2): 별도 학습 또는 기본값 사용

사용법:
    # Arch0 inference
    python inference.py --arch arch0 --sr_type mamba \
        --sr_weights /path/to/sr.pth \
        --yolo_weights /path/to/yolo.pt \
        --input /path/to/images \
        --output /path/to/results

    # Arch2 inference (Gate 포함)
    python inference.py --arch arch2 --sr_type mamba \
        --sr_weights /path/to/sr.pth \
        --yolo_weights /path/to/yolo.pt \
        --gate_weights /path/to/gate.pth \
        --input /path/to/images

    # Arch4 inference (2-pass)
    python inference.py --arch arch4 --sr_type rfdn \
        --sr_weights /path/to/sr.pth \
        --yolo_weights /path/to/yolo.pt \
        --conf_threshold 0.3 \
        --input /path/to/images
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import time
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from tqdm import tqdm

# 프로젝트 루트 추가
sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# Inference Engines
# =============================================================================

class BaseInference:
    """Inference 기본 클래스"""
    
    def __init__(
        self,
        sr_type: str = 'mamba',
        sr_weights: Optional[str] = None,
        yolo_weights: str = 'yolov8n.pt',
        device: str = 'cuda',
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        upscale_factor: int = 4
    ):
        self.sr_type = sr_type
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.upscale_factor = upscale_factor
        
        # SR 모델 로드
        self.sr_model = self._load_sr_model(sr_type, sr_weights)
        
        # YOLO 모델 로드
        self.yolo_model = self._load_yolo_model(yolo_weights)
        
        print(f"[{self.__class__.__name__}] Initialized")
        print(f"  SR: {sr_type.upper()}")
        print(f"  Device: {self.device}")
    
    def _load_sr_model(self, sr_type: str, weights_path: Optional[str]) -> nn.Module:
        """SR 모델 로드"""
        if sr_type == 'mamba':
            from src.models.sr_models.mamba_sr import MambaSR
            model = MambaSR(scale_factor=self.upscale_factor)
            if weights_path:
                model.load_pretrained(weights_path)
        else:  # rfdn
            from src.models.sr_models.rfdn import RFDN
            model = RFDN(
                in_channels=3,
                out_channels=3,
                nf=50,
                num_modules=4,
                upscale=self.upscale_factor
            )
            if weights_path:
                state_dict = torch.load(weights_path, map_location='cpu')
                if 'model_state_dict' in state_dict:
                    state_dict = state_dict['model_state_dict']
                model.load_state_dict(state_dict, strict=False)
        
        model.to(self.device)
        model.eval()
        return model
    
    def _load_yolo_model(self, weights_path: str):
        """YOLO 모델 로드"""
        from ultralytics import YOLO
        model = YOLO(weights_path)
        return model
    
    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """이미지 전처리: numpy (H,W,C) BGR → tensor (1,C,H,W) RGB normalized"""
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
        return img.to(self.device)
    
    def postprocess_sr(self, tensor: torch.Tensor) -> np.ndarray:
        """SR 결과 후처리: tensor → numpy BGR"""
        img = tensor.squeeze(0).cpu().clamp(0, 1)
        img = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img
    
    @torch.no_grad()
    def run_sr(self, lr_tensor: torch.Tensor) -> torch.Tensor:
        """SR 실행"""
        return self.sr_model(lr_tensor)
    
    def run_yolo(self, image: np.ndarray) -> Any:
        """YOLO 실행"""
        results = self.yolo_model(
            image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False
        )
        return results[0] if results else None
    
    def inference(self, image: np.ndarray) -> Dict[str, Any]:
        """Inference 실행 (서브클래스에서 구현)"""
        raise NotImplementedError
    
    def visualize(
        self,
        image: np.ndarray,
        detections: Any,
        save_path: Optional[str] = None
    ) -> np.ndarray:
        """결과 시각화"""
        vis_img = image.copy()
        
        if detections is not None and hasattr(detections, 'boxes'):
            boxes = detections.boxes
            for i in range(len(boxes)):
                # Box
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)
                conf = boxes.conf[i].cpu().item()
                
                # Draw
                cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{conf:.2f}"
                cv2.putText(vis_img, label, (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        if save_path:
            cv2.imwrite(save_path, vis_img)
        
        return vis_img


class Arch0Inference(BaseInference):
    """
    Architecture 0: Sequential Pipeline
    
    LR → SR → YOLO
    
    가장 단순한 파이프라인. 모든 이미지에 SR 적용 후 검출.
    """
    
    def inference(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Arch0 Inference
        
        Args:
            image: LR 이미지 (BGR, numpy)
        
        Returns:
            dict with:
                - sr_image: SR 결과 이미지
                - detections: YOLO 검출 결과
                - sr_applied: True (항상)
                - inference_time: 소요 시간
        """
        start_time = time.time()
        
        # 1. SR
        lr_tensor = self.preprocess(image)
        sr_tensor = self.run_sr(lr_tensor)
        sr_image = self.postprocess_sr(sr_tensor)
        
        # 2. YOLO on SR image
        detections = self.run_yolo(sr_image)
        
        inference_time = time.time() - start_time
        
        return {
            'sr_image': sr_image,
            'detections': detections,
            'sr_applied': True,
            'inference_time': inference_time,
            'num_detections': len(detections.boxes) if detections else 0
        }


class Arch2Inference(BaseInference):
    """
    Architecture 2: Soft Gate Pipeline
    
    LR → Gate → (SR or Bypass) → YOLO
    
    Gate가 이미지 복잡도를 판단하여 SR 적용 여부 결정.
    """
    
    def __init__(
        self,
        sr_type: str = 'mamba',
        sr_weights: Optional[str] = None,
        yolo_weights: str = 'yolov8n.pt',
        gate_weights: Optional[str] = None,
        gate_threshold: float = 0.5,
        **kwargs
    ):
        super().__init__(sr_type, sr_weights, yolo_weights, **kwargs)
        
        self.gate_threshold = gate_threshold
        self.gate_model = self._load_gate_model(gate_weights)
    
    def _load_gate_model(self, weights_path: Optional[str]) -> nn.Module:
        """Gate 모델 로드"""
        from src.models.gates.soft_gate import LightweightGate
        
        gate = LightweightGate(in_channels=3)
        
        if weights_path and Path(weights_path).exists():
            state_dict = torch.load(weights_path, map_location='cpu')
            if 'gate_state_dict' in state_dict:
                state_dict = state_dict['gate_state_dict']
            gate.load_state_dict(state_dict, strict=False)
            print(f"[Arch2] Gate weights loaded: {weights_path}")
        else:
            print(f"[Arch2] Using default gate (threshold: {self.gate_threshold})")
        
        gate.to(self.device)
        gate.eval()
        return gate
    
    @torch.no_grad()
    def compute_gate_score(self, lr_tensor: torch.Tensor) -> float:
        """Gate score 계산"""
        score = self.gate_model(lr_tensor)
        return score.mean().item()
    
    def inference(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Arch2 Inference
        
        Gate score > threshold → SR 적용
        Gate score <= threshold → Bypass (원본 사용)
        """
        start_time = time.time()
        
        lr_tensor = self.preprocess(image)
        
        # 1. Gate decision
        gate_score = self.compute_gate_score(lr_tensor)
        apply_sr = gate_score > self.gate_threshold
        
        # 2. SR or Bypass
        if apply_sr:
            sr_tensor = self.run_sr(lr_tensor)
            output_image = self.postprocess_sr(sr_tensor)
        else:
            # Bypass: LR을 HR 크기로 upscale만
            h, w = image.shape[:2]
            output_image = cv2.resize(
                image, 
                (w * self.upscale_factor, h * self.upscale_factor),
                interpolation=cv2.INTER_CUBIC
            )
        
        # 3. YOLO
        detections = self.run_yolo(output_image)
        
        inference_time = time.time() - start_time
        
        return {
            'sr_image': output_image,
            'detections': detections,
            'sr_applied': apply_sr,
            'gate_score': gate_score,
            'inference_time': inference_time,
            'num_detections': len(detections.boxes) if detections else 0
        }


class Arch5BInference:
    """
    Architecture 5B: Feature-Level Fusion Pipeline
    
    LR → SR Features + YOLO Features → Fusion → Detection
    
    SR과 YOLO의 feature를 융합하여 검출 성능 향상.
    학습된 Fusion 모듈 가중치 필요.
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        device: str = 'cuda',
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45
    ):
        """
        Args:
            checkpoint_path: Arch5B 학습된 체크포인트 경로 (best.pt)
            device: 디바이스
            conf_threshold: 검출 신뢰도 임계값
            iou_threshold: NMS IoU 임계값
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # 모델 로드
        self.model = self._load_model(checkpoint_path)
        
        print(f"[Arch5BInference] Initialized")
        print(f"  Checkpoint: {checkpoint_path}")
        print(f"  Device: {self.device}")
    
    def _load_model(self, checkpoint_path: str) -> nn.Module:
        """Arch5B 모델 로드"""
        from types import SimpleNamespace
        
        # 체크포인트 로드
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        
        # args에서 설정 추출
        args = ckpt.get('args', {})
        sr_type = args.get('sr_type', 'mamba')
        
        # Config 구성
        config = SimpleNamespace(
            device=str(self.device),
            model=SimpleNamespace(
                sr_type=sr_type,
                yolo=SimpleNamespace(
                    weights_path='yolov8n.pt',
                    num_classes=1
                ),
                rfdn=SimpleNamespace(nf=50, num_modules=4),
                mamba=SimpleNamespace(
                    embed_dim=48,
                    depths=[5, 5, 5, 5],
                    pretrain_path=None
                )
            ),
            data=SimpleNamespace(upscale_factor=4),
            training=SimpleNamespace(
                sr_weight=0.3,
                det_weight=0.7,
                freeze_detector=True
            )
        )
        
        # 모델 생성
        from src.models.pipelines.arch5b_fusion import Arch5BFusion
        model = Arch5BFusion(config)
        
        # 가중치 로드
        state_dict = ckpt['model_state_dict']
        model.load_state_dict(state_dict, strict=False)
        
        model.to(self.device)
        model.eval()
        
        print(f"[Arch5BInference] Model loaded (SR: {sr_type.upper()})")
        return model
    
    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """이미지 전처리"""
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
        return img.to(self.device)
    
    def postprocess_sr(self, tensor: torch.Tensor) -> np.ndarray:
        """SR 결과 후처리"""
        img = tensor.squeeze(0).cpu().clamp(0, 1)
        img = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img
    
    @torch.no_grad()
    def inference(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Arch5B Inference
        
        Args:
            image: LR 이미지 (BGR, numpy)
        
        Returns:
            dict with:
                - sr_image: SR 결과 이미지
                - detections: 검출 결과 (boxes, scores, classes)
                - inference_time: 소요 시간
        """
        start_time = time.time()
        
        # 전처리
        lr_tensor = self.preprocess(image)
        
        # Forward (Arch5B)
        outputs = self.model(lr_tensor)
        
        # outputs 구조 파싱
        # Arch5B returns: (sr_image, detection_output)
        if isinstance(outputs, tuple):
            sr_tensor = outputs[0]
            # 중첩 tuple 처리
            while isinstance(sr_tensor, tuple):
                sr_tensor = sr_tensor[0]
            det_output = outputs[1] if len(outputs) > 1 else None
        else:
            sr_tensor = outputs
            det_output = None
        
        # SR 이미지 후처리
        sr_image = self.postprocess_sr(sr_tensor)
        
        # Detection 결과 파싱
        detections = self._parse_detections(det_output, sr_image.shape[:2])
        
        inference_time = time.time() - start_time
        
        return {
            'sr_image': sr_image,
            'detections': detections,
            'sr_applied': True,
            'inference_time': inference_time,
            'num_detections': len(detections['boxes']) if detections else 0
        }
    
    def _parse_detections(
        self, 
        det_output: Any, 
        img_size: Tuple[int, int]
    ) -> Dict[str, np.ndarray]:
        """Detection 출력 파싱"""
        if det_output is None:
            return {'boxes': np.zeros((0, 4)), 'scores': np.zeros(0), 'classes': np.zeros(0)}
        
        # det_output이 tensor인 경우 (raw YOLO output)
        if isinstance(det_output, torch.Tensor):
            # NMS 적용 필요
            boxes, scores, classes = self._apply_nms(det_output, img_size)
        else:
            # 이미 처리된 결과
            boxes = np.zeros((0, 4))
            scores = np.zeros(0)
            classes = np.zeros(0)
        
        return {
            'boxes': boxes,
            'scores': scores,
            'classes': classes
        }
    
    def _apply_nms(
        self, 
        pred: torch.Tensor, 
        img_size: Tuple[int, int]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """NMS 적용"""
        try:
            from ultralytics.utils.ops import non_max_suppression
            
            # NMS
            results = non_max_suppression(
                pred,
                conf_thres=self.conf_threshold,
                iou_thres=self.iou_threshold
            )
            
            if results and len(results[0]) > 0:
                det = results[0].cpu().numpy()
                boxes = det[:, :4]
                scores = det[:, 4]
                classes = det[:, 5]
                return boxes, scores, classes
        except:
            pass
        
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0)
    
    def visualize(
        self,
        image: np.ndarray,
        detections: Dict[str, np.ndarray],
        save_path: Optional[str] = None
    ) -> np.ndarray:
        """결과 시각화"""
        vis_img = image.copy()
        
        boxes = detections.get('boxes', np.zeros((0, 4)))
        scores = detections.get('scores', np.zeros(0))
        
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i].astype(int)
            conf = scores[i] if i < len(scores) else 0
            
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{conf:.2f}"
            cv2.putText(vis_img, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        if save_path:
            cv2.imwrite(save_path, vis_img)
        
        return vis_img


class Arch4Inference(BaseInference):
    """
    Architecture 4: Adaptive 2-Pass Pipeline
    
    Pass 1: LR → YOLO (빠른 스캔)
    Pass 2: (조건부) SR → YOLO (정밀 검출)
    
    1차 검출 결과가 불확실하면 SR 적용 후 재검출.
    """
    
    def __init__(
        self,
        sr_type: str = 'mamba',
        sr_weights: Optional[str] = None,
        yolo_weights: str = 'yolov8n.pt',
        adaptive_threshold: float = 0.5,
        min_detections: int = 0,
        **kwargs
    ):
        super().__init__(sr_type, sr_weights, yolo_weights, **kwargs)
        
        self.adaptive_threshold = adaptive_threshold
        self.min_detections = min_detections
    
    def need_sr(self, detections: Any) -> bool:
        """SR 필요 여부 판단"""
        if detections is None or not hasattr(detections, 'boxes'):
            return True  # 검출 없음 → SR 필요
        
        boxes = detections.boxes
        
        # 검출 수가 적으면 SR
        if len(boxes) <= self.min_detections:
            return True
        
        # 평균 confidence가 낮으면 SR
        if len(boxes) > 0:
            avg_conf = boxes.conf.mean().item()
            if avg_conf < self.adaptive_threshold:
                return True
        
        return False
    
    def inference(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Arch4 Inference (2-Pass)
        
        Pass 1: LR에서 직접 검출 (빠름)
        Pass 2: 필요시 SR 후 재검출 (정밀)
        """
        start_time = time.time()
        
        # Upscale LR for YOLO (YOLO는 일정 크기 필요)
        h, w = image.shape[:2]
        lr_upscaled = cv2.resize(
            image,
            (w * self.upscale_factor, h * self.upscale_factor),
            interpolation=cv2.INTER_CUBIC
        )
        
        # Pass 1: LR에서 빠른 검출
        pass1_detections = self.run_yolo(lr_upscaled)
        pass1_time = time.time() - start_time
        
        # SR 필요 여부 판단
        apply_sr = self.need_sr(pass1_detections)
        
        if apply_sr:
            # Pass 2: SR 후 정밀 검출
            lr_tensor = self.preprocess(image)
            sr_tensor = self.run_sr(lr_tensor)
            sr_image = self.postprocess_sr(sr_tensor)
            
            pass2_detections = self.run_yolo(sr_image)
            final_detections = pass2_detections
            final_image = sr_image
        else:
            # Pass 1 결과 사용
            final_detections = pass1_detections
            final_image = lr_upscaled
        
        inference_time = time.time() - start_time
        
        return {
            'sr_image': final_image,
            'detections': final_detections,
            'sr_applied': apply_sr,
            'pass1_detections': len(pass1_detections.boxes) if pass1_detections else 0,
            'pass1_time': pass1_time,
            'inference_time': inference_time,
            'num_detections': len(final_detections.boxes) if final_detections else 0
        }


# =============================================================================
# Batch Processing
# =============================================================================

def process_folder(
    inference_engine: BaseInference,
    input_path: str,
    output_path: str,
    save_images: bool = True,
    save_json: bool = True
) -> Dict[str, Any]:
    """
    폴더 내 이미지 일괄 처리
    
    Args:
        inference_engine: Inference 엔진
        input_path: 입력 폴더
        output_path: 출력 폴더
        save_images: 결과 이미지 저장 여부
        save_json: 결과 JSON 저장 여부
    
    Returns:
        통계 정보
    """
    input_dir = Path(input_path)
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 이미지 파일 찾기
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(input_dir.glob(ext))
        image_files.extend(input_dir.glob(ext.upper()))
    
    print(f"\n[Processing] {len(image_files)} images from {input_dir}")
    
    # 통계
    stats = {
        'total_images': len(image_files),
        'total_detections': 0,
        'sr_applied_count': 0,
        'total_time': 0.0,
        'results': []
    }
    
    for img_path in tqdm(image_files, desc="Processing"):
        # 이미지 로드
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        
        # Inference
        result = inference_engine.inference(image)
        
        # 통계 업데이트
        stats['total_detections'] += result['num_detections']
        stats['total_time'] += result['inference_time']
        if result['sr_applied']:
            stats['sr_applied_count'] += 1
        
        # 결과 저장
        img_name = img_path.stem
        
        if save_images:
            # SR 이미지 저장
            sr_path = output_dir / f"{img_name}_sr.jpg"
            cv2.imwrite(str(sr_path), result['sr_image'])
            
            # 시각화 저장
            vis_path = output_dir / f"{img_name}_det.jpg"
            inference_engine.visualize(result['sr_image'], result['detections'], str(vis_path))
        
        # 결과 기록
        result_info = {
            'image': img_path.name,
            'num_detections': result['num_detections'],
            'sr_applied': result['sr_applied'],
            'inference_time': result['inference_time']
        }
        
        # Arch별 추가 정보
        if 'gate_score' in result:
            result_info['gate_score'] = result['gate_score']
        if 'pass1_detections' in result:
            result_info['pass1_detections'] = result['pass1_detections']
        
        stats['results'].append(result_info)
    
    # 평균 계산
    if stats['total_images'] > 0:
        stats['avg_time'] = stats['total_time'] / stats['total_images']
        stats['avg_detections'] = stats['total_detections'] / stats['total_images']
        stats['sr_ratio'] = stats['sr_applied_count'] / stats['total_images']
    
    # JSON 저장
    if save_json:
        json_path = output_dir / 'results.json'
        with open(json_path, 'w') as f:
            json.dump(stats, f, indent=2, default=str)
        print(f"[Saved] Results JSON: {json_path}")
    
    return stats


def print_stats(stats: Dict[str, Any], arch_name: str):
    """통계 출력"""
    print("\n" + "=" * 60)
    print(f"📊 {arch_name} Inference Results")
    print("=" * 60)
    print(f"  Total images:     {stats['total_images']}")
    print(f"  Total detections: {stats['total_detections']}")
    print(f"  Avg detections:   {stats.get('avg_detections', 0):.2f}")
    print(f"  SR applied:       {stats['sr_applied_count']} ({stats.get('sr_ratio', 0)*100:.1f}%)")
    print(f"  Avg time:         {stats.get('avg_time', 0)*1000:.1f} ms")
    print(f"  Total time:       {stats['total_time']:.2f} s")
    print("=" * 60)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Arch 0/2/4/5B Inference')
    
    # Architecture
    parser.add_argument('--arch', type=str, required=True,
                        choices=['arch0', 'arch2', 'arch4', 'arch5b'],
                        help='Architecture type')
    
    # Models
    parser.add_argument('--sr_type', type=str, default='mamba',
                        choices=['rfdn', 'mamba'], help='SR model type')
    parser.add_argument('--sr_weights', type=str, default=None,
                        help='SR model weights path')
    parser.add_argument('--yolo_weights', type=str, default='yolov8n.pt',
                        help='YOLO model weights path')
    parser.add_argument('--gate_weights', type=str, default=None,
                        help='Gate model weights (Arch2 only)')
    parser.add_argument('--arch5b_checkpoint', type=str, default=None,
                        help='Arch5B checkpoint path (best.pt)')
    
    # Input/Output
    parser.add_argument('--input', type=str, required=True,
                        help='Input image or folder path')
    parser.add_argument('--output', type=str, default='./inference_results',
                        help='Output folder path')
    
    # Detection settings
    parser.add_argument('--conf_threshold', type=float, default=0.25,
                        help='YOLO confidence threshold')
    parser.add_argument('--iou_threshold', type=float, default=0.45,
                        help='YOLO IoU threshold')
    
    # Arch-specific settings
    parser.add_argument('--gate_threshold', type=float, default=0.5,
                        help='Gate threshold (Arch2)')
    parser.add_argument('--adaptive_threshold', type=float, default=0.5,
                        help='Adaptive confidence threshold (Arch4)')
    
    # Options
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda/cpu)')
    parser.add_argument('--no_save_images', action='store_true',
                        help='Do not save result images')
    parser.add_argument('--no_save_json', action='store_true',
                        help='Do not save results JSON')
    
    args = parser.parse_args()
    
    # ==========================================================================
    # Create inference engine
    # ==========================================================================
    print(f"\n[Initializing] {args.arch.upper()} + {args.sr_type.upper()}")
    
    common_kwargs = {
        'sr_type': args.sr_type,
        'sr_weights': args.sr_weights,
        'yolo_weights': args.yolo_weights,
        'device': args.device,
        'conf_threshold': args.conf_threshold,
        'iou_threshold': args.iou_threshold
    }
    
    if args.arch == 'arch0':
        engine = Arch0Inference(**common_kwargs)
    
    elif args.arch == 'arch2':
        engine = Arch2Inference(
            **common_kwargs,
            gate_weights=args.gate_weights,
            gate_threshold=args.gate_threshold
        )
    
    elif args.arch == 'arch4':
        engine = Arch4Inference(
            **common_kwargs,
            adaptive_threshold=args.adaptive_threshold
        )
    
    elif args.arch == 'arch5b':
        if not args.arch5b_checkpoint:
            print("[Error] Arch5B requires --arch5b_checkpoint (path to best.pt)")
            return
        
        engine = Arch5BInference(
            checkpoint_path=args.arch5b_checkpoint,
            device=args.device,
            conf_threshold=args.conf_threshold,
            iou_threshold=args.iou_threshold
        )
    
    # ==========================================================================
    # Process
    # ==========================================================================
    input_path = Path(args.input)
    
    if input_path.is_file():
        # 단일 이미지
        print(f"\n[Processing] Single image: {input_path}")
        
        image = cv2.imread(str(input_path))
        result = engine.inference(image)
        
        # 결과 출력
        print(f"\n[Result]")
        print(f"  Detections: {result['num_detections']}")
        print(f"  SR applied: {result['sr_applied']}")
        print(f"  Time: {result['inference_time']*1000:.1f} ms")
        
        # 저장
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if not args.no_save_images:
            sr_path = output_dir / f"{input_path.stem}_sr.jpg"
            cv2.imwrite(str(sr_path), result['sr_image'])
            
            vis_path = output_dir / f"{input_path.stem}_det.jpg"
            
            # Arch5B는 dict 형태 detections
            if args.arch == 'arch5b':
                engine.visualize(result['sr_image'], result['detections'], str(vis_path))
            else:
                engine.visualize(result['sr_image'], result['detections'], str(vis_path))
            
            print(f"  Saved: {vis_path}")
    
    elif input_path.is_dir():
        # 폴더 처리
        stats = process_folder(
            engine,
            str(input_path),
            args.output,
            save_images=not args.no_save_images,
            save_json=not args.no_save_json
        )
        
        print_stats(stats, args.arch.upper())
    
    else:
        print(f"[Error] Invalid input path: {input_path}")
        return
    
    print("\n✓ Inference completed!")


if __name__ == '__main__':
    main()