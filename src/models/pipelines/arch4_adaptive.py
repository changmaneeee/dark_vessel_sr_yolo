import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Union,  Dict, Any
from pathlib import Path
from types import SimpleNamespace
from src.models.sr_models.mamba_sr import MambaSR
from src.models.detectors.yolo_wrapper import YOLOWrapper
from src.models.pipelines.base_pipeline import BasePipeline
from src.models.sr_models.rfdn import RFDN
from torchvision.ops import nms

# =============================================================================
# 1. Configuration Data Class
# =============================================================================

@dataclass
class Arch4Config:

    # system
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Dual YOLO Paths
    yolo_weights_hr: str = ""
    yolo_weights_lr: str = ""
    yolo_classes: int = 1  # Number of classes for YOLO models

    #SR Model Path
    sr_weights: str = ""
    sr_type: str = ""
    upscale_factor: int = 4

    # RFDN Specific
    rfdn_nf: int = 50
    rfdn_modules: int = 4

    # Adaptive strategy Thresholds
    pass1_conf: float = 0.1
    pass2_conf: float = 0.45
    final_conf: float = 0.25
    merge_iou: float = 0.5

    #Crop Setting
    roi_expansion: float = 1.5
    crop_size_lr: int = 32
    batch_size_sr: int= 32

    def __post_init__(self):
        if not self.yolo_weights_hr:
            raise ValueError("yolo_weights_hr must be specified")
        if not self.yolo_weights_lr:
            raise ValueError("yolo_weights_lr must be specified")
        

# =============================================================================
# 2. Arch4 Adaptive Pipeline
# =============================================================================

class Arch4Adaptive(BasePipeline):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.cfg = self._parse_yaml_config(config)

        self._init_models()
        self._print_info()


        print(f"\n[Arch4] Initialized with Architecture 4: Confidence-Adaptive")
        print(f"  - Scout (LR): {Path(self.cfg.yolo_weights_lr).name}")
        print(f"  - Sniper (HR): {Path(self.cfg.yolo_weights_hr).name}")
        print(f"  - SR Model: {self.cfg.sr_type.upper()} (x{self.cfg.upscale_factor})")
        print(f"  - Thresholds: Pass1={self.cfg.pass1_conf} | SkipSR={self.cfg.pass2_conf} | Final={self.cfg.final_conf}")

    def _parse_yaml_config(self, cfg_dict: Dict) -> Arch4Config:
        
        def get_val(path, default=None):
            keys = path.split('.')
            curr = cfg_dict
            for k in keys:
                if isinstance(curr, dict) and k in curr:
                    curr = curr[k]
                elif hasattr(curr, k):
                    curr = getattr(curr, k)
                else:
                    return default
            return curr
        
        return Arch4Config(

            upscale_factor=get_val('data.upscale_factor',4),

            yolo_weights_hr = get_val('model.yolo.weights_hr',''),
            yolo_weights_lr = get_val('model.yolo.weights_lr',''),
            yolo_classes = get_val('model.yolo.num_classes',1),

            sr_weights = get_val('model.sr.weights',''),
            sr_type= get_val('model.sr.type',''),

            #RFDN
            rfdn_nf = get_val('model.sr.rfdn.nf', 50),
            rfdn_modules = get_val('model.sr.rfdn.num_modules', 4),

            pass1_conf = get_val('model.arch4.pass1_conf', 0.1),
            pass2_conf = get_val('model.arch4.pass2_conf', 0.45),
            final_conf = get_val('model.arch4.final_conf', 0.25),
            merge_iou = get_val('model.arch4.merge_iou', 0.5),

            #crop
            roi_expansion = get_val('model.arch4.roi_expansion', 1.5),
            batch_size_sr = get_val('model.arch4.batch_size_sr', 32)

        )
    
    def _init_models(self):
        print(f"\n[Arch4] Loading Models...")

        # 1. SR MODEL
        if self.cfg.sr_type == 'mamba':
            if MambaSR is None:
                raise ImportError("MambaSR module not found!")
            print(f" > Loading SR: MambaSR (x{self.cfg.upscale_factor})")
            self.sr_model = MambaSR(
                scale_factor=self.cfg.upscale_factor,
                img_size=192,
                embed_dim=48,
                d_state=8)
            
        else:
            print(f"  > Loading SR: RFDN (x{self.cfg.upscale_factor}, nf={self.cfg.rfdn_nf}")
            self.sr_model = RFDN(
                in_channels= 3,
                out_channels=3, 
                nf=self.cfg.rfdn_nf,
                num_modules=self.cfg.rfdn_modules,
                upscale=self.cfg.upscale_factor,
                input_range='0-255' ##수정
            )

        # SR Weights 
        if self.cfg.sr_weights and Path(self.cfg.sr_weights).exists():
            ckpt = torch.load(self.cfg.sr_weights, map_location='cpu')
            state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
            try:
                self.sr_model.load_state_dict(state_dict, strict=True)
                print(f"   ✓ SR Weights loaded: {Path(self.cfg.sr_weights).name}")
            except Exception as e:
                print(f"   ! Warning: SR Weights loading issue: {e}")
                self.sr_model.load_state_dict(state_dict, strict=False)
        else:
            print(f" SR Weights not found! Initialized randomly.")

        self.sr_model.to(self.cfg.device).eval()

        # 2. Dual YOLO MODELS
        # LR YOLO
        print(f" > Loading Scout YOLO (LR)...{Path(self.cfg.yolo_weights_lr).name}")
        self.scout_detector = YOLOWrapper(
            model_path = self.cfg.yolo_weights_lr,
            num_classes = self.cfg.yolo_classes,
            device = self.cfg.device, verbose=False
        )
        self.scout_detector.eval()

        # HR YOLO
        print(f" > Loading Sniper YOLO (HR)...{Path(self.cfg.yolo_weights_hr).name}")
        self.sniper_detector = YOLOWrapper(
            model_path = self.cfg.yolo_weights_hr,
            num_classes = self.cfg.yolo_classes,
            device = self.cfg.device, verbose=False
        )
        self.sniper_detector.eval()

    def _print_info(self):
        print(f"\n[Arch4 Config]")
        print(f" - Scout YOLO Weights (LR): {self.cfg.pass1_conf}")
        print(f" - Sniper YOLO Weights (HR): {self.cfg.pass2_conf}")
        print(f" - Batch Strategy: {self.cfg.batch_size_sr} crops per SR pass")

# =============================================================================
# 3. Forward Logic
# =============================================================================

# =========================================================================
    # 3. Forward Logic (With Debug Mode & Safety Checks)
    # =========================================================================
    @torch.no_grad()
    def forward(self, lr_images: torch.Tensor, debug: bool = False) -> Dict[str, Any]:
        """
        [Process]
        1. Scout: LR 이미지 정찰
        2. Filter: A급(확실) / B급(애매) 분류
        3. Batch SR: B급만 모아서 SR & Sniper 수행
        4. Merge: 결과 합치기
        """
        batch_size, _, height, width = lr_images.shape
        self.scout_detector.eval()
        self.sr_model.eval()
        self.sniper_detector.eval()

        # [Debug Storage] 디버깅 데이터 저장소 초기화
        debug_info = {
            'pass1_raw': [],      # Scout 원본 결과
            'crops_lr': [],       # 잘라낸 LR 이미지들
            'crops_sr': [],       # SR 복원된 HR 이미지들
            'crop_meta': [],      # Crop 좌표 정보
            'pass2_raw': []       # Sniper 원본 결과
        }

        # --- Phase 1: Scout (정찰) ---
        pass1_preds = self.scout_detector.predict(
            lr_images, 
            conf=self.cfg.pass1_conf, 
            iou=self.cfg.merge_iou
        )
        
        # 디버그: 1차 결과 저장
        if debug:
            debug_info['pass1_raw'] = pass1_preds

        final_results = []
        all_crops_lr = []
        crop_metadata = []

        # --- Phase 2: Filter (분류) ---
        for b_idx, det in enumerate(pass1_preds):
            boxes, scores, classes = det['boxes'], det['scores'], det['classes']


            if debug and len(scores) > 0:
                print(f"\n[Img {b_idx}] Scout 결과: {len(scores)}개 발견 (Max Conf: {scores.max():.4f})")
                print(f"  - Thresholds: Pass1={self.cfg.pass1_conf}, High={self.cfg.pass2_conf}")

            # A급(확실) vs B급(애매) 분류
            confident_mask = scores >= self.cfg.pass2_conf
            confident_boxes = boxes[confident_mask]
            confident_scores = scores[confident_mask]
            confident_classes = classes[confident_mask]
            
            uncertain_mask = (~confident_mask) 
            uncertain_boxes = boxes[uncertain_mask]

            if debug:
                print(f"  -> A급(확실): {len(confident_boxes)}개")
                print(f"  -> B급(애매): {len(uncertain_boxes)}개 (SR 대상)")


            final_results.append({
                'boxes': [confident_boxes],
                'scores': [confident_scores],
                'classes': [confident_classes]
            })

            if len(uncertain_boxes) == 0:
                continue

            # B급은 Crop 수행
            # (아까 수정한 Safety Guard가 있는 _extract_crops가 호출됨)
            crops, coords = self._extract_crops(lr_images[b_idx], uncertain_boxes)
            
            for crop, coord in zip(crops, coords):
                all_crops_lr.append(crop)
                crop_metadata.append((b_idx, coord))

        # --- Phase 3: Batch SR & Sniper ---
        if len(all_crops_lr) > 0:
            # 1. Stack
            batch_crops_lr = torch.stack(all_crops_lr).to(self.cfg.device)
            
            # 2. SR 수행
            batch_crops_hr = self._run_batch_sr(batch_crops_lr)

            # 디버그: Crop 이미지들 저장 (CPU로 이동)
            if debug:
                debug_info['crops_lr'] = [c.cpu() for c in all_crops_lr]
                debug_info['crops_sr'] = [c.cpu() for c in batch_crops_hr]
                debug_info['crop_meta'] = crop_metadata

            # 3. Sniper 수행
            sniper_results = self.sniper_detector.predict(
                batch_crops_hr, 
                conf=self.cfg.final_conf, 
                iou=self.cfg.merge_iou
            )
            
            if debug:
                debug_info['pass2_raw'] = sniper_results

            # --- Phase 4: Assembly ---
            for i, res in enumerate(sniper_results):
                if len(res['boxes']) == 0: continue
                
                meta = crop_metadata[i]
                img_idx = meta[0]
                crop_x1, crop_y1, _, _ = meta[1]
                scale = self.cfg.upscale_factor
                
                # 좌표 복원
                global_boxes = res['boxes'].clone()
                global_boxes[:, [0, 2]] = (global_boxes[:, [0, 2]] / scale) + crop_x1
                global_boxes[:, [1, 3]] = (global_boxes[:, [1, 3]] / scale) + crop_y1

                # 결과 합류
                final_results[img_idx]['boxes'].append(global_boxes)
                final_results[img_idx]['scores'].append(res['scores'])
                final_results[img_idx]['classes'].append(res['classes'])

        # --- Final Merge (NMS) ---
        output_detections = []
        for res in final_results:
            if len(res['boxes']) == 0:
                output_detections.append({
                    'boxes': torch.empty(0,4).to(self.cfg.device), 
                    'scores': torch.empty(0).to(self.cfg.device), 
                    'classes': torch.empty(0).to(self.cfg.device)
                })
                continue
            
            all_boxes = torch.cat(res['boxes'], dim=0)
            all_scores = torch.cat(res['scores'], dim=0)
            all_classes = torch.cat(res['classes'], dim=0)
            
            if len(all_boxes) > 0:
                keep = nms(all_boxes, all_scores, self.cfg.merge_iou)
                output_detections.append({
                    'boxes': all_boxes[keep], 
                    'scores': all_scores[keep], 
                    'classes': all_classes[keep]
                })
            else:
                output_detections.append({
                    'boxes': all_boxes, 'scores': all_scores, 'classes': all_classes
                })

        # ★★★ 여기가 핵심입니다! ★★★
        # debug=True일 때만 딕셔너리에 'debug_info'를 넣어서 반환합니다.
        if debug:
            return {'detections': output_detections, 'debug_info': debug_info}
        else:
            return {'detections': output_detections}
            
    def compute_loss(self, outputs, targets, hr_gt=None):

        dummy_loss = torch.tensor(0.0, device=self.cfg.device, requires_grad=True)
        return {
            'total' : dummy_loss,
            'box_loss' : dummy_loss,
            'cls_loss' : dummy_loss,
            'dfl_loss' : dummy_loss,
            'sr_loss' : dummy_loss
        }
    
    #=======================================================================
    # Helpers Methods
    #=======================================================================
    def _extract_crops(self, image: torch.Tensor, boxes: torch.Tensor):
        crops = []
        coords = []
        _, img_h, img_w  = image.shape

        expansion = self.cfg.roi_expansion

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box.tolist()
            box_w, box_h = x2 - x1, y2 - y1

            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            nw, nh = box_w * expansion, box_h * expansion

            size = max(nw, nh, self.cfg.crop_size_lr)

            nx1 = cx - size / 2
            ny1 = cy - size / 2
            nx2 = cx + size / 2
            ny2 = cy + size / 2

            # [Safety Guard 1] 정수 변환 및 이미지 범위 안으로 강제 고정 (Clamping)
            ix1 = max(0, int(round(nx1)))
            iy1 = max(0, int(round(ny1)))
            ix2 = min(img_w, int(round(nx2)))
            iy2 = min(img_h, int(round(ny2)))

            # [Safety Guard 2] 유효성 검사 (가로/세로가 0 이하면 스킵)
            if ix2 <= ix1 or iy2 <= iy1:
                # 너무 구석이라 자를 게 없거나 에러인 경우 무시
                print(f"      ⚠️ [Skip] 너비/높이가 0입니다. (W={ix2-ix1}, H={iy2-iy1})")
                continue

            try:
                crop = image[:, iy1:iy2, ix1:ix2]
            
            except TypeError as e:
                print(f"      ⚠️ [Skip] Crop 추출 중 오류 발생: {e}")
                print(f"      ❌ [Error] 좌표 타입 오류: {type(iy1)}")
                continue

            if crop.numel() == 0:
                print(f"      ⚠️ [Skip] 텐서가 비었습니다.")
                continue
            
            try:
                crop_resized = F.interpolate(
                    crop.unsqueeze(0),
                    size = (self.cfg.crop_size_lr, self.cfg.crop_size_lr),
                    mode = 'bilinear', align_corners=False
                )[0]

                crops.append(crop_resized)
                coords.append((ix1, iy1, ix2, iy2))
            
            except Exception as e:
                print(f"      ⚠️ [Skip] Crop 처리 중 오류 발생: {e}")
                continue

        return crops, coords
    
    def _run_batch_sr(self, batch_lr: torch.Tensor) -> torch.Tensor:

        _, _,h, w = batch_lr.shape
        pad_h, pad_w = 0, 0

        if h < 32: pad_h = 32 - h
        if w < 32: pad_w = 32 - w

        if pad_h > 0 or pad_w > 0:
            batch_lr = F.pad(batch_lr, (0, pad_w, 0, pad_h), mode='replicate')

        batch_lr_255 = batch_lr * 255.0

        outputs=[]

        for i in range(0, len(batch_lr_255), self.cfg.batch_size_sr):
            
            chunk = batch_lr_255[i : i + self.cfg.batch_size_sr]

            with torch.no_grad():
                sr_chunk = self.sr_model(chunk)

            outputs.append(sr_chunk)
        full_sr = torch.cat(outputs, dim=0)

        if pad_h > 0 or pad_w > 0:
            scale = self.cfg.upscale_factor
            valid_h = h*scale
            valid_w = w*scale
            full_sr = full_sr[:, :, :valid_h, :valid_w]

        #full_sr = full_sr/255.0

        full_sr = torch.clamp(full_sr / 255.0, 0.0, 1.0)
        return full_sr


