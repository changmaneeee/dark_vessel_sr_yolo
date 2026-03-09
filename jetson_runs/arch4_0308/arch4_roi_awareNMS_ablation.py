import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

try:
    from torchvision.ops import batched_nms, box_iou
except Exception:  # pragma: no cover - runtime fallback for mismatched torchvision builds
    def box_iou(box_a: torch.Tensor, box_b: torch.Tensor) -> torch.Tensor:
        if box_a.numel() == 0 or box_b.numel() == 0:
            return torch.zeros((box_a.shape[0], box_b.shape[0]), dtype=torch.float32, device=box_a.device)
        a = box_a.float()
        b = box_b.float()
        lt = torch.maximum(a[:, None, :2], b[None, :, :2])
        rb = torch.minimum(a[:, None, 2:], b[None, :, 2:])
        wh = (rb - lt).clamp(min=0)
        inter = wh[..., 0] * wh[..., 1]
        area_a = ((a[:, 2] - a[:, 0]).clamp(min=0) * (a[:, 3] - a[:, 1]).clamp(min=0))[:, None]
        area_b = ((b[:, 2] - b[:, 0]).clamp(min=0) * (b[:, 3] - b[:, 1]).clamp(min=0))[None, :]
        union = area_a + area_b - inter
        return inter / union.clamp(min=1e-6)

    def _nms_single_class(boxes: torch.Tensor, scores: torch.Tensor, iou_thresh: float) -> torch.Tensor:
        if boxes.numel() == 0:
            return torch.empty((0,), dtype=torch.long, device=boxes.device)
        order = torch.argsort(scores, descending=True)
        keep = []
        while order.numel() > 0:
            i = order[0]
            keep.append(i)
            if order.numel() == 1:
                break
            rest = order[1:]
            ious = box_iou(boxes[i:i + 1], boxes[rest]).squeeze(0)
            order = rest[ious <= float(iou_thresh)]
        return torch.stack(keep) if keep else torch.empty((0,), dtype=torch.long, device=boxes.device)

    def batched_nms(boxes: torch.Tensor, scores: torch.Tensor, classes: torch.Tensor, iou_thresh: float) -> torch.Tensor:
        if boxes.numel() == 0:
            return torch.empty((0,), dtype=torch.long, device=boxes.device)
        keep_chunks = []
        for cls in classes.unique(sorted=True):
            cls = cls.to(classes.device)
            mask = classes == cls
            idx = torch.nonzero(mask, as_tuple=False).squeeze(1)
            kept_local = _nms_single_class(boxes[idx], scores[idx], iou_thresh)
            keep_chunks.append(idx[kept_local])
        if not keep_chunks:
            return torch.empty((0,), dtype=torch.long, device=boxes.device)
        keep = torch.cat(keep_chunks, dim=0)
        keep = keep[torch.argsort(scores[keep], descending=True)]
        return keep

from src.models.pipelines.arch4_adaptive import Arch4Adaptive, Arch4Config


@dataclass
class Arch4RoiAwareNMSConfig(Arch4Config):
    # Stage-wise suppression / merge controls
    scout_nms_iou: float = 0.50
    roi_merge_iou: float = 0.30
    roi_center_ratio: float = 0.35
    sniper_nms_iou: float = 0.45
    final_nms_iou: float = 0.50

    # If Sniper finds something in a ROI, drop Scout uncertain boxes from that ROI
    drop_uncertain_if_sniper_hits: bool = True

    # Optional score bias for Sniper outputs (kept 0 by default)
    sniper_score_bonus: float = 0.0

    # Crop refinement / ablation controls
    crop_refine_mode: str = "sr"  # sr | bilinear | bicubic | nearest | hr_ref
    sniper_imgsz_mode: str = "dynamic"  # dynamic | fixed
    sniper_imgsz_fixed: Optional[int] = None


class Arch4RoiAwareNMS(Arch4Adaptive):
    """
    Arch4 + ROI-aware hierarchical NMS + crop refinement ablation hooks.

    Default behavior stays the same as the original ROI-aware version:
      - crop_refine_mode='sr'
      - sniper_imgsz_mode='dynamic'

    Added capabilities:
      1) Run identical ROI groups with different crop refinement modes:
         - 'sr'       : original SR model (RFDN / Mamba)
         - 'bilinear' : interpolation only
         - 'bicubic'  : interpolation only
         - 'nearest'  : interpolation only
         - 'hr_ref'   : use paired HR crop as an oracle reference (requires hr_images)
      2) Make sniper imgsz configurable for crop-size / detector-domain studies.
      3) Return lightweight stage statistics so crop burden can be measured directly.
    """

    VALID_CROP_MODES = {"sr", "bilinear", "bicubic", "nearest", "hr_ref"}
    VALID_IMGSZ_MODES = {"dynamic", "fixed"}

    def _parse_yaml_config(self, cfg_dict: Dict) -> Arch4RoiAwareNMSConfig:
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

        yolo_classes = get_val('model.yolo.num_classes', get_val('model.yolo.classes', 1))
        pass2_conf = get_val('model.arch4.pass2_conf', get_val('model.arch4.high_conf', 0.45))
        final_conf = get_val('model.arch4.final_conf', 0.25)
        sniper_conf = get_val('model.arch4.sniper_conf', None)

        crop_refine_mode = get_val('model.arch4.crop_refine_mode', get_val('model.arch4.crop_sr_mode', 'sr'))
        sniper_imgsz_mode = get_val('model.arch4.sniper_imgsz_mode', 'dynamic')
        sniper_imgsz_fixed = get_val('model.arch4.sniper_imgsz_fixed', None)

        return Arch4RoiAwareNMSConfig(
            upscale_factor=get_val('data.upscale_factor', 4),
            yolo_weights_hr=get_val('model.yolo.weights_hr', ''),
            yolo_weights_lr=get_val('model.yolo.weights_lr', ''),
            yolo_classes=yolo_classes,
            sr_weights=get_val('model.sr.weights', ''),
            sr_type=get_val('model.sr.type', ''),
            rfdn_nf=get_val('model.sr.rfdn.nf', 50),
            rfdn_modules=get_val('model.sr.rfdn.num_modules', 4),
            pass1_conf=get_val('model.arch4.pass1_conf', 0.1),
            pass2_conf=pass2_conf,
            final_conf=final_conf,
            sniper_conf=sniper_conf,
            merge_iou=get_val('model.arch4.merge_iou', 0.5),
            roi_expansion=get_val('model.arch4.roi_expansion', 1.5),
            batch_size_sr=get_val('model.arch4.batch_size_sr', 32),
            crop_size_lr=get_val('model.arch4.crop_size_lr', 64),
            scout_nms_iou=get_val('model.arch4.scout_nms_iou', 0.50),
            roi_merge_iou=get_val('model.arch4.roi_merge_iou', 0.30),
            roi_center_ratio=get_val('model.arch4.roi_center_ratio', 0.35),
            sniper_nms_iou=get_val('model.arch4.sniper_nms_iou', 0.45),
            final_nms_iou=get_val('model.arch4.final_nms_iou', get_val('model.arch4.merge_iou', 0.50)),
            drop_uncertain_if_sniper_hits=get_val('model.arch4.drop_uncertain_if_sniper_hits', True),
            sniper_score_bonus=get_val('model.arch4.sniper_score_bonus', 0.0),
            crop_refine_mode=crop_refine_mode,
            sniper_imgsz_mode=sniper_imgsz_mode,
            sniper_imgsz_fixed=sniper_imgsz_fixed,
        )

    def _print_info(self):
        print(f"\n[Arch4 ROI-aware Config]")
        print(f" - Scout conf(pass1_conf):       {self.cfg.pass1_conf}")
        print(f" - High conf(pass2_conf):        {self.cfg.pass2_conf}")
        print(f" - Sniper conf(sniper_conf):     {self.cfg.sniper_conf}")
        print(f" - Final conf(final_conf):       {self.cfg.final_conf}")
        print(f" - Scout NMS IoU:                {self.cfg.scout_nms_iou}")
        print(f" - ROI merge IoU:                {self.cfg.roi_merge_iou}")
        print(f" - ROI center ratio:             {self.cfg.roi_center_ratio}")
        print(f" - Sniper NMS IoU:               {self.cfg.sniper_nms_iou}")
        print(f" - Final NMS IoU:                {self.cfg.final_nms_iou}")
        print(f" - Drop uncertain if sniper hits:{self.cfg.drop_uncertain_if_sniper_hits}")
        print(f" - Crop refine mode:             {self.cfg.crop_refine_mode}")
        print(f" - Sniper imgsz mode:            {self.cfg.sniper_imgsz_mode}")
        if self.cfg.sniper_imgsz_mode == 'fixed':
            print(f" - Sniper imgsz fixed:           {self.cfg.sniper_imgsz_fixed}")
        print(f" - Batch Strategy:               {self.cfg.batch_size_sr} crops per SR pass")

    @torch.no_grad()
    def forward(
        self,
        lr_images: torch.Tensor,
        hr_images: Optional[torch.Tensor] = None,
        debug: bool = False,
    ) -> Dict[str, Any]:
        batch_size, _, height, width = lr_images.shape
        self.scout_detector.eval()
        self.sr_model.eval()
        self.sniper_detector.eval()

        crop_mode = self._normalize_crop_refine_mode(self.cfg.crop_refine_mode)
        imgsz_mode = self._normalize_sniper_imgsz_mode(self.cfg.sniper_imgsz_mode)

        if crop_mode == 'hr_ref':
            if hr_images is None:
                raise ValueError("crop_refine_mode='hr_ref' requires hr_images to be passed into forward().")
            if hr_images.shape[0] != batch_size:
                raise ValueError(
                    f"hr_images batch size mismatch: expected {batch_size}, got {hr_images.shape[0]}"
                )
            if hr_images.device != lr_images.device:
                hr_images = hr_images.to(lr_images.device)

        debug_info = {
            'pass1_raw': [],
            'pass1_after_nms': [],
            'roi_groups': [],
            'crops_lr': [],
            'crops_refined': [],
            'crops_sr': [],  # kept for backward compatibility with old debug tooling
            'crop_meta': [],
            'pass2_raw': [],
            'pass2_after_nms': [],
        }

        stats: Dict[str, Any] = {
            'num_images': int(batch_size),
            'crop_refine_mode': crop_mode,
            'sniper_imgsz_mode': imgsz_mode,
            'sniper_imgsz_fixed': int(self.cfg.sniper_imgsz_fixed) if self.cfg.sniper_imgsz_fixed else None,
            'scout_boxes_total': 0,
            'confident_boxes_total': 0,
            'uncertain_boxes_total': 0,
            'roi_groups_total': 0,
            'roi_crops_total': 0,
            'sniper_hit_groups_total': 0,
            'fallback_groups_total': 0,
            'crop_invalid_groups_total': 0,
            'per_image': [],
        }

        pass1_preds = self.scout_detector.predict(
            lr_images,
            conf=self.cfg.pass1_conf,
            iou=self.cfg.scout_nms_iou,
        )
        if debug:
            debug_info['pass1_raw'] = pass1_preds

        final_results: List[Dict[str, List[torch.Tensor]]] = []
        all_crops_lr: List[torch.Tensor] = []
        crop_metadata: List[Dict[str, Any]] = []

        for b_idx, det in enumerate(pass1_preds):
            det = self._apply_batched_nms(det, self.cfg.scout_nms_iou)
            if debug:
                debug_info['pass1_after_nms'].append(det)

            boxes = det['boxes']
            scores = det['scores']
            classes = det['classes']

            confident_mask = scores >= self.cfg.pass2_conf
            confident_boxes = boxes[confident_mask]
            confident_scores = scores[confident_mask]
            confident_classes = classes[confident_mask]

            uncertain_boxes = boxes[~confident_mask]
            uncertain_scores = scores[~confident_mask]
            uncertain_classes = classes[~confident_mask]

            stats['scout_boxes_total'] += int(len(scores))
            stats['confident_boxes_total'] += int(len(confident_boxes))
            stats['uncertain_boxes_total'] += int(len(uncertain_boxes))

            final_results.append({
                'boxes': [confident_boxes],
                'scores': [confident_scores],
                'classes': [confident_classes],
            })

            roi_groups = self._build_roi_groups(
                uncertain_boxes,
                uncertain_scores,
                uncertain_classes,
            )
            stats['roi_groups_total'] += int(len(roi_groups))
            stats['per_image'].append({
                'image_index': int(b_idx),
                'scout_boxes': int(len(scores)),
                'confident_boxes': int(len(confident_boxes)),
                'uncertain_boxes': int(len(uncertain_boxes)),
                'roi_groups': int(len(roi_groups)),
            })

            if debug:
                debug_info['roi_groups'].append(roi_groups)

            for group in roi_groups:
                merged_box = group['merged_box'].unsqueeze(0)
                crops, coords = self._extract_crops(lr_images[b_idx], merged_box)
                group['fallback_boxes'] = group['member_boxes'][group['member_scores'] >= self.cfg.final_conf]
                group['fallback_scores'] = group['member_scores'][group['member_scores'] >= self.cfg.final_conf]
                group['fallback_classes'] = group['member_classes'][group['member_scores'] >= self.cfg.final_conf]

                if len(crops) == 0:
                    group['crop_valid'] = False
                    group['coord'] = None
                    stats['fallback_groups_total'] += 1
                    stats['crop_invalid_groups_total'] += 1
                    self._append_group_fallback(final_results[b_idx], group)
                    continue

                group['crop_valid'] = True
                group['coord'] = coords[0]
                all_crops_lr.append(crops[0])
                crop_metadata.append({
                    'img_idx': b_idx,
                    'group': group,
                })

        stats['roi_crops_total'] = int(len(all_crops_lr))

        if len(all_crops_lr) > 0:
            batch_crops_lr = torch.stack(all_crops_lr).to(self.cfg.device)
            batch_crops_refined = self._prepare_sniper_inputs(
                batch_crops_lr=batch_crops_lr,
                crop_metadata=crop_metadata,
                hr_images=hr_images,
            )

            hr_input_size = int(batch_crops_refined.shape[-1])
            for meta in crop_metadata:
                meta['sniper_input_size'] = hr_input_size

            if debug:
                debug_info['crops_lr'] = [c.cpu() for c in all_crops_lr]
                debug_info['crops_refined'] = [c.cpu() for c in batch_crops_refined]
                debug_info['crops_sr'] = [c.cpu() for c in batch_crops_refined]
                debug_info['crop_meta'] = crop_metadata

            sniper_imgsz = self._choose_sniper_imgsz(hr_input_size)
            sniper_results = self.sniper_detector.predict(
                batch_crops_refined,
                conf=float(self.cfg.sniper_conf),
                iou=self.cfg.sniper_nms_iou,
                imgsz=sniper_imgsz,
            )
            if debug:
                debug_info['pass2_raw'] = sniper_results

            for i, res in enumerate(sniper_results):
                res = self._apply_batched_nms(res, self.cfg.sniper_nms_iou)
                if debug:
                    debug_info['pass2_after_nms'].append(res)

                meta = crop_metadata[i]
                img_idx = meta['img_idx']
                group = meta['group']

                if len(res['boxes']) == 0:
                    stats['fallback_groups_total'] += 1
                    self._append_group_fallback(final_results[img_idx], group)
                    continue

                keep = res['scores'] >= float(self.cfg.final_conf)
                if keep.sum().item() == 0:
                    stats['fallback_groups_total'] += 1
                    self._append_group_fallback(final_results[img_idx], group)
                    continue

                res_boxes = res['boxes'][keep].clone().float()
                res_scores = res['scores'][keep].clone()
                res_classes = res['classes'][keep].clone()

                if self.cfg.sniper_score_bonus != 0.0:
                    res_scores = torch.clamp(res_scores + float(self.cfg.sniper_score_bonus), 0.0, 1.0)

                global_boxes = self._sniper_boxes_to_global(
                    res_boxes,
                    group['coord'],
                    width=width,
                    height=height,
                    hr_input_size=float(meta.get('sniper_input_size', hr_input_size)),
                )

                stats['sniper_hit_groups_total'] += 1

                if self.cfg.drop_uncertain_if_sniper_hits:
                    final_results[img_idx]['boxes'].append(global_boxes)
                    final_results[img_idx]['scores'].append(res_scores)
                    final_results[img_idx]['classes'].append(res_classes)
                else:
                    self._append_group_fallback(final_results[img_idx], group)
                    final_results[img_idx]['boxes'].append(global_boxes)
                    final_results[img_idx]['scores'].append(res_scores)
                    final_results[img_idx]['classes'].append(res_classes)

        output_detections: List[Dict[str, torch.Tensor]] = []
        for res in final_results:
            if len(res['boxes']) == 0:
                output_detections.append({
                    'boxes': torch.empty((0, 4), device=self.cfg.device),
                    'scores': torch.empty((0,), device=self.cfg.device),
                    'classes': torch.empty((0,), device=self.cfg.device),
                })
                continue

            all_boxes = torch.cat(res['boxes'], dim=0) if len(res['boxes']) > 0 else torch.empty((0, 4), device=self.cfg.device)
            all_scores = torch.cat(res['scores'], dim=0) if len(res['scores']) > 0 else torch.empty((0,), device=self.cfg.device)
            all_classes = torch.cat(res['classes'], dim=0) if len(res['classes']) > 0 else torch.empty((0,), device=self.cfg.device)

            if all_boxes.numel() == 0:
                output_detections.append({
                    'boxes': all_boxes,
                    'scores': all_scores,
                    'classes': all_classes,
                })
                continue

            keep = batched_nms(all_boxes, all_scores, all_classes.long(), self.cfg.final_nms_iou)
            output_detections.append({
                'boxes': all_boxes[keep],
                'scores': all_scores[keep],
                'classes': all_classes[keep],
            })

        stats['avg_scout_boxes_per_image'] = stats['scout_boxes_total'] / max(1, batch_size)
        stats['avg_roi_groups_per_image'] = stats['roi_groups_total'] / max(1, batch_size)
        stats['avg_roi_crops_per_image'] = stats['roi_crops_total'] / max(1, batch_size)
        stats['sniper_hit_rate_over_groups'] = (
            stats['sniper_hit_groups_total'] / max(1, stats['roi_groups_total'])
        )

        result = {'detections': output_detections, 'stats': stats}
        if debug:
            result['debug_info'] = debug_info
        return result

    def _append_group_fallback(self, result_dict: Dict[str, List[torch.Tensor]], group: Dict[str, Any]):
        fb_boxes = group.get('fallback_boxes')
        fb_scores = group.get('fallback_scores')
        fb_classes = group.get('fallback_classes')
        if fb_boxes is None or fb_boxes.numel() == 0:
            return
        result_dict['boxes'].append(fb_boxes)
        result_dict['scores'].append(fb_scores)
        result_dict['classes'].append(fb_classes)

    def _apply_batched_nms(self, det: Dict[str, torch.Tensor], iou_thresh: float) -> Dict[str, torch.Tensor]:
        boxes = det['boxes']
        scores = det['scores']
        classes = det['classes']
        if boxes.numel() == 0:
            return det
        keep = batched_nms(boxes, scores, classes.long(), iou_thresh)
        return {
            'boxes': boxes[keep],
            'scores': scores[keep],
            'classes': classes[keep],
        }

    def _build_roi_groups(
        self,
        boxes: torch.Tensor,
        scores: torch.Tensor,
        classes: torch.Tensor,
    ) -> List[Dict[str, Any]]:
        """Merge overlapping / nearby uncertain Scout boxes into ROI groups."""
        groups: List[Dict[str, Any]] = []
        if boxes.numel() == 0:
            return groups

        order = torch.argsort(scores, descending=True)
        boxes = boxes[order]
        scores = scores[order]
        classes = classes[order]
        used = torch.zeros((boxes.shape[0],), dtype=torch.bool, device=boxes.device)

        for i in range(boxes.shape[0]):
            if used[i]:
                continue
            used[i] = True
            members = [i]

            for j in range(i + 1, boxes.shape[0]):
                if used[j]:
                    continue
                if self._same_roi_group(boxes[i], boxes[j]):
                    used[j] = True
                    members.append(j)

            member_idx = torch.tensor(members, device=boxes.device, dtype=torch.long)
            member_boxes = boxes[member_idx]
            member_scores = scores[member_idx]
            member_classes = classes[member_idx]

            merged_box = torch.tensor([
                member_boxes[:, 0].min(),
                member_boxes[:, 1].min(),
                member_boxes[:, 2].max(),
                member_boxes[:, 3].max(),
            ], device=boxes.device, dtype=boxes.dtype)

            groups.append({
                'member_boxes': member_boxes,
                'member_scores': member_scores,
                'member_classes': member_classes,
                'merged_box': merged_box,
            })

        return groups

    def _same_roi_group(self, box_a: torch.Tensor, box_b: torch.Tensor) -> bool:
        iou = box_iou(box_a.unsqueeze(0), box_b.unsqueeze(0)).item()
        if iou >= float(self.cfg.roi_merge_iou):
            return True

        ax1, ay1, ax2, ay2 = box_a.tolist()
        bx1, by1, bx2, by2 = box_b.tolist()
        acx = 0.5 * (ax1 + ax2)
        acy = 0.5 * (ay1 + ay2)
        bcx = 0.5 * (bx1 + bx2)
        bcy = 0.5 * (by1 + by2)
        dx = acx - bcx
        dy = acy - bcy
        center_dist = (dx * dx + dy * dy) ** 0.5

        aw = max(ax2 - ax1, 1.0)
        ah = max(ay2 - ay1, 1.0)
        bw = max(bx2 - bx1, 1.0)
        bh = max(by2 - by1, 1.0)
        ref = max(max(aw, ah), max(bw, bh))
        return center_dist <= float(self.cfg.roi_center_ratio) * ref

    def _prepare_sniper_inputs(
        self,
        batch_crops_lr: torch.Tensor,
        crop_metadata: List[Dict[str, Any]],
        hr_images: Optional[torch.Tensor],
    ) -> torch.Tensor:
        mode = self._normalize_crop_refine_mode(self.cfg.crop_refine_mode)
        target_size = int(self.cfg.crop_size_lr * self.cfg.upscale_factor)

        if mode == 'sr':
            return self._run_batch_sr(batch_crops_lr)

        if mode in {'bilinear', 'bicubic', 'nearest'}:
            return self._upsample_crop_batch(batch_crops_lr, target_size=target_size, mode=mode)

        if mode == 'hr_ref':
            return self._extract_hr_reference_crops(
                batch_crops_lr=batch_crops_lr,
                hr_images=hr_images,
                crop_metadata=crop_metadata,
                target_size=target_size,
            )

        raise ValueError(
            f"Unsupported crop_refine_mode={self.cfg.crop_refine_mode!r}. "
            f"Valid modes: {sorted(self.VALID_CROP_MODES)}"
        )

    def _upsample_crop_batch(self, batch_crops_lr: torch.Tensor, target_size: int, mode: str) -> torch.Tensor:
        if mode == 'nearest':
            return F.interpolate(batch_crops_lr, size=(target_size, target_size), mode=mode)
        return F.interpolate(
            batch_crops_lr,
            size=(target_size, target_size),
            mode=mode,
            align_corners=False,
        )

    def _extract_hr_reference_crops(
        self,
        batch_crops_lr: torch.Tensor,
        hr_images: Optional[torch.Tensor],
        crop_metadata: List[Dict[str, Any]],
        target_size: int,
    ) -> torch.Tensor:
        if hr_images is None:
            raise ValueError("hr_images must be provided for crop_refine_mode='hr_ref'.")

        scale = float(self.cfg.upscale_factor)
        outputs: List[torch.Tensor] = []

        for idx, meta in enumerate(crop_metadata):
            img_idx = int(meta['img_idx'])
            coord = meta['group'].get('coord')
            if coord is None:
                raise ValueError("ROI group is missing 'coord' while preparing hr_ref crops.")

            ix1, iy1, ix2, iy2 = coord
            hr_img = hr_images[img_idx:img_idx + 1]
            _, _, hr_h, hr_w = hr_img.shape

            hx1 = max(0, min(hr_w, int(round(ix1 * scale))))
            hy1 = max(0, min(hr_h, int(round(iy1 * scale))))
            hx2 = max(0, min(hr_w, int(round(ix2 * scale))))
            hy2 = max(0, min(hr_h, int(round(iy2 * scale))))

            if hx2 <= hx1 or hy2 <= hy1:
                fallback = self._upsample_crop_batch(batch_crops_lr[idx:idx + 1], target_size=target_size, mode='bilinear')
                outputs.append(fallback[0])
                continue

            crop = hr_img[:, :, hy1:hy2, hx1:hx2]
            if crop.numel() == 0:
                fallback = self._upsample_crop_batch(batch_crops_lr[idx:idx + 1], target_size=target_size, mode='bilinear')
                outputs.append(fallback[0])
                continue

            crop_resized = F.interpolate(
                crop,
                size=(target_size, target_size),
                mode='bilinear',
                align_corners=False,
            )
            outputs.append(crop_resized[0])

        if not outputs:
            return torch.empty((0, batch_crops_lr.shape[1], target_size, target_size), device=batch_crops_lr.device)
        return torch.stack(outputs, dim=0)

    def _choose_sniper_imgsz(self, input_size: int) -> int:
        mode = self._normalize_sniper_imgsz_mode(self.cfg.sniper_imgsz_mode)
        if mode == 'fixed':
            return int(self.cfg.sniper_imgsz_fixed or input_size)
        return int(input_size)

    def _normalize_crop_refine_mode(self, mode: Any) -> str:
        norm = str(mode).strip().lower()
        aliases = {
            'interp': 'bilinear',
            'oracle': 'hr_ref',
            'gt': 'hr_ref',
            'hr': 'hr_ref',
        }
        norm = aliases.get(norm, norm)
        if norm not in self.VALID_CROP_MODES:
            raise ValueError(f"Invalid crop_refine_mode={mode!r}. Valid: {sorted(self.VALID_CROP_MODES)}")
        return norm

    def _normalize_sniper_imgsz_mode(self, mode: Any) -> str:
        norm = str(mode).strip().lower()
        if norm not in self.VALID_IMGSZ_MODES:
            raise ValueError(f"Invalid sniper_imgsz_mode={mode!r}. Valid: {sorted(self.VALID_IMGSZ_MODES)}")
        return norm

    def _sniper_boxes_to_global(
        self,
        res_boxes: torch.Tensor,
        coord: Tuple[int, int, int, int],
        width: int,
        height: int,
        hr_input_size: float,
    ) -> torch.Tensor:
        ix1, iy1, ix2, iy2 = coord
        crop_w = max(1, ix2 - ix1)
        crop_h = max(1, iy2 - iy1)
        lr_size = float(self.cfg.crop_size_lr)
        hr_input_size = max(1.0, float(hr_input_size))

        boxes_lr_resized = res_boxes.clone()
        boxes_lr_resized[:, [0, 2]] *= (lr_size / hr_input_size)
        boxes_lr_resized[:, [1, 3]] *= (lr_size / hr_input_size)

        boxes_lr = boxes_lr_resized.clone()
        boxes_lr[:, [0, 2]] *= (crop_w / lr_size)
        boxes_lr[:, [1, 3]] *= (crop_h / lr_size)

        global_boxes = boxes_lr.clone()
        global_boxes[:, [0, 2]] += ix1
        global_boxes[:, [1, 3]] += iy1
        global_boxes[:, [0, 2]] = global_boxes[:, [0, 2]].clamp(0, width - 1)
        global_boxes[:, [1, 3]] = global_boxes[:, [1, 3]].clamp(0, height - 1)
        return global_boxes
