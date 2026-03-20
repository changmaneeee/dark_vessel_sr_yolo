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

    # Merge policy controls
    merge_policy: str = "binary"  # binary | score_cond | size_cond
    sniper_replace_margin: float = 0.10
    roi_small_thresh: float = 32.0
    roi_large_thresh: float = 96.0
    large_roi_score_thresh: float = 0.50
    sniper_max_det_per_crop: int = 0

    # Crop refinement / ablation controls
    crop_refine_mode: str = "sr"  # sr | bilinear | bicubic | nearest | hr_ref
    sniper_imgsz_mode: str = "dynamic"  # dynamic | fixed
    sniper_imgsz_fixed: Optional[int] = None

    # Verifier controls
    use_verifier: bool = False
    verifier_mode: str = "iou"  # iou | topk
    verifier_iou_thresh: float = 0.10
    verifier_topk_cap: int = 3


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
    VALID_MERGE_POLICIES = {"binary", "score_cond", "size_cond"}

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
        merge_policy = get_val('model.arch4.merge_policy', 'binary')
        sniper_replace_margin = get_val('model.arch4.sniper_replace_margin', 0.10)
        roi_small_thresh = get_val('model.arch4.roi_small_thresh', 32.0)
        roi_large_thresh = get_val('model.arch4.roi_large_thresh', 96.0)
        large_roi_score_thresh = get_val('model.arch4.large_roi_score_thresh', 0.50)
        sniper_max_det_per_crop = get_val('model.arch4.sniper_max_det_per_crop', 0)
        use_verifier = get_val('model.arch4.use_verifier', False)
        verifier_mode = get_val('model.arch4.verifier_mode', 'iou')
        verifier_iou_thresh = get_val('model.arch4.verifier_iou_thresh', 0.10)
        verifier_topk_cap = get_val('model.arch4.verifier_topk_cap', 3)

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
            merge_policy=merge_policy,
            sniper_replace_margin=sniper_replace_margin,
            roi_small_thresh=roi_small_thresh,
            roi_large_thresh=roi_large_thresh,
            large_roi_score_thresh=large_roi_score_thresh,
            sniper_max_det_per_crop=sniper_max_det_per_crop,
            crop_refine_mode=crop_refine_mode,
            sniper_imgsz_mode=sniper_imgsz_mode,
            sniper_imgsz_fixed=sniper_imgsz_fixed,
            use_verifier=use_verifier,
            verifier_mode=verifier_mode,
            verifier_iou_thresh=verifier_iou_thresh,
            verifier_topk_cap=verifier_topk_cap,
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
        print(f" - Merge policy:                 {self.cfg.merge_policy}")
        if self.cfg.merge_policy == 'score_cond':
            print(f" - Sniper replace margin:        {self.cfg.sniper_replace_margin}")
        elif self.cfg.merge_policy == 'size_cond':
            print(f" - ROI small thresh:             {self.cfg.roi_small_thresh}")
            print(f" - ROI large thresh:             {self.cfg.roi_large_thresh}")
            print(f" - Large ROI score thresh:       {self.cfg.large_roi_score_thresh}")
            print(f" - Sniper replace margin:        {self.cfg.sniper_replace_margin}")
        if self.cfg.sniper_max_det_per_crop > 0:
            print(f" - Sniper max det per crop:      {self.cfg.sniper_max_det_per_crop}")
        print(f" - Crop refine mode:             {self.cfg.crop_refine_mode}")
        print(f" - Sniper imgsz mode:            {self.cfg.sniper_imgsz_mode}")
        if self.cfg.sniper_imgsz_mode == 'fixed':
            print(f" - Sniper imgsz fixed:           {self.cfg.sniper_imgsz_fixed}")
        print(f" - Use verifier:                 {self.cfg.use_verifier}")
        if self.cfg.use_verifier:
            print(f" - Verifier mode:                {self.cfg.verifier_mode}")
            if self.cfg.verifier_mode == 'topk':
                print(f" - Verifier topk cap:            {self.cfg.verifier_topk_cap}")
            else:
                print(f" - Verifier IoU thresh:          {self.cfg.verifier_iou_thresh}")
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
            'roi_debug': [],
            'crops_lr': [],
            'crops_refined': [],
            'crops_sr': [],  # kept for backward compatibility with old debug tooling
            'crop_meta': [],
            'pass2_raw': [],
            'pass2_after_nms': [],
            'final_records_pre_nms': [],
            'final_records_post_nms': [],
        }
        debug_group_map: Dict[Tuple[int, int], Dict[str, Any]] = {}

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
                'debug_records': [
                    self._make_box_records(
                        source='scout_confident',
                        image_index=b_idx,
                        roi_index=None,
                        boxes=confident_boxes,
                        scores=confident_scores,
                        classes=confident_classes,
                    )
                ],
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

            for roi_idx, group in enumerate(roi_groups):
                group['roi_index'] = int(roi_idx)
                merged_box = group['merged_box'].unsqueeze(0)
                crops, coords = self._extract_crops(lr_images[b_idx], merged_box)
                group['fallback_boxes'] = group['member_boxes'][group['member_scores'] >= self.cfg.final_conf]
                group['fallback_scores'] = group['member_scores'][group['member_scores'] >= self.cfg.final_conf]
                group['fallback_classes'] = group['member_classes'][group['member_scores'] >= self.cfg.final_conf]

                if debug:
                    group_debug = {
                        'image_index': int(b_idx),
                        'roi_index': int(roi_idx),
                        'merged_box_lr': self._tensor_to_box_list(group['merged_box'].unsqueeze(0)),
                        'member_boxes_lr': self._tensor_to_box_list(group['member_boxes']),
                        'member_scores': self._tensor_to_scalar_list(group['member_scores']),
                        'member_classes': self._tensor_to_int_list(group['member_classes']),
                        'fallback_boxes_lr': self._tensor_to_box_list(group['fallback_boxes']),
                        'fallback_scores': self._tensor_to_scalar_list(group['fallback_scores']),
                        'fallback_classes': self._tensor_to_int_list(group['fallback_classes']),
                        'crop_valid': False,
                        'coord_lr': None,
                        'sniper_input_size': None,
                        'sniper_raw_boxes_hr_input': [],
                        'sniper_raw_scores': [],
                        'sniper_raw_classes': [],
                        'sniper_after_nms_boxes_hr_input': [],
                        'sniper_after_nms_scores': [],
                        'sniper_after_nms_classes': [],
                        'sniper_kept_boxes_hr_input': [],
                        'sniper_kept_scores': [],
                        'sniper_kept_classes': [],
                        'mapped_global_boxes_lr': [],
                        'sniper_hit': False,
                        'fallback_used': False,
                        'final_kept_box_count': 0,
                        'final_kept_sources': [],
                        'final_kept_boxes_lr': [],
                    }
                    group['debug_entry'] = group_debug
                    debug_info['roi_debug'].append(group_debug)
                    debug_group_map[(int(b_idx), int(roi_idx))] = group_debug

                if len(crops) == 0:
                    group['crop_valid'] = False
                    group['coord'] = None
                    stats['fallback_groups_total'] += 1
                    stats['crop_invalid_groups_total'] += 1
                    if debug and 'debug_entry' in group:
                        group['debug_entry']['fallback_used'] = True
                    self._append_group_fallback(final_results[b_idx], group, image_index=b_idx)
                    continue

                group['crop_valid'] = True
                group['coord'] = coords[0]
                if debug and 'debug_entry' in group:
                    group['debug_entry']['crop_valid'] = True
                    group['debug_entry']['coord_lr'] = [int(v) for v in coords[0]]
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
                if debug and 'debug_entry' in meta['group']:
                    meta['group']['debug_entry']['sniper_input_size'] = int(hr_input_size)

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
                meta = crop_metadata[i]
                img_idx = meta['img_idx']
                group = meta['group']
                group_debug = group.get('debug_entry') if debug else None

                if group_debug is not None:
                    group_debug['sniper_raw_boxes_hr_input'] = self._tensor_to_box_list(res['boxes'])
                    group_debug['sniper_raw_scores'] = self._tensor_to_scalar_list(res['scores'])
                    group_debug['sniper_raw_classes'] = self._tensor_to_int_list(res['classes'])

                res = self._apply_batched_nms(res, self.cfg.sniper_nms_iou)
                if debug:
                    debug_info['pass2_after_nms'].append(res)
                if group_debug is not None:
                    group_debug['sniper_after_nms_boxes_hr_input'] = self._tensor_to_box_list(res['boxes'])
                    group_debug['sniper_after_nms_scores'] = self._tensor_to_scalar_list(res['scores'])
                    group_debug['sniper_after_nms_classes'] = self._tensor_to_int_list(res['classes'])

                if self.cfg.sniper_max_det_per_crop > 0 and len(res['boxes']) > self.cfg.sniper_max_det_per_crop:
                    topk = res['scores'].argsort(descending=True)[: int(self.cfg.sniper_max_det_per_crop)]
                    res = {
                        'boxes': res['boxes'][topk],
                        'scores': res['scores'][topk],
                        'classes': res['classes'][topk],
                    }

                if len(res['boxes']) == 0:
                    stats['fallback_groups_total'] += 1
                    if group_debug is not None:
                        group_debug['fallback_used'] = True
                    self._append_group_fallback(final_results[img_idx], group, image_index=img_idx)
                    continue

                keep = res['scores'] >= float(self.cfg.final_conf)
                if keep.sum().item() == 0:
                    stats['fallback_groups_total'] += 1
                    if group_debug is not None:
                        group_debug['fallback_used'] = True
                    self._append_group_fallback(final_results[img_idx], group, image_index=img_idx)
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

                if bool(self.cfg.use_verifier) and len(global_boxes) > 0:
                    member_boxes = group.get('member_boxes')
                    verifier_mode = str(getattr(self.cfg, 'verifier_mode', 'iou')).lower()
                    if verifier_mode == 'topk':
                        n_members = int(member_boxes.shape[0]) if member_boxes is not None and member_boxes.numel() > 0 else 0
                        topk_cap = max(1, int(getattr(self.cfg, 'verifier_topk_cap', 3)))
                        max_keep = max(1, min(topk_cap, n_members + 1 if n_members > 0 else topk_cap))
                        if len(global_boxes) > max_keep:
                            topk_idx = res_scores.argsort(descending=True)[:max_keep]
                            global_boxes = global_boxes[topk_idx]
                            res_boxes = res_boxes[topk_idx]
                            res_scores = res_scores[topk_idx]
                            res_classes = res_classes[topk_idx]
                        if group_debug is not None:
                            group_debug['verifier_topk_max_keep'] = int(max_keep)
                    elif member_boxes is not None and member_boxes.numel() > 0:
                        iou_mat = box_iou(member_boxes.float(), global_boxes.float())
                        verifier_thresh = float(self.cfg.verifier_iou_thresh)
                        kept_sniper_indices = set()
                        for scout_idx in range(iou_mat.shape[0]):
                            row = iou_mat[scout_idx]
                            best_idx = int(row.argmax().item())
                            if float(row[best_idx].item()) >= verifier_thresh:
                                kept_sniper_indices.add(best_idx)

                        if kept_sniper_indices:
                            kept_idx = sorted(kept_sniper_indices)
                            idx_tensor = torch.tensor(kept_idx, device=global_boxes.device, dtype=torch.long)
                            global_boxes = global_boxes[idx_tensor]
                            res_boxes = res_boxes[idx_tensor]
                            res_scores = res_scores[idx_tensor]
                            res_classes = res_classes[idx_tensor]
                            if group_debug is not None:
                                group_debug['verifier_kept_indices'] = [int(v) for v in kept_idx]
                        else:
                            stats['fallback_groups_total'] += 1
                            if group_debug is not None:
                                group_debug['fallback_used'] = True
                                group_debug['verifier_rejected_all'] = True
                            self._append_group_fallback(final_results[img_idx], group, image_index=img_idx)
                            continue

                stats['sniper_hit_groups_total'] += 1
                if group_debug is not None:
                    group_debug['sniper_hit'] = True
                    group_debug['sniper_kept_boxes_hr_input'] = self._tensor_to_box_list(res_boxes)
                    group_debug['sniper_kept_scores'] = self._tensor_to_scalar_list(res_scores)
                    group_debug['sniper_kept_classes'] = self._tensor_to_int_list(res_classes)
                    group_debug['mapped_global_boxes_lr'] = self._tensor_to_box_list(global_boxes)

                merge_policy = self._normalize_merge_policy(self.cfg.merge_policy)
                keep_sniper_only = bool(self.cfg.drop_uncertain_if_sniper_hits)
                if merge_policy == 'score_cond':
                    fallback_scores = group.get('fallback_scores')
                    sniper_best = float(res_scores.max().item()) if len(res_scores) > 0 else 0.0
                    fallback_best = (
                        float(fallback_scores.max().item())
                        if fallback_scores is not None and fallback_scores.numel() > 0
                        else 0.0
                    )
                    keep_sniper_only = sniper_best > (fallback_best + float(self.cfg.sniper_replace_margin))
                elif merge_policy == 'size_cond':
                    fallback_scores = group.get('fallback_scores')
                    sniper_best = float(res_scores.max().item()) if len(res_scores) > 0 else 0.0
                    fallback_best = (
                        float(fallback_scores.max().item())
                        if fallback_scores is not None and fallback_scores.numel() > 0
                        else 0.0
                    )
                    merged_box = group['merged_box']
                    roi_w = float((merged_box[2] - merged_box[0]).item())
                    roi_h = float((merged_box[3] - merged_box[1]).item())
                    roi_size = max(roi_w, roi_h)
                    small_thresh = float(self.cfg.roi_small_thresh)
                    large_thresh = float(self.cfg.roi_large_thresh)
                    if roi_size < small_thresh:
                        keep_sniper_only = False
                    elif roi_size > large_thresh:
                        high_mask = res_scores >= float(self.cfg.large_roi_score_thresh)
                        if high_mask.sum().item() == 0:
                            stats['fallback_groups_total'] += 1
                            if group_debug is not None:
                                group_debug['fallback_used'] = True
                            self._append_group_fallback(final_results[img_idx], group, image_index=img_idx)
                            continue
                        res_boxes = res_boxes[high_mask]
                        res_scores = res_scores[high_mask]
                        res_classes = res_classes[high_mask]
                        global_boxes = global_boxes[high_mask]
                        keep_sniper_only = True
                    else:
                        keep_sniper_only = sniper_best > (fallback_best + float(self.cfg.sniper_replace_margin))

                if not keep_sniper_only:
                    self._append_group_fallback(final_results[img_idx], group, image_index=img_idx)
                final_results[img_idx]['boxes'].append(global_boxes)
                final_results[img_idx]['scores'].append(res_scores)
                final_results[img_idx]['classes'].append(res_classes)
                final_results[img_idx]['debug_records'].append(
                    self._make_box_records(
                        source='sniper',
                        image_index=img_idx,
                        roi_index=group.get('roi_index'),
                        boxes=global_boxes,
                        scores=res_scores,
                        classes=res_classes,
                    )
                )

        output_detections: List[Dict[str, torch.Tensor]] = []
        for res in final_results:
            if len(res['boxes']) == 0:
                if debug:
                    debug_info['final_records_pre_nms'].append([])
                    debug_info['final_records_post_nms'].append([])
                output_detections.append({
                    'boxes': torch.empty((0, 4), device=self.cfg.device),
                    'scores': torch.empty((0,), device=self.cfg.device),
                    'classes': torch.empty((0,), device=self.cfg.device),
                })
                continue

            all_boxes = torch.cat(res['boxes'], dim=0) if len(res['boxes']) > 0 else torch.empty((0, 4), device=self.cfg.device)
            all_scores = torch.cat(res['scores'], dim=0) if len(res['scores']) > 0 else torch.empty((0,), device=self.cfg.device)
            all_classes = torch.cat(res['classes'], dim=0) if len(res['classes']) > 0 else torch.empty((0,), device=self.cfg.device)
            all_records = [record for record_batch in res.get('debug_records', []) for record in record_batch]

            if all_boxes.numel() == 0:
                if debug:
                    debug_info['final_records_pre_nms'].append(all_records)
                    debug_info['final_records_post_nms'].append([])
                output_detections.append({
                    'boxes': all_boxes,
                    'scores': all_scores,
                    'classes': all_classes,
                })
                continue

            keep = batched_nms(all_boxes, all_scores, all_classes.long(), self.cfg.final_nms_iou)
            if debug:
                kept_indices = keep.detach().cpu().tolist()
                kept_records = [all_records[idx] for idx in kept_indices] if all_records else []
                debug_info['final_records_pre_nms'].append(all_records)
                debug_info['final_records_post_nms'].append(kept_records)
                for record in kept_records:
                    roi_index = record.get('roi_index')
                    image_index = record.get('image_index')
                    if roi_index is None or image_index is None:
                        continue
                    entry = debug_group_map.get((int(image_index), int(roi_index)))
                    if entry is None:
                        continue
                    entry['final_kept_box_count'] += 1
                    entry['final_kept_sources'].append(record['source'])
                    entry['final_kept_boxes_lr'].append(record['box_lr'])
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

    def _append_group_fallback(
        self,
        result_dict: Dict[str, List[torch.Tensor]],
        group: Dict[str, Any],
        image_index: Optional[int] = None,
    ):
        fb_boxes = group.get('fallback_boxes')
        fb_scores = group.get('fallback_scores')
        fb_classes = group.get('fallback_classes')
        if fb_boxes is None or fb_boxes.numel() == 0:
            return
        result_dict['boxes'].append(fb_boxes)
        result_dict['scores'].append(fb_scores)
        result_dict['classes'].append(fb_classes)
        result_dict.setdefault('debug_records', []).append(
            self._make_box_records(
                source='scout_fallback',
                image_index=image_index,
                roi_index=group.get('roi_index'),
                boxes=fb_boxes,
                scores=fb_scores,
                classes=fb_classes,
            )
        )

    def _make_box_records(
        self,
        source: str,
        image_index: Optional[int],
        roi_index: Optional[int],
        boxes: torch.Tensor,
        scores: torch.Tensor,
        classes: torch.Tensor,
    ) -> List[Dict[str, Any]]:
        if boxes is None or boxes.numel() == 0:
            return []
        box_list = self._tensor_to_box_list(boxes)
        score_list = self._tensor_to_scalar_list(scores)
        class_list = self._tensor_to_int_list(classes)
        records: List[Dict[str, Any]] = []
        for box, score, cls in zip(box_list, score_list, class_list):
            records.append({
                'source': source,
                'image_index': None if image_index is None else int(image_index),
                'roi_index': None if roi_index is None else int(roi_index),
                'box_lr': box,
                'score': float(score),
                'class': int(cls),
            })
        return records

    def _tensor_to_box_list(self, boxes: torch.Tensor) -> List[List[float]]:
        if boxes is None or boxes.numel() == 0:
            return []
        return boxes.detach().cpu().float().tolist()

    def _tensor_to_scalar_list(self, values: torch.Tensor) -> List[float]:
        if values is None or values.numel() == 0:
            return []
        return [float(v) for v in values.detach().cpu().float().tolist()]

    def _tensor_to_int_list(self, values: torch.Tensor) -> List[int]:
        if values is None or values.numel() == 0:
            return []
        return [int(v) for v in values.detach().cpu().long().tolist()]

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

    def _normalize_merge_policy(self, mode: Any) -> str:
        norm = str(mode).strip().lower()
        if norm not in self.VALID_MERGE_POLICIES:
            raise ValueError(f"Invalid merge_policy={mode!r}. Valid: {sorted(self.VALID_MERGE_POLICIES)}")
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
