import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from torchvision.ops import batched_nms, box_iou

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


class Arch4RoiAwareNMS(Arch4Adaptive):
    """
    Arch4 + ROI-aware hierarchical NMS

    Key idea:
      1) Scout output is post-processed once more (stage-wise NMS)
      2) overlapping uncertain Scout boxes are merged into ROI groups
      3) each ROI group is cropped once and sent to SR + Sniper
      4) if Sniper succeeds on that ROI, uncertain Scout boxes from the same ROI are dropped
      5) final global class-aware NMS is applied only after source-aware merge
    """

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
        )

    def _print_info(self):
        print(f"\n[Arch4 ROI-aware Config]")
        print(f" - Scout conf(pass1_conf):     {self.cfg.pass1_conf}")
        print(f" - High conf(pass2_conf):      {self.cfg.pass2_conf}")
        print(f" - Sniper conf(sniper_conf):   {self.cfg.sniper_conf}")
        print(f" - Final conf(final_conf):     {self.cfg.final_conf}")
        print(f" - Scout NMS IoU:              {self.cfg.scout_nms_iou}")
        print(f" - ROI merge IoU:              {self.cfg.roi_merge_iou}")
        print(f" - ROI center ratio:           {self.cfg.roi_center_ratio}")
        print(f" - Sniper NMS IoU:             {self.cfg.sniper_nms_iou}")
        print(f" - Final NMS IoU:              {self.cfg.final_nms_iou}")
        print(f" - Drop uncertain if sniper hits: {self.cfg.drop_uncertain_if_sniper_hits}")
        print(f" - Batch Strategy:             {self.cfg.batch_size_sr} crops per SR pass")

    @torch.no_grad()
    def forward(self, lr_images: torch.Tensor, debug: bool = False) -> Dict[str, Any]:
        batch_size, _, height, width = lr_images.shape
        self.scout_detector.eval()
        self.sr_model.eval()
        self.sniper_detector.eval()

        debug_info = {
            'pass1_raw': [],
            'pass1_after_nms': [],
            'roi_groups': [],
            'crops_lr': [],
            'crops_sr': [],
            'crop_meta': [],
            'pass2_raw': [],
            'pass2_after_nms': [],
        }

        # Phase 1: Scout
        pass1_preds = self.scout_detector.predict(
            lr_images,
            conf=self.cfg.pass1_conf,
            iou=self.cfg.scout_nms_iou,
        )
        if debug:
            debug_info['pass1_raw'] = pass1_preds

        final_results = []
        all_crops_lr: List[torch.Tensor] = []
        crop_metadata: List[Dict[str, Any]] = []

        for b_idx, det in enumerate(pass1_preds):
            det = self._apply_batched_nms(det, self.cfg.scout_nms_iou)
            if debug:
                debug_info['pass1_after_nms'].append(det)

            boxes = det['boxes']
            scores = det['scores']
            classes = det['classes']

            if debug and len(scores) > 0:
                print(f"\n[Img {b_idx}] Scout 결과(ROI-aware): {len(scores)}개 (Max Conf: {scores.max():.4f})")
                print(f"  - Thresholds: Pass1={self.cfg.pass1_conf}, High={self.cfg.pass2_conf}, Final={self.cfg.final_conf}")

            confident_mask = scores >= self.cfg.pass2_conf
            confident_boxes = boxes[confident_mask]
            confident_scores = scores[confident_mask]
            confident_classes = classes[confident_mask]

            uncertain_boxes = boxes[~confident_mask]
            uncertain_scores = scores[~confident_mask]
            uncertain_classes = classes[~confident_mask]

            if debug:
                print(f"  -> A급(확실): {len(confident_boxes)}개")
                print(f"  -> B급(애매): {len(uncertain_boxes)}개")

            # final_results starts with confident Scout boxes only
            final_results.append({
                'boxes': [confident_boxes],
                'scores': [confident_scores],
                'classes': [confident_classes],
            })

            # Merge overlapping uncertain boxes into ROI groups
            roi_groups = self._build_roi_groups(
                uncertain_boxes,
                uncertain_scores,
                uncertain_classes,
            )
            if debug:
                debug_info['roi_groups'].append(roi_groups)
                print(f"  -> ROI groups after merge: {len(roi_groups)}")

            # For each merged ROI, create one crop only
            for group in roi_groups:
                merged_box = group['merged_box'].unsqueeze(0)
                crops, coords = self._extract_crops(lr_images[b_idx], merged_box)
                group['fallback_boxes'] = group['member_boxes'][group['member_scores'] >= self.cfg.final_conf]
                group['fallback_scores'] = group['member_scores'][group['member_scores'] >= self.cfg.final_conf]
                group['fallback_classes'] = group['member_classes'][group['member_scores'] >= self.cfg.final_conf]

                if len(crops) == 0:
                    group['crop_valid'] = False
                    group['coord'] = None
                    # No crop -> fallback to Scout uncertain boxes later
                    self._append_group_fallback(final_results[b_idx], group)
                    continue

                group['crop_valid'] = True
                group['coord'] = coords[0]
                all_crops_lr.append(crops[0])
                crop_metadata.append({
                    'img_idx': b_idx,
                    'group': group,
                })

        # Phase 2: SR + Sniper on merged ROI groups
        if len(all_crops_lr) > 0:
            batch_crops_lr = torch.stack(all_crops_lr).to(self.cfg.device)
            batch_crops_hr = self._run_batch_sr(batch_crops_lr)

            if debug:
                debug_info['crops_lr'] = [c.cpu() for c in all_crops_lr]
                debug_info['crops_sr'] = [c.cpu() for c in batch_crops_hr]
                debug_info['crop_meta'] = crop_metadata

            sniper_imgsz = int(batch_crops_hr.shape[-1])
            sniper_results = self.sniper_detector.predict(
                batch_crops_hr,
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
                    self._append_group_fallback(final_results[img_idx], group)
                    continue

                keep = res['scores'] >= float(self.cfg.final_conf)
                if keep.sum().item() == 0:
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
                )

                # source-aware rule:
                # if Sniper finds something in this ROI, we trust Sniper and drop uncertain Scout boxes
                if self.cfg.drop_uncertain_if_sniper_hits:
                    final_results[img_idx]['boxes'].append(global_boxes)
                    final_results[img_idx]['scores'].append(res_scores)
                    final_results[img_idx]['classes'].append(res_classes)
                else:
                    self._append_group_fallback(final_results[img_idx], group)
                    final_results[img_idx]['boxes'].append(global_boxes)
                    final_results[img_idx]['scores'].append(res_scores)
                    final_results[img_idx]['classes'].append(res_classes)

        # Phase 3: Final global class-aware NMS
        output_detections = []
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

        if debug:
            return {'detections': output_detections, 'debug_info': debug_info}
        return {'detections': output_detections}

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
        """
        Merge overlapping / nearby uncertain Scout boxes into ROI groups.

        A group means: "we will crop this area only once".
        """
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
        # Rule 1: enough overlap
        iou = box_iou(box_a.unsqueeze(0), box_b.unsqueeze(0)).item()
        if iou >= float(self.cfg.roi_merge_iou):
            return True

        # Rule 2: centers are sufficiently close relative to larger box size
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

    def _sniper_boxes_to_global(
        self,
        res_boxes: torch.Tensor,
        coord,
        width: int,
        height: int,
    ) -> torch.Tensor:
        ix1, iy1, ix2, iy2 = coord
        crop_w = max(1, ix2 - ix1)
        crop_h = max(1, iy2 - iy1)
        lr_size = float(self.cfg.crop_size_lr)
        scale = float(self.cfg.upscale_factor)

        boxes_lr_resized = res_boxes / scale
        boxes_lr = boxes_lr_resized.clone()
        boxes_lr[:, [0, 2]] *= (crop_w / lr_size)
        boxes_lr[:, [1, 3]] *= (crop_h / lr_size)

        global_boxes = boxes_lr.clone()
        global_boxes[:, [0, 2]] += ix1
        global_boxes[:, [1, 3]] += iy1
        global_boxes[:, [0, 2]] = global_boxes[:, [0, 2]].clamp(0, width - 1)
        global_boxes[:, [1, 3]] = global_boxes[:, [1, 3]].clamp(0, height - 1)
        return global_boxes
