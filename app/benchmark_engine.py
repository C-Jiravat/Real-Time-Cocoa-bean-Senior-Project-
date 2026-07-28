"""
benchmark_engine.py
Core logic for batch and single-image model benchmarking.
Computes Precision, Recall, F1, AP (per class) and mAP.
"""

import os, cv2, numpy as np, zipfile, tempfile, time, shutil, base64
from pathlib import Path

# ─── Class Definitions ────────────────────────────────────────────────────────
# Class IDs ตรงกับที่โมเดลถูก train มา
COLOR_CLASS_NAMES = {0: "ม่วง (Purple)", 1: "น้ำตาล (Brown)"}
DEFECT_CLASS_NAMES = {
    0: "ปกติ (Normal)",
    1: "งอก (Germinated)",
    2: "เทาหินชนวน (Rock/Hard)",
    3: "ขึ้นรา (Moldy)",
}

# ─── Placeholder Paths ────────────────────────────────────────────────────────
# TODO: แก้ path ให้ถูกต้องก่อนใช้งาน (หรืออัพโหลดผ่านหน้าเว็บแทน)
DEFAULT_COLOR_IMAGES_DIR  = r"D:\Chula\Senior_Project\color_mix_new_old-20250312T202002Z-001\images\test"
DEFAULT_COLOR_LABELS_DIR  = r"D:\Chula\Senior_Project\color_mix_new_old-20250312T202002Z-001\labels\test"
DEFAULT_DEFECT_IMAGES_DIR = r"D:\Chula\Senior_Project\defect_mix_new_old\images\test"
DEFAULT_DEFECT_LABELS_DIR = r"D:\Chula\Senior_Project\defect_mix_new_old\labels\test"

# ─── Label Parsing ────────────────────────────────────────────────────────────
def parse_yolo_label(label_path, img_w, img_h):
    """
    Parse a YOLO format .txt label file.
    Each line: class_id cx cy w h  (normalized 0..1)
    Returns: list of {"class_id": int, "bbox": [x1, y1, x2, y2]} in pixel coords
    """
    gts = []
    if not label_path or not os.path.exists(label_path):
        return gts
    with open(label_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(float(parts[0]))
            cx, cy, w, h = (float(p) for p in parts[1:5])
            x1 = max(0.0, (cx - w / 2) * img_w)
            y1 = max(0.0, (cy - h / 2) * img_h)
            x2 = min(float(img_w), (cx + w / 2) * img_w)
            y2 = min(float(img_h), (cy + h / 2) * img_h)
            gts.append({"class_id": cls_id, "bbox": [x1, y1, x2, y2]})
    return gts

# ─── IoU ──────────────────────────────────────────────────────────────────────
def compute_iou(box_a, box_b):
    """Compute Intersection-over-Union between two boxes [x1, y1, x2, y2]."""
    xa1 = max(box_a[0], box_b[0])
    ya1 = max(box_a[1], box_b[1])
    xa2 = min(box_a[2], box_b[2])
    ya2 = min(box_a[3], box_b[3])
    inter = max(0.0, xa2 - xa1) * max(0.0, ya2 - ya1)
    if inter == 0.0:
        return 0.0
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0

# ─── Detection Matching ───────────────────────────────────────────────────────
def match_detections(preds, gts, iou_thresh=0.5):
    """
    Greedy IoU matching: highest-IoU pair matched first.
    Returns: (matched_pairs [(pi, gi)], unmatched_pred_idxs, unmatched_gt_idxs)
    """
    if not preds or not gts:
        return [], list(range(len(preds))), list(range(len(gts)))

    iou_mat = np.zeros((len(preds), len(gts)), dtype=np.float32)
    for i, p in enumerate(preds):
        for j, g in enumerate(gts):
            iou_mat[i, j] = compute_iou(p['bbox'], g['bbox'])

    matched, used_p, used_g = [], set(), set()
    while True:
        max_val = float(iou_mat.max())
        if max_val < iou_thresh:
            break
        pi, gi = np.unravel_index(iou_mat.argmax(), iou_mat.shape)
        pi, gi = int(pi), int(gi)
        matched.append((pi, gi))
        used_p.add(pi)
        used_g.add(gi)
        iou_mat[pi, :] = -1.0
        iou_mat[:, gi] = -1.0

    unmatched_p = [i for i in range(len(preds)) if i not in used_p]
    unmatched_g = [j for j in range(len(gts))  if j not in used_g]
    return matched, unmatched_p, unmatched_g

# ─── Task-specific class colour maps (RGB tuples) ────────────────────────────
# Color task: matches main app color_map_weight1 (BGR→RGB)
#   BGR (128,0,128) purple  → RGB (128,0,128)
#   BGR (96,164,244) sandy  → RGB (244,164,96)  ← sandy brown
COLOR_TASK_RGB = {
    0: (160,  32, 240),   # ม่วง  — vivid purple
    1: (180, 100,  30),   # น้ำตาล — warm brown
}
# Defect task: matches main app color_map_weight2 (BGR→RGB)
#   BGR (0,255,0)   → RGB (0,255,0)   green    Normal
#   BGR (0,255,255) → RGB (255,255,0) yellow   งอก
#   BGR (255,0,0)   → RGB (0,0,255)   blue     เทาหินชนวน
#   BGR (0,0,255)   → RGB (255,0,0)   red      ขึ้นรา
DEFECT_TASK_RGB = {
    0: (  0, 220,   0),   # ปกติ      — green
    1: (255, 220,   0),   # งอก       — yellow
    2: ( 80, 160, 255),   # เทาหินชนวน — blue
    3: (255,  60,  60),   # ขึ้นรา    — red
}

def _task_color_map(class_names: dict) -> dict:
    """Return the correct RGB colour map based on number/names of classes."""
    if len(class_names) == 2:   # Color task
        return COLOR_TASK_RGB
    return DEFECT_TASK_RGB      # Defect task (4 classes)

# ─── PIL-based drawing helpers (Thai text support) ───────────────────────────
def _get_pil_font(size=14):
    """Try to load a CJK/Unicode font that covers Thai; fall back to default."""
    from PIL import ImageFont
    # Common paths for fonts that include Thai glyphs
    candidates = [
        # Windows
        r"C:\Windows\Fonts\THSarabunNew.ttf",
        r"C:\Windows\Fonts\tahoma.ttf",
        r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\NotoSansThai-Regular.ttf",
        # Linux / Docker
        "/usr/share/fonts/truetype/thai/TlwgMono.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansThai-Regular.ttf",
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()


def _draw_box_pil(pil_img, bbox, color_rgb, label, is_gt=False):
    """
    Draw one bbox + label onto a PIL Image (RGB).
    GT boxes: solid rectangle, label at bottom.
    Pred boxes: solid rectangle, label at top.
    """
    from PIL import ImageDraw
    draw = ImageDraw.Draw(pil_img, "RGBA")
    x1, y1, x2, y2 = (int(v) for v in bbox)
    
    lw = 2  # Thicker border like app_cpu.py
    
    # 1. Draw solid outer bbox for both
    draw.rectangle([x1, y1, x2, y2], outline=color_rgb + (255,), width=lw)

    # 2. Draw label background & text
    if label:
        font = _get_pil_font(32)  # Larger font (increased from 16 to 32)
        try:
            bbox_txt = draw.textbbox((0, 0), label, font=font)
            tw = bbox_txt[2] - bbox_txt[0]
            th = bbox_txt[3] - bbox_txt[1]
        except AttributeError:
            tw, th = draw.textsize(label, font=font)
        
        pad_x, pad_y = 5, 3
        
        if is_gt:
            # Place GT label at the top edge inside the box (to avoid overlap with Pred)
            bg_x1 = x1
            bg_y1 = y1
            bg_x2 = x1 + tw + pad_x * 2
            bg_y2 = y1 + th + pad_y * 2
        else:
            # Place Pred label at the top edge OUTSIDE the box 
            # If it goes off-screen (y < 0) or to ensure it stacks perfectly below GT label, push it inside below GT
            bg_y1 = y1 - th - pad_y * 2
            if bg_y1 < 0:
                bg_y1 = y1 + th + pad_y * 2
            bg_x1 = x1
            bg_x2 = x1 + tw + pad_x * 2
            bg_y2 = bg_y1 + th + pad_y * 2

        # Background rect (Solid, not transparent, to cover anything under it)
        draw.rectangle([bg_x1, bg_y1, bg_x2, bg_y2], fill=color_rgb + (255,))
        
        # Text
        tc = (255, 255, 255) if sum(color_rgb) < 400 else (0, 0, 0)
        draw.text((bg_x1 + pad_x, bg_y1 + pad_y - 2), label, font=font, fill=tc)


def draw_annotated_image(frame_bgr, gt_list, pred_list, class_names,
                          color_map=None, mismatch_only=False):
    """
    Render GT + Pred boxes on frame_bgr using Pillow (Thai-safe).

    color_map     : dict {class_id: (R,G,B)}.  Auto-detected if None.
    mismatch_only : if True, only draw boxes where GT ≠ Pred (FP / FN / wrong class).

    Returns annotated BGR numpy array.
    """
    from PIL import Image as PILImage, ImageDraw

    if color_map is None:
        color_map = _task_color_map(class_names)

    GT_COLOR = (210, 210, 210)   # light grey for GT
    MISMATCH_BORDER = (255, 60, 60)  # red border for mismatched preds
    LEGEND_H = 64  # Increased height to fit larger text

    # ── Build mismatch set (pred indices that are wrong) ─────────────────────
    # A pred is "wrong" if its class ≠ the GT class it was matched to,
    # OR if it is unmatched (FP).  GT boxes without a match are also shown (FN).
    # We recompute matching here from the stored bbox lists.
    mismatch_pred_idxs = set(range(len(pred_list)))  # default: all wrong
    mismatch_gt_idxs   = set(range(len(gt_list)))
    if pred_list and gt_list and not mismatch_only:
        # not needed unless mismatch_only
        pass
    if mismatch_only:
        from benchmark_engine import match_detections
        matched, unmatched_p, unmatched_g = match_detections(
            pred_list, gt_list, iou_thresh=0.5)
        correct_p = set()
        for pi, gi in matched:
            if pred_list[pi]["class_id"] == gt_list[gi]["class_id"]:
                correct_p.add(pi)
                mismatch_gt_idxs.discard(gi)   # this GT was correctly matched
        mismatch_pred_idxs = set(range(len(pred_list))) - correct_p

    # ── Convert BGR → RGBA PIL ────────────────────────────────────────────────
    pil = PILImage.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)).convert("RGBA")

    # ── Draw GT boxes ─────────────────────────────────────────────────────────
    for gi, g in enumerate(gt_list):
        if mismatch_only and gi not in mismatch_gt_idxs:
            continue
        cid  = g["class_id"]
        name = class_names.get(cid, str(cid))
        _draw_box_pil(pil, g["bbox"], GT_COLOR,
                      label=f"GT: {name}", is_gt=True)

    # ── Draw Predicted boxes ──────────────────────────────────────────────────
    for pi, p in enumerate(pred_list):
        if mismatch_only and pi not in mismatch_pred_idxs:
            continue
        cid  = p["class_id"]
        conf = p["confidence"] * 100
        name = class_names.get(cid, str(cid))
        rgb  = color_map.get(cid, (180, 180, 180))
        if mismatch_only:
            border_col = MISMATCH_BORDER
            _draw_box_pil(pil, p["bbox"], border_col,
                          label=f"✗ Pred: {name} {conf:.0f}%", is_gt=False)
        else:
            _draw_box_pil(pil, p["bbox"], rgb,
                          label=f"Pred: {name} {conf:.0f}%", is_gt=False)

    # ── Legend bar at bottom ──────────────────────────────────────────────────
    w, h = pil.size
    legend = PILImage.new("RGBA", (w, LEGEND_H), (15, 18, 26, 245))
    ld = ImageDraw.Draw(legend)
    font_sm = _get_pil_font(24)  # Increased font size for legend

    legend_items = [("GT", GT_COLOR, True)]
    seen = set()
    for p in pred_list:
        cid = p["class_id"]
        if cid not in seen:
            seen.add(cid)
            name = class_names.get(cid, str(cid))
            rgb  = color_map.get(cid, (180, 180, 180))
            legend_items.append((f"{name}", rgb, False))

    x_cur = 10
    for lbl, col, is_gt_leg in legend_items:
        sw = 24  # Increased swatch size
        sy = (LEGEND_H - sw) // 2
        if is_gt_leg:
            ld.rectangle([x_cur, sy, x_cur+sw, sy+sw], outline=col, width=2)
        else:
            ld.rectangle([x_cur, sy, x_cur+sw, sy+sw], fill=col)
        prefix = "GT: " if is_gt_leg else "Pred: "
        text = prefix + lbl
        try:
            bbox_txt = ld.textbbox((0,0), text, font=font_sm)
            tw = bbox_txt[2] - bbox_txt[0]
            th_txt = bbox_txt[3] - bbox_txt[1]
        except AttributeError:
            tw, th_txt = ld.textsize(text, font=font_sm)
        ld.text((x_cur + sw + 8, (LEGEND_H - th_txt) // 2 - 4), text, font=font_sm, fill=(200,200,200))
        x_cur += sw + 8 + tw + 24
        if x_cur > w - 20:
            break

    combined = PILImage.new("RGBA", (w, h + LEGEND_H), (0, 0, 0, 255))
    combined.paste(pil,    (0, 0))
    combined.paste(legend, (0, h))

    result_rgb = np.array(combined.convert("RGB"))
    return cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)


# ─── Per-Image Inference ──────────────────────────────────────────────────────
def _eval_one_image(frame, label_path, class_names,
                    detection_model, classifier_session,
                    detect_fn, batch_predict_fn,
                    model_mode, iou_thresh=0.5, conf_thres=0.45,
                    return_annotated=False):
    """
    Run inference + evaluation for ONE task on ONE image.

    model_mode='yolo'  → use detection_model's class prediction as final class
    model_mode='combo' → use detection_model for crop, classifier_session for class

    Returns a per-image result dict for metric accumulation.
    Extra keys when return_annotated=True:
        'annotated_img' : np.ndarray  (BGR, same size as frame)
        'timing'        : dict with yolo_sec, crop_sec, classifier_sec, total_sec
    """
    t_total_start = time.time()

    img_h, img_w = frame.shape[:2]
    gt = parse_yolo_label(label_path, img_w, img_h)
    n_classes = len(class_names)

    tp_dict  = {c: 0 for c in range(n_classes)}
    fp_dict  = {c: 0 for c in range(n_classes)}
    fn_dict  = {c: 0 for c in range(n_classes)}
    gt_count = {c: 0 for c in range(n_classes)}
    matches  = []
    scores   = {c: [] for c in range(n_classes)}

    for g in gt:
        c = g['class_id']
        if c in gt_count:
            gt_count[c] += 1

    # ── 1. YOLO Detection ─────────────────────────────────────────────────────
    t_yolo_start = time.time()
    try:
        raw_dets = detect_fn(detection_model, frame, conf_thres=conf_thres)
    except TypeError:
        # Fallback if detect_fn does not accept conf_thres keyword argument
        raw_dets = detect_fn(detection_model, frame)
    t_yolo = time.time() - t_yolo_start

    # ── 2. Image Cropping ─────────────────────────────────────────────────────
    t_crop_start = time.time()
    preds = []
    if model_mode == 'combo' or model_mode == 'yolo':
        crops, valid_idxs = [], []
        for i, d in enumerate(raw_dets):
            x1, y1, x2, y2 = map(int, d["bbox"])
            crop = frame[max(0, y1):y2, max(0, x1):x2]
            if crop.size == 0:
                continue
            crops.append(crop)
            valid_idxs.append(i)
    t_crop = time.time() - t_crop_start

    # ── 3. Classifier (ConvNeXt) ──────────────────────────────────────────────
    t_cls_start = time.time()
    cls_results = batch_predict_fn(classifier_session, crops, class_names) if crops else []
    t_cls = time.time() - t_cls_start

    for j, vi in enumerate(valid_idxs):
        # Multi-label support: Iterate through all labels returned for this crop
        for idx, _lbl, conf_pct in cls_results[j]:
            preds.append({
                "bbox":       raw_dets[vi]["bbox"],
                "class_id":   idx,
                "confidence": conf_pct / 100.0,
            })

    # ── 4. Post-process + Evaluate ('Match-Any' Multi-label Logic) ─────────────
    t_eval_start = time.time()
    
    # Tracking for confusion matrix and metrics
    gt_matched_flags = [False] * len(gt)
    pred_matched_flags = [False] * len(preds)

    # 1. Identify TRUE POSITIVES (matching any label at the location)
    for pi, p in enumerate(preds):
        if pred_matched_flags[pi]: continue
        
        pred_cls = p["class_id"]
        # Look for any overlapping GT with the same class
        best_iou = -1.0; best_gi = -1
        for gi, g in enumerate(gt):
            if gt_matched_flags[gi]: continue
            iou = compute_iou(p['bbox'], g['bbox'])
            if iou >= iou_thresh and g["class_id"] == pred_cls:
                if iou > best_iou:
                    best_iou = iou
                    best_gi = gi
        
        if best_gi != -1:
            # Found a match!
            tp_dict[pred_cls] = tp_dict.get(pred_cls, 0) + 1
            scores[pred_cls].append((p["confidence"], True))
            pred_matched_flags[pi] = True
            matches.append((pred_cls, pred_cls)) # Match the correct class
            
            # --- THE KEY CHANGE ---
            # Mark ALL GTs at this specific location as 'Found' 
            # so they don't count as FNs (Misses)
            for gi, g in enumerate(gt):
                if compute_iou(p['bbox'], g['bbox']) >= iou_thresh:
                    gt_matched_flags[gi] = True

    # 2. Check for remaining GTs (True Misses or Misclassifications)
    for gi, g in enumerate(gt):
        if gt_matched_flags[gi]: continue
        
        gt_cls = g["class_id"]
        # Is there any prediction overlapping this missed GT?
        found_overlap_p = -1
        for pi, p in enumerate(preds):
            if compute_iou(p['bbox'], g['bbox']) >= iou_thresh:
                found_overlap_p = pi
                break
        
        if found_overlap_p != -1:
            # Detected but class was wrong for ALL labels at this spot
            # (Otherwise it would have been matched in step 1)
            pred_cls = preds[found_overlap_p]["class_id"]
            matches.append((gt_cls, pred_cls))
            # Not a hit, so this GT is technically an FN or part of a wrong classification
            # We count it as FN because it was a missed class label
            fn_dict[gt_cls] = fn_dict.get(gt_cls, 0) + 1
        else:
            # Complete miss (No prediction box overlapped here)
            fn_dict[gt_cls] = fn_dict.get(gt_cls, 0) + 1
            matches.append((gt_cls, n_classes))

    # 3. Check for remaining Predictions (True False Positives)
    for pi, p in enumerate(preds):
        if pred_matched_flags[pi]: continue
            
        pred_cls = p["class_id"]
        # Check if this prediction overlaps with ANY GT
        overlaps_any = False
        for gi, g in enumerate(gt):
            if compute_iou(p['bbox'], g['bbox']) >= iou_thresh:
                overlaps_any = True
                break
        
        if not overlaps_any:
            # PURE FALSE POSITIVE (Background -> Class)
            fp_dict[pred_cls] = fp_dict.get(pred_cls, 0) + 1
            scores[pred_cls].append((p["confidence"], False))
            matches.append((n_classes, pred_cls))
        else:
            # Redundant detection or overlaps but wasn't a TP for any GT
            # Count as FP for the predicted class
            fp_dict[pred_cls] = fp_dict.get(pred_cls, 0) + 1
            scores[pred_cls].append((p["confidence"], False))

    t_eval = time.time() - t_eval_start
    t_total = time.time() - t_total_start

    # Count mismatches for summary
    n_wrong = sum(1 for pi, gi in matched
                  if preds[pi]["class_id"] != gt[gi]["class_id"])
    n_wrong += len(unmatched_p) + len(unmatched_g)

    result = {
        "tp": tp_dict, "fp": fp_dict, "fn": fn_dict,
        "gt_count": gt_count, "scores": scores,
        "matches": matches,
        "n_preds": len(preds), "n_gt": len(gt),
        "n_wrong": n_wrong,
        # 5-step timing
        "timing": {
            "object_detector_sec": round(t_yolo, 4),
            "image_seg_sec":       round(t_crop, 4),
            "classification_sec":  round(t_cls,  4),
            "postprocess_sec":     round(t_eval, 4),
            "total_sec":           round(t_total, 4),
        },
    }

    # ── Optional: return detailed crop info for gallery ──────────────────────
    if return_annotated:
        cmap = _task_color_map(class_names)
        result['annotated_img'] = draw_annotated_image(
            frame, gt, preds, class_names, color_map=cmap, mismatch_only=False)
        # Convert PIL images to NumPy (BGR) for OpenCV compatibility
        result['annotated_img'] = np.array(result['annotated_img'])[:, :, ::-1]
        result['mismatch_img']  = np.array(result['mismatch_img'])[:, :, ::-1]

        # Build crops_data for the UI gallery
        crops_data = [] # List of {crop_img, gt_name, pred_name, conf, is_correct, type}
        
        # 1. Matched (TP or Wrong Class)
        for pi, gi in matched:
            p = preds[pi]
            g = gt[gi]
            x1, y1, x2, y2 = map(int, p["bbox"])
            c_img = frame[max(0, y1):y2, max(0, x1):x2]
            if c_img.size == 0: continue
            
            p_cid = p["class_id"]
            g_cid = g["class_id"]
            is_correct = (p_cid == g_cid)
            
            crops_data.append({
                "img": c_img,
                "gt_name": class_names.get(g_cid, str(g_cid)),
                "pred_name": class_names.get(p_cid, str(p_cid)),
                "conf": p["confidence"] * 100,
                "is_correct": is_correct,
                "type": "matched"
            })

        # 2. Unmatched Predictions (FP)
        for pi in unmatched_p:
            p = preds[pi]
            x1, y1, x2, y2 = map(int, p["bbox"])
            c_img = frame[max(0, y1):y2, max(0, x1):x2]
            if c_img.size == 0: continue
            
            p_cid = p["class_id"]
            crops_data.append({
                "img": c_img,
                "gt_name": "Background/None",
                "pred_name": class_names.get(p_cid, str(p_cid)),
                "conf": p["confidence"] * 100,
                "is_correct": False,
                "type": "fp"
            })

        # 3. Unmatched Ground Truths (FN)
        for gi in unmatched_g:
            g = gt[gi]
            x1, y1, x2, y2 = map(int, g["bbox"])
            c_img = frame[max(0, y1):y2, max(0, x1):x2]
            if c_img.size == 0: continue
            
            g_cid = g["class_id"]
            crops_data.append({
                "img": c_img,
                "gt_name": class_names.get(g_cid, str(g_cid)),
                "pred_name": "Missed (None)",
                "conf": 0,
                "is_correct": False,
                "type": "fn"
            })
            
        result['crops_data'] = crops_data

    return result

# ─── Metric Aggregation ───────────────────────────────────────────────────────
def compute_metrics(all_results, class_names):
    """
    Aggregate per-image results and compute final metrics.
    Returns {"per_class": {0: {...}, 1: {...}, ...}, "mAP": float}
    """
    n_classes = len(class_names)
    tp_sum   = {c: 0 for c in range(n_classes)}
    fp_sum   = {c: 0 for c in range(n_classes)}
    fn_sum   = {c: 0 for c in range(n_classes)}
    gt_total = {c: 0 for c in range(n_classes)}
    all_scores = {c: [] for c in range(n_classes)}

    for r in all_results:
        for c in range(n_classes):
            tp_sum[c]   += r['tp'].get(c, 0)
            fp_sum[c]   += r['fp'].get(c, 0)
            fn_sum[c]   += r['fn'].get(c, 0)
            gt_total[c] += r['gt_count'].get(c, 0)
            all_scores[c].extend(r.get('scores', {}).get(c, []))

    class_metrics = {}
    aps = []

    for c in range(n_classes):
        tp, fp, fn = tp_sum[c], fp_sum[c], fn_sum[c]
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        # AP via 11-point interpolation
        scores_sorted = sorted(all_scores[c], key=lambda x: -x[0])
        n_gt = gt_total[c]
        if n_gt == 0 or not scores_sorted:
            ap = 0.0
        else:
            tp_cum = fp_cum = 0
            precs, recs = [], []
            for sc, is_tp in scores_sorted:
                if is_tp:
                    tp_cum += 1
                else:
                    fp_cum += 1
                precs.append(tp_cum / (tp_cum + fp_cum))
                recs.append(tp_cum / n_gt)
            ap = 0.0
            for t in np.arange(0.0, 1.1, 0.1):
                p_vals = [precs[i] for i in range(len(recs)) if recs[i] >= t]
                ap += max(p_vals) if p_vals else 0.0
            ap /= 11.0

        # Compute accuracy proxy: TP / (TP + FP + FN)
        acc = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

        class_metrics[c] = {
            'name':      class_names[c],
            'accuracy':  round(acc  * 100, 2),
            'precision': round(prec * 100, 2),
            'recall':    round(rec  * 100, 2),
            'f1':        round(f1   * 100, 2),
            'ap':        round(ap   * 100, 2),
            'tp': tp, 'fp': fp, 'fn': fn,
            'gt_count': gt_total[c],
        }
        aps.append(ap)

    mAP = round(float(np.mean(aps)) * 100, 2) if aps else 0.0
    
    # ── Confusion Matrix ──────────────────────────────────────────────────
    # size: (nc+1, nc+1) including background
    matrix = np.zeros((n_classes + 1, n_classes + 1), dtype=int)
    for r in all_results:
        for gt_c, pr_c in r.get('matches', []):
            if gt_c < matrix.shape[0] and pr_c < matrix.shape[1]:
                matrix[gt_c, pr_c] += 1
                
    return {
        "per_class": class_metrics, 
        "mAP": mAP,
        "confusion_matrix": matrix.tolist()
    }

# ─── Public API: Batch ────────────────────────────────────────────────────────
def run_batch_benchmark(images_dir, labels_dir, task, model_mode,
                        model_yolo_seed,
                        model_convnext_color, model_convnext_defect,
                        detect_fn, batch_predict_fn,
                        iou_thresh=0.5, conf_thres=0.45,
                        progress_callback=None,
                        save_annotated_dir=None):
    """
    Run benchmark on a directory of images + labels.

    task               : 'color' | 'defect'
    model_mode         : 'yolo'  | 'combo'
    save_annotated_dir : Folder to save JPGs with GT vs Pred boxes.
    """
    IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    img_files = sorted([f for f in Path(images_dir).iterdir()
                        if f.suffix.lower() in IMG_EXTS])

    color_results, defect_results = [], []
    batch_metadata = [] # List of {filename, n_wrong, img_url}
    
    t_start = time.time()
    skipped = 0

    # Ensure save dir exists
    if save_annotated_dir:
        os.makedirs(save_annotated_dir, exist_ok=True)

    for idx, img_path in enumerate(img_files):
        stem       = img_path.stem
        label_path = os.path.join(labels_dir, stem + '.txt')

        frame = cv2.imread(str(img_path))
        if frame is None:
            skipped += 1
            if progress_callback: progress_callback(idx + 1, len(img_files))
            continue

        class_names = COLOR_CLASS_NAMES if task == 'color' else DEFECT_CLASS_NAMES
        m_session   = model_convnext_color if task == 'color' else model_convnext_defect
        
        # Run evaluation
        r = _eval_one_image(
            frame, label_path, class_names,
            model_yolo_seed, m_session,
            detect_fn, batch_predict_fn, model_mode, iou_thresh, conf_thres,
            return_annotated=bool(save_annotated_dir)
        )
        
        if task == 'color': color_results.append(r)
        else: defect_results.append(r)

        # Save annotated image if requested
        if save_annotated_dir and r.get('annotated_img') is not None:
            out_name = f"{stem}_annotated.jpg"
            out_path = os.path.join(save_annotated_dir, out_name)
            cv2.imwrite(out_path, r['annotated_img'], [cv2.IMWRITE_JPEG_QUALITY, 85])
            
            batch_metadata.append({
                "filename":  img_path.name,
                "n_wrong":   r.get("n_wrong", 0),
                "image_url": out_name # Relative to session folder
            })

        if progress_callback:
            progress_callback(idx + 1, len(img_files))

    total_time = time.time() - t_start
    n_ok = len(img_files) - skipped

    result = {
        "summary": {
            "n_images":               len(img_files),
            "n_processed":            n_ok,
            "n_skipped":              skipped,
            "total_time_sec":         round(total_time, 2),
            "avg_time_per_image_sec": round(total_time / n_ok, 3) if n_ok > 0 else 0,
            "fps":                    round(n_ok / total_time, 2) if total_time > 0 else 0,
            "task":                   task,
            "model_mode":             model_mode,
            "iou_thresh":             iou_thresh,
        },
        "batch_viz": batch_metadata
    }
    if color_results:
        result["color"] = compute_metrics(color_results, COLOR_CLASS_NAMES)
    if defect_results:
        result["defect"] = compute_metrics(defect_results, DEFECT_CLASS_NAMES)

    return result

# ─── Public API: Single Image ─────────────────────────────────────────────────
def run_single_image_benchmark(frame, color_label_path, defect_label_path,
                               task, model_mode,
                               model_yolo_seed,
                               model_convnext_color, model_convnext_defect,
                               detect_fn, batch_predict_fn,
                               iou_thresh=0.5, conf_thres=0.45,
                               data_loader_sec=0.0):
    """
    Run benchmark on a single pre-loaded frame.
    data_loader_sec : time the caller spent decoding the image (step 1).

    Extra keys in result:
        annotated_color_b64 / annotated_defect_b64  : full annotation
        mismatch_color_b64  / mismatch_defect_b64   : wrong-predictions only
        timing_color / timing_defect                : 5-step timing dict
        n_wrong_color / n_wrong_defect              : # misclassified seeds
    """
    t_start = time.time()
    result  = {}

    def _enc(img):
        ok, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 88])
        return base64.b64encode(buf).decode('utf-8') if ok else None

    def _timing5(t):
        return {
            "data_loader_sec":     round(data_loader_sec, 4),
            "object_detector_sec": t.get("object_detector_sec", 0),
            "image_seg_sec":       t.get("image_seg_sec", 0),
            "classification_sec":  t.get("classification_sec", 0),
            "postprocess_sec":     t.get("postprocess_sec", 0),
            "total_sec":           t.get("total_sec", 0),
        }

    if task in ('color', 'both') and color_label_path:
        r = _eval_one_image(frame, color_label_path, COLOR_CLASS_NAMES,
                            model_yolo_seed, model_convnext_color,
                            detect_fn, batch_predict_fn, model_mode, iou_thresh, conf_thres,
                            return_annotated=True)
        result["color"]         = compute_metrics([r], COLOR_CLASS_NAMES)
        result["timing_color"]  = _timing5(r.get("timing", {}))
        result["n_wrong_color"] = r.get("n_wrong", 0)
        ann  = r.get("annotated_img")
        mism = r.get("mismatch_img")
        if ann  is not None: result["annotated_color_b64"] = _enc(ann)
        if mism is not None: result["mismatch_color_b64"]  = _enc(mism)
        
        # New: Encode individual crops for gallery
        crop_list = []
        for c in r.get("crops_data", []):
            crop_list.append({
                "b64": _enc(c["img"]),
                "gt": c["gt_name"],
                "pred": c["pred_name"],
                "conf": round(c["conf"], 1),
                "correct": c["is_correct"],
                "type": c["type"]
            })
        result["crops_color"] = crop_list

    if task in ('defect', 'both') and defect_label_path:
        r = _eval_one_image(frame, defect_label_path, DEFECT_CLASS_NAMES,
                            model_yolo_seed, model_convnext_defect,
                            detect_fn, batch_predict_fn, model_mode, iou_thresh, conf_thres,
                            return_annotated=True)
        result["defect"]         = compute_metrics([r], DEFECT_CLASS_NAMES)
        result["timing_defect"]  = _timing5(r.get("timing", {}))
        result["n_wrong_defect"] = r.get("n_wrong", 0)
        ann  = r.get("annotated_img")
        mism = r.get("mismatch_img")
        if ann  is not None: result["annotated_defect_b64"] = _enc(ann)
        if mism is not None: result["mismatch_defect_b64"]  = _enc(mism)
        
        # New: Encode individual crops for gallery
        crop_list = []
        for c in r.get("crops_data", []):
            crop_list.append({
                "b64": _enc(c["img"]),
                "gt": c["gt_name"],
                "pred": c["pred_name"],
                "conf": round(c["conf"], 1),
                "correct": c["is_correct"],
                "type": c["type"]
            })
        result["crops_defect"] = crop_list

    total_time = time.time() - t_start
    result["summary"] = {
        "n_images":       1,
        "n_processed":    1,
        "n_skipped":      0,
        "total_time_sec": round(total_time, 2),
        "fps":            round(1.0 / total_time, 2) if total_time > 0 else 0,
        "task":           task,
        "model_mode":     model_mode,
        "iou_thresh":     iou_thresh,
    }
    return result

# ─── ZIP Helper ───────────────────────────────────────────────────────────────
def extract_zip_to_temp(zip_bytes):
    """
    Extract ZIP bytes to a temp directory.
    Searches for 'images' and 'labels' subdirectories (case-insensitive, recursive).
    Returns (tmp_dir, images_dir, labels_dir).  Caller must clean up tmp_dir.
    Raises ValueError if structure is not valid.
    """
    tmp_dir = tempfile.mkdtemp(prefix='cocoa_bench_')
    zip_path = os.path.join(tmp_dir, 'upload.zip')
    with open(zip_path, 'wb') as f:
        f.write(zip_bytes)

    extract_dir = os.path.join(tmp_dir, 'extracted')
    os.makedirs(extract_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(extract_dir)

    images_dir = labels_dir = None
    for root, dirs, _files in os.walk(extract_dir):
        bn = os.path.basename(root).lower()
        if bn == 'images' and images_dir is None:
            images_dir = root
        if bn == 'labels' and labels_dir is None:
            labels_dir = root
        if images_dir and labels_dir:
            break

    if not images_dir or not labels_dir:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise ValueError("ไม่พบโฟลเดอร์ 'images/' หรือ 'labels/' ใน ZIP file")

    return tmp_dir, images_dir, labels_dir
