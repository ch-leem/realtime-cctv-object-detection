import os
import json
import time
import yaml
import cv2
import numpy as np
import onnxruntime as ort
from collections import Counter

# ===============================
# Utils
# ===============================
def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[0] + box1[2], box2[0] + box2[2])
    y2 = min(box1[1] + box1[3], box2[1] + box2[3])
    w = max(0, x2 - x1)
    h = max(0, y2 - y1)
    inter = w * h
    union = box1[2] * box1[3] + box2[2] * box2[3] - inter
    return inter / union if union > 0 else 0

def nms(boxes, scores, iou_thres):
    idxs = scores.argsort()[::-1]
    keep = []

    while idxs.size > 0:
        i = idxs[0]
        keep.append(i)
        if idxs.size == 1:
            break

        ious = np.array([
            compute_iou(boxes[i], boxes[j]) for j in idxs[1:]
        ])
        idxs = idxs[1:][ious < iou_thres]

    return keep

# ===============================
# Main
# ===============================
def main():
    cfg = load_yaml("../../configs/inference_onnx_yolo11s.yaml")

    onnx_dir = cfg["model"]["onnx_dir"]
    device = cfg["model"]["device"]

    test_dir = cfg["data"]["test_dir"]
    gt_json_path = cfg["data"]["gt_json"]
    img_w = cfg["data"]["img_width"]
    img_h = cfg["data"]["img_height"]
    names = cfg["data"]["names"]
    num_classes = len(names)

    imgsz = cfg["inference"]["imgsz"]
    conf_thres = cfg["inference"]["conf_thres"]
    iou_thres = cfg["inference"]["iou_thres"]

    IOUThreshold = cfg["evaluation"]["iou_threshold"]
    output_dir = cfg["output"]["output_dir"]
    ap_log_path = os.path.join(output_dir, cfg["evaluation"]["ap_log"])

    os.makedirs(output_dir, exist_ok=True)

    # ===============================
    # GT Load
    # ===============================
    with open(gt_json_path, "r") as f:
        gt_raw = json.load(f)

    gt_list = []
    for g in gt_raw:
        for ann in g["annotations"]:
            gt_list.append([
                g["images"]["img_id"],
                ann["lbl_nm"],
                eval(ann["annotations_info"])
            ])

    ap_file = open(ap_log_path, "w", encoding="utf-8")

    onnx_files = [f for f in os.listdir(onnx_dir) if f.endswith(".onnx")]
    results_summary = []

    scale_x = img_w / imgsz
    scale_y = img_h / imgsz

    for onnx_file in onnx_files:
        print(f"\n=== ONNX Inference: {onnx_file} ===")

        sess = ort.InferenceSession(
            os.path.join(onnx_dir, onnx_file),
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
            if device == "cuda" else ["CPUExecutionProvider"]
        )
        input_name = sess.get_inputs()[0].name

        output_json = []
        ann_id = 0
        start_time = time.time()

        for img_file in sorted(os.listdir(test_dir)):
            if not img_file.lower().endswith(".jpg"):
                continue

            img_id = os.path.splitext(img_file)[0]
            img_path = os.path.join(test_dir, img_file)

            img = cv2.imread(img_path)
            if img is None:
                continue

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (imgsz, imgsz))
            img = img.astype(np.float32) / 255.0
            img = np.transpose(img, (2, 0, 1))
            img = np.expand_dims(img, axis=0)

            preds = sess.run(None, {input_name: img})[0]
            preds = preds[0].T  # (N, 12)

            boxes = preds[:, :4]
            cls_scores = preds[:, 4:]

            cls_ids = np.argmax(cls_scores, axis=1)
            confs = np.max(cls_scores, axis=1)

            keep = confs > conf_thres
            boxes = boxes[keep]
            cls_ids = cls_ids[keep]
            confs = confs[keep]

            final_boxes, final_scores, final_cls = [], [], []

            for c in range(num_classes):
                idxs = np.where(cls_ids == c)[0]
                if len(idxs) == 0:
                    continue

                keep_idxs = nms(boxes[idxs], confs[idxs], iou_thres)
                for k in keep_idxs:
                    final_boxes.append(boxes[idxs][k])
                    final_scores.append(confs[idxs][k])
                    final_cls.append(c)

            image_info = {
                "images": {
                    "img_id": img_id,
                    "img_width": img_w,
                    "img_height": img_h
                },
                "annotations": []
            }

            for i in range(len(final_boxes)):
                x, y, w, h = final_boxes[i]

                x *= scale_x
                w *= scale_x
                y *= scale_y
                h *= scale_y

                image_info["annotations"].append({
                    "annotations_id": f"{img_id}_{ann_id}",
                    "lbl_id": final_cls[i],
                    "lbl_nm": names[final_cls[i]],
                    "annotations_info": f"[{x:.2f}, {y:.2f}, {w:.2f}, {h:.2f}]",
                    "confidence": float(final_scores[i])
                })
                ann_id += 1

            output_json.append(image_info)

        elapsed = time.time() - start_time
        print(f"Inference time: {elapsed:.2f}s")

        with open(
            os.path.join(output_dir, f"{os.path.splitext(onnx_file)[0]}.json"),
            "w",
            encoding="utf-8"
        ) as f:
            json.dump(output_json, f, indent=4, ensure_ascii=False)

        # ===============================
        # mAP
        # ===============================
        pr_list = []
        for img in output_json:
            for ann in img["annotations"]:
                pr_list.append([
                    img["images"]["img_id"],
                    ann["lbl_nm"],
                    eval(ann["annotations_info"]),
                    ann["confidence"]
                ])

        pr_list.sort(key=lambda x: x[3], reverse=True)

        ap = 0
        for cls_name in names:
            pr_c = [p for p in pr_list if p[1] == cls_name]
            gt_c = [g for g in gt_list if g[1] == cls_name]

            npos = len(gt_c)
            tp = np.zeros(len(pr_c))
            fp = np.zeros(len(pr_c))

            det = Counter([g[0] for g in gt_c])
            for k in det:
                det[k] = np.zeros(det[k])

            for i, p in enumerate(pr_c):
                gts = [g for g in gt_c if g[0] == p[0]]
                iou_max, jmax = 0, -1
                for j, g in enumerate(gts):
                    iou = compute_iou(p[2], g[2])
                    if iou > iou_max:
                        iou_max, jmax = iou, j

                if iou_max >= IOUThreshold and jmax >= 0 and det[p[0]][jmax] == 0:
                    tp[i] = 1
                    det[p[0]][jmax] = 1
                else:
                    fp[i] = 1

            acc_tp = np.cumsum(tp)
            acc_fp = np.cumsum(fp)
            rec = acc_tp / max(npos, 1)
            prec = acc_tp / np.maximum(acc_tp + acc_fp, 1e-6)

            recall_levels = np.linspace(0, 1, 11)[::-1]
            ap_cls = sum(
                np.max(prec[rec >= r]) if np.any(rec >= r) else 0
                for r in recall_levels
            ) / 11

            ap += ap_cls
            print(f"Class {cls_name} AP: {ap_cls:.4f}")
            ap_file.write(f"{onnx_file} {cls_name} AP: {ap_cls:.4f}\n")

        mAP = ap / num_classes
        print(f"mAP: {mAP:.4f}")
        results_summary.append({"model": onnx_file, "time": elapsed, "mAP": mAP})

    ap_file.close()

    print("\n=== Summary ===")
    for r in results_summary:
        print(f"{r['model']} | time {r['time']:.2f}s | mAP {r['mAP']:.4f}")

if __name__ == "__main__":
    main()