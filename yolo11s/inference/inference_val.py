import os
import json
import time
import yaml
import numpy as np
from ultralytics import YOLO
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

# ===============================
# Main
# ===============================
def main():
    cfg = load_yaml("./configs/inference_yolo11s.yaml")

    weights_dir = cfg["model"]["weights_dir"]
    device = cfg["model"]["device"]
    half = cfg["model"]["half"]

    test_dir = cfg["data"]["test_dir"]
    gt_json_path = cfg["data"]["gt_json"]
    img_w = cfg["data"]["img_width"]
    img_h = cfg["data"]["img_height"]

    imgsz = cfg["inference"]["imgsz"]
    conf_thres = cfg["inference"]["conf_thres"]
    iou_thres = cfg["inference"]["iou_thres"]
    stream = cfg["inference"]["stream"]

    IOUThreshold = cfg["evaluation"]["iou_threshold"]
    output_dir = cfg["output"]["output_dir"]
    ap_log_path = output_dir + '/' + cfg["evaluation"]["ap_log"]

    os.makedirs(output_dir, exist_ok=True)

    # 클래스 매핑
    class_mapping = {
        1: "bus",
        2: "car",
        3: "sign",
        4: "truck",
        5: "human",
        6: "special_vehicles",
        7: "taxi",
        8: "motorcycle"
    }

    # GT 로드
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

    model_files = [f for f in os.listdir(weights_dir) if f.endswith(".pt")]
    results_summary = []

    for model_file in model_files:
        print(f"\n=== Inference with {model_file} ===")
        weight_path = os.path.join(weights_dir, model_file)
        output_json_path = os.path.join(output_dir, f"{os.path.splitext(model_file)[0]}.json")

        model = YOLO(weight_path)

        start_time = time.time()
        results = model.predict(
            source=test_dir,
            imgsz=imgsz,
            conf=conf_thres,
            iou=iou_thres,
            stream=stream,
            device=device,
            half=half
        )

        output_json = []
        ann_id = 0

        for r in results:
            img_id = os.path.splitext(os.path.basename(r.path))[0]
            image_info = {
                "images": {
                    "img_id": img_id,
                    "img_width": img_w,
                    "img_height": img_h
                },
                "annotations": []
            }

            for box in r.boxes:
                cls = int(box.cls.item()) + 1
                conf = float(box.conf.item())
                x, y, w, h = box.xywh[0].tolist()
                x_min = x - w / 2
                y_min = y - h / 2

                image_info["annotations"].append({
                    "annotations_id": f"{img_id}_{ann_id}",
                    "lbl_id": cls,
                    "lbl_nm": class_mapping[cls],
                    "annotations_info": f"[{x_min:.2f}, {y_min:.2f}, {w:.2f}, {h:.2f}]",
                    "confidence": conf
                })
                ann_id += 1

            output_json.append(image_info)

        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(output_json, f, indent=4, ensure_ascii=False)

        elapsed = time.time() - start_time
        print(f"Inference time: {elapsed:.2f}s")

        # ===============================
        # mAP 계산
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
        for cls_name in class_mapping.values():
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
                iou_max = 0
                jmax = -1
                for j, g in enumerate(gts):
                    iou = compute_iou(p[2], g[2])
                    if iou > iou_max:
                        iou_max = iou
                        jmax = j

                if iou_max >= IOUThreshold and det[p[0]][jmax] == 0:
                    tp[i] = 1
                    det[p[0]][jmax] = 1
                else:
                    fp[i] = 1

            acc_tp = np.cumsum(tp)
            acc_fp = np.cumsum(fp)
            rec = acc_tp / max(npos, 1)
            prec = acc_tp / np.maximum(acc_tp + acc_fp, np.finfo(np.float64).eps)

            recall_levels = np.linspace(0, 1, 11)[::-1]
            ap_cls = 0
            for r in recall_levels:
                pmax = np.max(prec[rec >= r]) if np.any(rec >= r) else 0
                ap_cls += pmax
            ap_cls /= 11

            ap += ap_cls
            print(f"Class {cls_name} AP: {ap_cls:.4f}")
            ap_file.write(f"{model_file} {cls_name} AP: {ap_cls:.4f}\n")

        mAP = ap / len(class_mapping)
        print(f"mAP: {mAP:.4f}")

        results_summary.append({
            "model": model_file,
            "time": elapsed,
            "mAP": mAP
        })

    ap_file.close()

    print("\n=== Summary ===")
    for r in results_summary:
        print(f"{r['model']} | time {r['time']:.2f}s | mAP {r['mAP']:.4f}")

if __name__ == "__main__":
    main()