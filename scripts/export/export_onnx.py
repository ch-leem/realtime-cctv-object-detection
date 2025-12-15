import yaml
import shutil
from pathlib import Path
from ultralytics import YOLO

# ===============================
# Utils
# ===============================
def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

# ===============================
# Main
# ===============================
def main():
    cfg = load_yaml("../../configs/export_onnx.yaml")

    # -------------------------------
    # Model config
    # -------------------------------
    weights = Path(cfg["model"]["weights"]).resolve()
    device = cfg["model"]["device"]

    weight_dir = weights.parent
    weight_stem = weights.stem

    # -------------------------------
    # Export config
    # -------------------------------
    export_cfg = cfg["export"]
    imgsz = export_cfg["imgsz"]
    opset = export_cfg["opset"]
    dynamic = export_cfg["dynamic"]
    simplify = export_cfg["simplify"]
    half = export_cfg["half"]
    nms = export_cfg["nms"]
    output_dir = Path(export_cfg["output_dir"]).resolve()

    onnx_type = "FP16" if half else "FP32"
    export_name = f"yolo11s_{onnx_type}" if not nms else f"yolo11s_{onnx_type}_nms"

    print("===================================")
    print(" ONNX Export Config")
    print(f"  - weights : {weights}")
    print(f"  - device  : {device}")
    print(f"  - imgsz   : {imgsz}")
    print(f"  - opset   : {opset}")
    print(f"  - dynamic : {dynamic}")
    print(f"  - simplify: {simplify}")
    print(f"  - half    : {half} ({onnx_type})")
    print("===================================")

    # -------------------------------
    # Export ONNX
    # -------------------------------
    print("Exporting ONNX...")
    model = YOLO(str(weights))
    model.export(
        format="onnx",
        imgsz=imgsz,
        opset=opset,
        dynamic=dynamic,
        simplify=simplify,
        half=half,
        device=device,
        nms = nms
    )

    # -------------------------------
    # Move ONNX (weight_dir 기준)
    # -------------------------------
    src_onnx = weight_dir / f"{weight_stem}.onnx"
    dst_onnx = output_dir / f"{export_name}.onnx"

    if not src_onnx.exists():
        raise FileNotFoundError(f"ONNX file not found: {src_onnx}")

    shutil.move(src_onnx, dst_onnx)
    print(f"ONNX moved → {dst_onnx}")

    print("ONNX export completed successfully.")

# ===============================
# Entry
# ===============================
if __name__ == "__main__":
    main()