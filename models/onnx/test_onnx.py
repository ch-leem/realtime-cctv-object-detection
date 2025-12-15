import onnx
import onnxruntime as ort

path = "./yolo11s_FP32.onnx"

model = onnx.load(path)
onnx.checker.check_model(model)
print("ONNX model is valid.")

print("Inputs:")
for i in model.graph.input:
    print("   " + i.name)

print("\nOutputs:")
for o in model.graph.output:
    print("   " + o.name)

# https://netron.app <- 사이트 접속 후 onnx 그래프 확인

# GPU/CPU 추론 가능 여부 확인
sess = ort.InferenceSession(
    path,
    providers=['CUDAExecutionProvider'] # 'CPUExecutionProvider'
)

print(sess.get_inputs())
print(sess.get_outputs())
