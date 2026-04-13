# 주행 환경 차량용 신호등 인식 프로젝트

주행 환경에서 카메라 기반으로 차량용 신호등 상태를 인식하기 위한 객체 탐지 프로젝트입니다.  
본 저장소는 대회 기간 동안 수행한 데이터 전처리, 모델 학습, 성능 검증, 앙상블 실험, ONNX 변환까지의 전체 흐름과 구현 자산을 정리한 문서이며, 이력서에 기재한 성과와 실제 프로젝트 경험을 뒷받침하는 근거 자료로 활용할 수 있도록 구성했습니다.

## 프로젝트 개요

- 주제: 주행 환경에서 카메라를 통한 차량용 신호등 인식 모델 개발
- 형태: 4인 팀 프로젝트
- 기간: 2024.10 - 2024.11
- 총 진행 기간: 약 3주
- 결과: `mAP50 0.6533 -> 0.7362`, `+8.29%p` 향상
- 수상: `카카오모빌리티 대표상`

이 프로젝트의 핵심 목표는 차량용 신호등 14개 클래스를 안정적으로 인식하는 탐지 모델을 개발하고, 단일 모델 성능 한계를 넘기 위해 데이터 전략과 앙상블 전략을 함께 설계하는 것이었습니다.


## 나의 기여

이 저장소 기준으로 제가 기여한 핵심 내용은 아래 두 축으로 설명할 수 있습니다.

### 1. 데이터 불균형 해소 및 학습 안정성 확보

- 소수 클래스 성능 저하 문제를 분석
- 일부 클래스에 대해 `오버샘플링` 전략을 적용
- 실험 로그와 결과를 비교하며 학습 안정성을 검증

신호등 데이터는 클래스별 표본 수 차이가 크고, 특히 일부 `noSign`, `warning` 계열 클래스의 성능 편차가 컸습니다.  
이 문제를 줄이기 위해 단순 학습 반복이 아니라 데이터 분포 자체를 보완하는 방향으로 접근했습니다.

### 2. Class-wise Ensemble 기법 적용

- 모델별 강한 클래스와 약한 클래스를 비교 분석
- 클래스별 best detector를 조합하는 `Class-wise Ensemble` 관점으로 실험 설계
- 여러 detector 결과를 `NMS`, `NMW`, `WBF` 방식으로 조합하며 최적 조합 탐색

대회 후반부에는 단일 모델을 더 미세 조정하는 것보다, 클래스별로 강점을 가진 모델들을 조합하는 편이 더 큰 성능 향상을 만든다고 판단했습니다.  
실제 export 자료에도 클래스별 최고 성능 모델이 다르게 나타나며, 이를 기반으로 앙상블 전략을 정교화했습니다.

위 기여 내용은 단순 서술이 아니라, 저장소 내 실험표, export PDF, Todo 보드, legacy 실험 코드, 설정 파일을 통해 흐름을 추적할 수 있습니다.

## 성능 개선 성과

프로젝트 시작 시 기준 성능은 `mAP50 0.6533` 수준이었고, 최종적으로 `0.7362`까지 끌어올렸습니다.

- Baseline: `0.6533`
- Final: `0.7362`
- Improvement: `+8.29%p`

Notion export 및 실험 정리 자료 기준으로, 후반 실험에서는 `Co-DETR`, `Cascade R-CNN`, `DINO` 계열 모델을 조합한 앙상블 결과가 다수 정리되어 있으며, 검증 성능 `0.7393` 수준의 조합도 확인됩니다.  
이 수치는 최종 방향 설정 과정에서 단일 모델 개선뿐 아니라 앙상블 기반 접근이 유효했음을 보여줍니다.

## 문제 배경

이 프로젝트는 일반적인 객체 탐지보다 더 까다로운 조건을 다룹니다.

- 신호등 객체 자체가 작고 원거리에서 관측됨
- 유사한 클래스 간 구분이 어려움
- `go`, `stop`, `warning`, `noSign` 등 상태 기반 세분화가 필요함
- 클래스 불균형이 심하고 일부 클래스는 샘플 수가 매우 적음
- 실제 주행 환경 특성상 조명, 거리, 시야각, 날씨 변화의 영향이 큼

따라서 단순히 하나의 모델을 학습시키는 것으로는 충분하지 않았고, 데이터 전략과 앙상블 전략을 함께 설계해야 했습니다.

## 저장소 구성

이 저장소에는 크게 두 종류의 자산이 포함되어 있습니다.

### 1. 메인 파이프라인 코드

현재 기준의 메인 파이프라인은 `YOLO` 기반으로 정리되어 있습니다.

- 데이터 전처리
- fold 분할
- 학습
- PyTorch 검증
- ONNX export
- ONNX Runtime 검증

관련 디렉터리:

- `scripts/`
- `configs/`
- `models/`

### 2. 실험 및 대회 운영 히스토리

대회 기간 동안 사용한 실험 정리 자료와 export 자산도 함께 존재합니다.

- Todo 관리 자료
- 대회 개요 자료
- 앙상블 결과표
- 실험 자산 및 pretrained weight 정리 문서
- `legacy/` 하위 detector 실험 코드

즉, 이 저장소는 단순 코드 업로드가 아니라 실제 프로젝트 수행 과정을 검증할 수 있도록 남겨둔 아카이브 성격도 갖고 있습니다.

## 실험 근거 자료

사용자가 제공한 Notion export 자료 기준으로 다음 사실을 확인할 수 있습니다.

즉, 아래 자료들은 README에 적힌 프로젝트 설명뿐 아니라 이력서에 기재한 실험 수행 경험과 성능 개선 과정을 뒷받침하는 정리 자료입니다.

### 대회 및 일정 자료

- 자율주행 인공지능 챌린지 관련 PDF 포함
- 팀 단위 Todo 보드 포함
- 제출 일정 및 운영 링크 정리

이 자료를 통해 실제 대회형 프로젝트였다는 점, 팀 단위로 일정과 실험을 운영했다는 점을 확인할 수 있습니다.

### 앙상블 실험 자료

- `앙상블 Table` CSV/PDF 존재
- 실험 조합:
  - `Co-DETR(Obj365 1ep/2ep/3ep)`
  - `CascadeRCNN(SwinL 2048)`
  - `DINO`
- 앙상블 방식:
  - `NMS`
  - `NMW`
  - `WBF`
- threshold 및 weight 조합별 성능 비교 기록 존재

이 자료는 앙상블을 단순 아이디어 수준이 아니라 실제 조합 단위로 반복 실험했다는 근거가 됩니다.

### 클래스별 성능 분석 자료

PDF 내 클래스별 best 모델 요약이 포함되어 있습니다. 예를 들면:

- `veh_go`: `Co-DETR(Obj365, 2ep)`가 best
- `veh_stop`: `Co-DETR(Obj365, 3ep)`가 best
- `ped_stop`: `Co-DETR(Obj365, 2ep/3ep)`가 best
- `ped_noSign`, `bus_warning`, `bus_noSign` 계열은 상대적으로 어려운 클래스

이 자료는 제가 이력서에 적은 `Class-wise Ensemble` 기여를 뒷받침하는 정황 증거로 활용할 수 있습니다.  
즉, 클래스별 강점을 가진 모델이 다르다는 분석이 실제 실험 자료에도 드러나기 때문입니다.

또한 일부 난이도 높은 클래스에서 성능 편차가 큰 점은, 왜 데이터 불균형 대응과 클래스별 전략이 필요했는지를 설명하는 근거로도 사용할 수 있습니다.

## 기술 구현 흐름

현재 코드 기준의 메인 워크플로우는 아래와 같습니다.

### 1. 데이터 전처리

`scripts/pre_processing/`에서는 다음 작업을 수행합니다.

- 원본 JSON 어노테이션 정제
- 특정 구간 라벨 수정/삭제
- YOLO 포맷 라벨 생성
- `MultilabelStratifiedKFold` 기반 5-fold 분할
- 클래스 분포 리포트 출력

주요 파일:

- [scripts/pre_processing/data_cleaning.py](C:/Users/SSAFY/Desktop/ws/refactor/realtime-cctv-object-detection/scripts/pre_processing/data_cleaning.py:1)
- [scripts/pre_processing/split_data_vYOLO.py](C:/Users/SSAFY/Desktop/ws/refactor/realtime-cctv-object-detection/scripts/pre_processing/split_data_vYOLO.py:1)
- [scripts/pre_processing/print_distribution_fold.py](C:/Users/SSAFY/Desktop/ws/refactor/realtime-cctv-object-detection/scripts/pre_processing/print_distribution_fold.py:1)

이 중 `split_data_vYOLO.py`는 클래스 분포를 고려한 분할을 적용하고 있어, 데이터 불균형을 의식한 프로젝트라는 점을 보여줍니다.

### 2. 학습

[scripts/train/train.py](C:/Users/SSAFY/Desktop/ws/refactor/realtime-cctv-object-detection/scripts/train/train.py:1)는 [configs/train_yolo11s.yaml](C:/Users/SSAFY/Desktop/ws/refactor/realtime-cctv-object-detection/configs/train_yolo11s.yaml:1)을 읽어 학습을 수행합니다.

설정 항목:

- 모델 구조
- pretrained weights
- dataset yaml 경로
- epoch
- batch size
- image size
- device

### 3. 날씨 증강 확장

[scripts/train/trainers/weather_trainer.py](C:/Users/SSAFY/Desktop/ws/refactor/realtime-cctv-object-detection/scripts/train/trainers/weather_trainer.py:1)와 [scripts/train/datasets/weather_dataset.py](C:/Users/SSAFY/Desktop/ws/refactor/realtime-cctv-object-detection/scripts/train/datasets/weather_dataset.py:1)를 통해 날씨 기반 증강도 확장할 수 있습니다.

적용 가능한 증강:

- `RandomSnow`
- `RandomFog`
- `RandomRain`
- `RandomShadow`

이 부분은 실제 도로 환경 데이터의 도메인 특성을 고려한 확장 포인트로 해석할 수 있습니다.

### 4. 검증 및 배포 관점 평가

검증은 두 단계로 구성됩니다.

- [scripts/inference/inference_val.py](C:/Users/SSAFY/Desktop/ws/refactor/realtime-cctv-object-detection/scripts/inference/inference_val.py:1)
  - PyTorch `.pt` 모델 검증
- [scripts/inference/inference_val_onnx.py](C:/Users/SSAFY/Desktop/ws/refactor/realtime-cctv-object-detection/scripts/inference/inference_val_onnx.py:1)
  - ONNX Runtime 기반 검증

이 구조는 학습 성능만 보는 것이 아니라, 실제 배포 포맷 기준 성능과 추론 시간까지 확인하려는 목적을 보여줍니다.

### 5. ONNX 변환

[scripts/export/export_onnx.py](C:/Users/SSAFY/Desktop/ws/refactor/realtime-cctv-object-detection/scripts/export/export_onnx.py:1)는 학습된 모델을 ONNX로 변환합니다.

지원 옵션:

- dynamic shape
- simplify
- FP16/FP32
- NMS 포함 여부

즉, 연구용 코드에서 끝나지 않고 배포 경로까지 고려한 구성을 갖추고 있습니다.

## 저장소 구조

```text
realtime-cctv-object-detection/
├─ configs/
├─ scripts/
│  ├─ pre_processing/
│  ├─ train/
│  ├─ inference/
│  └─ export/
├─ models/
├─ legacy/
├─ environment.yml
└─ README_n.md
```

## 주요 엔지니어링 포인트

- 전처리 규칙을 코드화해 데이터 정제 이력을 남김
- 클래스 분포를 고려한 fold 분할 적용
- 학습/추론/배포 설정을 YAML로 분리
- 다중 모델 실험 결과를 CSV/PDF 형태로 보관
- ONNX 추론 검증 루틴 제공
- 단일 모델이 아닌 클래스별 강점 분석 기반의 앙상블 전략 수립

이 포인트들은 단순 구현 나열이 아니라, 실제로 프로젝트에서 어떤 문제를 인식했고 어떤 방식으로 해결했는지를 보여주는 경험 근거이기도 합니다.

## 실행 방법

### 환경 구성

```bash
conda env create -f environment.yml
conda activate RCOD
```

### 전처리

```bash
cd scripts/pre_processing
bash pre_processing.sh
```

### 학습

```bash
cd scripts/train
python train.py
```

### PyTorch 검증

```bash
cd scripts/inference
python inference_val.py
```

### ONNX 변환

```bash
cd scripts/export
python export_onnx.py
```

### ONNX 검증

```bash
cd scripts/inference
python inference_val_onnx.py
```

## 실행 환경 및 의존성

[environment.yml](C:/Users/SSAFY/Desktop/ws/refactor/realtime-cctv-object-detection/environment.yml:1)에 전체 환경이 정의되어 있습니다.

주요 라이브러리:

- `torch`
- `ultralytics`
- `opencv-python`
- `onnx`
- `onnxruntime-gpu`
- `albumentations`
- `pandas`
- `numpy`
- `iterative-stratification`

## 참고 사항

- 현재 저장소의 메인 실행 경로는 YOLO 파이프라인 기준으로 정리되어 있습니다.
- 과거 대회 실험 코드와 현재 코드가 완전히 하나의 구조로 통합된 것은 아닙니다.
- 데이터 및 일부 가중치는 저장소 외부 상대경로를 가정하고 있어, 코드만으로 즉시 재현되지는 않을 수 있습니다.
- 그러나 실험 자산, 결과표, 설정 파일, 전처리 코드가 함께 남아 있어 프로젝트의 실제 수행 과정을 확인하는 데에는 충분한 자료가 됩니다.

채용 담당자나 리뷰어 관점에서는 이 저장소를 "완전한 재현용 오픈소스"라기보다, 프로젝트 수행 과정과 기여 내용을 확인할 수 있는 포트폴리오형 근거 저장소로 보는 것이 적절합니다.

## 이력서 검증 포인트

이 저장소는 아래 이력서 문장을 검증하기 위한 프로젝트 자료로 활용할 수 있습니다.

- 주제: 주행 환경에서 카메라를 통한 차량용 신호등 인식 모델 개발 대회
- 기간: 2024.10 - 2024.11
- 형태: 4인 팀 프로젝트
- 성과: `mAP50 0.6533 -> 0.7362`, `+8.29%p`
- 수상: `카카오모빌리티 대표상`
- 기여:
  - 데이터 불균형 해소를 위한 오버샘플링 적용
  - 클래스별 강점 분석 기반 `Class-wise Ensemble` 기법 고안 및 적용

위 항목들은 다음 자료와 함께 교차 확인할 수 있습니다.

- 전처리 및 학습/검증 코드
- 설정 파일과 실험용 경로 구성
- `ExportBlock-*` 내부 Todo, 앙상블 표, 대회 정리 PDF
- `r2.pdf`에 정리된 실험 자산 및 운영 문서
- `legacy/` 내부의 과거 detector 실험 흔적

즉, 본 저장소는 단순 구현 코드가 아니라 문제 정의, 실험 설계, 성능 개선, 결과 정리까지 포함한 실제 프로젝트 경험의 근거 자료입니다.
