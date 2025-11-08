# FOV 기반 Viewpoint 샘플링 가이드

## 📋 목차
1. [개요](#개요)
2. [주요 기능](#주요-기능)
3. [카메라 스펙](#카메라-스펙)
4. [사용 방법](#사용-방법)
5. [출력 형식](#출력-형식)
6. [워크플로우](#워크플로우)
7. [예시](#예시)

---

## 개요

### 목적
카메라 FOV(Field of View) 스펙에 맞춰 3D 메시 객체를 효율적으로 커버하는 viewpoint를 샘플링하는 도구입니다. 로봇 비전 검사 시스템에서 카메라 경로 계획에 사용됩니다.

### 핵심 특징
- **FOV 기반 샘플링**: 카메라 시야각(41×30mm)을 고려한 viewpoint 생성
- **Working Distance**: 정확한 초점 거리(110mm) 유지
- **Depth of Field 검증**: 피사계 심도(0.5mm) 제약 확인
- **Overlap 관리**: 인접 view 간 25% 중첩으로 완전한 커버리지 보장
- **TSP 호환**: `mesh_to_tsp.py`와 호환되는 HDF5 형식 출력
- **커버리지 분석**: 표면 커버리지 통계 및 시각화

### 파일 정보
- **위치**: `/isaac-sim/curobo/vision_inspection/scripts/mesh_to_viewpoints.py`
- **라인 수**: ~650 lines
- **의존성**: Open3D, NumPy, Matplotlib, tsp_utils

---

## 주요 기능

### 1. CameraSpec 클래스
카메라 및 렌즈 사양 관리:

```python
@dataclass
class CameraSpec:
    sensor_width_px: int = 4096       # 센서 너비 (픽셀)
    sensor_height_px: int = 3000      # 센서 높이 (픽셀)
    pixel_size_um: float = 3.45       # 픽셀 크기 (μm)
    fov_width_mm: float = 41.0        # FOV 너비 (mm)
    fov_height_mm: float = 30.0       # FOV 높이 (mm)
    working_distance_mm: float = 110.0  # 작업 거리 (mm)
    depth_of_field_mm: float = 0.5    # 피사계 심도 (mm)
    overlap_ratio: float = 0.25       # 중첩 비율 (25%)
```

### 2. Viewpoint 생성 알고리즘

```
입력: 3D 메시 파일 (.obj)

처리 단계:
1. 메시 로드 및 법선 추정
2. Poisson disk sampling으로 표면 점 샘플링
3. 각 표면 점에 대해:
   - 법선 방향으로 WD(110mm) 오프셋하여 viewpoint 위치 계산
   - 카메라 방향: 법선의 반대 방향 (물체를 향함)
4. (선택) DOF 제약 검증:
   - 각 viewpoint에서 FOV 내 표면 depth variation 계산
   - 0.5mm 초과 시 경고

출력: Viewpoint 리스트 (위치 + 방향)
```

### 3. Depth of Field 검증

각 viewpoint에서 5×5 ray grid를 FOV 내에 샘플링하여 표면까지의 거리를 측정합니다:

```python
depth_variation = max_distance - min_distance

if depth_variation > DOF_limit:
    # 경고: 해당 viewpoint에서 일부 영역이 초점 밖
```

### 4. 커버리지 분석

- **표면 면적 계산**: 메시의 총 표면적
- **Viewpoint 커버리지**: 각 viewpoint가 커버하는 영역 (FOV × FOV)
- **커버리지 비율**: `(총 viewpoint 커버리지) / (메시 표면적)`
- **통계 출력**: 평균/최대 depth variation, DOF 위반 개수

---

## 카메라 스펙

### 기본 설정 (LG 비전 검사 시스템)

| 항목 | 값 | 단위 |
|------|-----|------|
| 센서 해상도 | 4096 × 3000 | pixel |
| 픽셀 크기 | 3.45 × 3.45 | μm |
| 광학 해상도 | 10 | μm |
| Working Distance (WD) | 110 | mm |
| Depth of Field (DOF) | 0.5 | mm |
| Field of View (FOV) | 41 × 30 | mm |
| Overlap 비율 | 25 | % |

### 실질 커버리지 계산

25% overlap을 고려한 실질적인 커버리지:

```
Effective Width  = FOV_width  × (1 - overlap) = 41.0 × 0.75 = 30.75 mm
Effective Height = FOV_height × (1 - overlap) = 30.0 × 0.75 = 22.50 mm
```

---

## 사용 방법

### 기본 사용

```bash
/isaac-sim/python.sh scripts/mesh_to_viewpoints.py \
  --mesh_file data/input/object/glass.obj \
  --num_points 500 \
  --save_path data/output/glass_fov_viewpoints.h5
```

### 전체 옵션

```bash
/isaac-sim/python.sh scripts/mesh_to_viewpoints.py \
  --mesh_file data/input/object/glass.obj \
  --num_points 500 \                    # 샘플링할 표면 점 개수
  --fov_width 41.0 \                    # FOV 너비 (mm)
  --fov_height 30.0 \                   # FOV 높이 (mm)
  --working_distance 110.0 \            # 작업 거리 (mm)
  --depth_of_field 0.5 \                # 피사계 심도 (mm)
  --overlap 0.25 \                      # 중첩 비율 (0-1)
  --check_dof \                         # DOF 제약 검증 활성화
  --remove_invalid_dof \                # DOF 위반 viewpoint 제거
  --save_path data/output/glass_fov_500.h5 \
  --plot \                              # 통계 그래프 저장
  --output data/output/stats.png \      # 그래프 출력 경로
  --visualize                           # Open3D 시각화 (GUI 필요)
```

### 커맨드라인 인터페이스

| 옵션 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `--mesh_file` | str | **필수** | 메시 파일 경로 (.obj) |
| `--save_path` | str | None | HDF5 출력 경로 |
| `--output` | str | viewpoint_stats.png | 통계 그래프 출력 경로 |
| `--fov_width` | float | 41.0 | FOV 너비 (mm) |
| `--fov_height` | float | 30.0 | FOV 높이 (mm) |
| `--working_distance` | float | 110.0 | 작업 거리 (mm) |
| `--depth_of_field` | float | 0.5 | 피사계 심도 (mm) |
| `--overlap` | float | 0.25 | 중첩 비율 (0-1) |
| `--num_points` | int | 1000 | 샘플링 점 개수 |
| `--check_dof` | flag | False | DOF 검증 활성화 |
| `--remove_invalid_dof` | flag | False | DOF 위반 제거 |
| `--visualize` | flag | False | Open3D 3D 시각화 |
| `--plot` | flag | False | Matplotlib 그래프 저장 |

---

## 출력 형식

### HDF5 Viewpoint 파일 구조 (간소화됨)

`mesh_to_viewpoints.py`에서 생성하는 간소화된 형식:

```
viewpoints.h5
│
├── metadata (group)
│   ├── num_viewpoints: int
│   ├── mesh_file: str
│   ├── timestamp: ISO datetime
│   ├── format: "viewpoints_only" (식별 마커)
│   └── camera_spec (group) - 카메라 스펙
│       ├── sensor_width_px: int
│       ├── sensor_height_px: int
│       ├── pixel_size_um: float
│       ├── fov_width_mm: float
│       ├── fov_height_mm: float
│       ├── working_distance_mm: float
│       ├── depth_of_field_mm: float
│       └── overlap_ratio: float
│
└── viewpoints (group)
    ├── positions: (N, 3) float32 - Viewpoint 좌표
    └── normals: (N, 3) float32 - 카메라 방향 벡터
```

**주요 변경점**:
- TSP tour 정보 제거 (tour는 mesh_to_tsp.py에서 계산)
- 정규화 정보 제거 (mesh_to_tsp.py에서 자동 처리)
- 카메라 스펙 메타데이터 추가
- 파일 크기 대폭 감소 (~60% 작음)

### 호환성

`mesh_to_tsp.py`는 두 가지 형식을 모두 로드 가능:
1. **간소화된 viewpoints** (`.h5` + `--use_viewpoints`)
2. **기존 메시/PCD** (`.obj`, `.pcd`)

### 통계 그래프

Matplotlib으로 생성되는 PNG 파일:

- **왼쪽 패널**:
  - Coverage ≤ 100%: 파이 차트 (커버/미커버)
  - Coverage > 100%: 막대 그래프 (중첩 표시)
- **오른쪽 패널**:
  - 카메라 스펙 요약
  - 샘플링 결과 통계
  - Depth variation 분석

---

## 워크플로우

### 방법 1: Viewpoint → TSP → 시뮬레이션 (권장)

```bash
# Step 1: FOV 기반 viewpoint 생성
/isaac-sim/python.sh scripts/mesh_to_viewpoints.py \
  --mesh_file data/input/object/glass.obj \
  --num_points 500 \
  --check_dof \
  --save_path data/output/glass_fov_500.h5 \
  --plot
```

**출력**:
- `data/output/glass_fov_500.h5` - Viewpoint 데이터 (간소화된 HDF5 형식)
- `viewpoint_stats.png` - 통계 그래프

```bash
# Step 2: TSP 경로 최적화 (--use_viewpoints 플래그 사용)
/isaac-sim/python.sh scripts/mesh_to_tsp.py \
  --mesh_file data/output/glass_fov_500.h5 \
  --use_viewpoints \
  --algorithm both \
  --num_starts 20 \
  --max_2opt_iterations 100 \
  --save_path data/output/glass_fov_500_tsp.h5 \
  --device cuda \
  --plot
```

**출력**:
- `data/output/glass_fov_500_tsp.h5` - TSP 최적화된 경로
- `tsp_tour_3d.png` - 경로 시각화

```bash
# Step 3: Isaac Sim에서 실행
/isaac-sim/python.sh scripts/run_app_v2.py \
  --headless_mode websocket \
  --robot robot_cfg/ur20.yml \
  --tour_file data/output/glass_fov_500_tsp.h5
```

### 방법 2: 기존 방식 (Poisson sampling → TSP)

```bash
# 기존 방식: mesh_to_tsp.py가 직접 샘플링
/isaac-sim/python.sh scripts/mesh_to_tsp.py \
  --mesh_file data/input/object/glass.obj \
  --num_points 500 \
  --algorithm both \
  --save_path data/output/glass_poisson_500_tsp.h5
```

**차이점**:
- 방법 1: FOV 고려한 최적 viewpoint 선택 (카메라 스펙 반영)
- 방법 2: 균일 샘플링 (단순히 표면을 균등하게 커버)

---

## 예시

### 예시 1: 기본 샘플링 (100 viewpoints)

```bash
/isaac-sim/python.sh scripts/mesh_to_viewpoints.py \
  --mesh_file data/input/object/glass.obj \
  --num_points 100 \
  --save_path data/output/glass_fov_100.h5
```

**출력**:
```
============================================================
FOV-based Viewpoint Sampling
============================================================
Camera Specifications:
  Sensor: 4096 x 3000 px
  Pixel size: 3.45 μm
  FOV: 41.0 x 30.0 mm
  Working Distance: 110.0 mm
  Depth of Field: 0.5 mm
  Overlap: 25.0%
  Effective coverage per view: 30.75 x 22.50 mm
============================================================
Loading mesh from: data/input/object/glass.obj
Loaded mesh: 461 vertices, 876 triangles
Surface area: 151379.20 mm²
Sampling 100 points using Poisson disk sampling...
Sampled 100 points

Computing viewpoints (WD = 110.0 mm)...
Generated 100 viewpoints

============================================================
RESULTS
============================================================
Number of viewpoints: 100
Mesh surface area: 151379.20 mm²
Total coverage: 123000.00 mm²
Coverage ratio: 81.3%
============================================================
```

### 예시 2: DOF 검증 활성화 (500 viewpoints)

```bash
/isaac-sim/python.sh scripts/mesh_to_viewpoints.py \
  --mesh_file data/input/object/glass.obj \
  --num_points 500 \
  --check_dof \
  --save_path data/output/glass_fov_500_dof.h5 \
  --plot
```

**출력**:
```
============================================================
RESULTS
============================================================
Number of viewpoints: 500
Mesh surface area: 151379.20 mm²
Total coverage: 615000.00 mm²
Coverage ratio: 406.3%
DOF violations: 259
Avg depth variation: 0.579 mm
Max depth variation: 1.365 mm
============================================================
```

**분석**:
- 259개 viewpoint (51.8%)가 DOF 제약(0.5mm) 위반
- 표면 곡률이 큰 영역에서 depth variation 증가
- 해결 방법:
  - `--remove_invalid_dof` 플래그로 위반 viewpoint 제거
  - 또는 DOF 값을 증가 (예: `--depth_of_field 1.0`)

### 예시 3: DOF 위반 제거

```bash
/isaac-sim/python.sh scripts/mesh_to_viewpoints.py \
  --mesh_file data/input/object/glass.obj \
  --num_points 500 \
  --check_dof \
  --remove_invalid_dof \
  --save_path data/output/glass_fov_500_filtered.h5
```

**출력**:
```
Checking DOF constraints (limit: 0.50 mm)...
Removed 259 viewpoints violating DOF constraints
Remaining viewpoints: 241

Number of viewpoints: 241
Coverage ratio: 195.9%
```

### 예시 4: 커스텀 카메라 스펙

```bash
/isaac-sim/python.sh scripts/mesh_to_viewpoints.py \
  --mesh_file data/input/object/glass.obj \
  --num_points 200 \
  --fov_width 50.0 \
  --fov_height 40.0 \
  --working_distance 150.0 \
  --depth_of_field 1.0 \
  --overlap 0.3 \
  --save_path data/output/glass_custom_camera.h5
```

### 예시 5: 시각화 포함

```bash
/isaac-sim/python.sh scripts/mesh_to_viewpoints.py \
  --mesh_file data/input/object/glass.obj \
  --num_points 100 \
  --visualize \
  --plot \
  --output data/output/glass_visualization.png
```

**시각화 요소**:
- **Open3D 3D viewer**:
  - 초록 구: Viewpoint 위치
  - 빨간 화살표: 카메라 방향
  - 회색 메시: 원본 객체
- **Matplotlib 그래프**:
  - 커버리지 파이 차트
  - 카메라 스펙 및 통계 요약

---

## 알고리즘 상세

### Surface Point Sampling

**Poisson Disk Sampling** (Open3D 구현):
- 메시 표면에 균일하게 분포된 점 생성
- 최소 거리 제약으로 균등한 간격 보장
- 자동 법선 추정

### Viewpoint Computation

각 표면 점 `p`와 법선 `n`에 대해:

```python
viewpoint_position = p + normalize(n) * working_distance
camera_direction = -normalize(n)  # 물체를 향함
```

### DOF Validation

각 viewpoint에서 5×5 ray grid 샘플링:

```python
# Local coordinate frame 생성
z_axis = camera_direction
x_axis = cross(helper, z_axis)
y_axis = cross(z_axis, x_axis)

# FOV 내 ray 샘플링
for u in [-FOV_w/2, ..., FOV_w/2]:
    for v in [-FOV_h/2, ..., FOV_h/2]:
        ray_direction = z_axis + u*x_axis + v*y_axis
        # Raycast to mesh
        distance = scene.cast_ray(viewpoint, ray_direction)

depth_variation = max(distances) - min(distances)
```

### Coverage Estimation

단순 추정 (평면 가정):

```
coverage_per_view = FOV_width × FOV_height
total_coverage = num_viewpoints × coverage_per_view
coverage_ratio = total_coverage / mesh_surface_area
```

**참고**:
- 실제 커버리지는 표면 곡률, 각도에 따라 달라짐
- Overlap으로 인해 coverage_ratio > 1 가능

---

## 좌표계

### Open3D 좌표계 (Y-up)

```
    Y (up)
    |
    |
    +---- X
   /
  Z
```

모든 데이터는 Open3D 좌표계로 저장됩니다.

### Isaac Sim 변환

`run_app_v2.py`에서 자동으로 Y-up → Z-up 변환:

```python
rotation_matrix = [
    [1,  0,  0],
    [0,  0, -1],
    [0,  1,  0]
]
```

---

## 제약 사항

1. **DOF 검증 정확도**:
   - 5×5 ray grid는 근사치
   - 더 정밀한 검증은 grid 크기 증가 필요 (성능 trade-off)

2. **Coverage 추정**:
   - 평면 FOV 가정
   - 곡면에서는 실제 커버리지가 다를 수 있음

3. **메시 품질 의존성**:
   - Low-poly 메시: 법선 추정 부정확
   - Self-intersection: Ray casting 오류 가능

4. **메모리 사용**:
   - DOF 검증은 메시 전체를 메모리에 로드
   - 대형 메시(>100MB)에서는 느려질 수 있음

---

## 문제 해결

### Coverage가 100% 미만

**원인**: 샘플링 점이 부족

**해결**:
```bash
--num_points 1000  # 점 개수 증가
```

### DOF 위반이 너무 많음

**원인**: 표면 곡률이 큼

**해결**:
1. DOF 증가:
   ```bash
   --depth_of_field 1.0
   ```
2. 위반 viewpoint 제거:
   ```bash
   --remove_invalid_dof
   ```

### HDF5 파일 크기가 큼

**원인**: Viewpoint 개수가 많음

**해결**:
```bash
--num_points 100  # 점 개수 감소
```

---

## 참고 자료

- **관련 스크립트**:
  - `mesh_to_tsp.py`: TSP 경로 최적화
  - `run_app_v2.py`: Isaac Sim 시뮬레이션
  - `tsp_utils.py`: HDF5 저장/로드 유틸리티

- **분석 문서**:
  - `MESH_TO_TSP_ANALYSIS.md`: TSP 알고리즘 상세
  - `CLAUDE.md`: 전체 시스템 개요

---

## 요약

### 핵심 개념

**mesh_to_viewpoints.py의 역할**:
- 3D 메시에서 카메라 FOV를 고려한 최적 viewpoint 샘플링
- TSP 경로는 계산하지 않음 (viewpoint만 저장)
- 간소화된 HDF5 형식 출력

**mesh_to_tsp.py의 역할**:
- Viewpoint들 간의 최적 방문 순서(TSP tour) 계산
- `--use_viewpoints` 플래그로 저장된 viewpoint 로드
- NN/Random Insertion + 2-opt 최적화

**파일 흐름**:
```
glass.obj
   ↓
mesh_to_viewpoints.py  →  glass_fov_500.h5 (viewpoints only)
   ↓
mesh_to_tsp.py --use_viewpoints  →  glass_fov_500_tsp.h5 (with TSP tour)
   ↓
run_app_v2.py  →  Isaac Sim 시뮬레이션
```

### 주요 장점

1. **모듈화**: Viewpoint 샘플링과 TSP 최적화 분리
2. **재사용성**: 같은 viewpoint로 여러 TSP 알고리즘 테스트 가능
3. **카메라 스펙 반영**: FOV, WD, DOF 제약 고려
4. **효율성**: 간소화된 파일 형식으로 빠른 I/O

### 빠른 시작

```bash
# 1. Viewpoint 생성
/isaac-sim/python.sh scripts/mesh_to_viewpoints.py \
  --mesh_file data/input/object/glass.obj \
  --num_points 100 \
  --save_path data/output/glass_viewpoints.h5

# 2. TSP 경로 계산
/isaac-sim/python.sh scripts/mesh_to_tsp.py \
  --mesh_file data/output/glass_viewpoints.h5 \
  --use_viewpoints \
  --algorithm both \
  --save_path data/output/glass_tsp.h5

# 완료! glass_tsp.h5를 run_app_v2.py에서 사용
```

---

**작성일**: 2025-11-02
**버전**: 2.0 (간소화된 형식)
**작성자**: Claude Code Assistant
