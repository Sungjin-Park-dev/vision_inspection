# Vision Inspection Pipeline

로봇 비전 검사 파이프라인입니다. 3D 메쉬에서 충돌 없는 로봇 경로를 생성합니다.

## 파이프라인 흐름

```
mesh → viewpoints → IK solutions → trajectory → collision-free → simulation
  ↓         ↓             ↓             ↓             ↓
 .obj      .h5          .h5          .csv          .csv
```

---

## 빠른 시작

전체 파이프라인 실행 예제:

```bash
# 1. Mesh → Viewpoints
omni_python scripts/mesh_to_viewpoints.py \
    --mesh_file data/object/glass.obj \
    --visualize

# 2. Viewpoints → IK Solutions
omni_python scripts/compute_ik_solutions.py \
    --viewpoints data/viewpoint/675/viewpoints.h5

# 3. IK Solutions → Trajectory
omni_python scripts/fk_gtsp_gpu_claude2.py \
    --h5 data/ik/675/ik_solutions.h5 \
    --knn 5 \
    --csv_out data/trajectory/675/dh_5.csv

# 4. Trajectory → Collision-Free
omni_python scripts/curobo_check.py \
    --trajectory data/trajectory/675/dh_5.csv

# 5. Simulation
omni_python scripts/simulate_trajectory.py \
    --trajectory data/trajectory/675/dh_5_collision_free.csv
```

---

## 스크립트 설명

### 1. mesh_to_viewpoints.py

**역할**: 3D 메쉬에서 카메라 뷰포인트 생성

**실행**:
```bash
omni_python scripts/mesh_to_viewpoints.py --mesh_file data/object/microwave.obj [OPTIONS]
```

**주요 옵션**:
- `--curvature_weight 0.5`: 곡률 영향 (0.0~1.0)
- `--min_tilt_angle 30.0`: 최소 기울기 각도 (도)
- `--visualize`: Open3D 시각화

**출력**: `data/viewpoint/{N}/viewpoints.h5` - 표면 위치 + 법선 벡터

---

### 2. compute_ik_solutions.py

**역할**: 각 뷰포인트에 대한 역기구학(IK) 계산

**실행**:
```bash
omni_python scripts/compute_ik_solutions.py \
    --viewpoints data/viewpoint/100/viewpoints.h5 \
    --output data/ik/100/ik_solutions.h5 \
    --mesh_file data/object/microwave.obj [OPTIONS]
```

**주요 옵션**:
- `--robot ur20.yml`: 로봇 설정 파일

**출력**: `data/ik/{N}/ik_solutions.h5` - 각 뷰포인트의 관절 각도 해

**특징**:
- GPU 가속 (CuRobo)
- 자기 충돌 + 환경 충돌 체크
- 여러 IK 해 생성

---

### 3. fk_gtsp_gpu_claude2.py

**역할**: GTSP 알고리즘으로 최적 경로 순서 결정

**실행**:
```bash
omni_python scripts/fk_gtsp_gpu_claude2.py \
    --ik_file data/ik/100/ik_solutions.h5 \
    --output data/tour/100/tour.h5
```

**출력**: `data/trajectory/{N}/joint_trajectory_dp.csv` - 순서가 정해진 경로

**특징**:
- GPU 가속 순방향 기구학
- 관절 공간 거리 최소화
- 각 뷰포인트에서 최적 IK 해 선택

**참고**: 안정성을 위해 리팩토링에서 제외됨

---

### 4. curobo_check.py

**역할**: 충돌 체크 및 모션 플래닝

**실행**:
```bash
omni_python scripts/curobo_check.py \
    --trajectory data/trajectory/100/joint_trajectory_dp.csv \
    --mesh data/object/microwave.obj [OPTIONS]
```

**주요 옵션**:
- `--robot ur20_safe.yml`: 로봇 설정 파일
- `--replan`: 충돌 구간 재계획 활성화
- `--replan_timeout 8.0`: 재계획 타임아웃 (초)
- `--replan_max_attempts 3`: 최대 재계획 시도 횟수

**출력**: `data/trajectory/{N}/joint_trajectory_dp_curobo_interpolated.csv` - 충돌 없는 경로

**특징**:
- 적응형 보간 (adaptive interpolation)
- 병렬 충돌 체크
- CuRobo 모션 플래닝

---

### 5. simulate_trajectory.py

**역할**: Isaac Sim에서 경로 시각화

**실행**:
```bash
# 대화형 모드
omni_python scripts/simulate_trajectory.py \
    --trajectory data/trajectory/100/joint_trajectory_dp_curobo_interpolated.csv [OPTIONS]

# Headless 모드
omni_python scripts/simulate_trajectory.py \
    --trajectory data/trajectory/100/joint_trajectory_dp_curobo_interpolated.csv \
    --headless native
```

**주요 옵션**:
- `--robot ur20.yml`: 로봇 설정 파일
- `--visualize_spheres`: 충돌 구체 시각화
- `--headless native`: Headless 모드

**특징**:
- 실시간 Isaac Sim 시각화
- 엔드이펙터 카메라 표시
- 타이밍 통계

---

## 설정

모든 설정은 `common/config.py`에 정리되어 있습니다.

**카메라 스펙**:
```python
CAMERA_FOV_WIDTH_MM = 41.0          # FOV 너비 (mm)
CAMERA_FOV_HEIGHT_MM = 30.0         # FOV 높이 (mm)
CAMERA_WORKING_DISTANCE_MM = 110.0  # 작업 거리 (mm)
CAMERA_OVERLAP_RATIO = 0.5          # 오버랩 비율 (50%)
```

**환경 설정**:
```python
GLASS_POSITION = np.array([1.00, 0.0, -0.172])  # 객체 위치
TABLE_POSITION = np.array([1.0, 0.0, -0.425])   # 테이블 위치
TABLE_DIMENSIONS = np.array([0.6, 1.0, 0.5])    # 테이블 크기
```

**알고리즘 파라미터**:
```python
IK_NUM_SEEDS = 20                    # IK 시드 개수
COLLISION_ADAPTIVE_INTERP = True     # 적응형 보간
REPLAN_ENABLED = True                # 재계획 활성화
REPLAN_TIMEOUT = 8.0                 # 재계획 타임아웃
```

---

## 공통 유틸리티

리팩토링으로 공통 기능을 `common/` 디렉토리로 분리했습니다.

### cli_utils.py
일관된 CLI 출력 포맷:
```python
from common.cli_utils import print_section_header, print_key_value

print_section_header("LOADING DATA", width=60)
print_key_value("Input file", file_path)
```

### world_setup.py
충돌 환경 설정 중앙화:
```python
from common.world_setup import setup_collision_world

world_cfg = setup_collision_world(
    table_position=config.TABLE_POSITION,
    table_dimensions=config.TABLE_DIMENSIONS,
    mesh_files=[mesh_file_path],
    verbose=True
)
```

### trajectory_io.py
경로 및 뷰포인트 파일 입출력:
```python
from common.trajectory_io import load_trajectory_csv, save_trajectory_csv

trajectory, joint_names = load_trajectory_csv("path/to/trajectory.csv")
save_trajectory_csv(trajectory, "path/to/output.csv", joint_names=joint_names)
```

---

## 좌표계

모든 스크립트는 **Z-up 좌표계** (Isaac Sim / URDF 규약) 사용:
- X: 오른쪽
- Y: 앞
- Z: 위

**메쉬 파일**: Z-up, 단위는 미터(m)

## 리팩토링 내역 (2024-11)

**생성된 공통 유틸리티**:
- `common/cli_utils.py` - 통일된 CLI 출력
- `common/world_setup.py` - 충돌 환경 설정 중앙화
- `common/trajectory_io.py` - 파일 입출력 통일

**리팩토링된 스크립트**:
- `compute_ik_solutions.py`
- `curobo_check.py`
- `simulate_trajectory.py`
- `mesh_to_viewpoints.py`

**제외된 스크립트**:
- `fk_gtsp_gpu_claude2.py` - 안정성 유지를 위해 제외
---

## 추가 문서

- `/docs/MESH_TO_VIEWPOINTS.md` - 뷰포인트 생성 상세 문서
- `/common/config.py` - 전체 설정 레퍼런스
- 각 스크립트의 docstring - 상세 사용법
