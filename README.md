# Vision Inspection Pipeline

로봇 비전 검사 파이프라인입니다. 3D 메쉬에서 충돌 없는 로봇 경로를 생성합니다.

## 파이프라인 흐름

```
mesh → viewpoints → IK solutions → trajectory → collision-free → simulation
  ↓         ↓             ↓             ↓             ↓
 .obj      .h5          .h5          .csv          .csv
```

## 📁 데이터 디렉토리 구조

```
data/
  {object_name}/          # 예: glass, phone, microwave
    mesh/
      target.obj          # 물체 메쉬 파일 (통일된 이름)
    viewpoint/
      {num_viewpoints}/   # 예: 500, 1000
        viewpoints.h5
    ik/
      {num_viewpoints}/
        ik_solutions.h5
    trajectory/
      {num_viewpoints}/
        gtsp.csv          # 초기 trajectory
        gtsp_final.csv    # collision-free trajectory
```

**예시**:
```
data/glass/mesh/target.obj
data/glass/viewpoint/500/viewpoints.h5
data/glass/ik/500/ik_solutions.h5
data/glass/trajectory/500/gtsp.csv
data/glass/trajectory/500/gtsp_final.csv
```

---

## 🎓 Jupyter Notebook (교육/데모용)

**새로운 방법**: 전체 파이프라인을 하나의 노트북에서 단계별로 실행

```bash
# Jupyter 실행
jupyter notebook vision_inspection_pipeline.ipynb
```

**장점**:
- ✅ 단계별 실행 및 시각화
- ✅ 파라미터 쉽게 조정 가능
- ✅ 중간 결과 확인 및 저장
- ✅ 교육용 설명 포함

**포함된 섹션**:
1. **Section 0**: 환경 설정
2. **Section 1**: Mesh → Viewpoints (완전 구현)
3. **Section 2-4**: IK, Trajectory, Collision (템플릿 제공)
4. **Section 5**: 결과 시각화

노트북은 기존 `scripts/` 및 `common/` 모듈을 임포트하여 사용하므로 코드 중복이 없습니다.

**진행 상황**: `PROGRESS.md` 참고

---

## ⚙️ 개별 스크립트 실행 (프로덕션용)

**간단한 사용법**: `--object_name`과 `--num_viewpoints`만 지정하면 경로 자동 생성

```bash
# 1. Mesh → Viewpoints
omni_python scripts/mesh_to_viewpoints.py \
    --object_name glass \
    --visualize

# 2. Viewpoints → IK Solutions
omni_python scripts/compute_ik_solutions.py \
    --object_name glass \
    --num_viewpoints 500

# 3. IK Solutions → Trajectory
omni_python scripts/fk_gtsp_gpu_claude2.py \
    --object_name glass \
    --num_viewpoints 500

# 4. Trajectory → Collision-Free
omni_python scripts/check_collision.py \
    --object_name glass \
    --num_viewpoints 500

# 5. Simulation
omni_python scripts/simulate_trajectory.py \
    --trajectory data/glass/trajectory/500/gtsp_final.csv
```

모든 중간 파일은 자동으로 다음 경로에 저장됩니다:
- Mesh: `data/glass/mesh/target.obj` (사전 준비 필요)
- Viewpoints: `data/glass/viewpoint/500/viewpoints.h5`
- IK: `data/glass/ik/500/ik_solutions.h5`
- Trajectory: `data/glass/trajectory/500/gtsp.csv`
- Final: `data/glass/trajectory/500/gtsp_final.csv`

---

## 스크립트 설명

### 1. mesh_to_viewpoints.py

**역할**: 3D 메쉬에서 카메라 뷰포인트 생성

**실행**:
```bash
omni_python scripts/mesh_to_viewpoints.py \
    --object_name glass \
    --visualize
```

**주요 옵션**:
- `--object_name`: 물체 이름 (자동 경로 생성)
- `--curvature_weight 0.5`: 곡률 영향 (0.0~1.0)
- `--visualize`: Open3D 시각화

**출력**:
- `data/{object_name}/viewpoint/{N}/viewpoints.h5`

---

### 2. compute_ik_solutions.py

**역할**: 각 뷰포인트에 대한 역기구학(IK) 계산

**실행**:
```bash
omni_python scripts/compute_ik_solutions.py \
    --object_name glass \
    --num_viewpoints 500
```

**주요 옵션**:
- `--object_name`: 물체 이름 (자동 경로 생성)
- `--num_viewpoints`: 뷰포인트 개수
- `--robot ur20.yml`: 로봇 설정 파일

**출력**:
- `data/{object_name}/ik/{N}/ik_solutions.h5`

**특징**:
- GPU 가속 (CuRobo)
- 자기 충돌 + 환경 충돌 체크
- 여러 IK 해 생성

---

### 3. fk_gtsp_gpu_claude2.py

**역할**: GTSP 알고리즘으로 최적 경로 순서 결정

**실행**:
```bash
# Basic usage
omni_python scripts/fk_gtsp_gpu_claude2.py \
    --object_name glass \
    --num_viewpoints 500

# With visualization (NEW)
omni_python scripts/fk_gtsp_gpu_claude2.py \
    --object_name glass \
    --num_viewpoints 500 \
    --visualize \
    --show-frames
```

**주요 옵션**:
- `--object_name`: 물체 이름 (자동 경로 생성)
- `--num_viewpoints`: 뷰포인트 개수
- `--lam_rot 1.0`: 회전 비용 가중치
- `--knn 5`: k-NN 이웃 개수
- **`--visualize`**: 생성 후 trajectory 시각화 (Open3D) 🆕
- **`--show-frames`**: 각 waypoint에 좌표계 표시 🆕
- **`--frame-size 0.02`**: 좌표계 크기 (미터) 🆕
- **`--mesh`**: 시각화용 메쉬 경로 (기본: 자동 감지) 🆕

**출력**:
- `data/{object_name}/trajectory/{N}/gtsp.csv`

**특징**:
- GPU 가속 순방향 기구학
- 관절 공간 거리 최소화
- 각 뷰포인트에서 최적 IK 해 선택
- **통합 시각화 (Open3D)**

---

### 4. check_collision.py

**역할**: 충돌 체크 및 모션 플래닝

**실행**:
```bash
omni_python scripts/check_collision.py \
    --object_name glass \
    --num_viewpoints 500
```

**주요 옵션**:
- `--object_name`: 물체 이름 (자동 경로 생성)
- `--num_viewpoints`: 뷰포인트 개수
- `--robot_config ur20.yml`: 로봇 설정 파일

**출력**:
- `data/{object_name}/trajectory/{N}/gtsp_final.csv`

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

**보간 간격**:
```python
COLLISION_ADAPTIVE_MAX_JOINT_STEP_DEG = 1.0 # 보간 간격
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

## 유틸리티 스크립트

### preprocess_mesh.py

**역할**: Multi-material OBJ 파일을 material별로 분리

**실행**:
```bash
# Using object name (recommended - NEW)
omni_python scripts/preprocess_mesh.py \
    --object_name glass \
    --material_rgb "0,255,0" \
    --visualize

# Using explicit paths
omni_python scripts/preprocess_mesh.py \
    --input data/object/sample_step_scaled.obj \
    --material-name "Opaque(0,255,0).001" \
    --output data/object/target_surface.ply
```

**주요 옵션**:
- **`--object_name`**: 물체 이름으로 경로 자동 생성 (e.g., "glass", "phone") 🆕
- `--input`: 입력 OBJ 파일 경로
- `--output`: 출력 PLY 파일 경로 (기본: 자동 생성)
- `--material-name`: Material 이름으로 선택
- `--material-rgb "R,G,B"`: RGB 색상으로 선택 (권장)
- `--color-tolerance 5.0`: RGB 매칭 허용 오차
- `--visualize`: Open3D 시각화
- `--no-save`: 저장 생략 (검사만)

**출력**:
- `data/{object_name}/mesh/target.ply` (when using --object_name)

**특징**:
- Trimesh 기반 정확한 material 파싱
- RGB 색상 매칭으로 material 자동 선택
- Binary PLY 압축 저장
- 좌표계: Z-up (Isaac Sim 호환)

---

## 추가 문서

- `/docs/MESH_TO_VIEWPOINTS.md` - 뷰포인트 생성 상세 문서
- `/common/config.py` - 전체 설정 레퍼런스
- 각 스크립트의 docstring - 상세 사용법
