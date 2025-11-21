# FOV 기반 Viewpoint 샘플링 가이드

## 📋 목차
1. [개요](#개요)
2. [카메라 스펙](#카메라-스펙)
3. [주요 기능](#주요-기능)
4. [알고리즘 상세](#알고리즘-상세)
5. [사용 방법](#사용-방법)
6. [출력 형식](#출력-형식)
7. [예시](#예시)
8. [좌표계](#좌표계)

---

## 개요

### 목적
카메라 FOV(Field of View) 스펙에 맞춰 3D 메시 객체를 효율적으로 커버하는 viewpoint를 샘플링하는 도구입니다. 로봇 비전 검사 시스템에서 카메라 경로 계획에 사용됩니다.

### 핵심 특징
- **자동 viewpoint 수 계산**: 표면적과 FOV를 고려하여 필요한 viewpoint 수 자동 추정
- **로봇 접근성 필터링**: 바닥면 제거 및 수평면 tilting으로 로봇이 접근 가능한 viewpoint만 생성
- **적응형 샘플링**: 표면 곡률에 따라 샘플링 밀도 조정
- **FOV 기반 샘플링**: 카메라 시야각(41×30mm)을 고려한 viewpoint 생성
- **Working Distance**: 정확한 초점 거리(110mm) 유지
- **Depth of Field 검증**: 피사계 심도(0.5mm) 제약 확인
- **Overlap 관리**: 인접 view 간 중첩 비율 설정 가능
- **정확한 커버리지 계산**: Voxel 기반 실제 커버리지 분석
- **TSP 호환**: `mesh_to_tsp.py`와 호환되는 HDF5 형식 출력

### 파일 정보
- **위치**: `/isaac-sim/curobo/vision_inspection/scripts/mesh_to_viewpoints.py`
- **라인 수**: ~1250 lines
- **의존성**: Open3D, NumPy, h5py, tsp_utils, common.config

---

## 카메라 스펙

### 기본 설정 (common/config.py)

```python
# Camera specifications (LG Vision Inspection System)
CAMERA_SENSOR_WIDTH_PX = 4096
CAMERA_SENSOR_HEIGHT_PX = 3000
CAMERA_PIXEL_SIZE_UM = 3.45
CAMERA_FOV_WIDTH_MM = 41.0
CAMERA_FOV_HEIGHT_MM = 30.0
CAMERA_WORKING_DISTANCE_MM = 110.0
CAMERA_DEPTH_OF_FIELD_MM = 0.5
CAMERA_OVERLAP_RATIO = 0.25
```

| 항목 | 값 | 단위 |
|------|-----|------|
| 센서 해상도 | 4096 × 3000 | pixel |
| 픽셀 크기 | 3.45 × 3.45 | μm |
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

## 주요 기능

### 1. CameraSpec 클래스

카메라 및 렌즈 사양 관리:

```python
@dataclass
class CameraSpec:
    sensor_width_px: int = 4096
    sensor_height_px: int = 3000
    pixel_size_um: float = 3.45
    fov_width_mm: float = 41.0
    fov_height_mm: float = 30.0
    working_distance_mm: float = 110.0
    depth_of_field_mm: float = 0.5
    overlap_ratio: float = 0.25

    def get_effective_coverage_mm(self) -> Tuple[float, float]:
        """중첩을 고려한 실제 커버리지 계산"""
        effective_width = self.fov_width_mm * (1.0 - self.overlap_ratio)
        effective_height = self.fov_height_mm * (1.0 - self.overlap_ratio)
        return effective_width, effective_height
```

### 2. 자동 Viewpoint 수 계산

`estimate_required_viewpoints()` 함수:

```python
def estimate_required_viewpoints(
    mesh: o3d.geometry.TriangleMesh,
    camera_spec: CameraSpec,
    target_coverage: float = 1.0,
    curvature_weight: float = 0.5
) -> int:
```

**동작 방식:**
1. 메시 표면적 계산
2. FOV 및 overlap을 고려한 viewpoint당 커버리지 계산
3. (선택) 표면 곡률 분석하여 고곡률 영역에 더 많은 viewpoint 할당
4. 필요한 총 viewpoint 수 반환

**Curvature weight:**
- `0.0`: 균일한 overlap (모든 곳에서 동일)
- `0.5`: 적당한 적응형 overlap (권장)
- `1.0`: 공격적인 적응형 overlap (고곡률 영역에 집중)

### 3. 샘플링 알고리즘

#### Uniform Sampling (기본)
`sample_points_uniform()` - Poisson disk sampling 사용

#### Adaptive Sampling (선택)

**옵션 1: Weighted Random Sampling** (기본)
`sample_points_adaptive()` - 표면 곡률 기반 밀도 조정

```python
def sample_points_adaptive(
    mesh: o3d.geometry.TriangleMesh,
    num_points: int,
    curvature_weight: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """Curvature 기반 weighted random sampling"""
```

**옵션 2: Curvature-Stratified Poisson Disk** (권장 ⭐)
`sample_points_adaptive_poisson()` - Curvature 층별 Poisson disk sampling

```python
def sample_points_adaptive_poisson(
    mesh: o3d.geometry.TriangleMesh,
    num_points: int,
    curvature_weight: float = 0.5,
    num_strata: int = 3
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Curvature-stratified Poisson disk sampling

    장점:
    - 샘플 간 최소 거리 보장 (blue noise distribution)
    - Curvature 기반 adaptive density
    - 더 균일한 공간 분포
    """
```

**동작 방식:**
1. 메시를 curvature 기준으로 3개 층(strata)으로 분할:
   - Low (0.00-0.33): 평면 영역
   - Medium (0.33-0.67): 중간 곡률
   - High (0.67-1.00): 모서리/코너
2. 각 층에 샘플 수 할당 (high-curvature에 더 많이)
3. 각 층에 독립적으로 Poisson disk sampling 적용
4. 모든 층의 샘플 병합

**Curvature 계산:**
```python
def compute_surface_curvature(mesh: o3d.geometry.TriangleMesh) -> np.ndarray:
    """각 vertex의 곡률 추정 (normal variation 기반)"""
```

고곡률 영역(모서리, 코너)에서 더 많은 point를 샘플링합니다.

### 4. 로봇 접근성 필터링

#### 바닥면 Viewpoint 제거

`filter_downward_facing_viewpoints()` 함수:

```python
def filter_downward_facing_viewpoints(
    viewpoints: List[Viewpoint],
    z_threshold: float = 0.0
) -> Tuple[List[Viewpoint], int]:
```

- Surface normal의 Z 성분이 0 미만인 viewpoint 제거
- 로봇이 물리적으로 접근 불가능한 하부 viewpoint 제거

#### 수평면 Tilting 적용

`apply_minimum_tilt_angle()` 함수:

```python
def apply_minimum_tilt_angle(
    viewpoints: List[Viewpoint],
    camera_spec: CameraSpec,
    min_tilt_deg: float = 30.0
) -> Tuple[List[Viewpoint], int]:
```

**목적:** 벽면(수평 normal)을 위에서 30도 각도로 보도록 조정

**동작:**
1. 거의 수평인 surface normal 감지 (`|normal.z| < sin(30°)`)
2. 같은 surface point를 보되, 카메라를 위쪽으로 배치
3. Working distance 분해:
   - 수평 거리: `wd × cos(tilt_angle)`
   - 수직 offset: `wd × sin(tilt_angle)`
4. Viewpoint 위치 재계산하여 원래 surface point를 정확히 바라봄

**수학:**
```python
# Surface point 복원
surface_point = vp.position + vp.normal * wd

# 수평 방향 유지, 위로 offset
horizontal_distance = wd * cos(min_tilt_rad)
vertical_offset = wd * sin(min_tilt_rad)

adjusted_position = surface_point + horizontal_dir * horizontal_distance
adjusted_position[2] += vertical_offset

# 카메라는 surface_point를 정확히 향함
adjusted_camera_direction = normalize(surface_point - adjusted_position)
```

### 5. Depth of Field 검증

`filter_viewpoints_by_dof()` 함수:

각 viewpoint에서 5×5 ray grid를 FOV 내에 샘플링하여 depth variation 계산:

```python
depth_variation = max_distance - min_distance

if depth_variation > DOF_limit:
    # 해당 viewpoint에서 일부 영역이 초점 밖
```

### 6. 커버리지 분석

#### Simple Coverage (with overlap)
```python
simple_coverage = sum(vp.coverage_area for vp in viewpoints)
simple_ratio = simple_coverage / mesh_surface_area
```

#### Voxel-based Coverage (accurate)
`compute_voxel_based_coverage()` 함수:

- 메시를 voxel grid로 변환 (기본: 2mm)
- 각 viewpoint에서 보이는 voxel 마킹
- 실제 커버된 voxel 개수 계산 (overlap 제거)

---

## 알고리즘 상세

### 전체 처리 흐름

```
1. 메시 로드 및 분석
   ├─ 표면적 계산
   ├─ 좌표 범위 확인 (단위 검증)
   └─ Normal 추정

2. 필요한 viewpoint 수 자동 계산
   ├─ FOV 및 overlap 기반 기본 추정
   └─ (선택) 곡률 기반 적응형 조정

3. Surface point 샘플링
   ├─ Uniform: Poisson disk sampling
   └─ Adaptive: 곡률 가중치 적용

4. Viewpoint 계산
   └─ position = surface + normal × working_distance
   └─ camera_direction = -normal

5. 로봇 접근성 필터링 (기본: 활성화)
   ├─ 바닥면 제거 (normal.z < 0)
   └─ 수평면 tilting (30도 위에서 보도록)

6. (선택) DOF 검증
   └─ Depth variation > 0.5mm 확인

7. (선택) Voxel 기반 커버리지 계산

8. HDF5 저장
   └─ Surface positions & normals
```

### Viewpoint Computation 상세

**입력:**
- Surface point: `p` (메시 표면의 점)
- Surface normal: `n` (표면에서 바깥으로)

**출력:**
- Viewpoint position: `p + normalize(n) × wd`
- Camera direction: `-normalize(n)` (표면을 향함)

**저장 시 변환:**
```python
# HDF5 저장 전: viewpoint position → surface position 변환
# (mesh_to_tsp.py와 run_app_v2.py는 surface position 기대)
surface_position = viewpoint.position + viewpoint.normal × wd
surface_normal = -viewpoint.normal
```

### Tilting 알고리즘 상세

**문제:** 벽면을 정면에서 보면 로봇 충돌 위험

**해결:** 같은 surface point를 30도 위에서 보도록 카메라 재배치

**계산 예시 (벽면):**
```
원래:
  surface_point = [0, 0, 0.5]
  normal = [1, 0, 0] (수평)
  viewpoint = [0.3, 0, 0.5]
  camera_dir = [-1, 0, 0] (정면)

Tilting 후 (30도):
  horizontal_dist = 0.3 × cos(30°) = 0.26m
  vertical_offset = 0.3 × sin(30°) = 0.15m

  adjusted_viewpoint = [0.26, 0, 0.65]  ← Z 높이 증가!
  adjusted_camera_dir = [-0.866, 0, -0.5]  ← 아래를 향함

  실제로 보는 지점:
    [0.26, 0, 0.65] + 0.3×[-0.866, 0, -0.5]
    = [0, 0, 0.5] ✓ (원래 surface point)
```

---

## 사용 방법

### 기본 사용 (자동 viewpoint 수 계산)

```bash
/isaac-sim/python.sh scripts/mesh_to_viewpoints.py \
  --mesh_file data/object/phone.obj \
  --save_path data/viewpoint/auto/viewpoints.h5
```

자동으로:
- 필요한 viewpoint 수 계산
- 바닥면 제거
- 수평면 30도 tilting 적용

### 전체 옵션

```bash
/isaac-sim/python.sh scripts/mesh_to_viewpoints.py \
  # 입력/출력
  --mesh_file data/object/phone.obj \
  --save_path data/viewpoint/500/viewpoints.h5 \

  # 카메라 스펙 (선택)
  --fov_width 41.0 \
  --fov_height 30.0 \
  --working_distance 110.0 \
  --depth_of_field 0.5 \
  --overlap 0.25 \

  # 샘플링 (자동 계산 사용 시 불필요)
  --adaptive_sampling \          # 곡률 기반 샘플링
  --use_poisson_disk \           # Poisson disk 사용 (균일 분포)
  --curvature_weight 0.5 \       # 곡률 영향도 (0-1)

  # 로봇 접근성 필터링 (기본: 활성화)
  --filter_downward \            # 바닥면 제거 (기본값)
  --apply_tilt \                 # 수평면 tilting (기본값)
  --min_tilt_angle 30.0 \        # Tilting 각도 (도)

  # DOF 검증 (선택)
  --check_dof \
  --remove_invalid_dof \

  # 커버리지 분석 (선택)
  --voxel_coverage \
  --voxel_size 2.0 \             # Voxel 크기 (mm)

  # 시각화 (선택)
  --visualize
```

### 명령줄 인자 상세

#### 필수
| 옵션 | 타입 | 설명 |
|------|------|------|
| `--mesh_file` | str | 메시 파일 경로 (.obj), Z-up 좌표계 |

#### 입출력
| 옵션 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `--save_path` | str | auto | HDF5 출력 경로 (미지정 시 자동 생성) |

#### 카메라 스펙
| 옵션 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `--fov_width` | float | 41.0 | FOV 너비 (mm) |
| `--fov_height` | float | 30.0 | FOV 높이 (mm) |
| `--working_distance` | float | 110.0 | 작업 거리 (mm) |
| `--depth_of_field` | float | 0.5 | 피사계 심도 (mm) |
| `--overlap` | float | 0.25 | 중첩 비율 (0-1) |

#### 샘플링 (자동 계산)
| 옵션 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `--adaptive_sampling` | flag | False | 곡률 기반 적응형 샘플링 |
| `--use_poisson_disk` | flag | False | Poisson disk 기반 adaptive sampling (균일 분포) |
| `--curvature_weight` | float | 0.5 | 곡률 영향도 (0=균일, 1=최대) |

#### 로봇 접근성
| 옵션 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `--filter_downward` | flag | **True** | 바닥면 viewpoint 제거 |
| `--no_filter_downward` | flag | - | 필터링 비활성화 |
| `--apply_tilt` | flag | **True** | 수평면 tilting 적용 |
| `--no_apply_tilt` | flag | - | Tilting 비활성화 |
| `--min_tilt_angle` | float | 30.0 | 최소 tilt 각도 (도) |

#### DOF 검증
| 옵션 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `--check_dof` | flag | False | DOF 검증 활성화 |
| `--remove_invalid_dof` | flag | False | DOF 위반 viewpoint 제거 |

#### 커버리지
| 옵션 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `--voxel_coverage` | flag | False | Voxel 기반 정확한 커버리지 계산 |
| `--voxel_size` | float | 2.0 | Voxel 크기 (mm) |

#### 시각화
| 옵션 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `--visualize` | flag | False | Open3D 3D 시각화 |

---

## 출력 형식

### HDF5 파일 구조

```
viewpoints.h5
│
├── metadata (group)
│   ├── num_viewpoints: int
│   ├── mesh_file: str
│   ├── timestamp: ISO datetime
│   ├── format: "viewpoints_only"
│   └── camera_spec (group)
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
    ├── positions: (N, 3) float32 - Surface 좌표 (Z-up)
    └── normals: (N, 3) float32 - Surface normal 벡터
```

**중요:** 저장되는 것은 **surface positions**입니다 (camera positions 아님).
`mesh_to_tsp.py`와 `run_app_v2.py`에서 `NORMAL_SAMPLE_OFFSET`을 적용하여 camera position을 계산합니다.

### 콘솔 출력 예시

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

Loading mesh from: data/object/phone.obj
Loaded mesh: 2341 vertices, 4680 triangles
Surface area: 24532.45 mm²

✓ Mesh coordinates appear to be in METERS (max range: 0.1523m)
✓ Using Z-up coordinate system (compatible with Isaac Sim / URDF / Pinocchio)

Automatic viewpoint estimation (adaptive overlap):
  Surface area: 24532.45 mm²
  FOV: 41.0 × 30.0 mm
  Curvature weight: 0.50
  Adaptive overlap range: 25% (flat) → 55% (curved)
  Average overlap: 31.2%
  Estimated viewpoints: 42

Sampling 42 points using adaptive (curvature-based) sampling...
Sampled 42 points

Computing viewpoints...
  Working distance: 110.0 mm = 0.11 m
  Offsetting surface points by 0.11 m along normals
  Generated 42 viewpoints

============================================================
FILTERING DOWNWARD-FACING VIEWPOINTS
============================================================
Filtered downward-facing viewpoints:
  Removed: 8
  Remaining: 34

============================================================
APPLYING MINIMUM TILT ANGLE
============================================================
Applying minimum tilt angle (30.0°)...
  Adjusted 12 nearly-horizontal viewpoints
  All viewpoints now view from >= 30.0° above horizontal
  Viewpoint Z-positions adjusted to maintain inspection coverage

============================================================
RESULTS
============================================================
Number of viewpoints: 34
Mesh surface area: 24532.45 mm²

Viewpoint filtering (robot accessibility):
  Downward-facing removed: 8
  Horizontal viewpoints adjusted: 12
  Minimum tilt angle: 30.0°
  Z-positions adjusted to maintain coverage height

Coverage (simple estimate, with overlap):
  Total coverage: 41820.00 mm²
  Coverage ratio: 170.5%
============================================================

Auto-generated save path: data/viewpoint/34/viewpoints.h5

============================================================
COORDINATE CONVERSION FOR HDF5 SAVE
============================================================
Converting viewpoint positions → surface positions
  Working distance: 110.0 mm = 0.110000 m
  Forward:  viewpoint_pos = surface_pos + surface_normal × WD
  Inverse:  surface_pos = viewpoint_pos - surface_normal × WD

Verification (first viewpoint):
  Original viewpoint pos:  [0.0523 0.0312 0.0845]
  Recovered surface pos:   [0.0412 0.0298 0.0734]
  Recomputed viewpoint:    [0.0523 0.0312 0.0845]
  Position error:          0.000000 mm
  ✓ Conversion verified (error < 1 μm)

Saving 34 surface positions to HDF5
============================================================

Done!
```

---

## 예시

### 예시 1: 기본 사용 (자동 계산)

```bash
/isaac-sim/python.sh scripts/mesh_to_viewpoints.py \
  --mesh_file data/object/phone.obj
```

**결과:**
- 자동으로 필요한 viewpoint 수 계산
- 바닥면 제거 및 30도 tilting 적용
- `data/viewpoint/{N}/viewpoints.h5` 자동 생성

### 예시 2: 적응형 샘플링 (Weighted Random)

```bash
/isaac-sim/python.sh scripts/mesh_to_viewpoints.py \
  --mesh_file data/object/phone.obj \
  --adaptive_sampling \
  --curvature_weight 0.8
```

**효과:**
- 고곡률 영역(모서리, 코너)에 더 많은 viewpoint
- `curvature_weight=0.8`: 공격적인 적응형 샘플링
- Weighted random sampling (샘플 clustering 가능)

### 예시 2b: 적응형 Poisson Disk 샘플링 (권장 ⭐)

```bash
/isaac-sim/python.sh scripts/mesh_to_viewpoints.py \
  --mesh_file data/object/phone.obj \
  --adaptive_sampling \
  --use_poisson_disk \
  --curvature_weight 0.5
```

**효과:**
- 고곡률 영역에 더 많은 viewpoint (adaptive)
- 샘플 간 최소 거리 보장 (blue noise)
- 더 균일한 공간 분포

**출력 예시:**
```
Sampling 675 points using curvature-stratified Poisson disk sampling...
  Curvature weight: 0.50
  Number of strata: 3

Stratum allocation:
  low [0.00-0.33]:    220 samples (30.1% area, factor: 1.08)
    → Sampled 220 points
  medium [0.33-0.67]: 500 samples (59.3% area, factor: 1.25)
    → Sampled 500 points
  high [0.67-1.00]:   101 samples (10.6% area, factor: 1.42)
    → Sampled 101 points

Total sampled: 821 points from 3 strata
  Adjusting from 821 to 675 points...
Final count: 675 points
```

### 예시 3: DOF 검증 및 위반 제거

```bash
/isaac-sim/python.sh scripts/mesh_to_viewpoints.py \
  --mesh_file data/object/phone.obj \
  --check_dof \
  --remove_invalid_dof
```

**결과:**
```
Checking DOF constraints (limit: 0.50 mm)...
Removed 15 viewpoints violating DOF constraints
Remaining viewpoints: 27

DOF constraints:
  Violations: 15
  Avg depth variation: 0.234 mm
  Max depth variation: 0.487 mm
```

### 예시 4: Voxel 기반 정확한 커버리지

```bash
/isaac-sim/python.sh scripts/mesh_to_viewpoints.py \
  --mesh_file data/object/phone.obj \
  --voxel_coverage \
  --voxel_size 2.0
```

**결과:**
```
Computing voxel-based coverage (12576 voxels)...
  Processing viewpoint 10/34...
  Processing viewpoint 20/34...
  Processing viewpoint 30/34...
  Coverage: 11234/12576 voxels (89.3%)

Coverage (voxel-based, no overlap):
  Voxels: 11234/12576
  Coverage ratio: 89.3%

Coverage (simple estimate, with overlap):
  Total coverage: 41820.00 mm²
  Coverage ratio: 170.5%
```

### 예시 5: 커스텀 tilting 각도

```bash
/isaac-sim/python.sh scripts/mesh_to_viewpoints.py \
  --mesh_file data/object/phone.obj \
  --min_tilt_angle 45.0
```

**효과:**
- 수평면을 45도 위에서 보도록 조정
- 더 가파른 각도 → 로봇 충돌 위험 감소
- 대신 coverage area 약간 증가

### 예시 6: 필터링 비활성화 (원래 동작)

```bash
/isaac-sim/python.sh scripts/mesh_to_viewpoints.py \
  --mesh_file data/object/phone.obj \
  --no_filter_downward \
  --no_apply_tilt
```

**효과:**
- 바닥면 viewpoint 포함
- 수평면 tilting 없음
- 기존 동작과 동일 (로봇 접근성 무시)

### 예시 7: 시각화

```bash
/isaac-sim/python.sh scripts/mesh_to_viewpoints.py \
  --mesh_file data/object/phone.obj \
  --visualize
```

**Open3D 시각화:**
- 초록 구: Viewpoint 위치
- 빨간 화살표: 카메라 방향
- 회색 메시: 원본 객체

---

## 좌표계

### 입력/출력 좌표계: Z-up (Isaac Sim / URDF / Pinocchio)

```
    Z (up)
    |
    |
    +---- X
   /
  Y
```

모든 데이터는 Z-up 좌표계로 저장됩니다:
- Mesh 파일: Z-up (.obj)
- HDF5 출력: Z-up (positions, normals)
- `mesh_to_tsp.py`: Z-up 입력 기대
- `run_app_v2.py`: Z-up에서 Isaac Sim 좌표계로 자동 변환

### Surface Normal 방향

- **Surface normal**: 표면에서 바깥으로 향함 (메시 외부)
- **Camera direction**: `-surface_normal` (표면을 향함)

### Viewpoint Position vs Surface Position

**계산 시:**
```python
viewpoint_position = surface_position + surface_normal × working_distance
camera_direction = -surface_normal
```

**저장 시:**
```python
# HDF5에는 surface_position과 surface_normal 저장
# (run_app_v2.py가 NORMAL_SAMPLE_OFFSET 적용하여 viewpoint 계산)
```

---

## 워크플로우

### Viewpoint → TSP → 시뮬레이션

```bash
# Step 1: Viewpoint 생성
/isaac-sim/python.sh scripts/mesh_to_viewpoints.py \
  --mesh_file data/object/phone.obj \
  --save_path data/viewpoint/auto/viewpoints.h5

# Step 2: TSP 경로 최적화
/isaac-sim/python.sh scripts/mesh_to_tsp.py \
  --mesh_file data/viewpoint/auto/viewpoints.h5 \
  --use_viewpoints \
  --algorithm both \
  --num_starts 20 \
  --save_path data/tour/auto/tour.h5
---

## 제약 사항

1. **메시 단위:**
   - 메시는 미터(m) 단위여야 함
   - 밀리미터 단위 메시는 ×0.001 스케일링 필요
   - 좌표 범위 자동 감지 및 경고 제공

2. **DOF 검증 정확도:**
   - 5×5 ray grid는 근사치
   - 더 정밀한 검증은 grid 크기 증가 필요 (성능 trade-off)

3. **Voxel 커버리지 성능:**
   - Viewpoint 많을수록 계산 시간 증가
   - 100+ viewpoints에서는 수 분 소요 가능

4. **Tilting 제약:**
   - 수평면만 조정 (수직면은 이미 접근 가능)
   - 같은 surface point를 보도록 보장
   - Working distance는 항상 유지

5. **Adaptive Poisson Disk Sampling:**
   - 층 간 경계에서 약간의 불연속 가능
   - 샘플 수 조정 시 random downsampling 사용
   - 각 층 내에서만 blue noise 특성 보장 (층 간은 보장 안됨)

---

## 샘플링 방법 비교

| 특성 | Uniform | Adaptive (Random) | Adaptive (Poisson) |
|------|---------|-------------------|-------------------|
| 최소 거리 보장 | ✅ 전체 | ❌ 없음 | ✅ 층별 |
| Curvature 적응 | ❌ 없음 | ✅ 가중치 | ✅ 층별 할당 |
| 공간 분포 | 매우 균일 | 불균일 가능 | 균일 (층별) |
| 계산 속도 | 빠름 | 빠름 | 빠름 |
| 추천 용도 | 단순 객체 | 빠른 테스트 | **복잡한 객체 (권장)** |

---

**작성일**: 2025-11-16
**버전**: 3.1 (Curvature-Stratified Poisson Disk 추가)
**업데이트**:
- v3.0: 바닥면 제거 및 수평면 tilting 기능 추가
- v3.1: Adaptive Poisson disk sampling 추가 (`--use_poisson_disk`)
