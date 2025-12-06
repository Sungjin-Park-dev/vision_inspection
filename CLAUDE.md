# Project: Vision Inspection Pipeline (cuRobo/Jupyter)

## 1. Project Goal
- **Context**: 4단계 검사 파이프라인(Viewpoint -> IK -> GTSP -> Collision)을 교육용 Notebook으로 통합.
- **Refactoring Goal**: `scripts/` 및 `common/` 모듈의 **API 일관성 유지** 및 **Tensor 연산 최적화**.
- **Constraint**: 리팩토링 후에도 `vision_inspection_pipeline.ipynb`가 수정 없이 작동해야 함.

## 2. Refactoring Principles
- **Notebook Compatibility**: 함수 시그니처 변경 시, 호출부(Notebook) 호환성 최우선 고려.
- **I/O Isolation**: 데이터 로딩/저장(HDF5/CSV)은 `common/`이나 별도 함수로 격리하여 비즈니스 로직 순수성 유지.