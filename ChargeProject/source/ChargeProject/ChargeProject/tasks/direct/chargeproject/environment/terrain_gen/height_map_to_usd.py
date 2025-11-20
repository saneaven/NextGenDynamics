import numpy as np
from pxr import Usd, UsdGeom, Vt, Gf, UsdPhysics, UsdShade

def save_height_map_to_usd(heightmap, usd_path, meter_per_grid=0.1):
    """
    heightmap: (rows, cols) float32 numpy array (높이 데이터)
    usd_path: 저장할 .usd 파일 경로 (예: "terrain.usd")
    meter_per_grid: 격자 간격
    """
    rows, cols = heightmap.shape
    
    # 1. USD 스테이지 생성
    stage = Usd.Stage.CreateNew(usd_path)
    
    # 기본 UpAxis를 Z축으로 설정 (Isaac Sim 표준)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

    # 기본 단위를 미터로 설정
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    
    # 1) 루트 Xform 정의
    root_prim = stage.DefinePrim("/World", "Xform")
    stage.SetDefaultPrim(root_prim)   
    # World를 기본 prim으로
    # 2. Mesh Prim 정의 (/World/terrain 경로에 생성)
    mesh_path = "/World/terrain"
    mesh_prim = UsdGeom.Mesh.Define(stage, mesh_path)
    
    
    # -------------------------------------------------------
    # 3. 데이터 생성 (Vectorization으로 고속 처리)
    # -------------------------------------------------------
    
    # (1) 정점(Points) 생성: (x, y, z)
    # x: 행 방향, y: 열 방향
    x_length = (cols-1) * meter_per_grid
    y_length = (rows-1) * meter_per_grid
    x = np.linspace(-x_length/2, x_length/2, rows)
    y = np.linspace(-y_length/2, y_length/2, cols)
    xv, yv = np.meshgrid(x, y, indexing='ij') # indexing='ij'로 행렬 좌표계 일치
    
    # 정점 배열 합치기 (N, 3)
    points = np.column_stack((xv.ravel(), yv.ravel(), heightmap.ravel()))
    
    # (2) 면(Topology) 생성: 격자를 삼각형으로 연결
    # Vertex 인덱스 계산을 위한 그리드 생성
    # 마지막 행과 열은 사각형의 시작점이 될 수 없으므로 제외
    r_idx = np.arange(rows - 1)
    c_idx = np.arange(cols - 1)
    rv, cv = np.meshgrid(r_idx, c_idx, indexing='ij')
    
    # 현재 점(top-left)의 1차원 인덱스들
    p_idxs = rv * cols + cv
    
    # 사각형 하나를 두 개의 삼각형으로 쪼개기 위한 인덱스 계산
    # Triangle 1: (tl, bl, tr) -> (idx, idx+cols, idx+1)
    # Triangle 2: (tr, bl, br) -> (idx+1, idx+cols, idx+cols+1)
    
    t1 = np.column_stack((p_idxs.ravel(), p_idxs.ravel() + cols, p_idxs.ravel() + 1))
    t2 = np.column_stack((p_idxs.ravel() + 1, p_idxs.ravel() + cols, p_idxs.ravel() + cols + 1))
    
    # 전체 인덱스 리스트 (Face Vertex Indices)
    face_vertex_indices = np.vstack((t1, t2)).reshape(-1)
    
    # 각 면은 삼각형이므로 점 3개씩 가짐 (Face Vertex Counts)
    face_vertex_counts = np.full(len(face_vertex_indices) // 3, 3)

    # -------------------------------------------------------
    # 4. USD 속성 설정 (pxr 타입으로 변환하여 입력)
    # -------------------------------------------------------
    
    # Points 설정
    mesh_prim.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(points.astype(np.float32)))
    
    # Topology 설정
    mesh_prim.GetFaceVertexCountsAttr().Set(Vt.IntArray.FromNumpy(face_vertex_counts.astype(np.int32)))
    mesh_prim.GetFaceVertexIndicesAttr().Set(Vt.IntArray.FromNumpy(face_vertex_indices.astype(np.int32)))
    
    # (선택) Extent(경계박스) 계산 - 퍼포먼스를 위해 중요
    mesh_prim.GetExtentAttr().Set(mesh_prim.ComputeExtent(mesh_prim.GetPointsAttr().Get()))
    
    # (1) CollisionAPI 적용: "이것은 충돌 가능한 물체입니다"
    UsdPhysics.CollisionAPI.Apply(mesh_prim.GetPrim())
    
    # (2) MeshCollisionAPI 적용: 충돌 모양 설정
    mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(mesh_prim.GetPrim())
    
    # 중요: 지형은 오목(Concave)하므로 Convex Hull로 감싸면 안 됩니다.
    # "none"으로 설정하면 원본 메쉬(Triangle Mesh) 그대로 충돌을 계산합니다.
    mesh_collision_api.CreateApproximationAttr().Set("none")
    
    # -------------------------------------------------------
    # [선택] 물리 재질(Material) 추가 (마찰력 등)
    # -------------------------------------------------------
    # 별도의 Material Prim 생성
    mat_path = "/World/material/terrain"
    material = UsdShade.Material.Define(stage, mat_path) # UsdShade import 필요
    
    # PhysicsMaterialAPI 적용
    phys_mat = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    phys_mat.CreateStaticFrictionAttr().Set(1.0)  # 정지 마찰력
    phys_mat.CreateDynamicFrictionAttr().Set(1.0) # 운동 마찰력
    phys_mat.CreateRestitutionAttr().Set(0.0)     # 반발력(통통 튀는 정도)

    # 메쉬에 재질 바인딩 (Binding)
    UsdShade.MaterialBindingAPI.Apply(mesh_prim.GetPrim()).Bind(material)

    # 5. 저장
    stage.GetRootLayer().Save()
    print(f"Successfully saved USD directly to: {usd_path}")