# Isaac Lab/Omniverse Kit 런타임 내 파이썬
from pxr import Usd, Sdf
import omni.usd

stage = omni.usd.get_context().get_stage()  # 현재 USD Stage 얻기
world = stage.GetPrimAtPath(Sdf.Path("/World"))
print("World valid?", world.IsValid())

# /World 바로 아래 자식 프림들 찍기
for child in world.GetChildren():
    print("child:", child.GetPath())

# 내가 찾는 경로가 실제로 유효한지 검사
p = stage.GetPrimAtPath(Sdf.Path("/World/terrain"))
print("/World/terrain valid?", p.IsValid())
