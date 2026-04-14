import json
from pprint import pprint

json_path = r"C:\Users\hyi8402\Downloads\dmd\gA\1\s1\gA_1_s1_2019-03-08T09;31;15+01;00_rgb_ann_distraction.json"   # 여기 경로 바꿔

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

print("=== 최상위 key ===")
print(data.keys())

openlabel = data.get("openlabel", {})
print("\n=== openlabel key ===")
print(openlabel.keys())

frames = openlabel.get("frames", {})
print(f"\n총 frame annotation 수: {len(frames)}")

# 첫 frame 하나만 구조 확인
if frames:
    first_frame_id = sorted(frames.keys(), key=lambda x: int(x))[0]
    print(f"\n=== 첫 frame id: {first_frame_id} ===")
    pprint(frames[first_frame_id])