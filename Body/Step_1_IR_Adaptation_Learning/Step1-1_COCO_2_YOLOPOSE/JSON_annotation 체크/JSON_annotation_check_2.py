import json

json_path = r"C:\Users\hyi8402\Downloads\dmd\gA\1\s1\gA_1_s1_2019-03-08T09;31;15+01;00_rgb_ann_distraction.json"

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

actions = data["openlabel"]["actions"]

for action_id, action_data in actions.items():
    print(f"\n=== action {action_id} ===")

    # 행동 이름
    print("type:", action_data.get("type"))

    # frame interval
    intervals = action_data.get("frame_intervals", [])
    print("interval 개수:", len(intervals))

    for i, interval in enumerate(intervals[:5]):
        print(f"  interval {i}: {interval}")