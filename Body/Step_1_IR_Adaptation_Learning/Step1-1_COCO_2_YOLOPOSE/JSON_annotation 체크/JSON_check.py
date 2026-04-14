import json
import os

def filter_coco_images(
    json_path,
    image_dir,
    remove_no_person=True
):
    with open(json_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    # 1. image_id → file_name 매핑
    id_to_filename = {img["id"]: img["file_name"] for img in coco["images"]}

    # 2. annotation 있는 image_id 수집
    image_ids_with_ann = set()
    for ann in coco["annotations"]:
        if ann.get("iscrowd", 0) == 1:
            continue
        if ann.get("category_id", None) != 1:
            continue

        # keypoints 없는 경우 제외
        kpts = ann.get("keypoints", [])
        if sum(kpts[2::3]) == 0:
            continue

        image_ids_with_ann.add(ann["image_id"])

    # 3. 유지할 파일 목록
    keep_files = set()

    for img_id, file_name in id_to_filename.items():
        if remove_no_person:
            if img_id in image_ids_with_ann:
                keep_files.add(file_name)
        else:
            keep_files.add(file_name)

    # 4. 실제 이미지 폴더에서 삭제
    all_files = os.listdir(image_dir)

    removed = 0
    for file_name in all_files:
        if file_name not in keep_files:
            path = os.path.join(image_dir, file_name)
            os.remove(path)
            removed += 1

    print(f"[DONE] removed {removed} images")
    print(f"[KEEP] {len(keep_files)} images")


if __name__ == "__main__":
    base_dir = r"C:\Users\hyi8402\Desktop\캡스톤\Dataset"

    train_json = os.path.join(base_dir, "annotations", "person_keypoints_train2017.json")
    train_images = os.path.join(base_dir, "train2017")

    filter_coco_images(
        json_path=train_json,
        image_dir=train_images,
        remove_no_person=True  # 사람 없는 이미지도 제거
    )