import json
import os
from collections import defaultdict

def coco_keypoints_to_yolo_pose(annotation_path, labels_dir):
    os.makedirs(labels_dir, exist_ok=True)

    with open(annotation_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    images = {img["id"]: img for img in coco["images"]}
    anns_by_image = defaultdict(list)

    for ann in coco["annotations"]:
        if ann.get("iscrowd", 0) == 1:
            continue
        if ann.get("category_id", None) != 1:
            continue

        kpts = ann.get("keypoints", [])
        if not kpts or sum(kpts[2::3]) == 0:
            continue

        anns_by_image[ann["image_id"]].append(ann)

    written = 0
    for image_id, anns in anns_by_image.items():
        img_info = images[image_id]
        img_w = img_info["width"]
        img_h = img_info["height"]

        file_name = img_info["file_name"]
        stem = os.path.splitext(file_name)[0]
        label_path = os.path.join(labels_dir, stem + ".txt")

        lines = []
        for ann in anns:
            x, y, w, h = ann["bbox"]

            cx = (x + w / 2) / img_w
            cy = (y + h / 2) / img_h
            nw = w / img_w
            nh = h / img_h

            row = [0, cx, cy, nw, nh]  # class=0 -> person

            keypoints = ann["keypoints"]
            for i in range(0, len(keypoints), 3):
                kx = keypoints[i]
                ky = keypoints[i + 1]
                kv = keypoints[i + 2]

                if kv > 0:
                    kx /= img_w
                    ky /= img_h
                else:
                    kx = 0.0
                    ky = 0.0

                row.extend([kx, ky, kv])

            lines.append(" ".join(map(str, row)))

        with open(label_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        written += 1

    print(f"[DONE] {annotation_path}")
    print(f"[LABELS] written: {written}")


if __name__ == "__main__":
    base_dir = r"C:\Users\hyi8402\Desktop\캡스톤\Dataset"

    train_json = os.path.join(base_dir, "annotations", "person_keypoints_train2017.json")
    val_json = os.path.join(base_dir, "annotations", "person_keypoints_val2017.json")

    train_labels = os.path.join(base_dir, "labels", "train2017")
    val_labels = os.path.join(base_dir, "labels", "val2017")

    coco_keypoints_to_yolo_pose(train_json, train_labels)
    coco_keypoints_to_yolo_pose(val_json, val_labels)