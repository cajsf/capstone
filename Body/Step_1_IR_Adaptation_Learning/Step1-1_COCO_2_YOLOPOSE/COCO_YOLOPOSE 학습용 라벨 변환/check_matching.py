import os

def check_match(image_dir, label_dir):
    img_stems = {os.path.splitext(f)[0] for f in os.listdir(image_dir) if f.lower().endswith(".jpg")}
    lbl_stems = {os.path.splitext(f)[0] for f in os.listdir(label_dir) if f.lower().endswith(".txt")}

    print(f"[IMAGE DIR] {image_dir}")
    print(f"[LABEL DIR] {label_dir}")
    print("images:", len(img_stems))
    print("labels:", len(lbl_stems))
    print("matched:", len(img_stems & lbl_stems))
    print("labels without image:", len(lbl_stems - img_stems))
    print("images without label:", len(img_stems - lbl_stems))
    print("-" * 50)

base_dir = r"C:\Users\hyi8402\Desktop\Capstone\Dataset"

check_match(
    os.path.join(base_dir, "train2017"),
    os.path.join(base_dir, "labels", "train2017")
)

check_match(
    os.path.join(base_dir, "val2017"),
    os.path.join(base_dir, "labels", "val2017")
)