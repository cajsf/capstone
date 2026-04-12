from ultralytics import YOLO

def main():
    model = YOLO("yolov8n-pose.pt")

    model.train(
        data=r"C:\Users\hyi8402\Desktop\Capstone\Dataset\coco_pose.yaml",
        epochs=50,
        imgsz=640,
        batch=48,
        device=0,
        workers=2,
        project="runs_pose",
        name="pose_debug",
        pretrained=True,
        cache=False,
        save=True,          # 중요
        save_period=1       # 매 epoch마다 저장   
    )

if __name__ == "__main__":
    main()