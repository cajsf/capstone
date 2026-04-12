

import cv2
from ultralytics import YOLO

# ===== 설정 =====
model_path = r"C:\Users\hyi8402\Desktop\Capstone\Code\2.code\runs\pose\runs_pose\pose_debug5\weights\best.pt"
img_path = r"C:\Users\hyi8402\Downloads\KakaoTalk_20260327_161718572.png"  # ← 여기만 바꿔

# ===== 모델 로드 =====
model = YOLO(model_path)

# ===== 추론 =====
results = model.predict(
    source=img_path,
    conf=0.25,
    verbose=False
)

# ===== 시각화 =====
vis = results[0].plot()

cv2.imshow("Pose Inference", vis)
cv2.waitKey(0)
cv2.destroyAllWindows()