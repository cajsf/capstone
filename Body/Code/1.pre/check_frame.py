import cv2

video_path = r"C:\Users\hyi8402\Downloads\dmd\gA\1\s1\gA_1_s1_2019-03-08T09;31;15+01;00_ir_hands.mp4"

cap = cv2.VideoCapture(video_path)

count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    count += 1

print(f"실제 프레임 수: {count}")

cap.release()