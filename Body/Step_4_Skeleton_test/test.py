import cv2
import math
import numpy as np
import torch
import torch.nn as nn
from collections import deque
from ultralytics import YOLO

# =========================
# 경로 / 설정
# =========================
VIDEO_PATH = r"C:\Users\hyi8402\Downloads\dmd\gZ\34\s7\gZ_34_s3_2019-04-04T15;33;18+02;00_ir_body.mp4"
POSE_MODEL_PATH = r"C:\Users\hyi8402\Downloads\best.pt"
#POSE_MODEL_PATH = r"C:\Users\hyi8402\Desktop\Capstone\Body\Step_1_IR_Adaptation_Learning\Step1-1_COCO_2_YOLOPOSE\COCO_POSE_result\weights\best.pt"
CLASSIFIER_PATH = r"C:\Users\hyi8402\Desktop\Capstone\Body\Code\3.result\record\tgcn_tsam_singleview_window_04122013\best.pt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
POSE_CONF_THRES = 0.25
INFER_EVERY = 1  # 1이면 매 프레임 추론, 느리면 2~3

# 저장 옵션
SAVE_OUTPUT = False
SAVE_PATH = r"C:\Users\hyi8402\Desktop\Capstone\Body\Code\3.result\demo_out.mp4"

# =========================
# 스켈레톤 정의
# =========================
NUM_JOINTS = 17
COCO_EDGES = [
    (0, 1), (0, 2),
    (1, 3), (2, 4),
    (5, 6),
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 11), (6, 12),
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16)
]

COCO_PARENTS = {
    0: 0,
    1: 0, 2: 0,
    3: 1, 4: 2,
    5: 0, 6: 0,
    7: 5, 8: 6,
    9: 7, 10: 8,
    11: 5, 12: 6,
    13: 11, 14: 12,
    15: 13, 16: 14
}

# =========================
# 유틸
# =========================
def build_adjacency(num_joints=17, edges=None, self_link=True):
    A = np.zeros((num_joints, num_joints), dtype=np.float32)

    if self_link:
        for i in range(num_joints):
            A[i, i] = 1.0

    for i, j in edges:
        A[i, j] = 1.0
        A[j, i] = 1.0

    D = np.sum(A, axis=1)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(D + 1e-6))
    A = D_inv_sqrt @ A @ D_inv_sqrt
    return A.astype(np.float32)

def center_scale_normalize_xy(xy, conf=None):
    """
    xy: (T, V, 2)
    conf: (T, V) or None
    """
    out = xy.copy()
    T, V, _ = out.shape

    for t in range(T):
        pts = out[t]

        if conf is not None:
            valid = conf[t] > 0
        else:
            valid = ~np.isnan(pts).any(axis=1)

        valid_pts = pts[valid]

        if len(valid_pts) < 2:
            out[t] = 0.0
            continue

        center = valid_pts.mean(axis=0)
        pts = pts - center

        min_xy = valid_pts.min(axis=0)
        max_xy = valid_pts.max(axis=0)
        scale = np.linalg.norm(max_xy - min_xy)
        if scale < 1e-6:
            scale = 1.0

        pts = pts / scale
        pts[~valid] = 0.0
        out[t] = pts

    return out

def compute_bone_feature(xy, parents):
    """
    xy: (T, V, 2)
    """
    T, V, C = xy.shape
    bone = np.zeros_like(xy, dtype=np.float32)

    for v in range(V):
        p = parents[v]
        if p == v:
            bone[:, v, :] = 0.0
        else:
            bone[:, v, :] = xy[:, v, :] - xy[:, p, :]
    return bone

def compute_velocity_feature(xy):
    """
    xy: (T, V, 2)
    """
    vel = np.zeros_like(xy, dtype=np.float32)
    if xy.shape[0] >= 2:
        vel[1:] = xy[1:] - xy[:-1]
    return vel

def get_window_valid_stats(conf_win, min_joint_ratio=0.15):
    joint_ratio_per_frame = np.mean(conf_win > 0, axis=1)
    valid_frame_mask = joint_ratio_per_frame >= min_joint_ratio
    n_valid_frames = int(valid_frame_mask.sum())
    valid_ratio = float(n_valid_frames / max(len(conf_win), 1))
    return n_valid_frames, valid_ratio

def choose_best_person(result):
    if result.keypoints is None:
        return None

    xy = result.keypoints.xy
    conf = result.keypoints.conf

    if xy is None or len(xy) == 0:
        return None

    xy = xy.cpu().numpy()
    conf = conf.cpu().numpy() if conf is not None else None

    if conf is None:
        return {
            "xy": xy[0],
            "conf": np.ones((xy.shape[1],), dtype=np.float32)
        }

    mean_conf = np.nanmean(conf, axis=1)
    best_idx = int(np.argmax(mean_conf))

    return {
        "xy": xy[best_idx],
        "conf": conf[best_idx]
    }

def draw_skeleton(frame, kpts, conf=None, conf_thres=0.1):
    if kpts is None:
        return frame

    # 선
    for i, j in COCO_EDGES:
        xi, yi = kpts[i]
        xj, yj = kpts[j]

        vi = not np.isnan(xi) and not np.isnan(yi)
        vj = not np.isnan(xj) and not np.isnan(yj)

        if conf is not None:
            vi = vi and (conf[i] > conf_thres)
            vj = vj and (conf[j] > conf_thres)

        if vi and vj:
            cv2.line(frame, (int(xi), int(yi)), (int(xj), int(yj)), (0, 255, 0), 2)

    # 점
    for idx, (x, y) in enumerate(kpts):
        valid = not np.isnan(x) and not np.isnan(y)
        if conf is not None:
            valid = valid and (conf[idx] > conf_thres)

        if valid:
            cv2.circle(frame, (int(x), int(y)), 3, (0, 0, 255), -1)

    return frame

# =========================
# 모델 정의 (학습 코드 동일)
# =========================
class GraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, A):
        super().__init__()
        self.register_buffer("A", torch.tensor(A, dtype=torch.float32))
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.conv(x)
        x = torch.einsum("nctv,vw->nctw", x, self.A)
        return x


class TGCBlock(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, dropout=0.1):
        super().__init__()
        self.gconv = GraphConv(in_channels, out_channels, A)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()

        self.tconv = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=(9, 1),
            stride=(stride, 1),
            padding=(4, 0),
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(dropout)

        if in_channels == out_channels and stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1), bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        res = self.residual(x)

        out = self.gconv(x)
        out = self.bn1(out)
        out = self.act(out)

        out = self.tconv(out)
        out = self.bn2(out)
        out = self.dropout(out)

        out = out + res
        out = self.act(out)
        return out


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)

        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        L = x.size(1)
        return x + self.pe[:, :L, :]


class TSAM(nn.Module):
    def __init__(self, channels, num_heads=4, dropout=0.1, max_len=4096):
        super().__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.pos_enc = SinusoidalPositionalEncoding(channels, max_len=max_len)

        self.attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.dropout = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(channels)

        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels * 4, channels),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        N, C, T, V = x.shape

        tokens = x.permute(0, 2, 3, 1).contiguous().view(N, T * V, C)
        tokens = self.norm1(tokens)
        tokens = self.pos_enc(tokens)

        attn_out, _ = self.attn(tokens, tokens, tokens, need_weights=False)
        tokens = tokens + self.dropout(attn_out)
        tokens = tokens + self.ffn(self.norm2(tokens))

        out = tokens.view(N, T, V, C).permute(0, 3, 1, 2).contiguous()
        return out


class TGCN_TSAM_Classifier(nn.Module):
    def __init__(self, in_channels, num_classes, A, num_joints=17,
                 model_dropout=0.1, tsam_heads=4, tsam_dropout=0.1, head_dropout=0.3):
        super().__init__()
        self.num_joints = num_joints

        self.data_bn = nn.BatchNorm1d(in_channels * num_joints)

        self.layer1 = TGCBlock(in_channels, 64, A, stride=1, dropout=model_dropout)
        self.layer2 = TGCBlock(64, 64, A, stride=1, dropout=model_dropout)
        self.layer3 = TGCBlock(64, 128, A, stride=2, dropout=model_dropout)
        self.layer4 = TGCBlock(128, 128, A, stride=1, dropout=model_dropout)
        self.layer5 = TGCBlock(128, 256, A, stride=2, dropout=model_dropout)
        self.layer6 = TGCBlock(256, 256, A, stride=1, dropout=model_dropout)

        self.tsam = TSAM(
            channels=256,
            num_heads=tsam_heads,
            dropout=tsam_dropout,
            max_len=4096
        )

        self.head = nn.Sequential(
            nn.Dropout(head_dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # x: (N, C, T, V)
        N, C, T, V = x.shape

        x = x.permute(0, 1, 3, 2).contiguous().view(N, C * V, T)
        x = self.data_bn(x)
        x = x.view(N, C, V, T).permute(0, 1, 3, 2).contiguous()

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)

        x = self.tsam(x)

        x = x.mean(dim=-1).mean(dim=-1)
        x = self.head(x)
        return x

# =========================
# 체크포인트 로드
# =========================
ckpt = torch.load(CLASSIFIER_PATH, map_location=DEVICE)

config = ckpt["config"]
label2id = ckpt["label2id"]
id2label = {v: k for k, v in label2id.items()}

WINDOW_SIZE = config["WINDOW_SIZE"]
MIN_WINDOW_VALID_FRAMES = config["MIN_WINDOW_VALID_FRAMES"]
MIN_WINDOW_VALID_RATIO = config["MIN_WINDOW_VALID_RATIO"]
MIN_WINDOW_VALID_JOINT_RATIO = config["MIN_WINDOW_VALID_JOINT_RATIO"]

USE_CONF_CHANNEL = config["USE_CONF_CHANNEL"]
USE_BONE_FEATURE = config["USE_BONE_FEATURE"]
USE_VELOCITY_FEATURE = config["USE_VELOCITY_FEATURE"]

TSAM_NUM_HEADS = config["TSAM_NUM_HEADS"]
TSAM_DROPOUT = config["TSAM_DROPOUT"]
MODEL_DROPOUT = config["MODEL_DROPOUT"]
HEAD_DROPOUT = config["HEAD_DROPOUT"]

in_channels = 2
if USE_BONE_FEATURE:
    in_channels += 2
if USE_VELOCITY_FEATURE:
    in_channels += 2
if USE_CONF_CHANNEL:
    in_channels += 1

A = build_adjacency(NUM_JOINTS, COCO_EDGES, self_link=True)

model = TGCN_TSAM_Classifier(
    in_channels=in_channels,
    num_classes=len(label2id),
    A=A,
    num_joints=NUM_JOINTS,
    model_dropout=MODEL_DROPOUT,
    tsam_heads=TSAM_NUM_HEADS,
    tsam_dropout=TSAM_DROPOUT,
    head_dropout=HEAD_DROPOUT
).to(DEVICE)

model.load_state_dict(ckpt["model_state_dict"])
model.eval()

print("[분류기 로드 완료]")
print(f"WINDOW_SIZE={WINDOW_SIZE}, in_channels={in_channels}, num_classes={len(label2id)}")

# =========================
# 포즈 모델 로드
# =========================
pose_model = YOLO(POSE_MODEL_PATH)
print("[포즈 모델 로드 완료]")

# =========================
# 실시간 전처리
# =========================
def make_model_input_from_buffer(kpt_buffer, conf_buffer):
    """
    kpt_buffer: deque of (17,2)
    conf_buffer: deque of (17,)
    return: x tensor (1, C, T, V) or None
    """
    keypoints = np.stack(kpt_buffer, axis=0).astype(np.float32)  # (T, V, 2)
    conf = np.stack(conf_buffer, axis=0).astype(np.float32)      # (T, V)

    conf = np.nan_to_num(conf, nan=0.0, posinf=0.0, neginf=0.0)
    keypoints = center_scale_normalize_xy(keypoints, conf=conf)
    keypoints = np.nan_to_num(keypoints, nan=0.0, posinf=0.0, neginf=0.0)

    n_valid_frames, valid_ratio = get_window_valid_stats(
        conf,
        min_joint_ratio=MIN_WINDOW_VALID_JOINT_RATIO
    )

    if n_valid_frames < MIN_WINDOW_VALID_FRAMES or valid_ratio < MIN_WINDOW_VALID_RATIO:
        return None, n_valid_frames, valid_ratio

    feats = [keypoints]

    if USE_BONE_FEATURE:
        bone = compute_bone_feature(keypoints, COCO_PARENTS)
        feats.append(bone)

    if USE_VELOCITY_FEATURE:
        vel = compute_velocity_feature(keypoints)
        feats.append(vel)

    if USE_CONF_CHANNEL:
        feats.append(conf[..., None])

    x = np.concatenate(feats, axis=-1).astype(np.float32)  # (T, V, C)
    x = np.transpose(x, (2, 0, 1)).astype(np.float32)      # (C, T, V)
    x = torch.from_numpy(x).unsqueeze(0).to(DEVICE)        # (1, C, T, V)

    return x, n_valid_frames, valid_ratio

# =========================
# 비디오 시작
# =========================
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError(f"비디오 열기 실패: {VIDEO_PATH}")

fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0:
    fps = 30.0

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

writer = None
if SAVE_OUTPUT:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(SAVE_PATH, fourcc, fps, (width, height))

kpt_buffer = deque(maxlen=WINDOW_SIZE)
conf_buffer = deque(maxlen=WINDOW_SIZE)

frame_idx = 0
last_pred = "..."
last_prob = 0.0
last_valid_frames = 0
last_valid_ratio = 0.0

fps_ma = deque(maxlen=30)

print("[추론 시작] ESC 누르면 종료")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    t0 = cv2.getTickCount()

    results = pose_model.predict(
        source=frame,
        conf=POSE_CONF_THRES,
        verbose=False,
        device=0 if DEVICE == "cuda" else "cpu"
    )

    picked = None
    if results:
        picked = choose_best_person(results[0])

    if picked is not None:
        kpts = picked["xy"].astype(np.float32)      # (17,2)
        conf = picked["conf"].astype(np.float32)    # (17,)
    else:
        kpts = np.full((NUM_JOINTS, 2), np.nan, dtype=np.float32)
        conf = np.zeros((NUM_JOINTS,), dtype=np.float32)

    kpt_buffer.append(kpts)
    conf_buffer.append(conf)

    # 시각화
    draw_skeleton(frame, kpts, conf=conf, conf_thres=0.1)

    # 추론
    if len(kpt_buffer) == WINDOW_SIZE and frame_idx % INFER_EVERY == 0:
        x, n_valid_frames, valid_ratio = make_model_input_from_buffer(kpt_buffer, conf_buffer)
        last_valid_frames = n_valid_frames
        last_valid_ratio = valid_ratio

        if x is not None:
            with torch.no_grad():
                logits = model(x)
                prob = torch.softmax(logits, dim=1)[0]
                pred = int(torch.argmax(prob).item())

            last_pred = id2label[pred]
            last_prob = float(prob[pred].item())
        else:
            last_pred = "invalid_window"
            last_prob = 0.0

    frame_idx += 1

    # FPS 계산
    infer_time = (cv2.getTickCount() - t0) / cv2.getTickFrequency()
    curr_fps = 1.0 / max(infer_time, 1e-6)
    fps_ma.append(curr_fps)
    avg_fps = sum(fps_ma) / len(fps_ma)

    # 오버레이
    cv2.putText(frame, f"Pred: {last_pred}", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    cv2.putText(frame, f"Prob: {last_prob:.3f}", (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    cv2.putText(frame, f"FPS: {avg_fps:.1f}", (20, 105),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, f"Valid frames: {last_valid_frames}/{WINDOW_SIZE}", (20, 140),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 255, 200), 2)
    cv2.putText(frame, f"Valid ratio: {last_valid_ratio:.2f}", (20, 170),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 255, 200), 2)

    cv2.imshow("DMD Real-time", frame)

    if writer is not None:
        writer.write(frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break

cap.release()
if writer is not None:
    writer.release()
cv2.destroyAllWindows()
print("[종료]")