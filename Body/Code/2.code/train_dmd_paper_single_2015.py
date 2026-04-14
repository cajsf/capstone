import os
import json
import math
import random
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split


# =========================================================
# 설정
# =========================================================
SEED = 42

DATA_ROOT = r"C:\Users\hyi8402\Desktop\Capstone\Body\Step_2_IR_Skeleton_Extract\dmd_finetune"
SAVE_ROOT = r"C:\Users\hyi8402\Desktop\Capstone\Body\Step_3_IR_Skeleton_dmd_train"

USE_LABEL_KEY = "primary_label"

# clip 단위 최소 유효 프레임
MIN_VALID_FRAMES = 16

# ---- window 기반 ----
WINDOW_SIZE = 48
WINDOW_STRIDE = 24
MAX_WINDOWS_PER_CLIP = None  # None이면 전부 사용

# window 품질 필터
MIN_WINDOW_VALID_FRAMES = 16
MIN_WINDOW_VALID_RATIO = 0.35
MIN_WINDOW_VALID_JOINT_RATIO = 0.15  # 프레임별로 유효 joint 비율이 너무 낮으면 invalid로 간주

BATCH_SIZE = 32
NUM_WORKERS = 0
EPOCHS = 100
LR = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 10
GRAD_CLIP_NORM = 1.0

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

USE_CONF_CHANNEL = True
USE_BONE_FEATURE = True
USE_VELOCITY_FEATURE = True

DROP_EMPTY_CLIPS = True
USE_CLASS_WEIGHT = True

# 구조가 바뀌었으므로 기본은 False 권장
RESUME = False
RESUME_PATH = None
SAVE_EVERY_EPOCH = True

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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

# COCO parent 정의
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

# TSAM 설정
TSAM_NUM_HEADS = 4
TSAM_DROPOUT = 0.1
MODEL_DROPOUT = 0.1
HEAD_DROPOUT = 0.3

# clip-level aggregation
CLIP_AGG_MODE = "topk_mean"   # "mean" or "topk_mean"
CLIP_TOPK = 3


# =========================================================
# 기본 유틸
# =========================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def center_crop_indices(length, target_len):
    if length <= 0:
        return np.zeros((target_len,), dtype=np.int64)

    if length >= target_len:
        start = (length - target_len) // 2
        return np.arange(start, start + target_len, dtype=np.int64)

    pad_total = target_len - length
    left_pad = pad_total // 2
    right_pad = target_len - length - left_pad

    center = np.arange(length, dtype=np.int64)
    left = np.full((left_pad,), 0, dtype=np.int64)
    right = np.full((right_pad,), length - 1, dtype=np.int64)

    return np.concatenate([left, center, right], axis=0)


def sliding_window_indices(length, window_size, stride):
    if length <= 0:
        return [np.zeros((window_size,), dtype=np.int64)]

    if length <= window_size:
        return [center_crop_indices(length, window_size)]

    windows = []
    start = 0
    while start + window_size <= length:
        windows.append(np.arange(start, start + window_size, dtype=np.int64))
        start += stride

    last_start = length - window_size
    last_window = np.arange(last_start, last_start + window_size, dtype=np.int64)

    if len(windows) == 0 or not np.array_equal(windows[-1], last_window):
        windows.append(last_window)

    return windows


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


def valid_frame_count(conf, min_joint_ratio=0.15):
    if conf is None:
        return 0
    joint_ratio = np.mean(conf > 0, axis=1)
    return int(np.sum(joint_ratio >= min_joint_ratio))


def compute_bone_feature(xy, parents):
    """
    xy: (T, V, 2)
    return bone: (T, V, 2)
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
    return vel: (T, V, 2)
    """
    vel = np.zeros_like(xy, dtype=np.float32)
    if xy.shape[0] >= 2:
        vel[1:] = xy[1:] - xy[:-1]
    return vel


def get_window_valid_stats(conf_win, min_joint_ratio=0.15):
    """
    conf_win: (Tw, V)
    """
    joint_ratio_per_frame = np.mean(conf_win > 0, axis=1)
    valid_frame_mask = joint_ratio_per_frame >= min_joint_ratio
    n_valid_frames = int(valid_frame_mask.sum())
    valid_ratio = float(n_valid_frames / max(len(conf_win), 1))
    return n_valid_frames, valid_ratio


# =========================================================
# 샘플 스캔
# =========================================================
def scan_clip_samples(data_root, label_key="primary_label"):
    data_root = Path(data_root)
    samples = []

    for clip_dir, _, files in os.walk(data_root):
        clip_dir = Path(clip_dir)
        files = set(files)

        if "skeleton_ir.npz" not in files:
            continue
        if "annotation.json" not in files:
            continue

        ann_path = clip_dir / "annotation.json"
        skel_path = clip_dir / "skeleton_ir.npz"

        try:
            ann = load_json(ann_path)
            label = ann.get(label_key, None)
            if label is None:
                continue

            npz = np.load(skel_path, allow_pickle=True)
            conf = npz["conf"] if "conf" in npz.files else None
            n_valid = valid_frame_count(conf, min_joint_ratio=MIN_WINDOW_VALID_JOINT_RATIO) if conf is not None else 0

            samples.append({
                "clip_dir": str(clip_dir),
                "skeleton_path": str(skel_path),
                "annotation_path": str(ann_path),
                "label_text": label,
                "n_valid_frames": n_valid,
            })

        except Exception as e:
            print(f"[스킵] {clip_dir} | 오류: {e}")

    return samples


def filter_samples(samples, min_valid_frames=16, drop_empty_clips=True):
    out = []
    for s in samples:
        if drop_empty_clips and s["n_valid_frames"] < min_valid_frames:
            continue
        out.append(s)
    return out


def build_label_map(samples):
    labels = sorted(list({s["label_text"] for s in samples}))
    label2id = {lb: i for i, lb in enumerate(labels)}
    id2label = {i: lb for lb, i in label2id.items()}
    return label2id, id2label


def stratified_split_samples(samples, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    labels = [s["label_text"] for s in samples]

    train_samples, temp_samples = train_test_split(
        samples,
        test_size=(1.0 - train_ratio),
        random_state=seed,
        stratify=labels
    )

    temp_labels = [s["label_text"] for s in temp_samples]
    val_portion_in_temp = val_ratio / (val_ratio + test_ratio)

    val_samples, test_samples = train_test_split(
        temp_samples,
        test_size=(1.0 - val_portion_in_temp),
        random_state=seed,
        stratify=temp_labels
    )

    return train_samples, val_samples, test_samples


# =========================================================
# 전처리 + preload
# =========================================================
def preprocess_sample_to_windows(
    sample,
    label2id,
    window_size=48,
    stride=24,
    use_conf_channel=True,
    use_bone_feature=True,
    use_velocity_feature=True,
    max_windows_per_clip=None,
    min_window_valid_frames=16,
    min_window_valid_ratio=0.35,
    min_window_valid_joint_ratio=0.15
):
    npz = np.load(sample["skeleton_path"], allow_pickle=True)

    keypoints = npz["keypoints"].astype(np.float32)  # (T, V, 2)
    conf = npz["conf"].astype(np.float32) if "conf" in npz.files else None

    if conf is not None:
        conf = np.nan_to_num(conf, nan=0.0, posinf=0.0, neginf=0.0)

    keypoints = center_scale_normalize_xy(keypoints, conf=conf)
    keypoints = np.nan_to_num(keypoints, nan=0.0, posinf=0.0, neginf=0.0)

    T, V, _ = keypoints.shape
    window_indices = sliding_window_indices(T, window_size, stride)

    if max_windows_per_clip is not None and len(window_indices) > max_windows_per_clip:
        select_idx = np.linspace(0, len(window_indices) - 1, max_windows_per_clip, dtype=int)
        window_indices = [window_indices[i] for i in select_idx]

    if conf is None:
        conf = np.ones((T, V), dtype=np.float32)

    bone = compute_bone_feature(keypoints, COCO_PARENTS) if use_bone_feature else None
    vel = compute_velocity_feature(keypoints) if use_velocity_feature else None

    y = label2id[sample["label_text"]]
    out = []

    for win_idx, idxs in enumerate(window_indices):
        kp_win = keypoints[idxs]    # (Tw, V, 2)
        conf_win = conf[idxs]       # (Tw, V)

        n_valid_frames, valid_ratio = get_window_valid_stats(
            conf_win,
            min_joint_ratio=min_window_valid_joint_ratio
        )
        if n_valid_frames < min_window_valid_frames or valid_ratio < min_window_valid_ratio:
            continue

        feats = [kp_win]

        if use_bone_feature:
            bone_win = bone[idxs]
            feats.append(bone_win)

        if use_velocity_feature:
            vel_win = vel[idxs]
            feats.append(vel_win)

        if use_conf_channel:
            feats.append(conf_win[..., None])

        x = np.concatenate(feats, axis=-1).astype(np.float32)  # (T,V,C)
        x = np.transpose(x, (2, 0, 1)).astype(np.float32)      # (C,T,V)

        out.append({
            "x": x,
            "y": y,
            "clip_dir": sample["clip_dir"],
            "label_text": sample["label_text"],
            "window_idx": win_idx,
            "num_windows_in_clip": len(window_indices),
            "n_valid_frames_window": n_valid_frames,
            "valid_ratio_window": valid_ratio,
        })

    return out


def preload_samples(
    samples,
    label2id,
    window_size=48,
    stride=24,
    use_conf_channel=True,
    use_bone_feature=True,
    use_velocity_feature=True,
    desc="Preloading",
    max_windows_per_clip=None,
    min_window_valid_frames=16,
    min_window_valid_ratio=0.35,
    min_window_valid_joint_ratio=0.15
):
    memory_data = []
    pbar = tqdm(
        samples,
        desc=desc,
        ncols=120,
        leave=True,
        ascii=True,
        mininterval=0.3
    )

    skipped_empty = 0

    for s in pbar:
        try:
            items = preprocess_sample_to_windows(
                s,
                label2id=label2id,
                window_size=window_size,
                stride=stride,
                use_conf_channel=use_conf_channel,
                use_bone_feature=use_bone_feature,
                use_velocity_feature=use_velocity_feature,
                max_windows_per_clip=max_windows_per_clip,
                min_window_valid_frames=min_window_valid_frames,
                min_window_valid_ratio=min_window_valid_ratio,
                min_window_valid_joint_ratio=min_window_valid_joint_ratio
            )
            if len(items) == 0:
                skipped_empty += 1
                continue
            memory_data.extend(items)
        except Exception as e:
            tqdm.write(f"[preload 스킵] {s['clip_dir']} | 오류: {e}")

    print(f"{desc} | window 0개가 되어 제외된 clip 수: {skipped_empty}")
    return memory_data


# =========================================================
# Dataset
# =========================================================
class MemorySkeletonWindowDataset(Dataset):
    def __init__(self, memory_data):
        self.memory_data = memory_data

    def __len__(self):
        return len(self.memory_data)

    def __getitem__(self, idx):
        item = self.memory_data[idx]
        return {
            "x": torch.from_numpy(item["x"]),
            "y": torch.tensor(item["y"], dtype=torch.long),
            "label_text": item["label_text"],
            "clip_dir": item["clip_dir"],
            "window_idx": torch.tensor(item["window_idx"], dtype=torch.long),
            "num_windows_in_clip": torch.tensor(item["num_windows_in_clip"], dtype=torch.long),
        }


# =========================================================
# 모델
# =========================================================
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
    def __init__(self, in_channels, num_classes, A, num_joints=17):
        super().__init__()
        self.num_joints = num_joints

        self.data_bn = nn.BatchNorm1d(in_channels * num_joints)

        self.layer1 = TGCBlock(in_channels, 64, A, stride=1, dropout=MODEL_DROPOUT)
        self.layer2 = TGCBlock(64, 64, A, stride=1, dropout=MODEL_DROPOUT)
        self.layer3 = TGCBlock(64, 128, A, stride=2, dropout=MODEL_DROPOUT)
        self.layer4 = TGCBlock(128, 128, A, stride=1, dropout=MODEL_DROPOUT)
        self.layer5 = TGCBlock(128, 256, A, stride=2, dropout=MODEL_DROPOUT)
        self.layer6 = TGCBlock(256, 256, A, stride=1, dropout=MODEL_DROPOUT)

        self.tsam = TSAM(
            channels=256,
            num_heads=TSAM_NUM_HEADS,
            dropout=TSAM_DROPOUT,
            max_len=4096
        )

        self.head = nn.Sequential(
            nn.Dropout(HEAD_DROPOUT),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
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


# =========================================================
# 평가 유틸
# =========================================================
def make_class_weights_from_memory(memory_data, num_classes):
    counts = np.zeros((num_classes,), dtype=np.float32)
    for item in memory_data:
        counts[item["y"]] += 1.0

    weights = 1.0 / np.maximum(counts, 1.0)
    weights = weights / weights.sum() * num_classes
    return torch.tensor(weights, dtype=torch.float32)


def save_checkpoint(path, model, optimizer, scheduler, epoch, best_val_f1, history, config, label2id):
    ckpt = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "best_val_f1": best_val_f1,
        "history": history,
        "config": config,
        "label2id": label2id
    }
    torch.save(ckpt, path)


def save_confusion_matrix(cm, labels, save_path):
    df = pd.DataFrame(cm, index=labels, columns=labels)
    df.to_csv(save_path, encoding="utf-8-sig")


def aggregate_probs_topk_mean(prob_list, topk=3):
    """
    prob_list: list of (C,)
    클래스별로 top-k probability 평균
    """
    probs = np.stack(prob_list, axis=0)  # (W, C)
    W, C = probs.shape
    k = min(topk, W)

    agg = np.zeros((C,), dtype=np.float32)
    for c in range(C):
        cls_scores = probs[:, c]
        topk_vals = np.sort(cls_scores)[-k:]
        agg[c] = float(np.mean(topk_vals))
    return agg


def aggregate_clip_level_results(targets, probs, clip_dirs, agg_mode="topk_mean", topk=3):
    clip_prob_sum = defaultdict(list)
    clip_target = {}

    for y, p, c in zip(targets, probs, clip_dirs):
        clip_prob_sum[c].append(np.asarray(p, dtype=np.float32))
        if c in clip_target:
            if clip_target[c] != y:
                raise ValueError(f"같은 clip_dir에 서로 다른 label 존재: {c}")
        else:
            clip_target[c] = y

    agg_targets = []
    agg_preds = []
    agg_probs = []
    agg_clip_dirs = []

    for c in clip_prob_sum.keys():
        prob_list = clip_prob_sum[c]

        if agg_mode == "mean":
            final_prob = np.mean(np.stack(prob_list, axis=0), axis=0)
        elif agg_mode == "topk_mean":
            final_prob = aggregate_probs_topk_mean(prob_list, topk=topk)
        else:
            raise ValueError(f"지원하지 않는 agg_mode: {agg_mode}")

        pred = int(np.argmax(final_prob))
        agg_targets.append(int(clip_target[c]))
        agg_preds.append(pred)
        agg_probs.append(final_prob)
        agg_clip_dirs.append(c)

    return {
        "targets": agg_targets,
        "preds": agg_preds,
        "probs": agg_probs,
        "clip_dirs": agg_clip_dirs,
        "acc": accuracy_score(agg_targets, agg_preds),
        "f1_macro": f1_score(agg_targets, agg_preds, average="macro")
    }


def run_one_epoch(model, loader, optimizer, criterion, device, train=True, epoch_idx=None, total_epochs=None):
    if train:
        model.train()
        phase = "Train"
    else:
        model.eval()
        phase = "Eval"

    total_loss = 0.0
    all_preds = []
    all_targets = []
    all_probs = []
    all_clip_dirs = []

    pbar_desc = f"[Epoch {epoch_idx}/{total_epochs}] {phase}" if epoch_idx is not None else phase
    pbar = tqdm(
        loader,
        desc=pbar_desc,
        leave=True,
        ncols=120,
        ascii=True,
        mininterval=0.3
    )

    for batch in pbar:
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)
        clip_dirs = batch["clip_dir"]

        if train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            logits = model(x)
            loss = criterion(logits, y)

            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                optimizer.step()

        total_loss += loss.item() * x.size(0)

        prob = torch.softmax(logits, dim=1)
        preds = torch.argmax(prob, dim=1)

        all_preds.extend(preds.detach().cpu().numpy().tolist())
        all_targets.extend(y.detach().cpu().numpy().tolist())
        all_probs.extend(prob.detach().cpu().numpy().tolist())
        all_clip_dirs.extend(list(clip_dirs))

        running_loss = total_loss / max(len(all_targets), 1)
        pbar.set_postfix(loss=f"{running_loss:.4f}")

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_targets, all_preds)
    f1_macro = f1_score(all_targets, all_preds, average="macro")

    result = {
        "loss": avg_loss,
        "acc": acc,
        "f1_macro": f1_macro,
        "targets": all_targets,
        "preds": all_preds,
        "probs": all_probs,
        "clip_dirs": all_clip_dirs,
    }

    if not train:
        clip_level = aggregate_clip_level_results(
            targets=all_targets,
            probs=all_probs,
            clip_dirs=all_clip_dirs,
            agg_mode=CLIP_AGG_MODE,
            topk=CLIP_TOPK
        )
        result["clip_level"] = clip_level

    return result


# =========================================================
# main
# =========================================================
def main():
    set_seed(SEED)
    ensure_dir(SAVE_ROOT)

    config = {
        "SEED": SEED,
        "DATA_ROOT": DATA_ROOT,
        "SAVE_ROOT": SAVE_ROOT,
        "USE_LABEL_KEY": USE_LABEL_KEY,
        "MIN_VALID_FRAMES": MIN_VALID_FRAMES,
        "WINDOW_SIZE": WINDOW_SIZE,
        "WINDOW_STRIDE": WINDOW_STRIDE,
        "MAX_WINDOWS_PER_CLIP": MAX_WINDOWS_PER_CLIP,
        "MIN_WINDOW_VALID_FRAMES": MIN_WINDOW_VALID_FRAMES,
        "MIN_WINDOW_VALID_RATIO": MIN_WINDOW_VALID_RATIO,
        "MIN_WINDOW_VALID_JOINT_RATIO": MIN_WINDOW_VALID_JOINT_RATIO,
        "BATCH_SIZE": BATCH_SIZE,
        "NUM_WORKERS": NUM_WORKERS,
        "EPOCHS": EPOCHS,
        "LR": LR,
        "WEIGHT_DECAY": WEIGHT_DECAY,
        "PATIENCE": PATIENCE,
        "GRAD_CLIP_NORM": GRAD_CLIP_NORM,
        "TRAIN_RATIO": TRAIN_RATIO,
        "VAL_RATIO": VAL_RATIO,
        "TEST_RATIO": TEST_RATIO,
        "USE_CONF_CHANNEL": USE_CONF_CHANNEL,
        "USE_BONE_FEATURE": USE_BONE_FEATURE,
        "USE_VELOCITY_FEATURE": USE_VELOCITY_FEATURE,
        "DROP_EMPTY_CLIPS": DROP_EMPTY_CLIPS,
        "USE_CLASS_WEIGHT": USE_CLASS_WEIGHT,
        "RESUME": RESUME,
        "DEVICE": DEVICE,
        "TSAM_NUM_HEADS": TSAM_NUM_HEADS,
        "TSAM_DROPOUT": TSAM_DROPOUT,
        "MODEL_DROPOUT": MODEL_DROPOUT,
        "HEAD_DROPOUT": HEAD_DROPOUT,
        "CLIP_AGG_MODE": CLIP_AGG_MODE,
        "CLIP_TOPK": CLIP_TOPK,
    }
    save_json(config, Path(SAVE_ROOT) / "config.json")

    print("[1] 샘플 스캔 중...")
    samples = scan_clip_samples(DATA_ROOT, label_key=USE_LABEL_KEY)
    print(f"전체 clip 수: {len(samples)}")

    samples = filter_samples(
        samples,
        min_valid_frames=MIN_VALID_FRAMES,
        drop_empty_clips=DROP_EMPTY_CLIPS
    )
    print(f"필터 후 clip 수: {len(samples)}")

    label2id, id2label = build_label_map(samples)
    save_json(label2id, Path(SAVE_ROOT) / "label2id.json")
    save_json(id2label, Path(SAVE_ROOT) / "id2label.json")

    print("\n[라벨 맵]")
    for k, v in label2id.items():
        print(f"{v}: {k}")

    split_info_path = Path(SAVE_ROOT) / "split_info.json"

    if split_info_path.exists():
        print("\n[기존 split_info.json 발견] 기존 split 재사용")
        split_info = load_json(split_info_path)

        train_set = set(split_info["train"])
        val_set = set(split_info["val"])
        test_set = set(split_info["test"])

        train_samples = [s for s in samples if s["clip_dir"] in train_set]
        val_samples = [s for s in samples if s["clip_dir"] in val_set]
        test_samples = [s for s in samples if s["clip_dir"] in test_set]
    else:
        print("\n[새 stratified split 생성]")
        train_samples, val_samples, test_samples = stratified_split_samples(
            samples,
            train_ratio=TRAIN_RATIO,
            val_ratio=VAL_RATIO,
            test_ratio=TEST_RATIO,
            seed=SEED
        )
        split_info = {
            "train": [s["clip_dir"] for s in train_samples],
            "val": [s["clip_dir"] for s in val_samples],
            "test": [s["clip_dir"] for s in test_samples],
        }
        save_json(split_info, split_info_path)

    print(f"Train clips: {len(train_samples)}")
    print(f"Val clips  : {len(val_samples)}")
    print(f"Test clips : {len(test_samples)}")

    print("\n[2] 메모리 preload 시작")
    train_mem = preload_samples(
        train_samples, label2id,
        window_size=WINDOW_SIZE,
        stride=WINDOW_STRIDE,
        use_conf_channel=USE_CONF_CHANNEL,
        use_bone_feature=USE_BONE_FEATURE,
        use_velocity_feature=USE_VELOCITY_FEATURE,
        desc="Preload Train",
        max_windows_per_clip=MAX_WINDOWS_PER_CLIP,
        min_window_valid_frames=MIN_WINDOW_VALID_FRAMES,
        min_window_valid_ratio=MIN_WINDOW_VALID_RATIO,
        min_window_valid_joint_ratio=MIN_WINDOW_VALID_JOINT_RATIO
    )
    val_mem = preload_samples(
        val_samples, label2id,
        window_size=WINDOW_SIZE,
        stride=WINDOW_STRIDE,
        use_conf_channel=USE_CONF_CHANNEL,
        use_bone_feature=USE_BONE_FEATURE,
        use_velocity_feature=USE_VELOCITY_FEATURE,
        desc="Preload Val",
        max_windows_per_clip=MAX_WINDOWS_PER_CLIP,
        min_window_valid_frames=MIN_WINDOW_VALID_FRAMES,
        min_window_valid_ratio=MIN_WINDOW_VALID_RATIO,
        min_window_valid_joint_ratio=MIN_WINDOW_VALID_JOINT_RATIO
    )
    test_mem = preload_samples(
        test_samples, label2id,
        window_size=WINDOW_SIZE,
        stride=WINDOW_STRIDE,
        use_conf_channel=USE_CONF_CHANNEL,
        use_bone_feature=USE_BONE_FEATURE,
        use_velocity_feature=USE_VELOCITY_FEATURE,
        desc="Preload Test",
        max_windows_per_clip=MAX_WINDOWS_PER_CLIP,
        min_window_valid_frames=MIN_WINDOW_VALID_FRAMES,
        min_window_valid_ratio=MIN_WINDOW_VALID_RATIO,
        min_window_valid_joint_ratio=MIN_WINDOW_VALID_JOINT_RATIO
    )

    print(f"\n메모리 적재 완료")
    print(f"Train windows: {len(train_mem)}")
    print(f"Val windows  : {len(val_mem)}")
    print(f"Test windows : {len(test_mem)}")

    train_ds = MemorySkeletonWindowDataset(train_mem)
    val_ds = MemorySkeletonWindowDataset(val_mem)
    test_ds = MemorySkeletonWindowDataset(test_mem)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    A = build_adjacency(NUM_JOINTS, COCO_EDGES, self_link=True)

    in_channels = 2
    if USE_BONE_FEATURE:
        in_channels += 2
    if USE_VELOCITY_FEATURE:
        in_channels += 2
    if USE_CONF_CHANNEL:
        in_channels += 1

    num_classes = len(label2id)

    model = TGCN_TSAM_Classifier(
        in_channels=in_channels,
        num_classes=num_classes,
        A=A,
        num_joints=NUM_JOINTS
    ).to(DEVICE)

    if USE_CLASS_WEIGHT:
        class_weights = make_class_weights_from_memory(train_mem, num_classes).to(DEVICE)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print(f"\nclass_weights(window 기준): {class_weights}")
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3
    )

    history = []
    best_val_f1 = -1.0
    best_epoch = -1
    no_improve = 0
    start_epoch = 1

    last_ckpt_path = Path(SAVE_ROOT) / "last.pt"
    resume_path = Path(RESUME_PATH) if RESUME_PATH is not None else last_ckpt_path

    if RESUME and resume_path.exists():
        print(f"\n[Resume] 체크포인트 로드: {resume_path}")
        ckpt = torch.load(resume_path, map_location=DEVICE)

        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        if ckpt.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])

        start_epoch = ckpt["epoch"] + 1
        best_val_f1 = ckpt.get("best_val_f1", -1.0)
        history = ckpt.get("history", [])

        if len(history) > 0:
            best_epoch = max(history, key=lambda x: x["val_clip_f1_macro"])["epoch"]

        print(f"[Resume 완료] start_epoch={start_epoch}, best_val_f1={best_val_f1:.4f}")
    else:
        print("\n[Resume 없음] 처음부터 학습 시작")

    print(f"\n사용 장치: {DEVICE}")
    print(f"in_channels: {in_channels}")
    print("[3] 학습 시작")

    for epoch in range(start_epoch, EPOCHS + 1):
        print(f"\n===== Epoch {epoch}/{EPOCHS} =====")

        train_out = run_one_epoch(
            model, train_loader, optimizer, criterion, DEVICE,
            train=True, epoch_idx=epoch, total_epochs=EPOCHS
        )
        val_out = run_one_epoch(
            model, val_loader, optimizer, criterion, DEVICE,
            train=False, epoch_idx=epoch, total_epochs=EPOCHS
        )

        val_clip = val_out["clip_level"]

        scheduler.step(val_clip["f1_macro"])
        current_lr = optimizer.param_groups[0]["lr"]

        row = {
            "epoch": epoch,
            "lr": current_lr,

            "train_window_loss": train_out["loss"],
            "train_window_acc": train_out["acc"],
            "train_window_f1_macro": train_out["f1_macro"],

            "val_window_loss": val_out["loss"],
            "val_window_acc": val_out["acc"],
            "val_window_f1_macro": val_out["f1_macro"],

            "val_clip_acc": val_clip["acc"],
            "val_clip_f1_macro": val_clip["f1_macro"],
        }
        history.append(row)

        print(
            f"Epoch {epoch}/{EPOCHS} | "
            f"LR: {current_lr:.8f} | "
            f"Train(window) loss: {train_out['loss']:.4f}, acc: {train_out['acc']:.4f}, f1: {train_out['f1_macro']:.4f} | "
            f"Val(window) loss: {val_out['loss']:.4f}, acc: {val_out['acc']:.4f}, f1: {val_out['f1_macro']:.4f} | "
            f"Val(clip) acc: {val_clip['acc']:.4f}, f1: {val_clip['f1_macro']:.4f}"
        )

        pd.DataFrame(history).to_csv(Path(SAVE_ROOT) / "metrics.csv", index=False, encoding="utf-8-sig")
        save_json(history, Path(SAVE_ROOT) / "history.json")

        if SAVE_EVERY_EPOCH:
            save_checkpoint(
                Path(SAVE_ROOT) / "last.pt",
                model, optimizer, scheduler, epoch, best_val_f1, history, config, label2id
            )

        val_window_cm = confusion_matrix(
            val_out["targets"],
            val_out["preds"],
            labels=list(range(num_classes))
        )
        save_confusion_matrix(
            val_window_cm,
            [id2label[i] for i in range(num_classes)],
            Path(SAVE_ROOT) / f"val_window_confusion_epoch_{epoch:03d}.csv"
        )

        val_clip_cm = confusion_matrix(
            val_clip["targets"],
            val_clip["preds"],
            labels=list(range(num_classes))
        )
        save_confusion_matrix(
            val_clip_cm,
            [id2label[i] for i in range(num_classes)],
            Path(SAVE_ROOT) / f"val_clip_confusion_epoch_{epoch:03d}.csv"
        )

        if val_clip["f1_macro"] > best_val_f1:
            best_val_f1 = val_clip["f1_macro"]
            best_epoch = epoch
            no_improve = 0

            save_checkpoint(
                Path(SAVE_ROOT) / "best.pt",
                model, optimizer, scheduler, epoch, best_val_f1, history, config, label2id
            )
            print(f"[Best 갱신] epoch={epoch}, val_clip_f1={best_val_f1:.4f}")
        else:
            no_improve += 1

        if no_improve >= PATIENCE:
            print(f"[Early Stopping] {PATIENCE} epoch 동안 개선 없음")
            break

    print(f"\n학습 종료 | best epoch={best_epoch}, best val_clip_f1={best_val_f1:.4f}")

    print("\n[4] Best 모델 Test 평가")
    best_ckpt = torch.load(Path(SAVE_ROOT) / "best.pt", map_location=DEVICE)
    model.load_state_dict(best_ckpt["model_state_dict"])

    test_out = run_one_epoch(
        model, test_loader, optimizer, criterion, DEVICE,
        train=False, epoch_idx=best_epoch, total_epochs=EPOCHS
    )
    test_clip = test_out["clip_level"]

    print(
        f"Test(window) loss: {test_out['loss']:.4f} | "
        f"Test(window) acc: {test_out['acc']:.4f} | "
        f"Test(window) f1_macro: {test_out['f1_macro']:.4f}"
    )
    print(
        f"Test(clip) acc: {test_clip['acc']:.4f} | "
        f"Test(clip) f1_macro: {test_clip['f1_macro']:.4f}"
    )

    test_window_cm = confusion_matrix(
        test_out["targets"],
        test_out["preds"],
        labels=list(range(num_classes))
    )
    save_confusion_matrix(
        test_window_cm,
        [id2label[i] for i in range(num_classes)],
        Path(SAVE_ROOT) / "test_window_confusion.csv"
    )

    test_clip_cm = confusion_matrix(
        test_clip["targets"],
        test_clip["preds"],
        labels=list(range(num_classes))
    )
    save_confusion_matrix(
        test_clip_cm,
        [id2label[i] for i in range(num_classes)],
        Path(SAVE_ROOT) / "test_clip_confusion.csv"
    )

    summary = {
        "best_epoch": best_epoch,
        "best_val_clip_f1": best_val_f1,

        "test_window_loss": test_out["loss"],
        "test_window_acc": test_out["acc"],
        "test_window_f1_macro": test_out["f1_macro"],

        "test_clip_acc": test_clip["acc"],
        "test_clip_f1_macro": test_clip["f1_macro"],

        "num_classes": num_classes,
        "labels": [id2label[i] for i in range(num_classes)],

        "n_train_clips": len(train_samples),
        "n_val_clips": len(val_samples),
        "n_test_clips": len(test_samples),

        "n_train_windows": len(train_mem),
        "n_val_windows": len(val_mem),
        "n_test_windows": len(test_mem),
    }
    save_json(summary, Path(SAVE_ROOT) / "summary.json")

    print("[저장 완료]")


if __name__ == "__main__":
    main()