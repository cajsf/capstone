import os
import json
import random
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


# =========================================================
# 설정
# =========================================================
SEED = 42

DATA_ROOT = r"C:\Users\hyi8402\Downloads\dmd_2"
SAVE_ROOT = r"C:\Users\hyi8402\Desktop\Capstone\Code\3.result\stgcn_runs\run_mem_preload"

USE_LABEL_KEY = "primary_label"
MIN_VALID_FRAMES = 4
TARGET_T = 48

BATCH_SIZE = 32
NUM_WORKERS = 0          # preload 쓸 때는 0~2 정도면 충분
EPOCHS = 100
LR = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 10

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

USE_CONF_CHANNEL = True
DROP_EMPTY_CLIPS = True
USE_CLASS_WEIGHT = True

RESUME = True
RESUME_PATH = None       # None이면 SAVE_ROOT/last.pt 자동 사용
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

    # 길이가 충분하면 중앙 연속 구간 48프레임
    if length >= target_len:
        start = (length - target_len) // 2
        return np.arange(start, start + target_len, dtype=np.int64)

    # 길이가 부족하면 양쪽 패딩
    # 앞은 첫 프레임 반복, 뒤는 마지막 프레임 반복
    pad_total = target_len - length
    left_pad = pad_total // 2
    right_pad = pad_total - left_pad

    center = np.arange(length, dtype=np.int64)
    left = np.full((left_pad,), 0, dtype=np.int64)
    right = np.full((right_pad,), length - 1, dtype=np.int64)

    return np.concatenate([left, center, right], axis=0)


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


def valid_frame_count(conf):
    if conf is None:
        return 0
    return int(np.sum(np.any(conf > 0, axis=1)))


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

            n_valid = valid_frame_count(conf) if conf is not None else 0

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


def filter_samples(samples, min_valid_frames=4, drop_empty_clips=True):
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


def split_samples(samples, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    samples = samples.copy()
    random.Random(seed).shuffle(samples)

    n = len(samples)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_samples = samples[:n_train]
    val_samples = samples[n_train:n_train + n_val]
    test_samples = samples[n_train + n_val:]
    return train_samples, val_samples, test_samples



# =========================================================
# 전처리 + preload
# =========================================================
def preprocess_one_sample(sample, label2id, target_t=48, use_conf_channel=True):
    npz = np.load(sample["skeleton_path"], allow_pickle=True)

    keypoints = npz["keypoints"].astype(np.float32)  # (T, V, 2)
    conf = npz["conf"].astype(np.float32) if "conf" in npz.files else None

    if conf is not None:
        conf = np.nan_to_num(conf, nan=0.0, posinf=0.0, neginf=0.0)

    keypoints = center_scale_normalize_xy(keypoints, conf=conf)
    keypoints = np.nan_to_num(keypoints, nan=0.0, posinf=0.0, neginf=0.0)

    T, V, _ = keypoints.shape
    idxs = center_crop_indices(T, target_t)

    keypoints = keypoints[idxs]
    if conf is not None:
        conf = conf[idxs]
    else:
        conf = np.ones((target_t, V), dtype=np.float32)

    if use_conf_channel:
        x = np.concatenate([keypoints, conf[..., None]], axis=-1)   # (T,V,3)
    else:
        x = keypoints                                                # (T,V,2)

    x = np.transpose(x, (2, 0, 1)).astype(np.float32)               # (C,T,V)
    y = label2id[sample["label_text"]]

    return {
        "x": x,
        "y": y,
        "clip_dir": sample["clip_dir"],
        "label_text": sample["label_text"]
    }


def preload_samples(samples, label2id, target_t=48, use_conf_channel=True, desc="Preloading"):
    memory_data = []
    pbar = tqdm(
        samples,
        desc=desc,
        ncols=120,
        leave=True,
        ascii=True,
        mininterval=0.3
    )

    for s in pbar:
        try:
            item = preprocess_one_sample(
                s,
                label2id=label2id,
                target_t=target_t,
                use_conf_channel=use_conf_channel
            )
            memory_data.append(item)
        except Exception as e:
            print(f"\n[preload 스킵] {s['clip_dir']} | 오류: {e}")

    return memory_data


# =========================================================
# Dataset
# =========================================================
class MemorySkeletonDataset(Dataset):
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
            "clip_dir": item["clip_dir"]
        }


# =========================================================
# ST-GCN 블록
# =========================================================
class GraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, A):
        super().__init__()
        self.A = nn.Parameter(torch.tensor(A), requires_grad=False)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.conv(x)  # (N,C,T,V)
        x = torch.einsum("nctv,vw->nctw", x, self.A)
        return x


class STGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, dropout=0.1):
        super().__init__()
        self.gcn = GraphConv(in_channels, out_channels, A)

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=(9, 1),
                stride=(stride, 1),
                padding=(4, 0)
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout)
        )

        if in_channels == out_channels and stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels)
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        res = self.residual(x)
        x = self.gcn(x)
        x = self.tcn(x)
        x = x + res
        x = self.relu(x)
        return x


class STGCNClassifier(nn.Module):
    def __init__(self, in_channels, num_classes, A):
        super().__init__()
        self.data_bn = nn.BatchNorm1d(in_channels * NUM_JOINTS)

        self.layer1 = STGCNBlock(in_channels, 64, A, stride=1, dropout=0.1)
        self.layer2 = STGCNBlock(64, 64, A, stride=1, dropout=0.1)
        self.layer3 = STGCNBlock(64, 128, A, stride=2, dropout=0.1)
        self.layer4 = STGCNBlock(128, 128, A, stride=1, dropout=0.1)
        self.layer5 = STGCNBlock(128, 256, A, stride=2, dropout=0.1)
        self.layer6 = STGCNBlock(256, 256, A, stride=1, dropout=0.1)

        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # x: (N,C,T,V)
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

        x = x.mean(dim=-1).mean(dim=-1)  # global avg pool
        x = self.head(x)
        return x


# =========================================================
# 학습 유틸
# =========================================================
def make_class_weights(samples, label2id):
    counts = Counter([s["label_text"] for s in samples])
    weights = np.zeros((len(label2id),), dtype=np.float32)

    for label_text, idx in label2id.items():
        weights[idx] = 1.0 / max(counts[label_text], 1)

    weights = weights / weights.sum() * len(weights)
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

        if train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            logits = model(x)
            loss = criterion(logits, y)

            if train:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * x.size(0)

        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.detach().cpu().numpy().tolist())
        all_targets.extend(y.detach().cpu().numpy().tolist())

        running_loss = total_loss / max(len(all_targets), 1)
        pbar.set_postfix(loss=f"{running_loss:.4f}")

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_targets, all_preds)
    f1_macro = f1_score(all_targets, all_preds, average="macro")

    return {
        "loss": avg_loss,
        "acc": acc,
        "f1_macro": f1_macro,
        "targets": all_targets,
        "preds": all_preds
    }


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
        "TARGET_T": TARGET_T,
        "BATCH_SIZE": BATCH_SIZE,
        "NUM_WORKERS": NUM_WORKERS,
        "EPOCHS": EPOCHS,
        "LR": LR,
        "WEIGHT_DECAY": WEIGHT_DECAY,
        "PATIENCE": PATIENCE,
        "TRAIN_RATIO": TRAIN_RATIO,
        "VAL_RATIO": VAL_RATIO,
        "TEST_RATIO": TEST_RATIO,
        "USE_CONF_CHANNEL": USE_CONF_CHANNEL,
        "DROP_EMPTY_CLIPS": DROP_EMPTY_CLIPS,
        "USE_CLASS_WEIGHT": USE_CLASS_WEIGHT,
        "RESUME": RESUME,
        "DEVICE": DEVICE,
    }
    save_json(config, Path(SAVE_ROOT) / "config.json")

    print("[1] 샘플 스캔 중...")
    samples = scan_clip_samples(DATA_ROOT, label_key=USE_LABEL_KEY)
    print(f"전체 샘플 수: {len(samples)}")

    samples = filter_samples(
        samples,
        min_valid_frames=MIN_VALID_FRAMES,
        drop_empty_clips=DROP_EMPTY_CLIPS
    )
    print(f"필터 후 샘플 수: {len(samples)}")

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

        train_samples = [s for s in samples if s["clip_dir"] in set(split_info["train"])]
        val_samples = [s for s in samples if s["clip_dir"] in set(split_info["val"])]
        test_samples = [s for s in samples if s["clip_dir"] in set(split_info["test"])]
    else:
        print("\n[새 split 생성]")
        train_samples, val_samples, test_samples = split_samples(
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

    print(f"Train: {len(train_samples)}")
    print(f"Val  : {len(val_samples)}")
    print(f"Test : {len(test_samples)}")

    print("\n[2] 메모리 preload 시작")
    train_mem = preload_samples(train_samples, label2id, TARGET_T, USE_CONF_CHANNEL, desc="Preload Train")
    val_mem = preload_samples(val_samples, label2id, TARGET_T, USE_CONF_CHANNEL, desc="Preload Val")
    test_mem = preload_samples(test_samples, label2id, TARGET_T, USE_CONF_CHANNEL, desc="Preload Test")

    print(f"\n메모리 적재 완료")
    print(f"Train preload: {len(train_mem)}")
    print(f"Val preload  : {len(val_mem)}")
    print(f"Test preload : {len(test_mem)}")

    train_ds = MemorySkeletonDataset(train_mem)
    val_ds = MemorySkeletonDataset(val_mem)
    test_ds = MemorySkeletonDataset(test_mem)

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
    in_channels = 3 if USE_CONF_CHANNEL else 2
    num_classes = len(label2id)

    model = STGCNClassifier(in_channels=in_channels, num_classes=num_classes, A=A).to(DEVICE)

    if USE_CLASS_WEIGHT:
        class_weights = make_class_weights(train_samples, label2id).to(DEVICE)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print(f"\nclass_weights: {class_weights}")
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
            best_epoch = max(history, key=lambda x: x["val_f1_macro"])["epoch"]

        print(f"[Resume 완료] start_epoch={start_epoch}, best_val_f1={best_val_f1:.4f}")
    else:
        print("\n[Resume 없음] 처음부터 학습 시작")

    print(f"\n사용 장치: {DEVICE}")
    print("[3] 학습 시작")

    for epoch in range(start_epoch, EPOCHS + 1):
        
        #tqdm.write(f"===== Epoch {epoch}/{EPOCHS} =====")

        train_out = run_one_epoch(
            model, train_loader, optimizer, criterion, DEVICE,
            train=True, epoch_idx=epoch, total_epochs=EPOCHS
        )
        val_out = run_one_epoch(
            model, val_loader, optimizer, criterion, DEVICE,
            train=False, epoch_idx=epoch, total_epochs=EPOCHS
        )

        scheduler.step(val_out["f1_macro"])
        current_lr = optimizer.param_groups[0]["lr"]

        row = {
            "epoch": epoch,
            "lr": current_lr,
            "train_loss": train_out["loss"],
            "train_acc": train_out["acc"],
            "train_f1_macro": train_out["f1_macro"],
            "val_loss": val_out["loss"],
            "val_acc": val_out["acc"],
            "val_f1_macro": val_out["f1_macro"],
        }
        history.append(row)

        print(
            f"Epoch {epoch}/{EPOCHS} | "
            f"LR: {current_lr:.8f} | "
            f"Train loss: {train_out['loss']:.4f}, acc: {train_out['acc']:.4f}, f1: {train_out['f1_macro']:.4f} | "
            f"Val loss: {val_out['loss']:.4f}, acc: {val_out['acc']:.4f}, f1: {val_out['f1_macro']:.4f}"
        )

        pd.DataFrame(history).to_csv(Path(SAVE_ROOT) / "metrics.csv", index=False, encoding="utf-8-sig")
        save_json(history, Path(SAVE_ROOT) / "history.json")

        if SAVE_EVERY_EPOCH:
            save_checkpoint(
                Path(SAVE_ROOT) / "last.pt",
                model, optimizer, scheduler, epoch, best_val_f1, history, config, label2id
            )

        val_cm = confusion_matrix(
            val_out["targets"],
            val_out["preds"],
            labels=list(range(num_classes))
        )
        save_confusion_matrix(
            val_cm,
            [id2label[i] for i in range(num_classes)],
            Path(SAVE_ROOT) / f"val_confusion_epoch_{epoch:03d}.csv"
        )

        if val_out["f1_macro"] > best_val_f1:
            best_val_f1 = val_out["f1_macro"]
            best_epoch = epoch
            no_improve = 0

            save_checkpoint(
                Path(SAVE_ROOT) / "best.pt",
                model, optimizer, scheduler, epoch, best_val_f1, history, config, label2id
            )
            print(f"[Best 갱신] epoch={epoch}, val_f1={best_val_f1:.4f}")
        else:
            no_improve += 1

        if no_improve >= PATIENCE:
            print(f"[Early Stopping] {PATIENCE} epoch 동안 개선 없음")
            break

    print(f"\n학습 종료 | best epoch={best_epoch}, best val_f1={best_val_f1:.4f}")

    print("\n[4] Best 모델 Test 평가")
    best_ckpt = torch.load(Path(SAVE_ROOT) / "best.pt", map_location=DEVICE)
    model.load_state_dict(best_ckpt["model_state_dict"])

    test_out = run_one_epoch(
        model, test_loader, optimizer, criterion, DEVICE,
        train=False, epoch_idx=best_epoch, total_epochs=EPOCHS
    )

    print(
        f"Test loss: {test_out['loss']:.4f} | "
        f"Test acc: {test_out['acc']:.4f} | "
        f"Test f1_macro: {test_out['f1_macro']:.4f}"
    )

    test_cm = confusion_matrix(
        test_out["targets"],
        test_out["preds"],
        labels=list(range(num_classes))
    )
    save_confusion_matrix(
        test_cm,
        [id2label[i] for i in range(num_classes)],
        Path(SAVE_ROOT) / "test_confusion.csv"
    )

    summary = {
        "best_epoch": best_epoch,
        "best_val_f1": best_val_f1,
        "test_loss": test_out["loss"],
        "test_acc": test_out["acc"],
        "test_f1_macro": test_out["f1_macro"],
        "num_classes": num_classes,
        "labels": [id2label[i] for i in range(num_classes)],
        "n_train": len(train_mem),
        "n_val": len(val_mem),
        "n_test": len(test_mem),
    }
    save_json(summary, Path(SAVE_ROOT) / "summary.json")

    print("[저장 완료]")


if __name__ == "__main__":
    main()