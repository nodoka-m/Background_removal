import os
import numpy as np
import torch
from PointNet import PointNet, load_pcd_as_tensor
from TCN import pcd_to_feature

# ===== 設定 =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1つのアクションフォルダを指定（適当に1つ）
action_path = r"/Users/nodoka-m/Desktop/research/ResearchData\PointClouds\person01\B1_BED_IN\ex1_pcd"

# ===== pcd一覧取得 =====
pcd_files = sorted([
    os.path.join(action_path, f)
    for f in os.listdir(action_path)
    if f.endswith(".pcd")
])

# ===== モデル =====
pointnet = PointNet().to(device)
# pointnet.load_state_dict(torch.load("..."), strict=False)  # 必要なら
pointnet.eval()

# ===== 1フレーム確認 =====
feat = pcd_to_feature(pcd_files[0], pointnet, device)

print("=== 1フレーム ===")
print("shape:", feat.shape)
print("mean/std:", feat.mean(), feat.std())
print("first10:", feat[:10])

# ===== 2フレーム差分 =====
feat2 = pcd_to_feature(pcd_files[1], pointnet, device)

print("\n=== フレーム差分 ===")
print("diff:", np.linalg.norm(feat - feat2))

# ===== シーケンス確認 =====
sequence = np.array([
    pcd_to_feature(p, pointnet, device)
    for p in pcd_files[:10]
])

print("\n=== シーケンス ===")
print("shape:", sequence.shape)

# ===== ウィンドウ確認 =====
def create_windows(sequence, window_size=30, stride=15):
    windows = []
    for i in range(0, len(sequence) - window_size + 1, stride):
        windows.append(sequence[i:i+window_size])
    return windows

windows = create_windows(sequence)

print("\n=== ウィンドウ ===")
print("num:", len(windows))
if len(windows) > 0:
    print("shape:", np.array(windows).shape)