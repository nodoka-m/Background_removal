# 抽出したベクトルをファイルに記述
# TCNで抽出した10個のベクトルをひたすらファイルに書き込んで、これを新しい学習データにする

import os
import open3d as o3d
import numpy as np
import torch
import torch.nn as nn
from PointNet import PointNet, load_pcd_as_tensor


# =========================
# 1. PCD → ベクトル変換（仮）
# =========================
def pcd_to_feature(pcd_path, model, device):
    model.eval()

    pcd = o3d.io.read_point_cloud(pcd_path)
    pts = load_pcd_as_tensor(pcd, num_points=2048)

    if pts is None:
        print(f"[SKIP] empty: {pcd_path}")
        return None

    
    pts = pts.unsqueeze(0).to(device)  # (1,N,3)

    with torch.no_grad():
        feat = model(pts, return_feature=True)  # (1,1024)

    return feat.squeeze(0).cpu().numpy()


# =========================
# 2. ウィンドウ化
# =========================
def create_windows(sequence, window_size=30, stride=15):
    windows = []
    for i in range(0, len(sequence) - window_size + 1, stride):
        windows.append(sequence[i:i+window_size])
    return windows


# =========================
# 3. TCNモデル
# =========================
class TCN(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=64, output_dim=33):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=2, dilation=2)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=4, dilation=4)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: (B, T, D) → (B, D, T)
        x = x.transpose(1, 2)

        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))

        x = self.pool(x).squeeze(-1)  # (B, hidden)
        x = self.fc(x)               # (B, 10)

        return x


# =========================
# 4. メイン処理
# =========================
def process_dataset(root_dir, save_path):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # PointNet
    model_pth = torch.load(r"C:\Users\bracy\Documents\Class\graduate shool\naist\U lab\study\Pointcloud_comparison\best_pointnet.pth")
    pointnet = PointNet()
    pointnet.load_state_dict(model_pth, strict = False)
    pointnet.to(device)
    pointnet.eval()

    # TCN
    tcn = TCN(input_dim=1024)
    tcn.to(device)
    tcn.eval()

    all_features = []

    for person in os.listdir(root_dir):
        person_path = os.path.join(root_dir, person)
        if not os.path.isdir(person_path):
            continue

        for action in os.listdir(person_path):
            action_path = os.path.join(person_path, action, "ex1_pcd")

            if not os.path.exists(action_path):
                continue

            print(f"Processing: {person}/{action}")

            pcd_files = sorted([
                os.path.join(action_path, f)
                for f in os.listdir(action_path)
                if f.endswith(".pcd")
            ])

            sequence = [pcd_to_feature(p, pointnet, device) for p in pcd_files]

            sequence = [s for s in sequence if s is not None]

            if len(sequence) < 30:
                continue

            windows = create_windows(sequence)

            for w in windows:
                w = torch.tensor(w, dtype=torch.float32).unsqueeze(0).to(device)

                with torch.no_grad():
                    vec = tcn(w).squeeze(0).cpu().numpy()

                all_features.append(vec)

    all_features = np.array(all_features)
    np.save(save_path, all_features)

    print(f"Saved: {save_path}")
    print(f"Shape: {all_features.shape}")

# =========================
# 実行
# =========================
if __name__ == "__main__":
    root_dir = r"C:\Users\bracy\Documents\Class\graduate shool\naist\U lab\study\Pointcloud_comparison\ResearchData\PointClouds"
    save_path = "tcn_features.npy"

    process_dataset(root_dir, save_path)