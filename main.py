import torch.nn as nn
import open3d as o3d
import torch.nn.functional as F
import torch
from torch import optim
import numpy as np
import os
import data_to_pointcloud
import PointNet as pn

# 研究データのルート
base_dir = r"C:\Users\bracy\Documents\Class\graduate shool\naist\U lab\study\Pointcloud_comparison\ResearchData"
camera_root = os.path.join(base_dir, "PointClouds")# アウトプット用

def main():
    # デバッグ用
    print("[DBG] start data_to_Pointcloud", flush=True)

    # device(GPUあれば使う、なければCPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    files, labels, person_ids, label_map = data_to_pointcloud.data_to_Pointcloud(base_dir, camera_root)
    num_classes = len(label_map)
    print("[DBG] done data_to_Pointcloud", len(files), len(labels), len(person_ids), "classes", len(label_map), flush=True)

    print("[DBG] sample label_map keys:", list(label_map.keys())[:5], flush=True)
    print("[DBG] person_ids sample:", person_ids[:5], flush=True)

    print("[DBG] start subject_split", flush=True)


    X_train, y_train, X_val, y_val, X_test, y_test = pn.subject_split(base_dir, files, labels, person_ids)
    print("train/val/test:", len(X_train), len(X_val), len(X_test))

    train_points = []
    train_labels = []
    val_points = []
    val_labels = []
    test_points = []
    test_labels = []

    epochs = pn.EXPERIMENT["epochs"]
    steps_per_epoch = pn.EXPERIMENT["steps_per_epoch"]
    batch_size = pn.EXPERIMENT["batch_size"]
    eval_batches = pn.EXPERIMENT["eval_batches"]
    lr = pn.EXPERIMENT['lr']
    momentum = pn.EXPERIMENT['momentum']
    num_points = 10000


    for train_pcd_path, train_label, in zip(X_train, y_train):
    # zipは同じインデックスの要素をペアにして同時に取り出すために使う

        # 点群読み込み
        train_pcd = o3d.io.read_point_cloud(train_pcd_path)
        train_pts = pn.load_pcd_as_tensor(train_pcd)   # (10000, 3)

        train_points.append(train_pts)
        train_labels.append(train_label)

    for val_pcd_path, val_label, in zip(X_val, y_val):

        val_pcd = o3d.io.read_point_cloud(val_pcd_path)
        val_pts = pn.load_pcd_as_tensor(val_pcd)   # (10000, 3)

        val_points.append(val_pts)
        val_labels.append(val_label)

    for test_pcd_path, test_label, in zip(X_test, y_test):

        test_pcd = o3d.io.read_point_cloud(test_pcd_path)
        test_pts = pn.load_pcd_as_tensor(test_pcd)   # (10000, 3)

        test_points.append(test_pts)
        test_labels.append(test_label)

    train_data = torch.stack(train_points).to(device)
    # train_data =
                # [
                #   点群1(10000, 3),
                #   点群2(10000, 3),
                #   点群3(10000, 3)
                # ]
                # = (B, 10000, 3)
    val_data = torch.stack(val_points).to(device)
    test_data = torch.stack(test_points).to(device)

    train_answer = torch.tensor(train_labels, dtype=torch.long).to(device)
    val_answer = torch.tensor(val_labels, dtype=torch.long).to(device)
    test_answer = torch.tensor(test_labels, dtype=torch.long).to(device)
    # answer_data = (B,)

    print("学習フェーズ")
    PointNet = pn.PointNet(num_points, classes=num_classes).to(device)
    # PointNetモデルのインスタンスを1つ作成され、重みが初期化される
    criterion = nn.NLLLoss()
    # 損失関数（loss function）を定義
    optimizer = optim.SGD(PointNet.parameters(), lr, momentum)
    # どうやって重みを更新するかを決める

    for i in range(epochs):

        PointNet.train()
        optimizer.zero_grad()
        outputs, trans3x3, trans64x64 = PointNet(train_data)
        loss = criterion(outputs, train_answer)
        loss.backward()
        # 「この loss を小さくするには、各重みをどっち向きにどれくらい動かせばいいか？」を計算する
        optimizer.step()
        # 逆伝搬

        print("評価フェーズ")
        PointNet.eval()
        with torch.no_grad():
            outputs, _, _ = PointNet(val_data)
        pred = outputs.argmax(dim=1)
        # outputs = (B, num_classes)
        correct = (pred == val_answer).sum().item()
        total = val_answer.size(0)

        print("検証フェーズ")
        with torch.no_grad():
            outputs_test, _, _ = PointNet(test_data)

        pred_test = outputs_test.argmax(dim=1)
        correct_test = (pred_test == test_answer).sum().item()
        total_test = test_answer.size(0)
        test_acc = correct_test / total_test

        val_acc = correct / total
        print(f"epoch {i+1} loss={loss.item():.4f} val_acc={val_acc:.3f} test_acc={test_acc:.3f}")


if __name__ == "__main__":
    print("[DBG] __main__ start", flush=True)
    main()
    print("[DBG] __main__ end", flush=True)
