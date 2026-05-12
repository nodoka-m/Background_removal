# Subset用に軽くしている
import torch.nn as nn
import open3d as o3d
import torch.nn.functional as F
import torch
from torch import optim
import numpy as np
import os
import data_to_pointcloud
import PointNet as pn
from collections import Counter

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

    print("[DBG] done subject_split", len(X_train), len(X_val), len(X_test), flush=True)
    print("[DBG] train label range:", int(min(y_train)), int(max(y_train)), flush=True)


    from collections import Counter
    print("train labels:", Counter(y_train).most_common(5))
    print("val labels:", Counter(y_val).most_common(5))
    print("test labels:", Counter(y_test).most_common(5))

    epochs = pn.EXPERIMENT["epochs"]
    steps_per_epoch = pn.EXPERIMENT["steps_per_epoch"]
    batch_size = pn.EXPERIMENT["batch_size"]
    eval_batches = pn.EXPERIMENT["eval_batches"]
    lr = pn.EXPERIMENT['lr']
    momentum = pn.EXPERIMENT['momentum']
    num_points = 2048

    print("学習フェーズ")
    PointNet = pn.PointNet(num_points, classes=num_classes).to(device)
    # PointNetモデルのインスタンスを1つ作成され、重みが初期化される
    criterion = nn.NLLLoss()
    # 損失関数（loss function）を定義
    optimizer = optim.SGD(PointNet.parameters(), lr = lr, momentum = momentum)
    # どうやって重みを更新するかを決める

    for i in range(epochs):
        PointNet.train()
        running_loss = 0.0
        for step in range(steps_per_epoch):
            xb, yb = pn.make_minibatch(X_train, y_train, batch_size, num_points, device)

            optimizer.zero_grad()
            out, _, _ = PointNet(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / steps_per_epoch
        
        print("評価フェーズ")
        PointNet.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for _ in range(eval_batches):
                xb, yb = pn.make_minibatch(X_val, y_val, batch_size, num_points, device)
                out, _, _ = PointNet(xb)
                pred = out.argmax(dim=1)
                correct += (pred == yb).sum().item()
                total += yb.numel()
            val_acc = correct / max(total, 1)

            print("検証フェーズ")
            correct = 0
            total = 0
            for _ in range(eval_batches):
                xb, yb = pn.make_minibatch(X_test, y_test, batch_size, num_points, device)
                out, _, _ = PointNet(xb)
                print("xb:", xb.shape, "yb:", yb.shape, "out:", out.shape)
                pred = out.argmax(dim=1)
                correct += (pred == yb).sum().item()
                total += yb.numel()
            test_acc = correct / max(total, 1)
        print(f"epoch {i+1} loss={avg_loss:.4f} val_acc={val_acc:.3f} test_acc={test_acc:.3f}")


if __name__ == "__main__":
    print("[DBG] __main__ start", flush=True)
    main()
    print("[DBG] __main__ end", flush=True)
