import torch.nn as nn
import open3d as o3d
import torch.nn.functional as F
import torch
from torch import optim
import numpy as np
import os
import glob
import re
import data_to_pointcloud_2
from torch import optim


EXPERIMENT = {
    "epochs": 50,
    "eval_batches": 20,
    "steps_per_epoch": 33, 
    "batch_size": 16,
    "lr": 0.0003,
    "momentum": 0.9,
    "optimizer": "adam"     # SGDより手軽に収束しやすい
}


class T_net(nn.Module):
    def __init__(self, pointNum=2048, mat_dim=3):
        super().__init__()
        self.pointNum = pointNum
        self.mat_dim = mat_dim
        self.register_buffer("iden", torch.eye(mat_dim).unsqueeze(0))  # (1,k,k)

        # 主要なレイヤー
        self.conv1_1 = nn.Conv1d(mat_dim, 64, 1)
        self.conv1_2 = nn.Conv1d(64, 128, 1)
        self.conv1_3 = nn.Conv1d(128, 1024, 1)
        self.MaxPool = nn.MaxPool1d(pointNum)
        self.fc1_1 = nn.Linear(1024, 512)
        self.fc1_2 = nn.Linear(512, 256)
        self.fc1_3 = nn.Linear(256, mat_dim * mat_dim)

        # すべてのレイヤーで共通で行うレイヤー
        self.bn_conv1_1 = nn.BatchNorm1d(64)
        self.bn_conv1_2 = nn.BatchNorm1d(128)
        self.bn_conv1_3 = nn.BatchNorm1d(1024)
        self.bn_fc1_1 = nn.BatchNorm1d(512)
        self.bn_fc1_2 = nn.BatchNorm1d(256)

    def forward(self, input):
        input_pcl = input

        # 畳み込み層
        input_pcl = self.conv1_1(input_pcl)
        input_pcl = self.filter_common(self.bn_conv1_1, input_pcl)
        input_pcl = self.conv1_2(input_pcl)
        input_pcl = self.filter_common(self.bn_conv1_2, input_pcl)
        input_pcl = self.conv1_3(input_pcl)
        input_pcl = self.filter_common(self.bn_conv1_3, input_pcl)

        # マックスプーリング
        input_pcl = self.MaxPool(input_pcl)
        input_pcl = nn.Flatten(1)(input_pcl)

        # 結合層
        input_pcl = self.fc1_1(input_pcl)
        input_pcl = self.filter_common(self.bn_fc1_1, input_pcl)
        input_pcl = self.fc1_2(input_pcl)
        input_pcl = self.filter_common(self.bn_fc1_2, input_pcl)

        # 変換行列
        # trans_mat = self.fc1_3(input_pcl)
        # trans_mat = trans_mat.view(-1, self.mat_dim, self.mat_dim)

        # weight = torch.eye(self.mat_dim).repeat(
        #     input_pcl.shape[0], 1, 1
        # )
        # trans_mat = trans_mat + weight

        # return trans_mat

        trans_mat = self.fc1_3(input_pcl).view(-1, self.mat_dim, self.mat_dim)
        trans_mat = trans_mat + self.iden.repeat(
            trans_mat.size(0), 1, 1
        )

        return trans_mat


    def filter_common(self, bn, inputdata):
        return F.relu(bn(inputdata))


class PointNet(nn.Module):
    def __init__(self, pointNum=2048, classes=10):
        super().__init__()
        self.pointNum = pointNum
        self.classes = classes
        # ここよくわからない
        self.tnet3  = T_net(pointNum=pointNum, mat_dim=3)
        self.tnet64 = T_net(pointNum=pointNum, mat_dim=64)


        # 主要なレイヤー
        self.input_transform = self.transform
        self.fc1 = nn.Linear(3, 64)
        self.feature_transfor = self.transform
        self.fc2_1 = nn.Linear(64, 64)
        self.fc2_2 = nn.Linear(64, 128)
        self.fc2_3 = nn.Linear(128, 1024)
        self.MaxPool = nn.MaxPool1d(pointNum)
        self.fc3_1 = nn.Linear(1024, 512)
        self.fc3_2 = nn.Linear(512, 256)
        self.fc3_3 = nn.Linear(256, classes)

        # 共通レイヤー
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2_1 = nn.BatchNorm1d(64)
        self.bn2_2 = nn.BatchNorm1d(128)
        self.bn2_3 = nn.BatchNorm1d(1024)
        self.bn3_1 = nn.BatchNorm1d(512)
        self.bn3_2 = nn.BatchNorm1d(256)
        self.DropOut = nn.Dropout(p=0.3)
        self.logsoftmax = nn.LogSoftmax(dim=1)

        self.SegmentationNW = nn.Sequential(
            nn.Linear(1088, 512),
            nn.Linear(512, 256),
            nn.Linear(256, 128),
            nn.Linear(128, classes),
        )

    def transform(self, input_pcl, k):
        if k == 3:
            trans_kxK = self.tnet3(input_pcl)
        elif k == 64:
            trans_kxK = self.tnet64(input_pcl)

        input_pcl = torch.bmm(
            torch.transpose(input_pcl, 1, 2), trans_kxK
        )
        return input_pcl, trans_kxK

    def forward(self, input, return_feature=False):
        input_pcl = input

        # 1. 入力変換
        input_pcl, trans3x3 = self.input_transform(
            input_pcl.transpose(1, 2), k=3
        )

        # 2. 一回目の MLP
        input_pcl = self.fc1(input_pcl)
        input_pcl = self.filter_common(self.bn1, input_pcl)

        # 3. 特徴変換
        input_pcl, trans64x64 = self.feature_transfor(
            input_pcl.transpose(1, 2), k=64
        )

        # 4. 二回目の MLP
        input_pcl = self.fc2_1(input_pcl)
        input_pcl = self.filter_common(self.bn2_1, input_pcl)

        input_pcl = self.fc2_2(input_pcl)
        input_pcl = self.filter_common(self.bn2_2, input_pcl)

        input_pcl = self.fc2_3(input_pcl)
        input_pcl = self.filter_common(self.bn2_3, input_pcl)

        # 5. マックスプーリング
        input_pcl = input_pcl.transpose(1, 2)   # (B, 1024, N)
        input_pcl = self.MaxPool(input_pcl)     # (B, 1024, 1)
        input_pcl = input_pcl.squeeze(-1)       # (B, 1024)

        # ここで保存
        global_feat = input_pcl
        # 追加
        if return_feature:
            return global_feat

        # 6. 三回目の MLP
        input_pcl = self.fc3_1(input_pcl)
        input_pcl = self.filter_common(self.bn3_1, input_pcl)

        # 7. 最終 MLP
        input_pcl = self.fc3_2(input_pcl)
        input_pcl = self.DropOut(input_pcl)
        input_pcl = self.filter_common(self.bn3_2, input_pcl)

        input_pcl = self.fc3_3(input_pcl)
        output_pcl = self.logsoftmax(input_pcl)

        return output_pcl, trans3x3, trans64x64

    def filter_common(self, bn, inputdata):
        if inputdata.ndim == 3:
            inputdata = inputdata.transpose(1, 2)
            inputdata = F.relu(bn(inputdata))
            inputdata = inputdata.transpose(1, 2)
        elif inputdata.ndim == 2:
            inputdata = F.relu(bn(inputdata))
        return inputdata


def subject_split(base_dir, pcds, labels, person_ids):
    persons = data_to_pointcloud_2.list_persons(base_dir)

    train_persons = set(persons[0:11])   # setにするとin判定が速い
    val_persons   = set(persons[11:13])
    test_persons  = set(persons[13:16])

    X_train, y_train = [], []
    X_val,   y_val   = [], []
    X_test,  y_test  = [], []

    for f, label, pid in zip(pcds, labels, person_ids):
        if pid in train_persons:
            X_train.append(f)
            y_train.append(label)
        elif pid in val_persons:
            X_val.append(f)
            y_val.append(label)
        elif pid in test_persons:
            X_test.append(f)
            y_test.append(label)
        else:
            print("Warning: Unknown person ID =", pid)

    return X_train, y_train, X_val, y_val, X_test, y_test
    # X_train = ["A.pcd", "D.pcd"], y_train = [0, 3], ...

def load_pcd_as_tensor(pcd, num_points=2048):
    pts = np.asarray(pcd.points)
    # pcdの中に入っている座標だけ取り出してnumpy配列に変換

    # 壊れてる点群を使わない
    if pts.shape[0] == 0:
        return None

    # 点数を揃える（ランダムサンプリング）
    if pts.shape[0] >= num_points:
        idx = np.random.choice(pts.shape[0], num_points, replace=False)

    # 点数少ない時には2048点になるよう重複で補うようにしているが、これでは正しくない精度が出る可能性高い（まあ実際2048点を下回る可能性低いからいいか）
    else:
        print(f"[WARN] points < 2048: {pts.shape[0]} points. Upsampling with replacement.")
        idx = np.random.choice(pts.shape[0], num_points, replace=True)
        
    pts = pts[idx]

    return torch.from_numpy(pts.astype(np.float32))
    # tensor([[x1., y1., z1.],
    #         [x2., y2., z3.]])

_PCD_CACHE = {}
def make_minibatch(pcd_paths, labels, batch_size, num_points, device):
    """
    pcd_paths: list[str]  .pcdのパスのリスト（train用など）
    labels:   list[int]  そのパスに対応するラベル
    """
    # バッチに使うインデックスをランダム抽出（重複ありでOK）
    idx = np.random.randint(0, len(pcd_paths), size=batch_size)
    batch_points = []
    batch_labels = []

    for j in idx:
        path = pcd_paths[j]

        if path in _PCD_CACHE:
            pts = _PCD_CACHE[path]  # (num_points, 3) CPU tensor
        else:
            pcd = o3d.io.read_point_cloud(path)
            pts = load_pcd_as_tensor(pcd, num_points=num_points)  # CPU tensor
            _PCD_CACHE[path] = pts

        batch_points.append(pts)
        batch_labels.append(labels[j])

    x = torch.stack(batch_points, dim=0).to(device, non_blocking=True)
    y = torch.tensor(batch_labels, dtype=torch.long).to(device, non_blocking=True)
    return x, y