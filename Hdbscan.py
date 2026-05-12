# HDBSCANでクラスタリングし、最大クラスタ(点数最多)を「人物」と仮定して抽出する

import hdbscan
import numpy as np
import open3d as o3d


def extract_largest_cluster_pcd(pcd: o3d.geometry.PointCloud,
                               min_cluster_size: int = 50,
                               min_samples: int | None = None,
                               keep_colors: bool = True,
                               keep_normals: bool = True):
    points = np.asarray(pcd.points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"pcd.points must be shape (N,3). got {points.shape}")

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples
    )
    labels = clusterer.fit_predict(points)  # (N,)

    # 最大クラスタを人物とみなす
    valid = labels[labels >= 0]
    if valid.size == 0:


        # クラスタが1つもできなかった
        return o3d.geometry.PointCloud(), -1, labels

    # labels が person_label の点だけ NumPy で抽出
    person_label = int(np.bincount(valid).argmax())
    idx = np.where(labels == person_label)[0].astype(np.int32)

    # open3dで抽出（属性も一緒に切り出せる）
    person_pcd = pcd.select_by_index(idx.tolist())

    # 念のため、不要なら属性を落とす
    if not keep_colors and person_pcd.has_colors():
        person_pcd.colors = o3d.utility.Vector3dVector()
    if not keep_normals and person_pcd.has_normals():
        person_pcd.normals = o3d.utility.Vector3dVector()

    return person_pcd, person_label, labels
