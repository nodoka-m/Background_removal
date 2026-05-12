import open3d as o3d
import numpy as np
import math

def pts2plane(points):
    """
    3点から平面パラメータ (a, b, c, d) を求める
    平面方程式: a x + b y + c z + d = 0を完成させることが目的
    ※法線は正規化してあるので、距離 = a x + b y + c z + d でOK
    """
    p1, p2, p3 = points  # points=[[x1, y1, z1],[x2, y2, z2],[x3, y3, z3]]
    v1 = p2 - p1
    v2 = p3 - p1

    # 法線ベクトル
    n = np.cross(v1, v2)
    norm = np.linalg.norm(n)
    if norm == 0:
        # 3点が一直線上などで平面を定義できない場合
        return None

    n = n / norm #正規化
    a, b, c = n
    d = -np.dot(n, p1)  # 平面上の1点を使って d を計算

    return np.array([a, b, c, d])

def dist2plane(plane, points):
    """
    点群と平面の符号付き距離を計算
    plane: (a, b, c, d)
    points: shape (N, 3)
    戻り値: shape (N,) の距離ベクトル：
    dist = [ 0.02, -0.15, 0.001, 1.23, ... ]

    """
    a, b, c, d = plane
    # 法線は正規化済と仮定しているので分母は不要
    return points @ np.array([a, b, c]) + d
    # points @ np.array([a, b, c]) ：N個の ax + by + cz + d を一気に計算する

def ransac(pcd):
    # 初期化
    bestSupport = 0
    bestPlane = None
    bestStd = float("inf")
    # bestStd =　∞
    bestInliers = None
    i = 0
    threshold = 0.01

    epsilon = 0.7 # 仮定
    alpha = 0.99 # ransacが99%で正しい仮説を引けるとする
    N = round(
        math.log(1 - alpha) /
        math.log(1 - (1 - epsilon)**3)
    )

    points_list = np.asarray(pcd.points)
    # pcd.points：各点の座標を取り出したもの（Open3D専用の型）
    while i < N:
        # N個の点群からランダムかつ重複なしで3つ取り出す
        idx = np.random.choice(len(points_list), 3, replace=False)
        sample_points = points_list[idx]
        plane = pts2plane(sample_points)
        # plane = (a, b, c, d)
        if plane is None:
            # 3点が一直線などで平面が定義できなかった場合はスキップ
            i += 1
            continue
        # dist = [ 0.02, -0.15, 0.001, 1.23, ... ]
        dis = dist2plane(plane, points_list)

        # dis 配列の中から、距離が t 以下の点だけを取り出し、その点たちのインデックス（番号）を返す
        inliers = np.where(np.abs(dis) <= threshold)[0]
        # 例：inliers = [3, 6, 45, 12]
        # インライアがゼロなら評価する意味がない
        if inliers.size == 0:
            i += 1
            continue
        # インライア点たちの距離の標準偏差
        std = np.std(np.abs(dis[inliers]))
        # インライア数が過去最高なら更新
        if len(inliers) > bestSupport:
            bestPlane = plane
            bestStd = std
            bestSupport = len(inliers)
            bestInliers = inliers
        elif len(inliers) == bestSupport:
            if std < bestStd:
                bestPlane = plane
                bestStd = std
                bestInliers = inliers
        i += 1
    return bestSupport, bestPlane, bestStd, bestInliers

def check_normal_direction(x, y, theta):
    nx = np.linalg.norm(x)
    ny = np.linalg.norm(y)
    if nx == 0 or ny == 0:
        raise ValueError("x, y に0ベクトルが含まれている")

    d = float(np.dot(x, y))
    # float でNumPy型 → Python型
    return (np.isclose(d, 0.0, atol=theta) or
            np.isclose(d, 1.0, atol=theta) or
            np.isclose(d, -1.0, atol=theta))

def check_inlier_ratio(pcd, inliers, min_inliers = 1000):

    n_pcd = len(pcd.points)
    n_inliers = len(inliers)

    return (n_inliers >= min_inliers)


def remove_plane(pcd: o3d.geometry.PointCloud, inliers):

    inliers = np.asarray(inliers, dtype=int)

    return pcd.select_by_index(inliers.tolist(), invert=True)
    # inliers.tolist()：NumPy配列の inliers を Pythonのlist に変換（Open3Dが list を期待するため）
"""
法線方向、位置制約、インライア比率、領域の形、などの複合制約で、
布団を省きながら背景を表す平面（基本4つ）を確実に除去することを目的としている
"""
def background_removal(pcd):
        
    pcd = o3d.io.read_point_cloud(pcd)
    while True:

        # RANSAC実行
        bestSupport, bestPlane, bestStd, bestInliers = ransac(pcd)

        # 平面が見つからなかった場合
        if bestInliers is None:
            print("No plane detected.")
            break

        print(f"Detected inliers: {len(bestInliers)}")

        # インライア数チェック
        if not check_inlier_ratio(pcd, bestInliers, min_inliers=1000):

            print("Plane too small. Stop removal.")
            break
        pcd = remove_plane(pcd, bestInliers)

    return pcd