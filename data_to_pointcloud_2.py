# RANSAC 点群の行動認識

import open3d as o3d
import pandas as pd
import os, zipfile
import cv2
import numpy as np
from io import BytesIO
import background_removal

# 研究データのルート
base_dir = r"/Users/nodoka-m/Desktop/research/ResearchData"

# カメラ内部パラメータ取得
def list_persons(base_dir: str):
    persons = []
    for d in os.listdir(base_dir):
        # d = Person001.zip
        p = os.path.join(base_dir, d)
        # pを定義してdが本当にフォルダかどうかチェック
        # p = C:\...\ResearchData\Person001.zip
        if d.lower().startswith("person") and (os.path.isdir(p) or d.lower().endswith(".zip")):

            persons.append(d)

    persons = sorted(persons)
    print("list_person:", persons)
    return persons
    # persons = ["Person001.zip", "Person002.zip", "Person003.zip", ...]

def list_actions(zip_path: str, person: str):
    pid = os.path.splitext(person)[0]
    actions = set()
    with zipfile.ZipFile(zip_path, "r") as zf:
        for f in zf.namelist():
#         zf.namelist() = [
                            # "Person001/",
                            # "Person001/B1_BED_IN/",
                            # "Person001/B1_BED_IN/depthImages/0.png",
                            # "Person001/E1_ENTER_HOUSE/depthImages/0.png",
                            # ...
                            # ]
            if not f.startswith(f"{pid}/"):
                continue
            rest = f[len(f"{pid}/"):]
            # rest = B1_BED_IN/depthImages/0.png
            parts = rest.split("/")
            # parts = ["B1_BED_IN", "depthImages", "0.png"]
            if len(parts) >= 2 and parts[0]:
                actions.add(parts[0])
        return sorted(actions)

    # 出力：["B1_BED_IN", "E1_ENTER_HOUSE", ...]

def read_intrinsics(csv_path: str):
    df = pd.read_csv(csv_path)
    # パラメータ情報、画像サイズはper-frameで保存されているが一定なので代表値（先頭行）を使用
    fx = float(df["K_0"].iloc[0])
    fy = float(df["K_4"].iloc[0])
    cx = float(df["K_2"].iloc[0])
    cy = float(df["K_5"].iloc[0])

    width  = int(df["width"].iloc[0])
    height = int(df["height"].iloc[0])

    return fx, fy, cx, cy, width, height

# 深度画像取得
def list_depthfiles(zip_path, person: str, action: str):
    pid = os.path.splitext(person)[0]
    depth_images = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in zf.namelist():
            if name.startswith(f"{pid}/{action}/depthImages/") and name.lower().endswith(".png"):
                depth_images.append(name)

    return sorted(depth_images, key=lambda s: int(os.path.splitext(os.path.basename(s))[0]))
    # 出力：["0.png", "1.png", ...]


# 深度画像→点群
def data_pointcloud_open3d(png_bytes, width, height, fx, fy, cx, cy):

    # depth_raw = o3d.io.read_image(depth_file)
    # # depth_raw: 深度画像そのもの　各ピクセルにカメラからその方向に何m離れているかが入っている

    # 疑似点群を作成
    buf = np.frombuffer(png_bytes, dtype=np.uint8)
    depth = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    # depth = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED)

    if depth.ndim == 3:
        depth = depth[:, :, 0]

    if depth.dtype == np.uint8:
        depth = (depth.astype(np.uint16) * (4000 // 255)).astype(np.uint16)
        depth_scale = 1000.0
        depth_trunc = 10.0
    else:
        depth_scale = 63.75
        depth_trunc = 10.0

    depth_raw = o3d.geometry.Image(depth)

    # ピンホールカメラモデルを仮定してパラメータ情報を代入
    intrinsic = o3d.camera.PinholeCameraIntrinsic()
    intrinsic.set_intrinsics(width, height, fx, fy, cx, cy)

    # 点群を生成
    pcd = o3d.geometry.PointCloud.create_from_depth_image(
        depth_raw,
        intrinsic,
        depth_scale=depth_scale,   # 最大距離4mと仮定
        depth_trunc=depth_trunc       # 10mより遠い点を除外
    )
    return pcd
    # 保存する場合
    # o3d.io.write_point_cloud(pcd_file, pcd)

def data_to_Pointcloud(base_dir, saved_root):
    stride = 15  # 30fps → 2fps
    action_names = []
    files = []
    labels = []
    person_ids = []
                
    # K camera rule
    k_camera_map = {
        "Person001": "K1",
        "Person002": "K1",
        "Person003": "K1",
        "Person004": "K2",
        "Person005": "K2",
        "Person006": "K2",
        "Person007": "K1",
        "Person008": "K1",
        "Person009": "K1",
        "Person010": "K1",
        "Person011": "K2",
        "Person012": "K2",
        "Person013": "K1",
        "Person014": "K2",
        "Person015": "K2",
        "Person016": "K2",
    }

    def is_valid_action(subject, act):

        # 例:
        # B1_BED_IN
        # K2_FRIDGE_OPEN

        camera = act.split("_", 1)[0]
        # camera == ["K1", "FRIDGE_OPEN"]

        # ===== K camera =====
        if camera.startswith("K"):

            return camera == k_camera_map[subject]

        # ===== synchronized camera =====
        else:

            # B1,D1などだけ使用
            return camera.endswith("1")

    # ラベル生成

    label_person = list_persons(base_dir)[0]
    person_id = os.path.splitext(label_person)[0]    # Person001 を基準に取得
    label_path = os.path.join(base_dir, list_persons(base_dir)[0])
    all_actions = list_actions(label_path, person_id)

    filtered_actions = []

    for act in all_actions:

        # 条件を満たすものだけ使う
        if is_valid_action(person_id, act):

            # B1_BED_IN → BED_IN
            action_name = act.split("_", 1)[1]

            filtered_actions.append(action_name)
            
            print("[DBG] accepted:", act, "->", action_name)    
    # 重複削除
    action_names = sorted(list(set(filtered_actions)))

    # ラベル生成
    label_map = {act: i for i, act in enumerate(action_names)}

    num_classes = len(action_names)

    print("✅ Action → Label map:", label_map)
    print("✅ Num classes:", num_classes)

    for person in list_persons(base_dir):
        zip_path = os.path.join(base_dir, person)
        for action in list_actions(zip_path, person):
            pid = os.path.splitext(person)[0]
            csv_path = os.path.join(saved_root, pid, action, "depth-camera_info.csv")
            if not os.path.exists(csv_path):
                continue
                # このactionは飛ばして次へ
            fx, fy, cx, cy, width, height = read_intrinsics(csv_path)
            label = label_map[action.split("_", 1)[1]]

            depth_images = list_depthfiles(zip_path, person, action)
            with zipfile.ZipFile(zip_path, "r") as zf:
                for member in depth_images:
                    frame_id = int(os.path.splitext(os.path.basename(member))[0])
                    print("[DBG] start frame", person, action, frame_id)
                    if frame_id % stride != 0:
                        continue

                    try:
                        png_bytes = zf.read(member)
                        print("[DBG] read png bytes", len(png_bytes))
                        pcd = data_pointcloud_open3d(
                            png_bytes,
                            width, height,
                            fx, fy, cx, cy
                        )
                        print("[DBG] made pcd. n=", len(pcd.points))
                        pcd = pcd.voxel_down_sample(voxel_size=0.03)
                        pcd, _, _ = background_removal.background_removal(pcd)
                        print("[DBG] hdbscan done. ")
                        if len(pcd.points) == 0:
                            continue
                        print("[DBG] skip empty cluster")                            
                        pcd_path = os.path.join(
                            saved_root,
                            pid,
                            action,
                            "ex2-pcd",
                            f"{frame_id}.pcd"
                        )
                        os.makedirs(os.path.dirname(pcd_path), exist_ok=True)
                        o3d.io.write_point_cloud(pcd_path, pcd)
                        print("[DBG] wrote", pcd_path)
                        files.append(pcd_path)
                        person_ids.append(person)
                        labels.append(label)

                    except Exception as e:
                        print("[ERROR] pointcloud failed")
                        print(" person:", person)
                        print(" action:", action)
                        print(" frame:", frame_id)
                        print(" png:", png_bytes)
                        print(" exception:", repr(e))
                        continue   # ← この1フレームだけ捨てて次へ

    return files, labels, person_ids, label_map


        
