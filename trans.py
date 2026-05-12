# 出力pcdを条件フォルダに分類するコード（ex-1は終わった）

import os
import shutil
import data_to_pointcloud

base_dir = r"C:\Users\bracy\Documents\Class\graduate shool\naist\U lab\study\Pointcloud_comparison\ResearchData"
camera_root = os.path.join(base_dir, "PointClouds")# アウトプット用

label_map = {
    "B1_BED_IN": 0,
    "B1_BED_OUT": 1,
    "B1_JACKET_OFF": 2,
    "B1_JACKET_ON": 3,
    "B2_BED_IN": 4,
    "B2_BED_OUT": 5,
    "B2_JACKET_OFF": 6,
    "B2_JACKET_ON": 7,
    "D1_EAT": 8,
    "D1_WATER": 9,
    "D2_EAT": 10,
    "D2_WATER": 11,
    "E1_ENTER_HOUSE": 12,
    "E1_LEAVE_HOUSE": 13,
    "E1_SHOES_OFF": 14,
    "E1_SHOES_ON": 15,
    "F1_CLEAN_BATH": 16,
    "F1_TAKE_BATH": 17,
    "K1_FRIDGE_CLOSE": 18,
    "K1_FRIDGE_OPEN": 19,
    "K1_PREPARE_MEAL": 20,
    "K2_FRIDGE_CLOSE": 21,
    "K2_FRIDGE_OPEN": 22,
    "K2_PREPARE_MEAL": 23,
    "L1_FALL_DOWN": 24,
    "L1_SIT_DOWN": 25,
    "L1_STAND_UP": 26,
    "L1_WATCH_TV": 27,
    "L2_FALL_DOWN": 28,
    "L2_SIT_DOWN": 29,
    "L2_STAND_UP": 30,
    "L2_WATCH_TV": 31,
    "W1_BRUSH_TEETH": 32,
}

def reorganize_pcd(pointcloud_root, label_map):
    id_to_action = {v: k for k, v in label_map.items()}

    persons = [d for d in os.listdir(pointcloud_root)
               if os.path.isdir(os.path.join(pointcloud_root, d)) and d.lower().startswith("person")]

    for person in persons:
        person_path = os.path.join(pointcloud_root, person)
        print(f"Processing {person}")

        for folder in os.listdir(person_path):
            label_path = os.path.join(person_path, folder)

            if not folder.isdigit():
                continue

            label_id = int(folder)
            if label_id not in id_to_action:
                print(f"  Unknown label id {label_id} in {person}")
                continue

            action_name = id_to_action[label_id]
            ex1_folder = os.path.join(person_path, action_name, "ex1_pcd")
            os.makedirs(ex1_folder, exist_ok=True)

            for fn in os.listdir(label_path):
                if not fn.lower().endswith(".pcd"):
                    continue

                src = os.path.join(label_path, fn)
                dst = os.path.join(ex1_folder, fn)

                # 上書き回避：同名があれば連番を付ける
                if os.path.exists(dst):
                    base, ext = os.path.splitext(fn)
                    k = 1
                    while True:
                        dst2 = os.path.join(ex1_folder, f"{base}_{k}{ext}")
                        if not os.path.exists(dst2):
                            dst = dst2
                            break
                        k += 1

                shutil.move(src, dst)

            # 空なら削除
            if not os.listdir(label_path):
                os.rmdir(label_path)

    print("Done.")

reorganize_pcd(camera_root, label_map)
