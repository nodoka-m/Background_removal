import os
import zipfile
import shutil
import time
import gc
from bagpy import bagreader

base_dir = r"C:\Users\bracy\Documents\Class\graduate shool\naist\U lab\study\Pointcloud_comparison\ResearchData"
output_root = os.path.join(base_dir, "PointClouds")
os.makedirs(output_root, exist_ok=True)

zip_files = [f for f in os.listdir(base_dir) if f.startswith("Person") and f.endswith(".zip")]
print(f"Found ZIP files: {zip_files}")

TOPIC = "depth/camera_info"

workdir = os.path.join(output_root, "_work_bagpy")
os.makedirs(workdir, exist_ok=True)

for zip_name in zip_files:
    zip_path = os.path.join(base_dir, zip_name)
    person_id = os.path.splitext(zip_name)[0]
    print(f"\n=== Processing {person_id} ===")

    with zipfile.ZipFile(zip_path, "r") as z:
        members = z.namelist()
        bag_paths = [m for m in members if m.endswith("camera_info.bag")]

        if not bag_paths:
            print(f"  ⚠️ No camera_info.bag in {zip_name}")
            continue

        for bag_inner_path in bag_paths:
            print(f"  ▶ Converting: {bag_inner_path}")
            action_name = bag_inner_path.split("/")[-2]   # ★安全にaction名

            out_dir = os.path.join(output_root, person_id, action_name)
            os.makedirs(out_dir, exist_ok=True)

            # ★毎回ユニーク名で保存（ロック残り対策）
            tmp_bag_path = os.path.join(workdir, f"{person_id}_{action_name}_camera_info.bag")

            # zip → 固定workdirへ展開
            with z.open(bag_inner_path) as bag_data, open(tmp_bag_path, "wb") as f:
                shutil.copyfileobj(bag_data, f)

            bag = None
            try:
                bag = bagreader(tmp_bag_path)
                csv_path = bag.message_by_topic(TOPIC)

                # bagを閉じる（念押し）
                try:
                    if hasattr(bag, "bag") and bag.bag is not None:
                        bag.bag.close()
                except Exception:
                    pass

                bag = None
                gc.collect()
                time.sleep(0.2)

                target_csv = os.path.join(out_dir, "depth-camera_info.csv")
                shutil.move(csv_path, target_csv)
                print(f"    ✔ Saved CSV → {target_csv}")

            except Exception as e:
                print(f"    ❌ Error converting bag: {e}")

            finally:
                # 念押しclose
                try:
                    if bag is not None and hasattr(bag, "bag") and bag.bag is not None:
                        bag.bag.close()
                except Exception:
                    pass

# ★最後に workdir は手動削除でOK（ロック残る可能性あるので）
print("\nDone. workdir:", workdir)
print("If needed, delete the folder after closing Python/VSCode:", workdir)
