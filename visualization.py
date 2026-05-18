# ex1
import os
import shutil


base_dir = r"/Users/nodoka-m/Desktop/research"


def save_sample_frames(pcd_dir):

    # 保存先
    dst_dir = os.path.join(
        os.path.dirname(pcd_dir),
        "ex1-pcd_visual"
    )

    os.makedirs(dst_dir, exist_ok=True)

    # pcd取得
    frames = sorted([
        f for f in os.listdir(pcd_dir)
        if f.endswith(".pcd")
    ])

    if len(frames) == 0:
        return

    # サンプル位置
    indices = [
        0,
        len(frames) // 4,
        len(frames) // 2,
        (len(frames) * 3) // 4,
        len(frames) - 1
    ]

    # 重複除去
    indices = sorted(list(set(indices)))

    for idx in indices:

        frame_name = frames[idx]

        src_path = os.path.join(pcd_dir, frame_name)

        dst_path = os.path.join(dst_dir, frame_name)

        shutil.copy(src_path, dst_path)

        print(f"Saved: {dst_path}")


# ==========================================
# 全被験者・全クラス
# ==========================================

subjects = sorted([
    d for d in os.listdir(base_dir)
    if d.startswith("Person")
])

for subject in subjects:

    subject_path = os.path.join(base_dir, subject)
    actions = sorted([
        d for d in os.listdir(subject_path)
        if os.path.isdir(os.path.join(subject_path, d))
    ])

    for action in actions:

        pcd_dir = os.path.join(
            subject_path,
            action,
            "ex1-pcd"
        )

        if not os.path.exists(pcd_dir):
            continue

        print(f"\nProcessing: {pcd_dir}")

        save_sample_frames(pcd_dir)