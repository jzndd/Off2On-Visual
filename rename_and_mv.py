import os
import shutil

# 原始路径和目标路径
src_dir = "/data/home/jzn/workspace/Off2On-Visual/outputs/2025.07.07/200400_soccer-v2_ppo_gae_wholetraj_usebc_use50traj_seed1_bs3200_minibs64/saved_models"
dst_dir = "/data/home/jzn/workspace/Off2On-Visual/ckpt/soccer-v2_ppo_2D_gae_sparse/save_data"

# 确保目标目录存在
os.makedirs(dst_dir, exist_ok=True)

# 遍历源目录中的所有 .pth 文件
for filename in os.listdir(src_dir):
    if filename.endswith(".pth"):
        base = os.path.splitext(filename)[0]
        new_filename = f"{base}_seed1.pth"
        
        src_path = os.path.join(src_dir, filename)
        dst_path = os.path.join(dst_dir, new_filename)

        # 复制并重命名文件
        shutil.copy(src_path, dst_path)
        print(f"Copied and renamed: {filename} -> {new_filename}")
