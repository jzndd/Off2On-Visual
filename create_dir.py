import os

# 任务列表
tasks = [
    "lever-pull-v2", "plate-slide-v2", "soccer-v2", "handle-pull-side-v2", "handle-pull-v2",
    "disassemble-v2", "coffee-pull-v2", "basketball-v2", "peg-unplug-side-v2", "door-close-v2",
    "push-back-v2", "door-lock-v2", "shelf-place-v2", "hammer-v2"
]

print(len(tasks))

# 根目录
base_dir = "ckpt"

# 批量创建目录
for task in tasks:
    dir_path = os.path.join(base_dir, f"{task}_ppo_2D_gae_seed0_sparse")
    os.makedirs(dir_path, exist_ok=True)  # 允许目录已存在
    print(f"Created or exists: {dir_path}")

print("All directories processed.")
