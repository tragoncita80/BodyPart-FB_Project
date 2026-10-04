import os
import numpy as np
from tqdm import tqdm
from pathlib import Path

def split_smpl_npz(input_path, output_path_base):
    """
    處理單個檔案的拆分邏輯
    input_path: 原始 npz 路徑
    output_path_base: 輸出的檔案路徑前綴 (不含 _upper.npz)
    """
    try:
        with np.load(input_path, allow_pickle=True) as loader:
            data = dict(loader)
            
        poses = data['poses']  # (Frames, 156)
        trans = data['trans']  # (Frames, 3)
        
        # 0:Pelvis, 1:L_Hip, 2:R_Hip, 4:L_Knee, 5:R_Knee, 7:L_Ankle, 8:R_Ankle, 10:L_Foot, 11:R_Foot
        lower_body_joints = [0, 1, 2, 4, 5, 7, 8, 10, 11]
        upper_body_joints = [3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21] + list(range(22, 52))

        def get_indices(joint_list):
            indices = []
            for j in joint_list:
                indices.extend([j*3, j*3+1, j*3+2])
            return indices

        lower_indices = get_indices(lower_body_joints)
        upper_indices = get_indices(upper_body_joints)

        lower_poses = np.zeros_like(poses)
        lower_poses[:, lower_indices] = poses[:, lower_indices]
        
        upper_poses = np.zeros_like(poses)
        upper_poses[:, upper_indices] = poses[:, upper_indices]
        upper_poses[:, 0:3] = poses[:, 0:3]
        upper_trans = np.zeros_like(trans)

        os.makedirs(os.path.dirname(output_path_base), exist_ok=True)
        
        lower_data = data.copy()
        lower_data['poses'] = lower_poses
        np.savez(f"{output_path_base}_lower.npz", **lower_data)

        upper_data = data.copy()
        upper_data['poses'] = upper_poses
        upper_data['trans'] = upper_trans
        np.savez(f"{output_path_base}_upper.npz", **upper_data)

    except Exception as e:
        print(f"\n[Error] 處理檔案 {input_path} 時出錯: {e}")

def batch_process_amass(src_root, dst_root):
    """
    遍歷所有子目錄並處理 npz
    """
    file_list = []
    for root, dirs, files in os.walk(src_root):
        for file in files:
            if file.endswith("_poses.npz")
                file_list.append(os.path.join(root, file))
    
    print(f"找到 {len(file_list)} 個檔案，準備開始拆分...")

    for input_path in tqdm(file_list, desc="Processing AMASS"):
        rel_path = os.path.relpath(input_path, src_root)
        out_prefix = os.path.join(dst_root, os.path.splitext(rel_path)[0])
        
        split_smpl_npz(input_path, out_prefix)

if __name__ == "__main__":
    # 原始 AMASS 根目錄
    amass_base = 'humenv/data_preparation/AMASS/datasets'
    output_base = 'humenv/data_preparation/AMASS/datasets_separated'
    
    batch_process_amass(amass_base, output_base)