import numpy as np
import torch
import smplx
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from tqdm import tqdm

def render_npz_to_mp4(npz_path, model_path, output_mp4, gender='neutral'):
    data = np.load(npz_path)
    poses = torch.from_numpy(data['poses']).float()
    trans = torch.from_numpy(data['trans']).float()
    betas = torch.from_numpy(data['betas'][:10]).float().unsqueeze(0)
    fps = data.get('mocap_framerate', 60.0)
    
    T = poses.shape[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = smplx.create(model_path, model_type='smpl', gender=gender).to(device)

    smpl_skeleton = [
        (0, 1), (0, 2), (0, 3),          # 骨盆到腿部與脊椎
        (1, 4), (2, 5), (3, 6),          # 大腿到膝蓋，脊椎向上
        (4, 7), (5, 8), (6, 9),          # 膝蓋到腳踝，脊椎到胸腔
        (7, 10), (8, 11), (9, 12),       # 腳踝到腳尖，胸腔到脖子
        (12, 15),                        # 脖子到頭
        (9, 13), (9, 14),                # 胸腔到左右肩膀
        (13, 16), (14, 17),              # 肩膀到肘部
        (16, 18), (17, 19),              # 肘部到腕部
        (18, 20), (19, 21)               # 腕部到手掌
    ]

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    writer = FFMpegWriter(fps=int(fps))

    with writer.saving(fig, output_mp4, dpi=100):
        for t in tqdm(range(T)):
            ax.cla()
            
            # SMPL Forward
            output = model(
                betas=betas.to(device),
                body_pose=poses[t:t+1, 3:72].to(device),
                global_orient=poses[t:t+1, :3].to(device),
                transl=trans[t:t+1].to(device)
            )
            
            joints = output.joints.detach().cpu().numpy()[0]
            
            x = joints[:, 0]
            y = joints[:, 2]
            z = -joints[:, 1]

            ax.scatter(x[:24], y[:24], z[:24], c='blue', s=20)

            for start, end in smpl_skeleton:
                ax.plot([x[start], x[end]], [y[start], y[end]], [z[start], z[end]], 
                        c='black', linewidth=2)

            ax.set_xlim(x[0]-1, x[0]+1)
            ax.set_ylim(y[0]-1, y[0]+1)
            ax.set_zlim(z[0]-1, z[0]+1)
            
            ax.set_xlabel('X')
            ax.set_ylabel('Y (Forward)')
            ax.set_zlabel('Z (Up)')
            ax.set_title(f"Frame {t}")
            
            writer.grab_frame()

    print(f"渲染完成：{output_mp4}")