from datetime import datetime
from pathlib import Path
import imageio
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

colors = plt.cm.get_cmap('tab20').colors
eval_counter = 0

initial_grip_trajs = None

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

def plot_grip_trajs(grip_trajs, save_path, expert_trajs=None):
    global eval_counter, initial_grip_trajs
    
    if eval_counter == 0:
        initial_grip_trajs = grip_trajs
        color = 'black'  
    else:
        color = colors[(eval_counter - 1) % len(colors)]  
    
    ax.cla()

    if initial_grip_trajs is not None:
        for grips in initial_grip_trajs:
            xs = [point[0] for point in grips]
            ys = [point[1] for point in grips]
            zs = [point[2] for point in grips]
            ax.plot(xs, ys, zs, color='black')
    
    if expert_trajs is not None:
        for grips in expert_trajs:
            xs = [point[0] for point in grips]
            ys = [point[1] for point in grips]
            zs = [point[2] for point in grips]
            ax.plot(xs, ys, zs, color='red')
        
    for grips in grip_trajs:
        xs = [point[0] for point in grips]
        ys = [point[1] for point in grips]
        zs = [point[2] for point in grips]
        ax.plot(xs, ys, zs, color=color, linewidth=0.7)

    plt.savefig(save_path)
    plt.draw()
    eval_counter += 1

class VideoRecorder:
    def __init__(self,
                 root_dir,
                 fps=20):
        if root_dir is not None:
            self.save_dir = Path(root_dir)
            self.save_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.save_dir = None

        self.fps = fps
        self.frames = []

    def init(self, env, enabled=True):
        self.frames = []
        self.enabled = self.save_dir is not None and enabled
        self.record(env)

    def record(self, env):
        if self.enabled:
            frame = env.render()
            frame = np.rot90(frame, 2)  # for corner 旋转180度 
            self.frames.append(frame)
            
    def save(self, file_name):
        if self.enabled:
            path = self.save_dir / file_name
            imageio.mimsave(str(path), self.frames, fps=self.fps)


# grip_trajs = np.load('/ailab/user/baichenjia/fcy/DPOK/videos/success_traj/coffee-button-v2_trajs_20.npy')
# plot_grip_trajs(grip_trajs, '/ailab/user/baichenjia/fcy/DPOK/videos/success_traj/coffee-button-v2_trajs_20.png')