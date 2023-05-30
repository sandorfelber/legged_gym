
import isaacgym # needed, otherwise isaacgym complains about being importde after PyTorch
import argparse
from matplotlib import pyplot as plt, widgets
import numpy as np
import torch

__all__ = ["plot_contacts_quality"]

class _Fmt:
    def __init__(self, heights):
        self.heights=heights

    def __mod__(self, other):
        return "%.2f" % (self.heights[int(other)])

def plot_contacts_quality(contacts_quality, *args):
    if len(args) == 1:
        cfg,  = args
        x = np.array(cfg.terrain.measured_points_x)
        y = np.array(cfg.terrain.measured_points_y)
        z = torch.arange(-cfg.normalization.clip_measurements,
                        cfg.normalization.clip_measurements+cfg.terrain.vertical_scale,
                        cfg.terrain.vertical_scale).numpy()
    elif len(args) == 3:
        x, y, z = args
    else:
        raise ValueError("""Cannot resolve overload. Expected args:\n\
                         - (contacts_quality: numpy.array, cfg: LeggedRobotCfg)
                         - (contacts_quality, x, y, z) all numpy.array""")

    x, y = np.meshgrid(x, y, indexing="ij") #ij to match PyTorch
    min, max = contacts_quality[..., 0].min(), contacts_quality[..., 0].max()

    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    img = ax.scatter(y.flatten(), x.flatten(),
                     c=min + (max-min) * np.random.rand(contacts_quality.shape[0]),
                     cmap="RdYlGn", vmin=min, vmax=max)
    plt.axis("scaled")
    fig.colorbar(img)

    def update(val):       
        z_ind = int(val)
        img.set_array(contacts_quality[:, z_ind, 0])

    z_init = z.shape[0]//2
    ax_slider = fig.add_axes([0.1,0.1,0.03,0.8])
    slider = widgets.Slider(ax_slider, "Height", 0, z.shape[0] - 1,
                            orientation="vertical", valinit=z_init, valstep=1, valfmt=_Fmt(z))
    slider.on_changed(update)

    update(z_init)

    plt.show(block=True)

if __name__ == "__main__":
   
    from legged_gym import LEGGED_GYM_ROOT_DIR
    from legged_gym.envs import *
    from legged_gym.utils.task_registry import task_registry, get_load_path, get_run_path
    import os
   
    parser = argparse.ArgumentParser() 
    parser.add_argument("--task", required=True, help="The task of which to load a run.")
    parser.add_argument("--run", default=-1, help="Name of the run to load the contacts quality map from. If -1 or omitted: will load the last run.")
    parser.add_argument("--checkpoint", default=-1, type=int, help="Saved model checkpoint number. If -1 or omitted: will load the last checkpoint.")

    args = parser.parse_args()

    cfg, train_cfg = task_registry.get_cfgs(args.task)
    
    root = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name)   
    run = get_run_path(root, args.run)
    cfg.update_from(os.path.join(run, "config.yaml"))

    
    path = get_load_path(run_path=run, checkpoint=args.checkpoint) 
    print("Loading checkpoint: ", path)
    contacts_quality = torch.load(path, map_location=torch.device("cpu"))["contacts_quality"].numpy()
    plot_contacts_quality(contacts_quality, cfg)
