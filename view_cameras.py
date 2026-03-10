"""Save a figure of all camera views at episode reset (no policy needed).

Usage:
    python view_cameras.py --task_name PickObjBanana --out /tmp/cameras.png
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from absl import app, flags

from aloha_sim import task_suite

flags.DEFINE_string("task_name", "PickObjBananaLeftLifted", "Task name to load")
flags.DEFINE_string("mjcf_root", None, "MJCF root directory, full path")
flags.DEFINE_string("out", "/tmp/cameras.png", "Output image path")


def main(_):
    env = task_suite.create_task_env(
        flags.FLAGS.task_name,
        time_limit=80.0,
        mjcf_root=flags.FLAGS.mjcf_root,
    )
    timestep = env.reset()
    obs = timestep.observation

    cameras = [c for c in task_suite.DEFAULT_CAMERAS if c in obs]

    ncols = 3
    nrows = -(-len(cameras) // ncols)  # ceiling division
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    axes = np.array(axes).flatten()

    for ax, cam in zip(axes, cameras):
        img = obs[cam]
        ax.imshow(img)
        ax.set_title(cam, fontsize=12)
        ax.axis("off")

    for ax in axes[len(cameras):]:
        ax.axis("off")

    fig.suptitle(f"Task: {flags.FLAGS.task_name}", fontsize=14)
    plt.tight_layout()
    out = flags.FLAGS.out
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    plt.savefig(out, dpi=120, bbox_inches="tight")
    print(f"Saved → {out}")


if __name__ == "__main__":
    app.run(main)
