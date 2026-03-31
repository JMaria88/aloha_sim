r"""Example to run one episode of LAP policy eval in ALOHA sim and save the video."""

import os
import time

import matplotlib
from absl import app, flags

matplotlib.use("Agg")
import dm_env
import matplotlib.pyplot as plt
import mediapy
import numpy as np
import scipy.spatial.transform as st
from dm_control.utils import inverse_kinematics as ik
from gdm_robotics.adapters import dm_env_to_gdmr_env_wrapper
from gdm_robotics.interfaces import environment as gdmr_environment
from openpi_client import image_tools, websocket_client_policy

from aloha_sim import task_suite
from aloha_sim.tasks.base import aloha2_task_left_lifted as aloha2_task

flags.DEFINE_string("mjcf_root", None, "MJCF root directory, full path")
flags.DEFINE_list("task_list", None, "List of task, default all")
flags.DEFINE_integer("num_episode_per_task", 1, "Num episode per task")
flags.DEFINE_string("remote_host", "0.0.0.0", "Remote host for policy server")
flags.DEFINE_integer("remote_port", 8000, "Remote port for policy server")
flags.DEFINE_integer("open_loop_horizon", 5, "Open loop horizon")
flags.DEFINE_boolean(
    "debug_viz", False, "Save debug visualization of policy inputs to /tmp/lap_debug/"
)
flags.DEFINE_integer("debug_viz_every", 1, "Save debug viz every N policy queries")

_DT = 0.02
_IMAGE_SIZE = (480, 640)
_ALOHA_CAMERAS = {
    "overhead_cam": _IMAGE_SIZE,
    "worms_eye_cam": _IMAGE_SIZE,
    "wrist_cam_left": _IMAGE_SIZE,
    "wrist_cam_right": _IMAGE_SIZE,
}
_ALOHA_JOINTS = {"joints_pos": 14}
_INIT_ACTION = np.concatenate([aloha2_task.LEFT_HOME_CTRL, aloha2_task.RIGHT_HOME_CTRL])
_PRINT_TIMES = True
ALOHA_CLOSED, ALOHA_OPEN = -0.06135, 1.5155

# Fixed rotation offset: maps initial gripper euler [0, 0.223, π] → [π, 0, 0].
# Applied consistently to all euler readings.
_R_EULER_OFFSET = (
    st.Rotation.from_euler("xyz", [np.pi, 0.0, 0.0])
    * st.Rotation.from_euler("xyz", [np.pi, 1.56859265, 0.0]).inv()
)

# LAP specific
# IMAGE_KEYS = (
#     "left_wrist_0_rgb",  # reference camera (left wrist)
#     "right_wrist_0_rgb",  # wrist camera (right wrist)
# )


def binarize_gripper_actions_np(
    actions: np.ndarray, threshold: float = 0.95
) -> np.ndarray:
    """
    Convert continuous gripper actions to binary (0 or 1) using backward propagation logic.
    """
    actions = actions.astype(np.float32)
    n = actions.shape[0]
    new_actions = np.zeros_like(actions)

    open_mask = actions > threshold
    closed_mask = actions < (1 - threshold)
    in_between_mask = ~(open_mask | closed_mask)

    carry = actions[-1] > threshold  # carry as boolean (True=open)

    for i in reversed(range(n)):
        if not in_between_mask[i]:
            carry = open_mask[i]
        new_actions[i] = float(carry)

    return new_actions


def invert_gripper_actions_np(actions: np.ndarray) -> np.ndarray:
    """Invert gripper binary actions: 1 → 0, 0 → 1."""
    return 1.0 - actions


def _append_task_instruction(
    timestep: dm_env.TimeStep, instruction: str
) -> dm_env.TimeStep:
    """Appends the task instruction to timestep observation."""
    new_observations = timestep.observation
    new_observations.update({"instruction": np.array(instruction)})
    return timestep._replace(observation=new_observations)


def extract_lap_observation(timestep_obs, physics, home_pos=None):
    """Extract observation in LAP format mirroring real-robot `shared.py`.

    Returns keys expected by the LAP pipeline: images, cartesian_position,
    gripper_position, joint_position, state, plus `euler` and `qpos`.

    home_pos: if provided, x and y are reflected about home to make deltas
    consistent with the action frame (dpos[0]*=-1, dpos[1]*=-1). The absolute
    value at home is preserved; only subsequent deltas are flipped.
    """
    SITE_NAME = "right\\gripper"

    # For LAP: left camera as reference, right arm as action doer
    right_image = timestep_obs["wrist_cam_left"][:, 104:744]
    wrist_image = timestep_obs["wrist_cam_right"][:, 104:744]
    overhead_image = timestep_obs["overhead_cam"][:, 104:744]
    overhead_image = overhead_image[204:, 173:]
    right_base_pov_image = timestep_obs["right_base_pov"][:, 104:744]

    h, w = right_base_pov_image.shape[:2]  # 480, 640
    start_x = (w - h) // 2  # (640 - 480) // 2 = 80
    end_x = start_x + h  # 80 + 480 = 560

    # Crop to the center square
    right_base_pov_image = right_base_pov_image[:, start_x:end_x]

    right_image = image_tools.resize_with_pad(right_image, 224, 224)
    wrist_image = image_tools.resize_with_pad(wrist_image, 224, 224)
    overhead_image = image_tools.resize_with_pad(overhead_image, 224, 224)
    right_base_pov_image = image_tools.resize_with_pad(right_base_pov_image, 224, 224)

    wrist_image = wrist_image[::-1, ::-1]

    # import matplotlib.pyplot as plt
    # plt.imshow(right_base_pov_image)
    # plt.show()
    # breakpoint()

    use_overhead_ref = False
    use_right_base_pov_ref = True

    if use_overhead_ref:
        ref_image = overhead_image
        # wrist_image = wrist_image.transpose(1, 0, 2)[:, ::-1]
    elif use_right_base_pov_ref:
        ref_image = right_base_pov_image
    else:
        ref_image = right_image

    # Joints: joints_pos is [left6, left_gripper, right6, right_gripper]
    qpos = timestep_obs["joints_pos"]  # (14,)
    left_joints = qpos[:6]
    left_gripper = qpos[6]
    right_joints = qpos[7:13]
    right_gripper = qpos[13]
    # Get right end-effector pose from physics (right arm is the action doer)
    try:
        right_pos = physics.named.data.site_xpos[SITE_NAME].copy()
        mat9 = physics.named.data.site_xmat[SITE_NAME]
        right_R = np.asarray(mat9).reshape(3, 3)
        euler = (_R_EULER_OFFSET * st.Rotation.from_matrix(right_R)).as_euler("xyz")
        # Transform to action-consistent frame with home position = [π, 0, 0].
        # Empirically: deuler[i]>0 causes raw_euler[2-i] to decrease, so we remap.
        # Offsets shift the home pose (raw=[π,0,0]) to [π,0,0] in the new frame.
        euler = np.array([np.pi - euler[2], -euler[1], np.pi - euler[0]])
        euler = np.arctan2(np.sin(euler), np.cos(euler))  # wrap to (-π, π]

        # Reflect x and y about home position so deltas are consistent with the
        # action frame (dpos[0]*=-1, dpos[1]*=-1), preserving the absolute home value.
        if home_pos is not None:
            right_pos[0] = 2 * home_pos[0] - right_pos[0]
            right_pos[1] = 2 * home_pos[1] - right_pos[1]

        print(right_pos)
        print(euler)

        # cartesian_6d: position + rot6d from euler
        def euler_to_rot6d(euler_angles: np.ndarray) -> np.ndarray:
            rot_matrix = st.Rotation.from_euler("xyz", euler_angles).as_matrix()
            return np.concatenate([rot_matrix[:, 0], rot_matrix[:, 1]], axis=0)

        cartesian_6d = np.concatenate([right_pos, euler_to_rot6d(euler)])
    except Exception:
        # Fallback to zeros if physics info unavailable
        euler = np.zeros(3)
        cartesian_6d = np.zeros(6)

    def binarize_gripper_actions_np(
        actions: np.ndarray, threshold: float = 0.95
    ) -> np.ndarray:
        actions = actions.astype(np.float32)
        n = actions.shape[0]
        new_actions = np.zeros_like(actions)
        open_mask = actions > threshold
        closed_mask = actions < (1 - threshold)
        in_between_mask = ~(open_mask | closed_mask)
        carry = actions[-1] > threshold
        for i in reversed(range(n)):
            if not in_between_mask[i]:
                carry = bool(open_mask[i])
            new_actions[i] = float(carry)
        return new_actions

    gripper_binary = binarize_gripper_actions_np(
        np.array([right_gripper]), (ALOHA_OPEN - ALOHA_CLOSED) / 2
    )[0]
    gripper_position = np.array([gripper_binary])
    joint_position = right_joints  # right arm joints (action doer)
    state = np.concatenate([cartesian_6d, gripper_position])

    return {
        "right_image": ref_image,
        "wrist_image": wrist_image,
        "cartesian_position": cartesian_6d,
        "gripper_position": gripper_position,
        "joint_position": joint_position,
        "state": state,
        "euler": euler,
        "qpos": qpos,
    }


def obs_to_request(curr_obs, instruction):
    """Convert observation to LAP request format."""
    # Ensure images are uint8 (server may expect uint8 RGB images)
    base_image = curr_obs["right_image"]
    wrist_image = curr_obs["wrist_image"]
    if base_image is not None:
        base_image = base_image.astype(np.uint8)
    if wrist_image is not None:
        wrist_image = wrist_image.astype(np.uint8)

    request = {
        "observation": {
            # Provide both legacy keys and model-expected keys.
            "base_image": base_image,
            "wrist_image": wrist_image,
            "base_0_rgb": base_image,
            "left_wrist_0_rgb": wrist_image,
            "cartesian_position": curr_obs["cartesian_position"],
            "gripper_position": curr_obs["gripper_position"],
            "joint_position": curr_obs["joint_position"],
            "state": curr_obs["state"],
        },
        "prompt": instruction,
        "batch_size": None,
    }
    return request


def get_action_from_response(response, curr_obs, physics):
    """Process LAP response to get actions using chained IK for cartesian deltas."""
    # pred_action_chunk = response["actions"].copy()
    pred_action_chunk = np.zeros((5, 14))  # Dummy zero actions for testing

    print(pred_action_chunk)

    # 1. Access site by name to prevent indexing errors if MJCF changes
    # Use the MuJoCo site name defined in your XML
    SITE_NAME = "right\\gripper"

    # Initialize "simulated" current pose with the actual physical pose
    # We update these inside the loop to "chain" the deltas
    virt_pos = physics.named.data.site_xpos[SITE_NAME].copy()
    virt_R = np.asarray(physics.named.data.site_xmat[SITE_NAME]).reshape(3, 3).copy()

    # Get joint IDs for the 8 right-side joints (arm + fingers)
    joint_ids = [
        physics.model.name2id(name, "joint") for name in aloha2_task._RIGHT_ARM_JOINTS
    ]

    full_actions = []
    for action in pred_action_chunk:
        dpos = action[:3]
        deuler = action[3:6]

        # deuler[0] = 0.05
        dpos[0] = 0.01

        dpos[0] *= -1
        dpos[1] *= -1

        roll = deuler[0]
        yaw = deuler[2]
        deuler[0] = -yaw
        deuler[2] = roll


        # Apply deltas directly in raw world frame.
        R_delta = st.Rotation.from_euler("xyz", deuler).as_matrix()
        R_target = virt_R @ R_delta

        target_quat_scipy = st.Rotation.from_matrix(R_target).as_quat()  # Scipy [x,y,z,w]
        target_quat = [
            target_quat_scipy[3],
            target_quat_scipy[0],
            target_quat_scipy[1],
            target_quat_scipy[2],
        ]
        target_pos = virt_pos + dpos

        # --- IK SOLVER ---
        result = ik.qpos_from_site_pose(
            physics=physics,
            site_name=SITE_NAME,
            target_pos=target_pos,
            target_quat=target_quat,
            tol=1e-6,
            max_steps=100,
        )

        if result.success:
            # Extract absolute target for the 8 right-side joints
            target_qpos_all = result.qpos[physics.model.jnt_qposadr[joint_ids]]
            # Slice to only include the 6 arm joints (Indices 7-12 of the actuator array)
            right_arm_joints = target_qpos_all[:6]

            # UPDATE VIRTUAL POSE:
            # Chaining allows action[1] to be relative to the result of action[0]
            virt_pos = target_pos.copy()
            virt_R = R_target.copy()
        else:
            # If IK fails, we maintain previous virtual pose and arm joints
            # (In a real eval, you might want to log this failure)
            current_qpos_all = np.array(
                [physics.named.data.qpos[j] for j in aloha2_task._RIGHT_ARM_JOINTS]
            )
            right_arm_joints = current_qpos_all[:6]

        # --- GRIPPER SCALING ---
        target_gripper_pos = ALOHA_CLOSED + (action[6] * (ALOHA_OPEN - ALOHA_CLOSED))

        # --- ASSEMBLY (14 Actuators) ---
        full_action = np.concatenate(
            [
                _INIT_ACTION[:7],  # Left Arm + Gripper
                right_arm_joints.squeeze(),  # Right Arm (6 joints)
                [target_gripper_pos],  # Right Gripper (1 actuator)
            ]
        )

        full_actions.append(full_action)

    return np.array(full_actions)


def visualize_policy_input(
    request, curr_obs, step: int, task_name: str, out_dir: str = "/tmp/lap_debug"
):
    """Save a debug figure showing images and state sent to the policy server."""
    os.makedirs(out_dir, exist_ok=True)

    obs = request["observation"]
    base_image = obs.get("base_image")
    wrist_image = obs.get("wrist_image")

    cartesian = curr_obs["cartesian_position"]
    gripper = curr_obs["gripper_position"]
    joint_pos = curr_obs["joint_position"]
    state = curr_obs["state"]
    euler = curr_obs["euler"]
    qpos = curr_obs["qpos"]

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(f"Policy Input Debug — task={task_name}  step={step}", fontsize=13)

    # ── Images ──────────────────────────────────────────────────────────────
    ax1 = fig.add_subplot(2, 4, 1)
    if base_image is not None:
        ax1.imshow(base_image)
    ax1.set_title("base_image (ref cam)")
    ax1.axis("off")

    ax2 = fig.add_subplot(2, 4, 2)
    if wrist_image is not None:
        ax2.imshow(wrist_image)
    ax2.set_title("wrist_image")
    ax2.axis("off")

    # ── State text panels ───────────────────────────────────────────────────
    def _text_panel(ax, title, labels, values):
        ax.axis("off")
        ax.set_title(title, fontsize=10)
        lines = [f"{l}: {v:.4f}" for l, v in zip(labels, values)]
        ax.text(
            0.05,
            0.95,
            "\n".join(lines),
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="#f0f0f0", alpha=0.8),
        )

    ax3 = fig.add_subplot(2, 4, 3)
    _text_panel(
        ax3,
        "cartesian_position (pos+rot6d)",
        ["x", "y", "z", "r0", "r1", "r2", "r3", "r4", "r5"],
        cartesian,
    )

    ax4 = fig.add_subplot(2, 4, 4)
    _text_panel(
        ax4,
        "euler (xyz) & gripper",
        ["roll", "pitch", "yaw", "gripper_bin"],
        list(euler) + list(gripper),
    )

    ax5 = fig.add_subplot(2, 4, 5)
    _text_panel(
        ax5,
        "joint_position (right arm, 6)",
        [f"j{i}" for i in range(len(joint_pos))],
        joint_pos,
    )

    ax6 = fig.add_subplot(2, 4, 6)
    _text_panel(
        ax6, "state (cart6d + gripper, 7)", [f"s{i}" for i in range(len(state))], state
    )

    ax7 = fig.add_subplot(2, 4, 7)
    _text_panel(ax7, "qpos full (14)", [f"q{i}" for i in range(len(qpos))], qpos)

    ax8 = fig.add_subplot(2, 4, 8)
    ax8.axis("off")
    ax8.set_title("prompt", fontsize=10)
    ax8.text(
        0.05,
        0.95,
        request.get("prompt", ""),
        transform=ax8.transAxes,
        fontsize=9,
        verticalalignment="top",
        wrap=True,
        bbox=dict(boxstyle="round", facecolor="#e8f4e8", alpha=0.8),
    )

    plt.tight_layout()
    out_path = os.path.join(out_dir, f"{task_name}_step{step:04d}.png")
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"[debug_viz] saved → {out_path}")


def run_episode(
    task_name: str,
    ep_idx: int,
    env: gdmr_environment.Environment,
    policy_client: websocket_client_policy.WebsocketClientPolicy,
    constant_instruction: str,
    open_loop_horizon: int,
    steps_per_vla_action: int,
) -> bool:
    """Runs an episode of the given task."""
    print("Task: ", task_name, " Episode: ", ep_idx)

    print("Homing...")
    timestep = _append_task_instruction(env.reset(), constant_instruction)
    # Capture home position for action-consistent position observations.
    SITE_NAME = "right\\gripper"
    home_pos = env.wrapped_env.physics.named.data.site_xpos[SITE_NAME].copy()
    frames = []
    wrist_frames = []
    right_wrist_frames = []
    policy_wrist_frames = []

    print("Running policy...")
    query_time = 0
    step_time = 0
    i = 0
    actions_from_chunk_completed = 0
    pred_action_chunk = None
    chunk_length = open_loop_horizon

    while not timestep.last() and i < 50:
        i += 1
        frame_start_time = time.time()
        curr_obs = extract_lap_observation(
            timestep.observation, env.wrapped_env.physics, home_pos=home_pos
        )

        # Wait for camera images to become available (camera observables may be
        # delayed). If missing, step the environment with the init action up to a
        # few times before sending a request to the policy server.
        max_wait_steps = 10
        waited = 0
        while (
            curr_obs["right_image"] is None or curr_obs["wrist_image"] is None
        ) and waited < max_wait_steps:
            timestep = env.step(action=_INIT_ACTION)
            timestep = _append_task_instruction(timestep, constant_instruction)
            curr_obs = extract_lap_observation(
                timestep.observation, env.wrapped_env.physics, home_pos=home_pos
            )
            waited += 1
        if curr_obs["right_image"] is None or curr_obs["wrist_image"] is None:
            raise RuntimeError("Camera observations not available after waiting")

        if pred_action_chunk is None or actions_from_chunk_completed >= chunk_length:
            actions_from_chunk_completed = 0
            request = obs_to_request(curr_obs, constant_instruction)
            if flags.FLAGS.debug_viz and (i % flags.FLAGS.debug_viz_every == 0):
                visualize_policy_input(request, curr_obs, step=i, task_name=task_name)
            # response = policy_client.infer(request)
            response = {}
            pred_action_chunk = get_action_from_response(
                response, curr_obs, env.wrapped_env.physics
            )

            chunk_length = min(
                open_loop_horizon,
                len(pred_action_chunk) if hasattr(pred_action_chunk, "__len__") else 1,
            )

        action = (
            pred_action_chunk[actions_from_chunk_completed]
            if isinstance(pred_action_chunk, np.ndarray) and pred_action_chunk.ndim > 1
            else pred_action_chunk
        )
        actions_from_chunk_completed += 1

        query_end_time = time.time()
        query_time += query_end_time - frame_start_time

        for sub_step in range(steps_per_vla_action):
            # Repeat the same action for a few environment steps to simulate open-loop control.
            timestep = env.step(action=action)
            timestep = _append_task_instruction(timestep, constant_instruction)

            if sub_step % 5 == 0:
                frames.append(timestep.observation["overhead_cam"])
                wrist_frames.append(timestep.observation["wrist_cam_left"])
                right_wrist_frames.append(timestep.observation["right_base_pov"])
                policy_wrist_frames.append(timestep.observation["wrist_cam_right"])

            if timestep.last():
                break

        step_time += time.time() - query_end_time
        if i % 100 == 0:
            if _PRINT_TIMES:
                print(
                    f"Step: {i}, Query time: {query_time / 100.0}, Step time:"
                    f" {step_time / 100.0}",
                    end="\r",
                )
            query_time = 0
            step_time = 0

    if timestep.reward >= 1.0:
        print("\nEpisode success.")
    else:
        print("\nEpisode failure.")
    success = timestep.reward >= 1.0
    success_str = "succ" if success else "fail"
    video_path = f"/tmp/{task_name}_ep{ep_idx}_{success_str}.mp4"
    print("Saving video to ", video_path)
    mediapy.write_video(video_path, frames, fps=100)
    # Save left wrist camera video for debugging (if recorded)
    if wrist_frames:
        wrist_video_path = f"/tmp/{task_name}_ep{ep_idx}_{success_str}_wrist_left.mp4"
        print("Saving wrist video to ", wrist_video_path)
        mediapy.write_video(wrist_video_path, wrist_frames, fps=100)

    if right_wrist_frames:
        right_wrist_video_path = (
            f"/tmp/{task_name}_ep{ep_idx}_{success_str}_wrist_right.mp4"
        )
        print("Saving wrist video to ", right_wrist_video_path)
        mediapy.write_video(right_wrist_video_path, right_wrist_frames, fps=100)

    if policy_wrist_frames:
        policy_wrist_video_path = (
            f"/tmp/{task_name}_ep{ep_idx}_{success_str}_policy_wrist.mp4"
        )
        print("Saving policy wrist video to ", policy_wrist_video_path)
        mediapy.write_video(policy_wrist_video_path, policy_wrist_frames, fps=100)
    return success


def main(_):
    # Create policy client
    # policy_client = websocket_client_policy.WebsocketClientPolicy(
    #     flags.FLAGS.remote_host, flags.FLAGS.remote_port
    # )

    success_rates = {}
    all_tasks = list(task_suite.TASK_FACTORIES.keys())
    if flags.FLAGS.task_list:
        selected_tasks = [t for t in flags.FLAGS.task_list if t in all_tasks]
    else:
        selected_tasks = all_tasks
    for task_name in selected_tasks:
        success_count = 0
        for ep_idx in range(flags.FLAGS.num_episode_per_task):
            env = task_suite.create_task_env(
                task_name, time_limit=10.0, mjcf_root=flags.FLAGS.mjcf_root
            )

            VLA_FREQ = 10  # Hz
            dt = env.physics.timestep()  # 0.002
            steps_per_vla_action = int((1 / VLA_FREQ) / dt)  # This will be 50

            # Wrap the composer environment to expose a proper GDMR interface.
            env = dm_env_to_gdmr_env_wrapper.DmEnvToGdmrEnvWrapper(env)

            # Prompt the user for an open-vocabulary instruction (blocking).
            user_instruction = None
            while not user_instruction:
                try:
                    user_instruction = input(
                        f"Enter instruction for task '{task_name}' (open vocabulary): "
                    ).strip()
                except EOFError:
                    # If input is closed, fall back to a non-empty default from the task.
                    user_instruction = env.task.get_instruction() or ""
                if not user_instruction:
                    print(
                        "Instruction cannot be empty. Please enter a non-empty instruction."
                    )

            success = run_episode(
                task_name,
                ep_idx,
                env,
                None,
                user_instruction,
                flags.FLAGS.open_loop_horizon,
                steps_per_vla_action=steps_per_vla_action,
            )
            if success:
                success_count += 1
        success_rates[task_name] = success_count / flags.FLAGS.num_episode_per_task
        print(f"----- Task: {task_name}, Success rate: {success_rates[task_name]}")

    print("All Task Success Rates:")
    for task_name, success_rate in success_rates.items():
        print(f"{task_name}:\t{success_rate}")


if __name__ == "__main__":
    app.run(main)
