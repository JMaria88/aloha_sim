r"""Example to run one episode of openVLA policy eval in ALOHA sim and save the video.

"""

import copy
from functools import partial
from sre_parse import FLAGS
import time

from absl import app
from absl import flags
from torch import device
from aloha_sim import task_suite
from aloha_sim.tasks.base import aloha2_task
import dm_env
from dm_control.utils import inverse_kinematics as ik
from dm_env import specs
from gdm_robotics.adapters import dm_env_to_gdmr_env_wrapper
from gdm_robotics.interfaces import environment as gdmr_environment
import mediapy
import numpy as np
import scipy.spatial.transform as st
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
import jax

from octo.model.octo_model import OctoModel
from octo.utils.gym_wrappers import HistoryWrapper, NormalizeProprio, RHCWrapper
from octo.utils.train_callbacks import supply_rng

flags.DEFINE_string('mjcf_root', None, 'MJCF root directory, full path')
flags.DEFINE_list('task_list', None, 'List of task, default all')
flags.DEFINE_integer('num_episode_per_task', 1, 'Num episode per task')
flags.DEFINE_integer('open_loop_horizon', 5, 'Open loop horizon')

_DT = 0.02
_IMAGE_SIZE = (480, 640)
_ALOHA_CAMERAS = {
    'overhead_cam': _IMAGE_SIZE,
    'worms_eye_cam': _IMAGE_SIZE,
    'wrist_cam_left': _IMAGE_SIZE,
    'wrist_cam_right': _IMAGE_SIZE,
}
_ALOHA_JOINTS = {'joints_pos': 14}
_INIT_ACTION = np.concatenate([aloha2_task.HOME_CTRL, aloha2_task.HOME_CTRL])
_PRINT_TIMES = True

def extract_octo_observation(timestep_obs, physics):
    """Extract observation in format suitable for openVLA using named access."""
    # Robust site access
    SITE_NAME = 'right\\gripper'
    
    # Image processing (remains the same as your setup)

    use_overhead_ref = False
    use_right_base_pov_ref = False
    use_left_base_pov_ref = True

    # Crop everything to 480x640 first since it's 480x848.
    if use_overhead_ref:
      overhead_image = timestep_obs['overhead_cam'][:, 104:744]
      ref_image = overhead_image[204:, 173:]
    elif use_right_base_pov_ref:
      ref_image = timestep_obs['right_base_pov'][:, 104:744]
    elif use_left_base_pov_ref:
      ref_image = timestep_obs['left_under_arm_pov'][:, 104:744]
    else:
      ref_image = timestep_obs['wrist_cam_left'][:, 104:744]

    h, w = ref_image.shape[:2]  # 480, 640
    start_x = (w - h) // 2      # (640 - 480) // 2 = 80
    end_x = start_x + h         # 80 + 480 = 560
    
    # Crop to the center square
    square_image = ref_image[:, start_x:end_x]
    square_right_image = timestep_obs['wrist_cam_right'][:, 104:744][:, start_x:end_x]

    square_image = Image.fromarray(square_image.astype(np.uint8))
    square_right_image = Image.fromarray(square_right_image.astype(np.uint8))

    # import matplotlib
    # matplotlib.use('TkAgg')
    # import matplotlib.pyplot as plt
    # plt.imshow(square_image)
    # breakpoint()

    image = square_image.resize((256, 256), resample=Image.Resampling.LANCZOS)
    right_wrist_image = square_right_image.resize((128, 128), resample=Image.Resampling.LANCZOS)

    # Joints: ALOHA joints_pos is typically [left7, right7] 
    qpos = timestep_obs['joints_pos']
    
    # Access right end-effector pose by name
    try:
        right_pos = physics.named.data.site_xpos[SITE_NAME]
        mat9 = physics.named.data.site_xmat[SITE_NAME]
        right_R = np.asarray(mat9).reshape(3, 3)
        euler = st.Rotation.from_matrix(right_R).as_euler('xyz')
        
        def euler_to_rot6d(euler_angles):
            rot_matrix = st.Rotation.from_euler('xyz', euler_angles).as_matrix()
            return np.concatenate([rot_matrix[:, 0], rot_matrix[:, 1]], axis=0)
        
        cartesian_6d = np.concatenate([right_pos, euler_to_rot6d(euler)])
    except Exception as e:
        print(f"Warning: Physics access failed: {e}")
        euler = np.zeros(3)
        cartesian_6d = np.zeros(6)

    # State for OpenVLA (Position + Euler + Gripper)
    # Most OpenVLA checkpoints expect 7D or 14D state depending on the wrapper
    right_gripper = qpos[13] if len(qpos) > 13 else 0.0
    state = np.concatenate([cartesian_6d, [right_gripper]])

    return {
        "image": image,
        "wrist_image": right_wrist_image,
        "state": state,
        "qpos": qpos,
        "site_pos": right_pos,
        "site_mat": right_R
    }


def process_octo_action(action, curr_obs, physics):
    """Process single octo action with robust naming and fixed indexing."""
    SITE_NAME = 'right\\gripper'
    
    # 1. Scaling
    ALOHA_CLOSED, ALOHA_OPEN = -0.06135, 1.5155
    target_gripper_pos = ALOHA_CLOSED + (action[6] * (ALOHA_OPEN - ALOHA_CLOSED))

    # 2. Target Pose (Relative to the observation's site pose)
    current_pos = curr_obs["site_pos"]
    current_R = curr_obs["site_mat"]
    
    R_delta = st.Rotation.from_euler('xyz', action[3:6]).as_matrix()
    R_target = current_R @ R_delta
    
    target_quat_scipy = st.Rotation.from_matrix(R_target).as_quat()
    target_quat = [target_quat_scipy[3], target_quat_scipy[0], target_quat_scipy[1], target_quat_scipy[2]]
    target_pos = current_pos + action[:3]

    # 3. Solve IK
    result = ik.qpos_from_site_pose(
        physics=physics,
        site_name=SITE_NAME,
        target_pos=target_pos,
        target_quat=target_quat,
        tol=1e-6,
        max_steps=100
    )

    # 4. Assemble 14-DoF Action
    full_action = np.zeros(14)
    full_action[0:7] = _INIT_ACTION[:7] # Left side home

    if result.success:
        joint_ids = [physics.model.name2id(name, 'joint') for name in aloha2_task._RIGHT_ARM_JOINTS]
        target_qpos_all = result.qpos[physics.model.jnt_qposadr[joint_ids]]
        # Slice out the fingers (last 2 of the 8) to keep it to 6 arm joints
        full_action[7:13] = target_qpos_all[:6]
    else:
        # If IK fails, hold the current right arm position
        current_right_qpos = np.array([physics.named.data.qpos[j] for j in aloha2_task._RIGHT_ARM_JOINTS])
        full_action[7:13] = current_right_qpos[:6]

    # Set Right Gripper to Index 13
    full_action[13] = target_gripper_pos

    return full_action


def run_episode(
    task_name: str,
    ep_idx: int,
    env: gdmr_environment.Environment,
    vla,
    task,
    policy_fn,
    open_loop_horizon: int, # We will use this to define chunk length
    steps_per_vla_action: int,
) -> bool:
    """Runs an episode using Octo with Temporal Aggregation (Open-Loop)."""
    print(f'Task: {task_name} | Episode: {ep_idx}')

    timestep = env.reset()
    frames, wrist_frames, right_wrist_frames, right_pov_frames = [], [], [], []
    i = 0
    MAX_STEPS = 20 # Total number of chunks to execute

    while not timestep.last() and i < MAX_STEPS:
        i += 1
        
        # 1. Get observation
        raw_obs = extract_octo_observation(timestep.observation, env.wrapped_env.physics)
        octo_obs = {
           "image_primary": np.array(raw_obs['image'])[None, None, ...], 
           "timestep_pad_mask": np.array([[True]]),
           "proprio": raw_obs['state'][None, None, ...].astype(np.float32),
           "image_wrist": np.array(raw_obs['wrist_image'])[None, None, ...],
        }

        # 2. Inference: Get the full chunk (e.g., shape [1, 4, 7])
        try:
            action_chunk = policy_fn(octo_obs, task) 
            # Remove batch dimension -> [4, 7]
            action_chunk = np.squeeze(action_chunk) 
        except Exception as e:
            print(f"Inference failed: {e}")
            break

        # 3. Temporal Aggregation Loop: Execute each step in the chunk
        # Note: 'open_loop_horizon' determines how many steps of the chunk to trust
        for chunk_idx in range(min(len(action_chunk), open_loop_horizon)):
            executed_action = action_chunk[chunk_idx]

            # Process the specific action in the sequence
            full_action = process_octo_action(executed_action, raw_obs, env.wrapped_env.physics)

            # Step simulation
            for sub_step in range(steps_per_vla_action):
                timestep = env.step(action=full_action)
                
                # Update raw_obs for the next step in the chunk to keep IK stable
                # Though in pure open-loop, we usually use the initial obs for the whole chunk
                
                if sub_step % 5 == 0:
                    frames.append(timestep.observation['overhead_cam'])
                    if 'left_under_arm_pov' in timestep.observation:
                        wrist_frames.append(timestep.observation['left_under_arm_pov'])
                    if 'wrist_cam_right' in timestep.observation:
                        right_wrist_frames.append(timestep.observation['wrist_cam_right'])
                    if 'right_base_pov' in timestep.observation:
                        right_pov_frames.append(timestep.observation['right_base_pov'])

                if timestep.last():
                    break
            
            if timestep.last():
                break
            
            # Update current observation for the next action in the chunk
            raw_obs = extract_octo_observation(timestep.observation, env.wrapped_env.physics)

    success = (timestep.reward >= 1.0)
    print(f'\nEpisode {"SUCCESS" if success else "FAILURE"}. Final Reward: {timestep.reward}')
    success_str = 'succ' if success else 'fail'
    video_path = f'/tmp/{task_name}_ep{ep_idx}_{success_str}.mp4'
    print('Saving video to ', video_path)
    mediapy.write_video(video_path, frames, fps=100)
    # Save left wrist camera video for debugging (if recorded)
    if wrist_frames:
        wrist_video_path = f'/tmp/{task_name}_ep{ep_idx}_{success_str}_wrist_left.mp4'
        print('Saving wrist video to ', wrist_video_path)
        mediapy.write_video(wrist_video_path, wrist_frames, fps=100)

    if right_wrist_frames:
        right_wrist_video_path = f'/tmp/{task_name}_ep{ep_idx}_{success_str}_wrist_right.mp4'
        print('Saving wrist video to ', right_wrist_video_path)
        mediapy.write_video(right_wrist_video_path, right_wrist_frames, fps=100)

    if right_pov_frames:
        right_pov_video_path = f'/tmp/{task_name}_ep{ep_idx}_{success_str}_right_base_pov.mp4'
        print('Saving right base pov video to ', right_pov_video_path)
        mediapy.write_video(right_pov_video_path, right_pov_frames, fps=100)
    return success


def main(_):
  # Load openVLA model
  vla = OctoModel.load_pretrained("hf://rail-berkeley/octo-base-1.5")
  print("Octo model loaded successfully.")

  policy_fn = supply_rng(
        partial(
            vla.sample_actions,
            unnormalization_statistics=vla.dataset_statistics["bridge_dataset"]["action"],
        ),
  )

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
          task_name, time_limit=80.0, mjcf_root=flags.FLAGS.mjcf_root
      )
      
      VLA_FREQ = 10  # Hz
      dt = env.physics.timestep()  # 0.002
      steps_per_vla_action = int((1 / VLA_FREQ) / dt)  # This will be 50
      instruction = env.task.get_instruction()

      task = vla.create_tasks(texts=[instruction])

      # Wrap the composer environment to expose a proper GDMR interface.
      env = dm_env_to_gdmr_env_wrapper.DmEnvToGdmrEnvWrapper(env)

      # Prompt the user for an open-vocabulary instruction (blocking).
      # user_instruction = None
      # while not user_instruction:
      #   try:
      #     user_instruction = input(
      #         f"Enter instruction for task '{task_name}' (open vocabulary): "
      #     ).strip()
      #   except EOFError:
      #     # If input is closed, fall back to a non-empty default from the task.
      #     user_instruction = env.task.get_instruction() or ""
      #   if not user_instruction:
      #     print('Instruction cannot be empty. Please enter a non-empty instruction.')


      success = run_episode(
          task_name,
          ep_idx,
          env,
          vla,
          task,
          policy_fn,
          flags.FLAGS.open_loop_horizon,
          steps_per_vla_action,
      )
      if success:
        success_count += 1
    success_rates[task_name] = success_count / flags.FLAGS.num_episode_per_task
    print(f'----- Task: {task_name}, Success rate: {success_rates[task_name]}')

  print('All Task Success Rates:')
  for task_name, success_rate in success_rates.items():
    print(f'{task_name}:\t{success_rate}')


if __name__ == '__main__':
  app.run(main)