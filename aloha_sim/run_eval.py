# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""Example to run one episode of Gemini Robotics On-Device Sim Eval and save the video.

"""

import copy
import time

from absl import app
from absl import flags
from aloha_sim import task_suite
import dm_env
from dm_env import specs
from gdm_robotics.adapters import dm_env_to_gdmr_env_wrapper
from gdm_robotics.interfaces import environment as gdmr_environment
import mediapy
import numpy as np

from safari_sdk.model import constants
from safari_sdk.model import gemini_robotics_policy

flags.DEFINE_string('mjcf_root', None, 'MJCF root directory, full path')
flags.DEFINE_list('task_list', None, 'List of task, default all')
flags.DEFINE_integer('num_episode_per_task', 5, 'Num episode per task')

_DT = 0.02
_IMAGE_SIZE = (480, 848)
_ALOHA_CAMERAS = {
    'overhead_cam': _IMAGE_SIZE,
    'worms_eye_cam': _IMAGE_SIZE,
    'wrist_cam_left': _IMAGE_SIZE,
    'wrist_cam_right': _IMAGE_SIZE,
}
_ALOHA_JOINTS = {'joints_pos': 14}
_INIT_ACTION = np.asarray([
    0.0,
    -0.96,
    1.16,
    0.0,
    -0.3,
    0.0,
    1.5,
    0.0,
    -0.96,
    1.16,
    0.0,
    -0.3,
    0.0,
    1.5,
])
_SERVE_ID = 'gemini_robotics_on_device'
_PRINT_TIMES = True


def _append_task_instruction(
    timestep: dm_env.TimeStep, instruction: str
) -> dm_env.TimeStep:
  """Appends the task instruction to timestep observation."""
  new_observations = timestep.observation
  new_observations.update({'instruction': np.array(instruction)})
  return timestep._replace(observation=new_observations)


def run_episode(
    task_name: str,
    ep_idx: int,
    env: gdmr_environment.Environment,
    policy: gemini_robotics_policy.GeminiRoboticsPolicy,
    constant_instruction: str,
) -> bool:
  """Runs an episode of the given task."""
  print('Task: ', task_name, ' Episode: ', ep_idx)

  print('Homing...')
  timestep = _append_task_instruction(env.reset(), constant_instruction)
  frames = []

  print('Running policy...')
  query_time = 0
  step_time = 0
  i = 0
  policy_state = policy.initial_state()
  while not timestep.last():
    i += 1
    frame_start_time = time.time()
    (action, _), policy_state = policy.step(timestep, policy_state)

    query_end_time = time.time()
    query_time += query_end_time - frame_start_time

    timestep = env.step(action=action)
    timestep = _append_task_instruction(timestep, constant_instruction)
    step_time += time.time() - query_end_time
    if i % 100 == 0:
      if _PRINT_TIMES:
        print(
            f'Step: {i}, Query time: {query_time / 100.}, Step time:'
            f' {step_time / 100.}',
            end='\r',
        )
      query_time = 0
      step_time = 0

    frames.append(timestep.observation['overhead_cam'])
  if timestep.reward >= 1.0:
    print('\nEpisode success.')
  else:
    print('\nEpisode failure.')
  success = timestep.reward >= 1.0
  success_str = 'succ' if success else 'fail'
  video_path = f'/tmp/{task_name}_ep{ep_idx}_{success_str}.mp4'
  print('Saving video to ', video_path)
  mediapy.write_video(video_path, frames)
  return success


def main(_):
  # Create policy
  env_for_spec = task_suite.create_task_env(
      list(task_suite.TASK_FACTORIES.keys())[0],
      time_limit=1.0,
      mjcf_root=flags.FLAGS.mjcf_root,
  )
  env_for_spec = dm_env_to_gdmr_env_wrapper.DmEnvToGdmrEnvWrapper(env_for_spec)
  timestep_spec = copy.deepcopy(env_for_spec.timestep_spec())
  assert isinstance(timestep_spec.observation, dict)
  timestep_spec.observation.update({'instruction': specs.StringArray(shape=())})
  env_for_spec.close()

  try:
    print('Creating policy...')
    policy = gemini_robotics_policy.GeminiRoboticsPolicy(
        serve_id=_SERVE_ID,
        task_instruction_key='instruction',
        image_observation_keys=_ALOHA_CAMERAS.keys(),
        proprioceptive_observation_keys=_ALOHA_JOINTS.keys(),
        min_replan_interval=25,
        inference_mode=constants.InferenceMode.SYNCHRONOUS,
        robotics_api_connection=constants.RoboticsApiConnectionType.LOCAL,
    )
    policy.step_spec(timestep_spec)  # Initialize the policy
    print('GeminiRoboticsPolicy initialized successfully.')
  except ValueError as e:
    print(f'Error initializing policy: {e}')
    raise
  except Exception as e:  # pylint: disable=broad-except
    print(f'An unexpected error occurred during initialization: {e}')
    raise

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
      instruction = env.task.get_instruction()
      # Wrap the composer environment to expose a proper GDMR interface.
      env = dm_env_to_gdmr_env_wrapper.DmEnvToGdmrEnvWrapper(env)
      success = run_episode(task_name, ep_idx, env, policy, instruction)
      if success:
        success_count += 1
    success_rates[task_name] = success_count / flags.FLAGS.num_episode_per_task
    print(f'----- Task: {task_name}, Success rate: {success_rates[task_name]}')

  print('All Task Success Rates:')
  for task_name, success_rate in success_rates.items():
    print(f'{task_name}:\t{success_rate}')


if __name__ == '__main__':
  app.run(main)
