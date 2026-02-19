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

"""Hand over task."""

import dataclasses
import os

from aloha_sim.tasks.base import aloha2_task
from aloha_sim.utils import oobb_utils
from aloha_sim.utils import success_detector_utils
from dm_control import composer
from dm_control import mjcf
from dm_control.composer import initializers
from dm_control.composer.variation import deterministic
from dm_control.composer.variation import distributions
from dm_control.composer.variation import rotations
from dm_control.mjcf import traversal_utils
import numpy as np

ROTATION_IDENTITY = np.array([1.0, 0.0, 0.0, 0.0])

TABLE_HEIGHT = 0.0
RESET_HEIGHT = 0.1

object_position = distributions.Uniform(
    low=[0.12, -0.1, TABLE_HEIGHT + RESET_HEIGHT],
    high=[0.18, 0.1, TABLE_HEIGHT + RESET_HEIGHT],
    single_sample=True,
)
object_rotation = rotations.QuaternionFromAxisAngle(
    axis=(0.0, 0.0, 1.0),
    angle=distributions.Uniform(
        -np.pi * 0.1 - np.pi * 0.5,
        np.pi * 0.1 - np.pi * 0.5,
        single_sample=True,
    ),
)


@dataclasses.dataclass(frozen=True)
class PickObjTaskConfig:
  """Configuration for a pick object task.

  Attributes:
    object_model: file path pointing to xml model of the object.
    success_threshold: threshold for distance between the object center of mass
      and the container center of mass for the task to be considered success.
    instruction: language goal for the task.
  """

  object_model: str
  success_threshold: float
  instruction: str


PICK_OBJ_CONFIGS = {
    # Hands over a banana and puts it in a bowl.
    'banana': PickObjTaskConfig(
        object_model='ycb/011_banana/google_64k/model.xml',
        success_threshold=0.1,
        instruction='move the gripper to the banana, then close it to grasp',
    ),
    # Hands over a pen and puts it in container.
    'pen': PickObjTaskConfig(
        object_model='edr/pen/model.xml',
        success_threshold=0.05,
        instruction='pick up the pen',
    ),
}


class PickObj(aloha2_task.AlohaTask):
  """Pick object task.

  The goal is to pick up the object with one gripper and put it in a container.
  """

  def __init__(
      self,
      object_name,
      **kwargs,
  ):
    """Initializes a new `PickObj` task.

    Args:
      object_name: str specifying the name of the object to be picked up and placed in a container.
      **kwargs: Additional args to pass to the base class.
    """
    super().__init__(
        **kwargs,
    )
    self._success_state = 0

    if object_name not in PICK_OBJ_CONFIGS.keys():
      raise ValueError(
          f'Invalid object name: {object_name}, must be one of'
          f' {PICK_OBJ_CONFIGS.keys()}'
      )
    task_config = PICK_OBJ_CONFIGS[object_name]
    self._dist_threshold = task_config.success_threshold
    self._instruction = task_config.instruction
    object_path = task_config.object_model

    # Adds object
    assets_dir = os.path.join(os.path.dirname(__file__), '../assets')
    self._object_prop = composer.ModelWrapperEntity(
        mjcf.from_path(
            os.path.join(
                assets_dir,
                object_path
            )
        )
    )
    self._scene.add_free_entity(self._object_prop)

    for prop in [self._object_prop]:
      freejoint = traversal_utils.get_freejoint(
          prop.mjcf_model.find_all('body')[0]
      )
      if freejoint:
        freejoint.remove()

    self._object_placers = [
        initializers.PropPlacer(
            props=[self._object_prop],
            position=object_position,
            quaternion=object_rotation,
            ignore_collisions=True,
            settle_physics=False,
        ),
        initializers.PropPlacer(
            props=[self._object_prop],
            position=deterministic.Identity(),
            quaternion=deterministic.Identity(),
            ignore_collisions=True,  # Collisions already resolved.
            settle_physics=True,
        ),
    ]

    # extra for object.
    extra_qpos = np.zeros((7,))
    scene_key = self.root_entity.mjcf_model.find('key', 'neutral_pose')
    scene_key.qpos = np.copy(np.concatenate([scene_key.qpos, extra_qpos]))

  def get_reward(self, physics):
    object_geom_ids = list(
        physics.bind(self._object_prop.mjcf_model.find_all('geom')).element_id
    )

    right_gripper_geom_ids = list(
        physics.bind(
            self.root_entity.mjcf_model.find(
                'body', r'right\gripper_link'
            ).find_all('geom')
        ).element_id
    )

    all_contact_pairs = []
    for contact in physics.data.contact:
      pair = (contact.geom2, contact.geom1)
      all_contact_pairs.append(pair)
      all_contact_pairs.append((contact.geom1, contact.geom2))
    def _touching(geom1_ids, geom2_ids):
      for contact_pair in all_contact_pairs:
        if contact_pair[0] in geom1_ids and contact_pair[1] in geom2_ids:
          return True
        if contact_pair[0] in geom2_ids and contact_pair[1] in geom1_ids:
          return True
      return False

    if self._success_state == 0:
      if _touching(right_gripper_geom_ids, object_geom_ids):
        self._success_state = 1
    elif self._success_state == 1:
      object_picked = False
      object_pos = self._object_prop.get_pose(physics)[0]
      object_z = object_pos[2]
      if np.linalg.norm(object_z) > self._dist_threshold:
        object_picked = True

      if object_picked:
        return 1.0
    else:
      raise ValueError('Invalid success state.')
    return 0.0

  def initialize_episode(self, physics, random_state):
    super().initialize_episode(physics, random_state)
    for prop_placer in self._object_placers:
      prop_placer(physics, random_state)
    self._success_state = 0

  def get_instruction(self):
    return self._instruction
