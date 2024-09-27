# Copyright 2024 The Kubric Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Worker file for the Multi-Object Video (MOVi) C (and CC) datasets.
  * The number of objects is randomly chosen between
    --min_num_objects (3) and --max_num_objects (10)
  * The objects are randomly chosen from the Google Scanned Objects dataset

  * Background is an random HDRI from the HDRI Haven dataset,
    projected onto a Dome (half-sphere).
    The HDRI is also used for lighting the scene.
"""


"""
kubric notes:
    // https://github.com/google-research/kubric/blob/main/kubric/core/objects.py: 
    
    bounds (Tuple[vec3d, vec3d]): An axis aligned bounding box around the object relative to its
                                  center, but ignoring any scaling or rotation.
        
    @property
    def bbox_3d(self):
         3D bounding box as an array of 8 corners (shape = [8, 3])
    
    @property
    def aabbox(self):
         Axis-aligned bounding box [(min_x, min_y, min_y), (max_x, max_y, max_z)].

Notes-to-self:
         
Singularity command to connect to the container where you can run this:
        singularity shell -p --nv       --bind /mnt/lustre/work/bethge/bkr857/projects/MADS:/src/MADS  kubric_image.sif

Then: cd /src/MADS/kubric/kubric-for-mads

command to run this script inside the container:
    /usr/bin/python3 -m pdb challenges/movi/movi_c_worker_segmentation_test.py  --camera=fixed_random


"""

import logging

import bpy
import kubric as kb
from kubric.simulator import PyBullet
from kubric.renderer import Blender
import numpy as np


# --- Some configuration values
# the region in which to place objects [(min), (max)]
SPAWN_REGION = [(-5, -5, 1), (5, 5, 5)]
VELOCITY_RANGE = [(-4., -4., 0.), (4., 4., 0.)]


def parse():
    # --- CLI arguments
    parser = kb.ArgumentParser()
    parser.add_argument("--objects_split", choices=["train", "test"],
                        default="train")
    # Configuration for the objects of the scene
    parser.add_argument("--min_num_objects", type=int, default=2,
                        help="minimum number of objects")
    parser.add_argument("--max_num_objects", type=int, default=3,
                        help="maximum number of objects")

    # Params added for MADS
    parser.add_argument("--num_occluded", choices=[1, 2], default=1,
                        help="Whether to use one or two occluded objects")
    parser.add_argument("--occluder_moves", action="store_true",
                        help="whether the occluder should move")
    parser.add_argument("--occluded_moves", action="store_true",
                        help="whether the occluded object should move (if both occluder and occluded move, they share the same velocity)")
    parser.add_argument("--occluded_relatable", action="store_true",
                        help="only used with two occluded objects; whether or not their edges, when interpolated, meet behind the occluder. (Not implemented yet, not sure I'll be able to.)")
    parser.add_argument("--occluded_textured", action="store_true",
                        help="whether the occluded object should have a texture (vs uniform color)")
    parser.add_argument("--movement_direction", choices=["parallel", "orthogonal"], default="orthogonal",
                        help="In which direction the moving objects move. Parallel: parallel to the axis along which "
                             "the occluded object(s) point out behind the occluder. In case of two occluded objects,"
                             " that's the axis along which the occluder is shorter.")

    # Configuration for the floor and background
    parser.add_argument("--floor_friction", type=float, default=0.0)
    parser.add_argument("--floor_restitution", type=float, default=0.0)
    parser.add_argument("--backgrounds_split", choices=["train", "test"],
                        default="train")

    parser.add_argument("--camera", choices=["fixed_random", "linear_movement"],
                        default="fixed_random")
    parser.add_argument("--max_camera_movement", type=float, default=4.0)


    # Configuration for the source of the assets
    parser.add_argument("--kubasic_assets", type=str,
                        default="gs://kubric-public/assets/KuBasic/KuBasic.json")
    parser.add_argument("--hdri_assets", type=str,
                        default="gs://kubric-public/assets/HDRI_haven/HDRI_haven.json")
    parser.add_argument("--gso_assets", type=str,
                        default="gs://kubric-public/assets/GSO/GSO.json")
    parser.add_argument("--save_state", dest="save_state", action="store_true")
    parser.set_defaults(save_state=False, frame_end=12, #24,
                        frame_rate=12,
                        resolution=256)
    return parser.parse_args()



def get_linear_camera_motion_start_end(
        rng,
        movement_speed: float,
        inner_radius: float = 8.,
        outer_radius: float = 12.,
        z_offset: float = 0.1,
    ):
      """Sample a linear path which starts and ends within a half-sphere shell."""
      while True:
        camera_start = np.array(kb.sample_point_in_half_sphere_shell(inner_radius,
                                                                     outer_radius,
                                                                     z_offset))
        direction = rng.rand(3) - 0.5
        movement = direction / np.linalg.norm(direction) * movement_speed
        camera_end = camera_start + movement
        if (inner_radius <= np.linalg.norm(camera_end) <= outer_radius and
            camera_end[2] > z_offset):
          return camera_start, camera_end


def main():
    FLAGS = parse()
    # FLAGS = parser.parse_args()

    # --- Common setups & resources
    scene, rng, output_dir, scratch_dir = kb.setup(FLAGS)
    scene.gravity = (0, 0, 0)
    simulator = PyBullet(scene, scratch_dir)
    renderer = Blender(scene, scratch_dir, samples_per_pixel=64)
    kubasic = kb.AssetSource.from_manifest(FLAGS.kubasic_assets)
    gso = kb.AssetSource.from_manifest(FLAGS.gso_assets)
    hdri_source = kb.AssetSource.from_manifest(FLAGS.hdri_assets)


    # --- Populate the scene
    # background HDRI
    train_backgrounds, test_backgrounds = hdri_source.get_test_split(fraction=0.1)
    if FLAGS.backgrounds_split == "train":
      logging.info("Choosing one of the %d training backgrounds...", len(train_backgrounds))
      hdri_id = rng.choice(train_backgrounds)
    else:
      logging.info("Choosing one of the %d held-out backgrounds...", len(test_backgrounds))
      hdri_id = rng.choice(test_backgrounds)
    background_hdri = hdri_source.create(asset_id=hdri_id)
    #assert isinstance(background_hdri, kb.Texture)
    logging.info("Using background %s", hdri_id)
    scene.metadata["background"] = hdri_id
    renderer._set_ambient_light_hdri(background_hdri.filename)

    # Dome
    dome = kubasic.create(asset_id="dome", name="dome",
                          friction=FLAGS.floor_friction,
                          restitution=FLAGS.floor_restitution,
                          static=True, background=True)
    assert isinstance(dome, kb.FileBasedObject)
    scene += dome
    dome_blender = dome.linked_objects[renderer]
    texture_node = dome_blender.data.materials[0].node_tree.nodes["Image Texture"]
    texture_node.image = bpy.data.images.load(background_hdri.filename)

    # Camera
    logging.info("Setting up the Camera...")
    scene.camera = kb.PerspectiveCamera(focal_length=35., sensor_width=32)
    if FLAGS.camera == "fixed_random":
      scene.camera.position = (  0, -1.5, 0.5)
          # kb.sample_point_in_half_sphere_shell(
          # inner_radius=7., outer_radius=9., offset=0.1))
      scene.camera.look_at((0, 0, 0.5))
      print("Camera position: ", scene.camera.position)
    elif FLAGS.camera == "linear_movement":
      # raise ValueError()
      camera_start, camera_end = get_linear_camera_motion_start_end(
          rng,
          movement_speed=rng.uniform(low=0., high=FLAGS.max_camera_movement)
      )
      # linearly interpolate the camera position between these two points
      # while keeping it focused on the center of the scene
      # we start one frame early and end one frame late to ensure that
      # forward and backward flow are still consistent for the last and first frames
      for frame in range(FLAGS.frame_start - 1, FLAGS.frame_end + 2):
        interp = ((frame - FLAGS.frame_start + 1) /
                  (FLAGS.frame_end - FLAGS.frame_start + 3))
        scene.camera.position = (interp * np.array(camera_start) +
                                 (1 - interp) * np.array(camera_end))
        scene.camera.look_at((0, 0, 0))
        scene.camera.keyframe_insert("position", frame)
        scene.camera.keyframe_insert("quaternion", frame)


    # Add random objects
    train_split, test_split = gso.get_test_split(fraction=0.1)
    if FLAGS.objects_split == "train":
      logging.info("Choosing one of the %d training objects...", len(train_split))
      active_split = train_split
    else:
      logging.info("Choosing one of the %d held-out objects...", len(test_split))
      active_split = test_split


    num_objects = FLAGS.num_occluded + 1 #rng.randint(FLAGS.min_num_objects,
                     #         FLAGS.max_num_objects+1)
    logging.info("Placing %d objects:", num_objects)
    occluder = None
    occluded = []

    def create_object(condition=None, ):
      # kb.move_until_no_overlap(obj, simulator, spawn_region=SPAWN_REGION, rng=rng)
      # initialize velocity randomly but biased towards center
      for i in range(500):
        obj = gso.create(asset_id=rng.choice(active_split))
        assert isinstance(obj, kb.FileBasedObject)
        scale = 1 #rng.uniform(0.75, 3.0)
        obj.scale = scale / np.max(obj.bounds[1] - obj.bounds[0])
        obj.metadata["scale"] = scale
        # objs.append(obj)
        if (condition is None or condition(obj)):
          return obj

      raise ValueError("Could not sample a valid object after 500 attempts")



    # def add_object_1():
    #   obj = create_object()
    #   # TODO
    #   return obj
    #
    # def add_object_2(obj_1):
    #   obj = create_object()
    #
    # def add_occluder(occluded: list):
    #   obj = create_object()


    # occluded.append(add_object_1())
    def condition_width_vs_height(obj):
        ''' Tests whether either width >> height or the other way around.
            Useful property for both occluders and occluded objs.'''
        width = (obj.bounds[1][0] - obj.bounds[0][0])
        height = (obj.bounds[1][2] - obj.bounds[0][2])
        if (width >= 3*height) or (height >= 3*width):
            return True
        return False

    def condition_fills_space(obj, threshold=0.5):
        '''return whether the true object volume comes close to the one computed by
           multiplying the sides of its bounding box.
           Over 0.5: Seem to be mostly rectangular objects.
           Below 0.2 can still be pans etc that have small volume but do fill space.'''
        return obj.metadata["volume"] > threshold * np.prod(obj.bounds[1] - obj.bounds[0])

    def condition_occludes_obj(obj, occluded_obj):
        return occludes_obj_in_one_direction(obj, occluded_obj)[0]
    def occludes_obj_in_one_direction(obj, occluded_obj):
        '''check whether occluded_obj can point out nicely in one direction and is hidden in the other.
           Also returns the direction - 0 for x, 2 for z - in which the occluded object points out.'''
        # import pdb
        # pdb.set_trace()
        width = (obj.bounds[1][0] - obj.bounds[0][0])
        height = (obj.bounds[1][2] - obj.bounds[0][2])
        occluded_width = (occluded_obj.bounds[1][0] - occluded_obj.bounds[0][0])
        occluded_height = (occluded_obj.bounds[1][2] - occluded_obj.bounds[0][2])
        if (width > occluded_width) and (height > occluded_height):
            return False, -1
        elif (width > 1.5 * occluded_width) and ((width - occluded_width) > 0.1) and (height < 0.5 * occluded_height):
            return True, 2
        elif (height > 1.5 * occluded_height) and ((height - occluded_height) > 0.1) and (width < 0.5 * occluded_width):
            return True, 0
        else:
            return False, -1

    def condition_occludes_two_objects(obj, occluded_1, occluded_2):
        return can_occlude_objects_in_one_direction(obj, occluded_1, occluded_2)[0]
    def can_occlude_objects_in_one_direction(obj, occluded_1, occluded_2):
        '''Test whether either in x or in z direction both objects fit behind the occluder in one direction.'''
        width = (obj.bounds[1][0] - obj.bounds[0][0])
        height = (obj.bounds[1][2] - obj.bounds[0][2])
        occluded1_width = (occluded_1.bounds[1][0] - occluded_1.bounds[0][0])
        occluded1_height = (occluded_1.bounds[1][2] - occluded_1.bounds[0][2])
        occluded2_width = (occluded_2.bounds[1][0] - occluded_2.bounds[0][0])
        occluded2_height = (occluded_2.bounds[1][2] - occluded_2.bounds[0][2])
        occluded_width = max(occluded1_width , occluded2_width)
        occluded_height = max(occluded1_height , occluded2_height)
        if (width > 1.5 * occluded_width):
            return (True, 2)
        elif (height > 1.5 * occluded_height):
            return True, 0
        else:
            return False, -1

    def determine_occluded_heights(occluder_height, occluded_1_height, occluded_2_height):
        ''' First check whether the occluder is higher than the average of the two occluded objects.
            If yes, place them so that for each, half of the object points out of the occluder.
            If not, place them so that they touch behind the center of the occluder.

            Edit: It's almost equivalent, but instead do: for each occluded object, place it
                    at +- 1/2*max(occludedi_height, occluder_height)
            '''
        z0 = - 0.5 * max(occluded1_height, occluder_height)
        z1 = 0.5 * max(occluded2_height, occluder_height)
        return z0, z1
        # avg_height = 0.5 * (occluded_1_height + occluded_2_height)
        # if avg_height <= occluder_height:
        #     z0 = -0.5 * occluder_height
        #     z1 = +0.5 * occluder_height
        #     return z0, z1
        # else:
        #     z0 = -occluded1_height
        #     z1 = occluded2_height
    # space_filling_objs = []
    # for i in range(10):
    #     obj = create_object(condition_fills_space)
    #     space_filling_objs.append(obj)
    # import pdb
    # pdb.set_trace()

    for i in range(FLAGS.num_occluded):
      obj = create_object(condition=(None))#None if FLAGS.num_occluded == 2 else condition_width_vs_height))
      if not FLAGS.occluded_moves:
          obj.static = True
      scene += obj
      occluded.append(obj)
      # bounds = obj.bounds
      # aabbox = obj.aabbox
      # sa = obj.metadata["surface_area"]
      # vol = obj.metadata["volume"]
      # # vol_from_bounds = np.prod(bounds[1] - bounds[0])
      # vol_from_aabbox = np.prod(aabbox[1] - aabbox[0])
      # # if vol

    # Occluder
    if FLAGS.num_occluded == 1:
        occluder = create_object(condition=lambda x: (condition_fills_space(x) and condition_width_vs_height(x)
                                                      and condition_occludes_obj(x, occluded[0])))
    else:
        occluder = create_object(condition=lambda x: (condition_fills_space(x) and condition_width_vs_height(x)
                                                      and condition_occludes_two_objects(x, occluded[0], occluded[1])))
    # occluder.position = (0, 0, 0.5) #(obj.bounds[1][2] - obj.bounds[0][2])  / 2 * 1.1)# 0.01) #obj.scale[2] / 2)
    if not FLAGS.occluder_moves:
        occluder.velocity = (0, 0, 0)
        # occluder.static = True

    occluder_width = (occluder.bounds[1][0] - occluder.bounds[0][0]) # width in x-direction
    occluder_height = (occluder.bounds[1][2] - occluder.bounds[0][2]) # height in z-direction
    occluder_depth = (occluder.bounds[1][1] - occluder.bounds[0][1]) # depth in y-direction

    assert FLAGS.movement_direction == "orthogonal", "Other direction not implemented yet!"

    # In case of one object, determine the direction of occlusion -> determine the movement direction, plus the velocity
    if FLAGS.num_occluded == 1:
        occluded_width = (occluded[0].bounds[1][0] - occluded[0].bounds[0][0]) # width in x-direction
        occluded_height = (occluded[0].bounds[1][2] - occluded[0].bounds[0][2]) # height in z-direction
        occluded_depth = (occluded[0].bounds[1][1] - occluded[0].bounds[0][1])
        width_diff = (occluder_width - occluded_width)
        height_diff = (occluder_height - occluded_height)

        occl, visible_direction = occludes_obj_in_one_direction(occluder, occluded[0])
        assert visible_direction != -1
        if visible_direction == 0:
            # Rotate both
            quat = np.array([np.cos(np.pi/4), 0, np.sin(np.pi/4), 0])
            # occluded[0].quaternion = quat
            # occluder.quaternion = quat
            occluded_width, occluded_height = occluded_height, occluded_width
            occluder_width, occluder_height = occluder_height, occluder_width
            width_diff, height_diff = height_diff, width_diff
            # # movement_direction = 2 if FLAGS.movement_direction == "orthogonal" else 0

        elif visible_direction == 2:
            pass

        if FLAGS.movement_direction == "orthogonal":
            movement_direction = 0
            if np.logical_xor((FLAGS.occluded_moves), (FLAGS.occluder_moves)):
                # In those cases we have a maximum distance the moving object can travel
                max_dist = width_diff
                logging.log(logging.INFO, f"Max distance to travel: {max_dist}")
                dist_per_step = 0.8 * max_dist / (FLAGS.frame_end - FLAGS.frame_start)
                print(f"Distance travelled per step: {dist_per_step}")
                if FLAGS.occluded_moves:
                    occluded[0].position = (-0.40 * max_dist, 0, 0.5)
                    occluded[0].velocity = (dist_per_step, 0, 0)
                    occluder.position = (0, (occluded_depth + occluder_depth) * 0.55, 0.5)
                    occluder.velocity = (0, 0, 0)
                else:
                    assert FLAGS.occluder_moves
                    occluded[0].position = (0, 0, 0.5)
                    occluded[0].velocity = (0, 0, 0)
                    occluder.position = (-0.4 * max_dist,  (occluded_depth + occluder_depth) * 0.55, 0.5)
                    occluder.velocity = (dist_per_step, 0, 0)

            elif (FLAGS.occluded_moves and FLAGS.occluder_moves):
                # dist_per_step = 0.5 * (occluded_width + occluder_width) / (FLAGS.frame_end - FLAGS.frame_start)
                dist_per_step = 0.8 * width_diff / (FLAGS.frame_end - FLAGS.frame_start) # for comparability to other setting, use width_diff
                print(f"Distance travelled per step: {dist_per_step}")
                start_ = -0.4 * width_diff
                occluder.position = (start_, (occluded_depth + occluder_depth) * 0.55, 0.5)
                occluded[0].position = (start_, 0, 0.5)
                occluder.velocity = (dist_per_step, 0, 0)
                occluded[0].velocity = (dist_per_step, 0, 0)
            else:
                # no movement, just place both centrally
                occluder.position = (0,  (occluded_depth + occluder_depth) * 0.55, 0.5)
                occluded[0].position = (0, 0, 0.5)
                occluded[0].static = True
                occluded[0].velocity = (0, 0, 0)

        # Parallel movement direction; not implemented yet:
        else:
            movement_direction = 2
            if np.logical_xor((FLAGS.occluded_moves), (FLAGS.occluder_moves)):
                # In those cases we have a maximum distance the moving object can travel
                max_dist = -height_diff
                dist_per_step =  0.8 * max_dist / (FLAGS.frame_end - FLAGS.frame_start)
                raise NotImplementedError()
            else:   # Todo
                raise NotImplementedError()

    else:
        # Two objects.
        assert FLAGS.num_occluded == 2
        occluded1_width = (occluded[0].bounds[1][0] - occluded[0].bounds[0][0])  # width in x-direction
        occluded1_height = (occluded[0].bounds[1][2] - occluded[0].bounds[0][2])  # height in z-direction
        occluded1_depth = (occluded[0].bounds[1][1] - occluded[0].bounds[0][1])  # depth in y-direction
        occluded2_width = (occluded[1].bounds[1][0] - occluded[1].bounds[0][0])  # width in x-direction
        occluded2_height = (occluded[1].bounds[1][2] - occluded[1].bounds[0][2])  # height in z-direction
        occluded2_depth = (occluded[1].bounds[1][1] - occluded[1].bounds[0][1])

        # Determine the direction in which the occluder can hide the two objects
        occludes, visible_direction = can_occlude_objects_in_one_direction(occluder, occluded[0], occluded[1])
        assert occludes
        if visible_direction == 0:
            # Rotate both
            quat = np.array([np.cos(np.pi/4), 0, np.sin(np.pi/4), 0])
            # occluded[0].quaternion = quat
            # occluded[1].quaternion = quat
            # occluder.quaternion = quat
            occluded1_width, occluded1_height = occluded1_height, occluded1_width
            occluded2_width, occluded2_height = occluded2_height, occluded2_width
            occluder_width, occluder_height = occluder_height, occluder_width

        if FLAGS.movement_direction == "orthogonal":
            movement_direction = 0
            max_dist = occluder_width - max(occluded1_width, occluded2_width)
            z0, z1 = determine_occluded_heights(occluder, occluded[0], occluded[1])
            if np.logical_xor((FLAGS.occluded_moves), (FLAGS.occluder_moves)):
                dist_per_step = 0.8 * max_dist / (FLAGS.frame_end - FLAGS.frame_start)
                print(f"Distance travelled per step: {dist_per_step}")
                if FLAGS.occluded_moves:
                    occluded[0].position = (-0.40 * max_dist, 0, 0.5 + z0)
                    occluded[0].velocity = (dist_per_step, 0, 0)
                    occluded[1].position = (-0.40 * max_dist, 0, 0.5 + z1)
                    occluded[1].velocity = (dist_per_step, 0, 0)
                    occluder.position = (0, (max(occluded1_depth, occluded2_depth) + occluder_depth) * 0.55, 0.5)
                    occluder.velocity = (0, 0, 0)
                else:
                    assert FLAGS.occluder_moves
                    occluded[0].position = (0, 0, 0.5 + z0)
                    occluded[0].velocity = (0, 0, 0)
                    occluded[1].position = (0, 0, 0.5 + z1)
                    occluded[1].velocity = (0, 0, 0)
                    occluder.position = (-0.4 * max_dist,  (max(occluded1_depth, occluded2_depth) + occluder_depth) * 0.55, 0.5)
                    occluder.velocity = (dist_per_step, 0, 0)

            elif (FLAGS.occluded_moves and FLAGS.occluder_moves):
                # dist_per_step = 0.5 * (occluded_width + occluder_width) / (FLAGS.frame_end - FLAGS.frame_start)
                dist_per_step = 0.8 * max_dist / (FLAGS.frame_end - FLAGS.frame_start) # for comparability to other setting, use width_diff
                print(f"Distance travelled per step: {dist_per_step}")
                start_ = -0.4 * max_dist
                occluder.position = (start_, (max(occluded1_depth, occluded2_depth) + occluder_depth) * 0.55, 0.5)
                occluded[0].position = (start_, 0, 0.5 + z0)
                occluded[1].position = (start_, 0, 0.5 + z1)
                occluder.velocity = (dist_per_step, 0, 0)
                occluded[0].velocity = (dist_per_step, 0, 0)
                occluded[1].velocity = (dist_per_step, 0, 0)
            else:
                # no movement, just place all centrally
                occluder.position = (0,  (max(occluded1_depth, occluded2_depth) + occluder_depth) * 0.55, 0.5)
                occluded[0].position = (0, 0, 0.5 + z0)
                occluded[0].static = True
                occluded[0].velocity = (0, 0, 0)
                occluded[1].position = (0, 0, 0.5 + z1)
                occluded[1].static = True
                occluded[1].velocity = (0, 0, 0)

        else:
            raise NotImplementedError()


    # import pdb
    # pdb.set_trace()

    quaternions = {}
    velocities = {}
    start_positions = {}
    objs = [occluder] + occluded
    for i, obj in enumerate(objs):
        # if i == 0:
        #     quaternions[i] = obj.quaternion
        # else:
        #     quaternions[i] = (1,0,0,0) #obj.quaternion
        velocities[i] = obj.velocity * 10 # for debugging
        start_positions[i] = obj.position
        logging.info("    Added %s at %s", obj.asset_id, obj.position)
        logging.info("    Position: %s, Velocity: %s", obj.position, obj.velocity)

      # if i == 0:
      #     pass
      #     # occluder
      # elif i == 1:
      #     occluder_width = (objs[0].bounds[1][0] - objs[0].bounds[0][0])
      #     obj.velocity = (0.1 * occluder_width, 0, 0)
      #     obj_1_start_pos = (- occluder_width * 5 , #/ 2,
      #                     (objs[0].bounds[1][1] - objs[0].bounds[0][1] + (obj.bounds[1][1] - obj.bounds[0][1])) / 2 * 1.02 ,#0.4,
      #                     0.5)
      #     obj.position =  obj_1_start_pos # no gravity -> can levitate
      #                     #(obj.bounds[1][2] - obj.bounds[0][2]) / 2 * 1.1)
      # else:
      #   obj.velocity = (rng.uniform(*VELOCITY_RANGE) -
      #                     [obj.position[0], obj.position[1], 0])
      # # import pdb
      # # pdb.set_trace()
      # # obj.mass = 0
      # obj.friction = 0
      # logging.info("    Added %s at %s", obj.asset_id, obj.position)
      # logging.info("    Position: %s, Velocity: %s", obj.position, obj.velocity)





    if FLAGS.save_state:
      logging.info("Saving the simulator state to '%s' prior to the simulation.",
                   output_dir / "scene.bullet")
      simulator.save_state(output_dir / "scene.bullet")

    # Run dynamic objects simulation
    logging.info("Running the simulation ...")
    # animation, collisions = simulator.run(frame_start=0,
    #                                       frame_end=10)#scene.frame_end+1)
    # Iterate over each frame
    # occluder_vel = 0
    # obj1_x_vel = 0.1 * occluder_width
    # obj1_y_vel = 0
    # import pdb
    # pdb.set_trace()

    n_frames = (FLAGS.frame_end - FLAGS.frame_start)
    for frame in range(FLAGS.frame_start, FLAGS.frame_end+1):
        # Manually set the locations of each object
        print(f"Len objs: {len(objs)}")
        for i, obj in enumerate(objs):
            obj.position = (start_positions[i][0] + velocities[i][0] * frame/n_frames,
                            start_positions[i][1] + velocities[i][1] * frame/n_frames,
                            start_positions[i][2] + velocities[i][2] * frame/n_frames)
            # obj.quaternion = quaternions[i]
            ## objs[0].position = (0 + occluder_vel * frame/n_frames, 0, 0.5) #(frame * 0.1, 0, 0)  # Example: move cube1 along the x-axis
            ## objs[1].position = (obj_1_start_pos[0] + obj1_x_vel * frame/n_frames,
            ##                     obj_1_start_pos[1] + obj1_y_vel * frame/n_frames,
            ##                     obj_1_start_pos[2])
            print(f"Frame: {frame}, Obj {i} position: {obj.position}, quaternion: {obj.quaternion}")
        # Update the scene for the current frame
        scene.frame_current = frame

        # This stores the initial position etc before running the simulation
        animation, collisions = simulator.run(frame_start=frame,
                                              frame_end=frame+1)  # scene.frame_end+1)
        # Todo: Use the following two commands for debugging; break into pdb, modify camera position, etc, and
        # re-render
        data_stack = renderer.render(frames=[frame], return_layers=("rgba",))
        # Save to image files
        kb.write_image_dict(data_stack, output_dir)

        print(f"Rendered frame {frame}")

    # import pdb
    # pdb.set_trace()

    # --- Rendering
    if FLAGS.save_state:
      logging.info("Saving the renderer state to '%s' ",
                   output_dir / "scene.blend")
      renderer.save_state(output_dir / "scene.blend")


    logging.info("Rendering the scene ...")
    data_stack = renderer.render()

    # --- Postprocessing
    kb.compute_visibility(data_stack["segmentation"], scene.assets)
    visible_foreground_assets = [asset for asset in scene.foreground_assets
                                 if np.max(asset.metadata["visibility"]) > 0]
    visible_foreground_assets = sorted(  # sort assets by their visibility
        visible_foreground_assets,
        key=lambda asset: np.sum(asset.metadata["visibility"]),
        reverse=True)

    data_stack["segmentation"] = kb.adjust_segmentation_idxs(
        data_stack["segmentation"],
        scene.assets,
        visible_foreground_assets)
    scene.metadata["num_instances"] = len(visible_foreground_assets)

    # Save to image files
    kb.write_image_dict(data_stack, output_dir)
    kb.post_processing.compute_bboxes(data_stack["segmentation"],
                                      visible_foreground_assets)

    # --- Metadata
    logging.info("Collecting and storing metadata for each object.")
    kb.write_json(filename=output_dir / "metadata.json", data={
        "flags": vars(FLAGS),
        "metadata": kb.get_scene_metadata(scene),
        "camera": kb.get_camera_info(scene.camera),
        "instances": kb.get_instance_info(scene, visible_foreground_assets),
    })
    kb.write_json(filename=output_dir / "events.json", data={
        "collisions":  kb.process_collisions(
            collisions, scene, assets_subset=visible_foreground_assets),
    })

    kb.done()


if __name__ == '__main__':
    main()