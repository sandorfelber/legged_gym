# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import numpy as np
from numpy.random import choice
from scipy import interpolate

from isaacgym import terrain_utils
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg

class Terrain:
    def __init__(self, cfg: LeggedRobotCfg.terrain, num_robots) -> None:

        self.cfg = cfg
        self.num_robots = num_robots
        self.type = cfg.mesh_type
        if self.type in ["none", 'plane']:
            return
        self.env_length = cfg.terrain_length
        self.env_width = cfg.terrain_width
        self.proportions = [np.sum(cfg.terrain_proportions[:i+1]) for i in range(len(cfg.terrain_proportions))]

        self.cfg.num_sub_terrains = cfg.num_rows * cfg.num_cols
        self.env_origins = np.zeros((cfg.num_rows, cfg.num_cols, 3))

        self.width_per_env_pixels = int(self.env_width / cfg.horizontal_scale)
        self.length_per_env_pixels = int(self.env_length / cfg.horizontal_scale)

        self.border = int(cfg.border_size/self.cfg.horizontal_scale)
        self.tot_cols = int(cfg.num_cols * self.width_per_env_pixels) + 2 * self.border
        self.tot_rows = int(cfg.num_rows * self.length_per_env_pixels) + 2 * self.border

        self.height_field_raw = np.zeros((self.tot_rows , self.tot_cols), dtype=np.int16)
        if cfg.curriculum:
            self.curiculum()
        elif cfg.selected:
            self.selected_terrain()
        else:    
            self.randomized_terrain()   
        
        self.heightsamples = self.height_field_raw
        if self.type=="trimesh":
            self.vertices, self.triangles = terrain_utils.convert_heightfield_to_trimesh(   self.height_field_raw,
                                                                                            self.cfg.horizontal_scale,
                                                                                            self.cfg.vertical_scale,
                                                                                            self.cfg.slope_treshold)
    
    def randomized_terrain(self):
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            choice = np.random.uniform(0, 1)
            difficulty = np.random.choice([0.5, 0.75, 0.9])
            terrain = self.make_terrain(choice, difficulty)
            self.add_terrain_to_map(terrain, i, j)
        
    def curiculum(self):
        for j in range(self.cfg.num_cols):
            for i in range(self.cfg.num_rows):
                difficulty = i / self.cfg.num_rows
                choice = j / self.cfg.num_cols + 0.001

                terrain = self.make_terrain(choice, difficulty)
                self.add_terrain_to_map(terrain, i, j)

    def selected_terrain(self):
        terrain_type = self.cfg.terrain_kwargs.pop('type')
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            terrain = terrain_utils.SubTerrain("terrain",
                              width=self.width_per_env_pixels,
                              length=self.width_per_env_pixels,
                              vertical_scale=self.vertical_scale,
                              horizontal_scale=self.horizontal_scale)

            eval(terrain_type)(terrain, **self.cfg.terrain_kwargs.terrain_kwargs)
            self.add_terrain_to_map(terrain, i, j)
    
    def make_terrain(self, choice, difficulty):
        terrain = terrain_utils.SubTerrain(   "terrain",
                                width=self.width_per_env_pixels,
                                length=self.width_per_env_pixels,
                                vertical_scale=self.cfg.vertical_scale,
                                horizontal_scale=self.cfg.horizontal_scale)
        slope = difficulty * self.cfg.slope_treshold
        step_height = self.cfg.steps_height_scale * (0.05 + 0.2 * difficulty)
        discrete_obstacles_height = self.cfg.steps_height_scale  * (0.05 + difficulty * 0.2)
        stepping_stones_size = 1.5 * (1.05 - difficulty)
        stone_distance = self.cfg.horizontal_difficulty_scale * (0.05 + 0.35 * difficulty)
        gap_size = self.cfg.horizontal_difficulty_scale * difficulty
        pit_depth = self.cfg.steps_height_scale * 0.5 * difficulty
        trench_width = self.cfg.horizontal_difficulty_scale  * 0.55 * (1 - (0.13 * difficulty))
        if choice < self.proportions[0]:
            if choice < self.proportions[0]/ 2:
                slope *= -1
            terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
        elif choice < self.proportions[1]:
            terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
            terrain_utils.random_uniform_terrain(terrain, min_height=-0.05, max_height=0.05, step=0.005, downsampled_scale=0.2)
        elif choice < self.proportions[3]:
            if choice<self.proportions[2]:
                step_height *= -1
            terrain_utils.pyramid_stairs_terrain(terrain, step_width=0.31, step_height=step_height, platform_size=3.)
        elif choice < self.proportions[4]:
            num_rectangles = 20
            rectangle_min_size = 1.
            rectangle_max_size = 2.
            terrain_utils.discrete_obstacles_terrain(terrain, discrete_obstacles_height, rectangle_min_size, rectangle_max_size, num_rectangles, platform_size=3.)
        elif choice < self.proportions[5]:
            terrain_utils.stepping_stones_terrain(terrain, stone_size=stepping_stones_size, stone_distance=stone_distance, max_height=0., platform_size=4.)
        elif choice < self.proportions[6]:
            #terrain_utils.wave_terrain(terrain)
            gap_terrain(terrain, gap_size=gap_size, platform_size=3.)
        elif choice < self.proportions[7]:
            radial_trench_terrain(terrain, wall_height=0.8, trench_width=trench_width, num_trenches=8, inner_untouched_diameter_percent=0.16, outer_untouched_diameter_percent=0.46)
        else:
            radial_trench_terrain_with_gaps(terrain, gap_depth=1., trench_width=trench_width, num_trenches=8, inner_untouched_diameter_percent=0.16, outer_untouched_diameter_percent=0.46)
            #pit_terrain(terrain, depth=pit_depth, platform_size=4.)
        
        return terrain

    def add_terrain_to_map(self, terrain, row, col):
        i = row
        j = col
        # map coordinate system
        start_x = self.border + i * self.length_per_env_pixels
        end_x = self.border + (i + 1) * self.length_per_env_pixels
        start_y = self.border + j * self.width_per_env_pixels
        end_y = self.border + (j + 1) * self.width_per_env_pixels
        self.height_field_raw[start_x: end_x, start_y:end_y] = terrain.height_field_raw

        env_origin_x = (i + 0.5) * self.env_length
        env_origin_y = (j + 0.5) * self.env_width
        x1 = int((self.env_length/2. - 1) / terrain.horizontal_scale)
        x2 = int((self.env_length/2. + 1) / terrain.horizontal_scale)
        y1 = int((self.env_width/2. - 1) / terrain.horizontal_scale)
        y2 = int((self.env_width/2. + 1) / terrain.horizontal_scale)
        env_origin_z = np.max(terrain.height_field_raw[x1:x2, y1:y2])*terrain.vertical_scale
        self.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]

def gap_terrain(terrain, gap_size, platform_size=1.):
    gap_size = int(gap_size / terrain.horizontal_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)

    center_x = terrain.length // 2
    center_y = terrain.width // 2
    x1 = (terrain.length - platform_size) // 2
    x2 = x1 + gap_size
    y1 = (terrain.width - platform_size) // 2
    y2 = y1 + gap_size
   
    terrain.height_field_raw[center_x-x2 : center_x + x2, center_y-y2 : center_y + y2] = -1000
    terrain.height_field_raw[center_x-x1 : center_x + x1, center_y-y1 : center_y + y1] = 0

def pit_terrain(terrain, depth, platform_size=1.):
    depth = int(depth / terrain.vertical_scale)
    platform_size = int(platform_size / terrain.horizontal_scale / 2)
    x1 = terrain.length // 2 - platform_size
    x2 = terrain.length // 2 + platform_size
    y1 = terrain.width // 2 - platform_size
    y2 = terrain.width // 2 + platform_size
    terrain.height_field_raw[x1:x2, y1:y2] = -depth

def radial_trench_terrain(terrain, wall_height, trench_width, num_trenches, inner_untouched_diameter_percent, outer_untouched_diameter_percent):
    """
    Modify the terrain height field to add radial walls, leaving the trenches at the original level.

    Parameters:
    terrain: The terrain object.
    wall_height: Height of the walls.
    trench_width: Width of each trench.
    num_trenches: Number of radial trenches.
    untouched_radius: Radius of the central area to be left at trench level.
    """
    # Adjusting wall dimensions based on terrain scale
    wall_height = int(wall_height / terrain.vertical_scale)
    trench_width = int(trench_width / terrain.horizontal_scale)

    # Determining the center of the terrain
    center_x = terrain.length // 2
    center_y = terrain.width // 2

    # Initialize a matrix to track trench points
    trench_points = np.zeros_like(terrain.height_field_raw, dtype=bool)
    # Marking trenches
    for i in range(num_trenches):
        angle = (i / num_trenches) * 2 * np.pi  # angle in radians
        
        for x1 in range(terrain.length):
            for y1 in range(terrain.width):
                # Calculating distance and angle to the current point
                dx = x1 - center_x
                dy = y1 - center_y
                distance_to_center = np.sqrt(dx**2 + dy**2)
                if distance_to_center <= terrain.length * inner_untouched_diameter_percent: #) - distance_to_center/1.05:
                    trench_points[x1, y1] = True
                elif distance_to_center > terrain.length * outer_untouched_diameter_percent:
                    trench_points[x1, y1] = True
                # Check if the point is within the trench
                else:
                    angle_to_point = np.arctan2(dy, dx)
                    angle_diff = min(abs(angle - angle_to_point), 
                                     abs(angle - angle_to_point - 2 * np.pi), 
                                     abs(angle - angle_to_point + 2 * np.pi))
                    if angle_diff < trench_width / distance_to_center:
                        trench_points[x1, y1] = True  # Mark as trench

    # Raising non-trench areas
    for x1 in range(terrain.length):
        for y1 in range(terrain.width):
            if not trench_points[x1, y1]:
                terrain.height_field_raw[x1, y1] += wall_height

def radial_trench_terrain_with_gaps(terrain, gap_depth, trench_width, num_trenches, inner_untouched_diameter_percent, outer_untouched_diameter_percent):
    """
    Modify the terrain height field to add radial gaps, leaving the trenches at the original level and creating
    gaps or cliffs between them.
    Parameters:
    terrain: The terrain object.
    gap_depth: Depth of the gaps/cliffs.
    trench_width: Width of each trench.
    num_trenches: Number of radial trenches.
    inner_untouched_diameter_percent: Inner diameter percentage that remains untouched.
    outer_untouched_diameter_percent: Outer diameter percentage that remains untouched.
    """
    # Adjusting gap dimensions based on terrain scale
    trench_width = int(trench_width / terrain.horizontal_scale)

    # Determining the center of the terrain
    center_x = terrain.length // 2
    center_y = terrain.width // 2

    # Initialize a matrix to track trench points
    trench_points = np.zeros_like(terrain.height_field_raw, dtype=bool)
    # Marking trenches
    for i in range(num_trenches):
        angle = (i / num_trenches) * 2 * np.pi  # angle in radians

        for x1 in range(terrain.length):
            for y1 in range(terrain.width):
                # Calculating distance and angle to the current point
                dx = x1 - center_x
                dy = y1 - center_y
                distance_to_center = np.sqrt(dx**2 + dy**2)
                if distance_to_center <= terrain.length * inner_untouched_diameter_percent: #) - distance_to_center/1.05:
                    trench_points[x1, y1] = True
                elif distance_to_center > terrain.length * outer_untouched_diameter_percent:
                    trench_points[x1, y1] = True
                # Check if the point is within the trench
                else:
                    angle_to_point = np.arctan2(dy, dx)
                    angle_diff = min(abs(angle - angle_to_point), 
                                     abs(angle - angle_to_point - 2 * np.pi), 
                                     abs(angle - angle_to_point + 2 * np.pi))
                    if angle_diff < trench_width / distance_to_center:
                        trench_points[x1, y1] = True  # Mark as trench

    # Raising non-trench areas
    for x1 in range(terrain.length):
        for y1 in range(terrain.width):
            if not trench_points[x1, y1]:
                terrain.height_field_raw[x1, y1] -= gap_depth*100.0

# def radial_trench_terrain(terrain, trench_depth, trench_width, num_trenches):
#     """
#     Modify the terrain height field to add radial trenches.

#     :param terrain: The terrain object.
#     :param trench_depth: Depth of the trenches.
#     :param trench_width: Width of each trench.
#     :param num_trenches: Number of radial trenches.
#     """
#     trench_depth = int(trench_depth / terrain.vertical_scale)
#     trench_width = int(trench_width / terrain.horizontal_scale)

#     center_x = terrain.length // 2
#     center_y = terrain.width // 2

#     for i in range(num_trenches):
#         angle = (i / num_trenches) * 2 * np.pi  # angle in radians

#         for x in range(terrain.length):
#             for y in range(terrain.width):
#                 dx = x - center_x
#                 dy = y - center_y
#                 distance_to_center = np.sqrt(dx**2 + dy**2)
#                 angle_to_point = np.arctan2(dy, dx)

#                 angle_diff = min(abs(angle - angle_to_point), abs(angle - angle_to_point - 2*np.pi), abs(angle - angle_to_point + 2*np.pi))
#                 if distance_to_center > 0:
#                     if angle_diff < trench_width / distance_to_center:
#                         terrain.height_field_raw[x, y] += trench_depth

# def radial_trench_terrain(terrain, trench_depth, trench_width, num_trenches, untouched_radius):
#     """
#     Modify the terrain height field to add radial trenches, leaving a small radius in the middle untouched,
#     and ensuring each point is only modified once to maintain two distinct height levels.

#     Parameters:
#     terrain: The terrain object.
#     trench_depth: Depth of the trenches.
#     trench_width: Width of each trench.
#     num_trenches: Number of radial trenches.
#     untouched_radius: Radius of the central area to be left untouched.
#     """
#     # Adjusting trench dimensions based on terrain scale
#     trench_depth = int(trench_depth / terrain.vertical_scale)
#     trench_width = int(trench_width / terrain.horizontal_scale)

#     # Determining the center of the terrain
#     center_x = terrain.length // 2
#     center_y = terrain.width // 2

#     # Initialize a matrix to track modified points
#     modified_points = np.zeros_like(terrain.height_field_raw, dtype=bool)

#     # Creating radial trenches
#     for i in range(num_trenches):
#         angle = (i / num_trenches) * 2 * np.pi  # angle in radians

#         for x in range(terrain.length):
#             for y in range(terrain.width):
#                 # Skip already modified points
#                 if modified_points[x, y]:
#                     continue

#                 # Calculating distance and angle to the current point
#                 dx = x - center_x
#                 dy = y - center_y
#                 distance_to_center = np.sqrt(dx**2 + dy**2)

#                 # Check if the point is outside the untouched central radius
#                 if distance_to_center > untouched_radius:
#                     angle_to_point = np.arctan2(dy, dx)

#                     # Determining if the point is within the trench
#                     angle_diff = min(abs(angle - angle_to_point), 
#                                      abs(angle - angle_to_point - 2 * np.pi), 
#                                      abs(angle - angle_to_point + 2 * np.pi))
#                     if angle_diff < trench_width / distance_to_center:
#                         terrain.height_field_raw[x, y] -= trench_depth
#                         modified_points[x, y] = True  # Mark as modified
                
# def trench_terrain(terrain, trench_depth, trench_width, spacing, num_trenches):
#     """
#     Modify the terrain height field to add trenches.

#     :param terrain: The terrain object.
#     :param trench_depth: Depth of the trenches.
#     :param trench_width: Width of each trench.
#     :param spacing: Spacing between trenches.
#     :param num_trenches: Number of trenches to create.
#     """
#     trench_depth = int(trench_depth / terrain.vertical_scale)
#     trench_width = int(trench_width / terrain.horizontal_scale)
#     spacing = int(spacing / terrain.horizontal_scale)

#     for i in range(num_trenches):
#         start_x = i * (trench_width + spacing)
#         end_x = start_x + trench_width

#         # Ensure the trench does not exceed terrain boundaries
#         if end_x > terrain.length:
#             break

#         terrain.height_field_raw[start_x:end_x, :] += trench_depth
    
