
import numpy as np
import pygsound as ps
import pandas as pd
from datetime import datetime
import os
from typing import List, Tuple, Optional
from itertools import product

def process_position_pair(args):
    mesh_path, src_pos, lis_pos, params, timestamp = args

    ctx = ps.Context()
    ctx.diffuse_count  = params['diffuse_count']
    ctx.specular_count = params['specular_count']
    ctx.channel_type   = ps.ChannelLayoutType.stereo

    mesh = ps.loadobj(mesh_path)
    scene = ps.Scene()
    scene.setMesh(mesh)

    src = ps.Source(src_pos)
    src.radius = params['source_radius']
    src.power  = params['source_power']

    lis = ps.Listener(lis_pos)
    lis.radius = params['listener_radius']

    path_data = scene.getPathData(
        [src], [lis], ctx,
        energy_percentage=params['energy_percentage'],
        max_rays=params['max_rays']
    )["path_data"][0]

    print(f"Processed: {src_pos} -> {lis_pos} ({path_data['num_paths']} paths)")

    data_dict = {
        'source_x':           [src_pos[0]] * path_data['num_paths'],
        'source_y':           [src_pos[1]] * path_data['num_paths'],
        'source_z':           [src_pos[2]] * path_data['num_paths'],
        'listener_x':         [lis_pos[0]] * path_data['num_paths'],
        'listener_y':         [lis_pos[1]] * path_data['num_paths'],
        'listener_z':         [lis_pos[2]] * path_data['num_paths'],
        'source_direction_x': path_data['source_directions'][:, 0],
        'source_direction_y': path_data['source_directions'][:, 1],
        'source_direction_z': path_data['source_directions'][:, 2],
        'listener_direction_x': path_data['listener_directions'][:, 0],
        'listener_direction_y': path_data['listener_directions'][:, 1],
        'listener_direction_z': path_data['listener_directions'][:, 2],
        'distance':           path_data['distances'],
        'relative_speed':     path_data['relative_speeds'],
        'speed_of_sound':     path_data['speeds_of_sound'],
    }

    intensities_df = pd.DataFrame(
        path_data['intensities'],
        columns=[f'intensity_band_{i}' for i in range(path_data['num_bands'])]
    )

    df = pd.DataFrame(data_dict)
    df = pd.concat([df, intensities_df], axis=1)
    df['param_num_bands']  = path_data['num_bands']
    df['param_timestamp']  = timestamp

    return df

class RayDataPipeline:
    def __init__(self,
                 diffuse_count=5000,
                 specular_count=1000,
                 source_radius=0.01,
                 source_power=1.0,
                 listener_radius=0.01,
                 energy_percentage=95.0,
                 max_rays=0):
        self.params = {
            'diffuse_count':    diffuse_count,
            'specular_count':   specular_count,
            'source_radius':    source_radius,
            'source_power':     source_power,
            'listener_radius':  listener_radius,
            'energy_percentage': energy_percentage,
            'max_rays':         max_rays
        }

    def process_coordinates(self, mesh_path, source_positions, listener_positions, output_path):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(output_path, exist_ok=True)

        work_items = [
            (mesh_path, src_pos, lis_pos, self.params.copy(), timestamp)
            for src_pos, lis_pos in product(source_positions, listener_positions)
        ]

        dfs = [process_position_pair(item) for item in work_items]
        final_df = pd.concat(dfs, ignore_index=True)

        output_filename = os.path.join(
            output_path,
            f"{timestamp}_{len(source_positions)}x{len(listener_positions)}_{len(final_df)}paths.parquet"
        )

        final_df.to_parquet(output_filename, index=False)
        print(f"\nSaved: {output_filename}")
        print(f"Total paths: {len(final_df)}")

        return output_filename