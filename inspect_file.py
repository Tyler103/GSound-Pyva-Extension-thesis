import numpy as np
import pygsound as ps
import pandas as pd

# Setup — same as the example
ctx = ps.Context()
ctx.diffuse_count = 5000
ctx.specular_count = 1000
ctx.channel_type = ps.ChannelLayoutType.stereo

mesh = ps.loadobj("cube.obj")
scene = ps.Scene()
scene.setMesh(mesh)

src = ps.Source([1, 1, 0.5])
src.radius = 0.01
src.power = 1.0

lis = ps.Listener([5, 3, 0.5])
lis.radius = 0.01

# Get raw path data
path_data = scene.getPathData([src], [lis], ctx)["path_data"][0]

# Inspect real arrays
print("num_paths:", path_data['num_paths'])
print("num_bands:", path_data['num_bands'])
print("total_energy:", path_data['total_energy'])

print("\nsource_directions shape:", np.array(path_data['source_directions']).shape)
print("listener_directions shape:", np.array(path_data['listener_directions']).shape)
print("distances shape:", np.array(path_data['distances']).shape)
print("intensities shape:", np.array(path_data['intensities']).shape)

# Build DataFrame exactly like the example
data_dict = {
    'source_direction_x': path_data['source_directions'][:, 0],
    'source_direction_y': path_data['source_directions'][:, 1],
    'source_direction_z': path_data['source_directions'][:, 2],
    'listener_direction_x': path_data['listener_directions'][:, 0],
    'listener_direction_y': path_data['listener_directions'][:, 1],
    'listener_direction_z': path_data['listener_directions'][:, 2],
    'distance':            path_data['distances'],
    'speed_of_sound':      path_data['speeds_of_sound'],
}

# Add intensity per band
intensities_df = pd.DataFrame(
    path_data['intensities'],
    columns=[f'intensity_band_{i}' for i in range(path_data['num_bands'])]
)

df = pd.DataFrame(data_dict)
df = pd.concat([df, intensities_df], axis=1)

# Compute arrival time
df['time'] = df['distance'] / df['speed_of_sound']

print("\nDataFrame shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nStats:")
print(df.describe())

# Find floor-hitting rays
# Rays traveling downward — source_direction_z negative
floor_rays = df[df['source_direction_z'] < -0.5]
print(f"\nFloor hitting rays: {len(floor_rays)} / {len(df)}")

total_energy_all   = df['intensity_band_0'].sum()
total_energy_floor = floor_rays['intensity_band_0'].sum()

print(f"Total rays: {len(df)}")
print(f"Floor rays: {len(floor_rays)}")
print(f"Floor rays energy: {total_energy_floor/total_energy_all*100:.1f}% of total")
print(f"Floor rays avg energy: {floor_rays['intensity_band_0'].mean():.6f}")
print(f"All rays avg energy:   {df['intensity_band_0'].mean():.6f}")