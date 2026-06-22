# /app/ray_generator/examples/full_pipeline.py

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import fftconvolve, resample
import sys
import os
import pygsound as ps

sys.path.insert(0, '/app')

from ray_pipeline import RayDataPipeline
from py_auralizer import Ambisonic_IR_Generator, create_dataset, _cart2sph

import pyva.properties.materialClasses as matC
import pyva.properties.structuralPropertyClasses as sProp

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D


# ─────────────────────────────────────
# CONFIG
# ─────────────────────────────────────
floor_width  = 10.0
floor_depth  = 10.0
floor_height = 3.0
sample_rate  = 48000

wall_x       = floor_width          # x-coordinate of the shared wall
room2_width  = 10.0                 # Room 2 extends from x=10 to x=20
wall_delay   = 0.2 / 3500 


import pyva.systems.acoustic3Dsystems as ac3
import pyva.systems.structure2Dsystems as st2
import pyva.coupling.junctions as jun

def get_TL_diffuse(frequencies, thickness, material='drywall'):
    """
    Diffuse field transmission loss — integrates angular transmission
    coefficient across all angles of incidence (0 to 90 degrees).
    More physically accurate than normal incidence for reverberant rooms.
    Captures coincidence effect at critical frequency.
    """

    # Convert frequencies to angular frequency needed for physics
    omega  = 2 * np.pi * np.array(frequencies, dtype=float)

    # Instantiates a pyva Fluid object with default air properties
    air    = matC.Fluid()

    # Material property blocks

    # E: Young's modulus — stiffness, resistance to deformation
    # rh0: Density
    # nu: Poisson's ratio — how much it expands laterally when compressed
    # eta: Loss factor — internal damping, fraction of energy dissipated per cycle (dimensionless)

    if material == 'concrete':
        mat       = matC.IsoMat(E=3.8e10, rho0=2300, nu=0.2, eta=0.02)
    elif material == 'drywall':
        mat       = matC.IsoMat(E=2.5e9, rho0=800, nu=0.3, eta=0.01)
    elif material == 'timber':
        mat       = matC.IsoMat(E=1.1e10, rho0=500, nu=0.3, eta=0.03)
    elif material == 'aluminium':
        mat = matC.IsoMat()

    else:
        raise ValueError(f"Unknown material: {material}. Choose 'concrete', 'drywall', or 'timber'.")
    
    # Build the plate object
    plate_prop  = sProp.PlateProp(thickness, mat)

    # theta = angle of incidence
    # tau = transmission coeefficiet
    tau_diffuse = np.zeros(len(omega))
    tau_diffuse = plate_prop.transmission_coefficient_diffuse(omega, fluid1=air)
    #Convert tau to decibels
    TL_db = -10 * np.log10(tau_diffuse + 1e-10)

    # TL in dB, raw τ values 
    return TL_db, tau_diffuse

frequencies = np.array([125, 250, 500, 1000, 2000, 4000, 8000, 16000], dtype=float)
# drywall: 0.012
# concrete: 0.2
# timber: 0.05
# alum: 0.006

material_chosen = 'aluminium'
thickniss = 0.006
TL_db, tau = get_TL_diffuse(frequencies, thickness=thickniss, material= material_chosen)


# prints the transmission loss over 8 frequency bands
print(f"Diffuse field TL ({material_chosen}):")
for f, tl, t in zip(frequencies, TL_db, tau):
    print(f"  {f:6.0f} Hz: TL={tl:.1f}dB  tau={t:.6f}  ({t*100:.4f}% survives)")

# ─────────────────────────────────────
# STEP 2: Generate Parquet via pipeline
# ─────────────────────────────────────
print("\nRunning RayDataPipeline...")

listener_grid = [
    (9.5, 2.0, 0.5),
    (9.5, 5.0, 0.5),
    (9.5, 8.0, 0.5),
    (9.5, 2.0, 1.5),
    (9.5, 5.0, 1.5),
    (9.5, 8.0, 1.5),
    (9.5, 2.0, 2.5),
    (9.5, 5.0, 2.5),
    (9.5, 8.0, 2.5),
]


# initialize ray creation pipeline
pipeline = RayDataPipeline(
    diffuse_count=5000,
    specular_count=1000,
    energy_percentage=95.0,
)

# simulates ray generation and saves to parquet
parquet_path = pipeline.process_coordinates(
    mesh_path='/app/ray_generator/examples/cube.obj',
    source_positions=[(1.0, 1.0, 0.5)],
    listener_positions=listener_grid,
    output_path='/app/ray_generator/examples/output'
)

df = pd.read_parquet(parquet_path)
print(f"Number of intensity bands: {df['param_num_bands'].iloc[0]}")
print(f"Band columns: {[c for c in df.columns if 'intensity_band' in c]}")
print(f"Frequencies array length: {len(frequencies)}")

print(f"Parquet saved: {parquet_path}")

# ─────────────────────────────────────
# STEP 3: Load dataset for auralizer
# ─────────────────────────────────────
print("\nLoading dataset...")

df_full = pd.read_parquet(parquet_path)

# filter to single central listener for Room 1 IR
df_single = df_full[
    (df_full['listener_x'] == 9.5) &
    (df_full['listener_y'] == 5.0) &
    (df_full['listener_z'] == 1.5)
].copy()

single_listener_path = parquet_path.replace('.parquet', '_single.parquet')
df_single.to_parquet(single_listener_path)

data = create_dataset(
    ray=single_listener_path,
    room='/app/ray_generator/examples/cube.obj'
)

# list of samples, one per source-listener pair
demo = data[0]

print(f"Source:      {demo['tx']}")
print(f"Listener:    {demo['rx']}")
print(f"Intensities: {demo['Intensities'].shape}")
print(f"Directions:  {demo['doa'].shape}")
print(f"Delays:      {demo['delay'].shape}")
print(f"Room volume: {demo['V']:.2f} m3")

# ─────────────────────────────────────
# STEP 4: Generate Room 1 Ambisonic IR
# ─────────────────────────────────────
print("\nGenerating Room 1 IR...")

# Instantiates the Ambisonic IR generator
auralizer = Ambisonic_IR_Generator(
    fs=sample_rate,
    order=1,
    imp_res_time=3.0
)

# output is 4 Ambisonic channels × IR length in samples
# 1. places each ray in time
# 2. encodes each ray spatially
# 3. Scales by per-band intensity

sir_room1 = auralizer.forward_ambsonics(demo)
print(f"Room 1 IR shape: {sir_room1.shape}")

# ─────────────────────────────────────
# STEP 5: Find wall-hitting rays
# ─────────────────────────────────────
print("\nFinding wall-hitting rays...")

df = df_full.copy()
df['time'] = df['distance'] / df['speed_of_sound']

band_cols = [f'intensity_band_{b}' for b in range(8)]
df['total_energy'] = df[band_cols].sum(axis=1)

# Rays headed toward the shared wall at x = wall_x (positive x direction)
wall_rays = df[df['listener_direction_x'] > 0.5].copy()

print(f"Wall rays: {len(wall_rays)} / {len(df)}")
print(f"Wall energy: {wall_rays['intensity_band_0'].sum()/df['intensity_band_0'].sum()*100:.1f}%")

# ─────────────────────────────────────
# STEP 6: Compute impact points on the shared wall per listener group
# ─────────────────────────────────────
print("\nComputing wall impact points...")

wall_rays['dist_norm']   = wall_rays['distance'] / wall_rays['distance'].max()
wall_rays['energy_norm'] = wall_rays['total_energy'] / wall_rays['total_energy'].max()
wall_rays['score']       = wall_rays['energy_norm'] / (wall_rays['dist_norm'] + 1e-10)

rays_per_listener = 6
all_impact_y      = []
all_impact_z      = []
all_impact_energy = []

for (lx, ly, lz), group in wall_rays.groupby(['listener_x', 'listener_y', 'listener_z']):
    top = group.nlargest(rays_per_listener, 'score').reset_index(drop=True)

    # t = how far along the ray direction until it hits x = wall_x
    t  = (wall_x - top['listener_x'].values) / (top['listener_direction_x'].values + 1e-10)
    iy = top['listener_y'].values + top['listener_direction_y'].values * t
    iz = top['listener_z'].values + top['listener_direction_z'].values * t

    iy = np.clip(iy, 0.05, floor_depth  - 0.05)
    iz = np.clip(iz, 0.05, floor_height - 0.05)

    all_impact_y.extend(iy)
    all_impact_z.extend(iz)
    all_impact_energy.extend(top['total_energy'].values)
    print(f"  Listener ({lx}, {ly}, {lz}): {len(top)} impact points on wall")

impact_y      = np.array(all_impact_y)
impact_z      = np.array(all_impact_z)
impact_energy = np.array(all_impact_energy)

print(f"\nTotal virtual sources: {len(impact_y)}")
print(f"Y range: {impact_y.min():.2f} to {impact_y.max():.2f}")
print(f"Z range: {impact_z.min():.2f} to {impact_z.max():.2f}")

# Virtual sources are just on the other side of the wall in Room 2
vsrc_positions = [
    (wall_x + 0.1, float(impact_y[i]), float(impact_z[i]))
    for i in range(len(impact_y))
]

print("Sample wall impact points (virtual sources in Room 2):")
for i in range(5):
    print(f"  ({wall_x + 0.1:.2f}, {impact_y[i]:.2f}, {impact_z[i]:.2f})")

room2_listener_local = (3.0, 5.0, 1.5)

vsrc_positions_local = [
    (0.1, float(impact_y[i]), float(impact_z[i]))
    for i in range(len(impact_y))
]

pipeline_room2 = RayDataPipeline(
    diffuse_count=5000,
    specular_count=1000,
    energy_percentage=95.0,
)

parquet_room2 = pipeline_room2.process_coordinates(
    mesh_path='/app/ray_generator/examples/cube.obj',
    source_positions=vsrc_positions_local,
    listener_positions=[room2_listener_local],
    output_path='/app/ray_generator/examples/output'
)
print(f"Room 2 parquet: {parquet_room2}")

# ─────────────────────────────────────
# STEP 7: Load Room 2 dataset
# ─────────────────────────────────────
data_room2 = create_dataset(ray=parquet_room2, room='/app/ray_generator/examples/cube.obj')


all_intensities = np.concatenate([d['Intensities'] for d in data_room2], axis=1)
all_doa         = np.concatenate([d['doa']         for d in data_room2], axis=1)
all_delays      = np.concatenate([d['delay']       for d in data_room2])

active_sources = sum(1 for d in data_room2 if len(d['delay']) > 0)
all_intensities = all_intensities / active_sources
print(f"Active sources: {active_sources} / {len(vsrc_positions)}")

# Apply pyva TL per band
for b in range(8):
    all_intensities[b] *= tau[b]

demo_room2 = {
    'tx':          np.array([wall_x + 0.1, np.mean(impact_y), np.mean(impact_z)]),
    'rx':          np.array([wall_x + room2_width - 7.0, 5.0, 1.5]),  # real-world Room 2 mic
    'Intensities': all_intensities,
    'doa':         all_doa,
    'delay':       all_delays + wall_delay,
    'V':           float(room2_width * floor_depth * floor_height)
}

# ─────────────────────────────────────
# STEP 8: Build Room 2 Ambisonic IR
# ─────────────────────────────────────
sir_room2 = auralizer.forward_ambsonics(demo_room2)
print(f"Room 2 IR shape: {sir_room2.shape}")

# ─────────────────────────────────────
# STEP 9: Load gunshot
# ─────────────────────────────────────
gunshot, fs = sf.read(
    '/app/ray_generator/examples/210766__acs272__gun-shot-in-anechoic-chamber.wav'
)

if gunshot.ndim > 1:
    gunshot = gunshot.mean(axis=1)
if fs != sample_rate:
    gunshot = resample(gunshot, int(len(gunshot) * sample_rate / fs))
    fs = sample_rate
gunshot = gunshot / np.max(np.abs(gunshot))
print(f"\nLoaded gunshot: {len(gunshot)} samples, {fs}Hz, {len(gunshot)/fs:.2f}s")

# ─────────────────────────────────────
# STEP 10: Convolve each Ambisonic channel
# ─────────────────────────────────────
print("\nConvolving...")

room1_channels = [fftconvolve(gunshot, sir_room1[ch]) for ch in range(4)]
room2_channels = [fftconvolve(gunshot, sir_room2[ch]) for ch in range(4)]

room1_recording = np.array(room1_channels)
room2_recording = np.array(room2_channels)

peak            = np.max(np.abs(room1_recording))
room1_recording = room1_recording / peak
room2_recording = room2_recording / peak

room2_audible = room2_recording / (np.max(np.abs(room2_recording)) + 1e-10)

sf.write('output_room1_W.wav',         room1_recording[0].astype(np.float32), fs)
sf.write('output_room2_W.wav',         room2_recording[0].astype(np.float32), fs)
sf.write('output_room2_audible_W.wav', room2_audible[0].astype(np.float32),   fs)
sf.write('output_room1_ambi.wav',      room1_recording.T.astype(np.float32),  fs)
sf.write('output_room2_ambi.wav',      room2_recording.T.astype(np.float32),  fs)

print("\nSaved:")
print("  output_room1_W.wav          — Room 1 mono (W channel)")
print("  output_room2_W.wav          — Room 2 mono (same scale)")
print("  output_room2_audible_W.wav  — Room 2 mono (normalized for listening)")
print("  output_room1_ambi.wav       — Room 1 4-channel Ambisonics")
print("  output_room2_ambi.wav       — Room 2 4-channel Ambisonics")


# ─────────────────────────────────────

# ─────────────────────────────────────
# STEP 11: Summary
# ─────────────────────────────────────
print("\n─── Summary ─────────────────────────────────────────")
print(f"Total rays:         {len(df):,}")
print(f"Wall rays:          {len(wall_rays):,}")
print(f"Wall energy %:      {wall_rays['total_energy'].sum() / df['total_energy'].sum() * 100:.1f}%")
print(f"Virtual sources:    {len(vsrc_positions)}")
print(f"Room 2 total rays:  {len(all_delays):,}")
print(f"Room 1 IR shape:    {sir_room1.shape}")
print(f"Room 2 IR shape:    {sir_room2.shape}")

atr = np.max(np.abs(room2_recording)) / np.max(np.abs(room1_recording))
il  = -20 * np.log10(atr)
energy_transmission = atr**2 * 100

print(f"Room 2 vs Room 1:   {atr*100:.4f}%  (ATR)")
print(f"Insertion Loss:     {il:.2f} dB")
print(f"Energy transmitted: {energy_transmission:.4f}%")
print(f"{material_chosen} {thickniss*1000:.0f}mm ({thickniss*39.3701:.2f} inches)")
print(f"TL range:           {TL_db[0]:.1f}dB (125Hz) to {TL_db[-1]:.1f}dB (16kHz)")
print(f"Ray selection:      top {rays_per_listener} per listener group ({len(listener_grid)} listeners)")
print("─────────────────────────────────────────────────────")

# ─────────────────────────────────────
# STEP 12: Save figures
# ─────────────────────────────────────
figures_dir = '/app/ray_generator/examples/figures'
os.makedirs(figures_dir, exist_ok=True)

src_x, src_y, src_z = 1.0, 1.0, 0.5

def draw_room(ax, x0, y0, z0, w, d, h, color, alpha=0.1, label=''):
    xx, yy = np.meshgrid([x0, x0+w], [y0, y0+d])
    ax.plot_surface(xx, yy, np.full_like(xx, z0),   alpha=alpha, color=color)
    ax.plot_surface(xx, yy, np.full_like(xx, z0+h), alpha=alpha, color=color)
    for x in [x0, x0+w]:
        yy2, zz = np.meshgrid([y0, y0+d], [z0, z0+h])
        ax.plot_surface(np.full_like(yy2, x), yy2, zz, alpha=alpha, color=color)
    for y in [y0, y0+d]:
        xx2, zz = np.meshgrid([x0, x0+w], [z0, z0+h])
        ax.plot_surface(xx2, np.full_like(xx2, y), zz, alpha=alpha, color=color)
    if label:
        ax.text(x0+w/2, y0+d/2, z0+h/2, label, ha='center', va='center',
                fontsize=9, fontweight='bold', color=color)

# ── Figure 1: 3D building overview ──────────────────────
fig1 = plt.figure(figsize=(10, 8))
ax1  = fig1.add_subplot(111, projection='3d')

draw_room(ax1, 0,          0, 0, floor_width, floor_depth, floor_height, 'steelblue', label='Room 1')
draw_room(ax1, floor_width, 0, 0, room2_width, floor_depth, floor_height, 'coral',    label='Room 2')

# Shared wall plane
yy, zz = np.meshgrid([0, floor_depth], [0, floor_height])
ax1.plot_surface(np.full_like(yy, wall_x), yy, zz, alpha=0.3, color='gray')

ax1.scatter([src_x], [src_y], [src_z], c='red', s=200, marker='*', zorder=5, label='Gunshot')
ax1.scatter([9.5], [5.0], [1.5], c='steelblue', s=100, marker='^', zorder=5, label='Mic Room 1')
ax1.scatter([wall_x + 3.0], [5.0], [1.5], c='coral', s=100, marker='^', zorder=5, label='Mic Room 2')
ax1.scatter(np.full(len(impact_y), wall_x), impact_y, impact_z,
            c='orange', s=50, marker='o', alpha=0.7, label='Wall ray exits')

ax1.set_xlabel('X (m)'); ax1.set_ylabel('Y (m)'); ax1.set_zlabel('Z (m)')
ax1.set_title('3D Building Overview — Side-by-Side Rooms')
ax1.legend(fontsize=7, loc='upper left')
fig1.tight_layout()
fig1.savefig(f'{figures_dir}/01_building_3d_overview.png', dpi=150, bbox_inches='tight')
plt.close(fig1)
print("Saved 01_building_3d_overview.png")

# ── Figure 2: Top-down floor plan ───────────────────────
fig2, ax2 = plt.subplots(figsize=(12, 6))
room1_rect = patches.Rectangle((0, 0), floor_width, floor_depth,
                                linewidth=2, edgecolor='black', facecolor='lightyellow')
room2_rect = patches.Rectangle((floor_width, 0), room2_width, floor_depth,
                                linewidth=2, edgecolor='black', facecolor='lightcyan')
ax2.add_patch(room1_rect)
ax2.add_patch(room2_rect)

# Shared wall line
ax2.axvline(wall_x, color='gray', linewidth=3, linestyle='--', label='Shared wall')

scatter = ax2.scatter(
    np.full(len(impact_y), wall_x),
    impact_y,
    c=impact_energy, cmap='hot', s=40, alpha=0.8, label='Wall ray impacts'
)
plt.colorbar(scatter, ax=ax2, label='Total Energy (all 8 bands)')
ax2.scatter([src_x], [src_y], c='red', s=300, marker='*', zorder=5, label='Source')

# Listener grid positions
listener_xs = [pos[0] for pos in listener_grid]
listener_ys = [pos[1] for pos in listener_grid]
ax2.scatter(listener_xs, listener_ys, c='steelblue', s=80, marker='^', zorder=5, label='Listener grid')

ax2.text(5.0, 9.5, 'Room 1', ha='center', fontsize=12, fontweight='bold', color='steelblue')
ax2.text(15.0, 9.5, 'Room 2', ha='center', fontsize=12, fontweight='bold', color='coral')
ax2.set_xlabel('X (m)'); ax2.set_ylabel('Y (m)')
ax2.set_title('Top-down: Wall Ray Distribution')
ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3); ax2.set_aspect('equal')
ax2.set_xlim(-0.5, floor_width + room2_width + 0.5)
ax2.set_ylim(-0.5, floor_depth + 0.5)
fig2.tight_layout()
fig2.savefig(f'{figures_dir}/02_wall_ray_distribution_topdown.png', dpi=150, bbox_inches='tight')
plt.close(fig2)
print("Saved 02_wall_ray_distribution_topdown.png")

# ── Figure 00: Listener grid ────────────────────────────
fig0, ax0 = plt.subplots(figsize=(7, 7))
room_rect = patches.Rectangle((0, 0), floor_width, floor_depth,
                               linewidth=2, edgecolor='black', facecolor='lightyellow')
ax0.add_patch(room_rect)
ax0.axvline(wall_x, color='gray', linewidth=3, linestyle='--', label='Shared wall (x=10)')
ax0.scatter(listener_xs, listener_ys, c='steelblue', s=150, marker='^', zorder=5, label='Listener grid')
for pos in listener_grid:
    ax0.annotate(f'({pos[0]}, {pos[1]}, {pos[2]})',
                 xy=(pos[0], pos[1]), xytext=(6, 6),
                 textcoords='offset points', fontsize=7, color='steelblue')
ax0.scatter([src_x], [src_y], c='red', s=300, marker='*', zorder=5, label='Gunshot source')
ax0.set_xlabel('X (m)', fontsize=11); ax0.set_ylabel('Y (m)', fontsize=11)
ax0.set_title('Room 1 — Source and Listener Grid (wall face)', fontsize=12)
ax0.legend(fontsize=9); ax0.grid(True, alpha=0.3); ax0.set_aspect('equal')
ax0.set_xlim(-0.5, floor_width + 0.5); ax0.set_ylim(-0.5, floor_depth + 0.5)
fig0.tight_layout()
fig0.savefig(f'{figures_dir}/00_listener_grid.png', dpi=150, bbox_inches='tight')
plt.close(fig0)
print("Saved 00_listener_grid.png")

# ── Figure 3: Wall ray energy histogram ─────────────────
fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(14, 5))

ax3a.hist(wall_rays['total_energy'].values, bins=50,
          color='steelblue', edgecolor='black', alpha=0.7)
ax3a.axvline(min(all_impact_energy), color='orange', linestyle='--', linewidth=2,
             label=f'Top {rays_per_listener} per listener threshold')
ax3a.set_xlabel('Total Ray Energy (all 8 bands)')
ax3a.set_ylabel('Count')
ax3a.set_title('Wall Ray Energy — Linear Scale')
ax3a.legend(); ax3a.grid(True, alpha=0.3)

data_hist = wall_rays['total_energy'].values
data_hist = data_hist[data_hist > 0]
ax3b.hist(data_hist, bins=np.logspace(np.log10(data_hist.min()), np.log10(data_hist.max()), 50),
          color='steelblue', edgecolor='black', alpha=0.7)
ax3b.axvline(min(all_impact_energy), color='orange', linestyle='--', linewidth=2,
             label=f'Top {rays_per_listener} per listener threshold')
ax3b.set_xscale('log')
ax3b.set_xlabel('Total Ray Energy (all 8 bands, log scale)')
ax3b.set_ylabel('Count')
ax3b.set_title('Wall Ray Energy — Log Scale')
ax3b.legend(); ax3b.grid(True, alpha=0.3)

fig3.suptitle(f'Wall Ray Energy Distribution ({len(wall_rays)} rays) — scored by energy/distance')
fig3.tight_layout()
fig3.savefig(f'{figures_dir}/03_wall_ray_energy_histogram.png', dpi=150, bbox_inches='tight')
plt.close(fig3)
print("Saved 03_wall_ray_energy_histogram.png")

# ── Figure 4: pyva TL curve ─────────────────────────────
fig4, ax4 = plt.subplots(figsize=(8, 5))
ax4.semilogx(frequencies, TL_db, 'o-', color='purple', linewidth=2, markersize=8)
ax4.fill_between(frequencies, TL_db, TL_db[-1]*1.1, alpha=0.2, color='purple')
ax4.set_xlabel('Frequency (Hz)'); ax4.set_ylabel('Transmission Loss (dB)')
ax4.set_title(f'pyva Transmission Loss — {material_chosen} {thickniss*1000:.0f}mm')
ax4.invert_yaxis(); ax4.grid(True, alpha=0.3); ax4.set_xlim(100, 20000)
for f, tl in zip(frequencies, TL_db):
    ax4.annotate(f'{tl:.0f}dB', (f, tl), textcoords="offset points",
                 xytext=(0, 8), ha='center', fontsize=8)
fig4.tight_layout()
fig4.savefig(f'{figures_dir}/04_pyva_TL_curve.png', dpi=150, bbox_inches='tight')
plt.close(fig4)
print("Saved 04_pyva_TL_curve.png")

# ── Figure 5: Room 1 spectrogram ────────────────────────
fig5, ax5 = plt.subplots(figsize=(8, 5))
ax5.specgram(room1_recording[0], Fs=fs, cmap='inferno', vmin=-100, vmax=-20)
ax5.set_title('Room 1 — Spectrogram (W channel)')
ax5.set_xlabel('Time (s)'); ax5.set_ylabel('Frequency (Hz)')
ax5.set_ylim(0, 8000); ax5.set_xlim(0, 0.5)
fig5.tight_layout()
fig5.savefig(f'{figures_dir}/05_room1_spectrogram.png', dpi=150, bbox_inches='tight')
plt.close(fig5)
print("Saved 05_room1_spectrogram.png")

# ── Figure 6: Room 2 spectrogram ────────────────────────
fig6, ax6 = plt.subplots(figsize=(8, 5))
ax6.specgram(room2_recording[0], Fs=fs, cmap='inferno', vmin=-100, vmax=-20)
ax6.set_title('Room 2 — Spectrogram (W channel, transmitted)')
ax6.set_xlabel('Time (s)'); ax6.set_ylabel('Frequency (Hz)')
ax6.set_ylim(0, 8000); ax6.set_xlim(0, 0.5)
fig6.tight_layout()
fig6.savefig(f'{figures_dir}/06_room2_spectrogram.png', dpi=150, bbox_inches='tight')
plt.close(fig6)
print("Saved 06_room2_spectrogram.png")

# ── Figure 7: Room 1 vs Room 2 waveform comparison ──────
fig7, (ax7a, ax7b) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
t = np.arange(len(room1_recording[0])) / fs
ax7a.plot(t, room1_recording[0], color='steelblue', linewidth=0.5)
ax7a.set_title('Room 1 — Waveform (W channel)')
ax7a.set_ylabel('Amplitude'); ax7a.grid(True, alpha=0.3)
ax7b.plot(t, room2_recording[0], color='coral', linewidth=0.5)
ax7b.set_title('Room 2 — Waveform (W channel, same scale)')
ax7b.set_ylabel('Amplitude'); ax7b.set_xlabel('Time (s)'); ax7b.grid(True, alpha=0.3)
ax7a.set_xlim(0, 0.5); ax7b.set_xlim(0, 0.5)
fig7.suptitle(f'Room 2 level: {np.max(np.abs(room2_recording))/np.max(np.abs(room1_recording))*100:.2f}% of Room 1')
fig7.tight_layout()
fig7.savefig(f'{figures_dir}/07_waveform_comparison.png', dpi=150, bbox_inches='tight')
plt.close(fig7)
print("Saved 07_waveform_comparison.png")

# ── Figure 8: Per-band TL applied ───────────────────────
fig8, ax8 = plt.subplots(figsize=(8, 5))
band_labels = [str(int(f)) for f in frequencies]
room1_band_energy = [df[f'intensity_band_{b}'].sum() for b in range(8)]
room2_band_energy = [all_intensities[b].sum() for b in range(8)]
x = np.arange(8)
width = 0.35
ax8.bar(x - width/2, room1_band_energy, width, label='Room 1', color='steelblue', alpha=0.8)
ax8.bar(x + width/2, room2_band_energy, width, label='Room 2 (post-TL)', color='coral', alpha=0.8)
ax8.set_xticks(x); ax8.set_xticklabels(band_labels)
ax8.set_xlabel('Frequency Band (Hz)'); ax8.set_ylabel('Total Ray Energy')
ax8.set_title('Per-band Energy: Room 1 vs Room 2 after pyva TL')
ax8.legend(); ax8.grid(True, alpha=0.3, axis='y')
fig8.tight_layout()
fig8.savefig(f'{figures_dir}/08_perband_energy_comparison.png', dpi=150, bbox_inches='tight')
plt.close(fig8)
print("Saved 08_perband_energy_comparison.png")

print(f"\nAll figures saved to {figures_dir}/")