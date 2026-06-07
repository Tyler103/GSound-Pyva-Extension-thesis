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
wall_delay   = 0.2 / 3500  # 20cm slab at 3500 m/s

# ─────────────────────────────────────
# STEP 1: pyva TL coefficients
# ─────────────────────────────────────
def get_TL(frequencies, material='drywall', thickness=0.012):
    omega = 2 * np.pi * frequencies
    air   = matC.Fluid()

    if material == 'drywall':
        mat       = matC.IsoMat(E=2.5e9, rho0=800, nu=0.3, eta=0.01)
        thickness = 0.012
    elif material == 'concrete':
        mat       = matC.IsoMat(E=3.8e10, rho0=2300, nu=0.2, eta=0.02)
        thickness = 0.2
    elif material == 'timber':
        mat       = matC.IsoMat(E=1.1e10, rho0=500, nu=0.3, eta=0.03)
        thickness = 0.05

    plate = sProp.PlateProp(thickness, mat)
    tau   = np.array([
        plate.transmission_coefficient_angular(w, theta=0, fluid1=air)
        for w in omega
    ])
    TL_db = -10 * np.log10(tau + 1e-10)
    return TL_db, tau

frequencies = np.array([125, 250, 500, 1000, 2000, 4000, 8000, 16000])
TL_db, tau  = get_TL(frequencies, material='drywall', thickness=0.012)

print("pyva TL (drywall 12mm):")
for f, tl, t in zip(frequencies, TL_db, tau):
    print(f"  {f:6d} Hz: TL={tl:.1f}dB  tau={t:.6f}  ({t*100:.4f}% survives)")

# ─────────────────────────────────────
# STEP 2: Generate Parquet via pipeline
# ─────────────────────────────────────
print("\nRunning RayDataPipeline...")

pipeline = RayDataPipeline(
    diffuse_count=5000,
    specular_count=1000,
    energy_percentage=95.0,
)

parquet_path = pipeline.process_coordinates(
    mesh_path='/app/ray_generator/examples/cube.obj',
    source_positions=[(1.0, 1.0, 0.5)],
    listener_positions=[(5.0, 3.0, 0.5)],
    output_path='/app/ray_generator/examples/output'
)

print(f"Parquet saved: {parquet_path}")

# ─────────────────────────────────────
# STEP 3: Load dataset for auralizer
# ─────────────────────────────────────
print("\nLoading dataset...")

data = create_dataset(
    ray=parquet_path,
    room='/app/ray_generator/examples/cube.obj'
)

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

auralizer = Ambisonic_IR_Generator(
    fs=sample_rate,
    order=1,
    imp_res_time=3.0
)

sir_room1 = auralizer.forward_ambsonics(demo)
print(f"Room 1 IR shape: {sir_room1.shape}")

# ─────────────────────────────────────
# STEP 5: Find floor hitting rays
# ─────────────────────────────────────
print("\nFinding floor hitting rays...")

df = pd.read_parquet(parquet_path)
df['time'] = df['distance'] / df['speed_of_sound']

floor_rays = df[df['source_direction_z'] < -0.5].copy()
print(f"Floor rays: {len(floor_rays)} / {len(df)}")
print(f"Floor energy: {floor_rays['intensity_band_0'].sum()/df['intensity_band_0'].sum()*100:.1f}%")

# ─────────────────────────────────────
# STEP 6: Run Room 2 via RayDataPipeline
# Virtual sources = top 50 floor ray impact points
# ─────────────────────────────────────
print("\nRunning Room 2 via RayDataPipeline...")


top_floor = floor_rays.nlargest(50, 'intensity_band_0').reset_index(drop=True)

# Ray hits floor at: source + direction * t, where t = source_z / -direction_z
t_floor = top_floor['source_z'].values / (-top_floor['source_direction_z'].values)

impact_x = top_floor['source_x'].values + top_floor['source_direction_x'].values * t_floor
impact_y = top_floor['source_y'].values + top_floor['source_direction_y'].values * t_floor

vsrc_positions = [
    (float(impact_x[i]), float(impact_y[i]), 0.1)
    for i in range(len(top_floor))
]

# Sanity check
print("Sample floor impact points:")
for i in range(5):
    print(f"  ({impact_x[i]:.2f}, {impact_y[i]:.2f}, 0.1)")

pipeline_room2 = RayDataPipeline(
    diffuse_count=5000,
    specular_count=1000,
    energy_percentage=95.0,
)

parquet_room2 = pipeline_room2.process_coordinates(
    mesh_path='/app/ray_generator/examples/cube.obj',
    source_positions=vsrc_positions,
    listener_positions=[(5.0, 3.0, 1.5)],
    output_path='/app/ray_generator/examples/output'
)
print(f"Room 2 parquet: {parquet_room2}")

# ─────────────────────────────────────
# STEP 7: Load Room 2 dataset
# ─────────────────────────────────────
data_room2 = create_dataset(ray=parquet_room2, room='/app/ray_generator/examples/cube.obj')

# Aggregate all 50 source->listener path sets into one demo dict
all_intensities = np.concatenate([d['Intensities'] for d in data_room2], axis=1)
all_doa         = np.concatenate([d['doa']         for d in data_room2], axis=1)
all_delays      = np.concatenate([d['delay']       for d in data_room2])

# Apply pyva TL per band
for b in range(8):
    all_intensities[b] *= tau[b]

demo_room2 = {
    'tx':          np.array([5.0, 5.0, 0.1]),
    'rx':          np.array([5.0, 3.0, 1.5]),
    'Intensities': all_intensities,
    'doa':         all_doa,
    'delay':       all_delays + wall_delay,
    'V':           float(floor_width * floor_depth * floor_height)
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
# STEP 12: Save figures
# ─────────────────────────────────────
figures_dir = '/app/ray_generator/examples/figures'
os.makedirs(figures_dir, exist_ok=True)

src_x, src_y, src_z = 1.0, 1.0, 0.5

# ── Figure 1: 3D building overview ──────────────────────
fig1 = plt.figure(figsize=(10, 8))
ax1  = fig1.add_subplot(111, projection='3d')

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

draw_room(ax1, 0, 0, 0,            floor_width, floor_depth, floor_height, 'steelblue', label='Room 1')
draw_room(ax1, 0, 0, floor_height, floor_width, floor_depth, floor_height, 'coral',     label='Room 2')

ax1.scatter([src_x], [src_y], [src_z],
            c='red', s=200, marker='*', zorder=5, label='Gunshot')
ax1.scatter([5.0], [3.0], [0.5],
            c='steelblue', s=100, marker='^', zorder=5, label='Mic Room 1')
ax1.scatter([5.0], [3.0], [floor_height+1.5],
            c='coral', s=100, marker='^', zorder=5, label='Mic Room 2')

top20 = floor_rays.nlargest(20, 'intensity_band_0')
ax1.scatter(top20['source_x'].values, top20['source_y'].values, np.zeros(len(top20)),
            c='orange', s=50, marker='o', alpha=0.7, label='Floor ray exits')

ax1.set_xlabel('X (m)'); ax1.set_ylabel('Y (m)'); ax1.set_zlabel('Z (m)')
ax1.set_title('3D Building Overview')
ax1.legend(fontsize=7, loc='upper left')
fig1.tight_layout()
fig1.savefig(f'{figures_dir}/01_building_3d_overview.png', dpi=150, bbox_inches='tight')
plt.close(fig1)
print("Saved 01_building_3d_overview.png")

# ── Figure 2: Top-down floor plan ───────────────────────
fig2, ax2 = plt.subplots(figsize=(8, 8))
room_rect = patches.Rectangle((0, 0), floor_width, floor_depth,
                                linewidth=2, edgecolor='black', facecolor='lightyellow')
ax2.add_patch(room_rect)
scatter = ax2.scatter(floor_rays['source_x'].values, floor_rays['source_y'].values,
                      c=floor_rays['intensity_band_0'].values,
                      cmap='hot', s=20, alpha=0.7, label='Floor rays')
plt.colorbar(scatter, ax=ax2, label='Energy (band 0)')
ax2.scatter([src_x], [src_y], c='red', s=300, marker='*', zorder=5, label='Gunshot')
ax2.scatter([5.0], [3.0], c='blue', s=150, marker='^', zorder=5, label='Mic Room 1')
ax2.scatter(impact_x, impact_y, c='orange', s=60, marker='o',
            alpha=0.8, label='Virtual sources (Room 2)')
ax2.set_xlabel('X (m)'); ax2.set_ylabel('Y (m)')
ax2.set_title('Top-down: Floor Ray Distribution')
ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3); ax2.set_aspect('equal')
ax2.set_xlim(0, floor_width); ax2.set_ylim(0, floor_depth)
fig2.tight_layout()
fig2.savefig(f'{figures_dir}/02_floor_ray_distribution_topdown.png', dpi=150, bbox_inches='tight')
plt.close(fig2)
print("Saved 02_floor_ray_distribution_topdown.png")

# ── Figure 3: Floor ray energy histogram ────────────────
fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(14, 5))

# Left: linear scale (original, shows the skew)
ax3a.hist(floor_rays['intensity_band_0'].values, bins=50,
          color='steelblue', edgecolor='black', alpha=0.7)
ax3a.axvline(top_floor['intensity_band_0'].min(), color='orange',
             linestyle='--', linewidth=2, label='Top 50 threshold')
ax3a.set_xlabel('Ray Energy (band 0)'); ax3a.set_ylabel('Count')
ax3a.set_title('Floor Ray Energy — Linear Scale')
ax3a.legend(); ax3a.grid(True, alpha=0.3)

# Right: log x-axis — reveals the full distribution
data = floor_rays['intensity_band_0'].values
data = data[data > 0]  # log scale needs positive values
ax3b.hist(data, bins=np.logspace(np.log10(data.min()), np.log10(data.max()), 50),
          color='steelblue', edgecolor='black', alpha=0.7)
ax3b.axvline(top_floor['intensity_band_0'].min(), color='orange',
             linestyle='--', linewidth=2, label='Top 50 threshold')
ax3b.set_xscale('log')
ax3b.set_xlabel('Ray Energy (band 0, log scale)'); ax3b.set_ylabel('Count')
ax3b.set_title('Floor Ray Energy — Log Scale')
ax3b.legend(); ax3b.grid(True, alpha=0.3)

fig3.suptitle('Floor Ray Energy Distribution (755 rays)')
fig3.tight_layout()
fig3.savefig(f'{figures_dir}/03_floor_ray_energy_histogram.png', dpi=150, bbox_inches='tight')
plt.close(fig3)
print("Saved 03_floor_ray_energy_histogram.png")

# ── Figure 4: pyva TL curve ─────────────────────────────
fig4, ax4 = plt.subplots(figsize=(8, 5))
ax4.semilogx(frequencies, TL_db, 'o-', color='purple', linewidth=2, markersize=8)
ax4.fill_between(frequencies, TL_db, TL_db[-1]*1.1, alpha=0.2, color='purple')
ax4.set_xlabel('Frequency (Hz)'); ax4.set_ylabel('Transmission Loss (dB)')
ax4.set_title('pyva Transmission Loss — drywall 12mm')
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
band_labels = [str(f) for f in frequencies]
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