# save as /app/ray_generator/examples/full_pipeline.py

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import fftconvolve, resample
from numpy.fft import rfft, irfft, rfftfreq
import sys
import os

# Add parent path for auralizer
sys.path.insert(0, '/app')

# ─────────────────────────────────────
# Import pipeline and auralizer
# ─────────────────────────────────────
from ray_pipeline import RayDataPipeline, process_position_pair
from py_auralizer import Ambisonic_IR_Generator, create_dataset

import pyva.properties.materialClasses as matC
import pyva.properties.structuralPropertyClasses as sProp

# ─────────────────────────────────────
# STEP 1: TL coefficients
# ─────────────────────────────────────
def get_TL(material='drywall_single'):
    materials = {
        'concrete_200mm': np.array([36, 42, 48, 54, 57, 60, 62, 63]),
        'concrete_100mm': np.array([30, 36, 42, 48, 51, 54, 56, 57]),
        'drywall_single': np.array([15, 20, 25, 30, 33, 36, 38, 39]),
        'drywall_double': np.array([25, 30, 35, 40, 45, 48, 50, 52]),
        'timber_floor':   np.array([20, 25, 30, 35, 38, 40, 42, 43]),
    }
    TL_db = materials[material]
    tau   = 10 ** (-TL_db / 20)
    return TL_db, tau

def apply_tl_filter(ir, frequencies, TL_db, sample_rate):
    N        = len(ir)
    IR_fft   = rfft(ir)
    freqs    = rfftfreq(N, d=1/sample_rate)
    gain_db  = np.interp(freqs, frequencies, -TL_db)
    gain_lin = 10 ** (gain_db / 20)
    return irfft(IR_fft * gain_lin, n=N)

frequencies     = np.array([125, 250, 500, 1000, 2000, 4000, 8000, 16000])
TL_db, tau      = get_TL('drywall_single')

print("TL coefficients:")
for f, tl in zip(frequencies, TL_db):
    print(f"  {f:6d} Hz: {tl}dB")

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

sample_rate = 48000

auralizer = Ambisonic_IR_Generator(
    fs=sample_rate,
    order=1,          # 1st order — 4 channels (W, X, Y, Z)
    imp_res_time=3.0  # 3 second IR
)

sir_room1 = auralizer.forward_ambsonics(demo)
print(f"Room 1 IR shape: {sir_room1.shape}")
# (4, num_samples) — 4 ambisonic channels

# ─────────────────────────────────────
# STEP 5: Build Room 2 IR
# Apply pyva tau to rays BEFORE auralizer
# ─────────────────────────────────────
print("\nBuilding Room 2 IR from floor rays...")

# Load parquet to find floor rays
df = pd.read_parquet(parquet_path)
df['time'] = df['distance'] / df['speed_of_sound']

# Floor hitting rays
floor_rays = df[df['source_direction_z'] < -0.5].copy()
print(f"Floor rays: {len(floor_rays)} / {len(df)}")

# Floor ray intensities — shape (8, num_floor_rays)
floor_intensities = floor_rays[[f'intensity_band_{i}' for i in range(8)]].to_numpy().T

# Apply pyva tau per frequency band BEFORE auralization
# This is physically correct — auralizer sees attenuated energies
for i in range(8):
    floor_intensities[i] *= tau[i]

print(f"Applied pyva TL — band 0 tau={tau[0]:.4f}, band 7 tau={tau[7]:.4f}")

# Convert directions to spherical
floor_doa_xyz = floor_rays[['source_direction_x',
                             'source_direction_y',
                             'source_direction_z']].to_numpy().T

from py_auralizer import _cart2sph
floor_doa = _cart2sph(floor_doa_xyz)

# Wall crossing delay
wall_delay = 0.2 / 3500  # seconds

demo_room2 = {
    'tx':          demo['tx'],
    'rx':          demo['rx'],
    'Intensities': floor_intensities / 8.0,  # pyva already applied
    'doa':         floor_doa,
    'delay':       floor_rays['time'].values + wall_delay,
    'V':           demo['V']
}

# Auralizer sees correct attenuated energies — no post filter needed
sir_room2 = auralizer.forward_ambsonics(demo_room2)
print(f"Room 2 IR shape: {sir_room2.shape}")

# ─────────────────────────────────────
# STEP 6: Load gunshot
# ─────────────────────────────────────
gunshot, fs = sf.read('/app/ray_generator/examples/210766__acs272__gun-shot-in-anechoic-chamber.wav')

if gunshot.ndim > 1:
    gunshot = gunshot.mean(axis=1)

if fs != sample_rate:
    gunshot = resample(gunshot, int(len(gunshot) * sample_rate / fs))
    fs = sample_rate

gunshot = gunshot / np.max(np.abs(gunshot))
print(f"\nLoaded gunshot: {len(gunshot)} samples, {fs}Hz")

# ─────────────────────────────────────
# STEP 7: Convolve each channel
# ─────────────────────────────────────
print("\nConvolving...")

# Room 1 — convolve each ambisonic channel
room1_channels = []
for ch in range(sir_room1.shape[0]):
    room1_channels.append(fftconvolve(gunshot, sir_room1[ch]))

# Room 2 — convolve each ambisonic channel
room2_channels = []
for ch in range(sir_room2.shape[0]):
    room2_channels.append(fftconvolve(gunshot, sir_room2[ch]))

# Stack to (channels, samples)
room1_recording = np.array(room1_channels)
room2_recording = np.array(room2_channels)

# Normalize to Room 1 peak
peak            = np.max(np.abs(room1_recording))
room1_recording = room1_recording / peak
room2_recording = room2_recording / peak

# Audible Room 2
room2_audible = room2_recording / (np.max(np.abs(room2_recording)) + 1e-10)

# Save — use W channel (omnidirectional) for mono comparison
sf.write('output_room1_W.wav', room1_recording[0].astype(np.float32), fs)
sf.write('output_room2_W.wav', room2_recording[0].astype(np.float32), fs)
sf.write('output_room2_audible_W.wav', room2_audible[0].astype(np.float32), fs)

# Save full 4-channel ambisonics
sf.write('output_room1_ambi.wav', room1_recording.T.astype(np.float32), fs)
sf.write('output_room2_ambi.wav', room2_recording.T.astype(np.float32), fs)

print("\nSaved:")
print("  output_room1_W.wav          — Room 1 mono (W channel)")
print("  output_room2_W.wav          — Room 2 mono (W channel, same scale)")
print("  output_room2_audible_W.wav  — Room 2 mono (normalized for listening)")
print("  output_room1_ambi.wav       — Room 1 4-channel Ambisonics")
print("  output_room2_ambi.wav       — Room 2 4-channel Ambisonics")

# ─────────────────────────────────────
# STEP 8: Summary
# ─────────────────────────────────────
print("\n─── Summary ─────────────────────────────────────────")
print(f"Total rays:         {len(df):,}")
print(f"Floor rays:         {len(floor_rays):,}")
print(f"Floor energy %:     {floor_rays['intensity_band_0'].sum()/df['intensity_band_0'].sum()*100:.1f}%")
print(f"Room 1 IR shape:    {sir_room1.shape}")
print(f"Room 2 IR shape:    {sir_room2.shape}")
print(f"Room 2 vs Room 1:   {np.max(np.abs(room2_recording))/np.max(np.abs(room1_recording))*100:.4f}%")
print(f"TL material:        drywall_single")
print(f"TL range:           {TL_db[0]}dB (125Hz) to {TL_db[-1]}dB (16kHz)")
print("─────────────────────────────────────────────────────")