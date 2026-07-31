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
import pyva.systems.acoustic3Dsystems as ac3
import pyva.systems.structure2Dsystems as st2
import pyva.coupling.junctions as jun
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.ndimage import gaussian_filter, maximum_filter, label
import time

# ─────────────────────────────────────
# CONFIG
# ─────────────────────────────────────
floor_width  = 5
floor_depth  = 5
floor_height = 2.5
# Room 2 width        
room2_width  = 5
sample_rate  = 48000

# x-coordinate of the shared wall
wall_x       = floor_width          
wall_delay   = 0.2 / 3500 

listener_grid = []

figures_dir = '/app/ray_generator/examples/figures'
audio_dir = '/app/ray_generator/examples/audio'
os.makedirs(figures_dir, exist_ok=True)
os.makedirs(audio_dir, exist_ok=True)

# drywall: 0.012, concrete: 0.18, timber: 0.05, alum: 0.006
frequencies = np.array([125, 250, 500, 1000, 2000, 4000, 8000, 16000], dtype=float)

src_x = 2.5
src_y = 2.5
src_z = 1.0

material_chosen = 'aluminium'
thick = 0.006

d_count = 1000
s_count = 5000

def compute_occupancy(H):
    """
    Fraction of wall grid bins that received at least one ray impact.
    Uses raw H (pre-smoothing) since Gaussian smoothing spreads energy
    into neighboring bins that weren't actually hit by any real ray.
    """
    n_total_bins = H.size
    n_occupied_bins = np.count_nonzero(H)
    occupancy_pct = (n_occupied_bins / n_total_bins) * 100
    return occupancy_pct


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
        mat       = matC.IsoMat(E=2.85e10, rho0=2286, nu=0.2, eta=0.02)
    elif material == 'drywall':
        mat = matC.IsoMat(E=2.05e9, rho0=767, nu=0.3, eta=0.0163)
    elif material == 'timber':
        mat       = matC.IsoMat(E=1.1e10, rho0=500, nu=0.3, eta=0.03)
    elif material == 'aluminium':
        mat = matC.IsoMat(E=7.1e10, rho0=2700.0, nu=0.34, eta=0.01)

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

    # prints the transmission loss over 8 frequency bands
    print(f"Diffuse field TL ({material_chosen}):")
    for f, tl, t in zip(frequencies, TL_db, tau_diffuse):
        print(f"  {f:6.0f} Hz: TL={tl:.1f}dB  tau={t:.6f}  ({t*100:.4f}% survives)")

    # TL in dB, raw τ values 
    return TL_db, tau_diffuse


def room1_simulate():
    print("\nRunning RayDataPipeline...")

    listener_grid.clear()

    
    fracs = np.linspace(0.2, 0.8, 3)

    for d in fracs:
        for h in fracs:
            x = floor_width - 0.5
            y = floor_depth * d
            z = floor_height * h
            listener_grid.append((x, y, z))

    #listener_grid.append((floor_width - 0.5, floor_depth * 0.5, floor_height * 0.5))



    # initialize ray creation pipeline
    pipeline = RayDataPipeline(
        diffuse_count=d_count,
        specular_count=s_count,
        energy_percentage=95.0,
    )

    batch_size = 9
    batch_paths = []
    output_dir = '/app/ray_generator/examples/output'


    for i in range(0, len(listener_grid), batch_size):
        batch = listener_grid[i: i + batch_size]
        print(f"Processing listeners {i} to {i + len(batch) - 1} of {len(listener_grid)}")

        # simulates ray generation and saves to parquet
        batch_path = pipeline.process_coordinates(
            mesh_path='/app/ray_generator/examples/cube.obj',
            source_positions=[(src_x , src_y, src_z)],
            listener_positions=batch,
            output_path=output_dir
        )
        batch_paths.append(batch_path)

    # combine all batch outputs into a list
    dfs = []
    for p in batch_paths:
        df = pd.read_parquet(p)
        dfs.append(df)

    # actually combine the batches
    combined = pd.concat(dfs, ignore_index=True)

    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    parquet_path = os.path.join(output_dir, f'combined_paths_{timestamp}.parquet')
    combined.to_parquet(parquet_path)

    # clean up intermediate batch files
    for p in batch_paths:
        os.remove(p)

    return parquet_path

def gen_histogram(df):
    df_hist = df.copy()

    print(f"Number of intensity bands: {df_hist['param_num_bands'].iloc[0]}")
    print(f"Band columns: {[c for c in df_hist.columns if 'intensity_band' in c]}")
    print(f"Frequencies array length: {len(frequencies)}")
    #print(f"Parquet saved: {parquet_path}")

    # add time and energy to table
    # how many seconds after the gunshot fired each 
    # individual ray arrived at the listener.
    df_hist['time'] = df_hist['distance'] / df_hist['speed_of_sound']

    band_cols = [f'intensity_band_{b}' for b in range(8)]
    df_hist['total_energy'] = df_hist[band_cols].sum(axis=1)

    # histogram settings, 50 bins across first 30ms window
    n_bins = 50
    time_range = (0, 0.03)

    n_listeners = len(listener_grid)

    # format for subplot grid
    ncols = int(np.ceil(np.sqrt(n_listeners)))
    nrows = int(np.ceil(n_listeners / ncols))

    # build blank canvas boxes
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(4 * ncols, 3 * nrows),
        sharex=True, sharey=True,
    )
    axes = np.atleast_1d(axes).flatten()

    # loop through each listner and draw histogram
    for i, (lx, ly, lz) in enumerate(listener_grid):
        ax = axes[i]
        # build a mask that isolates rays matching listener cord
        mask = (
            np.isclose(df_hist['listener_x'], lx) &
            np.isclose(df_hist['listener_y'], ly) &
            np.isclose(df_hist['listener_z'], lz)
        )
        sub = df_hist[mask]

        # safety if no rays matched
        if len(sub) == 0:
            ax.set_title(f'({lx:.1f}, {ly:.1f}, {lz:.1f})\nno rays', fontsize=8)
            ax.axis('off')
            continue
        
        # Actual histogram call
        # bin listener by time, sum energy per bin
        # tells us how much energy arrived in this time slice
        ax.hist(
            sub['time'], bins=n_bins, range=time_range,
            weights=sub['total_energy'],
            color='steelblue', alpha=0.85,
        )
        ax.set_title(f'({lx:.1f}, {ly:.1f}, {lz:.1f})', fontsize=8)
        ax.grid(True, alpha=0.3)

    # clean up unused grid cells
    for j in range(n_listeners, len(axes)):
        axes[j].axis('off')

    fig.supxlabel('Arrival time (s)')
    fig.supylabel('Summed energy (all 8 bands)')
    fig.suptitle('Per-Listener Energy Arrival Histograms', fontsize=13)
    fig.tight_layout()
    fig.savefig(f'{figures_dir }/00c_listener_histograms.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {figures_dir}/00c_listener_histograms.png")

def load_auralizer(df, parquet_path):
    print("\nLoading dataset...")

    df_full = df

    central_listener = (
        floor_width - 0.5,          
        floor_depth * 0.5,          
        floor_height * 0.5          
    )

    cx, cy, cz = central_listener
    print(f"{cx}, {cy}, {cz}")

    df_single = df_full[
        np.isclose(df_full['listener_x'], cx) &
        np.isclose(df_full['listener_y'], cy) &
        np.isclose(df_full['listener_z'], cz)
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

    return demo, df_full

def gen_IR(demo):
    print("\nGenerating Room 1 IR...")

    # Instantiates the Ambisonic IR generator
    auralizer = Ambisonic_IR_Generator(
        fs=sample_rate,
        order=1,
        imp_res_time=10.0
    )

    # output is 4 Ambisonic channels × IR length in samples
    # 1. places each ray in time
    # 2. encodes each ray spatially
    # 3. Scales by per-band intensity

    sir_room1 = auralizer.forward_ambsonics(demo)
    print(f"Room 1 IR shape: {sir_room1.shape}")

    return sir_room1, auralizer

def gen_wall_impacts(df_full):
   
    #  Find wall-hitting rays
    print("\nFinding wall-hitting rays...")

    df = df_full.copy()
    df['time'] = df['distance'] / df['speed_of_sound']

    # build a list of all 8 band column names
    band_cols = [f'intensity_band_{b}' for b in range(8)]

    # sums all 8 frequency bands for each ray into one total-energy column
    df['total_energy'] = df[band_cols].sum(axis=1)

    # threshold of wall ray arrivals
    threshold_deg = 60
    cos_cutoff = np.cos(np.radians(threshold_deg))
    wall_rays = df[df['listener_direction_x'] > cos_cutoff].copy()

    print(f"Wall rays: {len(wall_rays)} / {len(df)}")
    #print(f"Wall energy: {wall_rays['intensity_band_0'].sum()/df['intensity_band_0'].sum()*100:.1f}%")

    return wall_rays, band_cols, df


def cluster_vsc(wall_rays, band_cols):

    print("\nComputing wall impact points (heatmap + clustering)...")
    # project every wall ray onto the shared wall plane
    # wall x - listen locat / listener direction x
    t  = (wall_x - wall_rays['listener_x'].values) / (wall_rays['listener_direction_x'].values + 1e-10)
    iy_all = wall_rays['listener_y'].values + wall_rays['listener_direction_y'].values * t
    iz_all = wall_rays['listener_z'].values + wall_rays['listener_direction_z'].values * t

    valid_mask = (
        (iy_all >= 0.05) & (iy_all <= floor_depth  - 0.05) &
        (iz_all >= 0.05) & (iz_all <= floor_height - 0.05)
    )

    n_total   = len(iy_all)
    n_dropped = (~valid_mask).sum()
    print(f"Dropping {n_dropped} of {n_total} rays ({100*n_dropped/n_total:.2f}%) as out-of-bounds")

    iy_all = iy_all[valid_mask]
    iz_all = iz_all[valid_mask]

    # filter wall_rays itself to match — critical, since everything downstream
    # (hardness, band_cols, cluster_rays lookup) must stay aligned with iy_all/iz_all
    wall_rays = wall_rays.iloc[valid_mask].reset_index(drop=True)

    wall_rays['impact_y'] = iy_all
    wall_rays['impact_z'] = iz_all

    # "hardness" = how much energy each ray delivers at impact = total_energy (sum of 8 bands)
    hardness = wall_rays['total_energy'].values

    # bin into a 2D grid over the wall (y, z), weighted by hardness
    ny_bins, nz_bins = 40, 20   # tune resolution to wall size / ray density

    # divides the wall into a 40x20 grid 
    # figures out which cell a ray falls into and adds its intensity
    # result is 40×20 array of summed energy per cell-- heatmap
    H, y_edges, z_edges = np.histogram2d(
        iy_all, iz_all, bins=(ny_bins, nz_bins),
        range=[[0, floor_depth], [0, floor_height]],
        weights=hardness
    )

    occupancy_pct = compute_occupancy(H)
    print(f"Occupancy: {occupancy_pct:.2f}%  ({np.count_nonzero(H)}/{H.size} bins hit)")

    # H is the raw 2D histogram (ny_bins x nz_bins) of summed ray energy 
    #per wall cell.

    # blurs so nearby cells influence each other
    H_smooth = gaussian_filter(H, sigma=1.2)

    # not used for
    # find energy cluster centers (hot spots)
    neighborhood = 3
    # finds local maxiumum
    local_max  = (H_smooth == maximum_filter(H_smooth, size=neighborhood))


    # 50% of max
    threshold  = 0.5 * H_smooth.max()

    # cluster is found when energy is at least 50% of max
    # groups adjacent hot cells into connected blob
    cluster_labels, n_clusters_found = label(H_smooth > threshold)

    # mask is true when any cell energy > threshold
    # label finds the connected cell that touch eachother 
    # and becomes cluster
    print(f"Found {n_clusters_found} energy cluster(s) on wall (pre-budget)")

    # overall virtual-source budget
    TOTAL_RAYS = 10

    #this part focuses on turning clusters into a virtual source
    # Below: convert each cluster (a blob of grid cells) into one
    # "virtual source" — a single representative point + energy spectrum

    # Convert bin indices back into real coordinates
    # This computes the midpoint of every bin/grid box, converting bin boundaries into bin centers
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])

    # combines them into two 2D arrays,
    Y_grid, Z_grid = np.meshgrid(y_centers, z_centers, indexing='ij')

    # tells you which bin a ray lands in
    y_bin_idx = np.clip(np.digitize(iy_all, y_edges) - 1, 0, ny_bins - 1)
    z_bin_idx = np.clip(np.digitize(iz_all, z_edges) - 1, 0, nz_bins - 1)

    cluster_summaries = []  # (cluster_energy, cy, cz, per_band_sum, n_rays)

    # loops through each cluster
    for cluster_id in range(1, cluster_labels.max() + 1):
        mask = cluster_labels == cluster_id
        if not mask.any():
            continue

        # finds the center of value of cluster

        # totol energy of cluster
        cluster_energy = H_smooth[mask].sum()

        # weighted average y
        cy = (Y_grid[mask] * H_smooth[mask]).sum() / cluster_energy

        # weighted average z
        cz = (Z_grid[mask] * H_smooth[mask]).sum() / cluster_energy

        #  checks "is the box this ray landed in one of the cluster's boxes
        ray_in_cluster = mask[y_bin_idx, z_bin_idx]
        if not ray_in_cluster.any():
            continue

        # Trace back to the individual raw rays that physically landed in that cluster's region, 
        # and sum their real (un-smoothed) energy per frequency band.
        cluster_rays  = wall_rays.iloc[np.where(ray_in_cluster)[0]]
        per_band_sum  = cluster_rays[band_cols].sum(axis=0).values  # energy-conserving

        cluster_summaries.append((per_band_sum.sum(), cy, cz, per_band_sum, ray_in_cluster.sum()))


    # if more clusters than budget, keep the strongest TOTAL_RAYS by energy
    cluster_summaries.sort(key=lambda c: c[0], reverse=True)
    cluster_summaries = cluster_summaries[:TOTAL_RAYS]

    print(f"Budget: {TOTAL_RAYS} total virtual sources / {len(cluster_summaries)} clusters kept")

    all_impact_y, all_impact_z, all_impact_energy, all_impact_bands = [], [], [], []

    for idx, (energy, cy, cz, per_band_sum, n_rays) in enumerate(cluster_summaries):
        all_impact_y.append(cy)
        all_impact_z.append(cz)
        all_impact_energy.append(energy)
        all_impact_bands.append(per_band_sum)
        print(f"  Cluster {idx:02d}: centroid=({cy:.2f}, {cz:.2f})  n_rays={n_rays}  energy={energy:.4e}")

    impact_y      = np.array(all_impact_y)
    impact_z      = np.array(all_impact_z)
    impact_energy = np.array(all_impact_energy)
    impact_bands  = np.array(all_impact_bands)
  
    print(f"\nTotal virtual sources: {len(impact_y)}")
    print(f"Y range: {impact_y.min():.2f} to {impact_y.max():.2f}")
    print(f"Z range: {impact_z.min():.2f} to {impact_z.max():.2f}")
    print("____________________________________\n")

    #room2_listener_local = (room2_width * 0.5, floor_depth * 0.5, floor_height * 0.5)

    room2_listener_local = (4.5, 2.5, 1)

    vsrc_positions_local = [
        (0.1, float(impact_y[i]), float(impact_z[i]))
        for i in range(len(impact_y))
    ]

    return room2_listener_local, vsrc_positions_local, impact_bands, impact_y, impact_z, y_edges, z_edges, H_smooth, impact_energy, iy_all, iz_all, hardness

def room2_simulate(room2_listener_local, vsrc_positions_local, impact_bands, tau):
    # ─────────────────────────────────────
    # STEP 7: Room 2 ray tracing — per source with dampened power + shape correction
    # ─────────────────────────────────────
    print("\nRunning Room 2 ray tracing per source (dampened + shape-corrected)...")

    data_room2 = []
    parquet_room2 = None   # keep last path for summary

    for i, (src_pos, bands) in enumerate(zip(vsrc_positions_local, impact_bands)):
        injected = bands * tau
        injected_total = float(injected.sum())

        print(f"  Source {i:02d}: bands={bands}  tau={tau}  injected_total={injected_total:.6e}")

        if injected_total <= 0:
            continue

        pipeline_i = RayDataPipeline(diffuse_count=d_count, specular_count=s_count,
                                    energy_percentage=95.0, source_power=injected_total)

        pq = pipeline_i.process_coordinates(mesh_path='/app/ray_generator/examples/cube.obj',
                                            source_positions=[src_pos],
                                            listener_positions=[room2_listener_local],
                                            output_path='/app/ray_generator/examples/output')
        parquet_room2 = pq 

        print(f"  Source {i:02d}: parquet={pq}")  

        ds = create_dataset(ray=pq, room='/app/ray_generator/examples/cube.obj')
        
        print(f"  Source {i:02d}: parquet={pq}")
        try:
            df_check = pd.read_parquet(pq)
            print(f"    parquet rows: {len(df_check)}")
            print(f"    parquet source cols: {df_check[['source_x','source_y','source_z']].drop_duplicates().values if 'source_x' in df_check.columns else 'no source_x col'}")
            print(f"    parquet listener cols: {df_check[['listener_x','listener_y','listener_z']].drop_duplicates().values if 'listener_x' in df_check.columns else 'no listener_x col'}")
        except Exception as e:
            print(f"    failed to read parquet directly: {e}")
        print(f"  Source {i:02d}: ds length = {len(ds)}")
        if len(ds) == 0:
            print(f"  Source {i:02d}: no paths found, skipped  (src_pos={src_pos}, listener={room2_listener_local})")
            continue

        d = ds[0]

        # raw GSound output, no rescale, no shape correction
        raw_bands = d['Intensities'].sum(axis=1)   # sum over all rays -> 8-band vector
        raw_total = raw_bands.sum()
        print(f"  Source {i:02d}: raw_bands={raw_bands}  raw_total={raw_total:.6e}")

        data_room2.append(d)

    print(f"Room 2 sources traced: {len(data_room2)}")

    return data_room2, parquet_room2

def doa_handling(data_room2, impact_y, impact_z, auralizer, vsrc_positions_local, room2_listener_local):
    # Combine all per-source Room 2 data — used only for DOA sanity-check
    # printouts and the direction estimate below, NOT for IR generation
    all_intensities = np.concatenate([d['Intensities'] for d in data_room2], axis=1)
    all_doa         = np.concatenate([d['doa']         for d in data_room2], axis=1)
    all_delays      = np.concatenate([d['delay']       for d in data_room2])

    active_sources = len(data_room2)
    print(f"Active sources: {active_sources} / {len(vsrc_positions_local)}")

    print("Sanity check of doa of room 2 listener")
    print(f"\nRaw doa row 0 (az?) range: {all_doa[0].min():.3f} to {all_doa[0].max():.3f}")
    print(f"Raw doa row 1 (el?) range: {all_doa[1].min():.3f} to {all_doa[1].max():.3f}")
    print(f"Single source doa shape: {data_room2[0]['doa'].shape}")
    print(f"Single source doa sample: {data_room2[0]['doa'][:, 0]}")

    # Only use early-arriving energy for direction estimation — late,
    # highly-diffuse reflections carry no reliable directional information
    # and wash out the true source direction when included.
    early_window = 0.010  # seconds after first arrival; tune based on room size
    first_arrival = all_delays.min()
    early_mask = (all_delays - first_arrival) <= early_window

    weights = all_intensities[:, early_mask].sum(axis=0)
    weights = weights / (weights.sum() + 1e-10)

    az_sin = (np.sin(all_doa[1, early_mask]) * weights).sum()
    az_cos = (np.cos(all_doa[1, early_mask]) * weights).sum()
    az_mean = np.arctan2(az_sin, az_cos)
    if az_mean < 0:
        az_mean += 2 * np.pi

    col_mean = (all_doa[0, early_mask] * weights).sum()
    el_mean = np.pi/2 - col_mean

    # cartesian
    direction = np.array([
        np.sin(col_mean) * np.cos(az_mean),   # x
        np.sin(col_mean) * np.sin(az_mean),   # y
        np.cos(col_mean)                       # z
    ])
    direction = direction / (np.linalg.norm(direction) + 1e-10)

    print(f"Azimuth:   {np.degrees(az_mean):.1f}°")
    print(f"Elevation: {np.degrees(el_mean):.1f}°")
    print(f"Cartesian: [{direction[0]:.3f}, {direction[1]:.3f}, {direction[2]:.3f}]")

    # ── Build Room 2 Ambisonic IR — one call per source, each with its
    # ── own real tx position, then sum in the time domain (linear
    # ── superposition of independent sources) instead of merging raw
    # ── ray data into one dict with a fake averaged tx.
    room_volume = float(room2_width * floor_depth * floor_height)
    sir_room2 = None

    print(f"\nEarly window: {early_window*1000:.1f} ms after first arrival ({first_arrival*1000:.3f} ms)")
    print(f"Rays in early window: {early_mask.sum()} / {len(early_mask)} ({100*early_mask.sum()/len(early_mask):.2f}%)")

    # Check how many distinct virtual sources are actually contributing
    # to the early window, and how much energy each contributes
    ray_source_idx = np.concatenate([
        np.full(d['Intensities'].shape[1], i) for i, d in enumerate(data_room2)
    ])
    early_source_idx = ray_source_idx[early_mask]
    early_source_energy = all_intensities[:, early_mask].sum(axis=0)

    print("Per-source contribution within early window:")
    for i in range(len(data_room2)):
        src_mask = early_source_idx == i
        src_energy = early_source_energy[src_mask].sum() if src_mask.any() else 0.0
        print(f"  Source {i:02d}: {src_mask.sum()} rays, energy={src_energy:.4e}, local_pos={vsrc_positions_local[i]}")

    for i, d in enumerate(data_room2):
        demo_i = {
            'tx':          np.array(vsrc_positions_local[i]),  # this source's real position
            'rx':          np.array(room2_listener_local),
            'Intensities': d['Intensities'],
            'doa':         d['doa'],
            'delay':       d['delay'] + wall_delay,
            'V':           room_volume,
        }
        sir_i = auralizer.forward_ambsonics(demo_i)

        if sir_room2 is None:
            sir_room2 = sir_i.copy()
        else:
            sir_room2 += sir_i

        print(f"  Source {i:02d}: IR energy contribution = {np.sum(sir_i**2):.6e}")

    print(f"Room 2 IR shape: {sir_room2.shape}")

    return sir_room2, all_intensities, all_delays, direction, az_mean, el_mean

def convolveSignal(sir_room1, sir_room2):
    # Load gunshot
    gunshot, fs = sf.read(
        '/app/ray_generator/examples/210766__acs272__gun-shot-in-anechoic-chamber.wav'
    )

    print("test")
    print(np.sum(sir_room1**2))
    print(np.sum(sir_room2**2))




    if gunshot.ndim > 1:
        gunshot = gunshot.mean(axis=1)
    if fs != sample_rate:
        gunshot = resample(gunshot, int(len(gunshot) * sample_rate / fs))
        fs = sample_rate
    gunshot = gunshot / np.max(np.abs(gunshot))
    print(f"\nLoaded gunshot: {len(gunshot)} samples, {fs}Hz, {len(gunshot)/fs:.2f}s")


    #Convolve each Ambisonic channel
    print("\nConvolving...")

    room1_channels = [fftconvolve(gunshot, sir_room1[ch]) for ch in range(4)]
    room2_channels = [fftconvolve(gunshot, sir_room2[ch]) for ch in range(4)]

    room1_recording = np.array(room1_channels)
    room2_recording = np.array(room2_channels)

    peak            = np.max(np.abs(room1_recording))
    room1_recording = room1_recording / peak
    room2_recording = room2_recording / peak

    room2_audible = room2_recording / (np.max(np.abs(room2_recording)) + 1e-10)

    sf.write(os.path.join(audio_dir, 'output_room1_W.wav'),         room1_recording[0].astype(np.float32), fs)
    sf.write(os.path.join(audio_dir, 'output_room2_W.wav'),         room2_recording[0].astype(np.float32), fs)
    sf.write(os.path.join(audio_dir, 'output_room2_audible_W.wav'), room2_audible[0].astype(np.float32),   fs)
    sf.write(os.path.join(audio_dir, 'output_room1_ambi.wav'),      room1_recording.T.astype(np.float32),  fs)
    sf.write(os.path.join(audio_dir, 'output_room2_ambi.wav'),      room2_recording.T.astype(np.float32),  fs)
    
    print("\nSaved:")
    print("  output_room1_W.wav          — Room 1 mono (W channel)")
    print("  output_room2_W.wav          — Room 2 mono (same scale)")
    print("  output_room2_audible_W.wav  — Room 2 mono (normalized for listening)")
    print("  output_room1_ambi.wav       — Room 1 4-channel Ambisonics")
    print("  output_room2_ambi.wav       — Room 2 4-channel Ambisonics")

    return room1_recording, room2_recording, fs

def generate_metrics(
                            df, 
                            wall_rays,
                            vsrc_positions_local,
                            all_delays,
                            sir_room1,
                            sir_room2,
                            room1_recording,
                            room2_recording,
                            parquet_room2,
                            TL_db,
                            all_intensities

                            ):
    # ─────────────────────────────────────
    # STEP 11: Summary
    # ─────────────────────────────────────
    print("\n─── Summary ─────────────────────────────────────────")
    print(f"Total rays:         {len(df):,}")
    print(f"Wall rays:          {len(wall_rays):,}")
    print(f"Wall energy %:      {wall_rays['total_energy'].sum() / df['total_energy'].sum() * 100:.1f}%")
    print(f"Virtual sources:    {len(vsrc_positions_local)}")
    print(f"Room 2 total rays:  {len(all_delays):,}")
    print(f"Room 1 IR shape:    {sir_room1.shape}")
    print(f"Room 2 IR shape:    {sir_room2.shape}")

    energy_ratio = np.sum(room2_recording**2) / np.sum(room1_recording**2)
    il = -10 * np.log10(energy_ratio)
    energy_transmission = energy_ratio * 100
    atr = np.sqrt(energy_ratio)

    print(f"Room 2 vs Room 1:   {atr*100:.4f}%  (ATR)")
    print(f"Insertion Loss:     {il:.2f} dB")
    print(f"Energy transmitted: {energy_transmission:.4f}%")
    print(f"{material_chosen} {thick*1000:.0f}mm ({thick*39.3701:.2f} inches)")
    print(f"TL range:           {TL_db[0]:.1f}dB (125Hz) to {TL_db[-1]:.1f}dB (16kHz)")
    #print(f"Ray selection:      top {rays_per_listener} per listener group ({len(listener_grid)} listeners)")
    print("─────────────────────────────────────────────────────")

    if parquet_room2 is not None:
        df_room2_check = pd.read_parquet(parquet_room2)
        print(f"Room 2 actual paths (last source parquet): {len(df_room2_check):,}")
    else:
        print("Room 2 actual paths (last source parquet): N/A — no sources produced a parquet file")

    print(f"Room 2 actual paths (last source parquet): {len(df_room2_check):,}")
    print(f"Room 2 total combined paths: {all_intensities.shape[1]:,}")
    print(f"Room 2 requested per source: diffuse={d_count}, specular={s_count}")

def draw_room(ax, x0, y0, z0, w, d, h, color, alpha=0.1, label=''):
    # 8 corners of the box
    corners = np.array([
        [x0,   y0,   z0],
        [x0+w, y0,   z0],
        [x0+w, y0+d, z0],
        [x0,   y0+d, z0],
        [x0,   y0,   z0+h],
        [x0+w, y0,   z0+h],
        [x0+w, y0+d, z0+h],
        [x0,   y0+d, z0+h],
    ])

    # 12 edges connecting those corners
    edges = [
        (0,1), (1,2), (2,3), (3,0),   # bottom face
        (4,5), (5,6), (6,7), (7,4),   # top face
        (0,4), (1,5), (2,6), (3,7),   # vertical edges
    ]

    for i, j in edges:
        ax.plot(*zip(corners[i], corners[j]), color=color, linewidth=1.2, alpha=0.8)

    if label:
        ax.text(x0+w/2, y0+d/2, z0+h/2, label, ha='center', va='center',
                fontsize=9, fontweight='bold', color=color)



def generate_figures(y_edges, z_edges, H_smooth, room1_recording, room2_recording, TL_db, impact_energy, impact_y, impact_z, direction, az_mean, 
                    el_mean,fs, vsrc_positions_local,iy_all, iz_all, hardness ):

    
    # ── Figure 1: 3D building overview with wall energy heatmap, multiple angles ──────
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])
    Yg, Zg = np.meshgrid(y_centers, z_centers, indexing='ij')
    Xg = np.full_like(Yg, wall_x)

    norm = plt.Normalize(vmin=H_smooth.min(), vmax=H_smooth.max())
    facecolors = cm.jet(norm(H_smooth))

    r2_listener_global = (
        wall_x + floor_width * 0.5,
        floor_depth * 0.5,
        0.5
    )

    # name -> (elev, azim)
    views = {
        'front':  (0,   0),    # straight-on look at the shared wall face
    
        'angled': (20, -60),   # original 3/4 perspective view
    }

    for name, (elev, azim) in views.items():
        fig1 = plt.figure(figsize=(10, 8))
        ax1  = fig1.add_subplot(111, projection='3d')

        draw_room(ax1, 0,          0, 0, floor_width, floor_depth, floor_height, 'steelblue', label='Room 1')
        draw_room(ax1, floor_width, 0, 0, room2_width, floor_depth, floor_height, 'coral',    label='Room 2')

        wall_surf = ax1.plot_surface(
            Xg, Yg, Zg,
            facecolors=facecolors,
            rstride=1, cstride=1,
            shade=False, alpha=0.9, zorder=1
        )

        mappable = cm.ScalarMappable(norm=norm, cmap='jet')
        mappable.set_array(H_smooth)
        cbar = fig1.colorbar(mappable, ax=ax1, shrink=0.6, pad=0.1)
        cbar.set_label('Wall energy (summed intensity, all bands)')

        ax1.scatter([src_x], [src_y], [src_z], c='red', s=200, marker='*', zorder=5, label='source')
        ax1.scatter(*r2_listener_global, c='coral', s=150, marker='^',
                    edgecolors='black', zorder=5, label='Mic Room 2')
       
        ax1.set_xlabel('X (m)'); ax1.set_ylabel('Y (m)'); ax1.set_zlabel('Z (m)')

        if name == 'front':
            ax1.set_xlabel('')
            ax1.set_xticklabels([])
            ax1.set_xticks([])
        
        ax1.set_title(f'3D Building Overview — Shared Wall Energy — {name.capitalize()} View')
        ax1.legend(fontsize=7, loc='upper left')
        ax1.view_init(elev=elev, azim=azim)

        fig1.tight_layout()
        fig1.savefig(f'{figures_dir}/01_building_heatmap_{name}.png', dpi=150, bbox_inches='tight')
        plt.close(fig1)
        print(f"Saved 01_building_heatmap_{name}.png")


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

    ax2.text(floor_width * 0.5, floor_depth * 0.95, 'Room 1', ha='center', fontsize=12, fontweight='bold', color='steelblue')
    ax2.text(floor_width + room2_width * 0.5, floor_depth * 0.95,'Room 2', ha='center', fontsize=12, fontweight='bold', color='coral')
    ax2.set_xlabel('X (m)'); ax2.set_ylabel('Y (m)')
    ax2.set_title('Top-down: Wall Ray Distribution')
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3); ax2.set_aspect('equal')
    ax2.set_xlim(-0.5, floor_width + room2_width + 0.5)
    ax2.set_ylim(-0.5, floor_depth + 0.5)
    fig2.tight_layout()
    fig2.savefig(f'{figures_dir}/02_wall_ray_distribution_topdown.png', dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print("Saved 02_wall_ray_distribution_topdown.png")


    # ── Figure 00b: 3D Listener grid, multiple angles ────────────────────────
    listener_xs = [pos[0] for pos in listener_grid]
    listener_ys = [pos[1] for pos in listener_grid]
    listener_zs = [pos[2] for pos in listener_grid]

    # name -> (elev, azim)
    views = {
        'front':  (0,   0),
        'side':   (0,  -90),
        'top':    (90,  0),
        'angled': (20, -60),
    }

    for name, (elev, azim) in views.items():
        fig_lg = plt.figure(figsize=(10, 7))
        ax_lg  = fig_lg.add_subplot(111, projection='3d')

        draw_room(ax_lg, 0, 0, 0, floor_width, floor_depth, floor_height, 'steelblue')

        # Shared wall face
        yy, zz = np.meshgrid([0, floor_depth], [0, floor_height])
        ax_lg.plot_surface(np.full_like(yy, wall_x), yy, zz, alpha=0.2, color='gray')

        # Listener grid — coloured by z height
        sc = ax_lg.scatter(
            listener_xs, listener_ys, listener_zs,
            c=listener_zs, cmap='cool', s=120, marker='^',
            zorder=5, label='Listener grid', depthshade=True
        )
        plt.colorbar(sc, ax=ax_lg, label='Z height (m)', shrink=0.5)

        # Annotate each listener
        for pos in listener_grid:
            ax_lg.text(pos[0]+0.1, pos[1]+0.1, pos[2]+0.1,
                    f'({pos[0]:.1f},{pos[1]:.1f},{pos[2]:.1f})',
                    fontsize=6, color='steelblue')

        # Gunshot source
        ax_lg.scatter([src_x], [src_y], [src_z],
                    c='red', s=250, marker='*', zorder=5, label='Gunshot source')

        # Draw lines from each listener to the wall to show projection direction
        for pos in listener_grid:
            ax_lg.plot([pos[0], wall_x], [pos[1], pos[1]], [pos[2], pos[2]],
                    color='orange', linewidth=0.6, alpha=0.5, linestyle='--')

        ax_lg.set_xlabel('X (m)'); ax_lg.set_ylabel('Y (m)'); ax_lg.set_zlabel('Z (m)')

        if name == 'front':
            ax_lg.set_xlabel('')
            ax_lg.set_xticklabels([])
            ax_lg.set_xticks([])

        if name == 'side':
            ax_lg.set_ylabel('')
            ax_lg.set_yticklabels([])
            ax_lg.set_yticks([])

        if name == 'top':
            ax_lg.set_zlabel('')
            ax_lg.set_zticklabels([])
            ax_lg.set_zticks([])

        ax_lg.set_title(f'Room 1 — 3D Listener Grid — {name.capitalize()} View')
        ax_lg.legend(fontsize=8, loc='upper left')
        ax_lg.set_xlim(0, floor_width)
        ax_lg.set_ylim(0, floor_depth)
        ax_lg.set_zlim(0, floor_height)
        ax_lg.view_init(elev=elev, azim=azim)

        fig_lg.tight_layout()
        fig_lg.savefig(f'{figures_dir}/00b_listener_grid_3d_{name}.png', dpi=150, bbox_inches='tight')
        plt.close(fig_lg)
        print(f"Saved 00b_listener_grid_3d_{name}.png")

    # ── Figure 00c: Illustrative ray-bounce schematic ───────
    # NOTE: this is a simplified specular (billiard-style) bounce off the
    # axis-aligned room boundaries for illustration purposes — it does not
    # draw from GSound-SIR's actual traced ray paths.
    rng = np.random.default_rng(0)
    n_rays, n_bounces = 12, 6

    fig_rb = plt.figure(figsize=(10, 7))
    ax_rb = fig_rb.add_subplot(111, projection='3d')

    draw_room(ax_rb, 0, 0, 0, floor_width, floor_depth, floor_height, 'steelblue')

    yy, zz = np.meshgrid([0, floor_depth], [0, floor_height])
    ax_rb.plot_surface(np.full_like(yy, wall_x), yy, zz, alpha=0.15, color='gray')

    colors = cm.plasma(np.linspace(0, 1, n_rays))
    bounds_min = np.array([0.0, 0.0, 0.0])
    bounds_max = np.array([floor_width, floor_depth, floor_height])

    for i in range(n_rays):
        d = rng.normal(size=3)
        d /= np.linalg.norm(d)

        p = np.array([src_x, src_y, src_z])
        path = [p.copy()]

        for _ in range(n_bounces):
            t_candidates = []
            for axis in range(3):
                if abs(d[axis]) < 1e-9:
                    continue
                if d[axis] > 0:
                    t = (bounds_max[axis] - p[axis]) / d[axis]
                else:
                    t = (bounds_min[axis] - p[axis]) / d[axis]
                if t > 1e-9:
                    t_candidates.append((t, axis))
            if not t_candidates:
                break

            t_hit, hit_axis = min(t_candidates, key=lambda x: x[0])
            p = p + d * t_hit
            path.append(p.copy())

            d = d.copy()
            d[hit_axis] *= -1

        path = np.array(path)
        ax_rb.plot(path[:, 0], path[:, 1], path[:, 2],
                   color=colors[i], linewidth=1.0, alpha=0.8)
        ax_rb.scatter(path[1:-1, 0], path[1:-1, 1], path[1:-1, 2],
                      color=colors[i], s=15, marker='o', alpha=0.6)

    ax_rb.scatter([src_x], [src_y], [src_z], c='red', s=250, marker='*', zorder=5, label='Gunshot source')

    ax_rb.scatter(listener_xs, listener_ys, listener_zs,
                  c='steelblue', s=100, marker='^', zorder=5,
                  edgecolors='black', label='Listener grid')

    ax_rb.set_xlabel('X (m)'); ax_rb.set_ylabel('Y (m)'); ax_rb.set_zlabel('Z (m)')
    ax_rb.set_title(f'Room 1 — Illustrative Ray Bounces ({n_bounces} reflections, {n_rays} rays)')
    ax_rb.legend(fontsize=8, loc='upper left')
    ax_rb.set_xlim(0, floor_width)
    ax_rb.set_ylim(0, floor_depth)
    ax_rb.set_zlim(0, floor_height)
    fig_rb.tight_layout()
    fig_rb.savefig(f'{figures_dir}/00c_ray_bounce_schematic.png', dpi=150, bbox_inches='tight')
    plt.close(fig_rb)
    print("Saved 00c_ray_bounce_schematic.png")


    # ── Figure 1b: 3D Bar Histogram — Wall Energy, saved from multiple angles ───────────

    # h_smooth(2d histo): grid that each cells holds summed/smoothed energy
    # y_center/z_centers: midpoint between edges
    # y_edges/z_edges: boundary lines of grid

    # flatten grid into list of cord and heights
    # get one long list of 800 (y, z) positions and a matching 
    # list of 800 energy values (heights), one per wall cell. 
    y_pos, z_pos = np.meshgrid(y_centers, z_centers, indexing='ij')
    y_pos = y_pos.ravel()
    z_pos = z_pos.ravel()

    # height of each bar
    heights = H_smooth.ravel()

    # x cord to anchor to y and z
    x_base = np.full_like(y_pos, wall_x)

    # width of each bar
    dy = (y_edges[1] - y_edges[0]) * 0.9
    dz = (z_edges[1] - z_edges[0]) * 0.9

    norm = plt.Normalize(vmin=heights.min(), vmax=heights.max())
    colors = cm.viridis(norm(heights))

    # name -> (elev, azim)
    views = {
        'front':      (0,   0),    # looking straight at the wall face (Y-Z plane)
        'side':       (0,  -90),   # looking down the length of the bars (energy axis)
        'top':        (90,  0),    # bird's-eye view, looking straight down
        'angled':     (20, -60),   # your original 3/4 perspective view
    }

    for name, (elev, azim) in views.items():
        fig_bar3d = plt.figure(figsize=(10, 8))
        ax_bar3d = fig_bar3d.add_subplot(111, projection='3d')

        ax_bar3d.bar3d(x_base, y_pos, z_pos, heights, dy, dz, color=colors, shade=True)

        ax_bar3d.set_xlabel('Energy (into room, +X)')
        ax_bar3d.set_ylabel('Y (m, along wall)')
        ax_bar3d.set_zlabel('Z (m, height)')

        if name == 'front':
            ax_bar3d.set_xlabel('')
            ax_bar3d.set_xticklabels([])
            ax_bar3d.set_xticks([])

        if name == 'side':
            ax_bar3d.set_ylabel('')
            ax_bar3d.set_yticklabels([])
            ax_bar3d.set_yticks([])

        if name == 'top':
            ax_bar3d.set_zlabel('')
            ax_bar3d.set_zticklabels([])
            ax_bar3d.set_zticks([])


        ax_bar3d.set_title(f'Wall Energy Histogram — {name.capitalize()} View')

        ax_bar3d.view_init(elev=elev, azim=azim)

        mappable = cm.ScalarMappable(norm=norm, cmap='viridis')
        mappable.set_array(heights)
        fig_bar3d.colorbar(mappable, ax=ax_bar3d, shrink=0.6, pad=0.1, label='Energy')

        fig_bar3d.tight_layout()
        fig_bar3d.savefig(f'{figures_dir}/01b_wall_energy_bar3d_{name}.png', dpi=150, bbox_inches='tight')
        plt.close(fig_bar3d)
        print(f"Saved {figures_dir}/01b_wall_energy_bar3d_{name}.png")


    # ── Figure 1c: Room 2 — Estimated DOA visualization ───────────────────
    arrow_length = 1.5  # meters, just for visibility — scales the unit direction vector

    listener_pos = np.array(r2_listener_global)
    arrow_end = listener_pos + direction * arrow_length

    views = {
        'top':    (90,  0),
        'angled': (20, -60),
    }

    for name, (elev, azim) in views.items():
        fig_doa = plt.figure(figsize=(10, 8))
        ax_doa = fig_doa.add_subplot(111, projection='3d')

        draw_room(ax_doa, 0,          0, 0, floor_width, floor_depth, floor_height, 'steelblue', label='Room 1')
        draw_room(ax_doa, floor_width, 0, 0, room2_width, floor_depth, floor_height, 'coral',    label='Room 2')

        # source in Room 1
        ax_doa.scatter([src_x], [src_y], [src_z], c='red', s=200, marker='*', zorder=5, label='Source')

        # Room 2 listener/mic position
        ax_doa.scatter(*listener_pos, c='coral', s=150, marker='^',
                        edgecolors='black', zorder=5, label='Mic Room 2')

        # DOA arrow — points FROM the listener TOWARD where the sound is estimated to be coming from
        ax_doa.quiver(
            listener_pos[0], listener_pos[1], listener_pos[2],
            direction[0], direction[1], direction[2],
            length=arrow_length, color='green', linewidth=2.5,
            arrow_length_ratio=0.25, zorder=6, label='Estimated DOA'
        )

        ax_doa.set_xlabel('X (m)'); ax_doa.set_ylabel('Y (m)'); ax_doa.set_zlabel('Z (m)')


        if name == 'top':
            ax_doa.set_zlabel('')
            ax_doa.set_zticklabels([])
            ax_doa.set_zticks([])


        ax_doa.set_title(
            f'Room 2 — Estimated Direction of Arrival — {name.capitalize()} View\n'
            f'Az={np.degrees(az_mean):.1f}°, El={np.degrees(el_mean):.1f}°'
        )
        ax_doa.legend(fontsize=7, loc='upper left')
        ax_doa.view_init(elev=elev, azim=azim)

        fig_doa.tight_layout()
        fig_doa.savefig(f'{figures_dir}/01c_room2_doa_{name}.png', dpi=150, bbox_inches='tight')
        plt.close(fig_doa)
        print(f"Saved {figures_dir}/01c_room2_doa_{name}.png")

    # ── Figure: Room 2 ray-bounce schematic (full building context) ──
    # NOTE: illustrative specular bounce, confined to Room 2's bounding
    # box only. Room 1 is drawn for spatial context but has no rays here.
    rng = np.random.default_rng(1)
    n_rays_per_source = 4
    n_bounces = 5

    # Room 2 bounds in GLOBAL coordinates (Room 2 sits adjacent to Room 1,
    # offset by wall_x — same convention as r2_listener_global)
    room2_bounds_min = np.array([wall_x, 0.0, 0.0])
    room2_bounds_max = np.array([wall_x + room2_width, floor_depth, floor_height])

    fig_rb2 = plt.figure(figsize=(12, 8))
    ax_rb2 = fig_rb2.add_subplot(111, projection='3d')

    draw_room(ax_rb2, 0,          0, 0, floor_width,  floor_depth, floor_height, 'steelblue', label='Room 1')
    draw_room(ax_rb2, wall_x,     0, 0, room2_width,   floor_depth, floor_height, 'coral',     label='Room 2')

    colors_rb2 = cm.viridis(np.linspace(0, 1, len(vsrc_positions_local)))

    for src_idx, vsrc_local in enumerate(vsrc_positions_local):
        # convert local Room 2 virtual source position to global coordinates
        origin = np.array([wall_x + vsrc_local[0], vsrc_local[1], vsrc_local[2]])

        for r in range(n_rays_per_source):
            d = rng.normal(size=3)
            d /= np.linalg.norm(d)

            p = origin.copy()
            path = [p.copy()]

            for _ in range(n_bounces):
                t_candidates = []
                for axis in range(3):
                    if abs(d[axis]) < 1e-9:
                        continue
                    if d[axis] > 0:
                        t = (room2_bounds_max[axis] - p[axis]) / d[axis]
                    else:
                        t = (room2_bounds_min[axis] - p[axis]) / d[axis]
                    if t > 1e-9:
                        t_candidates.append((t, axis))
                if not t_candidates:
                    break

                t_hit, hit_axis = min(t_candidates, key=lambda x: x[0])
                p = p + d * t_hit
                path.append(p.copy())

                d = d.copy()
                d[hit_axis] *= -1

            path = np.array(path)
            ax_rb2.plot(path[:, 0], path[:, 1], path[:, 2],
                        color=colors_rb2[src_idx], linewidth=1.0, alpha=0.8)
            ax_rb2.scatter(path[1:-1, 0], path[1:-1, 1], path[1:-1, 2],
                           color=colors_rb2[src_idx], s=25, marker='o', alpha=0.9,
                           edgecolors='black', linewidths=0.3, zorder=6)

    # virtual source positions (global)
    vsrc_global = np.array([[wall_x + v[0], v[1], v[2]] for v in vsrc_positions_local])
    ax_rb2.scatter(vsrc_global[:, 0], vsrc_global[:, 1], vsrc_global[:, 2],
                   c='red', s=100, marker='*', zorder=5, label='Virtual sources')

    # Room 2 listener
    r2_listener_global = (wall_x + room2_width * 0.5, floor_depth * 0.5, floor_height * 0.5)
    ax_rb2.scatter(*r2_listener_global, c='black', s=150, marker='^',
                   edgecolors='white', zorder=5, label='Mic Room 2')

    ax_rb2.set_xlabel('X (m)'); ax_rb2.set_ylabel('Y (m)'); ax_rb2.set_zlabel('Z (m)')
    ax_rb2.set_title('Room 2 — Illustrative Ray Bounces from Virtual Sources')
    ax_rb2.legend(fontsize=7, loc='upper left')
    ax_rb2.set_xlim(0, wall_x + room2_width)
    ax_rb2.set_ylim(0, floor_depth)
    ax_rb2.set_zlim(0, floor_height)
    ax_rb2.view_init(elev=20, azim=-60)

    fig_rb2.tight_layout()
    fig_rb2.savefig(f'{figures_dir}/01d_room2_ray_bounce_schematic.png', dpi=150, bbox_inches='tight')
    plt.close(fig_rb2)
    print("Saved 01d_room2_ray_bounce_schematic.png")



    # ── Figure 1e: 3D scatter of raw wall impact points ─────
    fig_wi = plt.figure(figsize=(10, 8))
    ax_wi = fig_wi.add_subplot(111, projection='3d')

    draw_room(ax_wi, 0,          0, 0, floor_width, floor_depth, floor_height, 'steelblue', label='Room 1')
    draw_room(ax_wi, floor_width, 0, 0, room2_width, floor_depth, floor_height, 'coral',    label='Room 2')

    x_impact_raw = np.full_like(iy_all, wall_x)

    sc = ax_wi.scatter(
        x_impact_raw, iy_all, iz_all,
        c=hardness, cmap='hot', s=8, alpha=0.4,
        zorder=5, label='Wall impacts (raw)'
    )
    cbar = fig_wi.colorbar(sc, ax=ax_wi, shrink=0.6, pad=0.1)
    cbar.set_label('Impact energy (all 8 bands)')

    ax_wi.scatter([src_x], [src_y], [src_z], c='blue', s=200, marker='*', zorder=6, label='Source')

    ax_wi.set_xlabel('X (m)'); ax_wi.set_ylabel('Y (m)'); ax_wi.set_zlabel('Z (m)')
    ax_wi.set_title('3D Wall Impact Points (Raw, Pre-Clustering)')
    ax_wi.legend(fontsize=8, loc='upper left')
    ax_wi.set_xlim(0, floor_width + room2_width)
    ax_wi.set_ylim(0, floor_depth)
    ax_wi.set_zlim(0, floor_height)
    ax_wi.view_init(elev=20, azim=-60)

    fig_wi.tight_layout()
    fig_wi.savefig(f'{figures_dir}/01e_wall_impact_points_3d.png', dpi=150, bbox_inches='tight')
    plt.close(fig_wi)
    print("Saved 01e_wall_impact_points_3d.png")

    


    # ── Figure 4: pyva TL curve ─────────────────────────────
    fig4, ax4 = plt.subplots(figsize=(8, 5))
    ax4.semilogx(frequencies, TL_db, 'o-', color='purple', linewidth=2, markersize=8)
    ax4.fill_between(frequencies, TL_db, TL_db[-1]*1.1, alpha=0.2, color='purple')
    ax4.set_xlabel('Frequency (Hz)'); ax4.set_ylabel('Transmission Loss (dB)')
    ax4.set_title(f'Transmission Loss — {material_chosen}')
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

    print(f"\nAll figures saved to {figures_dir}/")


def per_band_energy_diagnostic(sir_room1, sir_room2, fs, frequencies, tau):
    """
    Breaks down IR energy into per-band buckets (via FFT) for both rooms,
    so we can see which specific band(s) are leaking more energy than
    the material's tau values should allow.
    """
    n_samples = sir_room1.shape[1]
    freqs = np.fft.rfftfreq(n_samples, d=1/fs)

    # define band edges as midpoints between adjacent center frequencies
    # (in log space, since these are octave-ish bands)
    log_f = np.log2(frequencies)
    edges = np.concatenate([
        [log_f[0] - (log_f[1]-log_f[0])/2],
        (log_f[:-1] + log_f[1:]) / 2,
        [log_f[-1] + (log_f[-1]-log_f[-2])/2]
    ])
    band_edges_hz = 2 ** edges  # back to linear Hz

    print(f"\n{'Band':>8} {'Range (Hz)':>18} {'Room1 E':>12} {'Room2 E':>12} {'Sim ratio':>12} {'Sim TL(dB)':>10} {'Theory TL(dB)':>13}")
    print("-" * 90)

    for i, f_center in enumerate(frequencies):
        lo, hi = band_edges_hz[i], band_edges_hz[i+1]
        mask = (freqs >= lo) & (freqs < hi)

        e1 = 0.0
        e2 = 0.0
        for ch in range(sir_room1.shape[0]):
            spec1 = np.fft.rfft(sir_room1[ch])
            spec2 = np.fft.rfft(sir_room2[ch])
            e1 += np.sum(np.abs(spec1[mask])**2)
            e2 += np.sum(np.abs(spec2[mask])**2)

        sim_ratio = e2 / (e1 + 1e-30)
        sim_tl = -10 * np.log10(sim_ratio + 1e-30)
        theory_tl = -10 * np.log10(tau[i] + 1e-30)

        print(f"{f_center:>8.0f} {lo:>8.1f}-{hi:<8.1f} {e1:>12.4e} {e2:>12.4e} {sim_ratio:>12.4e} {sim_tl:>10.2f} {theory_tl:>13.2f}")

    # broadband totals for reference
    e1_total = np.sum(sir_room1**2)
    e2_total = np.sum(sir_room2**2)
    print("-" * 90)
    print(f"{'TOTAL':>8} {'':>18} {e1_total:>12.4e} {e2_total:>12.4e} "
          f"{e2_total/e1_total:>12.4e} {-10*np.log10(e2_total/e1_total):>10.2f}")
    

def main():

    # Gets transmission loss values
    TL_db, tau = get_TL_diffuse(frequencies, thickness=thick, material=material_chosen)

    # simulate room 1
    parquet_path = room1_simulate()
    df = pd.read_parquet(parquet_path)

    # generates histogram per listner on grid
    gen_histogram(df)
    # set up auralizer for IR 1
    demo, df_full = load_auralizer(df, parquet_path)
    # get room 1 rir
    sir_room1, auralizer = gen_IR(demo)
    # get wall impact points
    wall_rays, band_cols, df_with_energy = gen_wall_impacts(df_full)
    # cluster into virtual sources based on weighted avergae of cluster
    room2_listener_local, vsrc_positions_local, impact_bands, impact_y, impact_z,  y_edges, z_edges, H_smooth, impact_energy, iy_all, iz_all, hardness = cluster_vsc(wall_rays, band_cols)
    # simulate room 2
    data_room2, parquet_room2 = room2_simulate(room2_listener_local, vsrc_positions_local, impact_bands, tau)
    # build IR for room 2 and handle DOA
    sir_room2, all_intensities, all_delays, direction, az_mean, el_mean = doa_handling(data_room2, impact_y, impact_z, auralizer, vsrc_positions_local, room2_listener_local)
    # convolve room ir with dry gunshot
    room1_recording, room2_recording, fs = convolveSignal(sir_room1, sir_room2)

    per_band_energy_diagnostic(sir_room1, sir_room2, sample_rate, frequencies, tau)

    # generates metrics and figures
    generate_metrics(
                    df_with_energy, 
                    wall_rays,
                    vsrc_positions_local,
                    all_delays,
                    sir_room1,
                    sir_room2,
                    room1_recording,
                    room2_recording,
                    parquet_room2,
                    TL_db,
                    all_intensities    
                    )

    generate_figures(
                    y_edges, 
                    z_edges, 
                    H_smooth, 
                    room1_recording, 
                    room2_recording,
                    TL_db,
                    impact_energy,
                    impact_y, 
                    impact_z,
                    direction, 
                    az_mean, 
                    el_mean,
                    fs,
                    vsrc_positions_local,
                    iy_all, 
                    iz_all, 
                    hardness

                    )

if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed = time.time() - start_time
    print(f"\nTotal runtime: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
