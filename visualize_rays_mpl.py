import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ── Room config ───────────────────────────────────────────────────────────────
ROOM_W   = 5.0
ROOM_D   = 5.0

FLOOR1_Z = 0.0
SLAB_BOT = 2.45
SLAB_TOP = 2.5
CEIL2_Z  = 5.0

SRC_X, SRC_Y, SRC_Z = 2.5, 2.5, 4.0

N_RAYS       = 50    # total room 2 rays
N_FLOOR_RAYS = 5     # rays that punch through slab
N_R1_RAYS    = 8     # rays per virtual source in room 1

FPS      = 10        # slower fps for dampening effect
DURATION = 16.0
DPI      = 120
SAVE     = 'floors.gif'

# ── Geometry ──────────────────────────────────────────────────────────────────

def draw_room_wire(ax, w, d, z_bot, z_top, color, alpha=0.28, lw=1.0):
    corners = np.array([
        [0,0,z_bot],[w,0,z_bot],[w,d,z_bot],[0,d,z_bot],
        [0,0,z_top],[w,0,z_top],[w,d,z_top],[0,d,z_top],
    ])
    for a,b in [(0,1),(1,2),(2,3),(3,0),
                (4,5),(5,6),(6,7),(7,4),
                (0,4),(1,5),(2,6),(3,7)]:
        ax.plot(*zip(corners[a], corners[b]),
                color=color, alpha=alpha, linewidth=lw)

def draw_plane(ax, w, d, z, color, alpha=0.25):
    verts = [[[0,0,z],[w,0,z],[w,d,z],[0,d,z]]]
    ax.add_collection3d(Poly3DCollection(
        verts, alpha=alpha, facecolor=color,
        edgecolor='#556677', linewidth=0.6))

def draw_slab(ax, w, d, z_bot, z_top):
    z = (z_bot + z_top) / 2
    corners = np.array([[0,0,z],[w,0,z],[w,d,z],[0,d,z],[0,0,z]])
    ax.plot(corners[:,0], corners[:,1], corners[:,2],
            color='#ffffff', alpha=0.6, linewidth=2.0)
    ax.plot([0,w],[0,d],[z,z], color='#ffffff', alpha=0.3, linewidth=0.8)
    ax.plot([w,0],[0,d],[z,z], color='#ffffff', alpha=0.3, linewidth=0.8)

# ── Ray tracer ────────────────────────────────────────────────────────────────

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def trace_ray(ox, oy, oz, dx, dy, dz, z_min, z_max, max_bounces=6):
    mag = np.sqrt(dx**2 + dy**2 + dz**2) + 1e-12
    dx, dy, dz = dx/mag, dy/mag, dz/mag
    max_len = 1.5 * np.sqrt(ROOM_W**2 + ROOM_D**2 + (z_max-z_min)**2)
    pts   = [(ox, oy, oz)]
    cx, cy, cz = ox, oy, oz
    total = 0.0
    for _ in range(max_bounces):
        t_list = []
        if dx >  1e-9: t_list.append(((ROOM_W - cx)/dx,   'xmax'))
        if dx < -1e-9: t_list.append((cx/(-dx),            'xmin'))
        if dy >  1e-9: t_list.append(((ROOM_D - cy)/dy,   'ymax'))
        if dy < -1e-9: t_list.append((cy/(-dy),            'ymin'))
        if dz >  1e-9: t_list.append(((z_max  - cz)/dz,   'zmax'))
        if dz < -1e-9: t_list.append(((cz - z_min)/(-dz), 'zmin'))
        t_list = [(t,s) for t,s in t_list if t > 1e-6]
        if not t_list:
            break
        t, surface = min(t_list, key=lambda x: x[0])
        if total + t > max_len:
            t = max_len - total
            pts.append((clamp(cx+dx*t,0,ROOM_W),
                        clamp(cy+dy*t,0,ROOM_D),
                        clamp(cz+dz*t,z_min,z_max)))
            break
        nx = clamp(cx+dx*t,0,ROOM_W)
        ny = clamp(cy+dy*t,0,ROOM_D)
        nz = clamp(cz+dz*t,z_min,z_max)
        pts.append((nx,ny,nz))
        total += t
        cx,cy,cz = nx,ny,nz
        if surface in ('xmin','xmax'): dx=-dx
        if surface in ('ymin','ymax'): dy=-dy
        if surface in ('zmin','zmax'): dz=-dz
    return pts

def generate_r2_rays(seed=42):
    rng  = np.random.default_rng(seed)
    rays = []
    for _ in range(N_RAYS):
        theta = rng.uniform(0, np.pi)
        phi   = rng.uniform(0, 2*np.pi)
        dx = np.sin(theta)*np.cos(phi)
        dy = np.sin(theta)*np.sin(phi)
        dz = np.cos(theta)
        rays.append(trace_ray(SRC_X, SRC_Y, SRC_Z,
                               dx, dy, dz,
                               SLAB_TOP, CEIL2_Z))
    return rays

def pick_floor_impact_points(seed=7):
    """
    Pick 5 evenly spread impact points on the slab.
    These are where rays 'punch through' the slab.
    """
    rng = np.random.default_rng(seed)
    pts = []
    # spread them across the floor using a simple grid-ish scatter
    xs = [1.0, 4.0, 2.5, 1.2, 3.8]
    ys = [1.0, 1.5, 2.5, 3.8, 3.5]
    for x, y in zip(xs, ys):
        pts.append((x, y))
    return pts

def make_transmission_path(src_x, src_y, src_z, impact_x, impact_y):
    """Straight line from source down to slab impact point."""
    return [
        (src_x,    src_y,    src_z),
        (impact_x, impact_y, SLAB_TOP),
        (impact_x, impact_y, SLAB_BOT),   # through slab
    ]

def generate_r1_rays(impact_x, impact_y, seed=0):
    """
    From each floor impact point, shoot N_R1_RAYS downward into room 1.
    Rays are dampened — shorter path, fewer bounces than room 2.
    """
    rng  = np.random.default_rng(seed + int(impact_x*100))
    rays = []
    for _ in range(N_R1_RAYS):
        # downward biased but spread in all horizontal directions
        theta = rng.uniform(np.pi*0.15, np.pi*0.85)
        phi   = rng.uniform(0, 2*np.pi)
        dx = np.sin(theta)*np.cos(phi)
        dy = np.sin(theta)*np.sin(phi)
        dz = -abs(np.cos(theta))   # always starts downward
        rays.append(trace_ray(impact_x, impact_y, SLAB_BOT,
                               dx, dy, dz,
                               FLOOR1_Z, SLAB_BOT,
                               max_bounces=3))   # fewer bounces = dampened
    return rays


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Generating rays...")
    r2_rays       = generate_r2_rays()
    impact_pts    = pick_floor_impact_points()
    trans_paths   = [make_transmission_path(SRC_X, SRC_Y, SRC_Z, ix, iy)
                     for ix, iy in impact_pts]
    r1_rays_all   = [generate_r1_rays(ix, iy, seed=i)
                     for i, (ix, iy) in enumerate(impact_pts)]

    total_frames = int(FPS * DURATION)

    fig = plt.figure(figsize=(12, 9), facecolor='#0a0a0f')
    ax  = fig.add_subplot(111, projection='3d')
    fig.patch.set_facecolor('#0a0a0f')
    ax.set_facecolor('#0a0a0f')

    # ── Static scene ──────────────────────────────────────────────────────────
    draw_room_wire(ax, ROOM_W, ROOM_D, FLOOR1_Z, SLAB_BOT,
                   '#00cfff', alpha=0.18, lw=0.8)
    draw_plane(ax, ROOM_W, ROOM_D, FLOOR1_Z, '#112233', alpha=0.20)
    ax.text(ROOM_W/2, ROOM_D*0.85, FLOOR1_Z+0.1,
            'FLOOR 1', color='#00cfff', fontsize=9,
            ha='center', alpha=0.55, fontweight='bold')

    draw_slab(ax, ROOM_W, ROOM_D, SLAB_BOT, SLAB_TOP)
    ax.text(ROOM_W/2, ROOM_D/2, SLAB_BOT-0.1,
            'CONCRETE SLAB', color='#aabbcc', fontsize=7,
            ha='center', va='top', alpha=0.80)

    draw_room_wire(ax, ROOM_W, ROOM_D, SLAB_TOP, CEIL2_Z,
                   '#ffcc00', alpha=0.35, lw=1.1)
    draw_plane(ax, ROOM_W, ROOM_D, CEIL2_Z, '#223344', alpha=0.15)
    ax.text(ROOM_W/2, ROOM_D*0.85, CEIL2_Z-0.15,
            'FLOOR 2', color='#ffcc00', fontsize=9,
            ha='center', alpha=0.75, fontweight='bold')

    # Gunshot
    ax.scatter([SRC_X],[SRC_Y],[SRC_Z], color='red', s=250, zorder=10)
    ax.text(SRC_X+0.15, SRC_Y, SRC_Z+0.18,
            'GUNSHOT', color='red', fontsize=10, fontweight='bold')

    ax.set_xlabel('X (m)', color='#aaaaaa', labelpad=6)
    ax.set_ylabel('Y (m)', color='#aaaaaa', labelpad=6)
    ax.set_zlabel('Z (m)', color='#aaaaaa', labelpad=6)
    ax.tick_params(colors='#aaaaaa', labelsize=7)
    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.fill = False
        pane.set_edgecolor('#1a1a2a')
    ax.set_xlim(0, ROOM_W)
    ax.set_ylim(0, ROOM_D)
    ax.set_zlim(FLOOR1_Z, CEIL2_Z)
    ax.view_init(elev=20, azim=-50)

    title = ax.set_title('', color='white', fontsize=11, pad=10)

    # ── Stage boundaries ──────────────────────────────────────────────────────
    # Stage 1  0.00 – 0.30  : room 2 rays appear
    # Stage 2  0.30 – 0.48  : 5 transmission rays punch through slab (slow)
    # Stage 3  0.48 – 0.62  : red virtual sources appear in room 1
    # Stage 4  0.62 – 0.85  : room 1 yellow rays spread out (dampened, slow)
    # Orbit    0.85 – 1.00  : camera orbit

    S = dict(r2=0.30, trans=0.48, vsrc=0.62, r1=0.85)

    drawn_r2    = []
    drawn_trans = []
    vsrc_scat   = [None]
    drawn_r1    = []

    def update(frame):
        t = frame / total_frames

        # ── Stage 1: Room 2 rays ─────────────────────────────────────────────
        if t <= S['r2']:
            prog   = t / S['r2']
            n_show = int(prog * N_RAYS)
            while len(drawn_r2) < n_show:
                i   = len(drawn_r2)
                pts = r2_rays[i]
                xs  = [p[0] for p in pts]
                ys  = [p[1] for p in pts]
                zs  = [p[2] for p in pts]
                ln, = ax.plot(xs, ys, zs,
                              color='#ff4400', alpha=0.65, linewidth=0.9)
                drawn_r2.append(ln)
            title.set_text(
                f'Stage 1 — Room 2 rays propagating  ({len(drawn_r2)}/{N_RAYS})')

        # ── Stage 2: 5 rays punch slowly through slab ────────────────────────
        elif t <= S['trans']:
            prog  = (t - S['r2']) / (S['trans'] - S['r2'])
            # reveal fractional segments so ray appears to travel slowly
            n_full = int(prog * N_FLOOR_RAYS)   # fully drawn transmission rays

            # draw complete ones
            while len(drawn_trans) < n_full:
                i    = len(drawn_trans)
                path = trans_paths[i]
                xs   = [p[0] for p in path]
                ys   = [p[1] for p in path]
                zs   = [p[2] for p in path]
                ln, = ax.plot(xs, ys, zs,
                              color='#ff4400', alpha=0.95,
                              linewidth=2.0, linestyle='-')
                drawn_trans.append(ln)

            # animate the next one segment by segment
            if n_full < N_FLOOR_RAYS:
                seg_prog = (prog * N_FLOOR_RAYS) - n_full
                path     = trans_paths[n_full]
                # interpolate along path
                total_pts = len(path)
                idx_f     = seg_prog * (total_pts - 1)
                idx_i     = int(idx_f)
                frac      = idx_f - idx_i
                sub_pts   = path[:idx_i+1]
                if idx_i + 1 < total_pts:
                    px = path[idx_i][0] + frac*(path[idx_i+1][0]-path[idx_i][0])
                    py = path[idx_i][1] + frac*(path[idx_i+1][1]-path[idx_i][1])
                    pz = path[idx_i][2] + frac*(path[idx_i+1][2]-path[idx_i][2])
                    sub_pts = sub_pts + [(px, py, pz)]
                if len(sub_pts) >= 2:
                    xs = [p[0] for p in sub_pts]
                    ys = [p[1] for p in sub_pts]
                    zs = [p[2] for p in sub_pts]
                    ax.plot(xs, ys, zs, color='#ff4400',
                            alpha=0.95, linewidth=2.0)

            title.set_text(
                f'Stage 2 — Energy punching through concrete slab  '
                f'({min(n_full+1, N_FLOOR_RAYS)}/{N_FLOOR_RAYS})')

       # ── Stage 3: Red virtual sources appear in room 1 ────────────────────
        elif t <= S['vsrc']:
            # hide all room 2 rays for clean visualization
            for ln in drawn_r2:
                ln.set_visible(False)
            for ln in drawn_trans:
                ln.set_visible(False)
        # ── Stage 4: Room 1 yellow rays, dampened (slow, fewer bounces) ──────
        elif t <= S['r1']:
            prog       = (t - S['vsrc']) / (S['r1'] - S['vsrc'])
            total_r1   = N_FLOOR_RAYS * N_R1_RAYS
            n_show     = int(prog * total_r1)
            while len(drawn_r1) < n_show:
                i       = len(drawn_r1)
                src_idx = i // N_R1_RAYS
                ray_idx = i  % N_R1_RAYS
                if src_idx < len(r1_rays_all):
                    pts = r1_rays_all[src_idx][ray_idx]
                    xs  = [p[0] for p in pts]
                    ys  = [p[1] for p in pts]
                    zs  = [p[2] for p in pts]
                    # yellow, more transparent = dampened energy
                    ln, = ax.plot(xs, ys, zs,
                                  color='#ffee00',
                                  alpha=0.45,
                                  linewidth=0.85)
                    drawn_r1.append(ln)
            title.set_text(
                f'Stage 4 — Dampened re-radiation in Room 1  '
                f'(yellow = attenuated energy)  ({len(drawn_r1)}/{total_r1})')

        # ── Orbit ─────────────────────────────────────────────────────────────
        else:
            prog = (t - S['r1']) / (1.0 - S['r1'])
            ax.view_init(elev=20, azim=-50 + prog*300)
            title.set_text('Multi-floor transmission simulation — complete')

        return []

    ani = animation.FuncAnimation(
        fig, update, frames=total_frames,
        interval=1000/FPS, blit=False)

    print(f"Rendering → {SAVE}  ({total_frames} frames @ {FPS}fps)...")
    writer = animation.PillowWriter(fps=FPS)
    ani.save(SAVE, writer=writer, dpi=DPI,
             savefig_kwargs={'facecolor': '#0a0a0f'})
    print("Done.")

if __name__ == '__main__':
    main()