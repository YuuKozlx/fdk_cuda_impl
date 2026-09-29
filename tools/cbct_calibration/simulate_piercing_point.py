# -*- coding: utf-8 -*-
"""Recover the principal point from radial-pair (phi, phi+pi) lines.

Phantom:
  - Upper ring: (r cos phi_k, r sin phi_k, +z0), k = 0..n-1
  - Lower ring: (r cos phi_k, r sin phi_k, -z0), same phi
  - Radial pair: upper phi_k  <->  lower phi_k+pi
    (equivalently upper k <-> lower (k + n/2) % n)
  Each such line passes through the world origin.
  Projected, all these lines intersect at the projection of the origin,
  which is the principal point.

Run:
    python simulate_piercing_point.py
"""

from __future__ import annotations

import numpy as np
import matplotlib
import matplotlib.pyplot as plt


# =====================================================================
# 1. Phantom
# =====================================================================
def make_two_ring_phantom(radius_mm=30.0, n_per_ring=8,
                          z_top_mm=25.0, z_bottom_mm=-25.0):
    """上圈、下圈同 phi 分布。径向对由配对方式决定。"""
    ang = np.linspace(0, 2 * np.pi, n_per_ring, endpoint=False)
    top = np.column_stack([radius_mm * np.cos(ang),
                           radius_mm * np.sin(ang),
                           np.full(n_per_ring, z_top_mm)])
    bottom = np.column_stack([radius_mm * np.cos(ang),
                              radius_mm * np.sin(ang),
                              np.full(n_per_ring, z_bottom_mm)])
    return np.vstack([top, bottom])


def rotation_xyz(rx, ry, rz):
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def apply_rigid(beads_body, R, t):
    return (R @ beads_body.T).T + t


# =====================================================================
# 2. Projection
# =====================================================================
def rotation_eta(eta_deg):
    t = np.deg2rad(eta_deg)
    return np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])


def project(sdd, sod, u0, v0, eta_deg, beads_world, angles_rad, pixel_size_mm):
    beads = np.asarray(beads_world, dtype=float)
    c, s = np.cos(angles_rad)[:, None], np.sin(angles_rad)[:, None]
    x = c * beads[:, 0] - s * beads[:, 1]
    y = s * beads[:, 0] + c * beads[:, 1]
    denom = sod + x
    if np.any(denom <= 0):
        raise ValueError("bead behind source plane")
    aligned = sdd * np.stack([-y / denom,
                              beads[:, 2][None, :] / denom], axis=-1)
    return (aligned @ rotation_eta(eta_deg)) / pixel_size_mm + [u0, v0]


# =====================================================================
# 3. Radial-pair intersection
# =====================================================================
def line_from_two_points(p1, p2):
    """归一化直线 (a, b, c)，a²+b²=1，a·u+b·v+c=0。"""
    (u1, v1), (u2, v2) = p1, p2
    a = v2 - v1
    b = u1 - u2
    c = u2 * v1 - u1 * v2
    norm = np.hypot(a, b)
    if norm < 1e-12:
        return None
    return np.array([a / norm, b / norm, c / norm])


def least_squares_intersection(lines):
    """线性最小二乘交点。lines: (n, 3)。"""
    A = lines[:, :2]
    c = lines[:, 2]
    sol, _, _, _ = np.linalg.lstsq(A, -c, rcond=None)
    return sol


def line_residuals(lines, point):
    return np.abs(lines @ np.r_[point, 1.0])


def piercing_point_one_view(uv_top, uv_bot):
    """径向对：上圈 k ↔ 下圈 (k + n/2) % n。"""
    n = len(uv_top)
    half = n // 2
    lines = []
    for k in range(half):         # 只做一半，避免重复
        L = line_from_two_points(uv_top[k], uv_bot[(k + half) % n])
        if L is not None:
            lines.append(L)
    if len(lines) < 2:
        raise ValueError("need >= 2 lines")
    lines = np.array(lines)
    Pc = least_squares_intersection(lines)
    res = line_residuals(lines, Pc)
    return Pc, res, lines


# =====================================================================
# 4. Simulation
# =====================================================================
def simulate(*, n_views=360, noise_px=0.1, seed=42,
             sdd=770.0, sod=440.0, u0=499.5, v0=535.5, eta_deg=1.0,
             pixel_size_mm=0.417,
             rigid_rx_deg=0.0, rigid_ry_deg=0.0, rigid_rz_deg=0.0,
             rigid_tx_mm=0.0, rigid_ty_mm=0.0, rigid_tz_mm=0.0):
    beads_body = make_two_ring_phantom()
    n_beads = len(beads_body)
    n_per_ring = n_beads // 2

    R_rigid = rotation_xyz(np.deg2rad(rigid_rx_deg),
                           np.deg2rad(rigid_ry_deg),
                           np.deg2rad(rigid_rz_deg))
    t_rigid = np.array([rigid_tx_mm, rigid_ty_mm, rigid_tz_mm])
    beads_world = apply_rigid(beads_body, R_rigid, t_rigid)

    angles = np.arange(n_views) * (2 * np.pi / n_views)
    pixel_size = np.array([pixel_size_mm, pixel_size_mm])
    tracks = project(sdd, sod, u0, v0, eta_deg,
                     beads_world, angles, pixel_size)
    rng = np.random.default_rng(seed)
    if noise_px > 0:
        tracks = tracks + rng.normal(0, noise_px, tracks.shape)

    Pc_list = []
    res_list = []
    for i in range(n_views):
        try:
            Pc, res, _ = piercing_point_one_view(
                tracks[i, :n_per_ring], tracks[i, n_per_ring:])
            Pc_list.append(Pc)
            res_list.append(res)
        except Exception as exc:
            print(f"[view {i}] failed: {exc}")
    Pc_list = np.array(Pc_list)
    Pc_median = np.median(Pc_list, axis=0)

    return {
        "tracks": tracks,
        "angles": angles,
        "pixel_size": pixel_size,
        "n_per_ring": n_per_ring,
        "n_views": n_views,
        "Pc_per_view": Pc_list,
        "res_per_view": res_list,
        "Pc_median": Pc_median,
        "truth": dict(u0_px=u0, v0_px=v0, sdd=sdd, sod=sod, eta_deg=eta_deg,
                      beads_world=beads_world, beads_body=beads_body,
                      R=R_rigid, t=t_rigid),
    }


# =====================================================================
# 5. Visualization
# =====================================================================
def visualize(sim):
    tracks = sim["tracks"]
    n_per_ring = sim["n_per_ring"]
    Pc_per_view = sim["Pc_per_view"]
    Pc_median = sim["Pc_median"]
    truth = sim["truth"]
    u0, v0 = truth["u0_px"], truth["v0_px"]
    n_views = sim["n_views"]

    fig = plt.figure(figsize=(18, 10))

    margin = 80
    u_min = min(tracks[..., 0].min() - margin, Pc_median[0] - 20)
    u_max = max(tracks[..., 0].max() + margin, Pc_median[0] + 20)
    v_min = min(tracks[..., 1].min() - margin, Pc_median[1] - 20)
    v_max = max(tracks[..., 1].max() + margin, Pc_median[1] + 20)

    def draw_view(ax, v, title):
        uv_top = tracks[v, :n_per_ring]
        uv_bot = tracks[v, n_per_ring:]
        ax.scatter(uv_top[:, 0], uv_top[:, 1], c="tab:blue", s=55,
                   marker="o", label="top ring",
                   edgecolors="k", linewidths=0.5)
        ax.scatter(uv_bot[:, 0], uv_bot[:, 1], c="tab:cyan", s=55,
                   marker="o", label="bottom ring",
                   edgecolors="k", linewidths=0.5)
        # 径向对连线：上圈 k -> 下圈 (k+half)%n
        half = n_per_ring // 2
        for k in range(n_per_ring):
            j = (k + half) % n_per_ring
            ax.plot([uv_top[k, 0], uv_bot[j, 0]],
                    [uv_top[k, 1], uv_bot[j, 1]],
                    color="gray", lw=0.9, alpha=0.6)
        Pc, res, _ = piercing_point_one_view(uv_top, uv_bot)
        ax.scatter([Pc[0]], [Pc[1]], c="red", s=160, marker="*",
                   label=f"Pc ({Pc[0]:.3f}, {Pc[1]:.3f})",
                   edgecolors="k", linewidths=0.5, zorder=5)
        ax.scatter([u0], [v0], c="lime", s=110, marker="x",
                   label=f"truth ({u0:.3f}, {v0:.3f})",
                   linewidths=2.5, zorder=6)
        ax.set_xlabel("u (px)")
        ax.set_ylabel("v (px)")
        ax.set_title(title)
        ax.set_xlim(u_min, u_max)
        ax.set_ylim(v_max, v_min)
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="best")

    ax00 = fig.add_subplot(2, 3, 1)
    draw_view(ax00, 0, "View 0: radial pairs + piercing point")

    ax01 = fig.add_subplot(2, 3, 2)
    draw_view(ax01, n_views // 3, f"View {n_views//3}")

    ax02 = fig.add_subplot(2, 3, 3)
    draw_view(ax02, 2 * n_views // 3, f"View {2*n_views//3}")

    ax10 = fig.add_subplot(2, 3, 4)
    ax10.scatter(Pc_per_view[:, 0], Pc_per_view[:, 1],
                 c="tab:red", s=10, alpha=0.6, label="Pc per view")
    ax10.scatter([Pc_median[0]], [Pc_median[1]], c="red", s=180,
                 marker="*", edgecolors="k", linewidths=0.5,
                 label=f"median ({Pc_median[0]:.3f}, {Pc_median[1]:.3f})")
    ax10.scatter([u0], [v0], c="lime", s=120, marker="x",
                 linewidths=2.5, label=f"truth ({u0:.3f}, {v0:.3f})")
    ax10.set_xlabel("u (px)")
    ax10.set_ylabel("v (px)")
    ax10.set_title("Piercing point across views")
    ax10.set_xlim(u_min, u_max)
    ax10.set_ylim(v_max, v_min)
    ax10.set_aspect("equal")
    ax10.grid(alpha=0.3)
    ax10.legend(fontsize=8)

    ax11 = fig.add_subplot(2, 3, 5)
    ax11.plot(Pc_per_view[:, 0], ".", ms=4, color="tab:blue", label="Pc u")
    ax11.plot(Pc_per_view[:, 1], ".", ms=4, color="tab:red", label="Pc v")
    ax11.axhline(u0, color="tab:blue", lw=1, ls="--",
                 label=f"truth u = {u0:.3f}")
    ax11.axhline(v0, color="tab:red", lw=1, ls="--",
                 label=f"truth v = {v0:.3f}")
    ax11.set_xlabel("view index")
    ax11.set_ylabel("pixel")
    ax11.set_title("Pc u, v across views")
    ax11.grid(alpha=0.3)
    ax11.legend(fontsize=7)

    ax12 = fig.add_subplot(2, 3, 6)
    all_res = np.concatenate([r for r in sim["res_per_view"]])
    ax12.hist(all_res, bins=40, color="tab:purple", alpha=0.75)
    ax12.set_xlabel("line-to-Pc distance (px)")
    ax12.set_ylabel("count")
    ax12.set_title(f"Line residuals\nmean = {all_res.mean():.4f} px, "
                   f"max = {all_res.max():.4f} px")
    ax12.grid(alpha=0.3)

    fig.suptitle(
        f"Piercing point = principal point   "
        f"(median = [{Pc_median[0]:.3f}, {Pc_median[1]:.3f}], "
        f"truth = [{u0:.3f}, {v0:.3f}])",
        fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


# =====================================================================
# 6. Main
# =====================================================================
if __name__ == "__main__":
    sim = simulate(
        n_views=360, noise_px=0.1, seed=42,
        sdd=770.0, sod=440.0, u0=499.5, v0=535.5, eta_deg=1.0,
        pixel_size_mm=0.417,
        rigid_rx_deg=0.0, rigid_ry_deg=0.0, rigid_rz_deg=0.0,
        rigid_tx_mm=0.0, rigid_ty_mm=0.0, rigid_tz_mm=0.0,
    )

    print(f"truth principal point: u0 = {sim['truth']['u0_px']:.4f}, "
          f"v0 = {sim['truth']['v0_px']:.4f}")
    print(f"Pc median            : u = {sim['Pc_median'][0]:.4f}, "
          f"v = {sim['Pc_median'][1]:.4f}")
    print(f"diff                 : du = "
          f"{sim['Pc_median'][0] - sim['truth']['u0_px']:+.4f}, "
          f"dv = {sim['Pc_median'][1] - sim['truth']['v0_px']:+.4f}")
    print(f"Pc std over views    : "
          f"{np.std(sim['Pc_per_view'], axis=0).tolist()}")

    visualize(sim)