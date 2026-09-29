# -*- coding: utf-8 -*-
"""Simulate two-ring phantom + unknown rigid pose + joint calibration,
then visualize everything in a SINGLE figure window (no files saved).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401


# =====================================================================
# Phantom + rigid
# =====================================================================
def make_two_ring_phantom(radius_mm=30.0, n_per_ring=8,
                          z_top_mm=25.0, z_bottom_mm=-25.0):
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


def rot_from_axis_angle(p):
    angle = float(np.linalg.norm(p))
    if angle < 1e-12:
        return np.eye(3)
    axis = p / angle
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


# =====================================================================
# Projection model
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
# Joint solve
# =====================================================================
def joint_solve(tracks, angles, pixel_size_mm, beads_body,
                initial_geometry, initial_rigid_params,
                center_sigma_px=0.1, max_nfev=2000):
    n_views, n_beads, _ = tracks.shape
    g = initial_geometry
    initial = np.r_[g["sdd"], g["sod"], g["u0"], g["v0"],
                    np.deg2rad(g["eta_deg"]),
                    initial_rigid_params]

    def unpack(p):
        sdd, sod, u0, v0, eta_rad = p[:5]
        R = rot_from_axis_angle(p[5:8])
        t = p[8:11]
        return sdd, sod, u0, v0, float(np.rad2deg(eta_rad)), R, t

    def residual(p):
        sdd, sod, u0, v0, eta_deg, R, t = unpack(p)
        beads_world = (R @ beads_body.T).T + t
        pred = project(sdd, sod, u0, v0, eta_deg, beads_world,
                       angles, pixel_size_mm)
        return (pred - tracks).ravel() / center_sigma_px

    lower = np.full(initial.size, -np.inf)
    lower[0] = 1e-6
    lower[1] = 1e-6
    opt = least_squares(residual, initial, bounds=(lower, np.inf),
                        x_scale="jac", max_nfev=max_nfev,
                        ftol=1e-12, xtol=1e-12, gtol=1e-10)

    sdd, sod, u0, v0, eta_deg, R, t = unpack(opt.x)
    beads_world = (R @ beads_body.T).T + t
    pred = project(sdd, sod, u0, v0, eta_deg, beads_world, angles, pixel_size_mm)
    res = pred - tracks
    return {
        "geometry": dict(sdd_mm=float(sdd), sod_mm=float(sod),
                         u0_px=float(u0), v0_px=float(v0), eta_deg=float(eta_deg)),
        "rigid": dict(R=R.tolist(), t=t.tolist()),
        "beads_world": beads_world,
        "predicted": pred,
        "residual_px": res,
        "rmse_px": float(np.sqrt(np.mean(res ** 2))),
    }


# =====================================================================
# Simulation driver
# =====================================================================
def simulate(*, n_views=359, noise_px=0.1, seed=42,
             sdd=770.0, sod=440.0, u0=499.5, v0=535.5, eta_deg=1.0,
             pixel_size_mm=0.417,
             rigid_rx_deg=2.0, rigid_ry_deg=-1.5, rigid_rz_deg=0.5,
             rigid_tx_mm=5.0, rigid_ty_mm=-3.0, rigid_tz_mm=8.0):
    beads_body = make_two_ring_phantom()
    n_beads = len(beads_body)

    R_truth = rotation_xyz(np.deg2rad(rigid_rx_deg),
                           np.deg2rad(rigid_ry_deg),
                           np.deg2rad(rigid_rz_deg))
    t_truth = np.array([rigid_tx_mm, rigid_ty_mm, rigid_tz_mm])
    beads_world_truth = apply_rigid(beads_body, R_truth, t_truth)

    angles = np.arange(n_views) * (2 * np.pi / n_views)
    pixel_size = np.array([pixel_size_mm, pixel_size_mm])
    tracks = project(sdd, sod, u0, v0, eta_deg,
                     beads_world_truth, angles, pixel_size)
    rng = np.random.default_rng(seed)
    if noise_px > 0:
        tracks = tracks + rng.normal(0, noise_px, tracks.shape)

    initial_geometry = dict(sdd=760.0, sod=450.0, u0=500.0, v0=535.0, eta_deg=0.8)
    initial_rigid_params = np.zeros(6)
    result = joint_solve(tracks, angles, pixel_size, beads_body,
                         initial_geometry, initial_rigid_params,
                         center_sigma_px=0.1)

    truth = dict(geometry=dict(sdd_mm=sdd, sod_mm=sod,
                                u0_px=u0, v0_px=v0, eta_deg=eta_deg),
                 R=R_truth, t=t_truth,
                 beads_body=beads_body,
                 beads_world=beads_world_truth)
    return dict(tracks=tracks, angles=angles, pixel_size=pixel_size,
                truth=truth, recovered=result, n_views=n_views)


# =====================================================================
# Single-window visualization
# =====================================================================
def visualize_all(sim):
    beads_body = sim["truth"]["beads_body"]
    world_truth = sim["truth"]["beads_world"]
    world_rec = sim["recovered"]["beads_world"]
    tracks = sim["tracks"]
    pred = sim["recovered"]["predicted"]
    res = sim["recovered"]["residual_px"]
    n_beads = len(beads_body)
    n_per_ring = n_beads // 2
    angles = sim["angles"]
    sod = sim["truth"]["geometry"]["sod_mm"]

    fig = plt.figure(figsize=(18, 10))

    # [0,0] 3D phantom
    ax00 = fig.add_subplot(2, 3, 1, projection="3d")
    top_t = world_truth[:n_per_ring]
    bot_t = world_truth[n_per_ring:]
    top_r = world_rec[:n_per_ring]
    bot_r = world_rec[n_per_ring:]
    ax00.scatter(top_t[:, 0], top_t[:, 1], top_t[:, 2],
                 c="tab:blue", s=60, marker="o", label="truth top",
                 edgecolors="k", linewidths=0.5)
    ax00.scatter(bot_t[:, 0], bot_t[:, 1], bot_t[:, 2],
                 c="tab:cyan", s=60, marker="o", label="truth bottom",
                 edgecolors="k", linewidths=0.5)
    ax00.scatter(top_r[:, 0], top_r[:, 1], top_r[:, 2],
                 c="tab:red", s=45, marker="x", label="recovered top")
    ax00.scatter(bot_r[:, 0], bot_r[:, 1], bot_r[:, 2],
                 c="tab:orange", s=45, marker="x", label="recovered bottom")
    for i in range(n_beads):
        ax00.plot([world_truth[i, 0], world_rec[i, 0]],
                  [world_truth[i, 1], world_rec[i, 1]],
                  [world_truth[i, 2], world_rec[i, 2]],
                  color="k", lw=0.8, alpha=0.6)
    ax00.set_xlabel("X (mm)")
    ax00.set_ylabel("Y (mm)")
    ax00.set_zlabel("Z (mm)")
    ax00.set_title("3D phantom: truth vs recovered")
    ax00.legend(fontsize=7, loc="upper left")

    # [0,1] 3D source orbit
    ax01 = fig.add_subplot(2, 3, 2, projection="3d")
    sx = sod * np.cos(angles)
    sy = sod * np.sin(angles)
    sz = np.zeros_like(angles)
    ax01.plot(sx, sy, sz, color="gray", lw=1.2, alpha=0.6, label="source orbit")
    ax01.scatter(world_truth[:, 0], world_truth[:, 1], world_truth[:, 2],
                 c="tab:blue", s=55, marker="o", label="phantom truth",
                 edgecolors="k", linewidths=0.5)
    ax01.scatter(world_rec[:, 0], world_rec[:, 1], world_rec[:, 2],
                 c="tab:red", s=40, marker="x", label="phantom recovered")
    ax01.plot([0, 0], [0, 0], [-100, 100], color="k", lw=1,
              linestyle="--", alpha=0.5, label="rotation axis")
    ax01.set_xlabel("X (mm)")
    ax01.set_ylabel("Y (mm)")
    ax01.set_zlabel("Z (mm)")
    ax01.set_title("Source orbit + phantom pose")
    ax01.legend(fontsize=7)

    # [0,2] Parameter comparison
    ax02 = fig.add_subplot(2, 3, 3)
    truth_g = sim["truth"]["geometry"]
    rec_g = sim["recovered"]["geometry"]
    keys = ["sdd_mm", "sod_mm", "u0_px", "v0_px", "eta_deg"]
    labels = ["SDD\n(mm)", "SOD\n(mm)", "u0\n(px)", "v0\n(px)", "eta\n(deg)"]
    true_vals = np.array([truth_g[k] for k in keys])
    rec_vals = np.array([rec_g[k] for k in keys])
    x = np.arange(len(keys))
    w = 0.35
    ax02.bar(x - w/2, true_vals, w, label="truth", color="tab:blue")
    ax02.bar(x + w/2, rec_vals, w, label="recovered", color="tab:red")
    for i, (t, r) in enumerate(zip(true_vals, rec_vals)):
        ax02.text(i - w/2, t, f"{t:.3f}", ha="center", va="bottom", fontsize=7)
        ax02.text(i + w/2, r, f"{r:.3f}", ha="center", va="bottom", fontsize=7)
    ax02.set_xticks(x)
    ax02.set_xticklabels(labels)
    ax02.set_ylabel("value")
    ax02.set_title("Recovered vs truth geometry")
    ax02.legend(fontsize=8)
    ax02.grid(axis="y", alpha=0.3)

    # [1,0] scatter view 0
    def scatter_view(ax, v, title):
        uv_t = tracks[v]
        uv_p = pred[v]
        ax.scatter(uv_t[:n_per_ring, 0], uv_t[:n_per_ring, 1],
                   c="tab:blue", s=45, marker="o", label="truth top",
                   edgecolors="k", linewidths=0.5)
        ax.scatter(uv_t[n_per_ring:, 0], uv_t[n_per_ring:, 1],
                   c="tab:cyan", s=45, marker="o", label="truth bottom",
                   edgecolors="k", linewidths=0.5)
        ax.scatter(uv_p[:n_per_ring, 0], uv_p[:n_per_ring, 1],
                   c="red", s=25, marker="x", label="recovered top")
        ax.scatter(uv_p[n_per_ring:, 0], uv_p[n_per_ring:, 1],
                   c="orange", s=25, marker="x", label="recovered bottom")
        ax.set_xlabel("u (px)")
        ax.set_ylabel("v (px)")
        ax.set_title(title)
        ax.invert_yaxis()
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)

    ax10 = fig.add_subplot(2, 3, 4)
    scatter_view(ax10, 0, "Detector scatter, view 0 (β=0°)")
    ax10.legend(fontsize=7, loc="best")

    ax11 = fig.add_subplot(2, 3, 5)
    v_mid = len(angles) // 4
    scatter_view(ax11, v_mid,
                 f"Detector scatter, view {v_mid} "
                 f"(β={np.rad2deg(angles[v_mid]):.0f}°)")
    ax11.legend(fontsize=7, loc="best")

    # [1,2] residual histograms
    ax12 = fig.add_subplot(2, 3, 6)
    ax12.hist(res[..., 0].ravel(), bins=50, color="tab:blue",
              alpha=0.7, label="Δu")
    ax12.hist(res[..., 1].ravel(), bins=50, color="tab:red",
              alpha=0.7, label="Δv")
    ax12.set_xlabel("residual (px)")
    ax12.set_ylabel("count")
    ax12.set_title(f"Reprojection residuals\n"
                   f"RMSE = {sim['recovered']['rmse_px']:.4f} px")
    ax12.legend(fontsize=8)
    ax12.grid(alpha=0.3)

    fig.suptitle("Two-ring phantom, unknown rigid pose, joint calibration",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()


# =====================================================================
# Main
# =====================================================================
if __name__ == "__main__":
    sim = simulate(noise_px=0.1, seed=42)

    print(f"truth:     SDD={sim['truth']['geometry']['sdd_mm']:.4f}, "
          f"SOD={sim['truth']['geometry']['sod_mm']:.4f}, "
          f"u0={sim['truth']['geometry']['u0_px']:.4f}, "
          f"v0={sim['truth']['geometry']['v0_px']:.4f}, "
          f"eta={sim['truth']['geometry']['eta_deg']:.4f}")
    print(f"recovered: SDD={sim['recovered']['geometry']['sdd_mm']:.4f}, "
          f"SOD={sim['recovered']['geometry']['sod_mm']:.4f}, "
          f"u0={sim['recovered']['geometry']['u0_px']:.4f}, "
          f"v0={sim['recovered']['geometry']['v0_px']:.4f}, "
          f"eta={sim['recovered']['geometry']['eta_deg']:.4f}")
    print(f"RMSE = {sim['recovered']['rmse_px']:.4f} px")

    visualize_all(sim)