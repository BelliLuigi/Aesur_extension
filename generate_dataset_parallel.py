#!/usr/bin/env python3
"""
Generates the NRSur7dq4v2 precessing waveform dataset (N=102,400) using 4 parallel
workers with chunksize=1024 and optimal transport ball sampling.
"""
import os
import sys
import time
import h5py
import numpy as np
from scipy.stats import qmc
from concurrent.futures import ProcessPoolExecutor
import gwsurrogate
from gwsurrogate.new.precessing_surrogate import rotateWaveform, quatInv, multiplyQuats

N_TOTAL = 102400
NUM_WORKERS = 4
CHUNKSIZE = 1024
LENGTH = 2048
DT = 2.0
Q_MIN, Q_MAX = 1.0, 4.0
A_MAX = 0.8
OUTPUT_PATH = "datasets/NRSur7dq4v2_dataset.hdf"

# ==============================================================================
# 1. 7D Parameter Sampling: Uniform q + OT-Uniform Ball Spins
# ==============================================================================
def sample_spin_ball(n, a_max=0.8, over_factor=1.04, iters=8):
    """Sample 3D spins with flat 1D marginals across [-0.78, 0.78] and |chi| <= a_max."""
    pool_n = max(n, 4096)
    alpha = 0.50
    u = 2.0 * np.random.beta(alpha, alpha, size=(pool_n, 3)) - 1.0
    u0, u1, u2 = u[:, 0], u[:, 1], u[:, 2]

    x = u0 * np.sqrt(np.clip(1.0 - u1**2 / 2.0 - u2**2 / 2.0 + (u1**2 * u2**2) / 3.0, 0.0, None)) * a_max
    y = u1 * np.sqrt(np.clip(1.0 - u0**2 / 2.0 - u2**2 / 2.0 + (u0**2 * u2**2) / 3.0, 0.0, None)) * a_max
    z = u2 * np.sqrt(np.clip(1.0 - u0**2 / 2.0 - u1**2 / 2.0 + (u0**2 * u1**2) / 3.0, 0.0, None)) * a_max
    pts = np.stack([x, y, z], axis=-1)

    target_grid = np.linspace(-a_max * over_factor, a_max * over_factor, pool_n)
    for _ in range(iters):
        for dim in range(3):
            order = np.argsort(pts[:, dim])
            jitter = np.random.uniform(-a_max / pool_n, a_max / pool_n, size=pool_n)
            pts[order, dim] = np.clip(target_grid + jitter, -a_max, a_max)

        norms = np.linalg.norm(pts, axis=-1, keepdims=True)
        mask = (norms > a_max).squeeze(-1)
        if np.any(mask):
            pts[mask] = pts[mask] / norms[mask] * (a_max * 0.9999)

    pts = pts[:n].astype(np.float32)
    norms32 = np.linalg.norm(pts, axis=-1, keepdims=True)
    over = (norms32 > a_max).squeeze(-1)
    if np.any(over):
        pts[over] = (pts[over] / norms32[over]) * np.float32(a_max * 0.999999)
    return pts

def sample_parameters_7d(n=N_TOTAL):
    sampler = qmc.Sobol(d=1, scramble=True, seed=42)
    u_q = sampler.random(n=n)[:, 0]
    q = Q_MIN + (Q_MAX - Q_MIN) * u_q
    chiA = sample_spin_ball(n, a_max=A_MAX)
    chiB = sample_spin_ball(n, a_max=A_MAX)
    return q, chiA, chiB

# ==============================================================================
# 2. Worker Waveform Task
# ==============================================================================
def worker_task(worker_id, q_slice, chiA_slice, chiB_slice, part_path):
    times = np.arange(-3996, 100, DT)
    n_part = len(q_slice)
    sur = gwsurrogate.LoadSurrogate("NRSur7dq4v2")
    precessing_opts = {"return_dynamics": True}
    num_chunks = int(np.ceil(n_part / CHUNKSIZE))

    with h5py.File(part_path, "w", libver="latest") as f:
        f.create_dataset("Waveforms", shape=(n_part, LENGTH), dtype=np.complex128)
        f.create_dataset("Waveforms_coorb", shape=(n_part, LENGTH), dtype=np.complex128)
        f.create_dataset("Parameters", shape=(n_part, 7), dtype=np.float32)
        g_dyn = f.create_group("Dynamics")
        d_q_copr = g_dyn.create_dataset("q_copr", shape=(n_part, 4, LENGTH), dtype=np.float32)
        d_chiA = g_dyn.create_dataset("chiA", shape=(n_part, LENGTH, 3), dtype=np.float32)
        d_chiB = g_dyn.create_dataset("chiB", shape=(n_part, LENGTH, 3), dtype=np.float32)
        d_orbphase = g_dyn.create_dataset("orbphase", shape=(n_part, LENGTH), dtype=np.float32)

        for c in range(num_chunks):
            start = c * CHUNKSIZE
            end = min(start + CHUNKSIZE, n_part)
            cur_n = end - start

            wave_chunk = np.empty((cur_n, LENGTH), dtype=np.complex128)
            wave_coorb_chunk = np.empty((cur_n, LENGTH), dtype=np.complex128)
            param_chunk = np.empty((cur_n, 7), dtype=np.float32)
            q_copr_chunk = np.empty((cur_n, 4, LENGTH), dtype=np.float32)
            chiA_chunk = np.empty((cur_n, LENGTH, 3), dtype=np.float32)
            chiB_chunk = np.empty((cur_n, LENGTH, 3), dtype=np.float32)
            orbphase_chunk = np.empty((cur_n, LENGTH), dtype=np.float32)

            for i in range(cur_n):
                q_val = float(q_slice[start + i])
                chiA0_val = chiA_slice[start + i]
                chiB0_val = chiB_slice[start + i]

                _, h, dyn = sur(
                    q=q_val,
                    chiA0=chiA0_val,
                    chiB0=chiB0_val,
                    f_low=0.0,
                    times=times,
                    ellMax=2,
                    precessing_opts=precessing_opts,
                )

                wave_chunk[i] = h[(2, 2)]

                # Co-orbital frame (2, 2) mode: steady slowly-varying envelope (zero carrier oscillations)
                h_in_arr = np.array([h[(2, m)] for m in range(-2, 3)])
                q_rot = np.array([
                    np.cos(dyn["orbphase"] / 2.0),
                    0.0 * dyn["orbphase"],
                    0.0 * dyn["orbphase"],
                    np.sin(dyn["orbphase"] / 2.0),
                ])
                qfull = multiplyQuats(dyn["q_copr"], q_rot)
                wave_coorb_chunk[i] = rotateWaveform(quatInv(qfull), h_in_arr)[4]

                param_chunk[i] = [q_val, *chiA0_val, *chiB0_val]
                q_copr_chunk[i] = dyn["q_copr"]
                chiA_chunk[i] = dyn["chiA"]
                chiB_chunk[i] = dyn["chiB"]
                orbphase_chunk[i] = dyn["orbphase"]

            f["Waveforms"][start:end] = wave_chunk
            f["Waveforms_coorb"][start:end] = wave_coorb_chunk
            f["Parameters"][start:end] = param_chunk
            d_q_copr[start:end] = q_copr_chunk
            d_chiA[start:end] = chiA_chunk
            d_chiB[start:end] = chiB_chunk
            d_orbphase[start:end] = orbphase_chunk
            f.flush()
            print(f"[Worker {worker_id}] Chunk {c+1}/{num_chunks} complete ({end}/{n_part})", flush=True)

    return part_path

# ==============================================================================
# 3. Main Pipeline: Sample -> Parallel Generate -> Merge
# ==============================================================================
def main():
    os.makedirs("datasets", exist_ok=True)

    print(f"1/3 Pre-sampling parameters for N={N_TOTAL:,} (ot_uniform_ball)...", flush=True)
    t0 = time.time()
    q_all, chiA_all, chiB_all = sample_parameters_7d(N_TOTAL)
    print(f"    Done in {time.time() - t0:.2f}s.", flush=True)

    n_per_worker = N_TOTAL // NUM_WORKERS
    part_files = [f"datasets/temp_part_{w}.hdf" for w in range(NUM_WORKERS)]
    worker_args = [
        (w, q_all[w * n_per_worker : (w + 1) * n_per_worker],
            chiA_all[w * n_per_worker : (w + 1) * n_per_worker],
            chiB_all[w * n_per_worker : (w + 1) * n_per_worker],
            part_files[w])
        for w in range(NUM_WORKERS)
    ]

    print(f"2/3 Generating {N_TOTAL:,} waveforms across {NUM_WORKERS} workers...", flush=True)
    t_gen = time.time()
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = [executor.submit(worker_task, *args) for args in worker_args]
        for fut in futures:
            fut.result()
    print(f"    Generation finished in {(time.time() - t_gen)/60:.2f} min.", flush=True)

    print(f"3/3 Merging parts into {OUTPUT_PATH}...", flush=True)
    t_merge = time.time()
    param_names = ["q", "chiA0_x", "chiA0_y", "chiA0_z", "chiB0_x", "chiB0_y", "chiB0_z"]
    with h5py.File(OUTPUT_PATH, "w", libver="latest") as f_out:
        f_out.create_dataset("Waveforms", shape=(N_TOTAL, LENGTH), dtype=np.complex128)
        f_out.create_dataset("Waveforms_coorb", shape=(N_TOTAL, LENGTH), dtype=np.complex128)
        d_params = f_out.create_dataset("Parameters", shape=(N_TOTAL, 7), dtype=np.float32)
        d_params.attrs["names"] = param_names

        g_dyn = f_out.create_group("Dynamics")
        d_q_copr = g_dyn.create_dataset("q_copr", shape=(N_TOTAL, 4, LENGTH), dtype=np.float32)
        d_chiA = g_dyn.create_dataset("chiA", shape=(N_TOTAL, LENGTH, 3), dtype=np.float32)
        d_chiB = g_dyn.create_dataset("chiB", shape=(N_TOTAL, LENGTH, 3), dtype=np.float32)
        d_orbphase = g_dyn.create_dataset("orbphase", shape=(N_TOTAL, LENGTH), dtype=np.float32)

        offset = 0
        block_size = 5120
        for p_file in part_files:
            with h5py.File(p_file, "r") as f_in:
                n_p = len(f_in["Parameters"])
                for b in range(0, n_p, block_size):
                    b_end = min(b + block_size, n_p)
                    cur_s = slice(b, b_end)
                    out_s = slice(offset + b, offset + b_end)
                    f_out["Waveforms"][out_s] = f_in["Waveforms"][cur_s]
                    f_out["Waveforms_coorb"][out_s] = f_in["Waveforms_coorb"][cur_s]
                    f_out["Parameters"][out_s] = f_in["Parameters"][cur_s]
                    d_q_copr[out_s] = f_in["Dynamics/q_copr"][cur_s]
                    d_chiA[out_s] = f_in["Dynamics/chiA"][cur_s]
                    d_chiB[out_s] = f_in["Dynamics/chiB"][cur_s]
                    d_orbphase[out_s] = f_in["Dynamics/orbphase"][cur_s]
                offset += n_p

    for p_file in part_files:
        if os.path.exists(p_file):
            os.remove(p_file)

    print(f"    Merge finished in {time.time() - t_merge:.1f}s.", flush=True)
    print(f"Dataset ready at {OUTPUT_PATH} (total: {(time.time() - t0)/60:.2f} min).", flush=True)

if __name__ == "__main__":
    main()
