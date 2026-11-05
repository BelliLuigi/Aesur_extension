# ==============================================================================
# Pipeline Dependencies (zero extra packages)
# ==============================================================================
import os
import argparse
import h5py
import numpy as np
from tqdm import tqdm
import gwsurrogate

# ==============================================================================
# 3D Spin Sampling (Uniform magnitude + isotropic direction)
# ==============================================================================
def sample_spin_3d(size=None, a_max=0.8):
    """
    Sample 3D spin vector with uniform magnitude in [0, a_max] and isotropic direction.
    Returns array of shape (3,) if size is None, else (size, 3).
    """
    a = np.random.uniform(0.0, a_max, size=size)
    cos_theta = np.random.uniform(-1.0, 1.0, size=size)
    phi = np.random.uniform(0.0, 2.0 * np.pi, size=size)
    sin_theta = np.sqrt(1.0 - cos_theta**2)

    chi_x = a * sin_theta * np.cos(phi)
    chi_y = a * sin_theta * np.sin(phi)
    chi_z = a * cos_theta
    return np.stack([chi_x, chi_y, chi_z], axis=-1)

# ==============================================================================
# Precessing Dataset Generation Pipeline
# ==============================================================================
def generate_dataset(
    output_path="datasets/NRSur7dq4v2_dataset.hdf",
    N=102400,
    chunksize=512,
    sur_name="NRSur7dq4v2",
    length=2048,
    dt=2.0,
    a_max=0.8,
    q_min=1.0,
    q_max=4.0,
):
    """
    Generate HDF5 dataset of precessing gravitational waves with 7 parameters and dynamics.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    sur = gwsurrogate.LoadSurrogate(sur_name)

    # Time grid aligned with merger at t=0 (index 1998 for length 2048, dt=2.0)
    times = np.arange(-3996, 100, dt)
    assert len(times) == length, f"Expected {length} time steps, got {len(times)}"

    param_names = ["q", "chiA0_x", "chiA0_y", "chiA0_z", "chiB0_x", "chiB0_y", "chiB0_z"]

    with h5py.File(output_path, "w", libver="latest") as f:
        # 1. Inertial Waveforms
        f.create_dataset("Waveforms", shape=(N, length), dtype=np.complex128)

        # 2. Intrinsic Parameters (7 values)
        d_params = f.create_dataset("Parameters", shape=(N, 7), dtype=np.float32)
        d_params.attrs["names"] = param_names

        # 3. Dynamics
        g_dyn = f.create_group("Dynamics")
        d_q_copr = g_dyn.create_dataset("q_copr", shape=(N, 4, length), dtype=np.float32)
        d_chiA = g_dyn.create_dataset("chiA", shape=(N, length, 3), dtype=np.float32)
        d_chiB = g_dyn.create_dataset("chiB", shape=(N, length, 3), dtype=np.float32)
        d_orbphase = g_dyn.create_dataset("orbphase", shape=(N, length), dtype=np.float32)

        # Store time grid attributes
        #f.attrs["times"] = times
        #f.attrs["dt"] = dt

        precessing_opts = {"return_dynamics": True}

        num_chunks = int(np.ceil(N / chunksize))
        print(f"Generating {N} precessing waveforms with dynamics in {num_chunks} chunks...")

        for c in tqdm(range(num_chunks), desc="Chunks"):
            start = c * chunksize
            end = min(start + chunksize, N)
            cur_chunk = end - start

            # Pre-allocated chunk buffers
            wave_chunk = np.empty((cur_chunk, length), dtype=np.complex128)
            param_chunk = np.empty((cur_chunk, 7), dtype=np.float32)
            q_copr_chunk = np.empty((cur_chunk, 4, length), dtype=np.float32)
            chiA_chunk = np.empty((cur_chunk, length, 3), dtype=np.float32)
            chiB_chunk = np.empty((cur_chunk, length, 3), dtype=np.float32)
            orbphase_chunk = np.empty((cur_chunk, length), dtype=np.float32)

            # Sample batch parameters
            q_batch = np.random.uniform(q_min, q_max, size=cur_chunk)
            chiA0_batch = sample_spin_3d(size=cur_chunk, a_max=a_max)
            chiB0_batch = sample_spin_3d(size=cur_chunk, a_max=a_max)

            for i in range(cur_chunk):
                q_val = float(q_batch[i])
                chiA0_val = chiA0_batch[i]
                chiB0_val = chiB0_batch[i]

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
                param_chunk[i] = [q_val, *chiA0_val, *chiB0_val]
                q_copr_chunk[i] = dyn["q_copr"]
                chiA_chunk[i] = dyn["chiA"]
                chiB_chunk[i] = dyn["chiB"]
                orbphase_chunk[i] = dyn["orbphase"]

            # Write chunk to HDF5
            f["Waveforms"][start:end] = wave_chunk
            f["Parameters"][start:end] = param_chunk
            d_q_copr[start:end] = q_copr_chunk
            d_chiA[start:end] = chiA_chunk
            d_chiB[start:end] = chiB_chunk
            d_orbphase[start:end] = orbphase_chunk
            f.flush()

    print(f"Dataset successfully created at {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate precessing waveform dataset with dynamics.")
    parser.add_argument("--sur", type=str, default="NRSur7dq4v2", help="Surrogate name")
    parser.add_argument("--output", type=str, default="datasets/NRSur7dq4v2_dataset.hdf", help="Output HDF5 path")
    parser.add_argument("-N", type=int, default=102400, help="Number of waveforms")
    parser.add_argument("--chunksize", type=int, default=512, help="Chunk size for writing")
    args = parser.parse_args()

    generate_dataset(output_path=args.output, N=args.N, chunksize=args.chunksize, sur_name=args.sur)
