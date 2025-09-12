import h5py
import numpy as np
import os

def find_hdf5_files(root_dir):
    hdf5_paths = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".hdf5"):
                hdf5_paths.append(os.path.join(root, file))
    return sorted(hdf5_paths)

data_dir = "path/to/your/dataset"
paths = find_hdf5_files(data_dir)

total_files = 0
nan_files = 0
bad_shape_files = 0

print(f"🧹 Checking {len(paths)} HDF5 files in: {data_dir}")

for path in paths:
    total_files += 1
    try:
        with h5py.File(path, 'r') as f:
            qpos = f['qpos'][:]
            if np.isnan(qpos).any() or np.isinf(qpos).any():
                print("❗ Found invalid values in:", path)
                nan_files += 1
            elif qpos.shape[1] != 32:
                print("⚠️ Shape issue in:", path, "Got shape:", qpos.shape)
                bad_shape_files += 1
    except Exception as e:
        print("❌ Error reading file:", path, "\n", e)

print("\n✅ Dataset check complete.")
print(f"Total files checked: {total_files}")
print(f"Files with NaN/Inf: {nan_files}")
print(f"Files with bad shape: {bad_shape_files}")
