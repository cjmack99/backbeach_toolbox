#!/usr/bin/env python3
import os
import csv
import platform
import multiprocessing
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")

import rasterio
from rasterio.transform import xy
from rasterio.plot import show
from scipy.ndimage import generic_filter
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# ────────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ────────────────────────────────────────────────────────────────────────────────

# Overwrite existing outputs?
replace = False

# Detect OS and set base paths
system = platform.system()
if system == "Darwin":
    input_csv   = "/Volumes/group/LiDAR/LidarProcessing/backbeach_dem_tools/tif_paths.csv"
    output_base = "/Volumes/group/LiDAR/LidarProcessing/backbeach_dem_tools/cliff_toe_lines/del_mar"
    jpeg_dir    = "/Volumes/group/LiDAR/LidarProcessing/backbeach_dem_tools/dems_with_toe"
elif system == "Linux":
    input_csv   = "/project/group/LiDAR/LidarProcessing/backbeach_dem_tools/tif_paths.csv"
    output_base = "/project/group/LiDAR/LidarProcessing/backbeach_dem_tools/cliff_toe_lines/del_mar"
    jpeg_dir    = "/project/group/LiDAR/LidarProcessing/backbeach_dem_tools/dems_with_toe"
else:
    raise RuntimeError(f"Unsupported OS: {system}")

os.makedirs(output_base, exist_ok=True)
os.makedirs(jpeg_dir, exist_ok=True)

# K-means and line detection settings
k_clusters = 4    # e.g. water / beach / cliff
min_run    = 5    # require this many consecutive non-beach pixels

# Determine number of workers: total CPU cores // 4, at least 1
n_workers = max(1, multiprocessing.cpu_count() // 4)

# ────────────────────────────────────────────────────────────────────────────────
# PROCESSING FUNCTION
# ────────────────────────────────────────────────────────────────────────────────
def process_tile(row):
    orig_tif = row["path"]
    # adjust path on Linux
    tif_path = orig_tif.replace("/Volumes/", "/project/", 1) if system == "Linux" else orig_tif
    if not os.path.isfile(tif_path):
        print(f"⚠️  Missing file: {tif_path}")
        return

    # derive survey_name and date_tag
    survey_name = os.path.basename(os.path.dirname(os.path.dirname(tif_path)))
    date_tag    = survey_name.split("_")[0]
    out_csv     = os.path.join(output_base, f"{survey_name}_xyz.csv")
    out_jpeg    = os.path.join(jpeg_dir,    f"{date_tag}.jpg")

    # skip if outputs exist and replace=False
    if not replace and os.path.exists(out_csv) and os.path.exists(out_jpeg):
        print(f"🔹 Skipping {survey_name}, outputs exist")
        return

    print(f"[Processing] {survey_name}")

    # Load DEM
    with rasterio.open(tif_path) as src:
        dem       = src.read(1, masked=True)
        transform = src.transform
        res_x, res_y = src.res
        height, width = dem.shape

    # Features: elevation, slope, roughness
    elev = dem.filled(np.nan)
    dy, dx = np.gradient(elev, res_y, res_x)
    slope   = np.sqrt(dx*dx + dy*dy)
    roughness = generic_filter(
        elev,
        function=lambda w: np.nanstd(w) if not np.isnan(w).all() else np.nan,
        size=5,
        mode="constant",
        cval=np.nan
    )

    # Stack & mask invalid
    stack = np.dstack([elev, slope, roughness])
    valid = (~dem.mask) & ~np.isnan(stack).any(axis=2)

    # Standardize & cluster
    X        = stack[valid]
    X_scaled = StandardScaler().fit_transform(X)
    km       = KMeans(n_clusters=k_clusters, random_state=0).fit(X_scaled)
    labels   = np.full(dem.shape, -1, dtype=int)
    labels[valid] = km.labels_

    # Identify beach cluster
    beach_idx = np.argmin(km.cluster_centers_.sum(axis=1))

    # Trace back-beach line
    shoreline_pts = []
    for i in range(height):
        in_beach = False
        j = 0
        while j < width:
            lbl = labels[i, j]
            if not in_beach:
                if lbl == beach_idx:
                    in_beach = True
                j += 1
            else:
                if not valid[i, j]:
                    j += 1; continue
                if lbl != beach_idx:
                    count, k2 = 1, j+1
                    while k2 < width and count < min_run:
                        if valid[i, k2] and labels[i, k2] != beach_idx:
                            count += 1; k2 += 1
                        elif not valid[i, k2]:
                            k2 += 1
                        else:
                            break
                    if count >= min_run:
                        x, y = xy(transform, i, j, offset="center")
                        shoreline_pts.append((x, y, float(dem[i, j])))
                        break
                j += 1

    if not shoreline_pts:
        print(f"  ⚠️ No shoreline points for {survey_name}")
        return

    # Smooth z outliers
    xs, ys, zs = zip(*shoreline_pts)
    zs_arr = np.array(zs)
    med = np.median(zs_arr)
    mad = np.median(np.abs(zs_arr - med))
    inlier = np.abs(zs_arr - med) <= 3*mad
    zs_smooth = zs_arr.copy()
    for ii, ok in enumerate(inlier):
        if not ok:
            prevs = np.where(inlier[:ii])[0]
            nexts = np.where(inlier[ii+1:])[0] + ii + 1
            if prevs.size and nexts.size:
                zs_smooth[ii] = 0.5*(zs_arr[prevs[-1]] + zs_arr[nexts[0]])
            elif prevs.size:
                zs_smooth[ii] = zs_arr[prevs[-1]]
            elif nexts.size:
                zs_smooth[ii] = zs_arr[nexts[0]]
            else:
                zs_smooth[ii] = med

    shoreline_smoothed = list(zip(xs, ys, zs_smooth))

    # Write CSV
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["x","y","z"])
        w.writerows(shoreline_smoothed)

    # Save JPEG of DEM + line
    fig, ax = plt.subplots(figsize=(8, 8))
    show(dem, transform=transform, ax=ax, cmap="terrain")
    xs2, ys2, _ = zip(*shoreline_smoothed)
    ax.plot(xs2, ys2, color="red", linewidth=1.5)
    ax.set_title(survey_name)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.tight_layout()
    fig.savefig(out_jpeg, dpi=300)
    plt.close(fig)

# ────────────────────────────────────────────────────────────────────────────────
# MAIN: parallel execution
# ────────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = pd.read_csv(input_csv)
    rows = df.to_dict("records")
    print(f"Running on {n_workers} workers…")
    with multiprocessing.Pool(n_workers) as pool:
        pool.map(process_tile, rows)
    print("All done!")
