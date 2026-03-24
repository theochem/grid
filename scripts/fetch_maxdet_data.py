import urllib.request
import os
import numpy as np

def download_maxdet(degree, npts):
    filename = f"md{degree:03d}.{npts:05d}"
    url = f"https://web.maths.unsw.edu.au/~rsw/Sphere/S2Pts/MD/{filename}"
    print(f"Downloading {url}...")
    try:
        with urllib.request.urlopen(url) as f:
            data = f.read().decode().splitlines()
        points_weights = []
        for line in data:
            if line.strip():
                points_weights.append([float(x) for x in line.split()])
        pw = np.array(points_weights)
        points = pw[:, :3]
        # Sum of weights in the file is already 4*pi
        # Let's normalize to 1 to stay consistent with other npz files
        weights = pw[:, 3] / (4 * np.pi)
        return points, weights
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return None, None

def save_as_npz(degree, npts, points, weights, target_dir):
    filename = f"maxdet_{degree}_{npts}.npz"
    filepath = os.path.join(target_dir, filename)
    print(f"Saving to {filepath}...")
    np.savez(filepath, degree=degree, size=npts, points=points, weights=weights)

if __name__ == "__main__":
    target_dir = r"C:\Users\valok\.gemini\antigravity\scratch\grid\src\grid\data\max_det"
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    
    # Degrees and corresponding number of points
    degrees = [1, 2, 3, 4, 5, 10, 20, 30] 
    for d in degrees:
        npts = (d + 1) ** 2
        p, w = download_maxdet(d, npts)
        if p is not None:
            save_as_npz(d, npts, p, w, target_dir)
