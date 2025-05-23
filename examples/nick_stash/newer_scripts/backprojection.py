import numpy as np

C = 3E8


def backprojection(pt_clouds, 
                   voxel_grid, 
                   start_gate, 
                   end_gate, 
                   bin_width, 
                   hists, 
                   voxel_params,
                   xlim, 
                   ylim,
                   zlim,
                   thresh = np.nan):
        """
        pt_clouds : List of length num_frames. Each entry contains np array [12, 12, 3]  
        voxel_grid: [num_voxels, 3]
        start_gate: scalar
        end_gate  : scalar
        bin_width : scalar
        hists     : List of length num_frames. Each entry contains np array [12, 12, numBins]
        """
        # === reshape hists and point clouds === #
        hists = [hist.reshape(144, -1) for hist in hists]
        pt_clouds = [pt_cloud.reshape(144, 3) for pt_cloud in pt_clouds]
        
        # === Extract voxel params === #
        num_x, num_y, num_z = voxel_params[0:3]
        num_voxels = voxel_grid.shape[0]

        # === Extract other parameters === #
        num_hists = len(hists)
        if np.isnan(thresh):
            thresh = bin_width * C

        # === Backprojection === #
        volume = np.zeros((num_voxels, 1))
        for k in range(num_hists):
            cur_pt_cloud = pt_clouds[k]
            for i, cur_pixel in enumerate(cur_pt_cloud):
                dists = np.linalg.norm(voxel_grid - cur_pixel.reshape(1, 3), axis=1).reshape(-1, 1)
                for j in range(start_gate, end_gate):
                    cur_radius = j * bin_width * C / 2
                    mask = np.abs(dists - cur_radius) < thresh
                    volume += hists[k][i, j] * mask

        volume = volume.reshape(num_x, num_y, num_z, order="C")

        # === filtering step === #
        volume_filter = filter_volume(volume, num_x, num_y)
        volume_filter = np.transpose(volume_filter, [2, 1, 0])
        volume_filter = np.flip(volume_filter, axis=(1, 2))

        # === Determine plotting === #
        x_tick_labels = np.linspace(0, num_x-1, 3)
        y_tick_labels = np.linspace(0, num_y-1, 3)
        z_tick_labels = np.linspace(0, num_z-1, 3)

        x_tick_vals = np.flip(np.linspace(xlim[0], xlim[1], 3))
        y_tick_vals = np.flip(np.linspace(ylim[0], ylim[1], 3))
        z_tick_vals = np.linspace(zlim[0], zlim[1], 3)

        axis_labels = [x_tick_labels, y_tick_labels, z_tick_labels]
        axis_vals = [x_tick_vals, y_tick_vals, z_tick_vals]

        return volume_filter, axis_labels, axis_vals


def filter_volume(volume: np.ndarray, num_x, num_y) -> np.ndarray:
        volume_unpadded = 2 * volume[:, :, 1:-1] - volume[:, :, :-2] - volume[:, :, 2:]
        zero_pad = np.zeros((num_x, num_y, 1))
        volume_padded = np.concatenate([zero_pad, volume_unpadded, zero_pad], axis=-1)
        return volume_padded


PIXEL_SIZE = [16.8e-6, 38.8e-6]  # Pixel size in meters
FOCAL_LENGTH = 400e-6  # Focal length in meters

C = 3e8  # Speed of light in meters per second

BIN_RESOLUTION = 100e-12
FOV_X = 41  # Field of view in x-direction [deg]
FOV_Y = 52  # Field of view in y-direction [deg]

def extract_point_cloud_interpolated(hists: np.ndarray, N: int) -> np.ndarray:
    fx, fy = FOCAL_LENGTH / PIXEL_SIZE[0] * 3, FOCAL_LENGTH / PIXEL_SIZE[1] * 2
    cx, cy = hists.shape[1] / 2, hists.shape[0] / 2
    points = []
    H, W, _ = hists.shape
    for i in range(H):
        for j in range(W):
            bin_index = np.argmax(hists[i, j])
            # TODO: why / 2?
            t = bin_index * BIN_RESOLUTION / 2
            depth = (C * t) / 2
            for u in range(N):
                for v in range(N):
                    x_sub = j + (v + 0.5) / N
                    y_sub = i + (u + 0.5) / N
                    X = (x_sub - cx) * depth / fx
                    Y = (y_sub - cy) * depth / fy
                    Z = depth
                    points.append([X, Y, Z])
    return np.array(points)