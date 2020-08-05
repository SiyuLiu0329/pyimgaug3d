from pyimgaug3d.augmentation.src.base_augmentation import BaseAugmentation
import tensorflow_addons as tfa
import tensorflow as tf
import numpy as np

# credit: https://stackoverflow.com/users/2187530/warbean
def griddify(rect, w_div, h_div):
    w = rect[2] - rect[0]
    h = rect[3] - rect[1]
    x_step = w / float(w_div)
    y_step = h / float(h_div)
    y = rect[1]
    grid_vertex_matrix = []
    for _ in range(h_div + 1):
        grid_vertex_matrix.append([])
        x = rect[0]
        for _ in range(w_div + 1):
            grid_vertex_matrix[-1].append([int(x), int(y)])
            x += x_step
        y += y_step
    grid = np.array(grid_vertex_matrix)
    return grid

def distort_grid(org_grid, max_shift):
    new_grid = np.copy(org_grid)
    x_min = np.min(new_grid[:, :, 0])
    y_min = np.min(new_grid[:, :, 1])
    x_max = np.max(new_grid[:, :, 0])
    y_max = np.max(new_grid[:, :, 1])
    new_grid += np.random.randint(- max_shift, max_shift + 1, new_grid.shape)
    new_grid[:, :, 0] = np.maximum(x_min, new_grid[:, :, 0])
    new_grid[:, :, 1] = np.maximum(y_min, new_grid[:, :, 1])
    new_grid[:, :, 0] = np.minimum(x_max, new_grid[:, :, 0])
    new_grid[:, :, 1] = np.minimum(y_max, new_grid[:, :, 1])
    return new_grid

class GridWarp(BaseAugmentation):
    def __init__(self, grid=(4, 4, 4), max_shift=10):
        self.grid = grid
        self.max_shift = max_shift

    def _merge_control_points(self, grid):
        pts = []
        for i in grid:
            for j in i:
                pts.append(j)
        pts = np.array(pts, dtype='float32')[None, :, :]
        return pts

    def _get_control_points(self, h, w, grid):
        x, y = grid
        dst_grid = griddify(((0, 0, h, w)), x, y)
        src_grid = distort_grid(dst_grid, self.max_shift)
        src = self._merge_control_points(src_grid)
        dst = self._merge_control_points(dst_grid)
        return src, dst

    def _get_ax_params(self, img):

        h, w, d, _ = img.shape
        params = {
            0: [
                lambda img: img, 
                self._get_control_points(w, d, (self.grid[1], self.grid[2]))
            ],
            1: [
                lambda img: tf.transpose(img, [1, 0, 2, 3]), 
                self._get_control_points(h, d, (self.grid[0], self.grid[2]))
            ],
            2: [
                lambda img: tf.transpose(img, [2, 1, 0, 3]), 
                self._get_control_points(w, h, (self.grid[1], self.grid[0]))
            ]
        }
        
        return params


    # convert to 3D
    def _warp_channel(self, img, params):
        img = tf.convert_to_tensor(img, dtype='float32')
        def warp_axis(img, axis):
            swap, pts = params[axis]
            img = swap(img)
            src, dst = pts
            src = np.concatenate([src] * img.shape[0])
            dst = np.concatenate([dst] * img.shape[0])
            img, _ = tfa.image.sparse_image_warp(
                img, 
                src,
                dst, interpolation_order=3
                
            )
            img = swap(img)
            return img
        for a in range(3):
            img = warp_axis(img, a)
        return img

    def __call__(self, imgs):
        img = imgs[0]
        params = self._get_ax_params(img)
        for i, img in enumerate(imgs):
            c = img.shape[-1]
            if c == 1:
                res = self._warp_channel(img, params)
            else:
                res = []
                for j in range(c):
                    res.append(self._warp_channel(img[:, :, :, j: j+1], params))
                res = tf.concat(res, axis=3)
            imgs[i] = res
        return imgs