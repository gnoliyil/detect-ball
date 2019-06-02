import logging
import cv2
import numpy as np
import pcl
import json
import freenect
import frame_convert2
import matplotlib.pylab as plt
import matplotlib
import imutils

import redis
import frame_convert2
import importlib
import time
from sklearn.cluster import KMeans
from scipy.optimize import curve_fit

def str_to_vec(s):
    return np.array([float(x) for x in s[1:-1].split(',')])

def key_to_vec(r, key_name):
    return str_to_vec(r.get(key_name))

def vec_to_str(v):
    return '[' + ','.join(x for x in v) + ']'

MAX_ALLOWED_DEPTH = 4.0

# https://jekel.me/2015/Least-Squares-Sphere-Fit/
def sphereFit(spX,spY,spZ):
    #   Assemble the A matrix
    spX = np.array(spX)
    spY = np.array(spY)
    spZ = np.array(spZ)
    A = np.zeros((len(spX),4))
    A[:,0] = spX*2
    A[:,1] = spY*2
    A[:,2] = spZ*2
    A[:,3] = 1

    #   Assemble the f matrix
    f = np.zeros((len(spX),1))
    f[:,0] = (spX*spX) + (spY*spY) + (spZ*spZ)
    C, residules, rank, singval = np.linalg.lstsq(A, f, rcond=None)

    #   solve for the radius
    t = (C[0]*C[0])+(C[1]*C[1])+(C[2]*C[2])+C[3]
    radius = np.sqrt(t)

    return radius, C[0], C[1], C[2]

reference_color = np.array([178, 181, 60]) / 255.
hue = matplotlib.colors.rgb_to_hsv(reference_color)[0]

class FindBall():

    def __init__(self):
        pass


    def get_depth(self):
        """
        Get depth image in millimeters (valued 0-10000)
        """
        value = freenect.sync_get_depth(format=freenect.DEPTH_REGISTERED)[0]
        return value

    def get_video(self):
        """
        Get BGR frames (valued 0-255)
        """
        return frame_convert2.video_cv(freenect.sync_get_video()[0])

    def find_color_pixel_mask(self, rgb):
        hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV) / 255.
        hue_hsv, sat_hsv, bright_hsv = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

        height, width = hsv.shape[0], hsv.shape[1]
        grid = np.zeros((height, width), dtype=np.uint8)
        grid[(hue_hsv < (hue + 15 / 360.)) & (hue_hsv > (hue - 25 / 360.)) & \
            (sat_hsv > 0.4) & (bright_hsv > 0.35)] = 255
        return grid

    def to_xyz(self, rgb, depth):
        fx, fy = 529.215, 525.5639
        cx, cy = 328.942, 267.480
        z = depth[rgb[:, 0], rgb[:, 1]]
        x = ((rgb[:, 1] - cx) * z) / fx
        y = ((rgb[:, 0] - cy) * z) / fy
        x, y, z = x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)
        xyz = np.hstack([x, y, z])
        xyz = xyz.astype(np.float32)
        return xyz[xyz[:, 2] > 0]

    def find_contour(self, mask, rgb):
        contours = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        c = max(contours, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        return (x, y), radius

    def get_mask_ball(self, mask, center, radius, depth):
        x, y = center
        mask_circle = np.zeros(mask.shape, dtype=np.uint8)
        mask_circle = cv2.circle(mask_circle, (int(x), int(y)), int(radius * 1.5), 255, thickness=-1)
        mask_ball_color = mask_circle & mask
        mask_ball_color = mask_ball_color & (depth > 0) & (depth < MAX_ALLOWED_DEPTH)

        mbc_sum = np.count_nonzero(mask_ball_color)
        mc_sum = np.count_nonzero(mask_circle)

        area = mbc_sum / mc_sum
        if mbc_sum < 20 or area < 0.2:
            return None
        else:
            return mask_ball_color

    def get_foreground_from_mask(self, center, radius, depth):
        x, y = center
        x, y, radius = int(x), int(y), int(radius)
        subimg = depth[y-radius:y+radius, x-radius:x+radius]
        mask = (subimg > 0) & (subimg < MAX_ALLOWED_DEPTH)
        positions = np.argwhere(mask > 0)
        sub_depth = np.array([subimg[u, v] for u, v in positions]).reshape(-1, 1)
        if len(positions) > 0:
            kmeans = KMeans(n_clusters=2, random_state=0).fit(sub_depth)
            argmin = np.argmin(kmeans.cluster_centers_)
            return positions[kmeans.labels_ == argmin] + [y-radius, x-radius]
        else:
            return None

    def filter_outlier(self, pc):
        n_pc = len(pc)
        cloud = pcl.PointCloud()
        cloud_filtered = pcl.PointCloud()
        cloud.from_array(pc)

        kdtree = cloud.make_kdtree_flann()
        [ind, sqdist] = kdtree.nearest_k_search_for_cloud(cloud, int(n_pc / 10))
        valid_mask = sqdist.max(axis=-1) < 1e-2
        pc_filtered = pc[valid_mask]
        return pc_filtered

    def find_ball(self, image=None, depth=None):
        if image is None or depth is None:
            image = self.get_video()
            depth = self.get_depth() / 1000

        color_mask = self.find_color_pixel_mask(image)
        center, radius = self.find_contour(color_mask, image)
        fg_mask = self.get_foreground_from_mask(center, radius * 1.5, depth)
        mask_ball = self.get_mask_ball(color_mask, center, radius, depth)

        if fg_mask is None or mask_ball is None:
            msg = ""
            if fg_mask is None: msg += "fg_mask is None "
            if mask_ball is None: msg += "mask_ball is None "
            raise Exception("ball not found: {}".format(msg))

        pc_mask = self.to_xyz(fg_mask, depth)
        pc_color = self.to_xyz(np.argwhere(mask_ball > 0), depth)

        # # Used for debugging:
        # mask_fg = np.zeros((480, 640))
        # for x,y in fg_mask: mask_fg[x, y] = 0.5
        # for x,y in np.argwhere(mask_ball > 0): mask_fg[x, y] += 0.5
        # plt.imshow(mask_fg, cmap='gist_rainbow', alpha=0.5)
        # plt.imshow(rgb, alpha=0.5)
        # plt.imshow(depth, alpha=0.5)

        rm, cxm, cym, czm = sphereFit(pc_mask[:, 0], pc_mask[:, 1], pc_mask[:, 2])
        rc, cxc, cyc, czc = sphereFit(pc_color[:, 0], pc_color[:, 1], pc_color[:, 2])
        rm_valid = rm > 0.01 and rm < 0.3
        rc_valid = rc > 0.01 and rc < 0.3

        if rm_valid and not rc_valid:
            return rm, cxm, cym, czm, 'm'
        if rc_valid and not rm_valid:
            return rc, cxc, cyc, czc, 'c'
        if rc_valid and rm_valid:
            if rc < rm: return rc, cxc, cyc, czc, 'c'
            if rm <= rc: return rm, cxm, cym, czm, 'm'
        raise Exception("ball not found: ball size invalid!")

if __name__ == "__main__":
    BALL_POSITION_KEY  = 'sai2::cs225a::cv::ball_position::xyz'
    BALL_TIMESTAMP_KEY = 'sai2::cs225a::cv::ball_position::time'

    redis = redis.Redis(charset='utf-8', decode_responses=True)
    fb = FindBall()
    while True:
        try:
            r, x, y, z, type_ = fb.find_ball()
            logging.info("Ball found! Radius = {:.3f}, [x, y, z] = [{:.3f}, {:.3f}, {:.3f}]".format(r, x, y, z))

            bp_str = [x, y, z]
            redis.set(BALL_POSITION_KEY, vec_to_str(bp_str))
            redis.set(BALL_TIMESTAMP_KEY, str(time.time()))
        except Exception as e:
            logging.error(e)

