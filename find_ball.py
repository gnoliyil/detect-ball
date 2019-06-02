import cv2
import numpy as np
import pcl
import json
import freenect
import frame_convert2
import matplotlib.pylab as plt
import matplotlib

import redis
import frame_convert2
import importlib
import time
from scipy.optimize import curve_fit

def str_to_vec(s):
    return np.array([float(x) for x in s[1:-1].split(',')])

def key_to_vec(r, key_name):
    return str_to_vec(r.get(key_name))

def vec_to_str(v):
    return '[' + ','.join(x for x in v) + ']'
    
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
    
reference_color = np.array([122, 120, 35]) / 255.
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
    
    def find_color_pixel_uv(self, rgb):
        hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV) / 255.
        hue_hsv, sat_hsv, bright_hsv = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        
        height, width = hsv.shape[0], hsv.shape[1]
        grid = np.zeros((height, width))    
        grid[(hue_hsv < (hue + 25 / 360.)) & (hue_hsv > (hue - 25 / 360.)) & (sat_hsv > 0.3) & (bright_hsv > 0.3)] = 1
        return np.argwhere(grid > 0)

    def to_xyz(self, rgb, depth):
        fx, fy = 529.215, 525.5639
        cx, cy = 328.942, 267.480
        z = depth[rgb[:, 0], rgb[:, 1]]
        x = ((rgb[:, 0] - cx) * z) / fx
        y = ((rgb[:, 1] - cy) * z) / fy
        x, y, z = x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)
        xyz = np.hstack([x, y, z])
        xyz = xyz.astype(np.float32)
        return xyz[xyz[:, 2] > 0]
        
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


    def find_ball(self):
        image = self.get_video()
        depth = self.get_depth() / 1000
        
        color_uv = self.find_color_pixel_uv(image)
        pc_color = self.to_xyz(color_uv, depth)

        pc_filtered = self.filter_outlier(pc_color)
        
        if len(pc_filtered) == 0:
            raise Exception("ball not found")
        # ball_center = np.mean(pc_filtered, axis=0)
        r, cx, cy, cz = sphereFit(pc_filtered[:, 0], pc_filtered[:, 1], pc_filtered[:, 2])
        return r[0], cx[0], cy[0], cz[0]

if __name__ == "__main__":
    BALL_POSITION_KEY  = 'sai2::cs225a::cv::ball_position::xyz'
    BALL_TIMESTAMP_KEY = 'sai2::cs225a::cv::ball_position::time'
    
    redis = redis.Redis(charset='utf-8', decode_responses=True)
    fb = FindBall()
    while True:
        redis.set(BALL_TIMESTAMP_KEY, str(time.time()))
        ball_pos = fb.find_ball()
        print(ball_pos)
        bp_str = [str(x) for x in ball_pos]
        redis.set(BALL_POSITION_KEY, vec_to_str(bp_str))