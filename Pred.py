#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from numpy_ringbuffer import RingBuffer
import redis, time, signal, sys
import os
import json
from sklearn.cluster import KMeans


# In[2]:


def str_to_vec(s):
    return np.array([float(x) for x in s[1:-1].split(' ')])

def key_to_vec(r, key_name):
    return str_to_vec(r.get(key_name))

def vec_to_str(v):
    return '[' + ','.join(x for x in v) + ']'
def signal_handler(signal, frame):
    global runloop
    runloop = False
    print(' ... Exiting')


# In[3]:


def predict(x, y, z, r, x_cross = -0.665):
#     pz = z + 4.9 * x * x
    coeffz = np.polyfit(x, z, 2)
    coeffy = np.polyfit(x, y, 1)
    z_cross = coeffz[2] + coeffz[1] * x_cross + coeffz[0] * x_cross *x_cross
    y_cross = coeffy[1] + coeffy[0] * x_cross

    y_cross -= -1.138
    z_cross -= 0.2
    r.set("sai2::FrankaPanda::control::target_position", vec_to_str([str(x_cross), str(y_cross), str(z_cross)]))
#     return np.array([x_cross, y_cross, z_cross])
    return y_cross, z_cross
#     print(y_cross, z_cross)


# In[20]:
if __name__ == "__main__":

    signal.signal(signal.SIGINT, signal_handler)
    runloop = True

    OPTI_TIMESTAMP_KEY = "sai2::optitrack::timestamp"
    OPTI_POS_KEY = "sai2::optitrack::pos_single_markers"
    r_server = redis.StrictRedis(host='localhost', port=6379, db=0)

    buffersize = 20
    minpts = 10

    buff = RingBuffer(capacity=buffersize, dtype=(np.float64, 4))

    ptcount = 0
    prev_time = 0

    pred_frequency = 500.0  # Hz
    pred_period = 1.0/pred_frequency
    t_init = time.time()
    t = t_init

    timeout_time = 1.5 # unit: second

    while runloop:
        t += pred_period

        opti_time = json.loads(r_server.get(OPTI_TIMESTAMP_KEY).decode("utf-8"))
        poslist = r_server.get(OPTI_POS_KEY).decode("utf-8").split(";")

        if opti_time != prev_time:
            opti_pos = json.loads("["+",".join(("".join(poslist)).split(" "))+"]")
            prev_time = opti_time

            # remove all timeout points
            while len(buff) > 0 and opti_time - buff[0, 3] >= timeout_time:
                buff.popleft()

            # check current point and add it to the end of buffer
            for i in range(len(poslist)):
                pos = np.array([opti_pos[3*i+2], opti_pos[3*i], opti_pos[3*i+1]])

                if pos[2] > 0.2 and pos[0] < 1.4 and pos[0] > -0.665 and pos[1] < -0.638 and pos[1] > -1.638:
                    buff.append(np.array([pos[0], pos[1], pos[2], opti_time]).reshape(1, 4))

                    ptcount = min(ptcount + 1, buffersize)
                    if ptcount > minpts:
                        xp = np.array(buff[:ptcount, 0])
                        yp = np.array(buff[:ptcount, 1])
                        zp = np.array(buff[:ptcount, 2])
                        yc, zc = predict(xp, yp, zp, r_server)
                        print(yc, zc)
                        break


# # In[4]:


# # file = np.loadtxt("opti/ball_traj_06-05-19_12-54-26")
# file = np.loadtxt("opti/ball_traj_06-05-19_12-54-11")
# # file = np.loadtxt("opti/ball_traj_06-05-19_12-53-59")

# xhis = file[:,1]
# yhis = file[:,2]
# zhis = file[:,3]

# ychis = np.empty(file.shape[0])
# zchis = np.empty(file.shape[0])

# buffersize = 10
# minpts = 2

# xbuff = RingBuffer(capacity=buffersize)
# ybuff = RingBuffer(capacity=buffersize)
# zbuff = RingBuffer(capacity=buffersize)

# ptcount = 0
# for i in range(file.shape[0]):
#     xbuff.appendleft(xhis[i])
#     ybuff.appendleft(yhis[i])
#     zbuff.appendleft(zhis[i])

#     ptcount = min(ptcount + 1, buffersize)

#     if ptcount > minpts:
#         xp = np.array(xbuff[:ptcount])
#         yp = np.array(ybuff[:ptcount])
#         zp = np.array(zbuff[:ptcount])
#         yc, zc = predict(xp, yp, zp, redis, xhis[-1])
#         ychis[i] = yc
#         zchis[i] = zc


# # In[5]:





# # In[ ]:




