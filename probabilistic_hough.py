from hough import erode_with_ellipse
from blob_detection import blob_detector
import os
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
import math
from collections import Counter
from statistics import mean

from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
import heapq

# input_path = 'C:\\Users\\handenur.caliskan\\Desktop\\highres_double\\double20.jpg'

# directory = "C:\\Users\\handenur.caliskan\\Desktop\\highres_double"
# img = cv.imread('C:\\Users\\handenur.caliskan\\Desktop\\highres_double\\double30.jpg')
# \\GitHub\\DRBox_keras-1\\training_kimlik_new_v1'

# for filename in os.listdir(directory):
#     if filename.endswith(".jpg"):
#         # read image

#         input_path = os.path.join(directory, filename)

def probabilistic_houghline(img):
        # img = cv.imread(input_path)
        # img_copy = img.copy()

    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

    # Step 1: Blurring
    blur = cv.GaussianBlur(gray,(11,11),0)

    # Step 2: Thresholding
    thresh = cv.adaptiveThreshold(blur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 5)

    # fig1 = plt.figure('thresh')
    # plt.imshow(cv.cvtColor(thresh, cv.COLOR_BGR2RGB))

    # Step 3: Canny Edge Detection:
    edges1 = cv.Canny(thresh, 50, 255)
    # edges2 = cv.Canny(dilated_with_rect, 50, 255)

    # fig4 = plt.figure('edges 1')
    # plt.imshow(cv.cvtColor(edges1, cv.COLOR_BGR2RGB))

    # plt.show()

    # Step 4: Apply Probabilistic Hough Transform
    # lines1 = cv.HoughLinesP(edges1, 1, np.pi / 180, 100, None, 75, 10)
    lines1 = cv.HoughLinesP(edges1, 1, np.pi / 180, 100, None, 75, 10)

    angles = []

    # Step 5: Draw Detected Lines on the Original Image, Find and Cluster Their Unique Angles
    if lines1 is not None:
        xs = []
        ys = []
        for line in lines1:
            leftx, boty, rightx, topy = line[0]
            xs.extend([leftx, rightx])
            ys.extend([boty,topy])
            angle = math.degrees(math.atan2((topy-boty),(rightx-leftx)))
            if angle < 0:
                angle = angle + 180
            if 0 <= angle < 10:
                cv.line(img, (leftx, boty), (rightx,topy), (255, 0, 0), 1)
            elif 80 <= angle < 100:
                cv.line(img, (leftx, boty), (rightx,topy), (0, 255, 0), 1)
            elif 170 <= angle < 180:
                cv.line(img, (leftx, boty), (rightx,topy), (0, 0, 255), 1)
            else:
                cv.line(img, (leftx, boty), (rightx,topy), (150, 122, 170), 3)
            angles.append(angle)
        unique_angles = np.unique(angles)
        # res = list(zip(*np.unique(angles, return_counts=True)))
        # print(res)
        xs_mean = mean(xs)
        ys_mean = mean(ys)
        if len(unique_angles) == 0:
            pass
        else:
            clusters = []
            eps = 2
            curr_point = unique_angles[0]
            curr_cluster = [curr_point]
            for point in unique_angles[1:]:
                if point <= curr_point + eps:
                    curr_cluster.append(point)
                else:
                    clusters.append(curr_cluster)
                    curr_cluster = [point]
                curr_point = point
            clusters.append(curr_cluster)
            print(clusters)
            most_commons = []
            lengths = []
            
            for cluster in clusters:
                # most_common_avg  = mean(cluster)
                # most_commons.append(most_common_avg)
                lengths.append(len(cluster))
            # clusters_dic = dict(zip(cluster, lengths))
            max4_inds = heapq.nlargest(4, range(len(lengths)), key=lengths.__getitem__)
            dirs = []
            for i in range(0,len(max4_inds)):
                dir = mean(clusters[max4_inds[i]])
                dirs.append(dir)
            dirs_mod90 = np.array(dirs)%90
    print(dirs)
    print(dirs_mod90)
    print(lengths)
    fig6 = plt.figure("Detected Lines edges1 - Probabilistic Hough Transform")
    plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.show()
    return dirs, xs_mean, ys_mean