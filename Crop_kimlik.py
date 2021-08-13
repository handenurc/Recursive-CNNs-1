import cv2 as cv
import numpy as np
import os
import math
import matplotlib.pyplot as plt
from blob_detection import blob_detector
from hough import hough_line_detector
from probabilistic_hough import probabilistic_houghline
from rotating_without_losing_boundaries import rotate_image
directory = "C:\\Users\\handenur.caliskan\\Desktop\\highres_double"
# img = cv.imread('C:\\Users\\handenur.caliskan\\Desktop\\highres_double\\double27.jpg')
# \\GitHub\\DRBox_keras-1\\training_kimlik_new_v1'

# for filename in os.listdir(directory):
#     if filename.endswith(".png"):
#         # read image
#         input_path = os.path.join(directory, filename)
#         img = cv.imread(input_path)
def crop_kimlik(img):

    if img is not None:
        cols = img.shape[0]
        rows = img.shape[1]
        copy_img = img.copy()
        (cx, cy, radius) = blob_detector(img)
        fig = plt.figure("Circles detected")
        plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))

        directions, xs_mean, ys_mean = probabilistic_houghline(copy_img)

        directions_mod90 = np.array(directions) % 90

        for i in range(0,len(directions_mod90)):
            if directions_mod90[i]> 80:
                directions_mod90[i] = 90 - directions_mod90[i]

        directions_mod90 = sorted(directions_mod90)
        # clusters_img = hough_line_detector(copy_img)
        # for i in range(0,len(clusters_img)):
        #     if clusters_img[i][0] < 80:
        #         alpha1 = clusters_img[i][0]
        #     elif 80 <= clusters_img[i][0] < 190:
        #         alpha2 = clusters_img[i][0]
        #     elif 190 <= clusters_img[i][0] < 280:
        #         alpha3 = clusters_img[i][0]
        #     elif 280 <= clusters_img[i][0] < 360:
        #         alpha4 = clusters_img[i][0]
        # print(alpha1)
        # cropping_point = cx+(radius*5)*(1/math.cos(math.radians(alpha1)))
        # ----------------------------------------------------------------
        
        # Which one is going to be applied in which case?

        
        groups = []
        eps = 10
        curr_point = directions_mod90[0]
        curr_cluster = [curr_point]
        for point in directions_mod90[1:]:
            if point <= curr_point + eps:
                curr_cluster.append(point)
            else:
                groups.append(curr_cluster)
                curr_cluster = [point]
            curr_point = point
        groups.append(curr_cluster)
        print(groups)

        # ----------------------------------------------------------------
        if cx < xs_mean:
            if len(groups) == 1:
                alpha1 = max(groups[0])
                if 20 <= alpha1 < 70:
                    rotated_img = rotate_image(img,(360-alpha1))
                    crop_point1 = cx+(radius*8)
                    cropped_img1 = rotated_img[:, int(crop_point1):rotated_img.shape[1]]
                else:
                    crop_point1 = cx+(radius*10)*math.cos(math.radians(abs(60-alpha1)))
                    # if 90 <= alpha1 < 180:
                    #     crop_point = cx+(radius*8)*math.cos(math.radians(abs(180-directions[1])))
                    cropped_img1 = img[:, int(crop_point1):rows]

                

            if len(groups) == 2:
                alpha1 = max(groups[0])
                crop_point1 = cx+(radius*10)*math.cos(math.radians(abs(60-alpha1)))
                cropped_img1 = img[:, int(crop_point1):rows]

                if len(groups[1]) == 2:
                    alpha2 = max(groups[1])
                    crop_point2 = cx+(radius*10)*math.cos(math.radians(abs(60-alpha2)))
                    cropped_img2 = img[:, int(crop_point2):rows]
                    # fig2 = plt.figure("Cropped img 2")
                    # plt.imshow(cv.cvtColor(cropped_img2, cv.COLOR_BGR2RGB))
                    
                else:
                    pass
                
            # if len(directions) >= 3:
            #     alpha3 = directions[2]
            #     crop_point3 = cx+(radius*4)*math.cos(math.radians(abs(60-(270-directions[2]))))
            # if len(directions) == 4:
            #     alpha4 = directions[3]
            #     crop_point4 = cx+(radius*6)*math.cos(math.radians(abs(60-(360-directions[3]))))

            

        else:
            if len(groups) == 1:
                alpha1 = max(groups[0])
                if 20 <= alpha1 < 70:
                    rotated_img = rotate_image(img,(360-alpha1))
                    crop_point1 = cx-(radius*4)
                    cropped_img1 = rotated_img[:, 0:int(crop_point1)]
                else:
                    crop_point1 = cx-(radius*4)*math.cos(math.radians(abs(60-alpha1)))
                # if 90 <= alpha1 < 180:
                #     crop_point = cx+(radius*8)*math.cos(math.radians(abs(180-directions[1])))
                    cropped_img1 = img[:, 0:int(crop_point1)]
                    fig1 = plt.figure("Cropped img 1")
                    plt.imshow(cv.cvtColor(cropped_img1, cv.COLOR_BGR2RGB))
                

            if len(groups) == 2:
                alpha1 = max(groups[0])
                crop_point1 = cx-(radius*4)*math.cos(math.radians(abs(60-alpha1)))
                cropped_img1 = img[:, 0:int(crop_point1)]

                if len(groups[1]) == 2:
                    alpha2 = max(groups[1])
                    crop_point2 = cx-(radius*4)*math.cos(math.radians(abs(60-alpha2)))
                    cropped_img2 = img[:, 0:int(crop_point2)]
                    # fig2 = plt.figure("Cropped img 2")
                    # plt.imshow(cv.cvtColor(cropped_img2, cv.COLOR_BGR2RGB))
                    
                else:
                    pass


            #     crop_point2 = cx+(radius*6)*math.cos(math.radians(abs(180-directions[1])))
            # elif len(directions) >= 3:
            #     alpha3 = directions[2]
            #     crop_point3 = cx+(radius*10)*math.cos(math.radians(abs(60-(270-directions[2]))))
            # elif len(directions) >= 4:
            #     alpha4 = directions[3]
            #     crop_point4 = cx+(radius*8)*math.cos(math.radians(abs(60-(360-directions[3]))))
            # crop_point needs to be assigned somewhat a different value in this case
            
            
            

        # if cropping_point < rows/2:
            
        #     cropped_img = img[:, int(cropping_point):rows]
        # else:
        #     cropping_point = cx-(radius*2)*(1/math.cos(math.radians(alpha1)))
        #     cropped_img = img[:, 0:int(cropping_point)]

        print(cx,cy,radius)
        print(directions, xs_mean, ys_mean)

        fig1 = plt.figure("Cropped img 1")
        plt.imshow(cv.cvtColor(cropped_img1, cv.COLOR_BGR2RGB))

        # fig2 = plt.figure("Cropped img 2")
        # plt.imshow(cv.cvtColor(cropped_img2, cv.COLOR_BGR2RGB))
        plt.show()
    return cropped_img1