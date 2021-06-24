import cv2
import numpy as np
import matplotlib.pyplot as plt

def perspective_transform(image,corner_addresses):

    rows,cols,ch = image.shape

    new_sizes = np.float32([[0,0],[300,0],[0,450],[300,450]])

    M = cv2.getPerspectiveTransform(np.float32(np.array(corner_addresses)),new_sizes)
    dst = cv2.warpPerspective(image,M,(300,450))

    # Coordinates of the rectangular area needs to be covered 
    # x1,y1 --> top left x2,y2 --> bottom right

    x1 = 100
    y1 = 20
    x2 = 200
    y2 = 55

    points = np.array([[x1, y1], [x2, y1],  [x2, y2], [x1, y2]],
                    dtype=np.int32)

    cv2.rectangle(dst, (x1, y1), (x2, y2), (255,0,0), 2)
    cv2.fillPoly(dst, [points], (255, 0, 0))

    return dst


def main():

    if __name__ == "__main__":

        img1 = cv2.imread('C:\\Users\\hcaliskan\\Recursive-CNNs\\TrainedModel\\kimlik57.jpg')

        tl = [110,100]
        tr = [350,80]
        bl = [160,450]
        br = [400, 390]

        transformed_image = perspective_transform(img1,tl,tr,bl,br)
        cv2.imshow("transformed image",transformed_image)
        cv2.waitKey(0)
        # To destroy and remove all created GUI windows from
        #screen and memory
        cv2.destroyAllWindows()
        main()


