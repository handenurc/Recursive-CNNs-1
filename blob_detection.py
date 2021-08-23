import cv2
import numpy as np
import os
import matplotlib.pyplot as plt


def blob_detector(img):

    # change to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    rows = gray.shape[0] #cols indeed!!!


    # Apply adaptive threshold on gray image-------------------------------------------------------
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 101, 3)
    #----------------------------------------------------------------------------------------------

    # Apply morphology close with rect element-----------------------------------------------------
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed_with_rect = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel2)
    #----------------------------------------------------------------------------------------------

    # Erode with ellipse element-------------------------------------------------------------------
    kernel4 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (12, 12))
    eroded_with_ellipse = cv2.erode(closed_with_rect, kernel4)
    #----------------------------------------------------------------------------------------------

    edges = cv2.Canny(eroded_with_ellipse, 50, 255)

    # Close edges with rect element----------------------------------------------------------------
    kernel_for_edges = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 20))
    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_for_edges)
    #----------------------------------------------------------------------------------------------

    # fig1 = plt.figure("Morphology Steps")

    # plt.subplot(2, 3, 1)
    # plt.title("original image")
    # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # plt.subplot(2, 3, 2)
    # plt.title("adaptive threshold")
    # plt.imshow(cv2.cvtColor(thresh, cv2.COLOR_BGR2RGB))
    # plt.subplot(2, 3, 3)
    # plt.title("closed with rect")
    # plt.imshow(cv2.cvtColor(closed_with_rect, cv2.COLOR_BGR2RGB))
    # plt.subplot(2, 3, 4)
    # plt.title("eroded with ellipse")
    # plt.imshow(cv2.cvtColor(eroded_with_ellipse, cv2.COLOR_BGR2RGB))
    # plt.subplot(2, 3, 5)
    # plt.title("edges")
    # plt.imshow(cv2.cvtColor(edges, cv2.COLOR_BGR2RGB))
    # plt.subplot(2, 3, 6)
    # plt.title("closed edges")
    # plt.imshow(cv2.cvtColor(closed_edges, cv2.COLOR_BGR2RGB))
    # plt.tight_layout()

    eroded_with_ellipse = (255 - eroded_with_ellipse)

    # Detect Hough circles--------------------------------------------------------------------------
    circles = cv2.HoughCircles(closed_edges, cv2.HOUGH_GRADIENT, 1, rows / 2,
                                param1=10, param2=10,
                                minRadius=int(rows/16), maxRadius=int(rows/8))
    # ----------------------------------------------------------------------------------------------

    # Detect Contours-------------------------------------------------------------------------------
    cnts_edges, hierarchy = cv2.findContours(
        eroded_with_ellipse, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # ----------------------------------------------------------------------------------------------

    eccentricities = []
    distances = []
    orientations = []
    areaArray = []
    cnt_areas = []

    if cnts_edges is not None:
        for cnt in cnts_edges:
            # Draw contour with magenta
            cv2.drawContours(img, [cnt], -1, (255, 0, 255), 1)
            # fit blue ellipse on contour
            ellipse = cv2.fitEllipse(cnt)
            cv2.ellipse(img, ellipse, (255,0, 0), 1, cv2.LINE_AA)
            # fig = plt.figure("Blobs")
            # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            # plt.show()
            # orientation gives the angle
            (center, axes, orientation) = ellipse
            orientations.append(orientation)

            # find eccentricity of the fitted circle
            majoraxis_length = max(axes)
            minoraxis_length = min(axes)
            eccentricity = (
                np.sqrt(1-(minoraxis_length/majoraxis_length)**2))
            eccentricities.append(eccentricity)
            # print("eccentricity: ", eccentricity)

            # find contour area
            cnt_area = cv2.contourArea(cnt)
            # print("cnt area: ", cnt_area)
            cnt_areas.append(cnt_area)

            # compute moment of the contour
            M = cv2.moments(cnt)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            area = cv2.contourArea(cnt)

            # find equivalent diameter of the contour
            equi_diameter = np.sqrt(4*area/np.pi)

            if circles is not None:

                circles = np.uint16(np.around(circles))

                for i in circles[0, :]:
                    circ_center = (i[0], i[1])
                    # circle center point
                    cv2.circle(img, circ_center, 2, (42, 50, 163), 2)
                    # circle outline with radius
                    radius = i[2]
                    cv2.circle(img, circ_center, radius, (42, 50, 163), 2)

                    dist = np.sqrt(abs(cX-i[0])**2 + abs(cY-i[1]**2))
                    diameter_ratio = equi_diameter/radius

                    if ((dist <= rows/2) & (eccentricity <= 0.5)):
                        # & (cnt_area >= (rows*(gray.shape[1]/50))):
                        # Draw the fitted ellipse in turquoise color
                        cv2.ellipse(img, ellipse, (255, 255, 40), 1, cv2.LINE_AA)
                        # Draw the actual contour in thick green color
                        cv2.drawContours(img, [cnt], -1, (0, 255, 0), 1)
                        # Draw moment point
                        cv2.circle(img, (cX, cY), 1, (255, 255, 255), -1)
                        # Put text as center of moment
                        cv2.putText(img, "center of moment", (cX - 20, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        print('circ_center with blob')
                        return cX,cY,radius
                    else:
                        return circ_center[0],circ_center[1],radius
            else:
                # When there is no circle detected by Hough Transform:
                if ((equi_diameter >= rows/8) & (eccentricity <= 0.5)):
                    # Draw the fitted ellipse in turquoise color
                    cv2.ellipse(img, ellipse, (255, 255, 40), 1, cv2.LINE_AA)
                    # Draw the actual contour in thick green color
                    cv2.drawContours(img, [cnt], -1, (0, 255, 0), 1)
                    # Draw moment point
                    cv2.circle(img, (cX, cY), 1, (255, 255, 255), -1)
                    # Put text as center of moment
                    cv2.putText(img, "center of moment", (cX - 20, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    print('cX, cY')
                    return (cX,cY,equi_diameter/2)
    else:
        if circles is not None:
            
            circles = np.uint16(np.around(circles))
            
            for i in circles[0, :]:
                circ_center = (i[0], i[1])
                # circle center
                cv2.circle(img, circ_center, 1, (0, 250, 13), 2)
                # circle outline
                radius = i[2]
                cv2.circle(img, circ_center,
                            radius, (0, 250, 13), 1)
                # circle2 = plt.Circle((circ_center), radius, color='b', fill=False)
                print('circ_center with no blob')
                return circ_center,radius
        else:
            print('Nothing found on image')
    plt.show()