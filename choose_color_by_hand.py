
import numpy as np
import cv2
import math

number_of_frames = 10
blue_counter = number_of_frames
green_counter = number_of_frames
red_counter = number_of_frames
yellow_counter = number_of_frames
colorIndex = 0  # the color blue by default


def change_color(frame):

    global number_of_frames
    global blue_counter, green_counter, red_counter, yellow_counter, colorIndex

    cv2.rectangle(frame, (400, 200), (635, 5), (0, 255, 0), 0)
    crop_img = frame[5:200, 400:635]
    grey = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(grey, (45, 45), 0)

    _, thresh1 = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    _, contours, hierarchy = cv2.findContours(thresh1.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # find contour with biggest area
    cnt = max(contours, key=lambda x: cv2.contourArea(x))

    # finding convex hull
    hull = cv2.convexHull(cnt)
    # drawing contours
    drawing = np.zeros(crop_img.shape, np.uint8)
    cv2.drawContours(drawing, [cnt], 0, (0, 255, 0), 0)
    cv2.drawContours(drawing, [hull], 0, (0, 0, 255), 0)
    # finding convex hull
    hull = cv2.convexHull(cnt, returnPoints=False)
    # finding convexity defects
    defects = cv2.convexityDefects(cnt, hull)
    count_defects = 0
    cv2.drawContours(thresh1, contours, -1, (0, 255, 0), 3)

    if defects is not None:
        for i in range(defects.shape[0]):
            s, e, f, _ = defects[i, 0]

            start = tuple(cnt[s][0])
            end = tuple(cnt[e][0])
            far = tuple(cnt[f][0])

            # find length of all sides of triangle
            a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
            b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
            c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)

            angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 57

            # ignore angles > 80 and highlight the others
            if angle <= 80:
                count_defects += 1
                cv2.circle(crop_img, far, 1, [0, 0, 255], -1)

        # define actions required
        if count_defects == 1:
            red_counter = number_of_frames
            green_counter = number_of_frames
            yellow_counter = number_of_frames
            blue_counter -= 1
            print(2)
            if blue_counter == 0:
                colorIndex = 0  # Blue
                blue_counter = number_of_frames
        elif count_defects == 2:
            print(3)
            blue_counter = number_of_frames
            green_counter -= 1
            red_counter = number_of_frames
            yellow_counter = number_of_frames
            if green_counter == 0:
                colorIndex = 1  # Green
                green_counter = number_of_frames

        elif count_defects == 3:
            print(4)
            blue_counter = number_of_frames
            green_counter = number_of_frames
            red_counter -= 1
            yellow_counter = number_of_frames

            if red_counter == 0:
                colorIndex = 2  # Red
                red_counter = number_of_frames
        elif count_defects == 4:
            print(5)
            blue_counter = number_of_frames
            green_counter = number_of_frames
            red_counter -= number_of_frames
            yellow_counter -= 1

            if yellow_counter == 0:
                colorIndex = 3  # Yellow
                yellow_counter = number_of_frames
        else:
            print("No hand")

        # all_img = np.hstack((drawing
        cv2.imshow('Convex', drawing)
        return colorIndex