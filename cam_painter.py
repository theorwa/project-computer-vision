import numpy as np
import cv2

from choose_color_by_hand import change_color

BLUE_INDEX = 0
GREEN_INDEX = 1
RED_INDEX = 2
YELLOW_INDEX = 3

BLUE_RGB = (255, 0, 0)
GREEN_RGB = (0, 255, 0)
RED_RGB = (0, 0, 255)
YELLOW_RGB = (0, 255, 255)

points = [[[]] for i in range(4)]
index_list = [0 for j in range(4)]
colors = [BLUE_RGB, GREEN_RGB, RED_RGB, YELLOW_RGB]

blue_mask = (np.array([100, 80, 80]), np.array([160, 255, 255]))
kernel = np.ones((5, 5), np.uint8)

def crop():
    global scan_flag, scan_paper
    image = orig_image
    clone = image.copy()
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", click_and_crop)
    while True:
        cv2.imshow("image", image)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("r"):
            image = clone.copy()

        elif key == ord("z"):
            break
    if len(refPt) == 2:
        clear_all()
        cv2.destroyWindow("image")
        scan_flag = True
        roi = clone[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
        scan_paper = roi
        paintWindow[100:roi.shape[0] + 100, 100:roi.shape[1] + 100] = roi


def click_and_crop(event, x, y, flags, param):
    global refPt
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = [(x, y)]
    elif event == cv2.EVENT_LBUTTONUP:
        refPt.append((x, y))
        cv2.rectangle(orig_image, refPt[0], refPt[1], (0, 255, 0), 2)


def clear_all():
    global scan_flag, index_list, points
    points = [[[]] for i in range(4)]
    index_list = [0 for i in range(4)]
    scan_flag = False


def init_board():
    global paintWindow, orig_image, hsv, frame
    paintWindow = np.zeros((471, 636, 3), dtype=np.uint8) + 255
    paintWindow = cv2.rectangle(paintWindow, (40, 1), (140, 65), (0, 0, 0), 2)
    cv2.putText(paintWindow, "CLEAR ALL", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.namedWindow('Paint', cv2.WINDOW_AUTOSIZE)
    paintWindow[:, :, :] = 255
    paintWindow = cv2.rectangle(paintWindow, (40, 1), (140, 65), (0, 0, 0), 2)
    cv2.putText(paintWindow, "CLEAR ALL", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    if scan_flag:
        paintWindow[100:scan_paper.shape[0] + 100, 100:scan_paper.shape[1] + 100] = scan_paper
    frame = cv2.flip(orig_image, 1)
    # Add the coloring options to the frame
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    frame = cv2.rectangle(frame, (40, 1), (140, 65), (122, 122, 122), -1)
    cv2.putText(frame, "CLEAR ALL", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)


def draw_lines():
    global points
    for i in range(len(points)):
        for j in range(len(points[i])):
            for k in range(1, len(points[i][j])):
                if points[i][j][k - 1] is None or points[i][j][k] is None:
                    continue
                # if points[i][j][k - 1][2] != 0:
                if j == len(points[i]) - 1:
                    cv2.line(frame, points[i][j][k - 1][:2], points[i][j][k][:2], colors[i],
                             points[i][j][k - 1][2])
                cv2.line(paintWindow, points[i][j][k - 1][:2], points[i][j][k][:2], colors[i],
                         points[i][j][k - 1][2])


def detect_brush():
    global orig_image, index_list, hsv, points, colors, paintWindow, blue_mask
    # Determine which pixels fall within the blue boundaries and then blur the binary image
    blueMask = cv2.inRange(hsv, blue_mask[0], blue_mask[1])
    blueMask = cv2.erode(blueMask, kernel, iterations=2)
    blueMask = cv2.morphologyEx(blueMask, cv2.MORPH_OPEN, kernel)
    blueMask = cv2.dilate(blueMask, kernel, iterations=1)
    # Find contours in the image
    (_, cnts, _) = cv2.findContours(blueMask.copy(), cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)

    # Check to see if any contours were found
    if len(cnts) > 0:
        # Sort the contours and find the largest one -- we
        # will assume this contour correspondes to the area of the bottle cap
        cnt = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
        # Get the radius of the enclosing circle around the found contour
        ((x, y), radius) = cv2.minEnclosingCircle(cnt)

        if radius < 20:
            rad = 0
        else:
            rad = radius - 20
            rad = int(rad / 5)

        # Draw the circle around the contour
        cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)

        # Get the moments to calculate the center of the contour (in this case Circle)
        M = cv2.moments(cnt)
        center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))

        if center[1] <= 65 and 40 <= center[0] <= 140:  # Clear All
            clear_all()
        else:
            cv2.circle(paintWindow, (int(x), int(y)), int(5), (0, 0, 0), 2)
            cv2.circle(frame, (int(x), int(y)), int(radius), colors[colorIndex], 2)
            if rad > 0:
                points[colorIndex][index_list[colorIndex]].append(center + (rad,))
    else:
        points[colorIndex].append([])
        index_list[colorIndex] += 1


def get_frame(camera):
    global orig_image
    (grabbed, image) = camera.read()
    if not grabbed:
        return False
    orig_image = image
    return True


def main():
    global frame, paintWindow, colorIndex
    camera = cv2.VideoCapture(0)
    while True:
        if not get_frame(camera):
            break
        init_board()
        detect_brush()
        colorIndex = change_color(frame)
        draw_lines()

        cv2.imshow("Tracking", frame)
        cv2.imshow("Paint", paintWindow)

        if cv2.waitKey(1) & 0xFF == ord("c"):
            crop()

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    scan_paper, orig_image, frame, hsv = None, None, None, None
    refPt = []
    rad = 0
    scan_flag = False
    frame_counter = 10
    colorIndex = 0
    main()