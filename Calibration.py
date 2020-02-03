import cv2
import numpy as np

def nothing(x):
    pass

vid = cv2.VideoCapture(0)

cv2.namedWindow("Trackbar")
cv2.createTrackbar("L-H", "Trackbar", 0, 180, nothing)
cv2.createTrackbar("L-S", "Trackbar", 0, 255, nothing)
cv2.createTrackbar("L-V", "Trackbar", 0, 255, nothing)
cv2.createTrackbar("U-H", "Trackbar", 180, 180, nothing)
cv2.createTrackbar("U-S", "Trackbar", 255, 255, nothing)
cv2.createTrackbar("U-V", "Trackbar", 255, 255, nothing)

while True:
    _, frame = vid.read()
    hsv= cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    l_hr = cv2.getTrackbarPos("L-H", "Trackbar")
    l_sr = cv2.getTrackbarPos("L-S", "Trackbar")
    l_vr = cv2.getTrackbarPos("L-V", "Trackbar")
    u_hr = cv2.getTrackbarPos("U-H", "Trackbar")
    u_sr = cv2.getTrackbarPos("U-S", "Trackbar")
    u_vr = cv2.getTrackbarPos("U-V", "Trackbar")

    lower_red = np.array([l_hr, l_sr, l_vr])
    upper_red = np.array([u_hr, u_sr, u_vr])

    mask = cv2.inRange(hsv, lower_red, upper_red)

    cv2.imshow("Normal", frame)
    cv2.imshow("Mask", mask)

    k = cv2.waitKey(1)
    if k == 27:
        break
vid.release()
cv2.destroyAllWindows()