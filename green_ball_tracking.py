import cv2 as cv
import math

cap = cv.VideoCapture(0)  # 640X480

font = cv.FONT_HERSHEY_SIMPLEX

# green
greenLower = (29, 86, 6)
greenUpper = (64, 255, 255)

x = 0
y = 0
angle_x = 0
angle_y = 0

while True:

    ret, frame = cap.read()
    cv.circle(frame, (320, 240), 5, (0, 0, 255), 2)
    # cv.line(frame, (320, 240), (320, 0), (0, 0, 255), 1)
    # cv.line(frame, (0, 240), (640, 240), (0, 0, 255), 1)
    # cv.line(frame, (320, 240), (640, 240), (0, 0, 255), 1)

    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv, greenLower, greenUpper)
    mask = cv.erode(mask, None, iterations=2)
    mask = cv.dilate(mask, None, iterations=2)

    cnts = cv.findContours(mask.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[-2]
    center = None

    if len(cnts) > 0:
        c = max(cnts, key=cv.contourArea)
        ((x, y), radius) = cv.minEnclosingCircle(c)
        M = cv.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        if radius > 10:
            cv.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
            cv.circle(frame, center, 5, (0, 0, 255), -1)

    if center is not None:
        cv.line(frame, (320, 240), center, (0, 0, 255), 1)
        angle_x = abs(int(math.atan2(240 - y, x - 320) * 180 / math.pi))
        angle_y = abs(int(math.atan2(x - 320, 240 - y) * 180 / math.pi))
        print(x, y)
        cv.putText(frame, "x " + str(angle_x), (350, 230), font, 0.8, (255, 255, 255), 2, cv.LINE_AA)
        cv.putText(frame, "y " + str(angle_y), (350, 200), font, 0.8, (255, 255, 255), 2, cv.LINE_AA)

    cv.putText(frame, "Press \'Q' to exit", (10, 20), font, 0.4, (255, 255, 255), 1, cv.LINE_AA)
    cv.imshow('frame', frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()