import cv2
import numpy as np
from matplotlib import pyplot as plt

def cannify(im):
    grayscale = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(grayscale, (5,5), 0)
    cannied = cv2.Canny(blur, 50, 150)
    return cannied

def mask(im):
    pts = np.array([[(70, 480), (800, 480), (375, 290)]])

    mask = np.zeros_like(im)
    cv2.fillPoly(mask, pts, 255)

    masked = cv2.bitwise_and(im, mask)

    return masked

def plotlines(im, lines):
    lineslist = lines[:, 0]
    for line in lineslist:
        cv2.line(im, (line[0], line[1]), (line[2], line[3]), (0,0,255), thickness=2)
    return im

def main():
    vid = cv2.VideoCapture("input.mp4")
    while vid.isOpened():
        _, frame = vid.read()

        canim = cannify(frame)
        maskim = mask(canim)
        houghlines = cv2.HoughLinesP(maskim, 2, np.pi/180, 100, np.array([]), minLineLength=100, maxLineGap=50)
        finout = plotlines(frame, houghlines)
        cv2.imshow("result", finout)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    vid.release()
    cv2.destroyAllWindows()

main()