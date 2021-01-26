from pdf2image import convert_from_path
from pdf2image.exceptions import (
    PDFInfoNotInstalledError,
    PDFPageCountError,
    PDFSyntaxError
)
import cv2
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    images = convert_from_path('./data.pdf')

    for i, img in enumerate(images):
        img = np.array(img, dtype=np.uint8)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray = cv2.bitwise_not(gray)

        bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)
        cols = bw.shape[1]
        horizontal_size = int(cols / 30)
        horizontal_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
        horizontal = cv2.morphologyEx(bw, cv2.MORPH_OPEN, horizontal_structure)

        rows = bw.shape[0]
        vertical_size = int(rows / 30)
        vertical_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_size))
        vertical = cv2.morphologyEx(bw, cv2.MORPH_OPEN, vertical_structure)

        dst = cv2.addWeighted(horizontal, 0.5, vertical, 0.5, 0)
        dst = np.where(dst > 0, 255, dst)
        contours, _ = cv2.findContours(dst, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        rects = []

        for j, cnt in enumerate(contours):
            if cv2.contourArea(cnt) > 10000.0:
                cv2.drawContours(img, contours, j, (0, 255, 0), 3)
                x, y, w, h = cv2.boundingRect(cnt)
                rects.append(['', img[y:y + h, x:x + w]])

        plt.figure(figsize=(12, 10))
        plt.imshow(img)
        plt.show()

        plt.figure(figsize=(12, 10))
        plt.imshow(vertical, cmap='gray')
        plt.show()

        plt.figure(figsize=(12, 10))
        plt.imshow(horizontal, cmap='gray')
        plt.show()

        plt.figure(figsize=(12, 10))
        plt.imshow(dst, cmap='gray')
        plt.show()
