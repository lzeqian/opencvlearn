import cv2
import numpy as np
import tkinter as tk
import tkinter
from tkinter import filedialog

class ColorRangeSelector:
    def __init__(self, image_path):
        self.image_path = image_path
        self.image = cv2.imread(image_path)
        self.color_range = None

    def get_color_range(self):
        cv2.namedWindow('image')
        cv2.setMouseCallback('image', self.mouse_callback)
        while True:
            cv2.imshow('image', self.image)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        cv2.destroyAllWindows()
        return self.color_range

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.crop_start = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.crop_end = (x, y)
            x1, y1 = self.crop_start
            x2, y2 = self.crop_end
            cropped_image = self.image[min(y1, y2):max(y1, y2), min(x1, x2):max(x1, x2)]
            self.color_range = self.get_color_range_from_image(cropped_image)

    def get_color_range_from_image(self, image):
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv_image)
        color_range = np.array([h.min(), s.min(), v.min()]), np.array([h.max(), s.max(), v.max()])
        return color_range


if __name__ == '__main__':
    root = tk.Tk()
    root.withdraw()
    image_path = tk.filedialog.askopenfilename()
    selector = ColorRangeSelector(image_path)
    color_range = selector.get_color_range()
    print('Selected color range:', color_range)