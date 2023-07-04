import json
import math
import os

import cv2
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
import numpy as np


class Plot:
    def __init__(self):
        self.plot = plt

    def subplot(self, loc, image, title, cmap=None, debug=False):
        if debug:
            self.plot.subplot(loc)
            self.plot.title(title)
            self.plot.imshow(image, cmap=cmap)

    def show(self, debug=False):
        if debug:
            self.plot.show()


def get_all_files(dir_path):
    all_files = []
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            file_path = os.path.join(root, file)
            if file_path.endswith('.jpg') or file_path.endswith('.jpeg') or file_path.endswith('.png'):
                all_files.append(file_path)
    return all_files


def get_all_files_next(dir_path):
    all_files = []
    for file in os.listdir(dir_path):
        file_path = os.path.join(dir_path, file)
        if file_path.endswith('.jpg') or file_path.endswith('.jpeg') or file_path.endswith('.png'):
            all_files.append(file_path)
    return all_files


def fillColor(img, color=[255, 255, 255], radio=0.05):
    # 计算填充的大小
    top = int(radio * img.shape[0])
    bottom = int(radio * img.shape[0])
    left = int(radio * img.shape[1])
    right = int(radio * img.shape[1])
    # 调用cv2.copyMakeBorder()函数进行填充
    img_with_border = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img_with_border


def show(dilate, title, cmap=None, debug=False):
    if debug:
        plt.title(title)
        plt.imshow(dilate, cmap=cmap)
        plt.show()


def clearBorder(img):
    # 将图像转为灰度图像
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lower = np.array([245])
    upper = np.array([255])
    mask = cv2.inRange(gray, lower, upper)
    img[mask == 255] = [0, 0, 0]
    return img


"""
   将图像按原始方向进行校正
   :param pts 传入一个矩形pt，[[左上角],[右上角],[右下角],[左上角]]
"""


def justImage(img, pts):
    # 计算四个点的中心点坐标
    center = np.mean(pts, axis=0)
    # 获取整张图片中心位置
    iheight, iwidth = img.shape[:2]  # 获取图片的高度和宽度
    center_x, center_y = int(iwidth / 2), int(iheight / 2)
    print(pts, pts)
    dy = pts[1][1] - pts[0][1]
    dx = pts[1][0] - pts[0][0]
    # 例如，如果最小矩形的长轴与水平方向的夹角为 30 度，则 cv2.minAreaRect() 函数返回的角度值为 -60 度，
    # 因为它是相对于水平方向的逆时针旋转的角度，即 -90 度到 0 度之间的负数。
    rect = cv2.minAreaRect(np.array(pts))
    # 提取最小外接矩形的长、宽和角度
    width = rect[1][0]
    height = rect[1][1]
    angle = rect[-1]
    # 如果长边不是水平方向，将角度修正为0~180度范围内
    print(width, height)
    if width < height:
        angle -= 90
    # 转换为弧度
    angle = math.radians(angle)
    # 计算正方形和x轴之间的夹角,，注意逆时针旋转需要+270度
    angle_deg = math.degrees(angle)
    # 创建旋转矩阵
    M = cv2.getRotationMatrix2D((center_x, center_y), angle_deg, scale=1)
    pointsNew = justPoint(pts, M);
    if pointsNew[0][1] < (iheight / 2):
        angle_deg -= 180
    M = cv2.getRotationMatrix2D((center_x, center_y), angle_deg, scale=1)
    rotated_img = cv2.warpAffine(img, M, (iwidth, iheight))
    return rotated_img, M


"""
   获取点在M旋转后生成的新点
   :param pts 传入一个矩形pt，[[左上角],[右上角],[右下角],[左上角]]
"""


def justPoint(pts, M):
    points = [[int(x) for x in point] for point in pts]
    pointsNew = [];
    for p in points:
        p_rotated_array = np.dot(M, np.array([p[0], p[1], 1]))
        p_rotated = (int(p_rotated_array[0]), int(p_rotated_array[1]))
        pointsNew.append(p_rotated)
    return pointsNew
def getNonmal(pts):
    [[x1, y1],[x2, y2],[x3, y3],[x4, y4]]=pts
    # 计算左上角和右下角的点
    tl_index = np.argmin([x1 + y1, x2 + y2, x3 + y3, x4 + y4])
    br_index = np.argmax([x1 + y1, x2 + y2, x3 + y3, x4 + y4])
    # 输出左上角和右下角的点的坐标
    left_pts=pts[tl_index]
    right_pts=pts[br_index]
    return (left_pts[0],left_pts[1],right_pts[0]-left_pts[0],right_pts[1]-left_pts[1])
def printPretty(jsonObj):
    # 使用json.dumps()函数将JSON对象转换为字符串，并设置indent参数为4实现缩进，确保输出的JSON易于阅读
    json_str = json.dumps(jsonObj, ensure_ascii=False, indent=4)
    # 使用print()函数打印输出JSON字符串，并在末尾加上换行符'\n'
    print(json_str + '\n')

# path='../images/idcard.jpg'
# show(clearBorder(path), "contours矩形",cmap="gray", debug=True)
