import os
import re

import cv2
import easyocr
import matplotlib
import numpy as np
import torch
from cnocr import CnOcr

from myutils import common
from myutils.common import Plot


class Ocr:
    def ocr(self,image):
        pass
    def ocr_for_single_line(data_area):
        pass

class CnOcrImpl(Ocr):
    def __init__(self):
        self.ocrTarget = CnOcr()
    def ocr(self,image):
        result=self.ocrTarget.ocr(image)
        return list(filter(lambda x: x["score"]>0.34, result))
    def ocr_for_single_line(self,image):
        return self.ocrTarget.ocr_for_single_line(image)
class EasyOcrImpl(Ocr):
    def __init__(self):
        self.ocrTarget = easyocr.Reader(['ch_sim','en'])
    def ocr(self,image):
        result = self.ocrTarget.readtext(image)
        print("ocr",result)
        return [{"text":i[1]} for i in result]
    def ocr_for_single_line(self,image):
        result=self.ocr(image)
        return result[0] if len(result)>0 else None
class Card:
    def __init__(self, imagePath, whminratio=6, whmaxratio=12, isDebug=False):
        matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体
        matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
        if not os.path.exists(imagePath):
            raise Exception(imagePath + "Image does not exist")
        data = np.fromfile(imagePath, dtype=np.uint8)
        self.img = cv2.imdecode(data, 1)
        common.show(self.img, "原图", debug=isDebug)
        self.grayImg = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        common.show(self.grayImg, "灰度图", cmap="gray", debug=isDebug)
        self.isDebug = isDebug
        self.whminratio = whminratio
        self.whmaxratio = whmaxratio
        # self.ocr = CnOcr(det_model_name='naive_det')
        self.ocr = CnOcrImpl()

    """
        获取身份证的区域
    """
    def getCardLocation(self):
        h, w = self.img.shape[:2]
        return (0, 0, w, h)
    """
        获取身份证的身份证号码区域
        :param whminratio:宽高比的最小值
        :param whmaxratio：宽高比的最大值
    """
    def getCardNoLocation(self):
        # 二值化处理，将黑字变成白色，偏白色区域转换成黑色
        _, thresholdImg = cv2.threshold(self.grayImg, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        common.show(self.grayImg, "二值化图", cmap="gray", debug=self.isDebug)
        kernel = np.ones((3, 3), np.uint8)
        # 将白色的部分进行膨胀处理
        dilate = cv2.dilate(thresholdImg, kernel, iterations=9)
        common.show(dilate, "膨胀", cmap="gray", debug=self.isDebug)
        # 進行开运算，腐蚀掉噪点，然后进行膨胀
        # morph_open = cv2.morphologyEx(dilate, cv2.MORPH_OPEN, kernel)
        # common.show(morph_open, "开运算", cmap="gray", debug=self.isDebug)
        # 轮廓检测
        contours, hierarchy = cv2.findContours(dilate, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        imgc = self.img.copy()
        maxarea = [];
        # 遍历所有轮廓
        for cnt in contours:
            # 进行多边形拟合
            approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
            x, y, w, h = cv2.boundingRect(cnt)
            if ((float(w) / h > self.whminratio and float(w) / h < self.whmaxratio) or (
                    float(h) / w > self.whminratio and float(h) / w < self.whmaxratio)):
                maxarea.append((x, y, w, h))
            # 绘制矩形框
        data_areas = sorted(maxarea, key=lambda x: x[2] * x[3], reverse=True)
        if (len(data_areas) > 0):
            x, y, w, h = data_areas[0]
            cv2.rectangle(imgc, (x, y), (x + w, y + h), (0, 0, 255), 1)
            common.show(imgc, "边框图", cmap="gray", debug=self.isDebug)
            self.fontHeight = h;
            return data_areas[0]
        return None;

    """
        截取身份证的图像返回
    """

    def getCardImg(self):
        cobj = self.getCardNoLocation()
        if cobj:
            x, y, w, h = cobj
            height, width = self.img.shape[:2]
            dst = self.img.copy()
            if w < h:
                k = 1;
                if x > y:
                    k = 3
                    x, y, w, h = height - y - h, cobj[0], cobj[3], cobj[2]
                else:
                    x, y, w, h = cobj[1], width - cobj[0] - w, cobj[3], cobj[2]
                dst = np.rot90(self.img, k=k)
            # cv2.rectangle(dst, (x, y), (x + w, y + h), (0, 0, 255), 1)
            # common.show(dst, "边框图", cmap="gray", debug=self.isDebug)
            ystart = (y - int(8.8 * h)) if (y - int(8.8 * h)) > 0 else 0
            xstart = (x - int(0.6 * w)) if (x - int(0.6 * w)) > 0 else 0
            crop_img = dst[ystart:y + h, xstart:x + w]
            common.show(crop_img, "截取图", debug=self.isDebug)
            return crop_img
        return None

    """
        截取身份证图像中文字对应的区域
    """

    def get_data_areas(self, img=None):
        if img is None:
            img = self.img
        grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resize = cv2.resize(grayImg, (1084, 669), interpolation=cv2.INTER_AREA)
        gray = resize.copy()
        threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        common.show(threshold, "threshold", cmap="gray", debug=self.isDebug)
        blur = cv2.medianBlur(threshold, 5)
        kernel = np.ones((3, 3), np.uint8)
        # 進行開運算
        morph_open = cv2.morphologyEx(blur, cv2.MORPH_OPEN, kernel)
        # kernel = np.ones((6, 6), np.uint8)
        # dilate = cv2.dilate(morph_open, kernel, iterations=7)
        kernel = np.ones((6, 6), np.uint8)
        #内核 (3, 5) 表示宽度为 3，高度为 5 的矩形
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 6))
        # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6, 6))

        dilate = cv2.dilate(morph_open, kernel, iterations=7)
        common.show(dilate, "获取开运算膨胀土", cmap="gray", debug=self.isDebug)
        contours, hierarchy = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        resize_copy = resize.copy()
        res = cv2.drawContours(resize_copy, contours, -1, (255, 0, 0), 2)
        common.show(res, "绘制边框", cmap="gray", debug=self.isDebug)
        data_areas = []
        resize_copy = resize.copy()
        for contour in contours:
            epsilon = 0.002 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            x, y, w, h = cv2.boundingRect(approx)
            if h > 50 and x > 100 and x < 620 and y > 0:
                res = cv2.rectangle(resize_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
                data_area = gray[y:(y + h), x:(x + w)]
                plot = Plot()
                plot.subplot(221, data_area, "原始图", cmap="gray", debug=self.isDebug)
                # common.show(data_area,"原始图",cmap="gray",debug=self.isDebug)
                data_area = cv2.resize(data_area, (int(w + w), int(h + h)), interpolation=cv2.INTER_AREA)
                plot.subplot(222, data_area, "扩大", cmap="gray", debug=self.isDebug)
                # common.show(data_area,"扩大",cmap="gray",debug=self.isDebug)
                data_area = cv2.medianBlur(data_area, 7)
                # _,data_area=cv2.threshold(data_area,0,255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                # common.show(data_area,"二值化",cmap="gray",debug=self.isDebug)
                # # 自适应直方图均衡化
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(2, 2))
                data_area = clahe.apply(data_area)
                plot.subplot(223, data_area, "直方图处理", cmap="gray", debug=self.isDebug)
                # common.show(data_area,"直方图处理",cmap="gray",debug=self.isDebug)
                # show(data_area,"原图",cmap="gray",debug=True)
                data_areas.append((x, y, w, h, data_area))
                plot.show(self.isDebug)
        data_areas = sorted(data_areas, key=lambda x: x[1])
        return data_areas

    """
        对每个区域进行ocr获取到身份证的各部分数据
    """

    def ocrArea(self, areas):
        row = 0;
        result = {"姓名": "", "性别": "", "民族": "", "住址": "","success":True}
        rowObj = {}
        # 先按照行归类
        for index, rect in enumerate(areas):
            x, y, w, h, data_area = rect
            if not row in rowObj:
                rowObj[row] = []
            rowObj[row].append((x, y, w, h, data_area))
            # print("第",row,"行，元素",(x, y, w, h,data_area))
            if index + 1 < len(areas):
                # 下一个元素的y值比当前元素大hre，是下一行,最好是根据身份证区域的高度来决定
                hre = 30
                if areas[index + 1][1] - y > hre:
                    row = 1 + row
        if self.isDebug:
            for key in rowObj:
                plot = Plot()
                for i, v in enumerate(rowObj[key]):
                    plot.subplot(int(str(len(rowObj[key]))+str(21 + i)), v[4], "level_" + str(key), cmap="gray", debug=self.isDebug)
                plot.show(self.isDebug)
        if len(rowObj.keys())<5:
            result["success"]=False
            return result
        # 解析第一行，姓名
        rowObj0Array = sorted(rowObj[0], key=lambda x: x[0])
        for row0Index,rowObj0 in enumerate(rowObj0Array):
            x, y, w, h, data_area = rowObj0
            # cv2.imwrite("xm.jpg", data_area)
            # out = self.ocr.ocr(data_area)
            # out =self.ocr.ocr_for_single_line(data_area) #姓名建议用单行的识别，地址用默认的多行识别
            # if len(rowObj0Array)>1 and row0Index==0 and (out["text"].find("姓")>=0 or out["text"].find("名")>=0): #过滤掉姓名，错误的：欢名
            #     continue
            # if out["text"] in ['姓名', '名']: #注意单行返回的字典不是数组
            #     continue
            # result["姓名"] += out["text"]
            out = self.ocr.ocr(data_area)
            # print(out)
            for ar in out:
                if len(rowObj0Array)>1 and row0Index==0 and (ar["text"].find("姓")>=0 or ar["text"].find("名")>=0): #过滤掉姓名，错误的：欢名
                    continue
                if ar["text"] in ['姓名', '名'] or len(ar["text"])>6:
                    continue
                result["姓名"] += ar["text"]
        rowObj1Array = sorted(rowObj[1], key=lambda x: x[0])
        for rowObj1 in rowObj1Array:
            x, y, w, h, data_area = rowObj1
            # out = self.ocr.ocr(data_area)
            out = self.ocr.ocr_for_single_line(data_area)  # 姓名建议用单行的识别，地址用默认的多行识别
            # print("第二行",out)
            text = out["text"]
            if text in ["男", "女"]:
                result["性别"] = out["text"]
            if text in ['安','如', '妙', '妍', '妤', '妞', '妹', '妻', '姊', '姐', '姒', '姓', '委', '娃', '娅', '娇', '娜', '娟', '姣', '姘', '娥']:
                result["性别"] ='女'
            if text.startswith("族") or text.startswith("民族"):
                result["民族"] = text.split("族")[1]
            nations = ['汉', '壮', '满', '回', '苗', '维吾尔', '土家', '彝', '蒙古', '藏', '布依', '侗', '瑶', '朝鲜', '白', '哈尼', '黎',
                       '哈萨克', '傣', '畲', '傈僳', '东乡', '仡佬', '拉祜', '佤', '水', '纳西', '羌', '土', '仫佬', '锡伯', '柯尔克孜', '景颇',
                       '达斡尔', '撒拉', '布朗', '毛南', '塔吉克', '普米', '阿昌', '怒', '鄂伦春', '赫哲', '门巴', '珞巴', '基诺', '德昂', '保安', '裕固',
                       '京族', '塔塔尔', '独龙', '鄂温克']
            if text in nations:
                result["民族"] = text
        rowObj2Array = sorted(rowObj[2], key=lambda x: x[0])
        for index2, rowObj2 in enumerate(rowObj2Array):
            x, y, w, h, data_area = rowObj2
            # out = self.ocr.ocr(data_area)
            out = self.ocr.ocr_for_single_line(data_area)  # 姓名建议用单行的识别，地址用默认的多行识别
            text = out["text"]
            if text in ['生', '出生']:
                continue
            if text.isdigit():
                if len(text) == 4:
                    result["出生年"] = text
                if len(text) >= 1 and len(text) <= 2:
                    if "出生年" in result and result["出生年"] is not None and result["出生年"] != "":
                        if not "出生月" in result or result["出生月"] is None or result["出生月"] == "":
                            result["出生月"] = text
                        else:
                            result["出生日"] = text

            if index2 == 0 and text.isdigit():  # 年份一定在月日的前面
                result["出生年"] = text
            if text.find("年") > 0:
                result["出生年"] = text.split("年")[0]
            if text.find("月") > 0:
                result["出生月"] = text.split("月")[0]
                if result["出生月"].find("年") >= 0:
                    result["出生月"] = result["出生月"].split("年")[1]
            if text.find("日") > 0:
                result["出生日"] = text.rstrip("日")
                if result["出生日"].find("月") >= 0:
                    result["出生日"] = result["出生日"].split("月")[1]
        rowObj3Array = sorted(rowObj[3], key=lambda x: x[0])
        for rowObj3 in rowObj3Array:
            x, y, w, h, data_area1 = rowObj3
            # common.show(data_area1, "地址", cmap="gray", debug=True)
            # _,data_area1=cv2.threshold(data_area1,60,255,cv2.THRESH_BINARY)
            out1 = self.ocr.ocr(data_area1)
            # cv2.imwrite("zz.jpg", data_area1)
            for ar in out1:
                if ar["text"] in ['住','住址', '址']:
                    continue
                result["住址"] += ar["text"]
            if result["住址"].startswith("住址"):
                result["住址"]=result["住址"].lstrip("住址")
        rowObj4Array = sorted(rowObj[4], key=lambda x: x[0])
        for rowObj4 in rowObj4Array:
            x, y, w, h, data_area1 = rowObj4
            # cv2.imwrite("zz.jpg", data_area1)
            # data_area1=cv2.medianBlur(data_area1,7)
            common.show(data_area1, "身份证图", cmap="gray", debug=self.isDebug)
            # _,data_area1=cv2.threshold(data_area1,0,255,cv2.THRESH_OTSU)
            # _,data_area1=cv2.threshold(data_area1,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            common.show(data_area1, "身份证图threshold", cmap="gray", debug=self.isDebug)
            out = self.ocr.ocr(data_area1)
            # out = self.ocr.ocr_for_single_line(data_area1)  # 姓名建议用单行的识别，地址用默认的多行识别
            if(len(out)==0):
                continue
            out=out[0]
            if out["text"] in ["公民身份证号码"]:
                continue
            result["身份证号码"] = out["text"]
            if result["身份证号码"] is None or  result["身份证号码"] =="":
                result["success"] = False
                return result
            lastChar = result["身份证号码"][len(result["身份证号码"]) - 1]
            result["身份证号码"] = re.sub(r'\D', '', result["身份证号码"])
            result["身份证号码"] = re.sub(r'\D', '0', result["身份证号码"])
            result["身份证号码"] = result["身份证号码"].replace("日", "0")
            if not lastChar.isdigit():
                result["身份证号码"] = result["身份证号码"][0:len(result["身份证号码"])] + "X"
        return result


"""
    使用yolov5定位到身份证的位置
"""


class Yolov5Card(Card):
    yolov5Path = None
    ptPath = None
    model = None

    @staticmethod
    def confYolov5(yolov5Path, ptPath):
        Yolov5Card.yolov5Path = yolov5Path
        Yolov5Card.ptPath = ptPath
        Yolov5Card.model = torch.hub.load(yolov5Path, 'custom', path=ptPath,
                                          source='local')  # local repo

    def __init__(self, imagePath, whminratio=6, whmaxratio=12, isDebug=False):
        super().__init__(imagePath, whminratio, whmaxratio, isDebug)

    """
        获取身份证的身份证号码区域
        :param whminratio:宽高比的最小值
        :param whmaxratio：宽高比的最大值
    """

    def getCardNoLocation(self):
        results = Yolov5Card.model(self.img)
        for result in results.xyxy[0]:
            x1, y1, x2, y2, conf, cls = result.tolist()
            if conf > 0.3:
                return int(x1), int(y1), int(x2 - x1), int(y2 - y1),Yolov5Card.model.names[int(cls)]
        return None;

    def getCardImg(self):
        cobj = self.getCardNoLocation()
        if cobj:
            x, y, w, h,labels = cobj
            print(cobj)
            height, width = self.img.shape[:2]
            dst = self.img.copy()
            if w < h:
                k = 1;
                if x > y:
                    k = 3
                    x, y, w, h = height - y - h, cobj[0], cobj[3], cobj[2]
                else:
                    x, y, w, h = cobj[1], width - cobj[0] - w, cobj[3], cobj[2]
                print("k",k)
                dst = np.rot90(self.img, k=k)
            crop_img = dst[y:y + h, x:x + w]
            common.show(crop_img, "截取图", debug=self.isDebug)
            return crop_img,labels
        return None,None
