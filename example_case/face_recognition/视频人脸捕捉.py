import cv2

def face_detect(img):
    # 加载人脸模型
    xml_dir = "D:/condaenv/tensorflowcpu/Library/etc/haarcascades/"
    # 加载人脸检测器

    face_cascade = cv2.CascadeClassifier(xml_dir + 'haarcascade_frontalface_alt2.xml')
    # 转换为灰度图像
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 检测人脸
    faces = face_cascade.detectMultiScale(gray)
    # 在图像中框出人脸
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imshow("demo",img)
cap=cv2.VideoCapture("./video/demo3.mp4")
while True:
    flag,frame=cap.read()
    if not flag:
        break;
    face_detect(frame)
    # 等待按下 q 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
cap.release()