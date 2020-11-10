                                    # SMRITI AMATYA
# TASK 1: OCR(OBJECT DETECTION)

import cv2

# Threshold to detect object
thres = 0.45

cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)
cap.set(10,70)

classNames= []
classFile = 'coco.names'
with open(classFile,'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

configPath = 'ssd.pbtxt'
weightsPath = 'frozen_inference_graph.pb'
net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

while True:
  success, img = cap.read()
  classIds, confs, bbox = net.detect(img, confThreshold=thres)
  print(classIds,bbox)
  if len(classIds) != 0:
    for classId, confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
      cv2.rectangle(img,box,color=(0,255,0),thickness=1)
      cv2.putText(img,classNames[classId-1].upper(),(box[0]+10,box[1]+20),
      cv2.FONT_ITALIC,1,(0,255,0),2)
      cv2.putText(img, str(round(confidence * 300, 2)), (box[0] + 200, box[1] + 30), cv2.FONT_ITALIC, 1, (0, 255, 0), 2)
      cv2.imshow("Output",img)
      cv2.waitKey(0)
cv2.destroyAllwindows()


