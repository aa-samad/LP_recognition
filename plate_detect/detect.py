import cv2 as cv
import sys


model_addr = "model/frozen_inference_graph.pb"
model_addr_txt = "model/graph.pbtxt"
image_addr = sys.argv[1]

LABELS = ["null","plate"]     

cvNet = cv.dnn.readNetFromTensorflow(model_addr, model_addr_txt)
# --- pre process
img = cv.imread(image_addr)
img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
clahe = cv.createCLAHE(clipLimit=4.0,tileGridSize=(4,4))
img[:, :, 2] = clahe.apply(img[:, :, 2])
img = cv.cvtColor(img, cv.COLOR_HSV2BGR)
# img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
# img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
# ---- feeding the net
rows = img.shape[0]
cols = img.shape[1]
cvNet.setInput(cv.dnn.blobFromImage(img, size=(300, 300), swapRB=True, crop=False))
cvOut = cvNet.forward()

# max_score = 0.5
score_thr = 0.2
max_area = 0
print(cvOut.shape)
for detection in cvOut[0, 0, :, :]:
    score = float(detection[2])
    if score > score_thr:
        
        label = "{}: {:.2f}%".format(LABELS[int(detection[1])], detection[2] * 100)
        # print("[INFO] {}".format(label))
        left = int(detection[3] * cols)
        top = int(detection[4] * rows)
        right = int(detection[5] * cols)
        bottom = int(detection[6] * rows)
        area = (bottom - top) * (right - left)
        # --- detect of bounding box has some ratio and bigger area than others
        if 0.1 < (bottom - top) / (right - left) < 0.4:
            max_area = area
            # --- if box was outside of image
            left = 0 if left < 0 else left
            top = 0 if top < 0 else top
            right = img.shape[1] if right > img.shape[1] else right
            bottom = img.shape[0] if bottom > img.shape[0] else bottom
            # --- create a margin
            left = left - 10 if left - 10 > 0 else left
            top = top - 10 if top - 10 > 0 else top
            right = right + 10 if right + 10 < img.shape[1] else right
            bottom = bottom + 10 if bottom + 10 < img.shape[0] else bottom
            
            # for i in range(3):            
            cv.imwrite(sys.argv[2], img[top:bottom, left:right])
            # print("left:{} top:{} right:{} bottom:{}".format(left,top,right,bottom))

            cv.rectangle(img, (int(left), int(top)), (int(right), int(bottom)), (23, 230, 210), thickness=2)
#             left = left - 15 if left - 15 > 15 else left + 15
#             top = top - 5
#             cv.putText(img, label, (int(left), int(top)),
#                 cv.FONT_HERSHEY_SIMPLEX, 0.5,(0,255,0), 2)

cv.imwrite(sys.argv[2], img)
        