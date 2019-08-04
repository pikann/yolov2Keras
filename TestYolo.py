import copy
import cv2
import tensorflow as tf
import math
import numpy as np
def custom_loss(y_true, y_pred):
  return y_pred - y_true
img=cv2.imread("long_term_accomodation-745x398.jpg")
w_img=img.shape[1]
h_img=img.shape[0]
model=tf.keras.models.load_model("model (4).h5",custom_objects={'custom_loss': custom_loss})
resized=cv2.resize(img,(416,416))
resized=tf.reshape(resized,[-1,416,416,3])
predict=model.predict(resized,steps=1)
def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def getXIntersect(x11,x12,x21,x22):
  if x11>x21:
    if x11>x22: return 0
    else: return min(x12,x22)-x11
  else:
    if x21>x12: return 0
    else: return min(x12,x22)-x21

def iou(box_1, box_2):
  wIntersect=getXIntersect(box_1[0]-box_1[2]/2,box_1[0]+box_1[2]/2,box_2[0]-box_2[2]/2,box_2[0]+box_2[2]/2)
  hIntersect=getXIntersect(box_1[1]-box_1[3]/2,box_1[1]+box_1[3]/2,box_2[1]-box_2[3]/2,box_2[1]+box_2[3]/2)
  intersect=wIntersect*hIntersect
  if (intersect/(box_1[2]*box_1[3]))>0.8: return 1
  if (intersect / (box_2[2] * box_2[3])) > 0.8: return 1
  union=box_1[2]*box_1[3]+box_2[2]*box_2[3]-intersect
  return intersect/union

faces=[]
for i, row in enumerate(predict[0]):
  for j, col in enumerate(row):
    for k, box in enumerate(col):
      if box[4]>0.01:
        face_box=[i,j,k]
        face_data=box
        ANCHORS = '1.08,1.19,  3.42,4.41,  6.63,11.38,  9.42,5.11,  16.62,10.52'
        ANCHORS = [float(ANCHORS.strip()) for ANCHORS in ANCHORS.split(',')]
        x_face=int((sigmoid(face_data[0])+face_box[1])*(w_img/13))
        y_face=int((sigmoid(face_data[1])+face_box[0])*(h_img/13))
        w_face=int(ANCHORS[2*k]*(math.e**face_data[2])*(w_img/13))
        h_face=int(ANCHORS[1+2*k]*(math.e**face_data[3])*(h_img/13))
        faces.append([x_face,y_face,w_face,h_face,box[4]])

thutu=list(reversed(np.argsort([face[4] for face in faces])))
loai=set([])
for i in range(len(thutu)):
  for j in range(i+1,len(thutu)):
    if iou(faces[thutu[i]],faces[thutu[j]])>0.35:
      loai.add(thutu[j])

img_copy=copy.copy(img)
for i in range(len(thutu)):
  if thutu[i] not in loai:
    face=faces[thutu[i]]
    x_begin=int(face[0]-face[2]/2)
    y_begin=int(face[1]-face[3]/2)
    x_end=int(face[0]+face[2]/2)
    y_end=int(face[1]+face[3]/2)
    cv2.rectangle(img_copy,(x_begin,y_begin),(x_end,y_end),(0,255,0),2)
cv2.imshow("...",img_copy)
cv2.waitKey()
cv2.destroyAllWindows()