# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 19:34:40 2020
Mask_detection_video
@author: SIBASIS SAHOO
"""


from torchvision import transforms
from imutils.video import VideoStream
import imutils
import pickle
import numpy as np
import torch
import cv2

test_transform=transforms.Compose([
                              transforms.ToPILImage(),
                              transforms.Resize(224),
                              transforms.CenterCrop(224),
                              transforms.ToTensor(),
                              transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
])
mask_dict={0:'with_mask',1:'without_mask'}
# load our serialized face detector model from disk
prototxtPath = r"face_detector/deploy.prototxt"
weightsPath = r"face_detector/res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
maskNet =  pickle.load(open('face_mask_detection_model.pkl', 'rb'))

# initialize the video stream
print("[INFO] starting video stream...")
def detect_and_predict_mask(frame,faceNet,maskNet):
  h,w=frame.shape[:2] 
  blob=cv2.dnn.blobFromImage(frame,1.0,(300,300),(104.0,177.0,123.0))
  faceNet.setInput(blob)
  detection=faceNet.forward()
  #initialize list of faces and their locations
  faces=[]
  locs=[]
  preds=[]
  for i in range(0,detection.shape[2]):
    confidence=detection[0,0,i,2]
    if confidence>0.5:
      #we need the X,Y coordinats
      box=detection[0,0,i,3:7]*np.array([w,h,w,h])
      (startX,startY,endX,endY)=box.astype('int')
      #ensure the bounding boxes fall within the dimentions of the frame
      (startX,startY)=(max(0,startX), max(0,startY))
      (endX,endY)=(min(w-1,endX), min(h-1,endY))
      #extract face ROI,conver it from BGR to RGB,resize it to 224,224
      face=frame[startY:endY, startX:endX]
      face=cv2.cvtColor(face,cv2.COLOR_BGR2RGB)
      maskNet.eval()
      faces.append(face)
      locs.append((startX,startY,endX,endY))
    if len(faces)>0:
      facess=np.squeeze(faces)
      facess=test_transform(facess)
      #facess=facess.cuda()
      facess = facess.unsqueeze(0)
      facess=torch.Tensor(facess)
      with torch.no_grad():
        preds=maskNet(facess)
    return (locs,preds)
vs=VideoStream(src=0).start()
while True:
  frame=vs.read()
  frame=imutils.resize(frame,width=600)
  (locs,preds)=detect_and_predict_mask(frame,faceNet,maskNet)
  for box,preds in zip(locs,preds):
    (startX,startY,endX,endY)=box
    (with_mask,without_mask)=preds
    label='With_mask' if with_mask>without_mask else 'No_mask'
    color=(0,255,0) if label=='With_mask' else (0,0,255)
    #include the probability in the label
    label="{}: {:.2f}%".format(label,max(with_mask,without_mask)*100)
    #display the label and bounding boxes
    cv2.putText(frame,label,(startX,startY-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,color,2)
    cv2.rectangle(frame,(startX,startY),(endX,endY),color,3)
  cv2.imshow("output", frame)
  key=cv2.waitKey(1)& 0xFF
  # if the `q` key was pressed, break from the loop
  if key == ord("q"):
     break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
