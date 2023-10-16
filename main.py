import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from tracker import*
import cvzone
from datetime import datetime
import os
from collections import defaultdict

now = datetime.now()
model=YOLO('yolov8l.pt')
#video_path = 'data/los_angeles.mp4'
video_path = '192.168.172.10_20230922_173123.mp4'

cap=cv2.VideoCapture(video_path)


COCO_data = open("yolov8counter/coco.txt", "r")
data = COCO_data.read()
class_list = data.split("\n") 
#print(class_list)

count=0
tracker=Tracker() 
area = [(3,300),(700,300),(700,550),(3,550)]
area_c = set()

'''
# Store the track history
track_history = defaultdict(lambda: [])
'''

'''function for save detected objects as png file'''
def imgwrite(img):
    now = datetime.now()
    current_time = now.strftime("%d_%m_%Y_%H_%M_%S")
    filename = '%s.png' % current_time
    cv2.imwrite(os.path.join(r"C:\Users\infoworks\Downloads\source_code\save",filename), img)

while True:    
    ret,frame = cap.read()
    if not ret:
        break
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    

    count += 1
    if count % 3 != 0:
        continue

    frame=cv2.resize(frame,(1020,500))

    '''N 프레임당 하나씩 이미지 추출하려면'''
    # 3 프레임당 한번씩 이미지 추출, 예측
    if(int(cap.get(1)) % 3 == 0):
        results=model.predict(frame, stream=True, iou=0.9) #, save_crop = True) crop 저장하려면 작업이 늘어서 처리 속도 느려짐
    results = list(results)
    # print(results)
    a=results[0].boxes.data
    px=pd.DataFrame(a).astype("float")
    print(px)
    lst=[]
    
    '''
    # Get the boxes and track IDs
    boxes = results[0].boxes.xywh.cpu()
    track_ids = results[0].boxes.id.int().cpu().tolist()

    # Visualize the results on the frame
    annotated_frame = results[0].plot()

    # Plot the tracks
    for box, track_id in zip(boxes, track_ids):
        x, y, w, h = box
        track = track_history[track_id]
        track.append((float(x), float(y)))  # x, y center point
        if len(track) > 30:  # retain 90 tracks for 90 frames
            track.pop(0)

        # Draw the tracking lines
        points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)
    ''' 

    for index,row in px.iterrows():
        print(row)
 
        x1=int(row[0])
        y1=int(row[1])
        x2=int(row[2])
        y2=int(row[3])
        d=int(row[5])
        
        c=class_list[d]
        if 'car' in c or 'truck' in c or 'motorcycle' in c or 'bus' in c:
            lst.append([x1,y1,x2,y2])
       
        
    bbox_id=tracker.update(lst)
    for bbox in bbox_id:
        x3,y3,x4,y4,id=bbox
        position=cv2.pointPolygonTest(np.array(area,np.int32),((x4,y4)),False) # 점이 외부에 있으면 -, 내부에 있으면 + 값 return 
        cx=int(x3+x4)//2
        cy=int(y3+y4)//2
        # cv2.circle(frame,(cx,cy),4,(255,0,255),-1)
        cv2.rectangle(frame,(x3,y3),(x4,y4),(0,0,255),2)
        '''label 표시에 차량 객체가 중첩되면 crop된 이미지가 불완전하다 -> id, class 시각화하지 않고 excel에 저장'''
        # cvzone.putTextRect(frame, f'{id}',(x3,y3),1,2)
        # cvzone.putTextRect(frame, f'{c}',(x4,y4),1,2)
        
        if position>=0:
            crop=frame[y3:y4,x3:x4]
            imgwrite(crop)
#            cv2.imshow(str(id),crop) 
            area_c.add(id)


    '''상행, 하행 방향 구분 시 선을 이용'''  
    # cv2.line(frame,(3,194),(1018,194),(0,255,0),2)
    # cv2.line(frame,(5,220),(1019,220),(0,255,255),2)
    
    # ROI 설정
    cv2.polylines(frame,[np.array(area,np.int32)],True,(255,0,0),2) # roi 다각형 설정
    print(area_c) # 영역 내의 객체 수
    k=len(area_c)
    cv2.putText(frame,str(k),(50,60),cv2.FONT_HERSHEY_PLAIN,5,(255,0,0),3)
    cv2.imshow("id_tracking", frame)
    # 좌표 일일이 모를 때는 마우스 API 사용해서 영상 보고 설정
    
    if cv2.waitKey(1)&0xFF==27:
        break
cap.release()
cv2.destroyAllWindows()

