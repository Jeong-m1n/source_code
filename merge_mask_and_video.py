import cv2 as cv
import numpy as np

def create_mask(img_path):
    pts = []  
    mask_list = []  

    def draw_mask_eventListener(event, x, y, flags, param):
        global pts
        img2 = img.copy()

        if event == cv.EVENT_LBUTTONDOWN:  
            pts.append((x, y))

        if event == cv.EVENT_RBUTTONDOWN: 
            pts.pop()

        if event == cv.EVENT_LBUTTONDBLCLK: 
            # 초기화
            mask_list.append(pts)
            pts = []

        if event == cv.EVENT_MBUTTONDOWN:  
            result_roi = np.zeros(img.shape, np.uint8)  #<- 이후 YOLO 모델 적용시 cv.addweighted() 인자 전달

            for point in mask_list:
                if not point: continue
                mask = np.zeros(img.shape, np.uint8)
                points = np.array(point, np.int32)
                points = points.reshape((-1, 1, 2)) 
                mask = cv.polylines(mask, [points], True, (255, 255, 255), 2) 
                mask2 = cv.fillPoly(mask.copy(), [points], (255, 255, 0)) 
        #        ROI = cv.bitwise_and(mask2, img) # 색깔 마스크 대신 원본 영상을 그대로 받고 싶으면(대신 차량도 함께 표시됨) 

                result_roi = cv.add(result_roi, mask2)  # 마스크 이미지끼리 더하기 - 원본 이미지 사이즈와 맞게 변경하면서 mask 부분 외에는 검은색으로 추출된 이미지
        
            cv.imshow('result_roi', result_roi)
            cv.imwrite('result_roi.jpg', result_roi) # 저장
        
        try:
            if len(pts) > 0:  # 마우스 포인트 원으로 지정
                cv.circle(img2, pts[-1], 3, (0, 0, 255), -1)
        except:
            pts = []

        if len(pts) > 1:  # 마우스 포인트 연결 라인 생성
            for i in range(len(pts) - 1):
                cv.circle(img2, pts[i], 5, (0, 0, 255), -1)
                cv.line(img=img2, pt1=pts[i], pt2=pts[i + 1], color=(255, 0, 0), thickness=2)

        if len(mask_list) > 0:  # 마스크 여러 개일때 포인트 연결 라인 생성
        
            area = []
            for m in mask_list:
                for i in range(len(m) - 1):
                    cv.circle(img2, m[i], 5, (0, 0, 255), -1)
                    cv.line(img=img2, pt1=m[i], pt2=m[i + 1], color=(255, 0, 0), thickness=2)
                area.append(m)
            print(area)


            '''차선 영역 구분, 이름 배당 위해 마스크 좌표 output file로 저장'''
            file_name = 'yolov8counter/mask.txt'
            with open(file_name, 'w+') as file:
                for idx, a in enumerate(area):
                    file.write(str(a)) # a는 튜플, file.write()는 str형태만 지원  
                    if idx != len(area)-1:
                        file.write('_')
            

        cv.imshow('ROI setting', img2)  # 이미지 화면 출력


    while True:
        img = cv.imread(img_path,  cv.IMREAD_UNCHANGED)
        img = cv.resize(img, (600, 400))

        
        cv.namedWindow('ROI setting')  
        cv.setMouseCallback('ROI setting', draw_mask_eventListener)  # 마우스 이벤트 발생시 리스너 호출
        
        key = cv.waitKey(1) & 0xFF 
        if key == 27:  # ESC
            cv.destroyAllWindows()
            break
      


#img_path = 'data/los_angeles.png'
#img_path = 'data/seo3.png'
img_path = 'data/seo10.png'
create_mask(img_path)
stored_path = 'result_roi.jpg'
#video_path = 'data/los_angeles.mp4'
#video_path = '192.168.172.3_20230923_205243.mp4' # 우측 영상
video_path = '192.168.172.10_20230922_173123.mp4' # 좌측 영상
cap=cv.VideoCapture(video_path)

while True:    
    ret,frame = cap.read()
    if not ret:
        break

    # 영상 정보 출력
    fps = cap.get(cv.CAP_PROP_FPS)
    print('영상 fps: ', fps)
    length = cap.get(cv.CAP_PROP_FRAME_COUNT)
    print('영상 길이: ', length)
    
    ########################################################
    ''' 이미지, 영상 합치기'''
    frame=cv.resize(frame,(600, 400))
    masked_img = cv.imread(stored_path)
    merged_img = cv.addWeighted(masked_img,0.1, frame, 0.9, 0)
    cv.imshow('merged_img', merged_img)
    
    ''' addWeighted() 대신 seamlessClone()으로 영상 합칠 경우
    mask = np.full_like(masked_img, 255)
    #--③ 합성 대상 좌표 계산(img2의 중앙)
    height, width = frame.shape[:2]
    center = (width//2, height//2)
    
    #--④ seamlessClone 으로 합성 
    normal = cv.seamlessClone(masked_img, frame, mask, center, cv.NORMAL_CLONE)
    cv.imshow("merging", normal)
    '''

    ''' 마스크 정보 읽어서 차선 정보 네이밍 -> 통행 차량 count '''
    mask_data = open("yolov8counter/mask.txt", "r")
    data = mask_data.read()
    masks = data.split("_") 
    print(masks)
    for m in masks:
        name = m.strip("'")
        print(name) # -> 이후 name 변수 이용해서 차선 구분

    ########################################################
    if cv.waitKey(1)&0xFF==27:
        break
cap.release()
cv.destroyAllWindows()