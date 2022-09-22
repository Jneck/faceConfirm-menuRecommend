import cv2
import numpy as np
import os

# 딕셔너리 주면 그 안에서 가장 value값이 큰 key값을 반환하는 함수
def most_count_dict(counted_dict):
    max_count = 0
    nick_name = ''
    for key, value in counted_dict.items():
        if value > max_count:
            max_count = value
            nick_name = str(key)
    return nick_name

def FR_model():
    labels = ["chan_youn", "hoi_eun", "nam_kyung", "seong_bin"]  # 라벨 지정
    nick_name_dict = {}

    face_cascade = cv2.CascadeClassifier('face_recognition/haarcascade_frontalface_default.xml')
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("face_recognition/yml_data/face-trainner.yml")  # 저장된 값 가져오기

    cap = cv2.VideoCapture(0)  # 카메라 실행

    cnt = 0
    cnt2 = 0
    if cap.isOpened() == False:  # 카메라 생성 확인
        exit()

    while True:
        ret, img = cap.read()  # 현재 이미지 가져오기
        img_temp = cv2.flip(img,1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 흑백으로 변환
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)  # 얼굴 인식

        for (x, y, w, h) in faces:
            b_center = (x + w / 2, y + h / 2)
            # print(b_center)
            if b_center[0] < 280 or b_center[0] > 360 or b_center[1] < 200 or b_center[1] > 280:
                continue
            roi_gray = gray[y:y + h, x:x + w]  # 얼굴 부분만 가져오기

            id_, conf = recognizer.predict(roi_gray)  # 얼마나 유사한지 확인

            if conf >= 55 and conf < 500:
                print(id_, conf)
                cnt += 1
                # 성빈이
                if id_ == 3:
                    cnt2 += 1
                font = cv2.FONT_HERSHEY_SIMPLEX  # 폰트 지정
                name = labels[id_]  # ID를 이용하여 이름 가져오기

                # 나온 사람 수를 딕셔너리에 닉네임, 등장횟수 꼴로 저장
                if name not in nick_name_dict:
                    nick_name_dict[str(name)] = 0
                # 등장했으면 횟수 1 추가
                nick_name_dict[str(name)] += 1


                cv2.putText(img_temp, name, (x, y), font, 1, (0, 0, 255), 2)
                cv2.rectangle(img_temp, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.rectangle(img_temp, (230, 150), (410, 330), (255, 0, 0), 2)
        cv2.imshow('Preview', img_temp)  # 이미지 보여주기
        if cv2.waitKey(10) >= 0:  # 키 입력 대기, 10ms
            break

    print('Frame width:', int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    print('Frame height:', int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    print('Frame count:', int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))

    fps = cap.get(cv2.CAP_PROP_FPS)
    print('FPS:', fps)

    print(cnt2, cnt, str(int(cnt2/cnt * 100)) + '%')
    # 전체 종료
    cap.release()
    cv2.destroyAllWindows()
    return most_count_dict(nick_name_dict)
