import cv2

def cut_image(video_path):
    # 영상의 의미지를 연속적으로 캡쳐할 수 있게 하는 class
    vidcap = cv2.VideoCapture(video_path)

    count = 0

    while (vidcap.isOpened()):
        # read()는 grab()와 retrieve() 두 함수를 한 함수로 불러옴
        # 두 함수를 동시에 불러오는 이유는 프레임이 존재하지 않을 때
        # grab() 함수를 이용하여 return false 혹은 NULL 값을 넘겨 주기 때문
        ret, image = vidcap.read()
        count += 1
        if count == 10:
            cv2.imwrite("face_recognition/data/img/user_img.jpg", image)
            break

    vidcap.release()
