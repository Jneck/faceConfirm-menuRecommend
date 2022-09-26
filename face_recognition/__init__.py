from flask import Flask, request, jsonify

def create_app():
    app = Flask(__name__)

    from .FR_model import FR_model
    from .ftp_request import request_video
    from .cut_image import cut_image
    @app.route('/')
    def index():
        # 1. FTP서버에 동영상 요청해서 받기 요청
        video_path = request_video('blob.mp4')

        # 2. 동영상에서 이미지 잘라서 저장하기
        cut_image(video_path)

        # 3. 모델 돌리기
        nickname = FR_model('face_recognition/video/blob.mp4')

        # 3. 결과값 반환
        print(nickname)
        return jsonify({'nickname': nickname})
    return app
