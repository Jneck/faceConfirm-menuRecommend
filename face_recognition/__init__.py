from flask import Flask, request, jsonify

def create_app():
    app = Flask(__name__)

    from .FR_model import FR_model
    from .ftp_request import request_video
    @app.route('/')
    def index():
        # 1. FTP서버에 동영상 요청해서 받기 요청
        ## 해야할 작업 (이건 나중에 지우기)
        # 1. FTP에서 동영상 잘 받아오기
        request_video('blob.mp4')

        # 2. 모델 돌리기
        ## 해야할 작업 (이건 나중에 지우기)
        # 1. 동영상만 주면 모델 연결되는지 확인
        # 2. 모델 안에 dict 이용하여 가장 많이 등장한 값을 반환 해주는 거까지
        nickname = FR_model('face_recognition/video/blob.mp4')

        # 3. 결과값 반환
        # return jsonify({'nickname': nickname})
        print(nickname)

        ## 찬영아 test 해보께
        return nickname
    return app
