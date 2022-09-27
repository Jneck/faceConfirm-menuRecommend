from flask import Flask, request, jsonify


def create_app():
    app = Flask(__name__)

    from .FR_model import FR_model
    from .ftp_request import request_video
    from .cut_image import cut_image
    from .rec_by_order import recommend_burger
    from .rec_by_order import recommend_side_menu

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

    @app.route('/rec_burger_by_order', methods=['GET'])
    def recommend_buger():
        # 요청 받은 1년치 주문 목록에서 오늘 주문한 user_id들의 burger_id_list 추천
        # 코사인 유사도 사용.
        return jsonify({'burger_id_list': recommend_burger(dict(request.json))})

    @app.route('/rec_side_menu_by_order_buger', methods=['GET'])
    def recommend_side():
        # 요청받은 user_id와 최근 주문 메뉴를 기반으로 사이드메뉴 추천
        # 연관분석 사용
        return jsonify({'side_menu_id_list': recommend_side_menu(dict(request.json))})

    return app
