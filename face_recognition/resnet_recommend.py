# http://shancarter.github.io/mr-data-converter/
import pandas as pd
import random
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


def resnet_recommend(data, path):
    print(data.head(5))
    # print(data['burger_calories'].sort_values(ascending=False))

    # device 확인->cuda
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 데이터 형태 바꾸기
    transforms_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 라벨링
    class_names = ['20대 남자', '20대 여자']

    # 학습된 모델 불러오기->결과 확인하기
    model = models.resnet34(pretrained=True)
    model.fc = nn.Linear(512, 3)
    model.load_state_dict(torch.load('./face_recognition/data/model_dict.pth'))
    model.eval()
    model = model.to(device)

    image = Image.open(path)
    image = transforms_test(image).unsqueeze(0).to(device)  # 배치 사이즈 추가하기
    print(image.shape)

    with torch.no_grad():
        outputs = model(image)
        _, preds = torch.max(outputs, 1)
        print(preds)
        print(class_names[preds[0]])

    # 데이터 전처리하기
    data_total = data[(data['burger_calories'] != 0) & (~data['burger_menu'].str.contains('세트')) & (~data['burger_menu'].str.contains('주니어')) & (
        ~data['burger_menu'].str.contains('팩'))]

    # resent 결과가 "20대 여자" 일 때
    if class_names[preds[0]] == '20대 여자':
        for_20_girls = []
        # 영양정보가 0인 데이터 제외하고 추출
        data_total = data_total.sort_values('burger_calories')
        for_20_girls = data_total[:5]['buger_id'].values.tolist()
        recommend_result = random.choice(for_20_girls)

    # resent 결과가 "20대 남자" 일 때
    elif class_names[preds[0]] == '20대 남자':
        for_20_boys = []
        # 영양정보가 0인 데이터 제외하고 추출
        data_total = data_total.sort_values('burger_protein', ascending=False)
        for_20_boys = data_total[:5]['buger_id'].values.tolist()
        recommend_result = random.choice(for_20_boys)
    else:
        pass

    return recommend_result
