import random

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
from datetime import datetime


def recommend_burger(data):
    # 요청 받은 데이터에서 유저당 주문한 메뉴아이디를 기반으로 카운팅하여 그룹화.
    request_df = pd.DataFrame.from_dict(data['one_year_order_list'])
    counting_by_user_id_and_menu_id = request_df.groupby(['user_id', 'menu_id']).size().reset_index(name='counts')

    d_list = []
    for x, y in counting_by_user_id_and_menu_id.iterrows():
        d_list.append([y['user_id'], y['menu_id'], y['counts']])

    counting_df_by_user_id_and_menu_id = pd.DataFrame(d_list, columns=['user_id', 'menu_id', 'counts'])

    # 정규화
    mean_rating_user = counting_df_by_user_id_and_menu_id.groupby('user_id')['counts'].mean().reset_index(
        name='Mean-Rating-User')
    mean_data = pd.merge(counting_df_by_user_id_and_menu_id, mean_rating_user, on='user_id')
    mean_data['Diff'] = mean_data['counts'] - mean_data['Mean-Rating-User']
    mean_data['Square'] = (mean_data['Diff']) ** 2
    norm_data = mean_data.groupby('user_id')['Square'].sum().reset_index(name='Mean-Square')
    norm_data['Root-Mean-Square'] = np.sqrt(norm_data['Mean-Square'])
    mean_data = pd.merge(norm_data, mean_data, on='user_id')
    mean_data['Norm-Rating'] = mean_data['Diff'] / (mean_data['Root-Mean-Square'])
    mean_data['Norm-Rating'] = mean_data['Norm-Rating'].fillna(0)
    max_rating = mean_data.sort_values('Norm-Rating')['Norm-Rating'].to_list()[-1]
    min_rating = mean_data.sort_values('Norm-Rating')['Norm-Rating'].to_list()[0]
    mean_data['Norm-Rating'] = 5 * (mean_data['Norm-Rating'] - min_rating) / (max_rating - min_rating)
    mean_data['Norm-Rating'] = np.ceil(mean_data['Norm-Rating']).astype(int)
    norm_ratings = mean_data[['user_id', 'menu_id', 'Norm-Rating']]
    mean_data.sort_values('Norm-Rating')

    burger_user = norm_ratings.pivot_table('Norm-Rating', index='user_id', columns='menu_id')
    burger_user.fillna(0, inplace=True)

    # 유저와 유저 간의 유사도
    user_based_collab = cosine_similarity(burger_user, burger_user)
    user_based_collab = pd.DataFrame(user_based_collab, index=burger_user.index, columns=burger_user.index)

    # 당일 주문한 user_id list
    today_date = datetime.today().strftime('%Y-%m-%d')
    today_order_user_id_list = request_df['user_id'][request_df['order_date'] == str(today_date)].to_list()

    user_id_rec_burger_id = {}
    for user_id in today_order_user_id_list:

        # 상위 3개 유사도 가진 유저
        if 3 > len(user_based_collab.index):
            top_3_list = user_based_collab[user_id].sort_values(ascending=False)[1:].index
        else:
            top_3_list = user_based_collab[user_id].sort_values(ascending=False)[1:4].index
        print(user_id, top_3_list)

        top_3_list_to_str = []
        for s in top_3_list:
            top_3_list_to_str.append(str(s))

        # 자신이 먹은 메뉴 제외한 메뉴 리스트 리턴.
        my_order_buger_id_list = norm_ratings['menu_id'][norm_ratings['user_id'] == int(user_id)].to_list()
        print(my_order_buger_id_list)

        # 유사도가 높은 유저기반으로 버거 리스트 만들기
        burger_list = []
        for i in top_3_list_to_str:
            burger_list.append(norm_ratings['menu_id'][norm_ratings['user_id'] == int(i)].to_list())
        burger_list_final = sum(burger_list, [])
        burger_list_final = list(set(burger_list_final) - set(my_order_buger_id_list))[:3]

        # 리턴할 유저 아이디별 버거 리스트
        user_id_rec_burger_id[user_id] = burger_list_final

    return user_id_rec_burger_id


def recommend_side_menu(data):
    request_df = pd.DataFrame.from_dict(data['one_year_order_list'])
    anal_df = request_df.reset_index(drop=True)
    anal_df['Index'] = anal_df.index

    # 늘먹던대로 user_id, "always_menu = menu_name" ,"always_id = menu_id"
    request_always_df = pd.DataFrame.from_dict(data['always'])

    cnt = len(anal_df['Index'].unique())

    # 당일 주문한 user_id list
    today_date = datetime.today().strftime('%Y-%m-%d')
    today_order_user_id_list = request_df['user_id'][request_df['order_date'] == str(today_date)].to_list()


    # 연관분석을 위한 배열 생성
    df_tmp_arr = [[] for i in range(cnt + 1)]  # 빈 리스트 제작
    num = 0
    for i, j in zip(anal_df['order_menu'], anal_df['side_menu']):
        df_tmp_arr[anal_df['Index'][num]].append(i)
        df_tmp_arr[anal_df['Index'][num]].append(j)
        num += 1

    num = 0
    for i in df_tmp_arr:
        df_tmp_arr[num] = list(set(df_tmp_arr[num]))  # 원래 중복값이 있던 자리에 unique값만 넣기
        num += 1
    # TransactionEncoder-> T/F값으로 unique한 리스트와 데이터 비교
    te = TransactionEncoder()
    te_ary = te.fit(df_tmp_arr).transform(df_tmp_arr)
    df = pd.DataFrame(te_ary, columns=te.columns_)

    frequent_itemsets = apriori(df, min_support=0.0001, use_colnames=True)  # 지지도 0.0001이상인 itemsets 계산
    rules = association_rules(frequent_itemsets, metric='lift', min_threshold=1)
    rules.sort_values(by='confidence', ascending=False)
    frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))

    # user가 최근 먹은 버거 기반으로 사이드 메뉴 추천  => 최근 먹은 버거 이름 필요!
    user_id_rec_side_menu_id = {}
    for user_id in today_order_user_id_list:
        menu_name = request_always_df[request_always_df['user_id'] == user_id]['always_menu'].iloc[0]

        frequent_itemsets_by_menu_name = frequent_itemsets[frequent_itemsets['itemsets'].apply(lambda x: menu_name in x)]
        frequent_itemsets_length_2 = frequent_itemsets_by_menu_name[frequent_itemsets_by_menu_name['length'] == 2]  # 길이가 2개인 세트

        # 자기 자신이 먹은 제품 제외하여 rec_side_menu_list에 이름으로 담기.
        item_list = []
        rec_side_menu_list = []
        for i in frequent_itemsets_length_2['itemsets']:
            item_list.append(list(i))
            rec_side_menu_list = sum(item_list, [])
            rec_side_menu_list = [i for i in rec_side_menu_list if i != menu_name]

        # rec_side_menu를 id로 반환하기 위함.
        tmp = []
        rec_side_id_list = []
        for i in rec_side_menu_list:
            # print(i)
            tmp.append(anal_df['side_id'][anal_df['side_menu'] == i].to_list())
            rec_side_id_list = sum(tmp, [])
            rec_side_id_list = list(set(rec_side_id_list))
        user_id_rec_side_menu_id[user_id] = random.sample(rec_side_id_list,3) # 사이드 메뉴중 랜덤 3개 추천
    return user_id_rec_side_menu_id
