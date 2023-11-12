from typing import List

from fastapi import FastAPI, HTTPException

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from schema import PostGet

import os
from catboost import CatBoostClassifier
import pandas as pd


SQLALCHEMY_DATABASE_URL = "postgresql://robot-startml-ro:pheiph0hahj1Vaif@postgres.lab.karpov.courses:6432/startml"

engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

### Загрузка таблицы пользователей
df_users = pd.read_sql("SELECT * FROM user_data", SQLALCHEMY_DATABASE_URL)

### Загрузка таблицы постов
# df_posts = pd.read_sql("SELECT * FROM post_text_df", SQLALCHEMY_DATABASE_URL)

### Загрузка таблицы постов с доп. признаками tf-idf/svd
df_posts_from_sql = pd.read_sql('SELECT * FROM i_n_20_embed_posts_fp3_1011',
                                con=engine, index_col='index')

### Загрузка файла модели
def get_model_path(path: str) -> str:
    if os.environ.get("IS_LMS") == "1":  # проверяем где выполняется код в лмс, или локально. Немного магии
        MODEL_PATH = '/workdir/user_input/model'
        # MODEL_PATH = '/workdir/user_input/model_5+5fds_idf_1305640-13'
    else:
        MODEL_PATH = path
    return MODEL_PATH

def load_models():
    model_path = get_model_path("model")
    from_file = CatBoostClassifier()
    from_file.load_model(model_path)
    return from_file


model_from_file = load_models()

app = FastAPI()


def get_db():
    with SessionLocal() as db:
        return db

@app.get("/post/recommendations/", response_model=List[PostGet])
def recommended_posts(id: int, limit: int = 5) -> List[PostGet]:
    # # Данные выбранного user_id:
    user_cols = ['gender', 'age', 'country', 'city', 'exp_group', 'os', 'source']
    cur_user_data = df_users[df_users['user_id'] == id]
    if len(cur_user_data) == 0:
        raise HTTPException(404, "User not found")
    cur_user_data = cur_user_data.drop('user_id', axis=1)

    post_topics = ['business', 'covid', 'entertainment', 'sport', 'politics', 'tech', 'movie']

    # # Собираем фрейм для заданного user_id:
    # temp_df = pd.DataFrame()
    specific_df = pd.DataFrame()
    # # !!! Посты с доп. признаками
    # specific_df = df_posts_from_sql.sample(n=2000)

    for cur_topic in post_topics:
        # 300 случайных постов по текущей теме
        temp_df = df_posts_from_sql[df_posts_from_sql['topic'] == cur_topic].sample(n=300)
        # накидываем в общий фрейм для предсказания
        specific_df = pd.concat([specific_df, temp_df], axis=0, ignore_index=True)

    # добавляем во фрейм для предсказания данные выбранного пользователя
    for col in user_cols:
        specific_df[col] = cur_user_data[col].values[0]

    # уберем post_id и text для предсказания
    df_for_predict = specific_df.drop(['post_id', 'text'], axis=1)

    # получаем вероятности для сформированного фрейма
    # и добавляем их в новую колонку
    pred = model_from_file.predict_proba(df_for_predict)
    specific_df['pred'] = pred[:, 1]
    # df_posts_from_sql['pred'] = pred[:, 1]

    # выделяем топ-5 из полученных вероятностей
    # top_5 = df_posts_from_sql.sort_values(by='pred', ascending=False).head(limit)
    top_pred = specific_df.sort_values(by='pred', ascending=False).head(limit)

    # получаем соответствующие 5 постов
    result = []
    temp_dict = {}
    for cur_id in top_pred['post_id'].values:
        # temp_df2 = df_posts_from_sql[df_posts_from_sql['post_id'] == cur_id]
        temp_df2 = specific_df[specific_df['post_id'] == cur_id]
        temp_dict = {
                      "id": temp_df2['post_id'].values[0],
                      "text": temp_df2['text'].values[0],
                      "topic": temp_df2['topic'].values[0]
                    }
        result.append(temp_dict)
        # result.append(df_posts[df_posts['post_id'] == cur_id])
    if not result:
        raise HTTPException(404, "Post not found")
    else:
        return result
