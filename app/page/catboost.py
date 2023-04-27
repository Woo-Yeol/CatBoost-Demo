import streamlit as st
from utils.page import Page
import pandas as pd
import numpy as np
import plotly_express as px
from utils.sidebar import filter_table_option
import json

from catboost import CatBoostRegressor

class CatBoost():
    def __init__(self,data=None) -> None:
        super().__init__()
        self.data = data
        self.use_best_model = True
        self.model = CatBoostRegressor()
    
    def load_model(self,pth_path='./page/saved_model/CatBoost.cbm'):
        self.model.load_model(pth_path)

    def pred(self, test):
        return self.model.predict(test)

class CatBoostPage(Page):
    def __init__(self, data, **kwargs):
        name = "CatBoost"
        super().__init__(name, data, **kwargs)

    def read_json(self):
        with open("./page/data/isbn2idx.json", "r") as st_json:
            self.isbn2idx = json.load(st_json)
        with open("./page/data/user2idx.json", "r") as st_json:
            self.user2idx = json.load(st_json)

    def read_csv(self):
        self.books = pd.read_csv('./page/data/books.csv')
        self.users = pd.read_csv('./page/data/users.csv')

    def content(self):
        st.markdown("## CatBoost Inference")
        self.read_json()
        self.read_csv()

        model = CatBoost()
        model.load_model()

        # Inference User ID, ISBN
        user_id, isbn = st.text_input('User ID','11676',key='user_id'),st.text_input('ISBN','0002005018',key='isbn')

        if user_id and isbn:
            # make context matrix
            if (user_id in self.user2idx) and (isbn in self.isbn2idx):
                user_id = self.user2idx[user_id]
                isbn = self.isbn2idx[isbn]

                context_vector = pd.DataFrame({'user_id' : {1 :user_id} , 'isbn':{1 : isbn}})
                context_vector = context_vector.merge(self.users[['user_id','age','location_city','location_state','location_country']], on='user_id', how='left').merge(self.books[['isbn', 'category', 'publisher', 'language', 'book_author', 'year_of_publication']], on='isbn', how='left')
                
                # print
                st.markdown("### Context Vector")
                st.dataframe(context_vector)

                # Prediction
                st.markdown("### Prediction")
                prediction = model.pred(context_vector)
                st.write(f"Predict Rating : {prediction[-1]}")
            else:
                st.markdown("### Cold Start")
                st.write(f"Mean Rating : {7.06}")

                

