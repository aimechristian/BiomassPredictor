import numpy as np
import pandas as pd
import streamlit as st
#import plotly.graph_objects as go
from sklearn import ensemble, tree
from sklearn.model_selection import train_test_split


data = pd.read_csv('Final_Dataset.csv')

#Feature-Label Split
features = data.loc[:, data.columns != 'Crop_Biomass_Aerial']
target = data.loc[:, data.columns == 'Crop_Biomass_Aerial']
introduction = "SEABEM (Stacked Ensemble Algorithms Biomass Estimator Model), is a free and open-sourced web application that enables\ 
            Sub-Saharan African farmers to predict yield before planting,\ 
            hence making better data-based operating decisions and becoming more profitable."

#Train-Test Split
x_train, x_test, y_train, y_test = train_test_split(features,target,test_size=0.25,random_state=20)

class StreamlitApp:

    def __init__(self):
        self.model = ensemble.StackingRegressor([('Gboost', ensemble.GradientBoostingRegressor()),
                                                 ('rf', ensemble.RandomForestRegressor()),
                                                 ('DTR', tree.DecisionTreeRegressor())])

    def train_data(self):
        self.model.fit(x_train, y_train)
        return self.model

    def construct_sidebar(self):

        cols = [col for col in features.columns]

        st.sidebar.markdown(
            '<p class="header-style">Biomass Prediction Variables</p>',
            unsafe_allow_html=True
        )
        Site_Latitude = st.sidebar.number_input(
            f"Input {cols[0]}"
        )

        Site_Longitude = st.sidebar.number_input(
            f"Input {cols[1]}"
        )

        Site_Soil_Sand_Percentage = st.sidebar.number_input(
            f"Input {cols[2]}"
        )

        Site_Soil_Silt_Percentage = st.sidebar.number_input(
            f"Input {cols[3]}"
        )
        
        Site_Soil_Clay_Percentage = st.sidebar.number_input(
            f"Input {cols[4]}"
        )
        
        Site_Precipitation_mm = st.sidebar.number_input(
            f"Input {cols[5]}"
        )
        
        Site_Temperature_Celsius = st.sidebar.number_input(
            f"Input {cols[6]}"
        )
        
        values = [Site_Latitude, Site_Longitude, Site_Soil_Sand_Percentage, Site_Soil_Silt_Percentage, Site_Soil_Clay_Percentage, Site_Precipitation_mm, Site_Temperature_Celsius]

        return values

    def construct_app(self):

        self.train_data()
        values = self.construct_sidebar()

        values_to_predict = np.array(values).reshape(1, -1)

        prediction = self.model.predict(values_to_predict)
        
        st.title("SEABEM")

        st.markdown(
            """
            <style>
            .header-style {
                font-size:25px;
                font-family:sans-serif;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <style>
            .font-style {
                font-size:20px;
                font-family:sans-serif;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
        
        
        
        st.markdown(
            '<p class="header-style"> introduction </p>',
            unsafe_allow_html=True
        )

        column_1, column_2 = st.columns(2)
        column_1.markdown(
            f'<p class="header-style" >Prediction (tons/ha)</p>',
            unsafe_allow_html=True
        )
        column_1.write(f"{prediction} ")

        column_2.markdown(
            '<p class="header-style" >Biomass Rank </p>',
            unsafe_allow_html=True
        )
        if prediction > 4:
            column_2.write(f"Very Good")
            
        elif prediction <2:
            column_2.write(f"Bad")
            
        else:
            column_2.write(f"Good")

        return self


sa = StreamlitApp()
sa.construct_app()
