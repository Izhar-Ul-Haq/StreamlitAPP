import streamlit as st
import pandas as pd
import sklearn as sk
import seaborn as sns
import numpy as np
import PIL as pil
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
header = st.container()
data_set = st.container()
model_train = st.container()
features = st.container()
st.header("Izhar Ul Haq aka Izhar Khan Khattak ")
with header:
    st.title("Hi this is just the title")
    st.text("This is just the simple text")
with data_set:
    st.header("Here we will import data set")
    st.text("Hi this is text for the data set")
    df = sns.load_dataset("titanic")
    st.write(df.head())
    st.line_chart(df["age"])
    st.bar_chart(df["sex"].value_counts())
    st.bar_chart(df["sex"].sample(50))
    st.line_chart(df["sex"].sample(50))
    df = df.dropna()
with features:
    st.header("Here we are gonna add some features to the App")
    st.text("This text is gonna be for features")
    st.markdown("1: Fatures 1: This is one of the features")
    st.markdown("2: Fatures 1: This is one of the features")
    st.markdown("3: Fatures 1: This is one of the features")
with model_train:
    st.header("Here we will train the model")
    st.text("Text for the model training")
    #pehly columns may ap ki selection points hun
    input, display = st.columns(2)
    max_depth = input.slider("How many people do you know?", min_value=0, max_value=100, value=20, step=5) 
#n_estimator
n_estimator = input.selectbox("How many node shoud be there in Random Forest", options=[10, 20, 30, 40, 50, 'No Limit'])
n_input = input.text_input("Which feature we should add?")
# st.header("Izhar Ul Haq Python StreamLit Project")
# st.text("Hi this is just a simple text")
# st.header("This is a big header")
# df = sns.load_dataset('iris')
# st.write(df.head(10))
# st.bar_chart(df['sepal_length'])
# st.line_chart(df['sepal_length'])
# print(df)
# #Working Titanic Data
# df1 = sns.load_dataset("titanic")
# print(df1)
# st.line_chart(df1["age"][1:50])
# st.write(df1.head())
# st.bar_chart(df1["sex"].value_counts())
model = RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimator)
X = df[[n_input]]
y = df[['fare']]
model.fit(X,y)
pred = model.predict(y)
# display
display.subheader("Mean Absolute error")
display.write(mean_absolute_error(y, pred))
display.subheader("Mean Squared error")
display.write(mean_squared_error(y, pred))
display.subheader("R Squared Score")
display.write(r2_score(y, pred))
input.write(df.columns)