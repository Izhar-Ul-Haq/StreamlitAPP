# import libraries
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
# import data_set
df = px.data.gapminder()
st.header("Izhar Ul Haq")
st.write(df)
st.title("I am gonna be a programmer")
st.write(df.head())
st.write(df.columns)
st.write(df.describe())
st.write(df.info())
# Data Management Using Plotly
year_option = df['year'].unique().tolist()
year = st.selectbox("Which year we should plot?", year_option, 0)
# df = df[df['year']==year]
# plotting
fig =px.scatter(df, x='gdpPercap', y = 'lifeExp', size = 'pop', color = 'country', hover_name='country',
log_x=True, size_max=55, range_x=[100, 10000], range_y=[20, 90], animation_frame='year', animation_group='country')
# fig1 =px.scatter(df, x='gdpPercap', y = 'lifeExp', size = 'pop', color = 'continent', hover_name='continent',
# log_x=True, size_max=55, range_x=[100, 10000], range_y=[20, 90])
st.write(fig)
# st.write(fig1)