# =========================================================
#                   StreamLit Learning
# =========================================================
import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
# =========================================================
#                   Display Text
# =========================================================
# st.text('Fixed width text')
# st.markdown('_Markdown_') # see *
# st.latex(r''' e^{i\pi} + 1 = 0 ''')
# st.write('Most objects') # df, err, func, keras!
# st.write(['st', 'is <', 3]) # see *
# st.title('My title')
# st.header('My header')
# st.subheader('My sub')
# st.code('for i in range(8): foo()')
# ==========================================================
st.header("This Video is brought to you by Izhar Khan Khattak")
st.text("Hi! This is a text in streamlit")
st.header("Izhar Khan Khattak Learning Streamlit")
df = sns.load_dataset("iris")
st.write(df.head(10))
st.write(df[["species", "sepal_length", "petal_length"]].head(15))
# =========================================================
#                   Bar Chart
# =========================================================
st.bar_chart(df['sepal_length'])
# =========================================================
#                   Line Chart
# =========================================================
st.line_chart(df['sepal_length'])
# =========================================================
# importing GapMinder data from px
df1 = px.data.gapminder()
st.header("GapMinder Data")
st.write(df1)
st.header("Head")
st.write(df1.head(5))
st.header("Columns")
st.write(df1.columns)
st.header("Data Summary")
st.write(df1.describe())
# =========================================================
#                   Data Management
# =========================================================
year_option = df1["year"].unique().tolist()
year = st.selectbox("Which year we should plot?", year_option,0)
df1 = df1[df1["year"]==year]
# =========================================================
#                   Ploting
# =========================================================
fig = px.scatter(df1, x = 'gdpPercap', y = 'lifeExp',
size = 'pop', color = 'country', hover_name = 'country',
log_x = True, size_max = 55, range_x = [100, 100000], 
range_y = [20, 90])
fig = fig.update_layout(width = 800)
st.write(fig)
# =========================================================