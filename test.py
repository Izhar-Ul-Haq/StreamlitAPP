import streamlit as st
import seaborn as sns
import numpy as np
st.header("Izhar Ul Haq Python StreamLit Project")
st.text("Hi this is just a simple text")
st.header("This is a big header")
df = sns.load_dataset('iris')
st.write(df.head(10))
st.bar_chart(df['sepal_length'])
st.line_chart(df['sepal_length'])
print(df)
#Working Titanic Data
df1 = sns.load_dataset("titanic")
print(df1)
st.line_chart(df1["age"][1:50])
st.write(df1.head())
st.bar_chart(df1["sex"].value_counts())