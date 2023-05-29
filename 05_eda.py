# This app is just for learning purposes created on 09/12/2022
import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
# WebPage ka Title
st.markdown('''
**Exploratory Data Analysis**
This app is Developed by Izhar Khan Khattak
EDA -> Eplorratory Data Analysis App 
''')
with st.sidebar.header("Upload your data from local PC: (.csv)"):
    uploaded_file = st.sidebar.file_uploader("Upload your fiel", type=['csv'])
    df = sns.load_dataset('titanic')
    st.sidebar.markdown("[Example CSV file]")
#Profiling report for pandas
if uploaded_file is not None:
    def load_csv():
        csv = pd.read_csv(uploaded_file)
        return csv
    df = load_csv
    pr = ProfileReport(df, explorative=True)
    st.header("**Input**")
    st.write(df)
    st.header("**Profiling Report with pandas**")
    st_profile_report(pr)
else:
    st.info("Awaiting for you man! Ab Upload b Kardo")
    if st.button("Press to use example data"):
        def load_data():
            a = pd.DataFrame(np.random.rand(100, 5),
            columns=["A", "B", "C", "D", "E"])
            return a
        df = load_data()
        pr = ProfileReport(df, explorative=True)
        st.header("**Input**")
        st.write(df)
        st.header("**Profiling Report with pandas**")
        st_profile_report(pr)