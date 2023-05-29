#import libraries
import sklearn as sk
import numpy as np
import pandas as pd
import matplotlib.pyplot as pt
import streamlit as st
import plotly.express as px
import plotly as plt
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
st.header("Izhar Khan Khattak Exploring Different Models")
#Data Set
data_set = st.sidebar.selectbox(
    'Select Data Set',
    ("Breast Cancer","Iris", "Wine")
)
classifier_name = st.sidebar.selectbox(
    "Select Classifier",
    ("KNN", "SVM", "Random Forest")
)
def get_data_set(data_set):
    data = None
    if data_set=="Iris":
        data = datasets.load_iris()
    elif data_set=="Wine":
        data = datasets.load_iris()
    else:
        data = datasets.load_breast_cancer()
    x = data.data
    y = data.target
    return x, y
X, y = get_data_set(data_set)
st.write("The shape of the data set is:", X.shape)
st.write("The number of the unique classes:", len(np.unique(y)))
def add_parameter_ui(classifier_name):
    params = dict()
    if classifier_name=="SVM":
        C = st.sidebar.slider("C",0.01, 10.0)
        params["C"] = C
    elif classifier_name=="KNN":
        K = st.sidebar.slider("K", 1, 15)
        params['K'] = K
    else:
        max_depth = st.sidebar.slider("max_depth", 2, 15)
        params['max_depth'] = max_depth 
        n_estimators = st.sidebar.slider("n_estimators", 1, 15)
        return params
params = add_parameter_ui(classifier_name)
def get_classifier(classifier_name, params):
    clf = None
    if classifier_name == "SVM":
        clf.SVC(C=params['C'])
    elif classifier_name == "KNN":
        clf = KNeighborsClassifier(n_neighbors =  params['K'])
    else:
        clf = RandomForestClassifier(n_estimators = params['n_estimators'],
        max_depth = params['max_depth'], random_state = 1234)
    
clf = get_classifier(classifier_name, params)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size  = 0.2, random_state =1234)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
st.write(f"Classifier: {classifier_name}")
st.write("Accuracy: {acc}")
#There is some problem with this code
#We have to uncoverd it later