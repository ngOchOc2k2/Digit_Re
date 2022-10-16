import streamlit as st
import pandas as pd
from keras.datasets import mnist
from matplotlib import pyplot as plt
import numpy as np
import model

datasets_name = ('Mnist Dataset', 'Iris data', 'None')
data = st.sidebar.selectbox('Dataset', datasets_name)
col = st.columns(4)
(X_train, y_train), (X_test,y_test) = mnist.load_data()
# X_trains, X_tests = model.Preprocess(X_train=X_train, y_train=y_train, X_test=X_test)

def display(data):
    if data == 'Mnist Dataset':
        selected_image = st.sidebar.radio('Select display Image Training Data', ('None', 'Single', '20 Image'))
    
        if selected_image == '20 Image':
            for i in range(5):
                a = [5*i, 5*i+1, 5*i+2, 5*i+3, 5*i+4]
                colima = st.columns(5)
                with colima[0]:
                    st.image(X_train[a[0]], caption='This number is ' + str(y_train[a[0]]))
                with colima[1]:
                    st.image(X_train[a[1]], caption='This number is ' + str(y_train[a[1]]))
                with colima[2]:
                    st.image(X_train[a[2]], caption='This number is ' + str(y_train[a[2]]))
                with colima[3]:
                    st.image(X_train[a[3]], caption='This number is ' + str(y_train[a[3]]))
                with colima[4]:
                    st.image(X_train[a[4]], caption='This number is ' + str(y_train[a[4]]))
        elif selected_image == 'Single':
            x = st.sidebar.slider('Seclect Image', 0, 50000, 1)
            st.image(X_train[x], caption='This number is ' + str(y_train[x]), width=70)

        model.plots(X_train, y_train)

        algothrim = st.sidebar.selectbox('Select Model', ('KNN', 'Neural Network', 'CNN', 'KMeans', 'None'))
        Select_Algo(algo=algothrim)
    return True

def Select_Algo(algo):
    if algo == 'Neural Network':
        colx = st.columns(3)
        with colx[0]:
            st.sidebar.slider('Select Layer', 1, 5, 1)
        with colx[1]:
            st.sidebar.slider('Select Node in Layer', 1, 10, 1)  

    elif algo == 'KNN':
        neighbors =  st.sidebar.slider('Select Neighbors', 1, 10, 1)
        norms = st.sidebar.slider('Select Norm ', 1, 4, 1)
        pre = st.button('Predi  ct')
        if pre:
            model.KNN(X_trains, y_train, X_tests,y_test, neighbors=neighbors, p=norms) 

    elif algo == 'KMeans':
        model.K_means(X_trains, y_train, X_tests,y_test, n_cluster=10)

if __name__ == '__main__':
    display(data=data)