import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans

def Preprocess(X_train, y_train, X_test, y_test):
    # X_train = X_train.reshape(60000, 784)
    # X_test = X_test.reshape(10000, 784)
    C = []
    for i in range(10):
        temp = (X_train[y_train == i][:3000])
        for items in temp:
            C.append(items)
    C = np.array(C)
    C = C.reshape(30000, 784)
    X_test= X_test.reshape(10000, 784)
    return C, X_test

def plots(X_train, y_train):
    name_chart = st.selectbox('Select Plot', ('Bar_chart', 'Line_chart'))
    a = []
    for i in range(10):
        a.append(X_train[y_train==i].shape[0])

    labels = np.arange(10)
    chart_data = pd.DataFrame(a)

    if name_chart == 'Bar_chart':
        st.bar_chart(chart_data)    
    else:
        st.line_chart(chart_data)

    return True

def KNN(X_train, y_train, X_test, y_test, neighbors = 1, p = 2):
    model = KNeighborsClassifier(n_neighbors=neighbors, p=p, weights='distance')
    model.fit(X_train, y_train)
    A = model.predict(X_test)
    st.write('Accuracy is ', 100*accuracy_score(A, y_test)) 
    return True

def K_means(X_train, y_train, X_test, y_test, n_cluster=2):
    model = KMeans(n_clusters=10)
    model.fit(X_train)
    lala = model.predict(X_test)
    st.write('Accuracy is', 100*accuracy_score(lala, y_test))
    return True