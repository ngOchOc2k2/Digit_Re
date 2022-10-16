import streamlit as st
from keras.datasets import mnist

(A, b), (C, d) = mnist.load_data()

def plot(data, label, name_plot='bar_chart'):
    if name_plot == 'bar_chart':
        a = []
        for i in range(10):
            a.append(data[label==i].shape[0])
        st.altair_chart(a)
    return True