import streamlit as st
import torch
from model import NeuralNetwork, vec

st.write("# Welcome in Coronavirus Tweet Classifier")
st.write("Tweet can be labeled with")
st.markdown("1. Extremely Negative")
st.markdown("2. Negative")
st.markdown("3. Neutral")
st.markdown("4. Positive")
st.markdown("5. Extremely Positive")

def get_label(value):
    if value == 0:
        return "Extremely Negative"
    elif value == 1:
        return "Extremely Positive"
    elif value == 2:
        return "Negative"
    elif value == 3:
        return "Neutral"
    elif value == 4:
        return "Positive"
    
model = torch.load("models/neuralnetwork.pth")

tweet = st.text_input("Enter your tweet")
label = model(vec(tweet))
l = label.squeeze().argmax().cpu().numpy()
st.text_input(get_label(l))