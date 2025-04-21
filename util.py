from IPython.display import display, HTML
import pandas as pd
import streamlit as st

def horizontal(dfs):
    html = '<div style="display:flex">'
    for df in dfs:
        html += '<div style="margin-right: 32px">'
        html += df.to_html()
        html += "</div>"
    html += "</div>"
    display(HTML(html))

@st.cache_data()
def load_data():
    DATA_URL_1 = ('data/alzheimers.csv')
    DATA_URL_2 = ('data/alzheimers_encoded.csv')
    alzheimers = pd.read_csv(DATA_URL_1)
    alzheimers_encoded = pd.read_csv(DATA_URL_2)
    return alzheimers, alzheimers_encoded