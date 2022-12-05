import streamlit as st
import time
import requests

# Add a Background Image from My Computer
import base64
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"jpg"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )

add_bg_from_local('background4.jpg')


# title & header
'''
# Text Summarization
This front queries the NEWSDATA.IO [API](https://newsdata.io/api/1/news?apikey=pub_140092ed3f8c155dd1a72797f696bcf187cf2&language=en&category=sports&q=badminton)
'''

# input parameters
with st.form(key='params_for_api'):
    # Date_from = st.date_input('Enter date : E.g 2022-11-11')
    Keyword = st.text_input('Enter keyword : E.g World Cup')
    #Topic = st.text_input('Enter topic : E.g Sports')

    st.form_submit_button('Enter')

params = dict(
    # date_from=Date_from,
    keywords=Keyword,
    #category=Topic
)

st.write(params)

with st.spinner('Loading...'):
    time.sleep(30)

backend_url = 'https://textsummary2-owfdsgrlca-as.a.run.app/predict'
response = requests.get(backend_url, params=params)
print(response.status_code)
prediction = response.json()

st.text_area('Sports Article', height=250, value=prediction['article'])

col1, col2 = st.columns(2)

with col1:
   st.text_area("Base Model Summary", height=250, value=prediction['summary'])

with col2:
   st.text_area("Finetuned Model Summary", height=250, value=prediction['summary2'])
