import streamlit as st
import requests
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from transformers import AutoTokenizer
import time


'''
# Text Summarization
This front queries the NEWSDATA.IO [API](https://newsdata.io/api/1/news?apikey=pub_140092ed3f8c155dd1a72797f696bcf187cf2&language=en&category=sports&q=badminton)
'''

with st.form(key='params_for_api'):
    # Date_from = st.date_input('Enter date : E.g 2022-11-11')
    Keyword = st.text_input('Enter keyword : E.g World Cup')
    Topic = st.text_input('Enter topic : E.g Sports')

    st.form_submit_button('Enter')

params = dict(
    # date_from=Date_from,
    keywords=Keyword,
    category=Topic
)

st.write(params)

with st.spinner('Loading...'):
    time.sleep(10)

news_article_api_url = 'http://127.0.0.1:8000/predict'
response = requests.get(news_article_api_url, params=params)
print(response.status_code)
prediction = response.json()
for item in prediction['results']:
    if item['content'] != None :
        st.caption(item['content'])
# pred = prediction['articles']
# pred = prediction['summary']
# st.header(f'{pred}')
# st.caption(item['content'])

def load_pb(path_to_pb):
    with tf.io.gfile.GFile(path_to_pb, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='')
        return graph

model = load_pb('pretrained_model/saved_model.pb')
# model = tf.saved_model.load('/home/jessicasai/code/JessicaSai/text_summary_frontend/pretrained_model')
# print(model)
# print('test')
model_checkpoint = 'sshleifer/distilbart-cnn-12-6'
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

if 't5' in model_checkpoint:
    document = "summarize: " + item['content']
tokenized = tokenizer([item['content']], return_tensors='np')
out = model.generate(**tokenized, max_length=128)

with tokenizer.as_target_tokenizer():
    print(tokenizer.decode(out[0]))


# st.markdown('''
# Remember that there are several ways to output content into your web page...

# Either as with the title by just creating a string (or an f-string). Or as with this paragraph using the `st.` functions
# ''')

# '''
# ## Here we would like to add some controllers in order to ask the user to select the parameters of the ride

# 1. Let's ask for:
# - date
# '''

# '''
# ## Once we have these, let's call our API in order to retrieve a prediction

# See ? No need to load a `model.joblib` file in this app, we do not even need to know anything about Data Science in order to retrieve a prediction...

# 🤔 How could we call our API ? Off course... The `requests` package 💡
# '''

# url = 'http://127.0.0.1:8000/predict'

# if url == 'http://127.0.0.1:8000/predict':

#     st.markdown('Maybe you want to use your own API for the prediction, not the one provided by Le Wagon...')

# '''

# 2. Let's build a dictionary containing the parameters for our API...

# 3. Let's call our API using the `requests` package...

# 4. Let's retrieve the prediction from the **JSON** returned by the API...

# ## Finally, we can display the prediction to the user
# '''
