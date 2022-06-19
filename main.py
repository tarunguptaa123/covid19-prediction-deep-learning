import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
from bs4 import BeautifulSoup
from keras.preprocessing import image
import cv2
import tensorflow as tf
import tempfile
from PIL import Image, ImageOps
model = tf.keras.models.load_model('2resnet50.hdf5')
st.set_page_config(layout="wide")
st.sidebar.markdown(f"<span style='color: black;font-size: 36px;font-weight: bold;'>Infector </span>", unsafe_allow_html=True)
st.sidebar.title("Data BY WHO 🧭")
with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>',unsafe_allow_html=True)
    animation_symbol='❅'
st.markdown(f"""<div class="snowflake">{animation_symbol}</div>
                <div class="snowflake">{animation_symbol}</div>
                <div class="snowflake">{animation_symbol}</div>
                <div class="snowflake">{animation_symbol}</div>
                <div class="snowflake">{animation_symbol}</div>
                <div class="snowflake">{animation_symbol}</div>
                <div class="snowflake">{animation_symbol}</div>
                <div class="snowflake">{animation_symbol}</div>
                <div class="snowflake">{animation_symbol}</div>
                <div class="snowflake">{animation_symbol}</div>
                <div class="snowflake">{animation_symbol}</div>
                <div class="snowflake">{animation_symbol}</div>
                <div class="snowflake">{animation_symbol}</div>
                <div class="snowflake">{animation_symbol}</div>
                <div class="snowflake">{animation_symbol}</div>
                <div class="snowflake">{animation_symbol}</div>
                <div class="snowflake">{animation_symbol}</div>
                <div class="snowflake">{animation_symbol}</div>
                <div class="snowflake">{animation_symbol}</div>
                <div class="snowflake">{animation_symbol}</div>""",unsafe_allow_html=True)

import base64


@st.cache(allow_output_mutation=True)
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()


def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = '''
    <style>
    body {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str

    st.markdown(page_bg_img, unsafe_allow_html=True)
    return

def predictor():
    with st.container():
        uploaded_file= st.file_uploader("Choose a file")
        if uploaded_file is not None:

            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())
            st.image(uploaded_file, caption = 'your uploded Xray',width=200)
            cap=cv2.imread(tfile.name,cv2.IMREAD_COLOR)
            plt.imshow(cap), plt.axis("off")
            #plt.show()
            cap=cv2.resize(cap,(128,128))
            resized_arr = tf.keras.preprocessing.image.img_to_array(cap)
            resized_arr=resized_arr/255
            resized_arr = np.expand_dims(resized_arr, axis = 0)
            result = model.predict(resized_arr)
            predicted_class_indices=np.argmax(result,axis=1)
            #print(predicted_class_indices)
            if predicted_class_indices==0:
                output="Infected"
            else:
                output="Normal"
            st.subheader(output)




def cases():
    url = "https://prsindia.org/covid-19/cases"
    html_text=requests.get(url).text
    soup=BeautifulSoup(html_text,'html')
    t=soup.find_all('table')
    rows=t[0].find_all('tr')
    l1=['index','State/UT', 'Confirmed Cases', 'Active Cases', 'Cured/Discharged','Death']
    l2=[]
    l2.append(l1)
    for row in rows:
        columns = row.find_all('td')
        l1=[]
        for column in columns:
            #print(column.text)
            l1.append(column.text)
        #print("*"*50)
        l2.append(l1)

    data_till_date=pd.DataFrame(l2)
    dates=soup.find_all("div",class_="left-navigation")
    till_date_data=dates[1].text
    till_date_data=till_date_data.split("\n")[1]


    df=pd.DataFrame(l2)
    #df=df.drop(0)
    #df=df.columns =['index','State/UT', 'Confirmed Cases', 'Active Cases', 'Cured/Discharged','Death']
    #df=df.drop(['index'], axis = 1)
    new_header = df.iloc[0] #grab the first row for the header
    df = df[1:] #take the data less the header row
    df.columns = new_header #set the header row as the df header
    df = df[1:]
    df=df.drop(['index'], axis = 1)
    df = df.reset_index(drop=True)
    df=df.set_index('State/UT').T.to_dict('list')
    keysList = list(df.keys())

    st.write(till_date_data)
    states = st.selectbox('Select a State',keysList)
    with st.container():

        co1, co2, co3, co4 = st.columns(4)
        co1.metric("Confirmed Cases",value= df[states][0])
        co2.metric("Active Cases",value=df[states][1] )
        co3.metric("Cured/Discharged",value=df[states][2])
        co4.metric("Death", value=df[states][3])


    st.write('You selected:', states)
    #st.write('Confirmed Cases',df[states][0])
    #st.write('Active Cases	',df[states][1])
    #st.write('Cured/Discharged',df[states][2])
    #st.write('Death',df[states][3])
    return df
def table():
    url = "https://www.worldometers.info/coronavirus/country/india/"
    html_text = requests.get(url).text
    soup = BeautifulSoup(html_text, 'html')
    k = soup.find_all("button", class_="btn btn-light date-btn")
    p = soup.find_all("li", class_="news_li")
    news_date = soup.find('div', class_="news_date")
    news_date = news_date.text
    l1 = []
    l2 = []
    l1.append(news_date)

    for i in k:
        i = i.text.split()
        i = i[0] + " " + i[1]
        l1.append(i)
    for j in p:
        j = j.text.split("\xa0")[0]
        l2.append(j)

    data = pd.DataFrame(list(zip(l1, l2)),
                        columns=['Date', 'Total_cases'])

    st.table(data)





st.sidebar.info("Welcome to Tarun's Webapp")

st.markdown("<h1 style='text-align: center;'>TARUN'S INFECTOR  🩺</h1>", unsafe_allow_html=True)
st.subheader("About Infector 🤔")
st.text("predict whether you have infected or normal lungs")
st.text("Tested it over 14000 Xrays and getting accuracy of 96.5 percent ")

predictor()

with st.sidebar:

    st.write("🥶 Daily cases and deaths by date reported to WHO [link](https://covid19.who.int/WHO-COVID-19-global-data.csv)")
    st.write("🐻 Latest reported counts of cases and deaths [link](https://covid19.who.int/WHO-COVID-19-global-table-data.csv)")
    st.write("🐯 Vaccination data [link]( https://covid19.who.int/who-data/vaccination-data.csv)")
    st.write("🍁Vaccination metadata [link](https://covid19.who.int/who-data/vaccination-metadata.csv)")
    table()





cases()

