import tensorflow as tf
model = tf.keras.models.load_model('2resnet50.hdf5')
import tempfile
import cv2
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
st.write("""
         # Rock-Paper-Scissor Hand Sign Prediction
         """
         )
st.write("This is a simple image classification web app to predict rock-paper-scissor hand sign")
uploaded_file = st.file_uploader("Please upload an image file", type=["jpg", "png"])


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
            st.balloons()
            result = model.predict(resized_arr)
            predicted_class_indices=np.argmax(result,axis=1)

            #print(predicted_class_indices)
            if predicted_class_indices==0:
                output="Infected"
            else:
                output="Normal"
            st.subheader(output)