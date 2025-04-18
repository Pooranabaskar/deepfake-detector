# @author Shruti and Shambhavi

#adding the capabilities of image URL as an input
# necessary packages are being imported in the following lines
import keras
import numpy as np
from PIL import Image
import streamlit as st
import requests
from skimage import transform
from io import BytesIO
from keras import backend as K 
import logging
import tensorflow as tf

tf.get_logger().setLevel(logging.ERROR)

#favicon = './icon.png'
#st.set_page_config(page_title='DeepFake Detector', page_icon = favicon, initial_sidebar_state = 'auto')
# favicon being an object of the same kind as the one you should provide st.image() with (ie. a PIL array for example) or a string (url or local file path)

hide_footer_style = """
    <style>
    .reportview-container .main footer {visibility: hidden;}    
    """
st.markdown(hide_footer_style, unsafe_allow_html=True)





#introduction of the app
st.write("""
## DeepFake Image Detector
""")

#using this count as a failsafe if the user does not upload the file and still clicks the "process image" button
count = 0

#upload button for the input image
uploaded_file = st.file_uploader("Choose an image and make sure it's in the right orientation for the best outputs.", type=['jpg', 'png', 'jpeg'])

if uploaded_file != None:

    count = count + 1

#st.write(uploaded_file)
#st.write(count)

#st.write(uploaded_file)

url = ""
url = st.text_input("Or paste the image URL here", 'https://thispersondoesnotexist.com/image')

if url != "https://thispersondoesnotexist.com/image":
    count = count + 1
else:
    count = count + 2
#
# st.write('final count')
#st.write(count)

if uploaded_file is not None:

    if count == 3:

        #getting the file extension of the uploaded file
        file_name = uploaded_file.name
        extension = file_name.split(".")[1]

        if extension == "png" or  extension == "PNG":
            uploaded_image = Image.open(uploaded_file)
            st.image(uploaded_image, caption='Uploaded Image', use_column_width=True)

        elif extension == "jpeg" or extension == "JPEG":
            uploaded_image = Image.open(uploaded_file)
            st.image(uploaded_image, caption='Uploaded Image', use_column_width=True)

        elif extension == "jpg" or extension == "JPG":
            # SHOW IMAGE IF JPG FORMAT
            uploaded_image = Image.open(uploaded_file)
            st.image(uploaded_image, caption='Uploaded Image', use_column_width=True)


        else:
            st.write("Please upload .JPG , .JPEG OR .PNG format file only!")

    elif count == 1:
        pass
        response = requests.get(url)
        uploaded_image = Image.open(BytesIO(response.content))
        st.image(uploaded_image, caption='Uploaded Image', use_column_width=True)

    elif count == 2:
        response = requests.get('https://thispersondoesnotexist.com/image')
        uploaded_image = Image.open(BytesIO(response.content))
        st.image(uploaded_image, caption='Uploaded Image', use_column_width=True)

    else:
        pass





if st.button('Check authenticity'):

    #checking if user uploaded any file
    if count == 1 :
        #st.write("Please choose a file!")
        response = requests.get(url)
        uploaded_image = Image.open(BytesIO(response.content))
        st.image(uploaded_image, caption='Uploaded Image', use_column_width=True)

        st.write("Processing...")

        # preprocessing the image to use in the trained model
        np_image = uploaded_image
        np_image = np.array(np_image).astype('float32') / 255
        np_image = transform.resize(np_image, (224, 224, 3))
        np_image = np.expand_dims(np_image, axis=0)

#the h5 model is being loaded here.
        model = keras.models.load_model("model.h5")
        probab = model.predict(np_image)[0][0]
        st.write("The probability of this image being real is: ")
        st.write(probab)

        if probab < 0.05:
            st.write("which means this image is most likely fake, do not trust everything you see on the internet.")
            st.image('fake_drake.png', caption="Drake doesn't approve", use_column_width=True)

        if probab > 0.9:
            st.write(
                "which means this image is most likely real, still do not trust everything you see on the internet.")
            st.image('real_drake.jpg', caption="Drake approves", use_column_width=True)

    elif count == 2 :
        #st.write("Please choose a file!")
        response = requests.get('https://thispersondoesnotexist.com/image')
        uploaded_image = Image.open(BytesIO(response.content))
        st.image(uploaded_image, caption='Uploaded Image', use_column_width=True)

        st.write("Processing...")

        # preprocessing the image to use in the trained model
        np_image = uploaded_image
        np_image = np.array(np_image).astype('float32') / 255
        np_image = transform.resize(np_image, (224, 224, 3))
        np_image = np.expand_dims(np_image, axis=0)


        model = keras.models.load_model("model.h5")
        probab = model.predict(np_image)[0][0]
        st.write("The probability of this image being real is: ")
        #probab = 0.9369558
        st.write(probab)

        if probab < 0.05:
            st.write("which means this image is most likely fake, do not trust everything you see on the internet.")
            st.image('fake_drake.png', caption="Drake doesn't approve", use_column_width=True)

        if probab > 0.9:
            st.write(
                "which means this image is most likely real, still do not trust everything you see on the internet.")
            st.image('real_drake.jpg', caption="Drake approves", use_column_width=True)

    elif count == 3:

        if extension == 'jpg' or extension == 'JPG':

            st.write("Processing...")

            #preprocessing the image to use in the trained model
            np_image = uploaded_image
            np_image = np.array(np_image).astype('float32') / 255
            np_image = transform.resize(np_image, (224, 224, 3))
            np_image = np.expand_dims(np_image, axis=0)


            model = keras.models.load_model("model.h5")
            probab = model.predict(np_image)[0][0]
            st.write("The probability of this image being real is: ")
            st.write(probab)


            if probab < 0.05 :
                st.write("which means this image is most likely fake, do not trust everything you see on the internet.")
                st.image('fake_drake.png', caption="Drake doesn't approve", use_column_width=True)

            if probab > 0.9 :
                st.write("which means this image is most likely real, still do not trust everything you see on the internet.")
                st.image('real_drake.jpg', caption="Drake approves", use_column_width=True)


        if extension == 'jpeg' or extension == 'JPEG':

            st.write("Processing...")

            #preprocessing the image to use in the trained model
            np_image = uploaded_image
            np_image = np.array(np_image).astype('float32') / 255
            np_image = transform.resize(np_image, (224, 224, 3))
            np_image = np.expand_dims(np_image, axis=0)

            model = keras.models.load_model("model.h5")
            probab = model.predict(np_image)[0][0]
            st.write("The probability of this image being real is: ")
            st.write(probab)


            if probab < 0.05 :
                st.write("which means this image is most likely fake, do not trust everything you see on the internet.")
                st.image('fake_drake.png', caption="Drake doesn't approve", use_column_width=True)

            if probab > 0.9 :
                st.write("which means this image is most likely real, still do not trust everything you see on the internet.")
                st.image('real_drake.jpg', caption="Drake approves", use_column_width=True)



        elif extension == 'png' or extension == 'PNG':
            st.write("Processing...")

            np_image = uploaded_image
            np_image = np.array(np_image).astype('float32') / 255
            np_image = transform.resize(np_image, (224, 224, 3))
            np_image = np.expand_dims(np_image, axis=0)

            #@st.cache
            model = keras.models.load_model("model.h5")
            probab = model.predict(np_image)[0][0]
            st.write("The probability of this image being real is: ")
            st.write(probab)

            if probab < 0.05:
                st.write("which means this image is most likely fake, do not trust everything you see on the internet.")
                st.image('fake_drake.png', caption="Drake doesn't approve", use_column_width=True)

            if probab > 0.9 :
                st.write("which means this image is most likely real, still do not trust everything you see on the internet.")
                st.image('real_drake.jpg', caption="Drake approves", use_column_width=True)

    else:
        st.write('Please refresh the page!')
        
    del model



st.write('\n')
st.write('\n')
st.write('\n')



K.clear_session()

