from numpy.testing._private.utils import suppress_warnings
import streamlit as st

from PIL import Image
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing import image as SImage
from keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model


DEMO_IMAGE = 'happy.jpg'

classifier = load_model('EmotionDetectionModel.h5')
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
class_labels=['Angry','Happy','Neutral','Sad','Surprise']


@st.cache
def detect_emotion(image):
    #resize the frame to process it quickly
    frame = image
   
    
    labels=[]
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=face_classifier.detectMultiScale(gray,1.3,5)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray=gray[y:y+h,x:x+w]
        roi_gray=cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray])!=0:
            roi=roi_gray.astype('float')/255.0
            roi=img_to_array(roi)
            roi=np.expand_dims(roi,axis=0)

            preds=classifier.predict(roi)[0]
            label=class_labels[preds.argmax()]
            
            label_position=(x,y)
            cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,255),3)
        else:
            cv2.putText(frame,'No Face Found',(20,20),cv2.FONT_HERSHEY_SIMPLEX,2,(255,255,0),2)
    
    
    
    return frame
    
                             




st.title('Emotion Detection Application')

st.markdown('''
            * This model helps you detect emotions in an Image\n
            * So give an image with where the face is not blurry or disoriented 
            ''')

img_file_buffer = st.file_uploader("Upload an image", type=[ "jpg", "jpeg",'png'])

if img_file_buffer is not None:
    image = np.array(Image.open(img_file_buffer))

else:
    demo_image = DEMO_IMAGE
    image = np.array(Image.open(demo_image))
    



st.subheader('Original Image')

st.image(image, caption=f"Original Image",use_column_width= True)

emotion_analysis = detect_emotion(image)
st.markdown('''
            This model detects only the following emotions 
            **Angry,Happy,Neutral,Sad and Surprise**
            ''')


st.subheader('Emotion Analysis')

st.image(emotion_analysis, caption=f"detected Image",use_column_width= True)

st.markdown('''
          # About Author \n 
             Hey this is ** Pavan Kunchala ** I hope you like the application \n
             
            I am looking for ** Collabration **,** Freelancing ** and  ** Job opportunities ** in the field of ** Deep Learning ** and 
            ** Computer Vision **  if you are interested in my profile you can check out my resume from 
            [here](https://drive.google.com/file/d/16aKmdHryldvx3OPNwmHhxW-DAoQOypvX/view?usp=sharing)\n
            
            If you're interested in collabrating you can mail me at ** pavankunchalapk@gmail.com ** \n
            You can check out my ** Linkedin ** Profile from [here](https://www.linkedin.com/in/pavan-kumar-reddy-kunchala/) \n
            You can check out my ** Github ** Profile from [here](https://github.com/Pavankunchala) \n
            You can also check my technicals blogs in ** Medium ** from [here](https://pavankunchalapk.medium.com/) \n
            If you are feeling generous you can buy me a cup of ** coffee ** from [here](https://www.buymeacoffee.com/pavankunchala)
             
            ''')






