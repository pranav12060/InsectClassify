import cv2
import numpy as np 
import os
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import joblib


import streamlit as st
import time
from PIL import Image
import numpy as np
from keras.preprocessing.image import load_img,img_to_array

kmeans, scale, svm, im_features =joblib.load("bovw.pkl")
def getDescriptors(sift, img):
    
    kp, des = sift.detectAndCompute(img, None)
    return des
def readImage(img_path):
    img = cv2.imread(img_path, 0)
    return cv2.resize(img,(150,150))
def vstackDescriptors(descriptor_list):
    descriptors = np.array(descriptor_list[0])
    for descriptor in descriptor_list[1:]:
        descriptors = np.vstack((descriptors, descriptor)) 
    return descriptors
def extractFeatures(kmeans, descriptor_list, image_count, no_clusters):
    im_features = np.array([np.zeros(no_clusters) for i in range(image_count)])
    for i in range(image_count):
        for j in range(len(descriptor_list[i])):
            feature = descriptor_list[i][j]
            feature = feature.reshape(1, 128)
            idx = kmeans.predict(feature)
            im_features[i][idx] += 1
    return im_features
def normalizeFeatures(scale, features):
    return scale.transform(features)
def testModel(path, kmeans, scale, svm, im_features, no_clusters, kernel):
    count = 0
    true = []
    descriptor_list = []

    name_dict =	{
        "0": "Aphid",
        "1": "caterpillar",
        "2":"corn flea beetle",
        "3":"Spider Mites",
        "4":"whitefly"
      
    }

    sift = cv2.SIFT_create()

    img_path=path
    img = readImage(img_path)
    des = getDescriptors(sift, img)
    if(des is not None):
        count += 1
        descriptor_list.append(des)

    descriptors = vstackDescriptors(descriptor_list)
    test_features = extractFeatures(kmeans, descriptor_list, count, no_clusters)
    test_features = scale.transform(test_features)    
    kernel_test = test_features
    if(kernel == "precomputed"):
        kernel_test = np.dot(test_features, im_features.T)
   
    predictions = [name_dict[str(int(i))] for i in svm.predict(kernel_test)]
    print("Given picture is of:")
    print(predictions)
    return predictions





def run():
    img1 = Image.open(r'C:\Users\Pranav Shinde\Downloads\BE project files\streamlut\butterfly.jpg')
    img1 = img1.resize((350,250))
    st.image(img1,use_column_width=False)
    st.title("Insect Classification")
    st.markdown('''<h4 style='text-align: middle; color: #8b70e5;font-family: Quando;font-size: 1em;text-transform:capitalize; '>Primates need good nutrition, to begin with. Not only fruits and plants, but insects as well</h4>''',unsafe_allow_html=True)

    img_file = st.file_uploader("Choose an Image of Insect", type=["jpg", "png"])
    if img_file is not None:
        st.write('uploaded image')
        st.image(img_file,use_column_width=False)
        save_image_path = './'+img_file.name
        with open(save_image_path, "wb") as f:
            f.write(img_file.getbuffer())

        if st.button("Predict"):
            result = testModel(save_image_path, kmeans, scale, svm, im_features, 10, "precomputed")
            st.success("Predicted Bird is: "+result[0])
run()


    







    

      






    
