import streamlit as st
from PIL import Image
import io
import os
import cv2
import numpy as np
from numpy import asarray
import matplotlib.pyplot as plt
from datasets import load_dataset, Image
import cv2 
from scipy.cluster.vq import vq
from scipy.cluster.vq import kmeans
from numpy.linalg import norm

st.set_page_config(layout="wide")
st.title('Bag of visual words project')

col_left, col_right = st.columns(2, gap='large')
col_left.header('Feature extraction')
col_right.header('Similar image retrieval')

#ucitavanje slike 
uploaded_file = col_left.file_uploader("Choose a image for processing")
def load_image():
    if uploaded_file is not None:
        # To read file as bytes:
        bytes_data = uploaded_file.getvalue()
        image = Image.open(io.BytesIO(bytes_data))
        numpydata = asarray(image)
        col_left.image(image)
        return numpydata
    else:
        return None
numpy_img = load_image()

#biramo feature extractor 
feature_extractor = None
fe_option = col_left.selectbox(
    'Choose a feature selector:',
    ('SIFT', 'SURF'))
if fe_option == 'SIFT':
    feature_extractor = cv2.SIFT_create()
elif fe_option == 'SURF':
    feature_extractor = cv2.FastFeatureDetector_create()

def feature_extract(feature_ext):
    if (numpy_img is not None):
        numpy_bw = cv2.cvtColor(numpy_img, cv2.COLOR_BGR2GRAY)
        keypoints_1 = feature_ext.detect(numpy_bw,None)
        img= cv2.drawKeypoints(numpy_bw,keypoints_1,numpy_img)
        col_left.image(img)

if(col_left.button("Feature extraction", type = 'primary')):
    feature_extract(feature_extractor)

#ucitavanje seta podataka nad kojim vrsimo pretrazivanje 
dataset_option = col_right.selectbox(
    'Choose a dataset for search:',
    ('imagenet-1k', 'food101'))

number_of_pictures = col_right.number_input('Insert a number of pictures from the dataset', min_value= 50,max_value=200, step= 10)
number = col_right.number_input('Insert a number of clusters', min_value= 100,max_value=200, step= 10)
number_of_images_to_ret = col_right.number_input('Insert a number of similar images to return', min_value= 1,max_value=5, step= 1)
loading_placeholder = col_right.empty()

def load_dataset_for_search():
    if dataset_option == 'imagenet-1k':
        with loading_placeholder:
            with st.spinner('Dataset is loading...'):
                dataset = load_dataset("imagenet-1k", split = "train", streaming = True)
    else:
        with loading_placeholder:
            with st.spinner('Dataset is loading...'):
                dataset = load_dataset("food101", split = "train", streaming="True")
    return dataset

def preprocess_images():
    #ucitavanje dataseta sa HuggingFace-a u streaming rezimu
    #odatle rad sa iterabilnim datasetom
    dataset = load_dataset_for_search()

    #predobrada slika za postupak ekstrakcije visual feature-a
    my_iterable_dataset = dataset.shuffle(seed=42, buffer_size=100)
    my_iterable_dataset = my_iterable_dataset.take(number_of_pictures)
    numpy_array_all = []
    with loading_placeholder:
        with st.spinner('Converting images to black and white...'):
            for example in my_iterable_dataset:
                numpy_array_all.append(cv2.cvtColor(np.array(example['image']), cv2.COLOR_BGR2GRAY))
    return preprocess_images()

def calc_visual_features():
    numpy_array_all = preprocess_images()
    #trazenje visual feature-a
    keypoints = []
    descriptors = []
    with loading_placeholder:
        with st.spinner('Detecting keypoints and descriptors...'):
            for img in numpy_array_all:
                img_keypoints, img_descriptors = feature_extractor.detectAndCompute(img, None)
                keypoints.append(img_keypoints)
                descriptors.append(img_descriptors)
    return keypoints, descriptors

def kmeans_func(n_clusters, n_iters):
    keypoints, descriptors = calc_visual_features()
    all_descriptors = []
    # prepakovanje podataka
    for img_descriptors in descriptors:
        for descriptor in img_descriptors:
            all_descriptors.append(descriptor)
    all_descriptors = np.stack(all_descriptors)
    with loading_placeholder:
        with st.spinner('Calculating KMeans...'):
            codebook, variance = kmeans(all_descriptors, n_clusters, n_iters)
    visual_words = []
    for img_descriptors in descriptors:
        img_visual_words, distance = vq(img_descriptors, codebook)
        visual_words.append(img_visual_words)
    return visual_words

def sparse_freq():
    visual_words = kmeans_func(number,2)
    frequency_vectors = []
    with loading_placeholder:
        with st.spinner('Calculating sparse frequency vectors...'):
            for img_visual_words in visual_words:
                img_frequency_vector = np.zeros(number)
                for word in img_visual_words:
                    img_frequency_vector[word] += 1
                frequency_vectors.append(img_frequency_vector)
            frequency_vectors = np.stack(frequency_vectors)
    return frequency_vectors

def tfidf_func():
    N = number_of_pictures
    frequency_vectors = sparse_freq()
    with loading_placeholder:
        with st.spinner('Calculating TF-IDF...'):
            df = np.sum(frequency_vectors > 0, axis=0)
            idf = np.log(N/ df)
            tfidf = frequency_vectors * idf
    return tfidf

def search_for_similar_images(no_of_similar_images):
    tfidf = tfidf_func()
    a = tfidf[20] #treba mi slika sa kojom uporedjujem
    b = tfidf
    cosine_similarity = np.dot(a, b.T)/(norm(a) * norm(b, axis=1))
    idx = np.argsort(-cosine_similarity)[:no_of_similar_images]
    # za prikaz rezultata
    for i in idx:
        print(f"{i}: {round(cosine_similarity[i], 4)}")
        col_right.image(image, caption='Sunrise by the mountains')
        #plt.imshow(bw_images[i], cmap='gray')
        #plt.show()

if(col_right.button("Get similar images", type = 'primary')):
    search_for_similar_images(number_of_images_to_ret)


