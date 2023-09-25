import streamlit as st
import faiss
from PIL import Image
import io
import cv2
import numpy as np
from numpy import asarray
import matplotlib.pyplot as plt
from datasets import load_dataset
import cv2 
from scipy.cluster.vq import vq, whiten
from scipy.cluster.vq import kmeans, kmeans2
from numpy.linalg import norm

st.set_page_config(layout="wide")
st.title('Bag of visual words project')

col_left ,col_right, col_pictures = st.columns(3, gap='large')
col_left.header('Data For Image Retrieval')
col_right.header('Search parameters')
col_pictures.header('Similar images')

#ucitavanje slike 
uploaded_file = col_left.file_uploader("Choose a image for processing", accept_multiple_files=False)
def load_image():
    if uploaded_file is not None:
        # To read file as bytes:
        bytes_data = uploaded_file.getvalue()
        image = Image.open(io.BytesIO(bytes_data))
        new_size = (256,256)
        image = image.resize(new_size)
        numpydata = asarray(image)
        col_left.image(image)
        return numpydata
    else:
        return None
    
numpy_img = load_image()
uploaded_custom_dataset = col_left.file_uploader("Choose a custom dataset for processing", accept_multiple_files=True)
#biramo feature extractor 
feature_extractor = None
fe_option = 'SIFT'
if fe_option == 'SIFT':
    feature_extractor = cv2.SIFT_create()
#ucitavanje seta podataka nad kojim vrsimo pretrazivanje 
dataset_option = col_left.selectbox(
    '...or choose a dataset for search:',
    ('pets', 'food101'))

number_of_pictures = col_left.number_input('Choose a number of pictures from the dataset', min_value= 50,max_value=300, step= 20)
number = col_right.slider('Choose a number of clusters', min_value =50, max_value = 200, value=100)
number_of_images_to_ret = col_right.number_input('Insert a number of similar images to return', min_value= 1,max_value=10, step= 1)
loading_placeholder = col_right.empty()

def load_dataset_for_search():
    global dataset
    dataset = []
    if dataset_option == 'pets':
        with loading_placeholder:
            with st.spinner('Dataset is loading...'):
                dataset = load_dataset("pcuenq/oxford-pets", split = "train" , streaming = True)
    elif dataset_option == "food101":
        with loading_placeholder:
            with st.spinner('Dataset is loading...'):
                dataset = load_dataset("food101", split = "train", streaming="True")
    return dataset

def load_custom_dataset():
    global dataset
    dataset = []
    dataset.append(numpy_img)
    newsize = (256, 256)
    if uploaded_custom_dataset is not None:
        # To read file as bytes:
        with loading_placeholder:
            with st.spinner('Dataset is loading...'):
                for image in uploaded_custom_dataset:
                    bytes_data = image.getvalue()
                    image = Image.open(io.BytesIO(bytes_data))
                    image = image.resize(newsize)
                    numpydata = asarray(image)
                    dataset.append(numpydata)
        return dataset
    else:
        return None

def preprocess_images():
    #ucitavanje dataseta sa HuggingFace-a u streaming rezimu
    #odatle rad sa iterabilnim datasetom
    dataset = load_dataset_for_search()

    #predobrada slika za postupak ekstrakcije visual feature-a
    #my_iterable_dataset = dataset.shuffle(seed=42, buffer_size=100)
    my_iterable_dataset = dataset.take(number_of_pictures)
    numpy_array_all
    numpy_array_all = []
    #moramo da dodamo i sliku koju korisnik bira
    if numpy_img is not None:
        numpy_array_all.append(cv2.cvtColor(numpy_img, cv2.COLOR_BGR2GRAY))
    with loading_placeholder:
        with st.spinner('Converting images to black and white...'):
            for example in my_iterable_dataset:
                numpy_array_all.append(cv2.cvtColor(np.array(example['image']), cv2.COLOR_BGR2GRAY))
    return numpy_array_all

def preprocess_custom_images():
    numpy_dataset = load_custom_dataset()
    numpy_array_all = []
    #moramo da dodamo i sliku koju korisnik bira
    if numpy_dataset is not None:
        with loading_placeholder:
            with st.spinner('Converting images to black and white...'):
                for example in numpy_dataset:
                    numpy_array_all.append(cv2.cvtColor(np.array(example), cv2.COLOR_BGR2GRAY))
    return numpy_array_all

def calc_visual_features():
    numpy_array_all = preprocess_custom_images()
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
    print(descriptors)
    all_descriptors = []
    # prepakovanje podataka
    for img_descriptors in descriptors:
        for descriptor in img_descriptors:
            all_descriptors.append(descriptor)
    all_descriptors = np.stack(all_descriptors)
    all_descriptors = whiten(all_descriptors)
    with loading_placeholder:
        with st.spinner('Calculating KMeans...'):
            #codebook -> tu su nam centroidi
            codebook, variance = kmeans(all_descriptors, n_clusters,  n_iters)
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
    print(frequency_vectors)
    return frequency_vectors

def tfidf_func():
    N = number_of_pictures
    frequency_vectors = sparse_freq()
    with loading_placeholder:
        with st.spinner('Calculating TF-IDF...'):
            df = np.sum(frequency_vectors > 0, axis=0)
            idf = np.log(N/ df)
            tfidf = frequency_vectors * idf
    print(df)
    return tfidf

def search_for_similar_images(no_of_similar_images):
    tfidf = tfidf_func()
    a = tfidf[0] #slika sa kojom uporedjujemo
    b = tfidf
    cosine_similarity = np.dot(a, b.T)/(norm(a) * norm(b, axis=1))
    idx = np.argsort(-cosine_similarity)[:(no_of_similar_images+1)]
    # za prikaz rezultata
    for i in idx:
        cos_sim = round(cosine_similarity[i], 3)
        col_pictures.image(np.array(dataset[i]), caption = cos_sim)

if(col_right.button("Get similar images", type = 'primary')):
    search_for_similar_images(number_of_images_to_ret)


