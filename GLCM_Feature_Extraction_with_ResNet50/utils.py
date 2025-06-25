# utils.py
import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from tensorflow.keras.applications.resnet50 import preprocess_input

def extract_glcm(img):
    img_uint8 = (img * 255).astype('uint8')
    glcm = graycomatrix(img_uint8, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    return np.array([
        graycoprops(glcm, 'contrast')[0, 0],
        graycoprops(glcm, 'dissimilarity')[0, 0],
        graycoprops(glcm, 'homogeneity')[0, 0],
        graycoprops(glcm, 'energy')[0, 0],
        graycoprops(glcm, 'correlation')[0, 0],
        graycoprops(glcm, 'ASM')[0, 0]
    ])

def prepare_image(image, resnet_model):
    img = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    img_resized = cv2.resize(img, (224, 224)) / 255.0

    glcm_feat = extract_glcm(img_resized).reshape(1, -1)

    img_rgb = cv2.cvtColor((img_resized * 255).astype('uint8'), cv2.COLOR_GRAY2RGB)
    img_rgb = preprocess_input(np.expand_dims(img_rgb, axis=0))
    resnet_feat = resnet_model.predict(img_rgb)

    return np.concatenate([resnet_feat, glcm_feat], axis=1), img_resized
