import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import shutil

def load_images_from_folder(folder):
    images = []
    filenames = []
    for filename in os.listdir(folder):
        filepath = os.path.join(folder, filename)
        if os.path.isfile(filepath):
            img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images.append(img)
                filenames.append(filename)
    return images, filenames

def preprocess_images(images):
    flat_images = [img.flatten() for img in images]
    flat_images = np.array(flat_images)
    scaler = StandardScaler()
    flat_images = scaler.fit_transform(flat_images)
    return flat_images

def cluster_images(images, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    kmeans.fit(images)
    return kmeans.labels_

def save_clustered_images(folder, images, filenames, labels, num_clusters):
    for i in range(num_clusters):
        cluster_folder = os.path.join(folder, f'cluster_{i}')
        if not os.path.exists(cluster_folder):
            os.makedirs(cluster_folder)
            
    for img, filename, label in zip(images, filenames, labels):
        cluster_folder = os.path.join(folder, f'cluster_{label}')
        cv2.imwrite(os.path.join(cluster_folder, filename), img)

def main(input_folder, output_folder, num_clusters):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    images, filenames = load_images_from_folder(input_folder)
    flat_images = preprocess_images(images)
    labels = cluster_images(flat_images, num_clusters)
    save_clustered_images(output_folder, images, filenames, labels, num_clusters)

if __name__ == "__main__":
    input_folder = "patches"
    output_folder = "output_kmeans"
    num_clusters = 18  # You can set this to the number of clusters you want
    
    main(input_folder, output_folder, num_clusters)