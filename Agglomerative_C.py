import os
import cv2
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, fcluster
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

def find_optimal_clusters(images, max_clusters=20):
    Z = linkage(images, method='ward')
    best_num_clusters = 2
    best_silhouette_score = -1
    
    for num_clusters in range(2, max_clusters + 1):
        labels = fcluster(Z, num_clusters, criterion='maxclust')
        silhouette_avg = silhouette_score(images, labels)
        if silhouette_avg > best_silhouette_score:
            best_silhouette_score = silhouette_avg
            best_num_clusters = num_clusters
    
    return best_num_clusters

def cluster_images(images, num_clusters):
    Z = linkage(images, method='ward')
    labels = fcluster(Z, num_clusters, criterion='maxclust')
    return labels

def save_clustered_images(folder, images, filenames, labels, num_clusters):
    for i in range(1, num_clusters + 1):
        cluster_folder = os.path.join(folder, f'cluster_{i}')
        if not os.path.exists(cluster_folder):
            os.makedirs(cluster_folder)
            
    for img, filename, label in zip(images, filenames, labels):
        cluster_folder = os.path.join(folder, f'cluster_{label}')
        cv2.imwrite(os.path.join(cluster_folder, filename), img)

def main(input_folder, output_folder, max_clusters=10):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    images, filenames = load_images_from_folder(input_folder)
    flat_images = preprocess_images(images)
    
    optimal_clusters = find_optimal_clusters(flat_images, max_clusters)
    labels = cluster_images(flat_images, optimal_clusters)
    
    save_clustered_images(output_folder, images, filenames, labels, optimal_clusters)
    print(f"Optimal number of clusters: {optimal_clusters}")

if __name__ == "__main__":
    input_folder = "patches"
    output_folder = "output_Agg"
    
    main(input_folder, output_folder, max_clusters=20)