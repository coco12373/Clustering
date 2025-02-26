import os
import cv2
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, fcluster
import shutil
from sklearn.decomposition import PCA


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

def detect_corners(image):
    """Detect corners in the image using the Shi-Tomasi corner detector."""
    corners = cv2.goodFeaturesToTrack(image, maxCorners=64, qualityLevel=0.01, minDistance=10)
    if corners is not None:
        corners = np.int0(corners)
        # corners = corners.reshape(-1, 2)
    return corners

def extract_corner_features(images, max_corners=32):
    features = []
    for img in images:
        corners = detect_corners(img)
        if corners is None:
            corners = np.zeros((0, 2))
        
        center = (img.shape[1] // 2, img.shape[0] // 2)  # 图片中心坐标
        
        relative_coords = corners - center  # 计算相对坐标
        
        if len(corners) > max_corners:
            relative_coords = relative_coords[:max_corners]  # 限制最大角点数量
        
        corner_coords = relative_coords.flatten()  # 展平为一维数组

        white_pixel_count = np.sum(img)  # 计算二值图像中白色像素点的数量
        
        # Ensure the feature vector has a consistent length
        # feature = np.hstack(([len(corners)], (0.00001*white_pixel_count).astype(int), corner_coords,))
        feature = corner_coords
        if len(feature) < (2 * max_corners + 2):
            feature = np.hstack((feature, np.zeros((2 * max_corners + 2) - len(feature))))
        
        features.append(feature)
    
    features = np.array(features)
    # Standardize the features
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    return features

def find_optimal_clusters(features, max_clusters=20):
    Z = linkage(features, method='ward')
    best_num_clusters = 2
    best_silhouette_score = -1
    
    for num_clusters in range(2, max_clusters + 1):
        labels = fcluster(Z, num_clusters, criterion='maxclust')
        silhouette_avg = silhouette_score(features, labels)
        if silhouette_avg > best_silhouette_score:
            best_silhouette_score = silhouette_avg
            best_num_clusters = num_clusters
    
    return best_num_clusters

def cluster_images(features, num_clusters):
    Z = linkage(features, method='ward')
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

def main(input_folder, output_folder, max_clusters=20, max_corners=64):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    images, filenames = load_images_from_folder(input_folder)
    features = extract_corner_features(images, max_corners)
    
    optimal_clusters = find_optimal_clusters(features, max_clusters)
    labels = cluster_images(features, optimal_clusters)
    
    save_clustered_images(output_folder, images, filenames, labels, optimal_clusters)
    print(f"Optimal number of clusters: {optimal_clusters}")

if __name__ == "__main__":
    input_folder = "random_10000"
    output_folder = "output_agg_new_10000"
    
    main(input_folder, output_folder, max_clusters=20, max_corners=64)
    # image_path = "F:/code/keypoint/patid_MX_Benchmark2_clip_nonhotspot1_6_orig_0_patches_patch_665.png"
    # img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # print(img.shape)
    # corners = detect_corners(img)
    # center = (100,100)  # 图片中心坐标
    # relative_coords = corners - center  # 计算相对坐标
    # corner_coords = relative_coords.flatten()  # 展平为一维数组
    # white_pixel_count = np.sum(img)  # 计算二值图像中白色像素点的数量
    # pca = PCA(n_components=2)
    # pca_features = pca.fit_transform(relative_coords)
    # pca_features = pca_features.flatten()
    # feature = np.hstack(([len(corners)], pca_features, (0.00001*white_pixel_count).astype(int)))
    
    # print(len(corners))
    # print(corners)
    # print(feature)



    # def extract_corner_features(images, max_corners=64):
#     features = []
#     for img in images:
#         corners = detect_corners(img)
#         if corners is None:
#             corners = np.zeros((0, 2))
#         corners = corners[:max_corners]  # Limit to max_corners
#         corner_coords = corners.flatten()
        
#         # Ensure the feature vector has a consistent length
#         feature = np.hstack(([len(corners)], corner_coords))
#         if len(feature) < (2 * max_corners + 1):
#             feature = np.hstack((feature, np.zeros((2 * max_corners + 1) - len(feature))))
        
#         features.append(feature)
    
#     features = np.array(features)
#     # Standardize the features
#     scaler = StandardScaler()
#     features = scaler.fit_transform(features)
#     return features