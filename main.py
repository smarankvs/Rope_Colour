from sklearn.cluster import KMeans
from collections import Counter
from PIL import Image
import numpy as np

def classify_rope_color(image_path):
    img = Image.open(image_path).convert('RGB')
    img_resized = img.resize((200, 200))  

    pixels = np.array(img_resized).reshape(-1, 3)

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    kmeans.fit(pixels)

    counts = Counter(kmeans.labels_)
    dominant_cluster = counts.most_common(1)[0][0]
    dominant_color = kmeans.cluster_centers_[dominant_cluster].astype(int)

    color_refs = {'green': [120,170,120], 'yellow': [129,127,118], 'white': [135,136,135]}
    def color_distance(c1, c2):
        return np.linalg.norm(np.array(c1) - np.array(c2))

    overall_color = min(color_refs, key=lambda c: color_distance(dominant_color, color_refs[c]))
    print("Dominant color (RGB):", dominant_color)
    print("Predicted overall rope color:", overall_color)

classify_rope_color('D:\\Smaran_Required\\ASPL\\rope_color\\Rope_folders\\20250903150024_IMG_9363.jpg')
