from sklearn.cluster import KMeans
from collections import Counter
from PIL import Image
import numpy as np

def get_dominant_rgb(image_path):
    img = Image.open(image_path).convert('RGB')
    img_resized = img.resize((200, 200))
    pixels = np.array(img_resized).reshape(-1, 3)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    kmeans.fit(pixels)
    counts = Counter(kmeans.labels_)
    dominant_cluster = counts.most_common(1)[0][0]
    dominant_color = kmeans.cluster_centers_[dominant_cluster].astype(int)
    return dominant_color.tolist()

def classify_rope_color(image_path, green_ref):
    img = Image.open(image_path).convert('RGB')
    img_resized = img.resize((200, 200))

    # Extract pixel RGB values as 2D array (num_pixels x 3)
    pixels = np.array(img_resized).reshape(-1, 3)

    # Apply KMeans clustering to identify 3 dominant colors
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    kmeans.fit(pixels)

    # Count the number of pixels in each cluster -> find dominant cluster
    counts = Counter(kmeans.labels_)
    dominant_cluster = counts.most_common(1)[0][0]
    dominant_color = kmeans.cluster_centers_[dominant_cluster].astype(int)

    color_refs = {
        'green': green_ref,
        'yellow': [176, 176, 180],  
        'white': [135, 136, 135]     
    }

    # Euclidean distance between two RGB colors
    def color_distance(c1, c2):
        return np.linalg.norm(np.array(c1) - np.array(c2))

    
    overall_color = min(color_refs, key=lambda c: color_distance(dominant_color, color_refs[c]))

    print("Dominant color (RGB):", dominant_color)
    print("Predicted overall rope color:", overall_color)



green_reference = get_dominant_rgb(r"D:/Smaran_Required/ASPL/Rope_folders/G2.jpg")  # Path to a green rope image


classify_rope_color(r"D:/Smaran_Required/ASPL/Rope_folders/W2.jpg", green_reference)  # Replace test_image.jpg with your input

