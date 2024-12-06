from flask import Flask, request, render_template, jsonify, send_from_directory
from PIL import Image
import pandas as pd
import torch
import torch.nn.functional as F
from open_clip import create_model_and_transforms, get_tokenizer
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances
import os
import numpy as np

app = Flask(__name__)

# Path to the directory containing images
IMAGE_FOLDER = 'coco_images_resized'

# Device setup
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Model configuration
MODEL_NAME = "ViT-B-32"
PRETRAINED = "openai"

# Load the CLIP model and utilities
model, preprocess_train, preprocess_val = create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED)
tokenizer = get_tokenizer(MODEL_NAME)
model = model.to(DEVICE).eval()

# Load image embeddings
EMBEDDINGS_PATH = 'image_embeddings.pickle'
with open(EMBEDDINGS_PATH, 'rb') as f:
    embeddings_df = pd.read_pickle(f)

# Load images for PCA
def load_images(image_dir, max_images=None, target_size=(224, 224)):
    images = []
    image_names = []
    for i, filename in enumerate(os.listdir(image_dir)):
        if filename.endswith('.jpg'):
            img = Image.open(os.path.join(image_dir, filename))
            img = img.convert('L')  # Convert to grayscale ('L' mode)
            img = img.resize(target_size)  # Resize to target size
            img_array = np.asarray(img, dtype=np.float32) / 255.0  # Normalize pixel values to [0, 1]
            images.append(img_array.flatten())  # Flatten to 1D
            image_names.append(filename)
        if max_images and i + 1 >= max_images:
            break
    return np.array(images), image_names

# Train PCA
train_images, train_image_names = load_images(IMAGE_FOLDER, max_images=2000)
pca = PCA()  # PCA initialized without a fixed number of components
pca.fit(train_images)
transform_images, transform_image_names = load_images(IMAGE_FOLDER, max_images=10000)
pca_embeddings = pca.transform(transform_images)

# Utility function: Compute Cosine Similarity
def compute_similarity(embedding1, embedding2):
    return F.cosine_similarity(embedding1, embedding2.unsqueeze(0)).item()

# Utility function: PCA Similarity (Euclidean distance)
def compute_pca_similarity(query_embedding, embeddings):
    distances = euclidean_distances([query_embedding], embeddings).flatten()
    nearest_indices = np.argsort(distances)
    return nearest_indices, distances[nearest_indices]

# Search function: Retrieve top k matches by similarity
def search_query(query_embedding, k=5, use_pca=False, pca_k=None):
    if use_pca:
        # Reduce PCA embeddings to the first k components
        reduced_pca_embeddings = pca_embeddings[:, :pca_k]
        query_embedding = query_embedding[:, :pca_k]  # Reduce query embedding
        indices, distances = compute_pca_similarity(query_embedding, reduced_pca_embeddings)
        results = [{"file_name": transform_image_names[idx], "similarity": 1 / (1 + distances[idx])} for idx in indices[:k]]
    else:
        results = []
        for _, row in embeddings_df.iterrows():
            similarity = compute_similarity(query_embedding, torch.tensor(row['embedding']).to(DEVICE))
            results.append({"file_name": row['file_name'], "similarity": similarity})
        results = sorted(results, key=lambda x: x["similarity"], reverse=True)
    return results[:k]

# Route to serve image files
@app.route('/media/<path:filename>')
def serve_image(filename):
    return send_from_directory(IMAGE_FOLDER, filename)

# Main route for rendering the search page
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle search requests
@app.route('/search', methods=['POST'])
def handle_search():
    query_type = request.form.get('query_type')
    use_pca = request.form.get('use_pca') == 'true'
    pca_k = int(request.form.get('pca_k', 50))  # Default to first 50 principal components
    text_query = request.form.get('text_query', '').strip()
    image_file = request.files.get('image_query')
    hybrid_weight = float(request.form.get('hybrid_weight', 0.5))

    query_embedding = None

    # Handle text queries
    if query_type == 'text' and text_query:
        tokens = tokenizer([text_query])
        query_embedding = F.normalize(model.encode_text(tokens.to(DEVICE)))

    # Handle image queries
    elif query_type == 'image' and image_file:
        image = Image.open(image_file).convert('RGB')
        processed_image = preprocess_val(image).unsqueeze(0).to(DEVICE)
        query_embedding = F.normalize(model.encode_image(processed_image))

    # Handle hybrid queries
    elif query_type == 'hybrid' and text_query and image_file:
        tokens = tokenizer([text_query])
        text_embedding = F.normalize(model.encode_text(tokens.to(DEVICE)))

        image = Image.open(image_file).convert('RGB')
        processed_image = preprocess_val(image).unsqueeze(0).to(DEVICE)
        image_embedding = F.normalize(model.encode_image(processed_image))

        query_embedding = F.normalize(
            hybrid_weight * text_embedding + (1 - hybrid_weight) * image_embedding
        )

    if query_embedding is not None:
        if use_pca:
            query_embedding = pca.transform(query_embedding.cpu().detach().numpy())  # Transform to PCA space
        matches = search_query(query_embedding, use_pca=use_pca, pca_k=pca_k)
        return jsonify(results=matches)

    return jsonify(results=[])

if __name__ == '__main__':
    app.run(debug=True)
