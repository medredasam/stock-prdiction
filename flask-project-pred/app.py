import os
import numpy as np
import cv2
from flask import Flask, render_template, request, send_from_directory, url_for
import tensorflow as tf
import matplotlib.pyplot as plt

# Configuration Flask
app = Flask(__name__)
UPLOAD_FOLDER = os.path.join('static', 'uploads')
VISUALIZATION_FOLDER = os.path.join('static', 'visualizations')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(VISUALIZATION_FOLDER, exist_ok=True)

# Modèle
MODEL_PATH = 'trading_trend_model.h5'
IMG_SIZE = (64, 64)
model = tf.keras.models.load_model(MODEL_PATH)

# --- Fonctions utiles ---

def preprocess_image(image_path):
    """Prétraitement de l'image pour la prédiction"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    day_width = img.shape[1] // 10
    sequence = []

    for i in range(10):
        day_img = img[:, i * day_width:(i + 1) * day_width]
        day_img = cv2.resize(day_img, IMG_SIZE)
        sequence.append(day_img)

    return np.array(sequence).reshape(1, 10, *IMG_SIZE, 1) / 255.0

def predict_trend(image_path):
    """Effectue la prédiction de la tendance"""
    processed_img = preprocess_image(image_path)
    if processed_img is None:
        return None

    pred = model.predict(processed_img)[0][0]
    trend = "Uptrend" if pred > 0.5 else "Downtrend"
    confidence = max(pred, 1 - pred)

    # Simulation des scores journaliers (à remplacer par des scores réels si disponibles)
    day_scores = np.random.rand(10).tolist()

    return {
        'trend': trend,
        'confidence': f"{confidence:.2%}",
        'raw_score': float(pred),
        'day_scores': day_scores
    }

def visualize_single(img_path, prediction, true_label="Unknown", output_path='static/visualizations/graph.png'):
    """Génère une visualisation et la sauvegarde"""
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(15, 8))

    # Image originale
    plt.subplot(2, 1, 1)
    plt.imshow(img)
    plt.title(f"{os.path.basename(img_path)}\nVrai: {true_label} | Prédit: {prediction['trend']} ({float(prediction['confidence'].strip('%')) / 100:.2%})")
    plt.axis('off')

    # Courbe des scores
    plt.subplot(2, 1, 2)
    plt.plot(range(1, 11), prediction['day_scores'], 'b-o')
    plt.axhline(0.5, color='r', linestyle='--')
    plt.xlabel('Jour')
    plt.ylabel('Score Uptrend')
    plt.xticks(range(1, 11))
    plt.ylim(0, 1)
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

# --- Routes Flask ---

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload')
def upload():
    return render_template('upload.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('upload.html', error='Aucun fichier sélectionné.')

    image_file = request.files['file']
    if image_file.filename == '':
        return render_template('upload.html', error='Aucune image sélectionnée.')

    try:
        image_path = os.path.join(UPLOAD_FOLDER, image_file.filename)
        image_file.save(image_path)

        result = predict_trend(image_path)
        if result is None:
            return render_template('upload.html', error='Erreur lors du traitement de l\'image.')

        # Créer la visualisation
        graph_filename = f"graph_{os.path.splitext(image_file.filename)[0]}.png"
        graph_path = os.path.join(VISUALIZATION_FOLDER, graph_filename)
        visualize_single(image_path, result, true_label="Unknown", output_path=graph_path)

        return render_template('result.html',
                               image_path=os.path.join('uploads', image_file.filename),
                               graph_path=os.path.join('visualizations', graph_filename),
                               result=result)

    except Exception as e:
        return render_template('upload.html', error=f"Erreur : {str(e)}")

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/visualizations/<filename>')
def visualization_file(filename):
    return send_from_directory(VISUALIZATION_FOLDER, filename)

# Lancer l'application
if __name__ == '__main__':
    app.run(debug=True,use_reloader=True)
