import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet import preprocess_input

# Taille des images attendues par le modèle
IMG_SIZE = (224, 224)

# Fonction de chargement des séquences
def load_all_sequences_for_backtest(base_dir="dataset_auto/sequences"):
    """
    Charge toutes les séquences d'images (5 images par séquence) depuis les sous-dossiers uptrend/downtrend.
    Retourne une liste de séquences sous forme de tableaux numpy.
    """
    all_sequences = []

    for trend_label in ["uptrend", "downtrend"]:
        trend_dir = os.path.join(base_dir, trend_label)
        for seq_folder in sorted(os.listdir(trend_dir)):
            seq_path = os.path.join(trend_dir, seq_folder)
            if os.path.isdir(seq_path):
                sequence = []
                for i in range(5):
                    img_path = os.path.join(seq_path, f"img_{i}.png")
                    if os.path.exists(img_path):
                        img = load_img(img_path, target_size=IMG_SIZE)
                        img = img_to_array(img)
                        img = preprocess_input(img)
                        sequence.append(img)
                if len(sequence) == 5:
                    all_sequences.append(np.array(sequence))
    return all_sequences

# Charger le modèle
model = load_model("end2end_model_20250530_203710.keras")

# Charger les séquences depuis les images générées
sequences = load_all_sequences_for_backtest()

# Prédire un signal (0 ou 1) pour chaque séquence
signals = []
for seq in sequences:
    seq_input = np.expand_dims(seq, axis=0)  # (1, 5, 224, 224, 3)
    prediction = model.predict(seq_input, verbose=0)
    signal = int(np.argmax(prediction))
    signals.append(signal)

# Afficher les signaux
print("Signaux prédits :", signals)
