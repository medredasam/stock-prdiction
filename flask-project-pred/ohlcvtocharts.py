import os
import pandas as pd
import mplfinance as mpf

# === Étape 1 : Chargement du fichier CSV ===
csv_file = 'tesla_ohlcv_365jours.csv'  # Assure-toi que ce fichier est dans ton dossier
df = pd.read_csv(csv_file, parse_dates=['datetime'], index_col='datetime')

# === Étape 2 : Paramètres ===
window_size = 20       # Nombre de jours pour chaque image de chandelier
sequence_len = 5       # Nombre d'images par séquence
lookahead = 4          # Nombre de jours pour déterminer la tendance future
stride = 5             # Décalage entre chaque séquence

# === Étape 3 : Création des dossiers de sauvegarde ===
base_dir = 'dataset_auto/sequences'
uptrend_base = os.path.join(base_dir, 'uptrend')
downtrend_base = os.path.join(base_dir, 'downtrend')
os.makedirs(uptrend_base, exist_ok=True)
os.makedirs(downtrend_base, exist_ok=True)

# === Étape 4 : Génération des séquences d'images ===
sequence_id = 0

for start_idx in range(0, len(df) - (window_size + lookahead + sequence_len - 1), stride):
    last_i = start_idx + sequence_len - 1
    last_window_end = last_i + window_size
    future_start = last_window_end
    future_end = future_start + lookahead

    if future_end > len(df):
        break

    # Déterminer la tendance
    current_close = df.iloc[last_window_end - 1]['close']
    future_mean = df.iloc[future_start:future_end]['close'].mean()

    if current_close < future_mean:
        label = 'uptrend'
        save_base = uptrend_base
    elif current_close > future_mean:
        label = 'downtrend'
        save_base = downtrend_base
    else:
        continue  # Skip si pas de tendance claire

    # Créer le dossier de séquence
    sequence_folder = os.path.join(save_base, f'seq_{sequence_id}')
    os.makedirs(sequence_folder, exist_ok=True)

    # Générer les images
    for t in range(sequence_len):
        i = start_idx + t
        past_window = df.iloc[i:i + window_size]
        chart_path = os.path.join(sequence_folder, f'img_{t}.png')

        # Tracer et sauvegarder le graphique en chandelier
        mpf.plot(
            past_window,
            type='candle',
            style='charles',
            volume=False,
            savefig=dict(fname=chart_path, dpi=100, bbox_inches='tight')
        )

    sequence_id += 1

print(f"✅ {sequence_id} séquences générées dans le dossier : {base_dir}")
