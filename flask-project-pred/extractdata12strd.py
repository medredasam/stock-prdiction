import os
import pandas as pd
import mplfinance as mpf

# --- Paramètres ---
csv_file = "NVDA.csv"  # Chemin vers le fichier CSV
window_size = 8
sequence_len = 5
lookahead = 3
stride = 12

# --- Chargement des données depuis le CSV ---
df = pd.read_csv(csv_file)

# Assurez-vous que les colonnes sont bien nommées et converties
df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].apply(pd.to_numeric)
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(by='date').reset_index(drop=True)
df['date'] = df['date']
df.set_index('date', inplace=True)

# --- Répertoires de sauvegarde ---
base_dir = 'NVDA_8Bougie/sequences'
uptrend_base = os.path.join(base_dir, 'uptrend')
downtrend_base = os.path.join(base_dir, 'downtrend')
os.makedirs(uptrend_base, exist_ok=True)
os.makedirs(downtrend_base, exist_ok=True)

# --- Génération des séquences ---
sequence_id = 0
for start_idx in range(0, len(df) - (window_size + lookahead + sequence_len - 1), stride):

    last_i = start_idx + sequence_len - 1
    last_window_end = last_i + window_size
    future_start = last_window_end
    future_end = future_start + lookahead

    if future_end > len(df):
        break

    current_close = df.iloc[last_window_end - 1]['close']
    future_mean = df.iloc[future_start:future_end]['close'].mean()

    if current_close < future_mean:
        label = 'uptrend'
        save_base = uptrend_base
    elif current_close > future_mean:
        label = 'downtrend'
        save_base = downtrend_base
    else:
        continue

    sequence_folder = os.path.join(save_base, f'seq_{sequence_id}')
    os.makedirs(sequence_folder, exist_ok=True)

    for t in range(sequence_len):
        i = start_idx + t
        past_window = df.iloc[i:i + window_size]
        chart_path = os.path.join(sequence_folder, f'img_{t}.png')

        mpf.plot(
            past_window,
            type='candle',
            style='charles',
            volume=False,
            savefig=dict(fname=chart_path, dpi=100, bbox_inches='tight')
        )

    sequence_id += 1