
# import os
# import numpy as np
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.image import load_img, img_to_array
# from tensorflow.keras.applications.resnet import preprocess_input

# # Taille des images attendues par le modèle
# IMG_SIZE = (224, 224)

# # Fonction de chargement des séquences
# def load_all_sequences_for_backtest(base_dir="dataset_auto/sequences"):
#     """
#     Charge toutes les séquences d'images (5 images par séquence) depuis les sous-dossiers uptrend/downtrend.
#     Retourne une liste de séquences sous forme de tableaux numpy.
#     """
#     all_sequences = []

#     for trend_label in ["uptrend", "downtrend"]:
#         trend_dir = os.path.join(base_dir, trend_label)
#         for seq_folder in sorted(os.listdir(trend_dir)):
#             seq_path = os.path.join(trend_dir, seq_folder)
#             if os.path.isdir(seq_path):
#                 sequence = []
#                 for i in range(5):
#                     img_path = os.path.join(seq_path, f"img_{i}.png")
#                     if os.path.exists(img_path):
#                         img = load_img(img_path, target_size=IMG_SIZE)
#                         img = img_to_array(img)
#                         img = preprocess_input(img)
#                         sequence.append(img)
#                 if len(sequence) == 5:
#                     all_sequences.append(np.array(sequence))
#     return all_sequences



# # Charger les séquences depuis les images générées
# sequences = load_all_sequences_for_backtest()

# # Prédire un signal (0 ou 1) pour chaque séquence
# signals = []
# for seq in sequences:
#     seq_input = np.expand_dims(seq, axis=0)  # (1, 5, 224, 224, 3)
#     prediction = model.predict(seq_input, verbose=0)
#     signal = int(np.argmax(prediction))
#     signals.append(signal)

# # Afficher les signaux
# print("Signaux prédits :", signals)


import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
from backtesting import Backtest, Strategy
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import traceback  


def run_backtest_model(model_name,model_path, dataset, stride, days_per_image):
    class Config:
        IMG_SIZE = (224, 224)
        SEQ_LEN = 5
        MODEL_NAME = model_name
        MODEL_PATH = model_path
        DATA_PATH = "tesla_ohlcv_365jours.csv"
        SEQUENCES_DIR = dataset
        DAYS_PER_IMAGE = days_per_image
        STRIDE = stride
        INITIAL_CASH = 10000
        COMMISSION = 0.002
        SLIPPAGE = 0.001
        STOP_LOSS = 0.10
        TAKE_PROFIT = 0.20
        RISK_FREE_RATE = 0.0

    def SMA(arr, n):
        return pd.Series(arr).rolling(n).mean().values

    def generate_signals(df):
        try:
            # Load model first before anything else
            if not os.path.exists(Config.MODEL_PATH):
                raise FileNotFoundError(f"Model file {Config.MODEL_PATH} not found")
                
            model = load_model(Config.MODEL_PATH)
            df['Signal'] = -1
            predictions, actuals, prediction_dates = [], [], []

            if not os.path.exists(Config.SEQUENCES_DIR):
                raise FileNotFoundError(f"Le dossier {Config.SEQUENCES_DIR} n'existe pas")

            for trend_dir in ['uptrend', 'downtrend']:
                dir_path = os.path.join(Config.SEQUENCES_DIR, trend_dir)
                if not os.path.exists(dir_path):
                    continue

                for seq_folder in sorted(os.listdir(dir_path), key=lambda x: int(x.split('_')[1])):
                    seq_path = os.path.join(dir_path, seq_folder)
                    if not os.path.isdir(seq_path):
                        continue

                    sequence = []
                    valid_sequence = True
                    for i in range(Config.SEQ_LEN):
                        img_path = os.path.join(seq_path, f'img_{i}.png')
                        if not os.path.exists(img_path):
                            valid_sequence = False
                            break
                        img = load_img(img_path, target_size=Config.IMG_SIZE)
                        img = img_to_array(img)
                        img = preprocess_input(img)
                        sequence.append(img)

                    if valid_sequence and len(sequence) == Config.SEQ_LEN:
                        seq_array = np.expand_dims(np.array(sequence), axis=0)
                        prediction = model.predict(seq_array, verbose=0)
                        signal = np.argmax(prediction)
                        seq_idx = int(seq_folder.split('_')[1])
                        candle_idx = seq_idx * Config.STRIDE + (Config.SEQ_LEN - 1) * Config.DAYS_PER_IMAGE

                        if candle_idx >= len(df):
                            continue

                        df.iloc[candle_idx, df.columns.get_loc('Signal')] = signal
                        predictions.append(signal)
                        actual = 0 if df.iloc[candle_idx]['Close'] > df.iloc[candle_idx - 1]['Close'] else 1
                        actuals.append(actual)
                        prediction_dates.append(df.index[candle_idx])

        except Exception as e:
            raise RuntimeError(f"Erreur dans generate_signals: {str(e)}")

        if len(predictions) == 0:
            raise ValueError("Aucun signal généré")

        return df, predictions, actuals, prediction_dates

    class MLStrategy(Strategy):
        stop_loss_pct = Config.STOP_LOSS
        take_profit_pct = Config.TAKE_PROFIT

        def init(self):
            close_prices = np.array(self.data.Close)
            self.sma20 = self.I(SMA, close_prices, 20, name='SMA20')
            self.sma50 = self.I(SMA, close_prices, 50, name='SMA50')
            self.signal = np.array(self.data.Signal)
            if len(np.unique(self.signal)) == 1 and self.signal[0] == -1:
                raise ValueError("Aucun signal valide trouvé")

        def next(self):
            current_idx = len(self.data.Close) - 1
            long_cond = (self.signal[current_idx] == 0) and (self.sma20[current_idx] > self.sma50[current_idx])
            short_cond = (self.signal[current_idx] == 1) and (self.sma20[current_idx] < self.sma50[current_idx])

            if not self.position:
                price = self.data.Close[-1]
                if long_cond:
                    self.buy(sl=price*(1-self.stop_loss_pct), tp=price*(1+self.take_profit_pct))
                elif short_cond:
                    self.sell(sl=price*(1+self.stop_loss_pct), tp=price*(1-self.take_profit_pct))
            else:
                if self.position.is_long and self.signal[current_idx] == 1:
                    self.position.close()
                elif self.position.is_short and self.signal[current_idx] == 0:
                    self.position.close()

    try:
        # Chargement des données
        if not os.path.exists(Config.DATA_PATH):
            raise FileNotFoundError(f"Fichier {Config.DATA_PATH} introuvable")
        
        df = pd.read_csv(Config.DATA_PATH, parse_dates=['datetime'])
        if df.empty:
            raise ValueError("Le fichier de données est vide")
        
        df.rename(columns={
            'datetime': 'Date', 'open': 'Open', 'high': 'High',
            'low': 'Low', 'close': 'Close', 'volume': 'Volume'
        }, inplace=True)
        df.sort_values('Date', inplace=True)
        df.set_index('Date', inplace=True)
        
        if len(df) < Config.SEQ_LEN * Config.DAYS_PER_IMAGE:
            raise ValueError("Données insuffisantes pour la longueur de séquence configurée")
        
        df, predictions, actuals, prediction_dates = generate_signals(df)
        df['SMA20'] = SMA(df['Close'], 20)
        df['SMA50'] = SMA(df['Close'], 50)
        
        # Exécution du backtest
        bt = Backtest(df, MLStrategy, cash=Config.INITIAL_CASH, commission=Config.COMMISSION)
        results = bt.run()
        
        # # Préparation des résultats pour Flask
        # trades_df = results['_trades']
        # num_trades = len(trades_df)
        # num_winning = len(trades_df[trades_df['PnL'] > 0])
        # num_losing = len(trades_df[trades_df['PnL'] < 0])
        # win_rate = (num_winning / num_trades) * 100 if num_trades else 0
        # profit_factor = trades_df['PnL'][trades_df['PnL'] > 0].sum() / abs(trades_df['PnL'][trades_df['PnL'] < 0].sum()) if num_losing else float('inf')
        
        # Préparation des résultats du backtest
        trades_df = results['_trades']

        # Création du graphique
        equity_curve = pd.DataFrame({'Equity': results['_equity_curve']['Equity']}, index=df.index[-len(results['_equity_curve']):])
        
        equity_series = results['_equity_curve']['Equity']

                # Calculate max drawdown
        equity = results['_equity_curve']['Equity']
        rolling_max = equity.cummax()
        drawdown = (equity - rolling_max) / rolling_max
        max_drawdown = drawdown.min() * 100  # as percentage

        num_trades = len(trades_df)
        num_winning = len(trades_df[trades_df['PnL'] > 0])
        num_losing = len(trades_df[trades_df['PnL'] < 0])
        win_rate = (num_winning / num_trades) * 100 if num_trades else 0
        profit_factor = trades_df['PnL'][trades_df['PnL'] > 0].sum() / abs(trades_df['PnL'][trades_df['PnL'] < 0].sum()) if num_losing else float('inf')

        # Calculate Sharpe ratio
        returns = equity.pct_change().dropna()
        sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std() if returns.std() != 0 else 0
        # Capital max/min
        capital_max = equity_series.max()
        capital_min = equity_series.min()

        # Dernière prédiction
        last_signal = predictions[-1] if predictions else None
        last_prediction_date = prediction_dates[-1] if prediction_dates else None

        # Capital à la date de la dernière prédiction
        last_equity = None
        if last_prediction_date and last_prediction_date in df.index:
            try:
                equity_curve_index = equity_curve.index.get_loc(last_prediction_date)
                last_equity = equity_series.iloc[equity_curve_index]
            except KeyError:
                last_equity = None

        # Création du graphique
        equity_curve = pd.DataFrame({'Equity': results['_equity_curve']['Equity']}, index=df.index[-len(results['_equity_curve']):])
        
        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05,
            row_heights=[0.7, 0.3], subplot_titles=("Graphique des Prix", "Courbe d'Équité")
        )
        
        fig.add_trace(go.Candlestick(
            x=df.index, open=df['Open'], high=df['High'],
            low=df['Low'], close=df['Close'], name='OHLC'),
            row=1, col=1
        )
        
        fig.add_trace(go.Scatter(
            x=df.index, y=df['SMA20'], name='SMA 20',
            line=dict(color='blue', width=1)), row=1, col=1
        )
        fig.add_trace(go.Scatter(
            x=df.index, y=df['SMA50'], name='SMA 50',
            line=dict(color='orange', width=1)), row=1, col=1
        )
        
        buy_signals = df[df['Signal'] == 0]
        sell_signals = df[df['Signal'] == 1]
        fig.add_trace(go.Scatter(
            x=buy_signals.index, y=buy_signals['Close'],
            mode='markers', name='Buy', marker=dict(color='green', size=10, symbol='triangle-up')),
            row=1, col=1
        )
        fig.add_trace(go.Scatter(
            x=sell_signals.index, y=sell_signals['Close'],
            mode='markers', name='Sell', marker=dict(color='red', size=10, symbol='triangle-down')),
            row=1, col=1
        )
        
        fig.add_trace(go.Scatter(
            x=equity_curve.index, y=equity_curve['Equity'],
            name='Equity', line=dict(color='purple', width=2)),
            row=2, col=1
        )
        
        fig.update_layout(
        title=f'Résultats du Backtest - {model_name}',
        height=800,
        showlegend=True,
        xaxis_rangeslider_visible=False,
        hovermode='x unified',
        plot_bgcolor='#f8f9fa',
        paper_bgcolor='#f8f9fa'
)


        fig.update_yaxes(title_text="Prix ($)", row=1, col=1)
        fig.update_yaxes(title_text="Capital ($)", row=2, col=1)
        
        # Génération du HTML
        plot_html = fig.to_html(
        full_html=False,
        include_plotlyjs='cdn',  # Important pour charger Plotly.js
        config={'responsive': True}
    )
        
        # Préparation des résultats
        backtest_results = {
            'success': True,
            'final_return': (results['Equity Final [$]'] - Config.INITIAL_CASH)/Config.INITIAL_CASH*100,
            'num_trades': num_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
             'max_drawdown': max_drawdown,  
            'sharpe_ratio': float(sharpe_ratio),  
            'capital_max': capital_max,
            'capital_min': capital_min,
            'last_prediction': {
                'date': str(last_prediction_date) if last_prediction_date else None,
                'signal': 'Buy' if last_signal == 0 else 'Sell' if last_signal == 1 else 'Unknown',
                'equity': last_equity
            },
            'plot_html': plot_html,
            'raw_results': results,
            'equity_curve': equity_curve.to_dict(),
            'signals': {
                'buy': buy_signals[['Close']].to_dict(),
                'sell': sell_signals[['Close']].to_dict()
            }
        }

        
        return backtest_results
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc() if 'traceback' in globals() else None
        }

