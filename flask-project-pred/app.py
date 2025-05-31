import os
import pandas as pd
from flask import Flask, render_template
import plotly.graph_objects as go
import plotly.io as pio

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')  # Accueil simple

@app.route('/backtest')
def backtest():
    # --- Lecture de la data ---
    df = pd.read_csv("tesla_ohlcv_365jours.csv", parse_dates=["datetime"])
    df.set_index('datetime', inplace=True)

    # --- Création du graphique ---
    fig = go.Figure(data=[
        go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name="TSLA"
        )
    ])

    fig.update_layout(
        title="Tesla - 365 derniers jours (bougies)",
        xaxis_title="Date",
        yaxis_title="Prix",
        xaxis_rangeslider_visible=True,
        template="plotly_white"
    )

    # --- Conversion en HTML ---
    backtest_html = pio.to_html(fig, full_html=False)

    return render_template("backtest.html", backtest_html=backtest_html)


if __name__ == '__main__':
    app.run(debug=True)
