import os
import pandas as pd
from flask import Flask, render_template
import plotly.graph_objects as go
import plotly.io as pio

app = Flask(__name__)

@app.route('/')
def home():
<<<<<<< HEAD
    return render_template('index.html')

@app.route('/backtest')
def backtest():
    # --- Lecture des données ---
=======
    return render_template('index.html')  # Accueil simple

@app.route('/backtest')
def backtest():
    # --- Lecture de la data ---
>>>>>>> c1add2d8e4227ac15d70239662641616769b8749
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
<<<<<<< HEAD
            name="TSLA",
            increasing_line_color="#00ff9f",
            decreasing_line_color="#ff4d4d"
=======
            name="TSLA"
>>>>>>> c1add2d8e4227ac15d70239662641616769b8749
        )
    ])

    fig.update_layout(
<<<<<<< HEAD
        xaxis_title="Date",
        yaxis_title="Prix",
        xaxis_rangeslider_visible=True,
        plot_bgcolor="#0f0f0f",
        paper_bgcolor="#0f0f0f",
        font=dict(color="white"),
        xaxis=dict(gridcolor="#333", linecolor="white"),
        yaxis=dict(gridcolor="#333", linecolor="white")
    )

=======
        title="Tesla - 365 derniers jours (bougies)",
        xaxis_title="Date",
        yaxis_title="Prix",
        xaxis_rangeslider_visible=True,
        template="plotly_white"
    )

    # --- Conversion en HTML ---
>>>>>>> c1add2d8e4227ac15d70239662641616769b8749
    backtest_html = pio.to_html(fig, full_html=False)

    return render_template("backtest.html", backtest_html=backtest_html)

<<<<<<< HEAD
=======

>>>>>>> c1add2d8e4227ac15d70239662641616769b8749
if __name__ == '__main__':
    app.run(debug=True)
