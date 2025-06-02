import os
import pandas as pd
from flask import Flask, render_template
import plotly.graph_objects as go
import plotly.io as pio
from flask import request
from backtest import run_backtest_model
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import traceback


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')
@app.route('/backtest')
def backtest():
    import pandas as pd
    import plotly.graph_objects as go
    import plotly.io as pio

    # Charger les données
    df = pd.read_csv("tesla_ohlcv_365jours.csv", parse_dates=["datetime"])
    df.set_index('datetime', inplace=True)

    # Créer le graphique en chandelier
    fig = go.Figure(data=[
        go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name="TSLA",
            increasing_line_color="#4fa98a",  # vert doux
            decreasing_line_color="#a94f4f",  # rouge doux
            hoverinfo="x+y+text"
        )
    ])

    # Personnalisation du layout
    fig.update_layout(
        width=680,  # Largeur augmentée
        height=400,
        shapes=[
            dict(
                type="rect",
                xref="paper", yref="paper",
                x0=0, y0=0, x1=1, y1=1,
                fillcolor="rgba(79,169,138,0.15)",  # vert clair transparent
                layer="below",
                line_width=0,
            )
        ],
        title=dict(
            text=" Your initial cash is 10000 $",
            font=dict(color="#84cc16", size=22),
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            title="Date",
            gridcolor="#333333",
            color="#d6f17e",
            linecolor="#5e7a3f",
            showspikes=True,
            spikecolor="#84cc16",
            spikethickness=2,
            rangeslider_visible=False,
        ),
        yaxis=dict(
            title="Prix (USD)",
            gridcolor="#333333",
            color="#d6f17e",
            linecolor="#5e7a3f",
            showspikes=True,
            spikecolor="#84cc16",
            spikethickness=2,
        ),
        paper_bgcolor="#101010",
        plot_bgcolor="rgba(25,25,25,0.7)",
        font=dict(color="#d6f17e", family="Inter, sans-serif"),
        hovermode="x unified",
        margin=dict(l=40, r=100, t=60, b=40),  # marge droite réduite
    )

    # Convertir en HTML
    backtest_html = pio.to_html(fig, full_html=False)

    # Retourner le template
    return render_template("backtest.html", backtest_html=backtest_html)

@app.route('/run-backtest', methods=['POST'])
def run_backtest():
    model_name = request.form.get('model_name')
    
    model_config = {
        "model_8": {
            "model_path": "models/8bougie_3strides_model_20250531_020107.keras",
            "dataset": "dataset_auto_8_bougie/sequences",
            "stride": 3,
            "days_per_image": 8
        },
        "model_10": {
            "model_path": "models/M_10bougie_5stride_model_20250531_110014.keras",
            "dataset": "dataset_auto_10_bougie/sequences", 
            "stride": 5,
            "days_per_image": 10
        },
        "model_20": {
            "model_path": "models/end2end_model_20250530_203710.keras",
            "dataset": "dataset_auto_20_bougie/sequences",
            "stride": 5,
            "days_per_image": 20
        }
    }

    if model_name not in model_config:
        return render_template("error.html", message="Modèle non reconnu"), 400

    config = model_config[model_name]
    
    try:
        # Vérification des fichiers avant exécution
        if not os.path.exists(config["model_path"]):
            return render_template("error.html", message=f"Fichier modèle introuvable: {config['model_path']}"), 400
            
        if not os.path.exists(config["dataset"]):
            return render_template("error.html", message=f"Dossier dataset introuvable: {config['dataset']}"), 400

        result = run_backtest_model(
            model_name=model_name,
            model_path=config["model_path"],
            dataset=config["dataset"],
            stride=config["stride"],
            days_per_image=config["days_per_image"]
        )
        
        if not result['success']:
            return render_template("error.html", message=result.get('error', 'Erreur inconnue')), 400

        # Ajout des dépendances JavaScript pour Plotly
        result['plot_html'] = result['plot_html'].replace(
            '<head>',
            '<head><script src="https://cdn.plot.ly/plotly-latest.min.js"></script>'
        )
        
        return render_template('result.html', results=result)

    except Exception as e:
        app.logger.error(f"Erreur dans run_backtest: {str(e)}\n{traceback.format_exc()}")
        return render_template("error.html", message=str(e))

if __name__ == '__main__':
    app.run(debug=True)