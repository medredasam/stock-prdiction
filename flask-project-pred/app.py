import os
import pandas as pd
from flask import Flask, render_template
import plotly.graph_objects as go
import plotly.io as pio

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/backtest')
def backtest():
    df = pd.read_csv("tesla_ohlcv_365jours.csv", parse_dates=["datetime"])
    df.set_index('datetime', inplace=True)

    fig = go.Figure(data=[
        go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name="TSLA",
            increasing_line_color="#4fa98a",  # muted teal/green from theme
            decreasing_line_color="#a94f4f",  # muted soft red from theme
            hoverinfo="x+y+text"
        )
    ])

    # Add a blurred rectangle shape as background to simulate blur effect
    # This rectangle is semi-transparent lime with a subtle glow
    fig.update_layout(
        shapes=[
            dict(
                type="rect",
                xref="paper", yref="paper",
                x0=0, y0=0, x1=1, y1=1,
                fillcolor="rgba(79,169,138,0.15)",  # light transparent lime
                layer="below",
                line_width=0,
                # no direct blur, but transparency + color mimics subtle glow
            )
        ],
        title=dict(
            text="Analyse technique TSLA - 365 jours",
            font=dict(color="#84cc16", size=22),  # lime accent color
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            title="Date",
            gridcolor="#333333",
            color="#d6f17e",  # lighter lime for axis labels
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
        plot_bgcolor="rgba(25,25,25,0.7)",  # slightly lighter black with opacity for blur effect
        font=dict(color="#d6f17e", family="Inter, sans-serif"),
        hovermode="x unified",
        margin=dict(l=40, r=100, t=60, b=40),
    )

    backtest_html = pio.to_html(fig, full_html=False)

    return render_template("backtest.html", backtest_html=backtest_html)
if __name__ == '__main__':
    app.run(debug=True)
