from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def home():
    # Données pour les cartes de fonctionnalités
    features = [
        {
            'title': 'DEX Trading',
            'content': 'There are many variations of passages of always available but the majority human'
        },
        {
            'title': 'Best Security',
            'content': 'Passages of always available but the majority human perception is tuned and prevents'
        },
        {
            'title': 'Low-risk Pools',
            'content': 'True generator on the internet: It uses a dictionary of over 200 Latin words combined'
        },
        {
            'title': 'Liquidity',
            'content': 'Mistaken idea of main denoun pleasure and praising pain was born there anyone'
        },
        {
            'title': 'Governance',
            'content': 'We use as filler text for layouts, non-readability is of great importance because'
        },
        {
            'title': 'Unlock Value',
            'content': 'Many variations of passages of always available but the majority human perception is tuned'
        }
    ]
    
    # Données pour les statistiques
    stats = [
        {'value': '289 M+', 'label': 'Total supply'},
        {'value': '563 K+', 'label': 'Total Trades'},
        {'value': '2.6T+', 'label': 'Trade Volume'},
        {'value': '$940M+', 'label': 'Market cap'}
    ]
    
    # Données pour les articles
    articles = [
        {'title': 'Launches new feature', 'date': 'Feb 10, 2023', 'content': 'allowing Swaps'},
        {'title': 'Partnership with major', 'date': 'Jun 29, 2023', 'content': 'DeFi Company'},
        {'title': 'Defi industry amid Rising', 'date': 'Jan 7, 2023', 'content': 'Popularity'}
    ]
    
    return render_template('index.html', features=features, stats=stats, articles=articles)

@app.route('/upload')
def upload():
    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)