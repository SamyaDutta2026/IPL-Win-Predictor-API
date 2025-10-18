from flask import Flask, render_template, jsonify, request
import pandas as pd
import joblib
import warnings

warnings.filterwarnings('ignore')

app = Flask(__name__)

DATA_FILE = 'ipl_matches_2008_2025.csv'
SUMMARY_FILE = 'team_win_summary.csv'

try:
    winner_model = joblib.load('model.pkl')
    winner_encoder = joblib.load('encoder.pkl')
except FileNotFoundError:
    winner_model = None
    winner_encoder = None

try:
    score_model = joblib.load('score_model.pkl')
except FileNotFoundError:
    score_model = None

try:
    toss_model = joblib.load('toss_model.pkl')
except FileNotFoundError:
    toss_model = None

try:
    df_summary = pd.read_csv(SUMMARY_FILE)
    all_teams = sorted(df_summary['team'].unique())
except FileNotFoundError:
    all_teams = ["Data file not found"]

try:
    df_full = pd.read_csv(DATA_FILE, low_memory=False)
    all_venues = sorted(df_full['venue'].dropna().unique())
except FileNotFoundError:
    all_venues = ["Data file not found"]


@app.route('/')
def dashboard():
    return render_template('index.html', all_teams=all_teams, all_venues=all_venues)

@app.route('/api/win_data')
def get_win_data():
    try:
        df = pd.read_csv(SUMMARY_FILE)
        df['total_wins'] = pd.to_numeric(df['total_wins'])
        data_for_chart = {
            'teams': df['team'].tolist(),
            'wins': df['total_wins'].tolist()
        }
        return jsonify(data_for_chart)
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/api/toss_data')
def get_toss_data():
    try:
        df = pd.read_csv(DATA_FILE, low_memory=False)
        match_df = df.drop_duplicates(subset=['match_id'])
        toss_counts = match_df['toss_decision'].value_counts()
        return jsonify(toss_counts.to_dict())
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/predict_winner', methods=['POST'])
def predict_winner():
    if not winner_model or not winner_encoder:
        return jsonify({"error": "Winner model not loaded."})

    try:
        data = request.get_json()
        team1 = data['team1']
        team2 = data['team2']

         
        sorted_input = sorted([team1, team2])
        input_df = pd.DataFrame([sorted_input], columns=['team1', 'team2'])
         
        
        input_encoded = winner_encoder.transform(input_df)

        prediction = winner_model.predict(input_encoded)
        prediction_proba = winner_model.predict_proba(input_encoded)
        winner_prob = prediction_proba[0][list(winner_model.classes_).index(prediction[0])]
        
        return jsonify({
            'predicted_winner': prediction[0],
            'win_probability': f"{winner_prob * 100:.2f}%"
        })
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/predict_score', methods=['POST'])
def predict_score():
    if not score_model:
        return jsonify({"error": "Score model not loaded."})
    
    try:
        data = request.get_json()
        batting_team = data['batting_team']
        bowling_team = data['bowling_team']
        venue = data['venue']

        input_df = pd.DataFrame([[batting_team, bowling_team, venue]], 
                                columns=['batting_team', 'bowling_team', 'venue'])
        
        prediction = score_model.predict(input_df)
        predicted_score = int(prediction[0])

        return jsonify({
            'predicted_score': f"{predicted_score - 5} to {predicted_score + 5}"
        })
    
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/predict_toss', methods=['POST'])
def predict_toss():
    if not toss_model:
        return jsonify({"error": "Toss model not loaded."})
    
    try:
        data = request.get_json()
        toss_winner = data['toss_winner']
        venue = data['venue']

        input_df = pd.DataFrame([[toss_winner, venue]], 
                                columns=['toss_winner', 'venue'])
        
        prediction = toss_model.predict(input_df)
        
        return jsonify({
            'predicted_decision': prediction[0].capitalize()
        })
    
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)