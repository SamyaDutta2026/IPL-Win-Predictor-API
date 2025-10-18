import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
import joblib
import warnings

warnings.filterwarnings('ignore')

print("Starting UPDATED model training...")

TEAM_NAME_MAP = {
    'Delhi Daredevils': 'Delhi Capitals',
    'Kings XI Punjab': 'Punjab Kings',
    'Rising Pune Supergiant': 'Rising Pune Supergiants',
    'Pune Warriors': 'Rising Pune Supergiants',
    'Gujarat Lions': 'Gujarat Titans'
}

DATA_FILE = 'ipl_matches_2008_2025.csv'

def get_match_data(df):
    print("Aggregating ball-by-ball data to match-level...")
    
    df['batting_team'] = df['batting_team'].replace(TEAM_NAME_MAP)
    df['bowling_team'] = df['bowling_team'].replace(TEAM_NAME_MAP)
    df['match_won_by'] = df['match_won_by'].replace(TEAM_NAME_MAP)
    
    df = df.dropna(subset=['match_won_by'])
    
    match_data = df.groupby('match_id').agg(
        teams_played=('batting_team', 'unique'),
        winner=('match_won_by', 'first')
    )
    
    match_data = match_data[match_data['teams_played'].apply(len) == 2]
    
   
    def sort_teams(teams):
        return sorted(teams)

    sorted_teams = match_data['teams_played'].apply(sort_teams)
    match_data['team1'] = sorted_teams.apply(lambda x: x[0])
    match_data['team2'] = sorted_teams.apply(lambda x: x[1])
     
    
    return match_data[['team1', 'team2', 'winner']]

try:
    df = pd.read_csv(DATA_FILE, low_memory=False)
except FileNotFoundError:
    print(f"Error: '{DATA_FILE}' not found.")
    exit()

print("EXTRACT: Reading data...")
model_df = get_match_data(df)

print("TRANSFORM: Preparing data for model...")
features = model_df[['team1', 'team2']]
target = model_df['winner']

encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
features_encoded = encoder.fit_transform(features)

print("TRAIN: Training the model...")
model = LogisticRegression()
model.fit(features_encoded, target)

print("LOAD: Saving 'model.pkl' and 'encoder.pkl'...")
joblib.dump(model, 'model.pkl')
joblib.dump(encoder, 'encoder.pkl')

print("UPDATED model training complete! ✅")