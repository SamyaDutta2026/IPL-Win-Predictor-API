import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib
import warnings

warnings.filterwarnings('ignore')

print("Starting Score Prediction Model training...")

TEAM_NAME_MAP = {
    'Delhi Daredevils': 'Delhi Capitals',
    'Kings XI Punjab': 'Punjab Kings',
    'Rising Pune Supergiant': 'Rising Pune Supergiants',
    'Pune Warriors': 'Rising Pune Supergiants',
    'Gujarat Lions': 'Gujarat Titans'
}

DATA_FILE = 'ipl_matches_2008_2025.csv'

def get_first_innings_data(df):
    print("Aggregating data for first innings scores...")
    
    df['batting_team'] = df['batting_team'].replace(TEAM_NAME_MAP)
    df['bowling_team'] = df['bowling_team'].replace(TEAM_NAME_MAP)
    
    first_innings_df = df[df['innings'] == 1]
    
    match_scores = first_innings_df.groupby('match_id').agg(
        batting_team=('batting_team', 'first'),
        bowling_team=('bowling_team', 'first'),
        venue=('venue', 'first'),
        total_runs=('runs_total', 'sum')
    )
    
    match_scores = match_scores.dropna()
    return match_scores

try:
    df = pd.read_csv(DATA_FILE, low_memory=False)
except FileNotFoundError:
    print(f"Error: '{DATA_FILE}' not found.")
    exit()

print("EXTRACT: Reading data...")
model_df = get_first_innings_data(df)

print("TRANSFORM: Preparing data for regression model...")
features = ['batting_team', 'bowling_team', 'venue']
target = 'total_runs'

X = model_df[features]
y = model_df[target]

categorical_transformer = OneHotEncoder(handle_unknown='ignore')
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, features)
    ])

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

print("TRAIN: Training the score model...")
model.fit(X, y)

print("LOAD: Saving 'score_model.pkl'...")
joblib.dump(model, 'score_model.pkl')

print("Score prediction model training complete! ✅")