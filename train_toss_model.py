import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib
import warnings

warnings.filterwarnings('ignore')

print("Starting Toss Decision Model training...")

TEAM_NAME_MAP = {
    'Delhi Daredevils': 'Delhi Capitals',
    'Kings XI Punjab': 'Punjab Kings',
    'Rising Pune Supergiant': 'Rising Pune Supergiants',
    'Pune Warriors': 'Rising Pune Supergiants',
    'Gujarat Lions': 'Gujarat Titans'
}

DATA_FILE = 'ipl_matches_2008_2025.csv'

def get_toss_data(df):
    print("Aggregating data for toss decisions...")
    
    df['toss_winner'] = df['toss_winner'].replace(TEAM_NAME_MAP)
    
    match_df = df.drop_duplicates(subset=['match_id'])
    
    match_df = match_df[['toss_winner', 'venue', 'toss_decision']]
    match_df = match_df.dropna()
    
    match_df = match_df[(match_df['toss_decision'] == 'bat') | (match_df['toss_decision'] == 'field')]
    
    return match_df

try:
    df = pd.read_csv(DATA_FILE, low_memory=False)
except FileNotFoundError:
    print(f"Error: '{DATA_FILE}' not found.")
    exit()

print("EXTRACT: Reading data...")
model_df = get_toss_data(df)

print("TRANSFORM: Preparing data for classification model...")
features = ['toss_winner', 'venue']
target = 'toss_decision'

X = model_df[features]
y = model_df[target]

categorical_transformer = OneHotEncoder(handle_unknown='ignore')
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, features)
    ])

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression())
])

print("TRAIN: Training the toss model...")
model.fit(X, y)

print("LOAD: Saving 'toss_model.pkl'...")
joblib.dump(model, 'toss_model.pkl')

print("Toss decision model training complete! ✅")