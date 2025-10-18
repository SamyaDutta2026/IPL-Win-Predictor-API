import pandas as pd
import warnings

warnings.filterwarnings('ignore')

print("Starting the UPDATED IPL Data Pipeline...")

TEAM_NAME_MAP = {
    'Delhi Daredevils': 'Delhi Capitals',
    'Kings XI Punjab': 'Punjab Kings',
    'Rising Pune Supergiant': 'Rising Pune Supergiants',
    'Pune Warriors': 'Rising Pune Supergiants',
    'Gujarat Lions': 'Gujarat Titans'
}

DATA_FILE = 'ipl_matches_2008_2025.csv'
OUTPUT_FILE = 'team_win_summary.csv'

def run_pipeline():
    try:
        df = pd.read_csv(DATA_FILE, low_memory=False)
    except FileNotFoundError:
        print(f"Error: '{DATA_FILE}' not found.")
        print(f"Please rename your new dataset to '{DATA_FILE}'.")
        return

    print("EXTRACT: Reading data...")
    
    df['match_won_by'] = df['match_won_by'].replace(TEAM_NAME_MAP)
    df = df.dropna(subset=['match_won_by'])

    print("TRANSFORM: Finding unique matches and winners...")
    match_df = df.drop_duplicates(subset=['match_id'])
    
    team_wins = match_df['match_won_by'].value_counts()

    win_summary_df = pd.DataFrame({
        'team': team_wins.index, 
        'total_wins': team_wins.values
    })

    print(f"LOAD: Saving the results to {OUTPUT_FILE}...")
    win_summary_df.to_csv(OUTPUT_FILE, index=False)
    
    print("\nPipeline finished successfully!")
    print(f"Check the output file: {OUTPUT_FILE}")
    print("\nHere is a preview of the new results:")
    print(win_summary_df.head())

if __name__ == "__main__":
    run_pipeline()