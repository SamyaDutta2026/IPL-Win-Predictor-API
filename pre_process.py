import pandas as pd
import json

DATA_FILE = 'ipl_matches_2008_2025.csv'

print(f"Reading the large file: {DATA_FILE}...")

 
df = pd.read_csv(DATA_FILE, low_memory=False)

 
print("Processing venues...")
all_venues = sorted(df['venue'].dropna().unique())
with open('venues.json', 'w') as f:
    json.dump(all_venues, f)
print("Saved venues.json")

 
print("Processing toss decisions...")
match_df = df.drop_duplicates(subset=['match_id'])
toss_counts = match_df['toss_decision'].value_counts()
with open('toss_data.json', 'w') as f:
    json.dump(toss_counts.to_dict(), f)
print("Saved toss_data.json")

print("\nPre-processing complete! You can now push these new .json files.")