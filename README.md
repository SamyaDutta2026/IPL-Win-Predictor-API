 # IPL Prediction Hub & Data Analysis Dashboard

This is a full-stack data science project that analyzes a complete 2008-2025 ball-by-ball IPL dataset. It performs complex data aggregation, trains three different machine learning models, and serves a live, interactive web dashboard built with Flask.

## 🚀 Features

This application provides a "Prediction Hub" with three separate ML-powered predictors, as well as two live-data visualizations:

### **Prediction Models:**
1.  **Match Winner Predictor:** Predicts the winning team based on the two opponents.
2.  **First Innings Score Predictor:** Predicts the projected final score for the team batting first, based on the batting team, bowling team, and venue.
3.  **Toss Decision Predictor:** Predicts whether a toss-winning captain will choose to "Bat" or "Field" based on the team and venue.

### **Data Visualizations:**
1.  **Team Win Summary:** A dynamic bar chart showing the total all-time wins for every team in the league.
2.  **League-Wide Toss Decisions:** A pie chart visualizing the overall "Bat" vs. "Field" decisions across the entire league.

## 🛠️ Tech Stack

* **Backend:** Python, Flask
* **Data Processing:** Pandas
* **Machine Learning:** Scikit-learn (Logistic Regression, Linear Regression)
* **Frontend:** HTML, CSS, JavaScript (Chart.js)

## 📁 Project Structure
. ├── .gitignore # Tells Git what to ignore ├── app.py # The main Flask application ├── process_ipl_data.py # ETL script: creates the win summary ├── train_model.py # Trains the winner prediction model ├── train_score_model.py # Trains the score prediction model ├── train_toss_model.py # Trains the toss prediction model ├── requirements.txt # All Python dependencies │ ├── ipl_matches_2008_2025.csv # (Ignored by Git) The raw dataset │ ├── /templates │ └── index.html # The frontend HTML file │ └── # (Generated Files - Ignored by Git) ├── model.pkl ├── encoder.pkl ├── score_model.pkl ├── toss_model.pkl └── team_win_summary.csv

## 🏃 How to Run This Project

### Step 1: Clone the Repository

```bash
git clone [https://github.com/SamyaDutta2026/IPL-Win-Predictor-API.git](https://github.com/SamyaDutta2026/IPL-Win-Predictor-API.git)



Step 2: Set Up the Environment
# Install all the required Python libraries.
pip install -r requirements.txt


Step 3: Get the Dataset

This project requires the ipl_matches_2008_2025.csv ball-by-ball dataset.
1. Download it from Kaggle (https://www.kaggle.com/datasets/chaitu20/ipl-dataset2008-2025).
2. Place the file inside your main project folder and 
3. ensure it is named ipl_matches_2008_2025.csv.

Step 4: Run the Data & Model Pipeline
You must run these scripts in order to clean the data and train the three models. 

# 1. Run the ETL script to create the win summary
python process_ipl_data.py

# 2. Run the training scripts to create the models
python train_model.py
python train_score_model.py
python train_toss_model.py


Step 5: Run the Web App
After the pipeline is complete, run the Flask application.

python app.py



Step 6: View Your Dashboard
Open your web browser and go to: http://127.0.0.1:5000/




📁 Project Structure
.
├── ipl_matches_2008_2025.csv   (The raw dataset you must download)
├── process_ipl_data.py         (ETL script: creates the win summary)
├── train_model.py              (Trains the winner prediction model)
├── train_score_model.py        (Trains the score prediction model)
├── train_toss_model.py         (Trains the toss prediction model)
├── app.py                      (The main Flask application)
├── requirements.txt            (All Python dependencies)
├── model.pkl                   (Generated winner model)
├── encoder.pkl                 (Generated winner encoder)
├── score_model.pkl             (Generated score model)
├── toss_model.pkl              (Generated toss model)
├── team_win_summary.csv        (Generated data for the bar chart)
└── /templates
    └── index.html              (The frontend HTML file)



👨‍💻 Author
Samya Dutta

GitHub: https://github.com/SamyaDutta2026