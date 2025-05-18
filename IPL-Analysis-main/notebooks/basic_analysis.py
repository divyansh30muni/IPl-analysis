#!/usr/bin/env python
# coding: utf-8

# # IPL Cricket Analysis with AI/ML
# 
# This notebook demonstrates how to analyze IPL cricket data using Python, data science techniques, and machine learning.

# ## Setup
# 
# First, let's import the necessary libraries and modules.

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# Add the project root to the path
sys.path.append('..')

# Import project modules
from scripts.data_collection import IPLDataCollector
from scripts.data_preprocessing import IPLDataProcessor
from scripts.ml_models import IPLMLModels
from scripts.visualization import IPLVisualizer

# Set up plotting
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('viridis')

# ## 1. Data Collection
# 
# First, let's collect IPL data or create sample data if real data collection fails.

data_dir = '../data'

# Initialize data collector
collector = IPLDataCollector(data_dir)

# Try to download real data
cricsheet_success = collector.download_cricsheet_data()
player_stats_success = collector.scrape_player_stats()

# If real data collection fails, create sample data
if not cricsheet_success or not player_stats_success:
    print("Creating sample data for development...")
    collector.create_sample_data()

# ## 2. Data Processing
# 
# Now, let's process the collected data.

# Initialize data processor
processor = IPLDataProcessor(data_dir)

# Process Cricsheet data
processor.process_cricsheet_data()

# Generate player statistics
processor.generate_player_stats()

# Prepare features for ML models
processor.prepare_features_for_ml()

# ## 3. Load and Explore Data
# 
# Let's load the processed data and explore it.

# Load matches data
matches_file = os.path.join(data_dir, 'processed', 'matches.csv')
if not os.path.exists(matches_file):
    matches_file = os.path.join(data_dir, 'processed', 'sample_matches.csv')
    
matches_df = pd.read_csv(matches_file)
print(f"Loaded {len(matches_df)} matches")
print(matches_df.head())

# Load player data
players_file = os.path.join(data_dir, 'processed', 'player_batting_stats.csv')
if not os.path.exists(players_file):
    players_file = os.path.join(data_dir, 'processed', 'sample_players.csv')
    
players_df = pd.read_csv(players_file)
print(f"\nLoaded {len(players_df)} players")
print(players_df.head())

# ## 4. Statistical Analysis
# 
# Let's perform some statistical analysis on the data.

# Team win counts
team_wins = matches_df['winner'].value_counts().reset_index()
team_wins.columns = ['team', 'wins']
print("\nTeam win counts:")
print(team_wins)

# Save wins chart to PNG
plt.figure(figsize=(12, 6))
plt.bar(team_wins['team'], team_wins['wins'])
plt.title('IPL Teams by Total Wins')
plt.xlabel('Team')
plt.ylabel('Total Wins')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(data_dir, '../visualizations', 'team_wins.png'))
plt.close()
print("Saved team wins visualization to 'visualizations/team_wins.png'")

# Impact of toss on match outcome
matches_df['toss_win_match_win'] = matches_df['toss_winner'] == matches_df['winner']
toss_impact = matches_df['toss_win_match_win'].mean() * 100

print(f"\nToss winner wins the match {toss_impact:.2f}% of the time")

# ## 5. Player Performance Analysis

# Determine the runs column name
runs_col = 'batting_runs' if 'batting_runs' in players_df.columns else 'total_runs'
player_name_col = 'name' if 'name' in players_df.columns else ('batsman' if 'batsman' in players_df.columns else 'player_id')

# Top batsmen
top_batsmen = players_df.sort_values(runs_col, ascending=False).head(10)
print("\nTop 10 batsmen by runs:")
print(top_batsmen[[player_name_col, runs_col]])

# Save top batsmen chart to PNG
plt.figure(figsize=(12, 6))
plt.bar(top_batsmen[player_name_col], top_batsmen[runs_col])
plt.title('Top 10 IPL Run Scorers')
plt.xlabel('Player')
plt.ylabel('Total Runs')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(data_dir, '../visualizations', 'top_batsmen.png'))
plt.close()
print("Saved top batsmen visualization to 'visualizations/top_batsmen.png'")

# ## 6. Machine Learning Models
# 
# Now, let's train some machine learning models to predict match outcomes.

# Initialize ML models
ml_models = IPLMLModels(data_dir)

# Train match outcome prediction model
ml_models.train_match_predictor()

# Make predictions for a sample match
prediction = ml_models.make_predictions(
    team1="Mumbai Indians", 
    team2="Chennai Super Kings", 
    venue="Stadium 1",
    toss_winner="Mumbai Indians", 
    toss_decision="bat"
)

if prediction:
    print(f"\nMatch Prediction: {prediction['team1']} vs {prediction['team2']}")
    print(f"Predicted winner: {prediction['predicted_winner']}")
    print(f"{prediction['team1']} win probability: {prediction['team1_win_probability']:.2f}")
    print(f"{prediction['team2']} win probability: {prediction['team2_win_probability']:.2f}")

# ## 7. Advanced Analysis
# 
# Let's perform some more advanced analysis.

# Home advantage analysis
if 'venue' in matches_df.columns:
    # This is a simplified analysis - in real data, we would need to know which venue is home for which team
    venue_team_wins = pd.crosstab(matches_df['venue'], matches_df['winner'])
    
    # Heatmap of venues vs teams
    plt.figure(figsize=(14, 10))
    sns.heatmap(venue_team_wins, annot=True, fmt='d', cmap='viridis')
    plt.title('Team Performance by Venue')
    plt.xlabel('Team')
    plt.ylabel('Venue')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, '../visualizations', 'venue_performance.png'))
    plt.close()
    print("Saved venue performance analysis to 'visualizations/venue_performance.png'")

# Win margin analysis
# Create bins for win margins
runs_bins = [0, 10, 20, 30, 50, 100, 200]
wickets_bins = [0, 1, 3, 5, 7, 10]

# Filter for wins by runs and by wickets
runs_wins = matches_df[matches_df['win_by_runs'] > 0]
wickets_wins = matches_df[matches_df['win_by_wickets'] > 0]

# Create histograms
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Wins by runs
ax1.hist(runs_wins['win_by_runs'], bins=runs_bins, color='blue', alpha=0.7)
ax1.set_title('Distribution of Wins by Runs')
ax1.set_xlabel('Runs')
ax1.set_ylabel('Number of Matches')

# Wins by wickets
ax2.hist(wickets_wins['win_by_wickets'], bins=wickets_bins, color='green', alpha=0.7)
ax2.set_title('Distribution of Wins by Wickets')
ax2.set_xlabel('Wickets')
ax2.set_ylabel('Number of Matches')

plt.tight_layout()
plt.savefig(os.path.join(data_dir, '../visualizations', 'win_margins.png'))
plt.close()
print("Saved win margins analysis to 'visualizations/win_margins.png'")

print("\nAnalysis complete! Check the visualizations folder for saved plots.") 