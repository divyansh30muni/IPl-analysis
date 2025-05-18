import os
import pandas as pd
import numpy as np
import json
import glob
from datetime import datetime

class IPLDataProcessor:
    def __init__(self, data_dir='data'):
        self.data_dir = data_dir
        self.raw_dir = os.path.join(data_dir, 'raw')
        self.processed_dir = os.path.join(data_dir, 'processed')
        
        # Ensure output directory exists
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)
    
    def process_cricsheet_data(self):
        """Process Cricsheet JSON data into analysis-ready DataFrames"""
        print("Processing Cricsheet JSON data...")
        
        json_dir = os.path.join(self.raw_dir, 'cricsheet')
        if not os.path.exists(json_dir):
            print(f"Cricsheet data directory not found at {json_dir}")
            return False
            
        # List all JSON files
        json_files = glob.glob(os.path.join(json_dir, '*.json'))
        if not json_files:
            print("No JSON files found in Cricsheet data directory")
            return False
            
        # Process match data
        matches_data = []
        batting_data = []
        bowling_data = []
        
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    match_data = json.load(f)
                
                # Basic match information
                match_info = match_data.get('info', {})
                
                match_id = os.path.basename(json_file).replace('.json', '')
                date = match_info.get('dates', [''])[0]
                venue = match_info.get('venue', '')
                teams = match_info.get('teams', [])
                toss = match_info.get('toss', {})
                outcome = match_info.get('outcome', {})
                
                if len(teams) < 2:
                    continue  # Skip incomplete data
                    
                match_row = {
                    'match_id': match_id,
                    'date': date,
                    'venue': venue,
                    'team1': teams[0],
                    'team2': teams[1],
                    'toss_winner': toss.get('winner', ''),
                    'toss_decision': toss.get('decision', ''),
                    'winner': outcome.get('winner', ''),
                    'win_by_runs': outcome.get('by', {}).get('runs', 0),
                    'win_by_wickets': outcome.get('by', {}).get('wickets', 0)
                }
                
                matches_data.append(match_row)
                
                # Process innings data if available
                innings = match_data.get('innings', [])
                for inning_idx, inning in enumerate(innings):
                    team_batting = inning.get('team', '')
                    team_bowling = teams[0] if team_batting == teams[1] else teams[1]
                    
                    # Process batting performances
                    for delivery in inning.get('overs', []):
                        for ball in delivery.get('deliveries', []):
                            batsman = ball.get('batter', '')
                            runs = ball.get('runs', {}).get('batter', 0)
                            
                            batting_data.append({
                                'match_id': match_id,
                                'date': date,
                                'team': team_batting,
                                'batsman': batsman,
                                'runs': runs,
                                'is_boundary': 1 if runs >= 4 else 0,
                                'is_six': 1 if runs == 6 else 0
                            })
                            
                            # Process bowling data
                            bowler = ball.get('bowler', '')
                            wicket = ball.get('wicket', None)
                            
                            bowling_data.append({
                                'match_id': match_id,
                                'date': date,
                                'team': team_bowling,
                                'bowler': bowler,
                                'runs_conceded': runs,
                                'wicket': 1 if wicket else 0,
                                'wicket_type': wicket.get('kind', '') if wicket else ''
                            })
            except Exception as e:
                print(f"Error processing {json_file}: {str(e)}")
                continue
        
        # Convert to DataFrames
        if matches_data:
            matches_df = pd.DataFrame(matches_data)
            batting_df = pd.DataFrame(batting_data)
            bowling_df = pd.DataFrame(bowling_data)
            
            # Save processed data
            matches_df.to_csv(os.path.join(self.processed_dir, 'matches.csv'), index=False)
            batting_df.to_csv(os.path.join(self.processed_dir, 'batting.csv'), index=False)
            bowling_df.to_csv(os.path.join(self.processed_dir, 'bowling.csv'), index=False)
            
            print(f"Processed {len(matches_data)} matches")
            return True
        else:
            print("No match data processed")
            return False
    
    def generate_player_stats(self):
        """Generate aggregated player statistics from batting and bowling data"""
        print("Generating player statistics...")
        
        batting_file = os.path.join(self.processed_dir, 'batting.csv')
        bowling_file = os.path.join(self.processed_dir, 'bowling.csv')
        
        if not (os.path.exists(batting_file) and os.path.exists(bowling_file)):
            # Try to use sample data if available
            batting_file = os.path.join(self.processed_dir, 'sample_players.csv')
            if os.path.exists(batting_file):
                print("Using sample player data")
                return True
            print("Required processed data files not found")
            return False
        
        try:
            # Read processed data
            batting_df = pd.read_csv(batting_file)
            bowling_df = pd.read_csv(bowling_file)
            
            # Aggregate batting statistics
            batting_stats = batting_df.groupby(['batsman', 'team']).agg(
                innings=('match_id', 'nunique'),
                total_runs=('runs', 'sum'),
                balls_faced=('batsman', 'count'),
                boundaries=('is_boundary', 'sum'),
                sixes=('is_six', 'sum')
            ).reset_index()
            
            batting_stats['strike_rate'] = (batting_stats['total_runs'] / batting_stats['balls_faced']) * 100
            
            # Aggregate bowling statistics
            bowling_stats = bowling_df.groupby(['bowler', 'team']).agg(
                matches=('match_id', 'nunique'),
                runs_conceded=('runs_conceded', 'sum'),
                wickets=('wicket', 'sum'),
                balls_bowled=('bowler', 'count')
            ).reset_index()
            
            bowling_stats['economy_rate'] = (bowling_stats['runs_conceded'] / (bowling_stats['balls_bowled'] / 6))
            bowling_stats['bowling_average'] = bowling_stats['runs_conceded'] / bowling_stats['wickets']
            
            # Save player statistics
            batting_stats.to_csv(os.path.join(self.processed_dir, 'player_batting_stats.csv'), index=False)
            bowling_stats.to_csv(os.path.join(self.processed_dir, 'player_bowling_stats.csv'), index=False)
            
            print("Generated player statistics successfully")
            return True
        except Exception as e:
            print(f"Error generating player statistics: {str(e)}")
            return False
    
    def prepare_features_for_ml(self):
        """Prepare features for machine learning models"""
        print("Preparing features for ML models...")
        
        matches_file = os.path.join(self.processed_dir, 'matches.csv')
        
        # If real data not available, use sample data
        if not os.path.exists(matches_file):
            matches_file = os.path.join(self.processed_dir, 'sample_matches.csv')
            if not os.path.exists(matches_file):
                print("No match data available for feature preparation")
                return False
        
        try:
            matches_df = pd.read_csv(matches_file)
            
            # Extract season info if available, otherwise use year from date
            if 'season' not in matches_df.columns:
                matches_df['season'] = pd.to_datetime(matches_df['date']).dt.year
                
            # One-hot encode categorical features
            features_df = pd.get_dummies(matches_df, columns=['venue', 'toss_decision'])
            
            # Add team strength features (win percentage, recent form)
            team_stats = {}
            
            for _, match in matches_df.iterrows():
                # Parse date if it's a string
                if isinstance(match['date'], str):
                    match_date = datetime.strptime(match['date'], '%Y-%m-%d')
                else:
                    match_date = match['date']
                    
                team1, team2 = match['team1'], match['team2']
                winner = match['winner']
                
                # Initialize team stats if needed
                for team in [team1, team2]:
                    if team not in team_stats:
                        team_stats[team] = {
                            'matches': 0,
                            'wins': 0,
                            'recent_form': []  # 1 for win, 0 for loss
                        }
                
                # Update stats
                team_stats[team1]['matches'] += 1
                team_stats[team2]['matches'] += 1
                
                if winner == team1:
                    team_stats[team1]['wins'] += 1
                    team_stats[team1]['recent_form'].append(1)
                    team_stats[team2]['recent_form'].append(0)
                elif winner == team2:
                    team_stats[team2]['wins'] += 1
                    team_stats[team2]['recent_form'].append(1)
                    team_stats[team1]['recent_form'].append(0)
                
                # Keep only last 5 matches for recent form
                for team in [team1, team2]:
                    team_stats[team]['recent_form'] = team_stats[team]['recent_form'][-5:]
            
            # Add team stats to features
            for idx, match in features_df.iterrows():
                team1, team2 = match['team1'], match['team2']
                
                if team1 in team_stats and team_stats[team1]['matches'] > 0:
                    features_df.at[idx, 'team1_win_pct'] = team_stats[team1]['wins'] / team_stats[team1]['matches']
                    features_df.at[idx, 'team1_form'] = sum(team_stats[team1]['recent_form'][-5:]) / len(team_stats[team1]['recent_form']) if team_stats[team1]['recent_form'] else 0
                else:
                    features_df.at[idx, 'team1_win_pct'] = 0.5
                    features_df.at[idx, 'team1_form'] = 0.5
                
                if team2 in team_stats and team_stats[team2]['matches'] > 0:
                    features_df.at[idx, 'team2_win_pct'] = team_stats[team2]['wins'] / team_stats[team2]['matches']
                    features_df.at[idx, 'team2_form'] = sum(team_stats[team2]['recent_form'][-5:]) / len(team_stats[team2]['recent_form']) if team_stats[team2]['recent_form'] else 0
                else:
                    features_df.at[idx, 'team2_win_pct'] = 0.5
                    features_df.at[idx, 'team2_form'] = 0.5
            
            # Save features for ML
            features_df.to_csv(os.path.join(self.processed_dir, 'ml_features.csv'), index=False)
            
            print("Prepared features for ML models")
            return True
        except Exception as e:
            print(f"Error preparing features: {str(e)}")
            return False

if __name__ == "__main__":
    processor = IPLDataProcessor()
    
    # Process data
    processor.process_cricsheet_data()
    processor.generate_player_stats()
    processor.prepare_features_for_ml() 