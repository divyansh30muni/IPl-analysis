import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class IPLVisualizer:
    def __init__(self, data_dir='data'):
        self.data_dir = data_dir
        self.processed_dir = os.path.join(data_dir, 'processed')
        self.viz_dir = os.path.join(data_dir, '../visualizations')
        
        # Create visualization directory if it doesn't exist
        if not os.path.exists(self.viz_dir):
            os.makedirs(self.viz_dir)
            
        # Set default style for matplotlib
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette('viridis')
    
    def load_data(self):
        """Load all available processed data"""
        data = {}
        
        # Try to load match data
        matches_file = os.path.join(self.processed_dir, 'matches.csv')
        if not os.path.exists(matches_file):
            matches_file = os.path.join(self.processed_dir, 'sample_matches.csv')
        
        if os.path.exists(matches_file):
            data['matches'] = pd.read_csv(matches_file)
            print(f"Loaded matches data: {len(data['matches'])} rows")
        
        # Try to load player data
        batting_file = os.path.join(self.processed_dir, 'player_batting_stats.csv')
        if not os.path.exists(batting_file):
            batting_file = os.path.join(self.processed_dir, 'sample_players.csv')
            
        if os.path.exists(batting_file):
            data['players'] = pd.read_csv(batting_file)
            print(f"Loaded player data: {len(data['players'])} rows")
            
        # Load bowling stats if available
        bowling_file = os.path.join(self.processed_dir, 'player_bowling_stats.csv')
        if os.path.exists(bowling_file):
            data['bowling'] = pd.read_csv(bowling_file)
            print(f"Loaded bowling data: {len(data['bowling'])} rows")
        
        # Load feature importance data if available
        importance_file = os.path.join(self.processed_dir, 'feature_importance.csv')
        if os.path.exists(importance_file):
            data['feature_importance'] = pd.read_csv(importance_file)
            print(f"Loaded feature importance data")
            
        return data
        
    def team_performance_analysis(self, save_fig=True):
        """Analyze and visualize team performances"""
        print("Analyzing team performances...")
        
        data = self.load_data()
        if 'matches' not in data:
            print("No match data available for team performance analysis")
            return False
            
        matches_df = data['matches']
        
        # Calculate team win counts
        team_wins = {}
        
        # Count wins for each team
        for _, match in matches_df.iterrows():
            if match['winner'] not in team_wins:
                team_wins[match['winner']] = 0
            
            team_wins[match['winner']] += 1
        
        # Convert to DataFrame for visualization
        team_performance = pd.DataFrame({
            'team': list(team_wins.keys()),
            'wins': list(team_wins.values())
        }).sort_values('wins', ascending=False)
        
        # Visualize team performances using Plotly
        fig = px.bar(
            team_performance,
            x='team',
            y='wins',
            title='IPL Teams by Total Wins',
            color='wins',
            color_continuous_scale='viridis',
            labels={'team': 'Team', 'wins': 'Total Wins'}
        )
        
        # Customize layout
        fig.update_layout(
            xaxis_tickangle=-45,
            plot_bgcolor='white',
            height=600
        )
        
        # Save interactive plot
        if save_fig:
            fig.write_html(os.path.join(self.viz_dir, 'team_wins.html'))
            print(f"Saved team performance visualization to {self.viz_dir}/team_wins.html")
        
        # Display plot
        fig.show()
        
        # Analyze win types (by runs vs by wickets)
        # Create a new DataFrame for win types
        win_types = []
        for _, match in matches_df.iterrows():
            if match['win_by_runs'] > 0:
                win_type = 'by runs'
                margin = match['win_by_runs']
            elif match['win_by_wickets'] > 0:
                win_type = 'by wickets'
                margin = match['win_by_wickets']
            else:
                # Skip if no winner or tie/draw
                continue
                
            win_types.append({
                'team': match['winner'],
                'win_type': win_type,
                'margin': margin
            })
            
        if win_types:
            win_types_df = pd.DataFrame(win_types)
            
            # Count win types by team
            win_type_counts = win_types_df.groupby(['team', 'win_type']).size().unstack(fill_value=0)
            
            # Visualize win types
            fig = px.bar(
                win_type_counts, 
                title='IPL Teams by Win Types',
                barmode='group',
                labels={'value': 'Number of Wins', 'team': 'Team', 'win_type': 'Win Type'}
            )
            
            # Save interactive plot
            if save_fig:
                fig.write_html(os.path.join(self.viz_dir, 'win_types.html'))
                print(f"Saved win types visualization to {self.viz_dir}/win_types.html")
            
            # Display plot
            fig.show()
        
        return True
    
    def player_performance_analysis(self, top_n=10, save_fig=True):
        """Analyze and visualize top player performances"""
        print("Analyzing player performances...")
        
        data = self.load_data()
        if 'players' not in data:
            print("No player data available for analysis")
            return False
            
        players_df = data['players']
        
        # Different column names depending on data source
        if 'batting_runs' in players_df.columns:
            runs_col = 'batting_runs'
            player_name_col = 'name' if 'name' in players_df.columns else 'player_id'
        elif 'total_runs' in players_df.columns:
            runs_col = 'total_runs'
            player_name_col = 'batsman' if 'batsman' in players_df.columns else 'name'
        else:
            print("No runs data found in player data")
            return False
        
        # Top run scorers
        top_batsmen = players_df.sort_values(runs_col, ascending=False).head(top_n)
        
        # Plot top batsmen
        fig = px.bar(
            top_batsmen,
            x=player_name_col, 
            y=runs_col,
            title=f'Top {top_n} IPL Run Scorers',
            color=runs_col,
            color_continuous_scale='viridis',
            labels={player_name_col: 'Player', runs_col: 'Total Runs'}
        )
        
        fig.update_layout(
            xaxis_tickangle=-45,
            plot_bgcolor='white',
            height=600
        )
        
        # Save interactive plot
        if save_fig:
            fig.write_html(os.path.join(self.viz_dir, 'top_batsmen.html'))
            print(f"Saved top batsmen visualization to {self.viz_dir}/top_batsmen.html")
        
        # Display plot
        fig.show()
        
        # Batting strike rate analysis if available
        if 'strike_rate' in players_df.columns:
            # Filter to players with significant runs
            significant_runs = players_df[players_df[runs_col] > 500]
            
            # Create scatter plot of runs vs strike rate
            fig = px.scatter(
                significant_runs,
                x=runs_col,
                y='strike_rate',
                title='IPL Batting: Runs vs Strike Rate',
                size=runs_col,
                color='team' if 'team' in players_df.columns else None,
                hover_name=player_name_col,
                labels={runs_col: 'Total Runs', 'strike_rate': 'Strike Rate'}
            )
            
            fig.update_layout(
                plot_bgcolor='white',
                height=600
            )
            
            if save_fig:
                fig.write_html(os.path.join(self.viz_dir, 'batting_strike_rate.html'))
                print(f"Saved strike rate analysis to {self.viz_dir}/batting_strike_rate.html")
            
            fig.show()
        
        # Bowling analysis if available
        if 'bowling' in data:
            bowling_df = data['bowling']
            
            if 'wickets' in bowling_df.columns:
                # Top wicket takers
                top_bowlers = bowling_df.sort_values('wickets', ascending=False).head(top_n)
                
                # Get bowler name column
                bowler_name_col = 'bowler' if 'bowler' in bowling_df.columns else 'player_id'
                
                fig = px.bar(
                    top_bowlers,
                    x=bowler_name_col,
                    y='wickets',
                    title=f'Top {top_n} IPL Wicket Takers',
                    color='wickets',
                    color_continuous_scale='viridis',
                    labels={bowler_name_col: 'Bowler', 'wickets': 'Total Wickets'}
                )
                
                fig.update_layout(
                    xaxis_tickangle=-45,
                    plot_bgcolor='white',
                    height=600
                )
                
                if save_fig:
                    fig.write_html(os.path.join(self.viz_dir, 'top_bowlers.html'))
                    print(f"Saved top bowlers visualization to {self.viz_dir}/top_bowlers.html")
                
                fig.show()
                
                # Economy rate analysis if available
                if 'economy_rate' in bowling_df.columns:
                    # Filter to bowlers with significant wickets
                    significant_wickets = bowling_df[bowling_df['wickets'] > 20]
                    
                    fig = px.scatter(
                        significant_wickets,
                        x='wickets',
                        y='economy_rate',
                        title='IPL Bowling: Wickets vs Economy Rate',
                        size='wickets',
                        color='team' if 'team' in bowling_df.columns else None,
                        hover_name=bowler_name_col,
                        labels={'wickets': 'Total Wickets', 'economy_rate': 'Economy Rate'}
                    )
                    
                    fig.update_layout(
                        plot_bgcolor='white',
                        height=600
                    )
                    
                    if save_fig:
                        fig.write_html(os.path.join(self.viz_dir, 'bowling_economy.html'))
                        print(f"Saved bowling economy analysis to {self.viz_dir}/bowling_economy.html")
                    
                    fig.show()
        
        return True
    
    def match_factors_analysis(self, save_fig=True):
        """Analyze factors affecting match outcomes"""
        print("Analyzing match factors...")
        
        data = self.load_data()
        if 'matches' not in data:
            print("No match data available for analysis")
            return False
            
        matches_df = data['matches']
        
        # Toss decision analysis
        toss_decisions = matches_df.groupby(['toss_decision', 'winner']).size().unstack(fill_value=0)
        
        # Check if we have toss data
        if toss_decisions.shape[0] > 0:
            fig = px.bar(
                toss_decisions,
                title='Impact of Toss Decision on Match Outcome',
                barmode='group',
                labels={'value': 'Number of Matches', 'toss_decision': 'Toss Decision', 'winner': 'Team'}
            )
            
            fig.update_layout(
                plot_bgcolor='white',
                height=600
            )
            
            if save_fig:
                fig.write_html(os.path.join(self.viz_dir, 'toss_impact.html'))
                print(f"Saved toss impact analysis to {self.viz_dir}/toss_impact.html")
            
            fig.show()
            
            # Calculate toss winner advantage
            matches_df['toss_winner_won_match'] = matches_df['toss_winner'] == matches_df['winner']
            toss_advantage = matches_df['toss_winner_won_match'].mean() * 100
            
            print(f"Toss winner wins the match {toss_advantage:.2f}% of the time")
        
        # Venue analysis if we have enough data
        if 'venue' in matches_df.columns:
            venue_wins = matches_df.groupby('venue').size()
            top_venues = venue_wins.sort_values(ascending=False).head(10).index.tolist()
            
            # Filter to top venues for cleaner visualization
            venue_results = matches_df[matches_df['venue'].isin(top_venues)]
            
            # Count wins by team at each venue
            venue_team_wins = pd.crosstab(venue_results['venue'], venue_results['winner'])
            
            # Create heatmap
            plt.figure(figsize=(16, 10))
            sns.heatmap(venue_team_wins, cmap='viridis', annot=True, fmt='d', linewidths=.5)
            plt.title('Team Performances by Venue')
            plt.xlabel('Team')
            plt.ylabel('Venue')
            
            if save_fig:
                plt.savefig(os.path.join(self.viz_dir, 'venue_performance.png'), dpi=300, bbox_inches='tight')
                print(f"Saved venue performance analysis to {self.viz_dir}/venue_performance.png")
                
            plt.close()
        
        # Feature importance visualization if available
        if 'feature_importance' in data:
            feat_imp = data['feature_importance']
            
            # Get top features
            top_features = feat_imp.sort_values('importance', ascending=False).head(15)
            
            fig = px.bar(
                top_features,
                x='importance',
                y='feature',
                title='Top Features for Match Outcome Prediction',
                orientation='h',
                color='importance',
                color_continuous_scale='viridis'
            )
            
            fig.update_layout(
                plot_bgcolor='white',
                height=600
            )
            
            if save_fig:
                fig.write_html(os.path.join(self.viz_dir, 'feature_importance.html'))
                print(f"Saved feature importance visualization to {self.viz_dir}/feature_importance.html")
            
            fig.show()
        
        return True

if __name__ == "__main__":
    visualizer = IPLVisualizer()
    
    # Generate all visualizations
    visualizer.team_performance_analysis()
    visualizer.player_performance_analysis()
    visualizer.match_factors_analysis() 