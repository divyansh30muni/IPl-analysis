import os
import requests
import zipfile
import json
import pandas as pd
from bs4 import BeautifulSoup
import time

class IPLDataCollector:
    def __init__(self, data_dir='data'):
        self.data_dir = data_dir
        self.raw_dir = os.path.join(data_dir, 'raw')
        self.processed_dir = os.path.join(data_dir, 'processed')
        
        # Ensure directories exist
        for directory in [self.raw_dir, self.processed_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)
    
    def download_cricsheet_data(self):
        """Download IPL data from Cricsheet"""
        print("Downloading IPL data from Cricsheet...")
        
        # URL for IPL matches in JSON format
        url = "https://cricsheet.org/downloads/ipl_json.zip"
        zip_path = os.path.join(self.raw_dir, "ipl_json.zip")
        
        # Download the ZIP file
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(zip_path, 'wb') as f:
                f.write(response.content)
            
            # Extract the ZIP file
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(os.path.join(self.raw_dir, 'cricsheet'))
            
            print(f"Downloaded and extracted IPL data to {self.raw_dir}")
            return True
        else:
            print(f"Failed to download data: {response.status_code}")
            return False
    
    def scrape_player_stats(self):
        """Scrape player statistics from IPLT20 website"""
        print("Scraping player statistics...")
        
        # This is a simplified demonstration - in a real scenario, you would 
        # need to handle pagination, error cases, and respect rate limits
        url = "https://www.iplt20.com/stats/all-time"
        
        try:
            response = requests.get(url)
            if response.status_code == 200:
                # Save the raw HTML for later processing
                with open(os.path.join(self.raw_dir, "player_stats.html"), "w", encoding="utf-8") as f:
                    f.write(response.text)
                print("Saved player stats HTML for parsing")
                
                # In a real implementation, you'd parse the HTML here with BeautifulSoup
                # But many sports sites have complex JS rendering, so often need Selenium
                return True
            else:
                print(f"Failed to scrape player stats: {response.status_code}")
                return False
        except Exception as e:
            print(f"Error scraping player stats: {str(e)}")
            return False
            
    def create_sample_data(self):
        """Creates sample data for development if real data collection fails"""
        print("Creating sample IPL data for development...")
        
        # Sample teams
        teams = ["Mumbai Indians", "Chennai Super Kings", "Royal Challengers Bangalore", 
                 "Kolkata Knight Riders", "Delhi Capitals", "Sunrisers Hyderabad",
                 "Rajasthan Royals", "Punjab Kings"]
        
        # Sample matches data
        matches_data = []
        for i in range(100):
            team1_idx = i % 8
            team2_idx = (i + 1) % 8
            
            winner = teams[team1_idx] if i % 3 != 0 else teams[team2_idx]
            
            matches_data.append({
                'match_id': 1000 + i,
                'season': 2016 + (i % 6),
                'date': f"2021-04-{(i % 30) + 1:02d}",
                'venue': f"Stadium {i % 10}",
                'team1': teams[team1_idx],
                'team2': teams[team2_idx],
                'toss_winner': teams[team1_idx] if i % 2 == 0 else teams[team2_idx],
                'toss_decision': 'bat' if i % 2 == 0 else 'field',
                'winner': winner,
                'win_by_runs': i % 20 if winner == teams[team1_idx] else 0,
                'win_by_wickets': i % 10 if winner == teams[team2_idx] else 0
            })
        
        # Sample player data
        players_data = []
        player_names = [
            "Virat Kohli", "MS Dhoni", "Rohit Sharma", "Jasprit Bumrah", 
            "AB de Villiers", "Suresh Raina", "Ravindra Jadeja", "David Warner",
            "Kane Williamson", "Jos Buttler", "Ben Stokes", "Andre Russell"
        ]
        
        for i, name in enumerate(player_names):
            players_data.append({
                'player_id': i + 1,
                'name': name,
                'team': teams[i % 8],
                'batting_runs': 2000 + (i * 150),
                'batting_average': 25 + (i * 2.5),
                'strike_rate': 120 + (i * 3),
                'centuries': i % 5,
                'fifties': 10 + (i % 15),
                'wickets': 20 + (i * 10) if i % 3 == 0 else i * 2,
                'bowling_average': 25 + (i * 1.2) if i % 3 == 0 else 40 + i,
                'economy_rate': 7.5 + (i * 0.2) if i % 3 == 0 else 8.5 + (i * 0.3)
            })
        
        # Convert to DataFrames and save as CSV
        matches_df = pd.DataFrame(matches_data)
        players_df = pd.DataFrame(players_data)
        
        matches_df.to_csv(os.path.join(self.processed_dir, "sample_matches.csv"), index=False)
        players_df.to_csv(os.path.join(self.processed_dir, "sample_players.csv"), index=False)
        
        print(f"Created sample data files in {self.processed_dir}")
        return True

if __name__ == "__main__":
    collector = IPLDataCollector()
    
    # Try to download real data
    cricsheet_success = collector.download_cricsheet_data()
    player_stats_success = collector.scrape_player_stats()
    
    # If either data collection fails, create sample data for development
    if not cricsheet_success or not player_stats_success:
        collector.create_sample_data() 