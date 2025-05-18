import os
import argparse
import pandas as pd
from scripts.data_collection import IPLDataCollector
from scripts.data_preprocessing import IPLDataProcessor
from scripts.ml_models import IPLMLModels
from scripts.visualization import IPLVisualizer

def main():
    """Run the IPL analysis pipeline"""
    parser = argparse.ArgumentParser(description='IPL Cricket Analysis with AI/ML')
    parser.add_argument('--collect', action='store_true', help='Collect data from sources')
    parser.add_argument('--process', action='store_true', help='Process raw data')
    parser.add_argument('--train', action='store_true', help='Train ML models')
    parser.add_argument('--visualize', action='store_true', help='Generate visualizations')
    parser.add_argument('--predict', nargs='+', help='Predict match outcome. Format: "team1,team2,venue[,toss_winner,toss_decision]"')
    parser.add_argument('--all', action='store_true', help='Run the complete pipeline')
    
    args = parser.parse_args()
    
    # If no args provided, show help
    if not any(vars(args).values()):
        parser.print_help()
        return
    
    # Run complete pipeline if --all is specified
    if args.all:
        args.collect = args.process = args.train = args.visualize = True
    
    # Setup directory structure
    data_dir = 'data'
    os.makedirs(data_dir, exist_ok=True)
    
    print("=" * 50)
    print("IPL Cricket Analysis with AI/ML")
    print("=" * 50)
    
    # Data Collection
    if args.collect:
        print("\n== Data Collection ==")
        collector = IPLDataCollector(data_dir)
        cricsheet_success = collector.download_cricsheet_data()
        player_stats_success = collector.scrape_player_stats()
        
        # If real data collection fails, create sample data
        if not cricsheet_success or not player_stats_success:
            print("Creating sample data for development...")
            collector.create_sample_data()
    
    # Data Processing
    if args.process:
        print("\n== Data Processing ==")
        processor = IPLDataProcessor(data_dir)
        processor.process_cricsheet_data()
        processor.generate_player_stats()
        processor.prepare_features_for_ml()
    
    # Model Training
    if args.train:
        print("\n== Model Training ==")
        ml_models = IPLMLModels(data_dir)
        ml_models.train_match_predictor()
        ml_models.player_performance_prediction()
    
    # Generate Visualizations
    if args.visualize:
        print("\n== Generating Visualizations ==")
        visualizer = IPLVisualizer(data_dir)
        visualizer.team_performance_analysis()
        visualizer.player_performance_analysis()
        visualizer.match_factors_analysis()
    
    # Make Predictions
    if args.predict:
        print("\n== Match Prediction ==")
        ml_models = IPLMLModels(data_dir)
        
        for match_str in args.predict:
            match_params = match_str.split(',')
            
            if len(match_params) < 3:
                print(f"Invalid prediction format: {match_str}")
                print("Format should be: 'team1,team2,venue[,toss_winner,toss_decision]'")
                continue
                
            team1 = match_params[0]
            team2 = match_params[1]
            venue = match_params[2]
            
            toss_winner = match_params[3] if len(match_params) > 3 else None
            toss_decision = match_params[4] if len(match_params) > 4 else None
            
            prediction = ml_models.make_predictions(
                team1=team1,
                team2=team2,
                venue=venue,
                toss_winner=toss_winner,
                toss_decision=toss_decision
            )
            
            if prediction:
                print(f"\nMatch Prediction: {prediction['team1']} vs {prediction['team2']}")
                print(f"Predicted winner: {prediction['predicted_winner']}")
                print(f"{prediction['team1']} win probability: {prediction['team1_win_probability']:.2f}")
                print(f"{prediction['team2']} win probability: {prediction['team2_win_probability']:.2f}")
            else:
                print(f"Could not make prediction for {team1} vs {team2}. Train models first.")
    
    print("\nIPL Analysis Complete!")

if __name__ == "__main__":
    main() 