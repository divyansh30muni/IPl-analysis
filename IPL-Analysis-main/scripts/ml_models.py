import os
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier

class IPLMLModels:
    def __init__(self, data_dir='data'):
        self.data_dir = data_dir
        self.processed_dir = os.path.join(data_dir, 'processed')
        self.models_dir = os.path.join(data_dir, '../models')
        
        # Ensure models directory exists
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)
            
    def load_data(self):
        """Load processed data for model training"""
        features_file = os.path.join(self.processed_dir, 'ml_features.csv')
        
        if not os.path.exists(features_file):
            # Try sample data if real data not available
            features_file = os.path.join(self.processed_dir, 'sample_matches.csv')
            if not os.path.exists(features_file):
                print("No data available for model training")
                return None
                
        df = pd.read_csv(features_file)
        print(f"Loaded {len(df)} rows for model training")
        return df
    
    def preprocess_match_outcome_data(self, df):
        """Preprocess data for match outcome prediction"""
        print("Preprocessing data for match outcome prediction")
        
        # Create target variable (1 if team1 wins, 0 if team2 wins)
        df['team1_win'] = df.apply(lambda row: 1 if row['winner'] == row['team1'] else 0, axis=1)
        
        # Select and prepare features
        # We need to be careful about information leakage
        feature_cols = [col for col in df.columns if not col.startswith('win_by') 
                       and col not in ['winner', 'team1_win', 'match_id', 'date']]
        
        # Handle categorical features
        # Team names need special handling - convert to team strength metrics
        # If we just used team names as categorical, the model would just memorize team performance
        for team_col in ['team1', 'team2', 'toss_winner']:
            if team_col in feature_cols:
                feature_cols.remove(team_col)
        
        # Clean and prepare features
        X = df[feature_cols].copy()
        y = df['team1_win']
        
        # Convert any remaining categorical columns
        cat_columns = X.select_dtypes(include=['object']).columns
        for col in cat_columns:
            label_encoder = LabelEncoder()
            X[col] = label_encoder.fit_transform(X[col].astype(str))
        
        # Handle missing values
        X = X.fillna(0)
        
        return X, y
        
    def train_match_predictor(self):
        """Train model to predict match outcomes"""
        print("Training match outcome prediction model...")
        
        df = self.load_data()
        if df is None:
            return False
            
        X, y = self.preprocess_match_outcome_data(df)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale numeric features
        scaler = StandardScaler()
        numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
        X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
        X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])
        
        # Train models
        models = {
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingClassifier(random_state=42),
            'logistic_regression': LogisticRegression(max_iter=1000, random_state=42),
            'xgboost': XGBClassifier(random_state=42)
        }
        
        best_accuracy = 0
        best_model_name = ''
        best_model = None
        
        for name, model in models.items():
            print(f"Training {name}...")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            print(f"{name} accuracy: {accuracy:.4f}")
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model_name = name
                best_model = model
        
        print(f"Best model: {best_model_name} with accuracy {best_accuracy:.4f}")
        
        # Save best model
        model_file = os.path.join(self.models_dir, 'match_predictor.pkl')
        with open(model_file, 'wb') as f:
            pickle.dump({
                'model': best_model,
                'scaler': scaler,
                'features': X.columns.tolist(),
                'name': best_model_name,
                'accuracy': best_accuracy
            }, f)
        
        # Detailed evaluation of best model
        y_pred = best_model.predict(X_test)
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Feature importance (if applicable)
        if hasattr(best_model, 'feature_importances_'):
            importances = best_model.feature_importances_
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            print("\nTop 10 most important features:")
            print(feature_importance.head(10))
            
            # Save feature importance
            feature_importance.to_csv(os.path.join(self.processed_dir, 'feature_importance.csv'), index=False)
        
        return True
    
    def player_performance_prediction(self):
        """Train model to predict player performance"""
        print("Training player performance prediction model...")
        
        # For player performance, we need different features
        batting_file = os.path.join(self.processed_dir, 'player_batting_stats.csv')
        if not os.path.exists(batting_file):
            # Try sample data
            players_file = os.path.join(self.processed_dir, 'sample_players.csv')
            if not os.path.exists(players_file):
                print("No player data available for model training")
                return False
            
            batting_df = pd.read_csv(players_file)
        else:
            batting_df = pd.read_csv(batting_file)
        
        # Simple model to predict runs
        # In a real scenario, we'd use more features and historical data
        if 'batting_runs' in batting_df.columns:
            target = 'batting_runs'
        elif 'total_runs' in batting_df.columns:
            target = 'total_runs'
        else:
            print("No suitable target found for player performance prediction")
            return False
        
        # Prepare features
        feature_cols = [col for col in batting_df.columns if col != target 
                       and col not in ['name', 'player_id', 'batsman']]
        
        # Handle categorical features
        X = batting_df[feature_cols].copy()
        y = batting_df[target]
        
        # Convert any categorical columns
        cat_columns = X.select_dtypes(include=['object']).columns
        for col in cat_columns:
            label_encoder = LabelEncoder()
            X[col] = label_encoder.fit_transform(X[col].astype(str))
        
        # Fill missing values
        X = X.fillna(0)
        
        # Train a simple model
        if len(X) > 20:  # Only if we have enough data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            model = GradientBoostingClassifier(random_state=42)
            model.fit(X_train, y_train)
            
            # Save model
            model_file = os.path.join(self.models_dir, 'player_performance_predictor.pkl')
            with open(model_file, 'wb') as f:
                pickle.dump({
                    'model': model,
                    'features': X.columns.tolist(),
                    'target': target
                }, f)
            
            print(f"Trained player performance prediction model")
            return True
        else:
            print("Not enough data for player performance model")
            return False
    
    def make_predictions(self, team1, team2, venue, toss_winner=None, toss_decision=None):
        """Make match outcome prediction using the trained model"""
        model_file = os.path.join(self.models_dir, 'match_predictor.pkl')
        
        if not os.path.exists(model_file):
            print("No trained model found. Please train the model first.")
            return None
        
        # Load model and metadata
        with open(model_file, 'rb') as f:
            model_data = pickle.load(f)
        
        model = model_data['model']
        scaler = model_data['scaler']
        features = model_data['features']
        
        # Need to build a feature vector similar to training data
        # This is simplified - in a real implementation, you would need to
        # calculate all the same features used in training
        
        # Get team stats from historical data
        matches_file = os.path.join(self.processed_dir, 'ml_features.csv')
        if os.path.exists(matches_file):
            matches_df = pd.read_csv(matches_file)
            
            # Get team win percentages and form
            team1_stats = matches_df[matches_df['team1'] == team1]
            team1_win_pct = team1_stats['team1_win_pct'].mean() if len(team1_stats) > 0 else 0.5
            team1_form = team1_stats['team1_form'].mean() if len(team1_stats) > 0 else 0.5
            
            team2_stats = matches_df[matches_df['team2'] == team2]
            team2_win_pct = team2_stats['team2_win_pct'].mean() if len(team2_stats) > 0 else 0.5
            team2_form = team2_stats['team2_form'].mean() if len(team2_stats) > 0 else 0.5
        else:
            # Default values if no historical data
            team1_win_pct, team1_form = 0.5, 0.5
            team2_win_pct, team2_form = 0.5, 0.5
        
        # Create a sample feature vector (simplified)
        # In real implementation, this would need to match the training features exactly
        sample = {
            'team1_win_pct': team1_win_pct,
            'team1_form': team1_form,
            'team2_win_pct': team2_win_pct,
            'team2_form': team2_form,
        }
        
        # Add venue encoding
        for feature in features:
            if feature.startswith('venue_') and venue in feature:
                sample[feature] = 1
            elif feature.startswith('venue_'):
                sample[feature] = 0
        
        # Add toss info if provided
        if toss_winner and toss_decision:
            for feature in features:
                if feature.startswith('toss_decision_') and toss_decision in feature:
                    sample[feature] = 1
                elif feature.startswith('toss_decision_'):
                    sample[feature] = 0
        
        # Convert to DataFrame with the right structure
        sample_df = pd.DataFrame([sample])
        
        # Ensure all features are present
        for feature in features:
            if feature not in sample_df.columns:
                sample_df[feature] = 0
        
        # Scale numeric features
        numeric_cols = sample_df.select_dtypes(include=['int64', 'float64']).columns
        sample_df[numeric_cols] = scaler.transform(sample_df[numeric_cols])
        
        # Make prediction
        prediction = model.predict_proba(sample_df)[0]
        
        result = {
            'team1': team1,
            'team2': team2,
            'team1_win_probability': prediction[1],
            'team2_win_probability': prediction[0],
            'predicted_winner': team1 if prediction[1] > prediction[0] else team2
        }
        
        return result

if __name__ == "__main__":
    ml_models = IPLMLModels()
    
    # Train models
    ml_models.train_match_predictor()
    ml_models.player_performance_prediction()
    
    # Sample prediction
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