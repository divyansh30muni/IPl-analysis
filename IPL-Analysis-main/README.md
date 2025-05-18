# IPL Cricket Analysis with AI/ML

This project provides a comprehensive analysis of Indian Premier League (IPL) cricket data using Python, data science techniques, and machine learning algorithms.

## Features

- **Data Collection**: Scripts to gather IPL match data, player statistics, and team information
- **Exploratory Data Analysis**: Statistical analysis of player performances, team results, and match outcomes
- **Machine Learning Models**: Predict match outcomes, player performances, and identify key performance factors
- **Visualizations**: Interactive charts and graphs to visualize insights and trends

## Project Structure

- `data/`: Raw and processed IPL datasets
  - `raw/`: Original data from sources
  - `processed/`: Cleaned and transformed data for analysis
- `notebooks/`: Jupyter notebooks and Python scripts for interactive analysis
- `scripts/`: Python modules for data processing and analysis
  - `data_collection.py`: Gather data from various sources
  - `data_preprocessing.py`: Clean and transform raw data
  - `ml_models.py`: Machine learning model implementations
  - `visualization.py`: Data visualization utilities
- `models/`: Trained machine learning models
- `visualizations/`: Generated plots and interactive visualizations
- `main.py`: Command-line interface to run the complete analysis pipeline

## Setup and Installation

1. Ensure you have Python 3.8+ installed on your system
2. Clone this repository:
   ```
   git clone https://github.com/yourusername/ipl-analysis.git
   cd ipl-analysis
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Command-line Interface

The project provides a command-line interface to run different parts of the analysis pipeline:

```
python main.py [options]
```

Options:
- `--collect`: Collect data from sources
- `--process`: Process raw data
- `--train`: Train ML models
- `--visualize`: Generate visualizations
- `--predict "team1,team2,venue[,toss_winner,toss_decision]"`: Predict match outcome
- `--all`: Run the complete pipeline

Examples:
```
# Run the complete pipeline
python main.py --all

# Only collect and process data
python main.py --collect --process

# Make a match prediction
python main.py --predict "Mumbai Indians,Chennai Super Kings,Wankhede Stadium,Mumbai Indians,bat"
```

### Interactive Analysis

For interactive analysis, you can use the Jupyter notebook:

```
cd notebooks
jupyter notebook ipl_analysis.ipynb
```

Alternatively, you can run the Python script for a quick analysis:

```
python notebooks/basic_analysis.py
```

## Analysis Areas

- **Team Performance Analysis**:
  - Overall team performance across seasons
  - Win percentage by venue
  - Impact of toss decisions

- **Player Analysis**:
  - Top batsmen by runs, strike rate, and consistency
  - Top bowlers by wickets and economy rate
  - Player value assessment

- **Match Outcome Prediction**:
  - Predict the outcome of matches using historical data
  - Analyze factors that influence match results
  - Quantify the impact of toss, venue, and team form

- **Advanced Insights**:
  - Home advantage analysis
  - Player performance trends over time
  - Team strategy analysis

## Sample Visualizations

The project generates various visualizations that provide insights into IPL data:

- Team win distributions
- Player performance comparisons
- Venue-specific analyses
- Feature importance for match outcomes
- Win margin distributions

These visualizations are saved in the `visualizations/` directory.

## Extending the Project

You can extend this project in several ways:

1. **Add New Data Sources**: Implement additional data collectors in `scripts/data_collection.py`
2. **Create New Features**: Add feature engineering in `scripts/data_preprocessing.py`
3. **Implement Additional Models**: Add new ML models in `scripts/ml_models.py`
4. **Create More Visualizations**: Implement new visualization types in `scripts/visualization.py`

## Dependencies

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- xgboost
- plotly
- jupyter
- requests
- beautifulsoup4

## License

This project is licensed under the MIT License - see the LICENSE file for details. 