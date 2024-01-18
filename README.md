# Music Playlist Generator

This Python project analyzes music data using machine learning techniques and creates a personalized playlist based on user mood and genre preferences.

## Overview

The project involves the following key steps:

1. **Data Analysis and Preprocessing:**
   - Utilizes pandas and seaborn for exploring and visualizing the dataset.
   - Implements hypothesis testing and regression analysis to understand the relationship between music features and user mood.

2. **Cluster Analysis:**
   - Applies K-Means clustering to group songs based on features like Valence, Energy, and Loudness.
   - Evaluates cluster quality using silhouette analysis and elbow plots.

3. **User Input and Playlist Generation:**
   - Takes user mood input within a 0.5 range and retrieves corresponding music clusters.
   - Allows users to select preferred genres.
   - Generates a playlist from the selected clusters and genres.

4. **Visualization:**
   - Visualizes the distribution of music features and genres within each cluster.

## Dependencies

- Python 3.x
- Required Python libraries: pandas, seaborn, matplotlib, scikit-learn

## Getting Started

1. **Clone the Repository:**
   
   git clone https://github.com/fbeyzaburkay/cs210.git
   cd cs210

   cd "CS210 Spotify playlist generator by mood.py"

