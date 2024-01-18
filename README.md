# Music Playlist Generator

This Python project analyzes my music data from Spotify using unsupervised machine learning and creates a personalized playlist based on user mood input and genre preferences input.

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

- Python 3.8.12
- Required Python libraries: [pandas](https://pandas.pydata.org), [seaborn](https://seaborn.pydata.org), [matplotlib](https://matplotlib.org), [scikit-learn](https://scikit-learn.org/stable/), [spotipy](https://spotipy.readthedocs.io), [statsmodels](https://www.statsmodels.org/stable/index.html), [scipy](https://scipy.org)


## Navigate to the Playlist Generator
[Link to My Project🖇️](https://github.com/fbeyzaburkay/cs210/blob/main/CS210%20Spotify%20playlist%20generator%20by%20mood.ipynb)


