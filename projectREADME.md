# Spotify Playlist Generator by Mood

This Python project analyzes my music data from Spotify using unsupervised machine learning and creates a personalized playlist based on user mood input and genre preferences input.

<img width="1028" alt="Screenshot 2024-01-18 at 20 39 31" src="https://github.com/fbeyzaburkay/cs210/assets/122027354/402dbe00-6e0e-479c-b05c-24f348cda183">

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

## EDA And Visualizations

<img width="557" alt="Screenshot 2024-01-18 at 19 56 40" src="https://github.com/fbeyzaburkay/cs210/assets/122027354/7402d6ce-0d77-4de8-b99b-490f35adf046">

![Unknown](https://github.com/fbeyzaburkay/cs210/assets/122027354/9b46bcfb-3108-4cf7-8ca7-a32d4059cec7)

<img width="1028" alt="Screenshot 2024-01-18 at 19 57 07" src="https://github.com/fbeyzaburkay/cs210/assets/122027354/ea4ecc8d-05a4-4bb1-821e-1f8e69f498f7">

<img width="887" alt="Screenshot 2024-01-18 at 19 57 22" src="https://github.com/fbeyzaburkay/cs210/assets/122027354/7fb0242f-c287-4ea8-b5a4-781a852a65ec">

![Unknown-2](https://github.com/fbeyzaburkay/cs210/assets/122027354/af592398-301e-44d8-a474-b0281c988510)

## Sample of Generated Playlist

<img width="847" alt="Screenshot 2024-01-18 at 20 30 52" src="https://github.com/fbeyzaburkay/cs210/assets/122027354/b809cf9a-34b7-490c-b66e-dde9f038c082">

## Acknowledgements

- Special thanks to [Spotipy](https://spotipy.readthedocs.io/) for providing a Python library to access the Spotify Web API.
- The dataset used in this project was self-extracted using Spotipy.

## License

This project is licensed under the [MIT License](https://choosealicense.com/licenses/mit/) - see the [LICENSE](LICENSE) file for details.
