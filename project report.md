# Spotify Playlist Generator by User Mood

This Python project analyzes my music data from Spotify using unsupervised machine learning and creates a personalized playlist based on user mood input and genre preferences input.

<img width="1028" alt="Screenshot 2024-01-18 at 20 39 31" src="https://github.com/fbeyzaburkay/cs210/assets/122027354/402dbe00-6e0e-479c-b05c-24f348cda183">

## Motivation
The motivation behind this project stems from my desire to explore the intersection of data science and music, leveraging Spotify's API to analyze and understand musical preferences based on user's mood and preferred genre.

## Data Source
The dataset utilized in this project was personally extracted using Spotipy, a Python library enabling access to the Spotify Web API.​

## Data Analysis Steps:
The project involves the following key steps:

1. **Data Extraction and Preprocessing:**

  - Extracted detailed information about tracks from multiple of my Spotify playlists using Spotipy with my Spotify API credentials.
  - Extracted track details from Spotify playlists, stores the data in a 'proj.csv' CSV file, and assigns each track a 'Main Genre' based on predefined conditions related to the 'Genres' column.

2. **EDA and Hypothesis Testing:**
  - Utilized pandas and seaborn for exploring and visualizing the dataset. Created heatmaps for showing the correlations of features. Subplots for easily exploring the dataset.
  - Implemented hypothesis testing and regression analysis to understand the relationship between music features and showed significance in relationship.

    *Correlation Heatmap*
    ![Correlation Heatmap](https://github.com/fbeyzaburkay/cs210/assets/122027354/7402d6ce-0d77-4de8-b99b-490f35adf046)

    
     *EDA Subplots*
![Unknown](https://github.com/fbeyzaburkay/cs210/assets/122027354/9b46bcfb-3108-4cf7-8ca7-a32d4059cec7)

3. **Cluster Analysis:**
  - Applied K-Means clustering to group songs based on features like Valence, Energy, and Loudness.
   - Evaluated cluster quality using silhouette analysis and elbow plots and decided on using k=5 clusters.

        *Silhouette Plots*
<img width="1028" alt="Screenshot 2024-01-18 at 19 57 07" src="https://github.com/fbeyzaburkay/cs210/assets/122027354/ea4ecc8d-05a4-4bb1-821e-1f8e69f498f7">


*Elbow Plots of SSE and Explained Variance*
<img width="887" alt="Screenshot 2024-01-18 at 19 57 22" src="https://github.com/fbeyzaburkay/cs210/assets/122027354/7fb0242f-c287-4ea8-b5a4-781a852a65ec">


4. **Feature Engineering and Visualization:**
  - Created a new User Mood column by taking weighted averages of Valence, Energy and Loudness.
  -  Visualized the distribution of music features and genres within each cluster. Each cluster shows different values for music features, showing the distinction amongst clusters.


 *Subplots for Cluster Analysis*
![Unknown-2](https://github.com/fbeyzaburkay/cs210/assets/122027354/af592398-301e-44d8-a474-b0281c988510)


5. **Playlist Generation:**
  - Took user mood input within a 0.5 range and retrieves corresponding music clusters.
   - Allowed users to select preferred genres.
   - Generated a playlist from the selected clusters and genres.

*Sample of Generated Playlist by User Mood and Genre Preference*
     <img width="847" alt="Screenshot 2024-01-18 at 20 30 52" src="https://github.com/fbeyzaburkay/cs210/assets/122027354/b809cf9a-34b7-490c-b66e-dde9f038c082">

## Findings

1. ** High User Mood Index::**
   - Most of my Pop and Rock songs exhibit high User Mood indexes, suggesting a preference for uplifting or energetic tracks.

2. **Low User Mood Index:**
   - Surprisingly, Electronic, Progressive Rock, and House songs tend to have lower User Mood indexes. This finding suggests a potential inclination towards genres that may have a more subdued or varied emotional tone, also may caused by the subtext of the song lyrics.

## Limitations and Future Work
In the future, I aim to improve the project by incorporating the `get.recommendations()` function from Spotipy authorization. Despite my current efforts, I encountered challenges in achieving this in Python. My future plans involve overcoming these obstacles and successfully implementing the function, providing users with personalized song recommendations based on the User Mood formula derived from Valence, Energy, and Loudness features. This enhancement aims to offer an enhanced music exploration experience, allowing users to discover new songs tailored to their mood and genre preferences.


## Navigate to the Playlist Generator
[Link to My Project🖇️](https://github.com/fbeyzaburkay/cs210/blob/main/CS210%20Spotify%20playlist%20generator%20by%20mood.ipynb)
