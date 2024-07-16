# README

## Music Recommendation System

### Overview

This repository contains code for various music recommendation systems, including collaborative filtering using K-Nearest Neighbors (KNN), matrix factorization using Singular Value Decomposition (SVD), and content-based filtering. These systems are built to recommend songs to users based on their listening history and song attributes.

### Table of Contents

- [Dataset](#dataset)
- [Dependencies](#dependencies)
- [File Descriptions](#file-descriptions)
- [Usage](#usage)
- [Results](#results)
- [Authors](#authors)

### Dataset

The dataset used in this project consists of two main parts:

1. **User Listening Data**: This dataset includes user IDs, song IDs, and the number of times each user has listened to each song.
2. **Song Metadata**: This dataset includes song IDs, song titles, artist names, and other metadata.

The datasets are available at:
- [User Listening Data](https://static.turi.com/datasets/millionsong/10000.txt)
- [Song Metadata](https://static.turi.com/datasets/millionsong/song_data.csv)

### Dependencies

To run the code in this repository, you need the following libraries:

- pandas
- numpy
- matplotlib
- seaborn
- scipy
- scikit-learn
- surprise
- fuzzywuzzy
- xgboost

You can install these dependencies using pip:

```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn surprise fuzzywuzzy xgboost
```

### File Descriptions

- **data_preparation.py**: Code for loading and preparing the dataset.
- **knn_recommender.py**: Implementation of the KNN-based collaborative filtering recommender.
- **svd_recommender.py**: Implementation of the SVD-based collaborative filtering recommender.
- **content_based_recommender.py**: Implementation of the content-based recommender system.
- **recommender_systems.ipynb**: Jupyter Notebook containing the code for training and evaluating the different recommender systems.

### Usage

1. **Prepare the Data**: Load and preprocess the dataset.

```python
import pandas as pd

# Load user listening data
song_info = pd.read_csv('https://static.turi.com/datasets/millionsong/10000.txt', sep='\t', header=None)
song_info.columns = ['user_id', 'song_id', 'listen_count']

# Load song metadata
song_actual = pd.read_csv('https://static.turi.com/datasets/millionsong/song_data.csv')
song_actual.drop_duplicates(['song_id'], inplace=True)

# Merge datasets
songs = pd.merge(song_info, song_actual, on="song_id", how="left")
songs.to_csv('songs.csv', index=False)
```

2. **KNN Recommender**: Train and evaluate the KNN-based collaborative filtering recommender.

```python
from recommeders.knn_recommender import Recommender

# Load preprocessed data
df_songs = pd.read_csv('songs.csv')

# Prepare the data for the recommender
# ... (code to prepare the data)

# Initialize and train the recommender
model = Recommender(metric='cosine', algorithm='brute', k=20, data=mat_songs_features, decode_id_song=decode_id_song)

# Make recommendations
song = 'I believe in miracles'
new_recommendations = model.make_recommendation(new_song=song, n_recommendations=10)
print(f"The recommendations for {song} are: {new_recommendations}")
```

3. **SVD Recommender**: Train and evaluate the SVD-based collaborative filtering recommender.

```python
from surprise import SVD
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split

# Load preprocessed data
df_songs = pd.read_csv('songs.csv')

# Prepare the data for the recommender
# ... (code to prepare the data)

# Train and evaluate the SVD model
reader = Reader(rating_scale=(1, 10))
data = Dataset.load_from_df(df_songs[['user_id', 'song_id', 'listen_count']], reader)
trainset, testset = train_test_split(data, test_size=.25)

svd = SVD(n_factors=160, n_epochs=100, lr_all=0.005, reg_all=0.1)
svd.fit(trainset)
predictions = svd.test(testset)
accuracy.rmse(predictions)
```

4. **Content-Based Recommender**: Train and evaluate the content-based recommender.

```python
from content_based_recommender import ContentBasedRecommender

# Load preprocessed data
songs = pd.read_csv('content based recommedation system/songdata.csv')

# Prepare the data for the recommender
# ... (code to prepare the data)

# Initialize and train the recommender
recommedations = ContentBasedRecommender(similarities)

# Make recommendations
recommendation = {"song": songs['song'].iloc[10], "number_songs": 4}
recommedations.recommend(recommendation)
```

### Results

The results of the recommendation systems are evaluated using metrics such as RMSE for collaborative filtering models. Detailed results and evaluation metrics can be found in the `recommender_systems.ipynb` notebook.
