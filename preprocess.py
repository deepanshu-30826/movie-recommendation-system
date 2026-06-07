import pandas as pd
import ast
import pickle

# -------------------------
# Load Datasets
# -------------------------

movies = pd.read_csv("tmdb_5000_movies.csv")
credits = pd.read_csv("tmdb_5000_credits.csv")

# Merge datasets
movies = movies.merge(credits, on="title")

# Required columns
movies = movies[
    [
        'movie_id',
        'title',
        'overview',
        'genres',
        'keywords',
        'cast',
        'crew',
        'vote_average',
        'release_date',
        'popularity'
    ]
]

# Remove null values
movies.dropna(inplace=True)

# -------------------------
# Helper Functions
# -------------------------

def convert(text):
    L = []

    for i in ast.literal_eval(text):
        L.append(i['name'])

    return L


def convert_cast(text):
    L = []

    counter = 0

    for i in ast.literal_eval(text):

        if counter < 3:
            L.append(i['name'])
            counter += 1
        else:
            break

    return L


def fetch_director(text):
    L = []

    for i in ast.literal_eval(text):

        if i['job'] == 'Director':
            L.append(i['name'])
            break

    return L

# -------------------------
# Data Processing
# -------------------------

movies['genres'] = movies['genres'].apply(convert)

movies['keywords'] = movies['keywords'].apply(convert)

movies['cast'] = movies['cast'].apply(convert_cast)

movies['crew'] = movies['crew'].apply(fetch_director)

movies['overview'] = movies['overview'].apply(
    lambda x: x.split()
)

# Remove spaces

movies['genres'] = movies['genres'].apply(
    lambda x: [i.replace(" ", "") for i in x]
)

movies['keywords'] = movies['keywords'].apply(
    lambda x: [i.replace(" ", "") for i in x]
)

movies['cast'] = movies['cast'].apply(
    lambda x: [i.replace(" ", "") for i in x]
)

movies['crew'] = movies['crew'].apply(
    lambda x: [i.replace(" ", "") for i in x]
)

# -------------------------
# Create Tags
# -------------------------

movies['tags'] = (
    movies['overview']
    + movies['genres']
    + movies['keywords']
    + movies['cast']
    + movies['crew']
)

# Save overview text separately

movies['overview_text'] = movies['overview'].apply(
    lambda x: " ".join(x)
)

# Final dataframe

new_df = movies[
    [
        'movie_id',
        'title',
        'tags',
        'overview_text',
        'vote_average',
        'release_date',
        'popularity'
    ]
]

# Convert tags list -> string

new_df['tags'] = new_df['tags'].apply(
    lambda x: " ".join(x)
)

# Lowercase

new_df['tags'] = new_df['tags'].apply(
    lambda x: x.lower()
)

# -------------------------
# Save Final Dataset
# -------------------------

pickle.dump(
    new_df,
    open("movies.pkl", "wb")
)

print("\nDataset Ready ✅")
print(new_df.head())

print("\nShape:")
print(new_df.shape)

print("\nmovies.pkl Saved Successfully ✅")