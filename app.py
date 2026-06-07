import streamlit as st
import pickle

# ---------------- PAGE CONFIG ---------------- #

st.set_page_config(
    page_title="AI Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)

# ---------------- LOAD DATA ---------------- #

@st.cache_resource
def load_data():

    movies = pickle.load(
        open("movies.pkl", "rb")
    )

    similarity = pickle.load(
        open("similarity.pkl", "rb")
    )

    return movies, similarity

movies, similarity = load_data()

movie_list = movies["title"].values

# ---------------- SIDEBAR ---------------- #

with st.sidebar:

    st.title("🎬 About")

    st.write("""
    AI Movie Recommendation System

    Dataset:
    TMDB 5000 Movies

    Algorithm:
    Content-Based Filtering

    Similarity:
    Cosine Similarity
    """)

    st.markdown("---")

    st.success("Machine Learning Project")

# ---------------- RECOMMEND FUNCTION ---------------- #

def recommend(movie):

    movie_index = movies[
        movies["title"] == movie
    ].index[0]

    distances = similarity[movie_index]

    movie_scores = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    recommendations = []

    for i in movie_scores:

        recommendations.append(
            {
                "title":
                movies.iloc[i[0]]["title"],

                "overview":
                movies.iloc[i[0]]["overview_text"],

                "rating":
                movies.iloc[i[0]]["vote_average"],

                "release_date":
                movies.iloc[i[0]]["release_date"],

                "popularity":
                movies.iloc[i[0]]["popularity"],

                "similarity":
                round(i[1] * 100, 2)
            }
        )

    return recommendations

# ---------------- HEADER ---------------- #

st.title("🎬 AI Movie Recommendation System")

st.markdown("""
### 🤖 Intelligent Movie Recommendation Engine

Discover similar movies using Machine Learning and Content-Based Filtering.
""")

# ---------------- METRICS ---------------- #

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Movies",
        len(movies)
    )

with col2:
    st.metric(
        "Features",
        "Genres + Overview"
    )

with col3:
    st.metric(
        "Algorithm",
        "Cosine"
    )

st.markdown("---")

# ---------------- SEARCH ---------------- #

selected_movie = st.selectbox(
    "🔍 Search Movie",
    movie_list
)

# ---------------- BUTTON ---------------- #

if st.button("🎯 Recommend"):

    with st.spinner(
        "Finding similar movies..."
    ):

        recommendations = recommend(
            selected_movie
        )

    st.subheader(
        "Recommended Movies"
    )

    for movie in recommendations:

        with st.container():

            st.markdown("---")

            st.markdown(
                f"## 🎬 {movie['title']}"
            )

            st.write(
                f"⭐ Rating: {movie['rating']}"
            )

            st.write(
                f"📅 Release Date: {movie['release_date']}"
            )

            st.write(
                f"🔥 Popularity: {movie['popularity']}"
            )

            st.write(
                f"🎯 Similarity Score: {movie['similarity']}%"
            )

            st.write(
                movie["overview"]
            )

# ---------------- FOOTER ---------------- #

st.markdown("---")

st.caption(
    "Developed by Deepanshu Sharma | Python • Streamlit • Scikit-Learn • TMDB Dataset"
)