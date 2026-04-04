from pathlib import Path
import pandas as pd

# 1. PATH SETUP
BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data" / "ml-1m"
OUTPUT_DIR = BASE_DIR / "output"
EMB_DIR = OUTPUT_DIR / "embeddings"

RATINGS_PATH = DATA_DIR / "ratings.dat"
MOVIES_PATH = DATA_DIR / "movies.dat"
USERS_PATH = DATA_DIR / "users.dat"

EMB64_PATH = EMB_DIR / "user_embeddings_64.csv"
EMB128_PATH = EMB_DIR / "user_embeddings_128.csv"
EMB256_PATH = EMB_DIR / "user_embeddings_256.csv"

# If exported final k-means assignments from R, put the file here
CLUSTER_ASSIGNMENTS_PATH = OUTPUT_DIR / "kmeans_cluster_assignments.csv"

INTERP_DIR = OUTPUT_DIR / "cluster_interpretations"


# 2. LOAD MOVIELENS FILES
def load_movielens():
    ratings = pd.read_csv(
        RATINGS_PATH,
        sep="::",
        engine="python",
        names=["UserID", "MovieID", "Rating", "Timestamp"]
    )

    movies = pd.read_csv(
        MOVIES_PATH,
        sep="::",
        engine="python",
        names=["MovieID", "Title", "Genres"],
        encoding="latin-1"
    )

    users = pd.read_csv(
        USERS_PATH,
        sep="::",
        engine="python",
        names=["UserID", "Gender", "Age", "Occupation", "Zip"]
    )

    return ratings, movies, users


# 3. LOAD EMBEDDINGS

def load_embeddings():
    embedding_paths = {
        64: EMB64_PATH,
        128: EMB128_PATH,
        256: EMB256_PATH
    }

    embeddings = {}

    for size, path in embedding_paths.items():
        if path.exists():
            df = pd.read_csv(path)
            embeddings[size] = df
            print(f"Loaded embedding {size}: {df.shape}")
        else:
            print(f"Missing embedding file: {path}")

    return embeddings

# 4. LOAD CLUSTER ASSIGNMENTS
def load_cluster_assignments():
    if not CLUSTER_ASSIGNMENTS_PATH.exists():
        raise FileNotFoundError(
            f"Cluster assignment file not found:\n{CLUSTER_ASSIGNMENTS_PATH}\n\n"
            "Make sure you exported it from R, for example:\n"
            'write.csv(cluster_df, "output/kmeans_cluster_assignments.csv", row.names = FALSE)'
        )

    cluster_df = pd.read_csv(CLUSTER_ASSIGNMENTS_PATH)

    expected_cols = {"UserID", "Cluster"}
    if not expected_cols.issubset(cluster_df.columns):
        raise ValueError(
            f"Cluster file must contain columns {expected_cols}, but has {set(cluster_df.columns)}"
        )

    return cluster_df


# 5. SUMMARIES
def summarize_ratings_by_cluster(merged):
    rating_summary = (
        merged.groupby("Cluster")
        .agg(
            avg_rating=("Rating", "mean"),
            sd_rating=("Rating", "std"),
            n_ratings=("Rating", "size")
        )
        .reset_index()
        .sort_values("Cluster")
    )
    return rating_summary


def summarize_user_activity(merged):
    user_activity = (
        merged.groupby(["UserID", "Cluster"])
        .size()
        .reset_index(name="num_ratings")
    )

    activity_summary = (
        user_activity.groupby("Cluster")
        .agg(
            avg_ratings_per_user=("num_ratings", "mean"),
            median_ratings_per_user=("num_ratings", "median"),
            min_ratings_per_user=("num_ratings", "min"),
            max_ratings_per_user=("num_ratings", "max")
        )
        .reset_index()
        .sort_values("Cluster")
    )

    return user_activity, activity_summary


def summarize_genres(merged, movies):
    # Expand genres once from movie table
    movie_genres = movies.copy()
    movie_genres["Genres"] = movie_genres["Genres"].str.split("|")
    movie_genres = movie_genres.explode("Genres")

    genre_summary = (
        merged[["MovieID", "Cluster"]]
        .merge(movie_genres[["MovieID", "Genres"]], on="MovieID", how="inner")
        .groupby(["Cluster", "Genres"])
        .size()
        .reset_index(name="count")
    )

    genre_summary["prop"] = (
        genre_summary.groupby("Cluster")["count"]
        .transform(lambda x: x / x.sum())
    )

    genre_summary = genre_summary.sort_values(
        ["Cluster", "prop"], ascending=[True, False]
    )

    top_genres = (
        genre_summary.groupby("Cluster", group_keys=False)
        .head(5)
        .reset_index(drop=True)
    )

    return genre_summary, top_genres


def summarize_demographics(cluster_df, users):
    demographic_summary = (
        cluster_df.merge(users, on="UserID", how="left")
        .groupby("Cluster")
        .agg(
            n_users=("UserID", "size"),
            pct_female=("Gender", lambda x: (x == "F").mean()),
            pct_male=("Gender", lambda x: (x == "M").mean()),
            avg_age_code=("Age", "mean")
        )
        .reset_index()
        .sort_values("Cluster")
    )

    return demographic_summary

# 6. MAIN INTERPRETATION PIPELINE
def main():
    print("Loading MovieLens data...")
    ratings, movies, users = load_movielens()
    print(f"Ratings shape: {ratings.shape}\n"
          f"Movies shape: {movies.shape}\n"
          f"Users shape: {users.shape}")

    print("\nLoading embeddings...")
    embeddings = load_embeddings()

    print("\nLoading cluster assignments...")
    cluster_df = load_cluster_assignments()
    print(f"Cluster assignment shape: {cluster_df.shape}")
    print(cluster_df.head())

    print(f"\nCluster sizes:\n"
          f"{cluster_df["Cluster"].value_counts().sort_index()}")

    print("\nMerging ratings with cluster assignments...")
    merged = ratings.merge(cluster_df, on="UserID", how="inner")
    print(f"Merged shape: {merged.shape}")

    print("\nSummarizing rating behavior by cluster...")
    rating_summary = summarize_ratings_by_cluster(merged)
    print(rating_summary)

    print("\nSummarizing user activity by cluster...")
    user_activity, activity_summary = summarize_user_activity(merged)
    print(activity_summary)

    print("\nSummarizing genre preferences by cluster...")
    genre_summary, top_genres = summarize_genres(merged, movies)
    print(top_genres)

    print("\nSummarizing demographics by cluster...")
    demographic_summary = summarize_demographics(cluster_df, users)
    print(demographic_summary)

    # Cluster size summary
    cluster_sizes = (
        cluster_df["Cluster"]
        .value_counts()
        .sort_index()
        .rename_axis("Cluster")
        .reset_index(name="n_users")
    )

    print(f"\nCluster size summary: {cluster_sizes}")

    # Combined cluster profile
    cluster_profile = (
        rating_summary
        .merge(activity_summary, on="Cluster", how="left")
        .merge(demographic_summary, on="Cluster", how="left")
        .sort_values("Cluster")
    )

    print(f"\nCluster profile (combined summary): {cluster_profile}")

    # Save outputs (CREATE FOLDER FIRST)
    INTERP_DIR.mkdir(parents=True, exist_ok=True)

    rating_summary.to_csv(INTERP_DIR / "py_rating_summary.csv", index=False)
    user_activity.to_csv(INTERP_DIR / "py_user_activity.csv", index=False)
    activity_summary.to_csv(INTERP_DIR / "py_activity_summary.csv", index=False)
    genre_summary.to_csv(INTERP_DIR / "py_genre_summary.csv", index=False)
    top_genres.to_csv(INTERP_DIR / "py_top_genres.csv", index=False)
    demographic_summary.to_csv(INTERP_DIR / "py_demographic_summary.csv", index=False)
    cluster_sizes.to_csv(INTERP_DIR / "py_cluster_sizes.csv", index=False)
    cluster_profile.to_csv(INTERP_DIR / "py_cluster_profile.csv", index=False)

    print(f"\nSaved interpretation outputs to: {INTERP_DIR}")

    # Quick embedding inspection
    print("\nEmbedding file summary:")
    for size, df in embeddings.items():
        print(f"Embedding {size}: {df.shape}")


if __name__ == "__main__":
    main()
