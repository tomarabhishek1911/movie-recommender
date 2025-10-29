import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MaxAbsScaler


def train_classifier_for_user(user_id, classifier_type, movies_df, ratings_df):
    # Merge movies and ratings
    movie_ratings = pd.merge(ratings_df, movies_df, on='movieId')

    user_ratings = movie_ratings[movie_ratings['userId'] == user_id]

    if user_ratings.empty:
        return [], 0

    X = user_ratings['genres'].fillna('')
    y = user_ratings['rating'].round().astype(int)

    vectorizer = CountVectorizer()
    X_vec = vectorizer.fit_transform(X)

    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X_vec.toarray())

    scaler = MaxAbsScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    # Choose model
    if classifier_type == 'random_forest':
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif classifier_type == 'decision_tree':
        model = DecisionTreeClassifier(max_depth=20, random_state=42)
    else:
        model = GradientBoostingClassifier(n_estimators=100, random_state=42)

    # Train model
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # Recommend high-rated movies for user
    top_movies = user_ratings[user_ratings['rating'] >= 4.0]['title'].drop_duplicates().tolist()

    return top_movies, acc
