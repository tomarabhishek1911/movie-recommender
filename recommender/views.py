import os
import pandas as pd
from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
from django.contrib import messages
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MaxAbsScaler
from django.conf import settings


# ===== HOME =====
def home(request):
    return render(request, 'home.html')


# ===== SIGNUP =====
def signup_view(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        confirm_password = request.POST.get('confirm_password')

        if password != confirm_password:
            messages.error(request, "Passwords do not match.")
            return redirect('signup')

        if User.objects.filter(username=username).exists():
            messages.error(request, "Username already exists.")
            return redirect('signup')

        User.objects.create_user(username=username, password=password)
        messages.success(request, "Account created successfully! Please login.")
        return redirect('login')

    return render(request, 'signup.html')


# ===== LOGIN =====
def login_view(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')

        user = authenticate(request, username=username, password=password)
        if user:
            login(request, user)
            return redirect('dashboard')
        else:
            messages.error(request, "Invalid username or password.")

    return render(request, 'login.html')


# ===== LOGOUT =====
def logout_view(request):
    logout(request)
    return redirect('home')


# ===== DASHBOARD =====
def dashboard(request):
    if not request.user.is_authenticated:
        return redirect('login')

    recommendations, accuracy = None, None

    if request.method == 'POST':
        movies_csv = request.FILES.get('movies_csv')
        ratings_csv = request.FILES.get('ratings_csv')
        model_type = request.POST.get('model_type')

        if not movies_csv or not ratings_csv:
            messages.error(request, "Please upload both CSV files.")
            return redirect('dashboard')

        # Save uploaded files in /media/uploads
        upload_dir = os.path.join(settings.MEDIA_ROOT, 'uploads')
        os.makedirs(upload_dir, exist_ok=True)

        movies_path = os.path.join(upload_dir, movies_csv.name)
        ratings_path = os.path.join(upload_dir, ratings_csv.name)

        with open(movies_path, 'wb+') as f:
            for chunk in movies_csv.chunks():
                f.write(chunk)

        with open(ratings_path, 'wb+') as f:
            for chunk in ratings_csv.chunks():
                f.write(chunk)

        # Train ML model
        try:
            recommendations, accuracy = train_classifier(movies_path, ratings_path, model_type)
        except Exception as e:
            messages.error(request, f"Error while training: {str(e)}")
            recommendations, accuracy = [], 0

    return render(request, 'dashboard.html', {
        'username': request.user.username,
        'recommendations': recommendations,
        'accuracy': accuracy
    })


# ===== MACHINE LEARNING MODEL =====
def train_classifier(movies_file, ratings_file, classifier_type='random_forest'):
    movies = pd.read_csv(movies_file)
    ratings = pd.read_csv(ratings_file)

    # Validate columns
    if 'movieId' not in movies.columns or 'movieId' not in ratings.columns:
        raise ValueError("Both CSVs must contain a 'movieId' column.")

    merged = pd.merge(ratings, movies, on='movieId')

    if merged.empty:
        raise ValueError("No matching movieId values found between movies.csv and ratings.csv.")

    # Prepare features and labels
    X = merged['genres'].fillna('')
    y = merged['rating'].round().astype(int)

    vectorizer = CountVectorizer()
    X_vec = vectorizer.fit_transform(X)

    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X_vec.toarray())

    scaler = MaxAbsScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=42)

    # Choose model
    if classifier_type == 'random_forest':
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif classifier_type == 'decision_tree':
        model = DecisionTreeClassifier(max_depth=20, random_state=42)
    else:
        model = GradientBoostingClassifier(n_estimators=100, random_state=42)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Generate top recommendations
    top_movies = merged[merged['rating'] >= 4.0]['title'].drop_duplicates().tolist()[:10]

    return top_movies, accuracy
