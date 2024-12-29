# Step 1 | Import Libraries
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# Step 2 | Read Dataset
df = pd.read_csv("./assets/dataset/heart.csv")

# Step 3 | Preprocessing
df.rename(columns={
    "age": "Age",
    "sex": "Sex",
    "cp": "ChestPain",
    "trestbps": "RestingBloodPressure",
    "chol": "Cholesterol",
    "fbs": "FastingBloodSugar",
    "restecg": "RestingECG",
    "thalach": "MaxHeartRate",
    "exang": "ExerciseAngina",
    "oldpeak": "OldPeak",
    "slope": "STSlope",
    "ca": "nMajorVessels",
    "thal": "Thalium",
    "target": "Status"
}, inplace=True)

# Step 4 | Feature Scaling
X = df.drop(["Status"], axis=1)  # Independent variables
y = df["Status"]                # Target variable

# Standardize numerical features
sc = StandardScaler()
X[X.columns] = sc.fit_transform(X[X.columns])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

# Step 5 | Model Training
clf_knn = KNeighborsClassifier()
parametrs_knn = {
    'n_neighbors': [3, 5, 7, 9, 11], 
    'metric': ['euclidean', 'manhattan', 'chebyshev'], 
    'weights': ['uniform', 'distance']
}

# Perform Grid Search
grid_clf_knn = GridSearchCV(clf_knn, parametrs_knn, cv=5, n_jobs=-1)
grid_clf_knn.fit(X_train, y_train)

# Save the best model
if grid_clf_knn.best_estimator_:
    with open('heart_disease_model.pkl', 'wb') as f:
        pickle.dump(grid_clf_knn.best_estimator_, f)
    print("Model training completed and saved as 'heart_disease_model.pkl'.")
else:
    print("Model training was not successful; no model to save.")