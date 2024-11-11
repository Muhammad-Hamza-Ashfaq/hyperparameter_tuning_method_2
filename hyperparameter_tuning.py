#Muhammad Hamza Ashfaq
#Importing necessary Libraries
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

# Loading the dataset 
data = pd.read_csv('emails.csv')

# Data Preprocessing
# Droping irrelevant columns like Email No. is non-feature Column
data = data.drop(['Email No.'], axis=1)  

# Handling missing values if any
data.fillna(0, inplace=True)

# Spliting features and target
X = data.drop('Prediction', axis=1)
y = data['Prediction']

# Scaling numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test spliting
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Defining the parameter grid for Grid Search and Random Search
rf_param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# Initializing the Random Forest classifier
rf_classifier = RandomForestClassifier(random_state=42)

# Grid Search for Random Forest
print("Running Grid Search on Random Forest...")
rf_grid_search = GridSearchCV(estimator=rf_classifier, param_grid=rf_param_grid, cv=5, scoring='f1', verbose=1, n_jobs=-1)
rf_grid_search.fit(X_train, y_train)
print("Best Random Forest parameters (Grid Search):", rf_grid_search.best_params_)
print("Best Random Forest F1 Score (Grid Search):", rf_grid_search.best_score_)

# Random Search for Random Forest
print("\nRunning Random Search on Random Forest...")
rf_random_search = RandomizedSearchCV(estimator=rf_classifier, param_distributions=rf_param_grid, n_iter=20, cv=5, scoring='f1', verbose=1, random_state=42, n_jobs=-1)
rf_random_search.fit(X_train, y_train)
print("Best Random Forest parameters (Random Search):", rf_random_search.best_params_)
print("Best Random Forest F1 Score (Random Search):", rf_random_search.best_score_)

# Evaluating the best model from Grid Search on the test set
best_rf_model_grid = rf_grid_search.best_estimator_
print("\nEvaluation on Test Set - Grid Search Optimized Model")
rf_predictions_grid = best_rf_model_grid.predict(X_test)
print(classification_report(y_test, rf_predictions_grid))

# Evaluating the best model from Random Search on the test set
best_rf_model_random = rf_random_search.best_estimator_
print("\nEvaluation on Test Set - Random Search Optimized Model")
rf_predictions_random = best_rf_model_random.predict(X_test)
print(classification_report(y_test, rf_predictions_random))
