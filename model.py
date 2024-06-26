import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

# Load the dataset
data = pd.read_csv("dataset.csv")

# Split data into features and target
X = data.drop(columns=["Target"])
y = data["Target"]

# Preprocessing for numerical features
numerical_features = X.select_dtypes(include=["int64", "float64"]).columns
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Preprocessing for categorical features
categorical_features = X.select_dtypes(include=["object"]).columns
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Define models to evaluate
models = [
    ('RandomForest', RandomForestClassifier()),
    ('GradientBoosting', GradientBoostingClassifier()),
    ('SVM', SVC())
]

# Evaluate each model using grid search
for name, model in models:
    print(f"Training {name}...")
    pipe = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
    param_grid = {}
    if name == 'RandomForest':
        param_grid = {
            'classifier__n_estimators': [100, 200, 300],
            'classifier__max_depth': [None, 10, 20, 30]
        }
    elif name == 'GradientBoosting':
        param_grid = {
            'classifier__n_estimators': [100, 200, 300],
            'classifier__learning_rate': [0.01, 0.1, 1],
            'classifier__max_depth': [3, 5, 7]
        }
    elif name == 'SVM':
        param_grid = {
            'classifier__C': [0.1, 1, 10],
            'classifier__gamma': ['scale', 'auto'],
            'classifier__kernel': ['linear', 'rbf']
        }
    grid_search = GridSearchCV(pipe, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X, y)
    
    print("Best Hyperparameters:", grid_search.best_params_)
    print("Best Accuracy:", grid_search.best_score_)

# Once you find the best model, you can use it to make predictions on new data
# For example:
# best_model = grid_search.best_estimator_
# X_new = ... # New data
# predictions = best_model.predict(X_new)
