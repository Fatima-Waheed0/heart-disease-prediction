from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import pickle

def train_model(X_train, y_train, preprocessor):
    """Train a Logistic Regression model and save it."""
    # Create the pipeline with preprocessing and logistic regression
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42))
    ])
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Save the trained model
    with open('heart_disease_logreg_model.pkl', 'wb') as model_file:
        pickle.dump(model, model_file)
    
    return model
