from src.data_preprocessing import load_data, preprocess_data
from src.model_training import train_model
from src.evaluation import evaluate_model
import pickle

def main():
    # Load and preprocess data
    df = load_data('data/heart_disease.csv')
    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(df)
    
    # Train the model
    model = train_model(X_train, y_train, preprocessor)
    
    # Evaluate the model
    accuracy, cm, report = evaluate_model(model, X_test, y_test)
    
    # Print results
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Confusion Matrix:\n{cm}')
    print(f'Classification Report:\n{report}')

if __name__ == "__main__":
    main()
