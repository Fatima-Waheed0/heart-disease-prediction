import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def load_data(filepath='../data/heart_disease.csv'):
    """Load the dataset from the provided CSV file."""
    df = pd.read_csv(filepath)
    return df

def preprocess_data(df):
    """Preprocess the dataset by separating features and target and applying transformations."""
    # Separate features and target variable
    X = df.drop(columns=['target'])
    y = df['target']
    
    # Define columns for scaling and encoding
    numeric_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
    
    # Column transformer for scaling and encoding
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(), categorical_features)
        ])
    
    # Split data into training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    return X_train, X_test, y_train, y_test, preprocessor
