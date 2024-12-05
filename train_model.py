import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

# Generate synthetic data
num_records = 1000
data = {
    "vehicle_count": np.random.randint(100, 1000, size=num_records),
    "average_speed": np.random.uniform(20, 120, size=num_records),
    "weather": np.random.choice(["Clear", "Rainy", "Foggy", "Snowy"], size=num_records),
    "incident": np.random.choice(["None", "Accident", "Roadwork"], size=num_records),
}
traffic_data = pd.DataFrame(data)

# Preprocess the data
X = traffic_data[['vehicle_count', 'weather', 'incident']]
y = traffic_data['average_speed']

# Set up preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), ['weather', 'incident']),
        ('num', StandardScaler(), ['vehicle_count'])
    ])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline with preprocessing and model
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', LinearRegression())
])

# Train the model
model_pipeline.fit(X_train, y_train)

# Save the model to a file
joblib.dump(model_pipeline, "traffic_model.pkl")
print("Model trained and saved as 'traffic_model.pkl'")