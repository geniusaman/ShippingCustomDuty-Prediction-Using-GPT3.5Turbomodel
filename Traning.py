from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor, AdaBoostRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from Preprocessing import preprocess_data


file_path = input("Enter file path: ")
X_data, y_data = preprocess_data(file_path)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_data,  # Only 'HS Code' as input
    y_data,  # Target variable
    test_size=0.2,
    random_state=42
)


# Define the pipeline with SimpleImputer, RandomForestRegressor
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('model', RandomForestRegressor())
])

# Training the pipeline
pipeline.fit(X_train.values.reshape(-1, 1), y_train)

# Model Evaluation
y_pred = pipeline.predict(X_test.values.reshape(-1, 1))
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error (MAE): {mae}")
r_squared = r2_score(y_test, y_pred)
print(f"R-squared (R2): {r_squared}")

# Save the trained model to a file
model_filename = 'trained_model.joblib'
joblib.dump(pipeline, model_filename)
print(f'Model saved as {model_filename}')
