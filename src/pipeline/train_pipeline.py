import os
import sys
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

from src.exception import CustomException
from src.utils import save_object


class TrainPipeline:
    def __init__(self):
        self.artifact_dir = os.path.join("artifacts")
        os.makedirs(self.artifact_dir, exist_ok=True)

        self.model_path = os.path.join(self.artifact_dir, "model.pkl")
        self.preprocessor_path = os.path.join(self.artifact_dir, "preprocessor.pkl")

    def initiate_training(self, train_data_path):

        try:
            df = pd.read_csv(train_data_path)

            # Target column
            target_column = "math_score"

            X = df.drop(columns=[target_column])
            y = df[target_column]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # Identify column types
            numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns
            categorical_cols = X.select_dtypes(include=["object"]).columns

            # Numerical pipeline
            num_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])

            # Categorical pipeline
            cat_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ])

            # Combine
            preprocessor = ColumnTransformer([
                ("num_pipeline", num_pipeline, numerical_cols),
                ("cat_pipeline", cat_pipeline, categorical_cols)
            ])

            # Transform data
            X_train_processed = preprocessor.fit_transform(X_train)
            X_test_processed = preprocessor.transform(X_test)

            # Model
            model = RandomForestRegressor(
                n_estimators=100,
                random_state=42
            )

            model.fit(X_train_processed, y_train)

            # Evaluation
            predictions = model.predict(X_test_processed)

            r2 = r2_score(y_test, predictions)
            rmse = np.sqrt(mean_squared_error(y_test, predictions))

            print("Model Performance:")
            print("R2 Score:", r2)
            print("RMSE:", rmse)

            # Save artifacts
            save_object(self.preprocessor_path, preprocessor)
            save_object(self.model_path, model)

            print("Artifacts saved successfully.")

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = TrainPipeline()
    obj.initiate_training("data/stud.csv")  # change path if needed