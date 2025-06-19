import os
import sys
from dataclasses import dataclass
import pandas as pd

from catboost import CatBoostRegressor
from sklearn.ensemble import(
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor
)

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor    
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
from src.exception import CustomException
from src.logger import logging

from src.utils import save_object, evaluate_model

@dataclass

class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()


    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and testing input and target variables")
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models = {
                'RandomForestRegressor': RandomForestRegressor(),
                'GradientBoostingRegressor': GradientBoostingRegressor(),
                'AdaBoostRegressor': AdaBoostRegressor(),
                'LinearRegression': LinearRegression(),
                'KNeighborsRegressor': KNeighborsRegressor(),
                'XGBRegressor': XGBRegressor(),
                'DecisionTreeRegressor': DecisionTreeRegressor(),
                'CatBoostRegressor': CatBoostRegressor(verbose=0)
            }

            model_report: dict = self.evaluate_model(X_train, y_train, X_test, y_test, models)
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]
            if best_model_score < 0.6:
                raise CustomException("No best model found")
            logging.info(f"Best model found: {best_model_name} with score: {best_model_score}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)
            logging.info(f"R2 score of the best model: {r2_square}")
            return r2_square

        except Exception as e:
            raise CustomException(e, sys)

    def evaluate_model(self, X_train, y_train, X_test, y_test, models):
        """
        Evaluates the given models and returns a report dictionary.
        """
        model_report = {}
        for name, model in models.items():
            model.fit(X_train, y_train)
            score = model.score(X_test, y_test)
            model_report[name] = score
        return model_report