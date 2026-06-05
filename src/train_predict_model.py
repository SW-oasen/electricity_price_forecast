import pandas as pd

# Split the data into training and testing sets based on the date
def train_test_split_by_date(df, date_column, target_column, split_date):
    train_data = df[df[date_column] < split_date]
    test_data = df[df[date_column] >= split_date]
    features_train = train_data.drop([date_column, target_column], axis=1)
    target_train = train_data[target_column]
    features_test = test_data.drop([date_column, target_column], axis=1)
    target_test = test_data[target_column]
    return features_train, target_train, features_test, target_test

def train_test_split_by_date_for_sarimax(df, date_column, target_column, split_date):
    train_data = df[df[date_column] < split_date]
    test_data = df[df[date_column] >= split_date]

    features_train = train_data.drop(target_column, axis=1)
    features_test = test_data.drop(target_column, axis=1)
    target_train = train_data[[date_column, target_column]]
    target_test = test_data[[date_column, target_column]]

    return features_train, target_train, features_test, target_test

# column transformer pipeline for preprocessing
from sklearn.compose import ColumnTransformer   
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def init_preprocessor(in_df):
    # identify numeric and categorical columns
    numeric_features = in_df.select_dtypes(include=['float64', 'float32']).columns
    categorical_features = in_df.select_dtypes(include=['int64', 'int32']).columns
    # numeric transformer
    numeric_transformer = Pipeline(steps=[ 
        ('scaler', StandardScaler()) 
    ])  
    # categorical transformer
    categorical_transformer = Pipeline(steps=[  
        ('onehot', OneHotEncoder(handle_unknown='ignore')) 
    ])  

    # combine transformers into a column transformer
    preprocessor = ColumnTransformer(   
        transformers=[ 
            ('num', numeric_transformer, numeric_features), 
            ('cat', categorical_transformer, categorical_features) 
        ])
    return preprocessor


# build the model pipeline
from sklearn.ensemble import RandomForestRegressor

def init_model_pipeline(in_df, model):
    preprocessor = init_preprocessor(in_df)
    model_pipeline = Pipeline(steps=[ 
        ('preprocessor', preprocessor), 
        ('model', model) 
    ])  
    return model_pipeline

def train_model_predict(model_pipeline, features_train, target_train, features_future):
    model_pipeline.fit(features_train, target_train)
    predictions = model_pipeline.predict(features_future)
    return predictions

import numpy as np
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def print_scores(model_name, target_test, predictions):
    mae = mean_absolute_error(target_test, predictions)
    rmse = np.sqrt(mean_squared_error(target_test, predictions))
    r2 = r2_score(target_test, predictions)
    print("-"*20 + f" scoring " + "-"*20)
    print(f"{'model':<15} {'MAE':>10} {'RMSE':>10} {'R²':>8}")
    print(f'{model_name:<16} {mae:>10.2f} {rmse:>10.2f} {r2:>8.2f}')


def rolling_origin_backtest(
    model_pipeline,
    df,
    date_column,
    target_column,
    first_split_date,
    step_hours=24,
    horizon_hours=24,
    max_folds=None,
):
    """
    Perform rolling-origin validation on a time-indexed dataset.

    Each fold trains on all rows before split_time and tests the next
    horizon_hours window. split_time then moves forward by step_hours.

    Returns:
        fold_scores: DataFrame with per-fold MAE/RMSE/R2 and fold timing
        summary_scores: dict with average MAE/RMSE/R2 and fold count
    """
    if step_hours <= 0 or horizon_hours <= 0:
        raise ValueError("step_hours and horizon_hours must be > 0")

    working_df = df.copy()
    working_df[date_column] = pd.to_datetime(working_df[date_column], utc=True)
    working_df = working_df.sort_values(date_column).reset_index(drop=True)

    split_time = pd.Timestamp(first_split_date)
    if split_time.tz is None:
        split_time = split_time.tz_localize("UTC")
    else:
        split_time = split_time.tz_convert("UTC")

    max_time = working_df[date_column].max()
    step_delta = pd.Timedelta(hours=step_hours)
    horizon_delta = pd.Timedelta(hours=horizon_hours)

    fold_results = []
    fold_id = 0

    while split_time + horizon_delta <= max_time + pd.Timedelta(hours=1):
        train_mask = working_df[date_column] < split_time
        test_mask = (working_df[date_column] >= split_time) & (
            working_df[date_column] < split_time + horizon_delta
        )

        train_data = working_df.loc[train_mask]
        test_data = working_df.loc[test_mask]

        if train_data.empty or test_data.empty:
            split_time += step_delta
            continue

        features_train = train_data.drop([date_column, target_column], axis=1)
        target_train = train_data[target_column]
        features_test = test_data.drop([date_column, target_column], axis=1)
        target_test = test_data[target_column]

        fold_model = clone(model_pipeline)
        fold_model.fit(features_train, target_train)
        predictions = fold_model.predict(features_test)

        fold_id += 1
        fold_results.append(
            {
                "fold": fold_id,
                "train_end_exclusive": split_time,
                "test_start": split_time,
                "test_end_exclusive": split_time + horizon_delta,
                "n_train": len(train_data),
                "n_test": len(test_data),
                "mae": mean_absolute_error(target_test, predictions),
                "rmse": np.sqrt(mean_squared_error(target_test, predictions)),
                "r2": r2_score(target_test, predictions),
            }
        )

        if max_folds is not None and fold_id >= max_folds:
            break

        split_time += step_delta

    fold_scores = pd.DataFrame(fold_results)
    if fold_scores.empty:
        summary_scores = {"folds": 0, "mae_mean": np.nan, "rmse_mean": np.nan, "r2_mean": np.nan}
        return fold_scores, summary_scores

    summary_scores = {
        "folds": int(fold_scores["fold"].count()),
        "mae_mean": float(fold_scores["mae"].mean()),
        "rmse_mean": float(fold_scores["rmse"].mean()),
        "r2_mean": float(fold_scores["r2"].mean()),
    }
    return fold_scores, summary_scores
    

from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, learning_curve, learning_curve

# tune the model using GridSearchCV
def tune_model_grid(model_pipeline, 
                    in_param_grid, 
                    in_features_train, 
                    in_target_train,
                    scoring='neg_mean_absolute_error'):
    tscv = TimeSeriesSplit(n_splits=5)
    grid_search = GridSearchCV(estimator=model_pipeline, 
                               param_grid=in_param_grid, 
                               cv=tscv,
                               scoring=scoring, 
                               n_jobs=-1)
    grid_search.fit(in_features_train, in_target_train)
    #print(f'Best parameters: {grid_search.best_params_}')
    return grid_search.best_estimator_, grid_search.best_params_

try:
    from skopt import BayesSearchCV
except ImportError:
    BayesSearchCV = None

# tune the model using Bayesian optimization
def tune_model_bayesian(model_pipeline, 
                        in_param_bayes, 
                        in_features_train, 
                        in_target_train,
                        scoring='neg_mean_absolute_error'):
    tscv = TimeSeriesSplit(n_splits=5)
    bayes_search = BayesSearchCV(estimator=model_pipeline, 
                                 search_spaces=in_param_bayes, 
                                 cv=tscv,
                                 scoring=scoring, 
                                 n_jobs=-1)
    bayes_search.fit(in_features_train, in_target_train)
    #print(f'Best parameters: {bayes_search.best_params_}')
    return bayes_search.best_estimator_, bayes_search.best_params_

import pickle

# save trained model to pickle file
def save_model_to_pickle(model_pipeline, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(model_pipeline, f)

# load trained model from pickle file
def load_model_from_pickle(file_path):
    with open(file_path, 'rb') as f:
        model_pipeline = pickle.load(f)
    return model_pipeline

from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt

# learn curve for training and testing data
def plot_learning_curve(model_pipeline, model_name, features_train, target_train):
    tscv = TimeSeriesSplit(n_splits=3)
    train_sizes, train_scores, test_scores = learning_curve(model_pipeline, 
                                                            features_train, 
                                                            target_train, 
                                                            cv=tscv, 
                                                            scoring='neg_mean_absolute_error', n_jobs=-1)
    train_scores_mean = -train_scores.mean(axis=1)
    test_scores_mean = -test_scores.mean(axis=1)

    plt.figure(figsize=(6, 3))
    plt.plot(train_sizes, train_scores_mean, 'o-', color='orange', label='Training score')
    plt.plot(train_sizes, test_scores_mean, 'o-', color='steelblue', label='Cross-validation score')
    plt.title(f'Learning Curve of {model_name}')
    plt.xlabel('Training examples')
    plt.ylabel('MAE')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()