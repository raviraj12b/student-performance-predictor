import pandas as pd
import pickle
import os
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from config import DATA_PATH, MODEL_PATH, TEST_SIZE, RANDOM_STATE
from logger import get_logger

log = get_logger(__name__)


# ── Load & prepare data ────────────────────────────────
def load_data():
    log.info(f"Loading dataset from {DATA_PATH}")

    if not os.path.exists(DATA_PATH):
        log.error(f"Dataset not found at {DATA_PATH}")
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.replace(' ', '_').str.lower()
    df['average_score'] = (
        df['math_score'] + df['reading_score'] + df['writing_score']
    ) / 3

    log.info(f"Dataset loaded → {df.shape[0]} rows, {df.shape[1]} columns")
    return df


# ── Build features ─────────────────────────────────────
def build_features(df):
    X = df.drop(columns=[
        'math_score', 'reading_score',
        'writing_score', 'average_score'
    ])
    y = df['average_score']

    log.info(f"Features: {X.columns.tolist()}")
    log.info(f"Target: average_score | range {y.min():.1f} – {y.max():.1f}")
    return X, y


# ── Build preprocessor ─────────────────────────────────
def build_preprocessor(X):
    categorical_cols = X.select_dtypes(include='object').columns.tolist()
    numerical_cols   = X.select_dtypes(include='number').columns.tolist()

    log.info(f"Categorical columns: {categorical_cols}")
    log.info(f"Numerical columns:   {numerical_cols}")

    transformers = [
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ]

    return ColumnTransformer(transformers=transformers, remainder='drop')


# ── Train & compare all models ─────────────────────────
def train_all_models(X_train, X_test, y_train, y_test, preprocessor):

    models = {
        'Linear Regression':   LinearRegression(),
        'Decision Tree':       DecisionTreeRegressor(random_state=RANDOM_STATE),
        'Random Forest':       RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE),
        'Gradient Boosting':   GradientBoostingRegressor(n_estimators=100, random_state=RANDOM_STATE),
    }

    results = {}

    log.info("=" * 58)
    log.info(f"{'Model':<25} {'R2':>8} {'MAE':>8} {'RMSE':>8}")
    log.info("-" * 58)

    for name, model in models.items():
        pipe = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model',        model)
        ])

        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        r2   = round(r2_score(y_test, y_pred), 4)
        mae  = round(mean_absolute_error(y_test, y_pred), 4)
        rmse = round(mean_squared_error(y_test, y_pred) ** 0.5, 4)

        results[name] = {
            'pipeline': pipe,
            'R2':  r2,
            'MAE': mae,
            'RMSE': rmse
        }

        log.info(f"{name:<25} {r2:>8} {mae:>8} {rmse:>8}")

    log.info("=" * 58)
    return results


# ── Select best model ──────────────────────────────────
def select_best(results):
    best_name = max(results, key=lambda k: results[k]['R2'])
    best      = results[best_name]

    log.info(f"Best model  : {best_name}")
    log.info(f"R2 Score    : {best['R2']}")
    log.info(f"MAE         : {best['MAE']}")
    log.info(f"RMSE        : {best['RMSE']}")

    return best_name, best['pipeline']


# ── Save model ─────────────────────────────────────────
def save_model(pipeline):
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(pipeline, f)

    log.info(f"Model saved → {MODEL_PATH}")


# ── Verify saved model ─────────────────────────────────
def verify_model():
    with open(MODEL_PATH, 'rb') as f:
        loaded = pickle.load(f)

    sample = pd.DataFrame({
        'gender':                       ['female'],
        'race/ethnicity':               ['group B'],
        'parental_level_of_education':  ["bachelor's degree"],
        'lunch':                        ['standard'],
        'test_preparation_course':      ['completed']
    })

    score = round(float(loaded.predict(sample)[0]), 2)
    log.info(f"Verification prediction on sample student: {score} / 100")
    log.info("Model verified successfully ✓")


# ── Main entry point ───────────────────────────────────
def main():
    log.info("━" * 58)
    log.info("  STUDENT PERFORMANCE PREDICTOR — Model Training")
    log.info("━" * 58)

    # 1. Load
    df = load_data()

    # 2. Features
    X, y = build_features(df)

    # 3. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    log.info(f"Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")

    # 4. Preprocessor
    preprocessor = build_preprocessor(X_train)

    # 5. Train all models
    results = train_all_models(X_train, X_test, y_train, y_test, preprocessor)

    # 6. Select best
    best_name, best_pipeline = select_best(results)

    # 7. Save
    save_model(best_pipeline)

    # 8. Verify
    verify_model()

    log.info("━" * 58)
    log.info("  Training complete! Run  python app.py  to start the app.")
    log.info("━" * 58)


if __name__ == '__main__':
    main()