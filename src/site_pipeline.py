"""End-to-end site pipeline: preprocess, train, evaluate and save models for O3 and NO2.

Usage example:
python src/site_pipeline.py --input site_1_train_data.csv --out_dir models/site_1 --test_size 0.25 --random_state 42
"""
"""End-to-end site pipeline: preprocess, train, evaluate and save models for O3 and NO2.

Usage example:
  python src/site_pipeline.py --input site_1_train_data.csv --out_dir models/site_1 --test_size 0.25 --random_state 42
"""
import argparse
import os
import json
import joblib
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


def ria_score(y_true, y_pred):
    """Refined Index of Agreement (Willmott refined d1 variant using absolute errors)."""
    num = np.sum(np.abs(y_pred - y_true))
    ybar = np.mean(y_true)
    den = np.sum(np.abs(y_pred - ybar) + np.abs(y_true - ybar))
    if den == 0:
        return np.nan
    return 1.0 - (num / den)


def build_datetime(df):
    df = df.copy()
    for col in ['year', 'month', 'day', 'hour']:
        if col in df.columns:
            # safe cast
            df[col] = df[col].astype(int)
    df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
    df = df.set_index('datetime').sort_index()
    return df


def align_daily_satellite(df, sat_cols):
    if not sat_cols:
        return df
    df = df.copy()
    df['date'] = df.index.date
    for c in sat_cols:
        if c in df.columns:
            daily = df.groupby('date')[c].first()
            df[c] = df['date'].map(daily.to_dict())
    df = df.drop(columns=['date'])
    return df


def add_past_target_features(df, targets, lags=[1,2,3,6,12,24], rolls=[24,168]):
    df = df.copy()
    for t in targets:
        if t not in df.columns:
            df[t] = np.nan
        for lag in lags:
            df[f'{t}_lag_{lag}'] = df[t].shift(lag)
        for w in rolls:
            df[f'{t}_roll_{w}_mean'] = df[t].rolling(window=w, min_periods=1).mean()
    return df


def split_days_mask(df, test_size=0.25, seed=42):
    dates = np.array(sorted(pd.Series(df.index.date).unique()))
    train_days, test_days = train_test_split(dates, test_size=test_size, random_state=seed)
    mask = df.index.date
    train_mask = np.isin(mask, train_days)
    return train_mask


def train_models(X_train, y_train, X_val, y_val, params=None, num_boost=1000, early_stopping=50):
    # Use sklearn API which supports early_stopping_rounds across different lightgbm builds
    params = params or {'learning_rate': 0.05, 'num_leaves': 31}
    models = {}
    preds = {}
    for col in y_train.columns:
        model = lgb.LGBMRegressor(**params, n_estimators=num_boost)
        Xtr = X_train.fillna(-999)
        Xv = X_val.fillna(-999)
        ytr = y_train[col].fillna(-999)
        yv = y_val[col].fillna(-999)
        try:
            # some lightgbm versions accept early_stopping_rounds directly
            model.fit(Xtr, ytr, eval_set=[(Xv, yv)], early_stopping_rounds=early_stopping, verbose=False)
        except TypeError:
            # fallback to callbacks API
            callbacks = [lgb.early_stopping(stopping_rounds=early_stopping), lgb.log_evaluation(period=0)]
            model.fit(Xtr, ytr, eval_set=[(Xv, yv)], callbacks=callbacks)
        models[col] = model
        preds[col] = model.predict(Xv)
    return models, preds


def evaluate(y_true, y_pred):
    metrics = {}
    for col in y_true.columns:
        y_t = y_true[col].values
        y_p = y_pred[col]
        rmse = np.sqrt(mean_squared_error(y_t, y_p))
        mae = mean_absolute_error(y_t, y_p)
        bias = float(np.mean(y_p - y_t))
        r2 = r2_score(y_t, y_p)
        ria = ria_score(y_t, y_p)
        metrics[col] = {'rmse': float(rmse), 'mae': float(mae), 'bias': bias, 'r2': float(r2), 'ria': float(ria)}
    return metrics


def pipeline(input_csv, out_dir, test_size=0.25, random_state=42):
    print('Loading', input_csv)
    df = pd.read_csv(input_csv)
    df = build_datetime(df)

    # normalize column names
    df.columns = [c.strip() for c in df.columns]

    targets = [c for c in df.columns if c.endswith('_target')]
    if not set(['O3_target', 'NO2_target']).issubset(set(targets)):
        raise ValueError('Expected O3_target and NO2_target in data')

    sat_cols = [c for c in ['NO2_satellite', 'HCHO_satellite', 'ratio_satellite'] if c in df.columns]
    df = align_daily_satellite(df, sat_cols)

    df = add_past_target_features(df, ['O3_target', 'NO2_target'])

    forecast_cols = [c for c in df.columns if c.endswith('_forecast')]
    lag_cols = [c for c in df.columns if ('_lag_' in c) or ('_roll_' in c)]
    time_cols = ['year', 'month', 'day', 'hour']
    features = forecast_cols + sat_cols + lag_cols + time_cols
    features = [f for f in features if f in df.columns]

    df = df.dropna(subset=['O3_target', 'NO2_target'])
    df = df.dropna(subset=lag_cols, how='any')

    df['date_only'] = df.index.date
    train_mask = df['date_only'].isin(train_test_split(np.array(sorted(df['date_only'].unique())), test_size=test_size, random_state=random_state)[0])

    X_train = df.loc[train_mask, features]
    y_train = df.loc[train_mask, ['O3_target', 'NO2_target']]
    X_val = df.loc[~train_mask, features]
    y_val = df.loc[~train_mask, ['O3_target', 'NO2_target']]

    print('Training rows:', len(X_train), 'Validation rows:', len(X_val))

    models, preds = train_models(X_train, y_train, X_val, y_val)
    y_pred = pd.DataFrame(preds, index=y_val.index)
    # save validation predictions + observations for downstream analysis
    try:
        vp = y_val.copy()
        vp_preds = y_pred.copy()
        vp = vp.join(vp_preds, how='left', rsuffix='_pred')
        vp_reset = vp.reset_index().rename(columns={'index': 'datetime'})
        vp_path = os.path.join(out_dir, 'validation_predictions.csv')
        vp_reset.to_csv(vp_path, index=False)
        print('Saved validation predictions to', vp_path)
    except Exception:
        pass

    metrics = evaluate(y_val, y_pred)
    print('Evaluation metrics:')
    for k, v in metrics.items():
        print(k, v)

    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, 'site_models.joblib')
    joblib.dump({'models': models, 'features': features, 'metrics': metrics}, model_path)
    print('Saved models to', model_path)
    metrics_path = os.path.join(out_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print('Saved metrics to', metrics_path)
    # Validation report: per-hour statistics
    try:
        vr = []
        for hour in sorted(y_val.index.hour.unique()):
            mask = y_val.index.hour == hour
            row = {}
            row['hour'] = int(hour)
            for col in y_val.columns:
                y_t = y_val.loc[mask, col]
                y_p = y_pred.loc[mask, col]
                row[f'{col}_rmse'] = float(np.sqrt(mean_squared_error(y_t, y_p)))
                row[f'{col}_mae'] = float(mean_absolute_error(y_t, y_p))
                row[f'{col}_bias'] = float((y_p - y_t).mean())
            vr.append(row)
        vrdf = pd.DataFrame(vr).sort_values('hour')
        vrpath = os.path.join(out_dir, 'validation_report.csv')
        vrdf.to_csv(vrpath, index=False)
        print('Saved validation report to', vrpath)
    except Exception:
        pass
    # Monthly and seasonal breakdowns
    try:
        month_rows = []
        for m in sorted(y_val.index.month.unique()):
            mask = y_val.index.month == m
            row = {'month': int(m)}
            for col in y_val.columns:
                y_t = y_val.loc[mask, col]
                y_p = y_pred.loc[mask, col]
                row[f'{col}_rmse'] = float(np.sqrt(mean_squared_error(y_t, y_p)))
                row[f'{col}_mae'] = float(mean_absolute_error(y_t, y_p))
                row[f'{col}_bias'] = float((y_p - y_t).mean())
            month_rows.append(row)
        month_df = pd.DataFrame(month_rows).sort_values('month')
        month_path = os.path.join(out_dir, 'monthly_report.csv')
        month_df.to_csv(month_path, index=False)
        print('Saved monthly report to', month_path)

        # seasons mapping
        seasons = {'DJF': [12,1,2], 'MAM': [3,4,5], 'JJA': [6,7,8], 'SON': [9,10,11]}
        season_rows = []
        for s, months in seasons.items():
            mask = y_val.index.month.isin(months)
            if mask.sum() == 0:
                continue
            row = {'season': s}
            for col in y_val.columns:
                y_t = y_val.loc[mask, col]
                y_p = y_pred.loc[mask, col]
                row[f'{col}_rmse'] = float(np.sqrt(mean_squared_error(y_t, y_p)))
                row[f'{col}_mae'] = float(mean_absolute_error(y_t, y_p))
                row[f'{col}_bias'] = float((y_p - y_t).mean())
            season_rows.append(row)
        season_df = pd.DataFrame(season_rows)
        season_path = os.path.join(out_dir, 'seasonal_report.csv')
        season_df.to_csv(season_path, index=False)
        print('Saved seasonal report to', season_path)
    except Exception:
        pass
    return model_path, metrics


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True, help='site CSV input')
    p.add_argument('--out_dir', required=True)
    p.add_argument('--test_size', type=float, default=0.25)
    p.add_argument('--random_state', type=int, default=42)
    args = p.parse_args()
    pipeline(args.input, args.out_dir, args.test_size, args.random_state)


def run(site_csv, unseen_csv, out_dir, config_path=None):
    """Train on `site_csv`, save models to `out_dir`, and predict on `unseen_csv` if provided.

    Returns (model_path, metrics)
    """
    # train and save
    model_path, metrics = pipeline(site_csv, out_dir)

    # predict unseen if provided
    if unseen_csv and os.path.exists(unseen_csv):
        # load model bundle
        bundle = joblib.load(model_path)
        models = bundle['models']
        features = bundle.get('features', None)

        # load history and unseen
        hist = pd.read_csv(site_csv)
        hist = build_datetime(hist)
        unseen = pd.read_csv(unseen_csv)
        unseen = build_datetime(unseen)

        # align satellite and attach history to compute lag features
        sat_cols = [c for c in ['NO2_satellite', 'HCHO_satellite', 'ratio_satellite'] if c in hist.columns or c in unseen.columns]
        hist = align_daily_satellite(hist, sat_cols)
        unseen = align_daily_satellite(unseen, sat_cols)

        combined = pd.concat([hist, unseen])
        combined = add_past_target_features(combined, ['O3_target', 'NO2_target'])

        # select features for unseen
        if features is None:
            # fallback to all columns except targets
            features = [c for c in combined.columns if c not in ['O3_target', 'NO2_target']]

        X_unseen = combined.loc[unseen.index, features].fillna(-999)
        # If there are duplicated datetimes (e.g., same datetimes in history and unseen), keep the last (unseen)
        if X_unseen.index.duplicated().any():
            X_unseen = X_unseen.groupby(level=0).last()

        preds = {t: models[t].predict(X_unseen) for t in models}
        pred_df = pd.DataFrame(preds, index=X_unseen.index)
        out_pred = os.path.join(out_dir, 'unseen_predictions.csv')
        pred_df.to_csv(out_pred, index_label='datetime')
        print('Wrote predictions to', out_pred)

    return model_path, metrics
