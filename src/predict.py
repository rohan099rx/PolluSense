"""Load trained models and produce hourly forecasts for N steps from a given start time.
"""
import argparse
import yaml
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def load_model(path):
    return joblib.load(path)


def recursive_forecast(models_bundle, history_df, steps, features):
    # naive autoregressive forecast using lag features present in history_df
    from src.site_pipeline import add_past_target_features
    
    preds = []
    df = history_df.copy()
    target_cols = ['O3_target', 'NO2_target']
    
    for s in range(steps):
        feat_row = df.iloc[-1:][features].fillna(-999)
        row_pred = {}
        for t, m in models_bundle['models'].items():
            # LightGBM expects 2D
            p = m.predict(feat_row)[0]
            row_pred[t] = p
        # append predictions as new row (datetime index incremented)
        next_idx = df.index[-1] + pd.Timedelta('1h')
        new_row = pd.Series(row_pred, name=next_idx)
        # ensure all feature columns exist
        for col in features:
            if col not in new_row.index:
                new_row[col] = np.nan
        df = pd.concat([df, new_row.to_frame().T])
        
        # CRITICAL FIX: Recalculate lag features after adding the new prediction
        # Without this, lag features remain NaN and model predictions become garbage
        df = add_past_target_features(df, target_cols)
        
        preds.append((next_idx, row_pred))
    return preds


def main(config_path, model_path, start, steps, history_path=None):
    cfg = yaml.safe_load(open(config_path))
    models = load_model(model_path)
    # load history to build features
    if history_path:
        hist = pd.read_parquet(history_path)
    else:
        # use model training split as fallback
        raise ValueError('history_path required for now')

    hist = hist.sort_index()
    # build list of features from training data columns excluding targets
    targets = cfg['model']['target_vars']
    features = [c for c in hist.columns if c not in targets]

    # ensure start is present in history index
    start = pd.to_datetime(start)
    if start not in hist.index:
        # find nearest earlier index
        hist = hist[hist.index <= start]
    preds = recursive_forecast(models, hist, steps, features)
    for idx, row in preds:
        print(idx.isoformat(), row)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--start', required=True)
    p.add_argument('--steps', type=int, default=24)
    p.add_argument('--history', required=True)
    args = p.parse_args()
    main(args.config, args.model, args.start, args.steps, args.history)
