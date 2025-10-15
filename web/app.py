import streamlit as st
from pathlib import Path
import joblib
import pandas as pd
import sys
from datetime import datetime, time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for better rendering

# ensure repo root is importable
repo_root = str(Path(__file__).resolve().parents[1])
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.predict import recursive_forecast
from src.site_pipeline import add_past_target_features


st.set_page_config(page_title='PolluSense Forecast', layout='wide', initial_sidebar_state='expanded')
st.title('🌍 PolluSense — Air Quality Forecast Dashboard')
st.markdown('**24/48-hour O₃ and NO₂ hourly forecasting for Delhi**')
st.markdown('---')

# Site selector: list subfolders in models/
models_root = Path('models')
sites = [p.name for p in models_root.iterdir() if p.is_dir()]
# Default to cleaned model if available
default_site = 'site_1_cleaned' if 'site_1_cleaned' in sites else (sites[0] if sites else 'site_1')
site_choice = st.sidebar.selectbox('🏭 Select Site', sites if sites else ['site_1'], index=sites.index(default_site) if default_site in sites else 0, help='Choose monitoring site')
site_dir = models_root / site_choice
model_path = site_dir / 'site_models.joblib'
metrics_path = site_dir / 'metrics.json'

# Use cleaned data by default if selecting site_1_cleaned
train_csv_default = 'site_1_train_data_cleaned.csv' if site_choice == 'site_1_cleaned' else 'site_1_train_data.csv'
train_csv = st.sidebar.text_input('📁 Training CSV (for history)', value=train_csv_default)

if not model_path.exists():
    st.sidebar.error(f'❌ Model bundle not found at {model_path}. Run training first.')
    st.error(f'**Model not found:** {model_path}\n\nPlease train the model first using `src/site_pipeline.py`')
    st.stop()

@st.cache_resource
def load_model_bundle(path):
    return joblib.load(path)

@st.cache_data
def load_history_parquet(train_csv_path):
    df = pd.read_csv(train_csv_path)
    for col in ['year', 'month', 'day', 'hour']:
        if col in df.columns:
            df[col] = df[col].astype(int)
    df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
    df = df.set_index('datetime').sort_index()
    # Add lag/roll features
    df = add_past_target_features(df, ['O3_target', 'NO2_target'])
    return df

@st.cache_data
def compute_forecast(_models_bundle, history_df, start_ts, steps, features):
    # use leading-underscore for models bundle so Streamlit doesn't attempt to hash it
    hist = history_df.copy()
    start_ts = pd.to_datetime(start_ts)
    if start_ts not in hist.index:
        hist = hist[hist.index <= start_ts]
    preds = recursive_forecast(_models_bundle, hist, steps, features)
    rows = []
    for idx, row in preds:
        r = {'datetime': idx}
        r.update(row)
        rows.append(r)
    outdf = pd.DataFrame(rows).set_index('datetime')
    
    # Rename columns to match plotting expectations
    outdf = outdf.rename(columns={
        'O3_target': 'O3_target_pred',
        'NO2_target': 'NO2_target_pred'
    })
    return outdf

def load_metrics(path):
    try:
        return pd.read_json(path)
    except Exception:
        return None

def get_feature_importances(models):
    # models is dict of target -> model
    fi = {}
    for t, m in models.items():
        try:
            arr = np.array(m.feature_importances_)
            cols = getattr(m, 'feature_name_', None) or None
            fi[t] = (cols, arr)
        except Exception:
            fi[t] = (None, None)
    return fi

def plot_styled_forecast(outdf, title='Forecast'):
    """Create a professional-looking forecast plot"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.patch.set_facecolor('white')
    
    # O3 plot
    if 'O3_target_pred' in outdf.columns:
        ax1.plot(outdf.index, outdf['O3_target_pred'], 
                marker='o', markersize=4, linewidth=2, 
                color='#2E86AB', label='O₃ Forecast')
        ax1.fill_between(outdf.index, 0, outdf['O3_target_pred'], 
                         alpha=0.3, color='#2E86AB')
        ax1.set_ylabel('O₃ (µg/m³)', fontsize=12, fontweight='bold')
        ax1.legend(loc='upper right', fontsize=10)
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.set_title('Ozone (O₃) Forecast', fontsize=14, fontweight='bold', pad=10)
    
    # NO2 plot
    if 'NO2_target_pred' in outdf.columns:
        ax2.plot(outdf.index, outdf['NO2_target_pred'], 
                marker='s', markersize=4, linewidth=2, 
                color='#A23B72', label='NO₂ Forecast')
        ax2.fill_between(outdf.index, 0, outdf['NO2_target_pred'], 
                         alpha=0.3, color='#A23B72')
        ax2.set_ylabel('NO₂ (µg/m³)', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Datetime', fontsize=12, fontweight='bold')
        ax2.legend(loc='upper right', fontsize=10)
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.set_title('Nitrogen Dioxide (NO₂) Forecast', fontsize=14, fontweight='bold', pad=10)
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return fig

def plot_feature_importance(fi_dict, target_name):
    """Create enhanced feature importance plot"""
    cols, arr = fi_dict.get(target_name, (None, None))
    if arr is None or cols is None:
        return None
    
    df_fi = pd.DataFrame({'feature': cols, 'importance': arr})
    df_fi = df_fi.sort_values('importance', ascending=False).head(15)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(df_fi)))
    bars = ax.barh(df_fi['feature'], df_fi['importance'], color=colors)
    ax.set_xlabel('Importance', fontsize=12, fontweight='bold')
    ax.set_title(f'Top 15 Features — {target_name}', fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, axis='x', linestyle='--')
    ax.invert_yaxis()
    plt.tight_layout()
    return fig


bundle = None
if model_path.exists():
    bundle = load_model_bundle(str(model_path))
    models = bundle.get('models')
    features = bundle.get('features')
else:
    models = {}
    features = None

st.sidebar.markdown('---')
st.sidebar.header('⚙️ Forecast Controls')

today = pd.Timestamp.now().normalize()
date_val = st.sidebar.date_input('📅 Start date', value=today)
time_val = st.sidebar.time_input('🕐 Start time', value=time(0, 0))
start_dt = datetime.combine(date_val, time_val)

horizon = st.sidebar.radio('🔭 Forecast Horizon', (24, 48), index=0, help='Choose 24h or 48h forecast')
run_btn = st.sidebar.button('▶️ Run Forecast', type='primary', use_container_width=True)

# Model info with better formatting
st.sidebar.markdown('---')
st.sidebar.header('📊 Model Performance')
if metrics_path.exists():
    try:
        import json
        metrics = json.loads(open(metrics_path).read())
        for target, vals in metrics.items():
            with st.sidebar.expander(f'**{target}**', expanded=False):
                for k, v in vals.items():
                    if isinstance(v, float):
                        st.write(f'**{k}:** {v:.4f}')
                    else:
                        st.write(f'**{k}:** {v}')
    except Exception:
        st.sidebar.write('Could not read metrics')
else:
    st.sidebar.write('No metrics.json found')

# Accuracy report
accuracy_path = site_dir / 'accuracy_report.json'
if accuracy_path.exists():
    st.sidebar.markdown('---')
    st.sidebar.header('🎯 Accuracy Summary')
    try:
        import json
        acc = json.loads(open(accuracy_path).read())
        for pollutant, metrics in acc.items():
            with st.sidebar.expander(f'**{pollutant}**', expanded=False):
                st.metric('RMSE', f"{metrics.get('rmse', 0):.2f} µg/m³")
                st.metric('MAE', f"{metrics.get('mae', 0):.2f} µg/m³")
                st.metric('R²', f"{metrics.get('r2', 0):.3f}")
                st.metric('% within ±10 µg/m³', f"{metrics.get('pct_within_abs_10ug/m3', 0):.1f}%")
                st.metric('% within ±20%', f"{metrics.get('pct_within_rel_20pct', 0):.1f}%")
    except Exception as e:
        st.sidebar.write(f'Could not load accuracy: {e}')


# Main content area
tab1, tab2, tab3 = st.tabs(['📈 Forecast', '🔍 Feature Importance', '🧠 SHAP Analysis'])

with tab1:
    st.header(f'Forecast Output — {site_choice}')
    
    if run_btn:
        try:
            with st.spinner('Loading history and computing forecast...'):
                hist = load_history_parquet(train_csv)
                models_bundle = {'models': models}
                feat_list = features or [c for c in hist.columns if not c.endswith('_target')]
                outdf = compute_forecast(models_bundle, hist, start_dt.isoformat(), horizon, feat_list)
            
            st.success(f'✅ Generated {horizon}-hour forecast from {start_dt}')
            
            # Debug: Show what columns we have
            if len(outdf.columns) == 0:
                st.error("❌ Forecast dataframe is empty! No columns generated.")
                st.write("Debug info:")
                st.write(f"- Features used: {len(feat_list)}")
                st.write(f"- Models available: {list(models.keys())}")
                st.stop()
            
            # Show styled forecast plot
            fig = plot_styled_forecast(outdf)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
            
        except Exception as e:
            st.error(f"❌ Error generating forecast: {str(e)}")
            st.write("Debug information:")
            st.write(f"- Train CSV: {train_csv}")
            st.write(f"- Start datetime: {start_dt}")
            st.write(f"- Horizon: {horizon}")
            st.write(f"- Models loaded: {list(models.keys()) if models else 'No models'}")
            import traceback
            st.code(traceback.format_exc())
            st.stop()
        
        # Statistics summary
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if 'O3_target_pred' in outdf.columns:
                st.metric('O₃ Mean', f"{outdf['O3_target_pred'].mean():.2f} µg/m³")
        with col2:
            if 'O3_target_pred' in outdf.columns:
                st.metric('O₃ Max', f"{outdf['O3_target_pred'].max():.2f} µg/m³")
        with col3:
            if 'NO2_target_pred' in outdf.columns:
                st.metric('NO₂ Mean', f"{outdf['NO2_target_pred'].mean():.2f} µg/m³")
        with col4:
            if 'NO2_target_pred' in outdf.columns:
                st.metric('NO₂ Max', f"{outdf['NO2_target_pred'].max():.2f} µg/m³")
        
        st.markdown('---')
        st.subheader('📋 Forecast Data Table')
        st.dataframe(outdf.reset_index(), use_container_width=True, height=400)
        
        # Download button
        csv_data = outdf.reset_index().to_csv(index=False)
        st.download_button(
            label='⬇️ Download Forecast CSV',
            data=csv_data,
            file_name=f'forecast_{site_choice}_{horizon}h_{start_dt.strftime("%Y%m%d_%H%M")}.csv',
            mime='text/csv',
            use_container_width=True
        )
    else:
        st.info('👈 Set start date/time and click **Run Forecast** to generate predictions.')
        
        # Show validation sample
        vp = site_dir / 'validation_predictions.csv'
        if vp.exists():
            st.markdown('### Recent Validation Performance (Sample)')
            df_vp = pd.read_csv(vp, parse_dates=['datetime']).set_index('datetime')
            
            # Plot validation sample
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
            sample = df_vp.head(168)  # 1 week
            
            ax1.plot(sample.index, sample['O3_target'], 'o-', label='Observed', alpha=0.7, markersize=3)
            ax1.plot(sample.index, sample['O3_target_pred'], 's-', label='Predicted', alpha=0.7, markersize=3)
            ax1.set_ylabel('O₃ (µg/m³)', fontweight='bold')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_title('O₃ Validation Sample (1 week)', fontweight='bold')
            
            ax2.plot(sample.index, sample['NO2_target'], 'o-', label='Observed', alpha=0.7, markersize=3)
            ax2.plot(sample.index, sample['NO2_target_pred'], 's-', label='Predicted', alpha=0.7, markersize=3)
            ax2.set_ylabel('NO₂ (µg/m³)', fontweight='bold')
            ax2.set_xlabel('Datetime', fontweight='bold')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.set_title('NO₂ Validation Sample (1 week)', fontweight='bold')
            
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

with tab2:
    st.header('🔍 Feature Importance Analysis')
    
    if models:
        fi = get_feature_importances(models)
        
        for target in models.keys():
            st.subheader(f'{target}')
            fig = plot_feature_importance(fi, target)
            if fig:
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
            else:
                st.warning(f'No feature importance data available for {target}')
    else:
        st.warning('No models loaded. Train a model first.')

with tab3:
    st.header('🧠 SHAP (SHapley Additive exPlanations) Analysis')
    st.markdown("""
    **SHAP values** explain predictions by computing the contribution of each feature to the model output.
    - **Positive SHAP values** (red) → feature pushes prediction higher
    - **Negative SHAP values** (blue) → feature pushes prediction lower
    - **Feature value** (color intensity) → shows if feature value is high or low
    
    This analysis is computationally intensive and may take 30-60 seconds.
    """)
    
    # SHAP configuration
    col_shap1, col_shap2 = st.columns(2)
    with col_shap1:
        sample_size = st.slider('Sample size (for performance)', 100, 1000, 500, 100,
                                help='Larger samples = more accurate but slower')
    with col_shap2:
        top_n_features = st.slider('Top N features to display', 5, 20, 10, 1,
                                   help='Number of most important features to show')
    
    shap_btn = st.button('🔄 Compute SHAP Values', type='primary', use_container_width=True)
    
    if shap_btn:
        try:
            import shap
            
            with st.spinner('📊 Loading history data...'):
                hist = load_history_parquet(train_csv)
                hist_clean = hist.dropna()
                
                if len(hist_clean) < sample_size:
                    st.warning(f'⚠️ Only {len(hist_clean)} samples available (requested {sample_size})')
                    sample_size = len(hist_clean)
                
                hist_sample = hist_clean.sample(min(sample_size, len(hist_clean)), random_state=42)
                st.info(f'📌 Using {len(hist_sample)} samples for SHAP analysis')
            
            for target_name, model in models.items():
                st.markdown('---')
                st.subheader(f'🎯 SHAP Analysis — {target_name}')
                
                with st.spinner(f'🔄 Computing SHAP values for {target_name}... (may take 30-60s)'):
                    # Create explainer
                    explainer = shap.TreeExplainer(model)
                    X_sample = hist_sample[features].fillna(-999)
                    shap_values = explainer.shap_values(X_sample)
                    
                    # Get expected value (baseline prediction)
                    expected_value = explainer.expected_value
                
                # Tab layout for different SHAP visualizations
                shap_tab1, shap_tab2, shap_tab3, shap_tab4, shap_tab5 = st.tabs([
                    '📊 Summary Plot', 
                    '📈 Bar Chart', 
                    '🎯 Waterfall', 
                    '🔥 Dependence Plot',
                    '📋 Data Table'
                ])
                
                with shap_tab1:
                    st.markdown('#### SHAP Summary Plot (Beeswarm)')
                    st.markdown("""
                    Each dot represents a sample:
                    - **X-axis**: SHAP value (impact on prediction)
                    - **Y-axis**: Feature name (sorted by importance)
                    - **Color**: Feature value (red=high, blue=low)
                    """)
                    
                    # Summary plot creates its own figure
                    plt.figure(figsize=(14, 8))
                    shap.summary_plot(shap_values, X_sample, max_display=top_n_features, 
                                    show=False, plot_size=(14, 8))
                    plt.tight_layout()
                    
                    # Get current figure and display it
                    fig_summary = plt.gcf()
                    st.pyplot(fig_summary, use_container_width=True)
                    plt.close(fig_summary)
                
                with shap_tab2:
                    st.markdown('#### Mean Absolute SHAP Values (Feature Importance)')
                    st.markdown("""
                    Shows average impact magnitude of each feature (regardless of direction).
                    Features with larger bars have more influence on predictions.
                    """)
                    
                    # Bar plot creates its own figure
                    plt.figure(figsize=(12, 6))
                    shap.summary_plot(shap_values, X_sample, plot_type='bar', 
                                    max_display=top_n_features, show=False)
                    plt.tight_layout()
                    
                    # Get current figure and display it
                    fig_bar = plt.gcf()
                    st.pyplot(fig_bar, use_container_width=True)
                    plt.close(fig_bar)
                    
                    # Calculate and show feature importance table
                    mean_abs_shap = np.abs(shap_values).mean(axis=0)
                    importance_df = pd.DataFrame({
                        'Feature': features,
                        'Mean |SHAP|': mean_abs_shap
                    }).sort_values('Mean |SHAP|', ascending=False).head(top_n_features)
                    
                    st.markdown(f'##### Top {top_n_features} Most Important Features')
                    st.dataframe(importance_df.reset_index(drop=True), use_container_width=True)
                
                with shap_tab3:
                    st.markdown('#### SHAP Waterfall Plot (Single Prediction Explanation)')
                    st.markdown("""
                    Shows how each feature contributes to moving the prediction from the 
                    base value (average prediction) to the final prediction for one sample.
                    """)
                    
                    # Select a sample to explain
                    sample_idx = st.selectbox(
                        'Select sample to explain:',
                        range(min(20, len(hist_sample))),
                        format_func=lambda x: f"Sample {x} (actual: {hist_sample[target_name].iloc[x]:.2f})"
                    )
                    
                    # Create SHAP Explanation object
                    explanation = shap.Explanation(
                        values=shap_values[sample_idx],
                        base_values=expected_value,
                        data=X_sample.iloc[sample_idx],
                        feature_names=features
                    )
                    
                    # Waterfall plot creates its own figure
                    plt.figure(figsize=(12, 8))
                    shap.waterfall_plot(explanation, max_display=top_n_features, show=False)
                    plt.tight_layout()
                    
                    # Get current figure and display it
                    fig_waterfall = plt.gcf()
                    st.pyplot(fig_waterfall, use_container_width=True)
                    plt.close(fig_waterfall)
                    
                    # Show prediction details
                    actual_pred = model.predict(X_sample.iloc[sample_idx:sample_idx+1])[0]
                    st.info(f"""
                    **Prediction Details:**
                    - Base value (avg): {expected_value:.2f} µg/m³
                    - Final prediction: {actual_pred:.2f} µg/m³
                    - Actual value: {hist_sample[target_name].iloc[sample_idx]:.2f} µg/m³
                    - Prediction error: {abs(actual_pred - hist_sample[target_name].iloc[sample_idx]):.2f} µg/m³
                    """)
                
                with shap_tab4:
                    st.markdown('#### SHAP Dependence Plot')
                    st.markdown("""
                    Shows how a feature's value affects predictions across all samples.
                    - **X-axis**: Feature value
                    - **Y-axis**: SHAP value (impact on prediction)
                    - **Color**: Another feature that may interact with this one
                    """)
                    
                    # Get top features for dropdown
                    mean_abs_shap = np.abs(shap_values).mean(axis=0)
                    top_features = np.array(features)[np.argsort(-mean_abs_shap)][:10]
                    
                    selected_feature = st.selectbox(
                        'Select feature to analyze:',
                        top_features,
                        help='Shows how this feature affects predictions'
                    )
                    
                    feature_idx = features.index(selected_feature)
                    
                    # Dependence plot creates its own figure
                    plt.figure(figsize=(12, 6))
                    shap.dependence_plot(
                        feature_idx,
                        shap_values,
                        X_sample,
                        show=False
                    )
                    plt.tight_layout()
                    
                    # Get current figure and display it
                    fig_dep = plt.gcf()
                    st.pyplot(fig_dep, use_container_width=True)
                    plt.close(fig_dep)
                    
                    # Statistics for selected feature
                    col_dep1, col_dep2, col_dep3, col_dep4 = st.columns(4)
                    with col_dep1:
                        st.metric('Feature Mean', f"{X_sample[selected_feature].mean():.2f}")
                    with col_dep2:
                        st.metric('Feature Std', f"{X_sample[selected_feature].std():.2f}")
                    with col_dep3:
                        st.metric('Mean SHAP', f"{shap_values[:, feature_idx].mean():.3f}")
                    with col_dep4:
                        st.metric('SHAP Std', f"{shap_values[:, feature_idx].std():.3f}")
                
                with shap_tab5:
                    st.markdown('#### SHAP Values Data Table')
                    st.markdown('Raw SHAP values for all samples and features (top 10 features shown)')
                    
                    # Create DataFrame with SHAP values
                    top_feature_indices = np.argsort(-np.abs(shap_values).mean(axis=0))[:10]
                    top_feature_names = [features[i] for i in top_feature_indices]
                    
                    shap_df = pd.DataFrame(
                        shap_values[:, top_feature_indices],
                        columns=top_feature_names
                    )
                    shap_df['Prediction'] = model.predict(X_sample)
                    shap_df['Actual'] = hist_sample[target_name].values
                    shap_df['Error'] = shap_df['Prediction'] - shap_df['Actual']
                    
                    st.dataframe(shap_df, use_container_width=True, height=400)
                    
                    # Download SHAP values
                    csv_shap = shap_df.to_csv(index=False)
                    st.download_button(
                        label='⬇️ Download SHAP Values CSV',
                        data=csv_shap,
                        file_name=f'shap_values_{target_name}_{site_choice}.csv',
                        mime='text/csv',
                        use_container_width=True
                    )
                
                st.success(f'✅ SHAP analysis complete for {target_name} ({len(hist_sample)} samples analyzed)')
        
        except ImportError:
            st.error('❌ SHAP library not installed.')
            st.code('pip install shap', language='bash')
            st.markdown('[Install SHAP](https://github.com/slundberg/shap) to enable explainability features.')
        except Exception as e:
            st.error(f'❌ SHAP computation failed: {str(e)}')
            import traceback
            with st.expander('Show error details'):
                st.code(traceback.format_exc())
    else:
        st.info('👆 Click **Compute SHAP Values** above to start the analysis.')
        st.markdown("""
        ### What You'll Get:
        
        1. **📊 Summary Plot**: Overview of feature impacts across all samples
        2. **📈 Bar Chart**: Ranking of features by importance
        3. **🎯 Waterfall Plot**: Detailed explanation of individual predictions
        4. **🔥 Dependence Plot**: How feature values affect predictions
        5. **📋 Data Table**: Raw SHAP values for further analysis
        
        ### Why Use SHAP?
        
        - **Model Transparency**: Understand what drives your predictions
        - **Feature Selection**: Identify which features matter most
        - **Trust & Validation**: Verify model makes sensible decisions
        - **Debugging**: Find issues like data leakage or spurious correlations
        """)

# Footer with monthly/seasonal reports
st.markdown('---')
with st.expander('📊 View Monthly & Seasonal Error Reports', expanded=False):
    col_a, col_b = st.columns(2)
    
    with col_a:
        mr = site_dir / 'monthly_report.csv'
        if mr.exists():
            st.markdown('#### Monthly RMSE')
            mdf = pd.read_csv(mr)
            fig_m, ax = plt.subplots(figsize=(10, 5))
            x = np.arange(len(mdf))
            width = 0.35
            ax.bar(x - width/2, mdf['O3_target_rmse'], width, label='O₃', color='#2E86AB')
            ax.bar(x + width/2, mdf['NO2_target_rmse'], width, label='NO₂', color='#A23B72')
            ax.set_xlabel('Month', fontweight='bold')
            ax.set_ylabel('RMSE (µg/m³)', fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(mdf['month'], rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_title('Monthly RMSE Comparison', fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig_m, use_container_width=True)
            plt.close(fig_m)
        else:
            st.info('No monthly report available')
    
    with col_b:
        sr = site_dir / 'seasonal_report.csv'
        if sr.exists():
            st.markdown('#### Seasonal RMSE')
            sdf = pd.read_csv(sr)
            fig_s, ax = plt.subplots(figsize=(10, 5))
            x = np.arange(len(sdf))
            width = 0.35
            ax.bar(x - width/2, sdf['O3_target_rmse'], width, label='O₃', color='#2E86AB')
            ax.bar(x + width/2, sdf['NO2_target_rmse'], width, label='NO₂', color='#A23B72')
            ax.set_xlabel('Season', fontweight='bold')
            ax.set_ylabel('RMSE (µg/m³)', fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(sdf['season'])
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_title('Seasonal RMSE Comparison', fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig_s, use_container_width=True)
            plt.close(fig_s)
        else:
            st.info('No seasonal report available')

st.markdown('---')
st.caption('🤖 Forecasts are autoregressive and deterministic. Results are cached for faster repeated runs. | PolluSense v1.0')
