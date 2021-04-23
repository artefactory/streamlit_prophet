import pandas as pd
from lib.utils.mapping import convert_into_nb_of_days, convert_into_nb_of_seconds
from datetime import timedelta


def get_forecast_components(models: dict, forecasts: dict) -> pd.DataFrame:
    """
    Return a dataframe with only the relevant components to sum to get the prediction
    """
    fcst = forecasts['eval'].copy()
    components_col_names = get_forecast_components_col_names(fcst) + ['ds']
    components = fcst[components_col_names]
    for col in components_col_names:
        if col in models['eval'].component_modes["multiplicative"]:
            components[col] *= components["trend"]
    components = components.set_index("ds")
    return components


def get_forecast_components_col_names(forecast: pd.DataFrame) -> list:
    components_col = [
        col.replace('_lower', '') for col in forecast.columns
        if 'lower' in col
           and 'yhat' not in col
           and 'multiplicative' not in col
           and 'additive' not in col
    ]
    return components_col


def get_df_cv_with_hist(forecasts: dict, datasets: dict) -> pd.DataFrame:
    df_cv = forecasts['cv'].drop(['cutoff'], axis=1)
    df_past = datasets['full'].loc[datasets['full']['ds'] < df_cv.ds.min()][['ds', 'y']]
    df_past = pd.concat([df_past] + [df_past[['y']]] * 3, axis=1)
    df_past.columns = ['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'y']
    df_cv = pd.concat([df_cv, df_past], axis=0).sort_values('ds').reset_index(drop=True)
    return df_cv


def get_cv_dates_dict(dates: dict, resampling: dict) -> dict:
    freq = resampling['freq'][-1]
    train_start = dates['train_start_date']
    horizon = dates['folds_horizon']
    cv_dates = dict()
    for i, cutoff in sorted(enumerate(dates['cutoffs']), reverse=True):
        cv_dates[f"Fold {i+1}"] = dict()
        cv_dates[f"Fold {i+1}"]['train_start'] = train_start
        cv_dates[f"Fold {i+1}"]['val_start'] = cutoff
        if freq in ['s', 'H']:
            multiplier = convert_into_nb_of_seconds(freq, 1)
            cv_dates[f"Fold {i+1}"]['train_end'] = cutoff - timedelta(seconds=1)
            cv_dates[f"Fold {i+1}"]['val_end'] = cutoff + timedelta(seconds=horizon * multiplier)
        else:
            multiplier = convert_into_nb_of_days(freq, 1)
            cv_dates[f"Fold {i+1}"]['train_end'] = cutoff - timedelta(days=1)
            cv_dates[f"Fold {i+1}"]['val_end'] = cutoff + timedelta(days=horizon * multiplier)
    return cv_dates
