import pandas as pd


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
    df_past = pd.concat([df_past] + [df_past[['y']]]*3, axis=1)
    df_past.columns = ['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'y']
    df_cv = pd.concat([df_cv, df_past], axis=0).sort_values('ds').reset_index(drop=True)
    return df_cv
