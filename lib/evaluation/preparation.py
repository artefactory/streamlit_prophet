import pandas as pd


def get_evaluation_df(datasets, forecasts, dates, eval, use_cv) -> pd.DataFrame:
    if use_cv:
        evaluation_df = forecasts['cv'].rename(columns={'y': 'truth', 'yhat': 'forecast'})
        mapping = {cutoff: f"Fold {i + 1}" for i, cutoff in enumerate(sorted(evaluation_df['cutoff'].unique(),
                                                                             reverse=True))}
        evaluation_df['Fold'] = evaluation_df['cutoff'].map(mapping)
        evaluation_df = evaluation_df.sort_values('ds')
    else:
        evaluation_df = pd.DataFrame()
        if eval['set'] == 'Validation':
            evaluation_df['ds'] = datasets['val'].ds.copy()
            evaluation_df['truth'] = list(datasets['val'].y)
            evaluation_df['forecast'] = list(forecasts['eval'].query(f'ds >= "{dates["val_start_date"]}" & '
                                                                     f'ds <= "{dates["val_end_date"]}"').yhat)
        elif eval['set'] == 'Training':
            evaluation_df['ds'] = datasets['train'].ds.copy()
            evaluation_df['truth'] = list(datasets['train'].y)
            evaluation_df['forecast'] = list(forecasts['eval'].query(f'ds >= "{dates["train_start_date"]}" & '
                                                                     f'ds <= "{dates["train_end_date"]}"').yhat)
    return evaluation_df


def add_time_groupers(evaluation_df):
    df = evaluation_df.copy()
    df['Global'] = 'Global'
    df['Daily'] = df['ds'].astype(str).map(lambda x: x[0:10])
    df['Weekly'] = df['ds'].dt.year.astype(str) + ' - W' + df['ds'].dt.week.astype(str)\
                                                                   .map(lambda x: '0'+x if len(x) < 2 else x)
    df['Monthly'] = df['ds'].dt.year.astype(str) + ' - M' + df['ds'].dt.month.astype(str)\
                                                                    .map(lambda x: '0'+x if len(x) < 2 else x)
    df['Quarterly'] = df['ds'].dt.year.astype(str) + ' - Q' + df['ds'].dt.quarter.astype(str)
    df['Yearly'] = df['ds'].dt.year.astype(str)
    return df