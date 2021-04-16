import pandas as pd


def get_evaluation_series(datasets, forecasts, dates, eval_set):
    if eval_set == 'Validation':
        y_true = datasets['val'].y
        y_pred = forecasts['eval'].query(f'ds >= "{dates["val_start_date"]}" & ds <= "{dates["val_end_date"]}"').yhat
    elif eval_set == 'Training':
        y_true = datasets['train'].y
        y_pred = forecasts['eval'].query(f'ds >= "{dates["train_start_date"]}" & ds < "{dates["val_start_date"]}"').yhat
    return y_true, y_pred


def get_evaluation_df(datasets, forecasts, dates, eval_set):
    y_true, y_pred = get_evaluation_series(datasets, forecasts, dates, eval_set)
    eval_df = pd.DataFrame()
    eval_df['ds'] = datasets['val'].ds.copy() if eval_set == 'Validation' else datasets['train'].ds.copy()
    eval_df['truth'] = y_true
    eval_df['forecast'] = y_pred
    return eval_df