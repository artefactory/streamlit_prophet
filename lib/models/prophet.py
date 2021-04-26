from fbprophet import Prophet
from lib.utils.logging import suppress_stdout_stderr
from lib.dataprep.clean import exp_transform
from lib.dataprep.split import make_eval_df, make_future_df
from lib.models.prophet_cv import cross_validation
from lib.exposition.preparation import get_df_cv_with_hist
from lib.utils.mapping import convert_into_nb_of_days, convert_into_nb_of_seconds


def instantiate_prophet_model(params, use_regressors=True):
    seasonality_params = {'yearly_seasonality': params['seasonalities']['yearly']['prophet_param'],
                          'weekly_seasonality': params['seasonalities']['weekly']['prophet_param']}
    model = Prophet(**{**params['prior_scale'], **seasonality_params, **params['other']})
    for _, values in params['seasonalities'].items():
        if 'custom_param' in values:
            model.add_seasonality(**values['custom_param'])
    for country in params['holidays']:
        model.add_country_holidays(country)
    if use_regressors:
        for regressor in params['regressors'].keys():
            model.add_regressor(regressor,
                                prior_scale=params['regressors'][regressor]['prior_scale'],
                                mode=params['regressors'][regressor]['mode']
                                )
    return model


def forecast_workflow(config: dict, use_cv: bool, make_future_forecast: bool, cleaning: dict, resampling: dict,
                      params: dict, dates: dict, datasets: dict):
    models, forecasts = dict(), dict()
    with suppress_stdout_stderr():
        models['eval'] = instantiate_prophet_model(params)
        models['eval'].fit(datasets['train'], seed=config["global"]["seed"])
        if use_cv:
            forecasts['cv'] = cross_validation(models['eval'],
                                               cutoffs=dates['cutoffs'],
                                               horizon=get_prophet_cv_horizon(dates, resampling),
                                               parallel='processes'
                                               )
            forecasts['cv_with_hist'] = get_df_cv_with_hist(forecasts, datasets, models)
        else:
            datasets = make_eval_df(datasets)
            forecasts['eval'] = models['eval'].predict(datasets['eval'])
        if make_future_forecast:
            models['future'] = instantiate_prophet_model(params, use_regressors=False)
            models['future'].fit(datasets['full'], seed=config["global"]["seed"])
            datasets = make_future_df(dates, datasets, cleaning)
            forecasts['future'] = models['future'].predict(datasets['future'])
    if cleaning['log_transform']:
        datasets, forecasts = exp_transform(datasets, forecasts)
    return datasets, models, forecasts


def get_prophet_cv_horizon(dates: dict, resampling: dict) -> str:
    freq = resampling['freq'][-1]
    horizon = dates['folds_horizon']
    if freq in ['s', 'H']:
        multiplier = convert_into_nb_of_seconds(freq, 1)
        prophet_horizon = f"{multiplier * horizon} seconds"
    else:
        multiplier = convert_into_nb_of_days(freq, 1)
        prophet_horizon = f"{multiplier * horizon} days"
    return prophet_horizon
