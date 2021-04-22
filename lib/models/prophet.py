from fbprophet import Prophet
from lib.utils.logging import suppress_stdout_stderr
from lib.dataprep.clean import exp_transform
from lib.dataprep.split import make_eval_df, make_future_df
from lib.models.prophet_cv import cross_validation


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


def forecast_workflow(config, use_cv, make_future_forecast, cleaning, params, dates, datasets, models, forecasts):
    with suppress_stdout_stderr():
        models['eval'] = instantiate_prophet_model(params)
        models['eval'].fit(datasets['train'], seed=config["global"]["seed"])
        if use_cv:
            forecasts['cv'] = cross_validation(models['eval'],
                                               cutoffs=dates['cutoffs'],
                                               horizon=f"{dates['folds_horizon']} days"
                                               )
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
