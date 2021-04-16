from fbprophet import Prophet
from lib.utils.logging import suppress_stdout_stderr
from lib.dataprep.split import make_eval_df, make_future_df
from lib.models.prophet_cv import cross_validation


def instantiate_prophet_model(params):
    seasonality_params = {'yearly_seasonality': params['seasonalities']['yearly']['prophet_param'],
                          'weekly_seasonality': params['seasonalities']['weekly']['prophet_param']}
    model = Prophet(**{**params['prior_scale'], **seasonality_params, **params['other']})
    for _, values in params['seasonalities'].items():
        if 'custom_param' in values:
            model.add_seasonality(**values['custom_param'])
    for country in params['holidays']:
        model.add_country_holidays(country)
    for regressor in params['regressors'].keys():
        model.add_regressor(regressor,
                            prior_scale=params['regressors'][regressor]['prior_scale'],
                            mode=params['regressors'][regressor]['mode']
                            )
    return model


def forecast_workflow(config, use_cv, make_future_forecast, df, params, dates, datasets, models, forecasts):
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
            models['future'] = instantiate_prophet_model(params)
            models['future'].fit(datasets['full'], seed=config["global"]["seed"])
            datasets = make_future_df(df, dates, datasets, include_history=True)
            # TODO : Appliquer le même cleaning / les mêmes filtres sur la donnée future que sur l'historique
            forecasts['future'] = models['future'].predict(datasets['future'])
    return datasets, models, forecasts
