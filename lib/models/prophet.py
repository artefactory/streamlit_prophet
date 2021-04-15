from fbprophet import Prophet

def get_prophet_model(prior_scale_params, other_params, seasonalities, holidays):
    seasonality_params = {'yearly_seasonality': seasonalities['yearly']['prophet_param'],
                          'weekly_seasonality': seasonalities['weekly']['prophet_param']}
    model = Prophet(**{**prior_scale_params, **seasonality_params, **other_params})
    for _, values in seasonalities.items():
        if 'custom_param' in values:
            model.add_seasonality(**values['custom_param'])
    for country in holidays:
        model = model.add_country_holidays(country)
    return model