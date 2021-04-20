def convert_into_nb_of_days(forecast_freq, forecast_horizon):
    mapping = {'day': forecast_horizon,
               'week': forecast_horizon * 7,
               'month': forecast_horizon * 30,
               'year': forecast_horizon * 365
               }
    return mapping[forecast_freq]


def dayname_to_daynumber(days: list):
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    mapping = {i: day for i, day in enumerate(day_names)}
    return [mapping[day] for day in days]
