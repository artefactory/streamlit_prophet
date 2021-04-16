def convert_into_nb_of_days(forecast_freq, forecast_horizon):
    mapping = {'day': forecast_horizon,
               'week': forecast_horizon * 7,
               'month': forecast_horizon * 30,
               'year': forecast_horizon * 365
               }
    return mapping[forecast_freq]


def dayname_to_daynumber(days: list):
    mapping = {'Monday': 0,
               'Tuesday': 1,
               'Wednesday': 2,
               'Thursday': 3,
               'Friday': 4,
               'Saturday': 5,
               'Sunday': 6
               }
    return [mapping[day] for day in days]
