def convert_into_nb_of_days(forecast_freq: str, forecast_horizon: int) -> int:
    mapping = {'day': forecast_horizon,
               'week': forecast_horizon * 7,
               'month': forecast_horizon * 30,
               'year': forecast_horizon * 365
               }
    return mapping[forecast_freq]


def dayname_to_daynumber(days: list) -> list:
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    mapping = {day: i for i, day in enumerate(day_names)}
    return [mapping[day] for day in days]


def mapping_country_names(countries: list) -> list:
    mapping = {'France': 'FR',
               'United States': 'US',
               'United Kingdom': 'UK',
               'Canada': 'CA',
               'Brazil': 'BR',
               'Mexico': 'MX',
               'India': 'IN',
               'China': 'CN',
               'Japan': 'JP',
               'Germany': 'DE',
               'Italy': 'IT',
               'Russia': 'RU',
               'Belgium': 'BE',
               'Portugal': 'PT',
               'Poland': 'PL'
               }
    return mapping, [mapping[country] for country in countries]
