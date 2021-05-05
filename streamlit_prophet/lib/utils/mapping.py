def convert_into_nb_of_days(freq: str, horizon: int) -> int:
    mapping = {
        "s": horizon // (24 * 60 * 60),
        "H": horizon // 24,
        "D": horizon,
        "W": horizon * 7,
        "M": horizon * 30,
        "Q": horizon * 90,
        "Y": horizon * 365,
    }
    return mapping[freq]


def convert_into_nb_of_seconds(freq: str, horizon: int) -> int:
    mapping = {
        "s": horizon,
        "H": horizon * 60 * 60,
        "D": horizon * 60 * 60 * 24,
        "W": horizon * 60 * 60 * 24 * 7,
        "M": horizon * 60 * 60 * 24 * 30,
        "Q": horizon * 60 * 60 * 24 * 90,
        "Y": horizon * 60 * 60 * 24 * 365,
    }
    return mapping[freq]


def dayname_to_daynumber(days: list) -> list:
    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    mapping = {day: i for i, day in enumerate(day_names)}
    return [mapping[day] for day in days]


def mapping_country_names(countries: list) -> list:
    mapping = {
        "France": "FR",
        "United States": "US",
        "United Kingdom": "UK",
        "Canada": "CA",
        "Brazil": "BR",
        "Mexico": "MX",
        "India": "IN",
        "China": "CN",
        "Japan": "JP",
        "Germany": "DE",
        "Italy": "IT",
        "Russia": "RU",
        "Belgium": "BE",
        "Portugal": "PT",
        "Poland": "PL",
    }
    return mapping, [mapping[country] for country in countries]


def mapping_freq_names(freq: str) -> str:
    mapping = {
        "s": "seconds",
        "H": "hours",
        "D": "days",
        "W": "weeks",
        "M": "months",
        "Q": "quarters",
        "Y": "years",
    }
    return mapping[freq]
