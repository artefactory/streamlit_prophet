from typing import Any, Dict, List, Tuple


def convert_into_nb_of_days(freq: str, horizon: int) -> int:
    """Converts a forecasting horizon in number of days.

    Parameters
    ----------
    freq : str
        Dataset frequency.
    horizon : int
        Forecasting horizon in dataset frequency units.

    Returns
    -------
    int
        Forecasting horizon in days.
    """
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
    """Converts a forecasting horizon in number of seconds.

    Parameters
    ----------
    freq : str
        Dataset frequency.
    horizon : int
        Forecasting horizon in dataset frequency units.

    Returns
    -------
    int
        Forecasting horizon in seconds.
    """
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


def dayname_to_daynumber(days: List[Any]) -> List[Any]:
    """Converts a list of day names into a list of day numbers from 0 (Monday) to 6 (Sunday).

    Parameters
    ----------
    days : list
        Day names.

    Returns
    -------
    list
        Day numbers from 0 (Monday) to 6 (Sunday).
    """
    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    mapping = {day: i for i, day in enumerate(day_names)}
    return [mapping[day] for day in days]


def mapping_country_names(countries: List[Any]) -> Tuple[Dict[Any, Any], List[Any]]:
    """Converts a list of country long names into a list of country short names.

    Parameters
    ----------
    countries : list
        Country long names.

    Returns
    -------
    dict
        Mapping used for the conversion.
    list
        Country short names.
    """
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
    """Converts a short frequency name into a long frequency name.

    Parameters
    ----------
    freq : str
        Short frequency name.

    Returns
    -------
    str
        Long frequency name.
    """
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
