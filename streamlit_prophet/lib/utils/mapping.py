from typing import Any, Dict, List, Tuple

from streamlit_prophet.lib.utils.holidays import get_school_holidays_FR

COUNTRY_NAMES_MAPPING = {
    "FR": "France",
    "US": "United States",
    "UK": "United Kingdom",
    "CA": "Canada",
    "BR": "Brazil",
    "MX": "Mexico",
    "IN": "India",
    "CN": "China",
    "JP": "Japan",
    "DE": "Germany",
    "IT": "Italy",
    "RU": "Russia",
    "BE": "Belgium",
    "PT": "Portugal",
    "PL": "Poland",
}

COVID_LOCKDOWN_DATES_MAPPING = {
    "FR": [
        ("2020-03-17", "2020-05-11"),
        ("2020-10-30", "2020-12-15"),
        ("2021-03-20", "2021-05-03"),
    ]
}

SCHOOL_HOLIDAYS_FUNC_MAPPING = {
    "FR": get_school_holidays_FR,
}


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
    mapping = {v: k for k, v in COUNTRY_NAMES_MAPPING.items()}
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
