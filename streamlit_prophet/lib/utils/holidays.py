from typing import List

import re

import pandas as pd
from vacances_scolaires_france import SchoolHolidayDates


def lockdown_format_func(lockdown_idx: int) -> str:
    return f"Lockdown {lockdown_idx + 1}"


def get_school_holidays_FR(years: List[int]) -> pd.DataFrame:
    """Retrieve french school holidays and transform it into a Prophet holidays compatible df

    Parameters
    ----------
    years: List[int]
        List of years for which to retrieve holidays.

    Returns
    -------
    pd.DataFrame
        Holidays dataframe with columns 'ds' and 'holiday'.
    """

    def _get_school_holidays_FR_for_year(year: int) -> pd.DataFrame:
        fr_holidays = SchoolHolidayDates()
        df_vacances = pd.DataFrame.from_dict(fr_holidays.holidays_for_year(year)).T.reset_index(
            drop=True
        )
        df_vacances = df_vacances.rename(columns={"date": "ds", "nom_vacances": "holiday"})
        df_vacances["holiday"] = df_vacances["holiday"].apply(
            lambda x: re.sub(r"^Vacances (De|D')? ?(La )?", "School holiday: ", x.title())
        )
        df_vacances["ds"] = pd.to_datetime(df_vacances["ds"])
        return df_vacances

    school_holidays = pd.concat(map(_get_school_holidays_FR_for_year, years))
    holidays_df = school_holidays[["holiday", "ds"]]
    return holidays_df
