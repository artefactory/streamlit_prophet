from typing import List

import pandas as pd
from vacances_scolaires_france import SchoolHolidayDates


def get_school_holidays_FR(years: List[int]) -> pd.DataFrame:
    def _get_school_holidays_FR_for_year(year: int):
        fr_holidays = SchoolHolidayDates()
        df_vacances = pd.DataFrame.from_dict(fr_holidays.holidays_for_year(year)).T.reset_index(
            drop=True
        )
        df_vacances = df_vacances.rename(columns={"date": "ds", "nom_vacances": "holiday"})
        df_vacances["ds"] = pd.to_datetime(df_vacances["ds"])
        return df_vacances

    school_holidays = pd.concat(map(_get_school_holidays_FR_for_year, years))
    holidays_df = school_holidays[["holiday", "ds"]]
    return holidays_df
