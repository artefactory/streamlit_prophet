from streamlit_prophet.lib.utils.mapping import convert_into_nb_of_days, convert_into_nb_of_seconds


def get_prophet_cv_horizon(dates: dict, resampling: dict) -> str:
    """Returns cross-validation horizon at the right format for Prophet cross_validation function.

    Parameters
    ----------
    dates : dict
        Dictionary containing cross-validation information.
    resampling : dict
        Dictionary containing dataset frequency information.

    Returns
    -------
    str
        Cross-validation horizon at the right format for Prophet cross_validation function.
    """
    freq = resampling["freq"][-1]
    horizon = dates["folds_horizon"]
    if freq in ["s", "H"]:
        prophet_horizon = f"{convert_into_nb_of_seconds(freq, horizon)} seconds"
    else:
        prophet_horizon = f"{convert_into_nb_of_days(freq, horizon)} days"
    return prophet_horizon
