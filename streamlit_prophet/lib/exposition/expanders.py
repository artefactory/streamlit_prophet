import streamlit as st
from streamlit_prophet.lib.exposition.preparation import get_cv_dates_dict
from streamlit_prophet.lib.exposition.visualize import plot_cv_dates


def display_expander(readme: dict, section: str) -> None:
    """Displays a streamlit expander with information about a section of the dashboard.

    Parameters
    ----------
    readme : dict
        Dictionary containing explanations about the section.
    section : str
        Section of the dashboard on top of which the expander will be displayed.
    """
    st.write("")
    with st.beta_expander("More info on this plot", expanded=False):
        st.write(readme["plots"][section])
        st.write("")


def display_expanders_performance(
    use_cv: bool, dates: dict, resampling: dict, style: dict, readme: dict
) -> None:
    """Displays a streamlit expander with information about performance section.

    Parameters
    ----------
    use_cv : bool
        Whether or not cross-validation is used.
    dates : dict
        Dictionary containing cross-validation dates information.
    resampling : dict
        Resampling specifications (granularity, dataset frequency).
    style : dict
        Style specifications for the graph (colors).
    readme : dict
        Dictionary containing explanations about the section.
    """
    st.write("")
    with st.beta_expander("More info on evaluation metrics", expanded=False):
        st.write(readme["plots"]["metrics"])
        st.write("")
        _display_metrics()
        st.write("")
    if use_cv:
        cv_dates = get_cv_dates_dict(dates, resampling)
        with st.beta_expander("See cross-validation folds", expanded=False):
            st.plotly_chart(plot_cv_dates(cv_dates, resampling, style))
        st.write("")
        st.write("")
    else:
        st.write("")
        st.write("")


def _display_metrics() -> None:
    """Displays formulas for all performance metrics."""
    if st.checkbox("Show metric formulas", value=False):
        st.write("If N is the number of distinct dates in the evaluation set:")
        st.latex(r"MAPE = \dfrac{1}{N}\sum_{t=1}^{N}|\dfrac{Truth_t - Forecast_t}{Truth_t}|")
        st.latex(r"RMSE = \sqrt{\dfrac{1}{N}\sum_{t=1}^{N}(Truth_t - Forecast_t)^2}")
        st.latex(
            r"SMAPE = \dfrac{1}{N}\sum_{t=1}^{N}\dfrac{2|Truth_t - Forecast_t]}{|Truth_t| + |Forecast_t|}"
        )
        st.latex(r"MSE = \dfrac{1}{N}\sum_{t=1}^{N}(Truth_t - Forecast_t)^2")
        st.latex(r"MAE = \dfrac{1}{N}\sum_{t=1}^{N}|Truth_t - Forecast_t|")
