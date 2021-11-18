from typing import Any, Dict

import plotly.graph_objects as go
import streamlit as st
from streamlit_prophet.lib.exposition.preparation import get_cv_dates_dict, get_hover_template_cv


def plot_cv_dates(
    cv_dates: Dict[Any, Any], resampling: Dict[Any, Any], style: Dict[Any, Any]
) -> go.Figure:
    """Creates a plotly bar plot showing training and validation dates for each cross-validation fold.

    Parameters
    ----------
    cv_dates : Dict
        Dictionary containing training and validation dates of each cross-validation fold.
    resampling : Dict
        Resampling specifications (granularity, dataset frequency).
    style : Dict
        Style specifications for the graph (colors).

    Returns
    -------
    go.Figure
        Plotly bar plot showing training and validation dates for each cross-validation fold.
    """
    hover_data, hover_template = get_hover_template_cv(cv_dates, resampling)
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=list(cv_dates.keys()),
            x=[cv_dates[fold]["val_end"] for fold in cv_dates.keys()],
            name="",
            orientation="h",
            text=hover_data,
            hoverinfo="y+text",
            hovertemplate=hover_template,
            marker=dict(color=style["colors"][1], line=dict(color=style["colors"][1], width=2)),
        )
    )
    fig.add_trace(
        go.Bar(
            y=list(cv_dates.keys()),
            x=[cv_dates[fold]["train_start"] for fold in cv_dates.keys()],
            name="",
            orientation="h",
            text=hover_data,
            hoverinfo="y+text",
            hovertemplate=hover_template,
            marker=dict(color=style["colors"][0], line=dict(color=style["colors"][1], width=2)),
        )
    )
    fig.add_trace(
        go.Bar(
            y=list(cv_dates.keys()),
            x=[cv_dates[fold]["train_end"] for fold in cv_dates.keys()],
            name="",
            orientation="h",
            text=hover_data,
            hoverinfo="y+text",
            hovertemplate=hover_template,
            marker=dict(color=style["colors"][0], line=dict(color=style["colors"][1], width=2)),
        )
    )
    fig.update_layout(
        showlegend=False,
        barmode="overlay",
        xaxis_type="date",
        title_text="Cross-Validation Folds",
        title_x=0.5,
        title_y=0.85,
    )
    return fig


def display_expander(
    readme: Dict[Any, Any], section: str, title: str, add_blank: bool = False
) -> None:
    """Displays a streamlit expander with information about a section of the dashboard.

    Parameters
    ----------
    readme : Dict
        Dictionary containing explanations about the section.
    section : str
        Section of the dashboard on top of which the expander will be displayed.
    title : str
        Title for the expander.
    add_blank : bool
        Whether or not to add a blank after the expander.
    """
    with st.expander(title, expanded=False):
        st.write(readme["plots"][section])
        st.write("")
    if add_blank:
        st.write("")
        st.write("")


def display_expanders_performance(
    use_cv: bool,
    dates: Dict[Any, Any],
    resampling: Dict[Any, Any],
    style: Dict[Any, Any],
    readme: Dict[Any, Any],
) -> None:
    """Displays a streamlit expander with information about performance section.

    Parameters
    ----------
    use_cv : bool
        Whether or not cross-validation is used.
    dates : Dict
        Dictionary containing cross-validation dates information.
    resampling : Dict
        Resampling specifications (granularity, dataset frequency).
    style : Dict
        Style specifications for the graph (colors).
    readme : Dict
        Dictionary containing explanations about the section.
    """
    st.write("")
    with st.expander("More info on evaluation metrics", expanded=False):
        st.write(readme["plots"]["metrics"])
        st.write("")
        _display_metrics()
        st.write("")
    if use_cv:
        cv_dates = get_cv_dates_dict(dates, resampling)
        with st.expander("See cross-validation folds", expanded=False):
            st.plotly_chart(plot_cv_dates(cv_dates, resampling, style))


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
