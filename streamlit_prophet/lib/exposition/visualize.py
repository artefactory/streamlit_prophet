# type: ignore

import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import streamlit as st
from fbprophet.plot import plot_plotly
from plotly.subplots import make_subplots
from streamlit_prophet.lib.evaluation.metrics import get_perf_metrics
from streamlit_prophet.lib.evaluation.preparation import get_evaluation_df
from streamlit_prophet.lib.exposition.preparation import (
    get_cv_dates_dict,
    get_forecast_components,
    get_hover_template_cv,
)
from streamlit_prophet.lib.utils.misc import reverse_list


def plot_overview(
    make_future_forecast: bool,
    use_cv: bool,
    models: dict,
    forecasts: dict,
    target_col: str,
    cleaning: dict,
    readme: dict,
) -> None:
    _display_expander(readme, "overview")
    bool_param = False if cleaning["log_transform"] else True
    if make_future_forecast:
        st.plotly_chart(
            plot_plotly(
                models["future"],
                forecasts["future"],
                ylabel=target_col,
                changepoints=bool_param,
                trend=bool_param,
                uncertainty=bool_param,
            )
        )
    elif use_cv:
        st.plotly_chart(
            plot_plotly(
                models["eval"],
                forecasts["cv_with_hist"],
                ylabel=target_col,
                changepoints=bool_param,
                trend=bool_param,
                uncertainty=bool_param,
            )
        )
    else:
        st.plotly_chart(
            plot_plotly(
                models["eval"],
                forecasts["eval"],
                ylabel=target_col,
                changepoints=bool_param,
                trend=bool_param,
                uncertainty=bool_param,
            )
        )


def plot_performance(
    use_cv: bool,
    target_col: str,
    datasets: dict,
    forecasts: dict,
    dates: dict,
    eval: dict,
    resampling: dict,
    config: dict,
    readme: dict,
) -> None:
    style = config["style"]
    _display_expanders_performance(use_cv, dates, resampling, style, readme)
    evaluation_df = get_evaluation_df(datasets, forecasts, dates, eval, use_cv)
    metrics_df, metrics_dict = get_perf_metrics(
        evaluation_df, eval, dates, resampling, use_cv, config
    )
    st.dataframe(metrics_df)
    plot_perf_metrics(metrics_dict, eval, use_cv, style)
    st.plotly_chart(plot_forecasts_vs_truth(evaluation_df, target_col, use_cv, style))
    st.plotly_chart(plot_truth_vs_actual_scatter(evaluation_df, use_cv, style))
    st.plotly_chart(plot_residuals_distrib(evaluation_df, use_cv, style))


def plot_components(
    use_cv: bool,
    target_col: str,
    models: dict,
    forecasts: dict,
    cleaning: dict,
    resampling: dict,
    config: dict,
    readme: dict,
) -> None:
    style = config["style"]
    _display_expander(readme, "components")
    if use_cv:
        forecast_df = forecasts["cv_with_hist"].copy()
        forecast_df = forecast_df.loc[forecast_df["ds"] < forecasts["cv"].ds.min()]
    else:
        forecast_df = forecasts["eval"].copy()
    st.plotly_chart(
        make_separate_components_plot(models, forecast_df, target_col, cleaning, resampling, style)
    )


def plot_future(
    models: dict, forecasts: dict, dates: dict, target_col: str, cleaning: dict, readme: dict
) -> None:
    _display_expander(readme, "future")
    bool_param = False if cleaning["log_transform"] else True
    fig = plot_plotly(
        models["future"],
        forecasts["future"],
        ylabel=target_col,
        changepoints=bool_param,
        trend=bool_param,
        uncertainty=bool_param,
    )
    fig.update_layout(xaxis_range=[dates["forecast_start_date"], dates["forecast_end_date"]])
    st.plotly_chart(fig)


def plot_forecasts_vs_truth(eval_df: pd.DataFrame, target_col: str, use_cv: bool, style: dict):
    if use_cv:
        colors = reverse_list(style["colors"], eval_df["Fold"].nunique())
        fig = px.line(eval_df, x="ds", y="forecast", color="Fold", color_discrete_sequence=colors)
        fig.add_trace(
            go.Scatter(
                x=eval_df["ds"],
                y=eval_df["truth"],
                name="Truth",
                mode="lines",
                line={"color": style["color_axis"], "dash": "dot", "width": 1.5},
            )
        )
    else:
        fig = px.line(
            eval_df, x="ds", y=["truth", "forecast"], color_discrete_sequence=style["colors"][1:]
        )
    fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list(
                [
                    dict(count=7, label="1w", step="day", stepmode="backward"),
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=3, label="3m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all"),
                ]
            )
        ),
    )
    fig.update_layout(
        yaxis_title=target_col,
        legend_title_text="",
        height=500,
        title_text="Forecast vs Truth",
        title_x=0.5,
        title_y=1,
    )
    return fig


def plot_truth_vs_actual_scatter(eval_df: pd.DataFrame, use_cv: bool, style: dict):
    if use_cv:
        colors = reverse_list(style["colors"], eval_df["Fold"].nunique())
        fig = px.scatter(
            eval_df,
            x="truth",
            y="forecast",
            color="Fold",
            opacity=0.5,
            color_discrete_sequence=colors,
        )
    else:
        fig = px.scatter(
            eval_df,
            x="truth",
            y="forecast",
            opacity=0.5,
            color_discrete_sequence=style["colors"][2:],
        )
    fig.add_trace(
        go.Scatter(
            x=eval_df["truth"],
            y=eval_df["truth"],
            name="optimal",
            mode="lines",
            line=dict(color=style["color_axis"], width=1.5),
        )
    )
    fig.update_layout(xaxis_title="Truth", yaxis_title="Forecast", legend_title_text="", height=450)
    return fig


def plot_residuals_distrib(eval_df: pd.DataFrame, use_cv: bool, style: dict):
    eval_df["residuals"] = eval_df["truth"] - eval_df["forecast"]
    if len(eval_df) >= 10:
        x_min, x_max = eval_df["residuals"].quantile(0.01), eval_df["residuals"].quantile(0.99)
    else:
        x_min, x_max = eval_df["residuals"].min(), eval_df["residuals"].max()
    if use_cv:
        labels = sorted(eval_df["Fold"].unique(), reverse=True)
        residuals = [eval_df.loc[eval_df["Fold"] == fold, "residuals"] for fold in labels]
        residuals = [x[x.between(x_min, x_max)] for x in residuals]
    else:
        labels = [""]
        residuals = pd.Series(eval_df["residuals"])
        residuals = [residuals[residuals.between(x_min, x_max)]]
    colors = (
        reverse_list(style["colors"], eval_df["Fold"].nunique()) if use_cv else [style["colors"][2]]
    )
    fig = ff.create_distplot(residuals, labels, show_hist=False, colors=colors)
    fig.update_layout(
        title_text="Distribution of residuals",
        title_x=0.5,
        title_y=0.85,
        xaxis_title="Residuals (Truth - Forecast)",
        yaxis_showticklabels=False,
        showlegend=True if use_cv else False,
        xaxis_zeroline=True,
        xaxis_zerolinecolor=style["color_axis"],
        xaxis_zerolinewidth=1,
        yaxis_zeroline=True,
        yaxis_zerolinecolor=style["color_axis"],
        yaxis_zerolinewidth=1,
        yaxis_rangemode="tozero",
        height=500,
    )
    return fig


def plot_perf_metrics(perf: dict, eval: dict, use_cv: bool, style: dict):
    # TODO : Tout inclure dans un unique subplot
    for i, metric in enumerate(perf.keys()):
        color_sequence = style["colors"] if use_cv else [style["colors"][i % len(style["colors"])]]
        if perf[metric][eval["granularity"]].nunique() > 1:
            fig = px.bar(
                perf[metric],
                x=eval["granularity"],
                y=metric,
                color=eval["granularity"],
                color_discrete_sequence=color_sequence,
            )
            fig.update_layout(xaxis_title="", showlegend=False, height=400)
            st.plotly_chart(fig)


def make_separate_components_plot(
    models: dict,
    forecast_df: pd.DataFrame,
    target_col: str,
    cleaning: dict,
    resampling: dict,
    style: dict,
):
    """
    Create an area chart with the components of the prediction, each one on its own subplot.
    """
    components = get_forecast_components(models, forecast_df)
    features = components.columns
    n_features = len(components.columns)
    fig = make_subplots(rows=n_features, cols=1, subplot_titles=features)
    for i, col in enumerate(features):
        if col == "daily":
            hours = forecast_df["ds"].groupby(forecast_df.ds.dt.hour).last()
            values = forecast_df.loc[forecast_df.ds.isin(hours), ("ds", col)]
            values = values.iloc[values.ds.dt.hour.values.argsort()]  # sort by hour order
            y = values[col]
            x = values.ds.map(lambda h: h.strftime("%H:%M"))
        elif col == "weekly":
            days = forecast_df["ds"].groupby(forecast_df.ds.dt.dayofweek).last()
            values = forecast_df.loc[forecast_df.ds.isin(days), ("ds", col)]
            values = values.iloc[
                values.ds.dt.dayofweek.values.argsort()
            ]  # sort by day of week order
            y = values[col]
            x = values.ds.dt.day_name()
        elif col == "monthly":
            days = forecast_df["ds"].groupby(forecast_df.ds.dt.day).last()
            values = forecast_df.loc[forecast_df.ds.isin(days), ("ds", col)]
            values = values.iloc[values.ds.dt.day.values.argsort()]  # sort by day of month order
            y = values[col]
            x = values.ds.dt.day
        elif col == "yearly":
            year = forecast_df["ds"].max().year - 1
            days = pd.date_range(start=f"{year}-01-01", end=f"{year}-12-31")
            y = forecast_df.loc[forecast_df["ds"].isin(days), col]
            x = days.dayofyear
        else:
            x = components.index
            y = components[col]
        fig.append_trace(
            go.Scatter(
                x=x,
                y=y,
                fill="tozeroy",
                name=col,
                mode="lines",
                line=dict(color=style["colors"][i % len(style["colors"])]),
            ),
            row=i + 1,
            col=1,
        )

        y_label = f"log {target_col}" if cleaning["log_transform"] else target_col
        fig.update_yaxes(title_text=f"{y_label} / {resampling['freq']}", row=i + 1, col=1)
        if col == "yearly":
            fig["layout"][f"xaxis{i + 1}"].update(
                tickmode="array",
                tickvals=[1, 61, 122, 183, 244, 305],
                ticktext=["Jan", "Mar", "May", "Jul", "Sep", "Nov"],
            )
    fig.update_layout(height=200 * n_features)
    return fig


def plot_cv_dates(cv_dates: dict, resampling: dict, style: dict):
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


def _display_expander(readme: dict, section: str):
    st.write("")
    with st.beta_expander("More info on this plot", expanded=False):
        st.write(readme["plots"][section])
        st.write("")


def _display_expanders_performance(
    use_cv: bool, dates: dict, resampling: dict, style: dict, readme: dict
):
    st.write("")
    with st.beta_expander("More info on evaluation metrics", expanded=False):
        st.write(readme["plots"]["metrics"])
        st.write("")
        __display_metrics()
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


def __display_metrics():
    if st.checkbox("Show metric formulas", value=False):
        st.write("If N is the number of distinct dates in the evaluation set:")
        st.latex(r"MAPE = \dfrac{1}{N}\sum_{t=1}^{N}|\dfrac{Truth_t - Forecast_t}{Truth_t}|")
        st.latex(r"RMSE = \sqrt{\dfrac{1}{N}\sum_{t=1}^{N}(Truth_t - Forecast_t)^2}")
        st.latex(
            r"SMAPE = \dfrac{1}{N}\sum_{t=1}^{N}\dfrac{2|Truth_t - Forecast_t]}{|Truth_t| + |Forecast_t|}"
        )
        st.latex(r"MSE = \dfrac{1}{N}\sum_{t=1}^{N}(Truth_t - Forecast_t)^2")
        st.latex(r"MAE = \dfrac{1}{N}\sum_{t=1}^{N}|Truth_t - Forecast_t|")
