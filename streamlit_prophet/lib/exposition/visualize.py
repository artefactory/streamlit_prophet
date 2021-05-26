# type: ignore

import datetime

import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import streamlit as st
from fbprophet.plot import plot_plotly
from plotly.subplots import make_subplots
from streamlit_prophet.lib.evaluation.metrics import get_perf_metrics
from streamlit_prophet.lib.evaluation.preparation import get_evaluation_df
from streamlit_prophet.lib.exposition.expanders import (
    display_expander,
    display_expanders_performance,
)
from streamlit_prophet.lib.exposition.export import display_2_download_links, display_download_link
from streamlit_prophet.lib.exposition.preparation import get_forecast_components
from streamlit_prophet.lib.inputs.dates import input_waterfall_dates
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
    """Plots a graph with predictions and actual values, with explanations.

    Parameters
    ----------
    make_future_forecast : bool
        Whether or not a forecast is made on future dates.
    use_cv : bool
        Whether or not cross-validation is used.
    models : dict
        Dictionary containing a model fitted on evaluation data and another model fitted on the whole dataset.
    forecasts : dict
        Dictionary containing evaluation forecasts and future forecasts if a future forecast is made.
    target_col : str
        Name of target column.
    cleaning : dict
        Cleaning specifications.
    readme : dict
        Dictionary containing explanations about the graph.
    """
    display_expander(readme, "overview", "More info on this plot")
    bool_param = False if cleaning["log_transform"] else True
    if make_future_forecast:
        model = models["future"]
        forecast = forecasts["future"]
    elif use_cv:
        model = models["eval"]
        forecast = forecasts["cv_with_hist"]
    else:
        model = models["eval"]
        forecast = forecasts["eval"]
    st.plotly_chart(
        plot_plotly(
            model,
            forecast,
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
    """Plots several graphs showing model performance, with explanations.

    Parameters
    ----------
    use_cv : bool
        Whether or not cross-validation is used.
    target_col : str
        Name of target column.
    datasets : dict
        Dictionary containing evaluation dataset.
    forecasts : dict
        Dictionary containing evaluation forecasts.
    dates : dict
        Dictionary containing evaluation dates.
    eval : dict
        Evaluation specifications (metrics, evaluation set, granularity).
    resampling : dict
        Resampling specifications (granularity, dataset frequency).
    config : dict
        Cleaning specifications.
    readme : dict
        Dictionary containing explanations about the graphs.
    """
    style = config["style"]
    evaluation_df = get_evaluation_df(datasets, forecasts, dates, eval, use_cv)
    metrics_df, metrics_dict = get_perf_metrics(
        evaluation_df, eval, dates, resampling, use_cv, config
    )
    st.write("## Performance metrics")
    display_expanders_performance(use_cv, dates, resampling, style, readme)
    display_expander(readme, "helper_metrics", "How to evaluate my model?", True)
    st.write("### Global performance")
    display_global_metrics(evaluation_df, eval, dates, resampling, use_cv, config)
    st.write("### Deep dive")
    plot_detailed_metrics(metrics_df, metrics_dict, eval, use_cv, style)
    display_2_download_links(
        evaluation_df,
        "evaluation_data",
        "Export evaluation data",
        metrics_df,
        "performance_metrics",
        "Export performance metrics",
    )
    st.write("## Error analysis")
    display_expander(readme, "helper_errors", "How to troubleshoot forecasting errors?", True)
    st.plotly_chart(plot_forecasts_vs_truth(evaluation_df, target_col, use_cv, style))
    st.plotly_chart(plot_truth_vs_actual_scatter(evaluation_df, use_cv, style))
    st.plotly_chart(plot_residuals_distrib(evaluation_df, use_cv, style))


def plot_components(
    use_cv: bool,
    make_future_forecast: bool,
    target_col: str,
    models: dict,
    forecasts: dict,
    cleaning: dict,
    resampling: dict,
    config: dict,
    readme: dict,
) -> None:
    """Plots a graph showing the different components of prediction, with explanations.

    Parameters
    ----------
    use_cv : bool
        Whether or not cross-validation is used.
    make_future_forecast : bool
        Whether or not a future forecast is made.
    target_col : str
        Name of target column.
    models : dict
        Dictionary containing a model fitted on evaluation data.
    forecasts : dict
        Dictionary containing evaluation forecasts.
    cleaning : dict
        Cleaning specifications.
    resampling : dict
        Resampling specifications (granularity, dataset frequency).
    config : dict
        Cleaning specifications.
    readme : dict
        Dictionary containing explanations about the graph.
    """
    style = config["style"]
    st.write("## Global impact")
    display_expander(readme, "components", "More info on this plot")
    if make_future_forecast:
        forecast_df = forecasts["future"].copy()
        model = models["future"]
    elif use_cv:
        forecast_df = forecasts["cv_with_hist"].copy()
        forecast_df = forecast_df.loc[forecast_df["ds"] < forecasts["cv"].ds.min()]
        model = models["eval"]
    else:
        forecast_df = forecasts["eval"].copy()
        model = models["eval"]
    display_download_link(forecast_df, "forecast_components", "Export forecast components", True)
    st.plotly_chart(
        make_separate_components_plot(model, forecast_df, target_col, cleaning, resampling, style)
    )
    st.write("## Local impact")
    display_expander(readme, "waterfall", "More info on this plot", True)
    start_date, end_date = input_waterfall_dates(forecast_df, resampling)
    st.plotly_chart(
        make_waterfall_components_plot(
            model, forecast_df, start_date, end_date, target_col, cleaning, resampling, style
        )
    )


def plot_future(
    models: dict, forecasts: dict, dates: dict, target_col: str, cleaning: dict, readme: dict
) -> None:
    """Plots a graph with predictions for future dates, with explanations.

    Parameters
    ----------
    models : dict
        Dictionary containing a model fitted on the whole dataset.
    forecasts : dict
        Dictionary containing future forecast.
    dates : dict
        Dictionary containing future forecast dates.
    target_col : str
        Name of target column.
    cleaning : dict
        Cleaning specifications.
    readme : dict
        Dictionary containing explanations about the graph.
    """
    display_expander(readme, "future", "More info on this plot")
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
    display_download_link(forecasts["future"], "future_forecasts", "Export future forecasts", True)
    st.plotly_chart(fig)


def plot_forecasts_vs_truth(
    eval_df: pd.DataFrame, target_col: str, use_cv: bool, style: dict
) -> go.Figure:
    """Creates a plotly line plot showing forecasts and actual values on evaluation period.

    Parameters
    ----------
    eval_df : pd.DataFrame
        Evaluation dataframe.
    target_col : str
        Name of target column.
    use_cv : bool
        Whether or not cross-validation is used.
    style : dict
        Style specifications for the graph (colors).

    Returns
    -------
    go.Figure
        Plotly line plot showing forecasts and actual values on evaluation period.
    """
    if use_cv:
        colors = reverse_list(style["colors"], eval_df["Fold"].nunique())
        fig = px.line(
            eval_df,
            x="ds",
            y="forecast",
            color="Fold",
            color_discrete_sequence=colors,
        )
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
            eval_df,
            x="ds",
            y=["truth", "forecast"],
            color_discrete_sequence=style["colors"][1:],
            hover_data={"variable": True, "value": ":.4f", "ds": False},
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
        width=800,
        title_text="Forecast vs Truth",
        title_x=0.5,
        title_y=1,
        hovermode="x unified",
    )
    return fig


def plot_truth_vs_actual_scatter(eval_df: pd.DataFrame, use_cv: bool, style: dict) -> go.Figure:
    """Creates a plotly scatter plot showing forecasts and actual values on evaluation period.

    Parameters
    ----------
    eval_df : pd.DataFrame
        Evaluation dataframe.
    use_cv : bool
        Whether or not cross-validation is used.
    style : dict
        Style specifications for the graph (colors).

    Returns
    -------
    go.Figure
        Plotly scatter plot showing forecasts and actual values on evaluation period.
    """
    eval_df["date"] = eval_df["ds"].map(lambda x: x.strftime("%A %b %d %Y"))
    if use_cv:
        colors = reverse_list(style["colors"], eval_df["Fold"].nunique())
        fig = px.scatter(
            eval_df,
            x="truth",
            y="forecast",
            color="Fold",
            opacity=0.5,
            color_discrete_sequence=colors,
            hover_data={"date": True, "truth": ":.4f", "forecast": ":.4f"},
        )
    else:
        fig = px.scatter(
            eval_df,
            x="truth",
            y="forecast",
            opacity=0.5,
            color_discrete_sequence=style["colors"][2:],
            hover_data={"date": True, "truth": ":.4f", "forecast": ":.4f"},
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
    fig.update_layout(
        xaxis_title="Truth", yaxis_title="Forecast", legend_title_text="", height=450, width=800
    )
    return fig


def plot_residuals_distrib(eval_df: pd.DataFrame, use_cv: bool, style: dict) -> go.Figure:
    """Creates a plotly distribution plot showing distribution of residuals on evaluation period.

    Parameters
    ----------
    eval_df : pd.DataFrame
        Evaluation dataframe.
    use_cv : bool
        Whether or not cross-validation is used.
    style : dict
        Style specifications for the graph (colors).

    Returns
    -------
    go.Figure
        Plotly distribution plot showing distribution of residuals on evaluation period.
    """
    eval_df["residuals"] = eval_df["forecast"] - eval_df["truth"]
    if len(eval_df) >= 10:
        x_min, x_max = eval_df["residuals"].quantile(0.005), eval_df["residuals"].quantile(0.995)
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
        title_text="Distribution of errors",
        title_x=0.5,
        title_y=0.85,
        xaxis_title="Error (Forecast - Truth)",
        showlegend=True if use_cv else False,
        xaxis_zeroline=True,
        xaxis_zerolinecolor=style["color_axis"],
        xaxis_zerolinewidth=1,
        yaxis_zeroline=True,
        yaxis_zerolinecolor=style["color_axis"],
        yaxis_zerolinewidth=1,
        yaxis_rangemode="tozero",
        height=500,
        width=800,
    )
    return fig


def plot_detailed_metrics(
    metrics_df: pd.DataFrame, perf: dict, eval: dict, use_cv: bool, style: dict
) -> None:
    """Displays a dataframe or plots graphs showing model performance on selected metrics.

    Parameters
    ----------
    metrics_df : pd.DataFrame
        Dataframe containing model performance on different metrics at the desired granularity.
    perf : dict
        Dictionary containing model performance on different metrics at the desired granularity.
    eval : dict
        Evaluation specifications (evaluation set, selected metrics, granularity).
    use_cv : bool
        Whether or not cross-validation is used.
    style : dict
        Style specifications for the graph (colors).
    """
    metrics = [metric for metric in perf.keys() if perf[metric][eval["granularity"]].nunique() > 1]
    if len(metrics) > 0:
        fig = make_subplots(
            rows=len(metrics) // 2 + len(metrics) % 2, cols=2, subplot_titles=metrics
        )
        for i, metric in enumerate(metrics):
            colors = (
                style["colors"]
                if use_cv
                else [style["colors"][i % len(style["colors"])]]
                * perf[metric][eval["granularity"]].nunique()
            )
            fig_metric = go.Bar(
                x=perf[metric][eval["granularity"]], y=perf[metric][metric], marker_color=colors
            )
            fig.append_trace(fig_metric, row=i // 2 + 1, col=i % 2 + 1)
        fig.update_layout(
            height=300 * (len(metrics) // 2 + len(metrics) % 2),
            width=1000,
            showlegend=False,
        )
        st.plotly_chart(fig)
    else:
        st.dataframe(metrics_df)


def make_separate_components_plot(
    model,
    forecast_df: pd.DataFrame,
    target_col: str,
    cleaning: dict,
    resampling: dict,
    style: dict,
) -> go.Figure:
    """Creates plotly area charts with the components of the prediction, each one on its own subplot.

    Parameters
    ----------
    model : Prophet
        Fitted model.
    forecast_df : pd.DataFrame
        Predictions of Prophet model.
    target_col : str
        Name of target column.
    cleaning : dict
        Cleaning specifications.
    resampling : dict
        Resampling specifications (granularity, dataset frequency).
    style : dict
        Style specifications for the graph (colors).

    Returns
    -------
    go.Figure
        Plotly area charts with the components of the prediction, each one on its own subplot.
    """
    components = get_forecast_components(model, forecast_df)
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
        fig.update_xaxes(showgrid=False)
        if col == "yearly":
            fig["layout"][f"xaxis{i + 1}"].update(
                tickmode="array",
                tickvals=[1, 61, 122, 183, 244, 305],
                ticktext=["Jan", "Mar", "May", "Jul", "Sep", "Nov"],
            )
    fig.update_layout(height=200 * n_features if n_features > 1 else 300, width=800)
    return fig


def make_waterfall_components_plot(
    model,
    forecast_df: pd.DataFrame,
    start_date: datetime.date,
    end_date: datetime.date,
    target_col: str,
    cleaning: dict,
    resampling: dict,
    style: dict,
) -> go.Figure:
    """Creates a waterfall chart with the components of the prediction.

    Parameters
    ----------
    model : Prophet
        Fitted model.
    forecast_df : pd.DataFrame
        Predictions of Prophet model.
    start_date : datetime.date
        Start date for components computation.
    end_date : datetime.date
        End date for components computation.
    target_col : str
        Name of target column.
    cleaning : dict
        Cleaning specifications.
    resampling : dict
        Resampling specifications (granularity, dataset frequency).
    style : dict
        Style specifications for the graph (colors).

    Returns
    -------
    go.Figure
        Waterfall chart with the components of prediction.
    """
    components = get_forecast_components(model, forecast_df, True).reset_index()
    waterfall = components.loc[
        (components["ds"] >= pd.to_datetime(start_date))
        & (components["ds"] <= pd.to_datetime(end_date))
    ]
    waterfall = waterfall.mean(axis=0, numeric_only=True)
    waterfall = waterfall[waterfall != 0]
    waterfall = waterfall[~waterfall.index.str.endswith("holidays")]
    fig = go.Figure(
        go.Waterfall(
            orientation="v",
            measure=["relative"] * (len(waterfall) - 1) + ["total"],
            x=[x.capitalize() for x in list(waterfall.index)[:-1] + ["Forecast"]],
            y=list(waterfall.values),
            textposition="auto",
            text=[
                "+" + str(round(x, 3)) if x > 0 else "" + str(round(x, 3))
                for x in list(waterfall.values)[:-1]
            ]
            + [str(round(waterfall.values[-1], 3))],
            decreasing={"marker": {"color": style["colors"][1]}},
            increasing={"marker": {"color": style["colors"][0]}},
            totals={"marker": {"color": style["colors"][2]}},
        )
    )
    y_label = f"log {target_col}" if cleaning["log_transform"] else target_col
    fig.update_yaxes(title_text=f"{y_label} / {resampling['freq']}")
    fig.update_layout(
        title=f"Forecast decomposition "
        f"(from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')})",
        title_x=0.2,
        width=800,
    )
    return fig


def display_global_metrics(
    evaluation_df: pd.DataFrame,
    eval: dict,
    dates: dict,
    resampling: dict,
    use_cv: bool,
    config: dict,
) -> None:
    """Displays all global metrics.

    Parameters
    ----------
    evaluation_df : pd.DataFrame
        Evaluation dataframe.
    eval : dict
        Evaluation specifications.
    dates : dict
        Dictionary containing all dates information.
    resampling : dict
        Resampling specifications.
    use_cv : bool
        Whether or note cross-validation is used.
    config : dict
        Lib configuration dictionary.
    """
    eval_all = {
        "granularity": "cutoff" if use_cv else "Global",
        "metrics": ["RMSE", "MAPE", "MAE", "MSE", "SMAPE"],
        "get_perf_on_agg_forecast": eval["get_perf_on_agg_forecast"],
    }
    metrics_df, _ = get_perf_metrics(evaluation_df, eval_all, dates, resampling, use_cv, config)
    if use_cv:
        st.dataframe(metrics_df)
    else:
        col1, col2, col3, col4, col5 = st.beta_columns(5)
        col1.markdown(
            f"<p style='color: {config['style']['colors'][1]}; "
            f"font-weight: bold; font-size: 20px;'> {eval_all['metrics'][0]}</p>",
            unsafe_allow_html=True,
        )
        col1.write(metrics_df.loc["Global", eval_all["metrics"][0]])
        col2.markdown(
            f"<p style='color: {config['style']['colors'][1]}; "
            f"font-weight: bold; font-size: 20px;'> {eval_all['metrics'][1]}</p>",
            unsafe_allow_html=True,
        )
        col2.write(metrics_df.loc["Global", eval_all["metrics"][1]])
        col3.markdown(
            f"<p style='color: {config['style']['colors'][1]}; "
            f"font-weight: bold; font-size: 20px;'> {eval_all['metrics'][2]}</p>",
            unsafe_allow_html=True,
        )
        col3.write(metrics_df.loc["Global", eval_all["metrics"][2]])
        col4.markdown(
            f"<p style='color: {config['style']['colors'][1]}; "
            f"font-weight: bold; font-size: 20px;'> {eval_all['metrics'][3]}</p>",
            unsafe_allow_html=True,
        )
        col4.write(metrics_df.loc["Global", eval_all["metrics"][3]])
        col5.markdown(
            f"<p style='color: {config['style']['colors'][1]}; "
            f"font-weight: bold; font-size: 20px;'> {eval_all['metrics'][4]}</p>",
            unsafe_allow_html=True,
        )
        col5.write(metrics_df.loc["Global", eval_all["metrics"][4]])
