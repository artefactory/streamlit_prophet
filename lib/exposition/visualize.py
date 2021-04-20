import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from fbprophet.plot import plot_components_plotly, plot_plotly
from lib.evaluation.preparation import get_evaluation_df
from lib.evaluation.metrics import get_perf_metrics
from lib.exposition.preparation import get_forecast_components


def plot_overview(make_future_forecast, use_cv, models, forecasts):
    if make_future_forecast:
        st.plotly_chart(plot_plotly(models['future'], forecasts['future'], changepoints=True, trend=True))
        st.plotly_chart(plot_components_plotly(models['future'], forecasts['future']))
    elif use_cv:
        st.write("Plot overview not implemented yet with cv")
    else:
        st.plotly_chart(plot_plotly(models['eval'], forecasts['eval'], changepoints=True, trend=True))
        #st.plotly_chart(plot_components_plotly(models['eval'], forecasts['eval']))


def plot_performance(use_cv, target_col, datasets, forecasts, dates, eval):
    if use_cv:
        st.write("Plot performance not implemented yet with cv")
        st.dataframe(forecasts['cv'])
    else:
        evaluation_df = get_evaluation_df(datasets, forecasts, dates, eval)
        metrics_df, perf = get_perf_metrics(evaluation_df, eval)
        st.dataframe(metrics_df.set_index(eval['granularity']))
        plot_perf_metrics(perf, eval)
        st.plotly_chart(plot_forecasts_vs_truth(evaluation_df, target_col))
        st.plotly_chart(plot_truth_vs_actual_scatter(evaluation_df))
        st.plotly_chart(plot_residuals_distrib(evaluation_df))


def plot_components(use_cv, target_col, models, forecasts):
    if use_cv:
        st.write("Plot components not implemented yet with cv")
    else:
        st.plotly_chart(make_separate_components_plot(models, forecasts, target_col))


def plot_forecasts_vs_truth(eval_df: pd.DataFrame, target_col: str):
    """
    Parameters
    ----------
    - eval_df : pd.DataFrame
        a dataframe structured with : 1 column date named 'ds', then 1 column per timeseries.
    """
    value_cols = [col for col in eval_df.columns if col != 'ds']
    fig = px.line(eval_df, x='ds', y=value_cols)
    fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=7, label="1w", step="day", stepmode="backward"),
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=3, label="3m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(step="all")
            ])
        )
    )
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title=target_col,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1,
            xanchor="right",
            x=1,

        ),
        legend_title_text=""
    )
    return fig


def plot_truth_vs_actual_scatter(eval_df: pd.DataFrame):
    fig = go.Figure(data=go.Scatter(
        x=eval_df["truth"], y=eval_df["forecast"],
        mode="markers",
        name='values',
        marker=dict(color="#00828c", opacity=0.2),
    ))
    fig.add_trace(go.Scatter(x=eval_df["truth"], y=eval_df["truth"], name='optimal'))
    fig.update_layout(
        xaxis_title="Truth",
        yaxis_title="Forecast"
    )
    return fig


def plot_residuals_distrib(eval_df: pd.DataFrame):
    """
    Plot the distribution of residuals.
    We expect it to be symmetric, approximately normal, and centered at zero.
    Parameters
    ----------
    - residuals: pd.Series
        Series containing y - yhat
    """
    residuals = (eval_df["truth"] - eval_df["forecast"])
    fig = px.histogram(
        pd.DataFrame(residuals, columns=['residuals']),
        x='residuals', color_discrete_sequence=["#00828c"]
    )
    fig.update_layout(
        xaxis_title="Residual (truth - forecast)",
        yaxis_title="Count"
    )
    return fig


def plot_perf_metrics(perf: dict, eval:dict):
    for metric in perf.keys():
        if perf[metric][eval['granularity']].nunique() > 1:
            fig = px.bar(perf[metric], x=eval['granularity'], y=metric)
            st.plotly_chart(fig)


def make_separate_components_plot(models: dict, forecasts: dict, target_col: str):
    """
    Create an area chart with the components of the prediction, each one on its own subplot.
    """
    components = get_forecast_components(models, forecasts)
    features = components.columns
    n_features = len(components.columns)
    fig = make_subplots(rows=n_features, cols=1, subplot_titles=features)
    for i, col in enumerate(features):
        if col == "weekly":
            days = forecasts['eval']["ds"].groupby(forecasts['eval'].ds.dt.dayofweek).last()
            values = forecasts['eval'].loc[forecasts['eval'].ds.isin(days), ("ds", col)]
            values = values.iloc[values.ds.dt.dayofweek.values.argsort()]  # sort by day of week order
            y = values[col]
            x = values.ds.dt.day_name()
        elif col == "yearly":
            year = forecasts['eval']["ds"].max().year - 1
            days = pd.date_range(start=f'{year}-01-01', end=f"{year}-12-31")
            y = forecasts['eval'].loc[forecasts['eval']["ds"].isin(days), col]
            x = days
        else:
            x = components.index
            y = components[col]
        fig.append_trace(go.Scatter(x=x, y=y, fill='tozeroy', name=col), row=i+1, col=1)
        fig.update_yaxes(title_text=f"{target_col} / day", row=i+1, col=1)
    fig.update_layout(height=200 * n_features)
    return fig