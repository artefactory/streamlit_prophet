import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from fbprophet.plot import plot_components_plotly, plot_plotly
from lib.evaluation.preparation import get_evaluation_series, get_evaluation_df
from lib.evaluation.metrics import get_perf_metrics, prettify_metrics


def plot_overview(make_future_forecast, use_cv, models, forecasts):
    if make_future_forecast:
        st.plotly_chart(plot_plotly(models['future'], forecasts['future'], changepoints=True, trend=True))
        st.plotly_chart(plot_components_plotly(models['future'], forecasts['future']))
    elif use_cv:
        st.write("Plot overview not implemented yet with cv")
    else:
        st.plotly_chart(plot_plotly(models['eval'], forecasts['eval'], changepoints=True, trend=True))
        st.plotly_chart(plot_components_plotly(models['eval'], forecasts['eval']))


def plot_performance(use_cv, metrics, target_col, datasets, forecasts, dates, eval_set):
    if use_cv:
        st.write("Plot performance not implemented yet with cv")
        st.dataframe(forecasts['cv'])
    else:
        y_true, y_pred = get_evaluation_series(datasets, forecasts, dates, eval_set)
        eval_df = get_evaluation_df(datasets, forecasts, dates, eval_set)
        perf = get_perf_metrics(y_true, y_pred, metrics)
        st.success(prettify_metrics(perf))
        st.plotly_chart(plot_forecasts_vs_truth(eval_df, target_col))
        st.plotly_chart(plot_truth_vs_actual_scatter(eval_df))
        st.plotly_chart(plot_residuals_distrib(eval_df))


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
            y=1.1,
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
