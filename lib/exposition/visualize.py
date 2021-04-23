import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from fbprophet.plot import plot_plotly
from lib.evaluation.preparation import get_evaluation_df
from lib.evaluation.metrics import get_perf_metrics
from lib.exposition.preparation import get_forecast_components
from lib.utils.palette import cat10_strong


def plot_overview(make_future_forecast, use_cv, models, forecasts, target_col):
    if make_future_forecast:
        st.plotly_chart(plot_plotly(models['future'], forecasts['future'],
                                    changepoints=True, trend=True, ylabel=target_col))
    elif use_cv:
        st.plotly_chart(plot_plotly(models['eval'], forecasts['cv_with_hist'], ylabel=target_col))
    else:
        st.plotly_chart(plot_plotly(models['eval'], forecasts['eval'],
                                    changepoints=True, trend=True, ylabel=target_col))


def plot_performance(use_cv, target_col, datasets, forecasts, dates, eval, resampling):
    evaluation_df = get_evaluation_df(datasets, forecasts, dates, eval, use_cv)
    metrics_df, metrics_dict = get_perf_metrics(evaluation_df, eval, dates, resampling, use_cv)
    st.dataframe(metrics_df)
    plot_perf_metrics(metrics_dict, eval)
    st.plotly_chart(plot_forecasts_vs_truth(evaluation_df, target_col, use_cv))
    st.plotly_chart(plot_truth_vs_actual_scatter(evaluation_df, use_cv))
    st.plotly_chart(plot_residuals_distrib(evaluation_df, use_cv))


def plot_components(use_cv, target_col, datasets, models, forecasts, cleaning, resampling):
    if use_cv:
        forecasts['eval'] = models['eval'].predict(datasets['train'].drop('y', axis=1))
    st.plotly_chart(make_separate_components_plot(models, forecasts, target_col, cleaning, resampling))


def plot_forecasts_vs_truth(eval_df: pd.DataFrame, target_col: str, use_cv: bool):
    """
    Parameters
    ----------
    - eval_df : pd.DataFrame
        a dataframe structured with : 1 column date named 'ds', then 1 column per timeseries.
    """
    if use_cv:
        fig = px.line(eval_df, x='ds', y='forecast', color='Fold', color_discrete_sequence=cat10_strong)
        fig.add_trace(go.Scatter(x=eval_df['ds'], y=eval_df['truth'], name='Truth', mode='lines',
                                 line={'color': '#d62728', 'dash': 'dot', 'width': 1.5}))
    else:
        fig = px.line(eval_df, x='ds', y=['truth', 'forecast'])
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
    fig.update_layout(yaxis_title=target_col, legend_title_text="")
    return fig


def plot_truth_vs_actual_scatter(eval_df: pd.DataFrame, use_cv: bool):
    if use_cv:
        fig = px.scatter(eval_df, x='truth', y='forecast',
                         color='Fold', opacity=0.5, color_discrete_sequence=cat10_strong)
    else:
        fig = px.scatter(eval_df, x='truth', y='forecast', opacity=0.5)
    fig.add_trace(go.Scatter(x=eval_df["truth"], y=eval_df["truth"], name='optimal',
                             mode='lines', line=dict(color='#d62728', width=1.5)))
    fig.update_layout(xaxis_title="Truth", yaxis_title="Forecast")
    return fig


def plot_residuals_distrib(eval_df: pd.DataFrame, use_cv):
    eval_df['residuals'] = eval_df["truth"] - eval_df["forecast"]
    x_min, x_max = eval_df['residuals'].quantile(.01), eval_df['residuals'].quantile(.99)
    if use_cv:
        labels = sorted(eval_df['Fold'].unique(), reverse=True)
        residuals = [eval_df.loc[eval_df['Fold'] == fold, 'residuals'] for fold in labels]
        residuals = [x[x.between(x_min, x_max)] for x in residuals]
    else:
        labels = ['']
        residuals = pd.Series(eval_df['residuals'])
        residuals = [residuals[residuals.between(x_min, x_max)]]
    fig = ff.create_distplot(residuals, labels, show_hist=False, colors=cat10_strong if use_cv else ["#00828c"])
    fig.update_layout(xaxis_title="Residuals distribution (Truth - Forecast)",
                      yaxis_showticklabels=False,
                      showlegend=True if use_cv else False,
                      xaxis_zeroline=True,
                      xaxis_zerolinecolor='#d62728',
                      xaxis_zerolinewidth=1,
                      yaxis_zeroline=True,
                      yaxis_zerolinecolor='#d62728',
                      yaxis_zerolinewidth=1,
                      yaxis_rangemode='tozero'
                      )
    return fig


def plot_perf_metrics(perf: dict, eval:dict):
    for metric in perf.keys():
        if perf[metric][eval['granularity']].nunique() > 1:
            fig = px.bar(perf[metric], x=eval['granularity'], y=metric)
            fig.update_layout(xaxis_title='')
            st.plotly_chart(fig)


def make_separate_components_plot(models: dict, forecasts: dict, target_col: str, cleaning:dict, resampling: dict):
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
        y_label = f"log {target_col}" if cleaning['log_transform'] else target_col
        fig.update_yaxes(title_text=f"{y_label} / {resampling['freq']}", row=i+1, col=1)
    fig.update_layout(height=200 * n_features)
    return fig