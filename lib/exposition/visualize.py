import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

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