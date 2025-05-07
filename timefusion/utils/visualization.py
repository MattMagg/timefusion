"""
Visualization utilities for TimeFusion.

This module provides utilities for visualizing time series data,
forecasts, and evaluation results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Optional, List, Tuple, Dict, Any, Union, Callable

# Check if seaborn is available for enhanced styling
try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False

# Check if plotly is available for interactive plots
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


def set_style(style: str = 'seaborn') -> None:
    """
    Set the matplotlib style.
    
    Args:
        style: Matplotlib style
    """
    if SEABORN_AVAILABLE and style == 'seaborn':
        sns.set_style('whitegrid')
    else:
        plt.style.use(style)


def plot_time_series(
    data: Union[pd.DataFrame, pd.Series],
    columns: Optional[List[str]] = None,
    title: str = 'Time Series Plot',
    xlabel: str = 'Time',
    ylabel: str = 'Value',
    figsize: Tuple[int, int] = (10, 6),
    color_map: Optional[str] = None,
    grid: bool = True,
    legend: bool = True,
    date_format: Optional[str] = None,
    save_path: Optional[str] = None,
    interactive: bool = False
) -> Union[plt.Figure, Dict[str, Any]]:
    """
    Plot time series data.
    
    Args:
        data: Time series data
        columns: Columns to plot (if None, all columns are plotted)
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        figsize: Figure size
        color_map: Matplotlib colormap name
        grid: Whether to show grid
        legend: Whether to show legend
        date_format: Date format for x-axis
        save_path: Path to save the figure (if None, the figure is not saved)
        interactive: Whether to create an interactive plot using Plotly
        
    Returns:
        Union[plt.Figure, Dict[str, Any]]: Figure object or Plotly figure
    """
    if interactive and not PLOTLY_AVAILABLE:
        print("Warning: Plotly is not available. Using Matplotlib instead.")
        interactive = False
    
    # Convert Series to DataFrame
    if isinstance(data, pd.Series):
        data = data.to_frame()
    
    # Select columns to plot
    if columns is None:
        columns = data.columns
    
    if interactive:
        # Create interactive plot using Plotly
        fig = go.Figure()
        
        for column in columns:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data[column],
                    mode='lines',
                    name=column
                )
            )
        
        fig.update_layout(
            title=title,
            xaxis_title=xlabel,
            yaxis_title=ylabel,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            ),
            template='plotly_white'
        )
        
        if save_path:
            fig.write_image(save_path)
        
        return fig
    else:
        # Create static plot using Matplotlib
        fig, ax = plt.subplots(figsize=figsize)
        
        if color_map:
            cmap = plt.get_cmap(color_map)
            colors = [cmap(i) for i in np.linspace(0, 1, len(columns))]
        else:
            colors = None
        
        for i, column in enumerate(columns):
            color = colors[i] if colors else None
            ax.plot(data.index, data[column], label=column, color=color)
        
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        
        if grid:
            ax.grid(True)
        
        if legend:
            ax.legend()
        
        if date_format and isinstance(data.index, pd.DatetimeIndex):
            ax.xaxis.set_major_formatter(mdates.DateFormatter(date_format))
            fig.autofmt_xdate()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        
        return fig
def plot_forecast(
    actual: Union[pd.DataFrame, pd.Series],
    forecast: Union[pd.DataFrame, pd.Series],
    column: Optional[str] = None,
    confidence_intervals: Optional[Dict[str, pd.DataFrame]] = None,
    title: str = 'Forecast vs Actual',
    xlabel: str = 'Time',
    ylabel: str = 'Value',
    figsize: Tuple[int, int] = (10, 6),
    actual_color: str = 'blue',
    forecast_color: str = 'red',
    ci_color: str = 'gray',
    ci_alpha: float = 0.2,
    grid: bool = True,
    legend: bool = True,
    date_format: Optional[str] = None,
    save_path: Optional[str] = None,
    interactive: bool = False
) -> Union[plt.Figure, Dict[str, Any]]:
    """
    Plot forecast vs actual values.
    
    Args:
        actual: Actual values
        forecast: Forecast values
        column: Column to plot (if None, the first column is used)
        confidence_intervals: Dictionary of confidence intervals
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        figsize: Figure size
        actual_color: Color for actual values
        forecast_color: Color for forecast values
        ci_color: Color for confidence intervals
        ci_alpha: Alpha for confidence intervals
        grid: Whether to show grid
        legend: Whether to show legend
        date_format: Date format for x-axis
        save_path: Path to save the figure (if None, the figure is not saved)
        interactive: Whether to create an interactive plot using Plotly
        
    Returns:
        Union[plt.Figure, Dict[str, Any]]: Figure object or Plotly figure
    """
    if interactive and not PLOTLY_AVAILABLE:
        print("Warning: Plotly is not available. Using Matplotlib instead.")
        interactive = False
    
    # Convert Series to DataFrame
    if isinstance(actual, pd.Series):
        actual = actual.to_frame()
    if isinstance(forecast, pd.Series):
        forecast = forecast.to_frame()
    
    # Select column to plot
    if column is None:
        column = actual.columns[0]
    
    if interactive:
        # Create interactive plot using Plotly
        fig = go.Figure()
        
        # Add actual values
        fig.add_trace(
            go.Scatter(
                x=actual.index,
                y=actual[column],
                mode='lines',
                name='Actual',
                line=dict(color=actual_color)
            )
        )
        
        # Add forecast values
        fig.add_trace(
            go.Scatter(
                x=forecast.index,
                y=forecast[column],
                mode='lines',
                name='Forecast',
                line=dict(color=forecast_color)
            )
        )
        
        # Add confidence intervals
        if confidence_intervals:
            for level, ci in confidence_intervals.items():
                fig.add_trace(
                    go.Scatter(
                        x=ci.index,
                        y=ci[f'{column}_lower'],
                        mode='lines',
                        line=dict(width=0),
                        showlegend=False
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=ci.index,
                        y=ci[f'{column}_upper'],
                        mode='lines',
                        fill='tonexty',
                        name=f'{level}% CI',
                        line=dict(width=0),
                        fillcolor=ci_color,
                        opacity=ci_alpha
                    )
                )
        
        fig.update_layout(
            title=title,
            xaxis_title=xlabel,
            yaxis_title=ylabel,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            ),
            template='plotly_white'
        )
        
        if save_path:
            fig.write_image(save_path)
        
        return fig
    else:
        # Create static plot using Matplotlib
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot actual values
        ax.plot(actual.index, actual[column], label='Actual', color=actual_color)
        
        # Plot forecast values
        ax.plot(forecast.index, forecast[column], label='Forecast', color=forecast_color)
        
        # Plot confidence intervals
        if confidence_intervals:
            for level, ci in confidence_intervals.items():
                ax.fill_between(
                    ci.index,
                    ci[f'{column}_lower'],
                    ci[f'{column}_upper'],
                    color=ci_color,
                    alpha=ci_alpha,
                    label=f'{level}% CI'
                )
        
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        
        if grid:
            ax.grid(True)
        
        if legend:
            ax.legend()
        
        if date_format and (isinstance(actual.index, pd.DatetimeIndex) or isinstance(forecast.index, pd.DatetimeIndex)):
            ax.xaxis.set_major_formatter(mdates.DateFormatter(date_format))
            fig.autofmt_xdate()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        
        return fig
def plot_residuals(
    residuals: Union[pd.DataFrame, pd.Series],
    column: Optional[str] = None,
    title: str = 'Residual Analysis',
    figsize: Tuple[int, int] = (12, 8),
    color: str = 'blue',
    grid: bool = True,
    save_path: Optional[str] = None,
    interactive: bool = False
) -> Union[plt.Figure, Dict[str, Any]]:
    """
    Plot residual analysis.
    
    Args:
        residuals: Residuals
        column: Column to plot (if None, the first column is used)
        title: Plot title
        figsize: Figure size
        color: Color for plots
        grid: Whether to show grid
        save_path: Path to save the figure (if None, the figure is not saved)
        interactive: Whether to create an interactive plot using Plotly
        
    Returns:
        Union[plt.Figure, Dict[str, Any]]: Figure object or Plotly figure
    """
    if interactive and not PLOTLY_AVAILABLE:
        print("Warning: Plotly is not available. Using Matplotlib instead.")
        interactive = False
    
    # Convert Series to DataFrame
    if isinstance(residuals, pd.Series):
        residuals = residuals.to_frame()
    
    # Select column to plot
    if column is None:
        column = residuals.columns[0]
    
    # Extract residuals as numpy array
    res = residuals[column].values
    
    if interactive:
        # Create interactive plot using Plotly
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Residuals vs Time',
                'Residual Histogram',
                'Q-Q Plot',
                'Autocorrelation'
            )
        )
        
        # Residuals vs Time
        fig.add_trace(
            go.Scatter(
                x=residuals.index,
                y=res,
                mode='lines',
                name='Residuals'
            ),
            row=1, col=1
        )
        
        # Residual Histogram
        fig.add_trace(
            go.Histogram(
                x=res,
                name='Histogram'
            ),
            row=1, col=2
        )
        
        # Q-Q Plot
        from scipy import stats
        qq_x = np.linspace(np.min(res), np.max(res), 100)
        qq_y = stats.norm.ppf(np.linspace(0.01, 0.99, 100))
        
        fig.add_trace(
            go.Scatter(
                x=qq_x,
                y=qq_y,
                mode='lines',
                name='Q-Q Line'
            ),
            row=2, col=1
        )
        
        # Autocorrelation
        from statsmodels.tsa.stattools import acf
        acf_values = acf(res, nlags=20)
        
        fig.add_trace(
            go.Bar(
                x=list(range(len(acf_values))),
                y=acf_values,
                name='ACF'
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title=title,
            height=600,
            width=800,
            showlegend=False,
            template='plotly_white'
        )
        
        if save_path:
            fig.write_image(save_path)
        
        return fig
    else:
        # Create static plot using Matplotlib
        fig, axs = plt.subplots(2, 2, figsize=figsize)
        
        # Residuals vs Time
        axs[0, 0].plot(residuals.index, res, color=color)
        axs[0, 0].set_title('Residuals vs Time')
        axs[0, 0].set_xlabel('Time')
        axs[0, 0].set_ylabel('Residuals')
        if grid:
            axs[0, 0].grid(True)
        
        # Residual Histogram
        axs[0, 1].hist(res, bins=20, color=color, alpha=0.7)
        axs[0, 1].set_title('Residual Histogram')
        axs[0, 1].set_xlabel('Residuals')
        axs[0, 1].set_ylabel('Frequency')
        if grid:
            axs[0, 1].grid(True)
        
        # Q-Q Plot
        from scipy import stats
        stats.probplot(res, plot=axs[1, 0])
        axs[1, 0].set_title('Q-Q Plot')
        if grid:
            axs[1, 0].grid(True)
        
        # Autocorrelation
        from statsmodels.tsa.stattools import acf
        acf_values = acf(res, nlags=20)
        axs[1, 1].bar(range(len(acf_values)), acf_values, color=color)
        axs[1, 1].set_title('Autocorrelation')
        axs[1, 1].set_xlabel('Lag')
        axs[1, 1].set_ylabel('ACF')
        if grid:
            axs[1, 1].grid(True)
        
        plt.suptitle(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        
        return fig
def plot_metrics(
    metrics: Dict[str, float],
    title: str = 'Model Metrics',
    figsize: Tuple[int, int] = (10, 6),
    color: str = 'blue',
    grid: bool = True,
    save_path: Optional[str] = None,
    interactive: bool = False
) -> Union[plt.Figure, Dict[str, Any]]:
    """
    Plot model metrics.
    
    Args:
        metrics: Dictionary of metrics
        title: Plot title
        figsize: Figure size
        color: Color for bars
        grid: Whether to show grid
        save_path: Path to save the figure (if None, the figure is not saved)
        interactive: Whether to create an interactive plot using Plotly
        
    Returns:
        Union[plt.Figure, Dict[str, Any]]: Figure object or Plotly figure
    """
    if interactive and not PLOTLY_AVAILABLE:
        print("Warning: Plotly is not available. Using Matplotlib instead.")
        interactive = False
    
    # Sort metrics by name
    metrics = dict(sorted(metrics.items()))
    
    if interactive:
        # Create interactive plot using Plotly
        fig = go.Figure()
        
        fig.add_trace(
            go.Bar(
                x=list(metrics.keys()),
                y=list(metrics.values()),
                marker_color=color
            )
        )
        
        fig.update_layout(
            title=title,
            xaxis_title='Metric',
            yaxis_title='Value',
            template='plotly_white'
        )
        
        if save_path:
            fig.write_image(save_path)
        
        return fig
    else:
        # Create static plot using Matplotlib
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.bar(metrics.keys(), metrics.values(), color=color)
        
        ax.set_title(title)
        ax.set_xlabel('Metric')
        ax.set_ylabel('Value')
        
        if grid:
            ax.grid(True)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        
        return fig


def plot_comparison(
    results: Dict[str, Dict[str, float]],
    metrics: Optional[List[str]] = None,
    title: str = 'Model Comparison',
    figsize: Tuple[int, int] = (12, 8),
    color_map: Optional[str] = None,
    grid: bool = True,
    save_path: Optional[str] = None,
    interactive: bool = False
) -> Union[plt.Figure, Dict[str, Any]]:
    """
    Plot model comparison.
    
    Args:
        results: Dictionary of model results
        metrics: List of metrics to plot (if None, all metrics are plotted)
        title: Plot title
        figsize: Figure size
        color_map: Matplotlib colormap name
        grid: Whether to show grid
        save_path: Path to save the figure (if None, the figure is not saved)
        interactive: Whether to create an interactive plot using Plotly
        
    Returns:
        Union[plt.Figure, Dict[str, Any]]: Figure object or Plotly figure
    """
    if interactive and not PLOTLY_AVAILABLE:
        print("Warning: Plotly is not available. Using Matplotlib instead.")
        interactive = False
    
    # Get all metrics if not specified
    if metrics is None:
        metrics = set()
        for model_metrics in results.values():
            metrics.update(model_metrics.keys())
        metrics = sorted(list(metrics))
    
    # Create DataFrame for plotting
    df = pd.DataFrame(index=metrics)
    for model, model_metrics in results.items():
        df[model] = [model_metrics.get(metric, np.nan) for metric in metrics]
    
    if interactive:
        # Create interactive plot using Plotly
        fig = go.Figure()
        
        for model in df.columns:
            fig.add_trace(
                go.Bar(
                    x=metrics,
                    y=df[model],
                    name=model
                )
            )
        
        fig.update_layout(
            title=title,
            xaxis_title='Metric',
            yaxis_title='Value',
            barmode='group',
            template='plotly_white'
        )
        
        if save_path:
            fig.write_image(save_path)
        
        return fig
    else:
        # Create static plot using Matplotlib
        fig, ax = plt.subplots(figsize=figsize)
        
        if color_map:
            cmap = plt.get_cmap(color_map)
            colors = [cmap(i) for i in np.linspace(0, 1, len(df.columns))]
        else:
            colors = None
        
        x = np.arange(len(metrics))
        width = 0.8 / len(df.columns)
        
        for i, model in enumerate(df.columns):
            offset = (i - len(df.columns) / 2 + 0.5) * width
            color = colors[i] if colors else None
            ax.bar(x + offset, df[model], width, label=model, color=color)
        
        ax.set_title(title)
        ax.set_xlabel('Metric')
        ax.set_ylabel('Value')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, rotation=45)
        
        if grid:
            ax.grid(True)
        
        ax.legend()
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        
        return fig
def plot_feature_importance(
    feature_importance: Dict[str, float],
    title: str = 'Feature Importance',
    figsize: Tuple[int, int] = (10, 6),
    color: str = 'blue',
    grid: bool = True,
    save_path: Optional[str] = None,
    interactive: bool = False
) -> Union[plt.Figure, Dict[str, Any]]:
    """
    Plot feature importance.
    
    Args:
        feature_importance: Dictionary of feature importance
        title: Plot title
        figsize: Figure size
        color: Color for bars
        grid: Whether to show grid
        save_path: Path to save the figure (if None, the figure is not saved)
        interactive: Whether to create an interactive plot using Plotly
        
    Returns:
        Union[plt.Figure, Dict[str, Any]]: Figure object or Plotly figure
    """
    if interactive and not PLOTLY_AVAILABLE:
        print("Warning: Plotly is not available. Using Matplotlib instead.")
        interactive = False
    
    # Sort feature importance
    feature_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
    
    if interactive:
        # Create interactive plot using Plotly
        fig = go.Figure()
        
        fig.add_trace(
            go.Bar(
                x=list(feature_importance.values()),
                y=list(feature_importance.keys()),
                orientation='h',
                marker_color=color
            )
        )
        
        fig.update_layout(
            title=title,
            xaxis_title='Importance',
            yaxis_title='Feature',
            template='plotly_white'
        )
        
        if save_path:
            fig.write_image(save_path)
        
        return fig
    else:
        # Create static plot using Matplotlib
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.barh(list(feature_importance.keys()), list(feature_importance.values()), color=color)
        
        ax.set_title(title)
        ax.set_xlabel('Importance')
        ax.set_ylabel('Feature')
        
        if grid:
            ax.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        
        return fig


def plot_correlation_matrix(
    data: pd.DataFrame,
    title: str = 'Correlation Matrix',
    figsize: Tuple[int, int] = (10, 8),
    cmap: str = 'coolwarm',
    annot: bool = True,
    save_path: Optional[str] = None,
    interactive: bool = False
) -> Union[plt.Figure, Dict[str, Any]]:
    """
    Plot correlation matrix.
    
    Args:
        data: Input data
        title: Plot title
        figsize: Figure size
        cmap: Colormap
        annot: Whether to annotate the heatmap
        save_path: Path to save the figure (if None, the figure is not saved)
        interactive: Whether to create an interactive plot using Plotly
        
    Returns:
        Union[plt.Figure, Dict[str, Any]]: Figure object or Plotly figure
    """
    if interactive and not PLOTLY_AVAILABLE:
        print("Warning: Plotly is not available. Using Matplotlib instead.")
        interactive = False
    
    # Calculate correlation matrix
    corr = data.corr()
    
    if interactive:
        # Create interactive plot using Plotly
        fig = go.Figure()
        
        fig.add_trace(
            go.Heatmap(
                z=corr.values,
                x=corr.columns,
                y=corr.index,
                colorscale=cmap,
                zmin=-1,
                zmax=1,
                text=corr.values if annot else None,
                texttemplate='%{text:.2f}' if annot else None,
                colorbar=dict(title='Correlation')
            )
        )
        
        fig.update_layout(
            title=title,
            template='plotly_white'
        )
        
        if save_path:
            fig.write_image(save_path)
        
        return fig
    else:
        # Create static plot using Matplotlib
        fig, ax = plt.subplots(figsize=figsize)
        
        if SEABORN_AVAILABLE:
            sns.heatmap(corr, annot=annot, cmap=cmap, vmin=-1, vmax=1, ax=ax)
        else:
            im = ax.imshow(corr, cmap=cmap, vmin=-1, vmax=1)
            ax.set_xticks(np.arange(len(corr.columns)))
            ax.set_yticks(np.arange(len(corr.index)))
            ax.set_xticklabels(corr.columns)
            ax.set_yticklabels(corr.index)
            plt.colorbar(im, ax=ax)
            
            if annot:
                for i in range(len(corr.index)):
                    for j in range(len(corr.columns)):
                        ax.text(j, i, f'{corr.iloc[i, j]:.2f}', ha='center', va='center')
        
        ax.set_title(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        
        return fig
