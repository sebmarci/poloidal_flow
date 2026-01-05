"""
Visualization utilities for cross-correlation function analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Literal
import flap

from ..core.analysis import CorrelationAnalysis, gaussian_func


class CCFPlotter:
    """
    Plotting functions for single channel or all channel CCFs.

    Parameters
    ----------
    analyzer : CorrelationAnalysis
        CorrelationAnalysis instance containing data and configuration.

    Attributes
    ----------
    analyzer : CorrelationAnalysis
        Reference to the correlation analysis instance.
    """

    def __init__(self, analyzer: CorrelationAnalysis):
        self.analyzer = analyzer

    def plot_single(
        self,
        time: float,
        channel: int,
        method: Literal['gaussian', 'spline'] = 'gaussian',
        show_fit: bool = True,
        **kwargs
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot cross-correlation function for a single channel at a given time.

        Parameters
        ----------
        time : float
            Time point in seconds for the analysis window center.
        channel : int
            ABES channel number (1-40).
        method : {'gaussian', 'spline'}, default='gaussian'
            Fitting method to visualize.
        ax : matplotlib.axes.Axes, optional
            Axes object to plot on. If None, creates a new figure.
        show_fit : bool, default=True
            Whether to show the fitted curve.
        show_peak : bool, default=True
            Whether to mark the peak position.
        show_error : bool, default=False
            Whether to show fit error for Gaussian method.
        **kwargs
            Additional keyword arguments passed to matplotlib plotting functions.

        Returns
        -------
        fig : matplotlib.figure.Figure
            Figure object.
        ax : matplotlib.axes.Axes
            Axes object with the plot.

        Notes
        -----
        The plot displays:
        - Scatter points: Raw CCF data
        - Solid line: Fitted curve (Gaussian or cubic spline)
        - Marker: Peak position with maximum correlation value
        - Title: Channel number, time point, and time delay
        """
        # Create figure if not provided
        fig, ax = plt.subplots(figsize = (8, 6))

        # Slice data for the specified time window and channel
        data0 = self.analyzer.data_defl0.slice_data(
            slicing={
                'Time': flap.Intervals(
                    time - self.analyzer.config.xcorr_window/2,
                    time + self.analyzer.config.xcorr_window/2
                ),
                'Channel number': channel
            },
            options={'Interpolation': 'Linear'}
        )

        data1 = self.analyzer.data_defl1.slice_data(
            slicing={
                'Time': flap.Intervals(
                    time - self.analyzer.config.xcorr_window/2,
                    time + self.analyzer.config.xcorr_window/2
                ),
                'Channel number': channel
            },
            options={'Interpolation': 'Linear'}
        )

        # Compute cross-correlation
        ccf = self.analyzer.ccf_window_single(data0, data1)
        time_lags = ccf.coordinate('Time lag')[0] * 1e6  # Convert to microseconds

        # Plot raw CCF data
        scatter_kwargs = {
            'alpha': kwargs.get('alpha', 0.7),
            'label': 'CCF data',
            's': kwargs.get('s', 30)
        }
        ax.scatter(time_lags, ccf.data, **scatter_kwargs)

        # Fit and plot curve
        if show_fit:
            ts = np.linspace(min(time_lags), max(time_lags), 1000)

            if method == 'gaussian':
                tau_max, ccf_max, popt = self.analyzer.fit_gaussian(ccf)
                ax.plot(ts, gaussian_func(ts, *popt),
                       label='Gaussian fit',
                       color=kwargs.get('color', 'C1'),
                       linewidth=kwargs.get('linewidth', 2))

            elif method == 'spline':
                tau_max, ccf_max, cs = self.analyzer.fit_cubic_spline(ccf)
                ax.plot(ts, cs(ts),
                       label='Cubic spline',
                       color=kwargs.get('color', 'C1'),
                       linewidth=kwargs.get('linewidth', 2))
                title_str = f'Channel {channel}, t = {time:.2f} s\n$\\tau$ = {tau_max:.2f} $\\mu$s'
            else:
                raise ValueError(f"Invalid method '{method}'. Must be 'gaussian' or 'spline'.")

            # Mark peak position
            ax.scatter(tau_max, ccf_max,
                        color='red',
                        s=kwargs.get('peak_size', 100),
                        zorder=5,
                        marker='o',
                        edgecolors='black',
                        linewidths=1.5,
                        label=f'Peak ($\\tau$ = {tau_max:.2f} $\\mu$s)')
            ax.axvline(x=tau_max,
                        color='red',
                        linestyle='--',
                        alpha=0.5,
                        linewidth=1)
            
        else:
            title_str = f'Channel {channel}, t = {time:.2f} s'

        # Styling
        ax.set_xlabel('Time lag $\\tau$ [$\\mu$s]', fontsize=kwargs.get('fontsize', 12))
        ax.set_ylabel('Normalized CCF', fontsize=kwargs.get('fontsize', 12))
        ax.set_title(title_str, fontsize=kwargs.get('title_fontsize', 14))
        ax.grid(kwargs.get('grid', True), alpha=0.3)
        ax.legend(fontsize=kwargs.get('legend_fontsize', 10))

        return fig, ax

    def plot_all_channels(
        self,
        time: float,
        method: Literal['gaussian', 'spline'] = 'spline',
        channels: Optional[range] = None,
        figsize: Tuple[int, int] = (40, 40),
        nrows: int = 8,
        ncols: int = 5,
        **kwargs
    ) -> Tuple[plt.Figure, np.ndarray]:
        """
        Plot cross-correlation functions for multiple channels at a single time point.

        Parameters
        ----------
        time : float
            Time point in seconds for the analysis window center.
        method : {'gaussian', 'spline'}, default='spline'
            Fitting method to use for all plots.
        channels : range, optional
            Range of channel numbers to plot. Default is range(1, 41) for all 40 channels.
        figsize : tuple, default=(40, 40)
            Figure size as (width, height) in inches.
        nrows : int, default=8
            Number of subplot rows.
        ncols : int, default=5
            Number of subplot columns.
        **kwargs
            Additional keyword arguments for plot customization.

        Returns
        -------
        fig : matplotlib.figure.Figure
            Figure object.
        axes : numpy.ndarray
            Array of axes objects for each subplot.

        Notes
        -----
        Creates a subplot grid (default 8x5 for 40 channels) where each subplot shows:
        - Scatter plot of CCF data points
        - Fitted curve (spline or Gaussian)
        - Peak position marker and vertical line
        - Time delay value in the subplot title

        The main figure title indicates the fitting method and time point.
        """
        if channels is None:
            channels = range(1, 41)

        # Create subplot grid
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

        method_name = 'Cubic Spline' if method == 'spline' else 'Gaussian'
        fig.suptitle(
            f'{method_name} CCF Fits - All Channels at t = {time:.2f} s',
            fontsize=kwargs.get('suptitle_fontsize', 20)
        )

        axes_flat = axes.flat

        # Slice data once for all channels
        data0 = self.analyzer.data_defl0.slice_data(
            slicing={'Time': flap.Intervals(
                time - self.analyzer.config.xcorr_window/2,
                time + self.analyzer.config.xcorr_window/2
            )},
            options={'Interpolation': 'Linear'}
        )

        data1 = self.analyzer.data_defl1.slice_data(
            slicing={'Time': flap.Intervals(
                time - self.analyzer.config.xcorr_window/2,
                time + self.analyzer.config.xcorr_window/2
            )},
            options={'Interpolation': 'Linear'}
        )

        # Plot each channel
        for i, ch in enumerate(channels):
            ax = axes_flat[i]

            # Slice for specific channel
            data0_ch = data0.slice_data(slicing={'Channel number': ch})
            data1_ch = data1.slice_data(slicing={'Channel number': ch})

            # Compute CCF
            ccf = self.analyzer.ccf_window_single(data0_ch, data1_ch)
            time_lags = ccf.coordinate('Time lag')[0] * 1e6
            ts = np.linspace(min(time_lags), max(time_lags), 1000)

            # Fit and plot
            if method == 'spline':
                tau_max, ccf_max, cs = self.analyzer.fit_cubic_spline(ccf)
                ax.plot(ts, cs(ts), color='C1', linewidth=1.5)
            elif method == 'gaussian':
                tau_max, ccf_max, popt = self.analyzer.fit_gaussian(ccf)
                ax.plot(ts, gaussian_func(ts, *popt), color='C1', linewidth=1.5)
            else:
                raise ValueError(f"Invalid method '{method}'. Must be 'gaussian' or 'spline'.")

            # Plot raw data and peak
            ax.scatter(time_lags, ccf.data, alpha=0.6, s=20)
            ax.scatter(tau_max, ccf_max, color='red', s=50, zorder=5, edgecolors='black', linewidths=1)
            ax.axvline(x=tau_max, color='red', linestyle='--', alpha=0.4, linewidth=1)

            # Styling
            ax.set_title(f'CH{ch}: $\\tau$ = {tau_max:.2f} $\\mu$s', fontsize=10)
            ax.set_xlabel('$\\tau$ [$\\mu$s]', fontsize=9)
            ax.set_ylabel('CCF', fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=8)

        # Hide unused subplots
        for j in range(i + 1, len(axes_flat)):
            axes_flat[j].axis('off')

        plt.tight_layout()

        return fig, axes
