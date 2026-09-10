"""
Visualization utilities for cross-correlation function analysis.

Diagnostic plots of the CCFs a `CorrelationAnalysis` computes: the data with its
error band, the fitted peak and the resulting time lag, for one channel or for
all 40 at once. Useful for checking that the lag interval, the bandpass and the
fit behave before running a whole shot through the analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Literal
import flap

from .analysis import CorrelationAnalysis, gaussian_func, parabolic_func


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

    Notes
    -----
    The CCFs are recomputed here rather than taken from the analysis results, so
    the window, lag interval and subinterval count all follow the analyzer's
    configuration.

    This is the only remaining caller of the deprecated ``fit_gaussian`` and
    ``fit_cubic_spline`` methods. The fitting method is chosen per call and is
    independent of ``config.xcorr_fitting_method``.

    Both methods call ``ccf_window_single(defl0, defl1)``, the opposite argument
    order from the ``get_max_time_lag_*`` methods, so the plotted time lags come
    out with the opposite sign to the analysed ones.

    Examples
    --------
    >>> plotter = CCFPlotter(analyzer)
    >>> fig, ax = plotter.plot_single(time=7.0, channel=20, method='parabola')
    """

    def __init__(self, analyzer: CorrelationAnalysis):
        self.analyzer = analyzer

    def plot_single(
        self,
        time: float,
        channel: int,
        method: Literal['gaussian', 'spline', 'parabola'] = 'gaussian',
        show_fit: bool = True,
        **kwargs
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot the cross-correlation function of a single channel at a given time.

        Parameters
        ----------
        time : float
            Centre of the analysis window in seconds.
        channel : int
            ABES channel number (1-40).
        method : {'gaussian', 'spline', 'parabola'}, default='gaussian'
            Peak fitting method to visualize.
        show_fit : bool, default=True
            Whether to fit and draw the peak. When False only the CCF data is
            plotted.
        **kwargs
            Plot styling overrides: 'alpha', 'error_alpha', 'color',
            'linewidth', 'peak_size', 'grid', 'fontsize', 'title_fontsize',
            'legend_fontsize'.

        Returns
        -------
        fig : matplotlib.figure.Figure
            Figure object.
        ax : matplotlib.axes.Axes
            Axes object with the plot.

        Raises
        ------
        ValueError
            If `method` is not one of the three listed above.

        Notes
        -----
        The plot displays the CCF data with a shaded +/-1-sigma error band, the
        fitted curve, and the peak position as a red marker with a vertical
        line; for 'parabola' the peak also carries a horizontal error bar. The
        title repeats the channel, time and fitted time lag.

        The window is sliced with linear interpolation, and the two deflection
        states are not truncated to a common sample count as they are in the
        analysis.
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

        # Plot raw CCF data with shaded error band
        ax.fill_between(
            time_lags, ccf.data - ccf.error, ccf.data + ccf.error,
            color='C0', alpha=kwargs.get('error_alpha', 0.2)
        )
        ax.plot(
            time_lags, ccf.data,
            color='C0',
            alpha=kwargs.get('alpha', 0.7),
            marker = 'o',
            ls = '--',
            label='CCF data'
        )

        # Fit and plot curve
        if show_fit:
            ts = np.linspace(min(time_lags), max(time_lags), 1000)

            if method == 'gaussian':
                tau_max, ccf_max, popt = self.analyzer.fit_gaussian(ccf)
                ax.plot(ts, gaussian_func(ts, *popt),
                       label='Gaussian fit',
                       color=kwargs.get('color', 'C1'),
                       linewidth=kwargs.get('linewidth', 2))
                title_str = f'Channel {channel}, t = {time:.2f} s\n$\\tau$ = {tau_max:.2f} $\\mu$s'

            elif method == 'spline':
                tau_max, ccf_max, cs = self.analyzer.fit_cubic_spline(ccf)
                ax.plot(ts, cs(ts),
                       label='Cubic spline',
                       color=kwargs.get('color', 'C1'),
                       linewidth=kwargs.get('linewidth', 2))
                title_str = f'Channel {channel}, t = {time:.2f} s\n$\\tau$ = {tau_max:.2f} $\\mu$s'

            elif method == 'parabola':
                tau_max, tau_err, ccf_max, popt = self.analyzer.fit_parabola(ccf)
                if popt is not None:
                    ind_max = np.argmax(np.abs(ccf.data))
                    time_lag_slice = time_lags[ind_max-2:ind_max+3]
                    ts_par = np.linspace(time_lag_slice[0], time_lag_slice[-1], 200)
                    ax.plot(ts_par, parabolic_func(ts_par, *popt),
                           label='Parabola fit',
                           color=kwargs.get('color', 'C1'),
                           linewidth=kwargs.get('linewidth', 2))
                title_str = (f'Channel {channel}, t = {time:.2f} s\n'
                             f'$\\tau$ = {tau_max:.2f} $\\pm$ {tau_err:.2f} $\\mu$s')

            else:
                raise ValueError(f"Invalid method '{method}'. Must be 'gaussian', 'spline', or 'parabola'.")

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

            if method == 'parabola':
                ax.errorbar(tau_max, ccf_max, xerr=tau_err,
                            color='red', capsize=4, capthick=1.5,
                            elinewidth=1.5, zorder=4)
            
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
        method: Literal['gaussian', 'spline', 'parabola'] = 'spline',
        channels: Optional[range] = None,
        figsize: Tuple[int, int] = (40, 40),
        nrows: int = 8,
        ncols: int = 5,
        **kwargs
    ) -> Tuple[plt.Figure, np.ndarray]:
        """
        Plot the cross-correlation functions of many channels at one time point.

        Parameters
        ----------
        time : float
            Centre of the analysis window in seconds.
        method : {'gaussian', 'spline', 'parabola'}, default='spline'
            Peak fitting method to use in every subplot.
        channels : range, optional
            Channel numbers to plot. Default is ``range(1, 41)``, all 40
            channels.
        figsize : tuple, default=(40, 40)
            Figure size as (width, height) in inches.
        nrows : int, default=8
            Number of subplot rows.
        ncols : int, default=5
            Number of subplot columns. ``nrows * ncols`` must be at least
            ``len(channels)``.
        **kwargs
            Plot styling overrides: 'suptitle_fontsize'.

        Returns
        -------
        fig : matplotlib.figure.Figure
            Figure object.
        axes : numpy.ndarray
            2D array of the subplot axes, including the unused ones, which are
            turned off.

        Raises
        ------
        ValueError
            If `method` is not one of the three listed above.

        Notes
        -----
        One subplot per channel, each showing the CCF data with its error band,
        the fitted curve, the peak marker and the fitted time lag in the
        subplot title. The figure title repeats the method and time point.

        The time window is sliced once for all channels, then per channel, which
        is the faster order.
        """
        if channels is None:
            channels = range(1, 41)

        # Create subplot grid
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

        method_name = {'spline': 'Cubic Spline', 'gaussian': 'Gaussian', 'parabola': 'Parabola'}.get(method, method)
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
            elif method == 'parabola':
                tau_max, tau_err, ccf_max, popt = self.analyzer.fit_parabola(ccf)
                if popt is not None:
                    ind_max = np.argmax(np.abs(ccf.data))
                    time_lag_slice = time_lags[ind_max-2:ind_max+3]
                    ts_par = np.linspace(time_lag_slice[0], time_lag_slice[-1], 200)
                    ax.plot(ts_par, parabolic_func(ts_par, *popt), color='C1', linewidth=1.5)
            else:
                raise ValueError(f"Invalid method '{method}'. Must be 'gaussian', 'spline', or 'parabola'.")

            # Plot raw data with shaded error band
            ax.fill_between(time_lags, ccf.data - ccf.error, ccf.data + ccf.error,
                            color='C0', alpha=0.2)
            ax.scatter(time_lags, ccf.data, color='C0', alpha=0.6, s=20)
            ax.scatter(tau_max, ccf_max, color='red', s=50, zorder=5, edgecolors='black', linewidths=1)
            ax.axvline(x=tau_max, color='red', linestyle='--', alpha=0.4, linewidth=1)
            if method == 'parabola':
                ax.errorbar(tau_max, ccf_max, xerr=tau_err,
                            color='red', capsize=3, capthick=1, elinewidth=1, zorder=4)

            # Styling
            if method == 'parabola':
                ax.set_title(
                    f'CH{ch}: $\\tau$ = {tau_max:.2f} $\\pm$ {tau_err:.2f} $\\mu$s',
                    fontsize=10
                )
            else:
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
