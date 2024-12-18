"""SPAD dashboard based on Dash for visualization in a browser."""

import threading
from pathlib import Path

import dash
import numpy as np
import plotly.graph_objs as go
from dash import dcc, html
from dash.dependencies import Input, Output, State
from plotly.subplots import make_subplots

from cc_hardware.drivers.spads import SPADDashboard
from cc_hardware.utils import get_logger


class DashDashboard(SPADDashboard):
    """
    Dashboard implementation using Dash and Plotly for web-based visualization.
    """

    def setup(
        self,
        *,
        fullscreen: bool = False,
        headless: bool = False,
        save: Path | None = None,
    ):
        """
        Sets up the layout and figures for the Dash application

        Args:
            fullscreen (bool): Unused parameter for DashDashboard.
            headless (bool): Whether to run in headless mode.
            save (Path | None): If provided, save the dashboard to this file.
        """
        if fullscreen:
            get_logger().warning(
                "Fullscreen functionality is not applicable for DashDashboard."
            )
        if headless:
            get_logger().warning(
                "Headless mode was set, but Dash is a web-based application."
            )
        if save:
            raise NotImplementedError(
                "Save functionality is not implemented for DashDashboard."
            )

        self.app = dash.Dash(__name__)

        self.lock = threading.Lock()
        self.num_updates = 0
        self.blocking = False
        self.thread: threading.Thread = None

        self.bins = np.arange(self.min_bin, self.max_bin)
        self.histograms = np.zeros((self.num_channels, self.max_bin - self.min_bin))
        rows = int(np.ceil(np.sqrt(self.num_channels)))
        cols = int(np.ceil(self.num_channels / rows))

        fig = make_subplots(
            rows=rows,
            cols=cols,
            subplot_titles=[f"Channel {ch}" for ch in self.channel_mask],
            shared_xaxes="all",
            shared_yaxes="all",
        )

        for idx, channel in enumerate(self.channel_mask):
            row = idx // cols + 1
            col = idx % cols + 1
            trace = go.Bar(
                x=self.bins,
                y=self.histograms[idx],
                name=f"Channel {channel}",
                marker_color="blue",
            )
            fig.add_trace(trace, row=row, col=col)
            fig.update_xaxes(title_text="Bin", row=row, col=col, showticklabels=True)
            fig.update_yaxes(
                title_text="Photon Counts", row=row, col=col, showticklabels=True
            )
            fig.update_xaxes(range=[self.min_bin, self.max_bin], row=row, col=col)
            if self.ylim is not None:
                fig.update_yaxes(range=[0, self.ylim], row=row, col=col)

        fig.update_layout(autosize=True, showlegend=False)

        self.app.layout = html.Div(
            [
                dcc.Graph(
                    id="live-update-graph",
                    figure=fig,
                    style={"height": "100vh", "width": "100vw"},
                ),
                dcc.Interval(
                    id="interval-component",
                    interval=1,
                    n_intervals=0,
                    max_intervals=self.num_frames,
                ),
            ],
        )
        self.existing_fig = fig

        @self.app.callback(
            Output("live-update-graph", "figure"),
            [Input("interval-component", "n_intervals")],
            [State("live-update-graph", "figure")],
        )
        def update_graph_live(n_intervals, existing_fig):
            """
            Updates the live graph with new histogram data.

            Args:
                n_intervals (int): The number of intervals that have passed.
                existing_fig (dict): The existing figure to update.
            """
            return self.update(n_intervals, existing_fig=existing_fig)

    def run(self):
        """
        Executes the Dash dashboard application.
        """
        self.blocking = True
        self.app.run(debug=False, use_reloader=False)

    def update(
        self,
        n_intervals: int,
        histograms: np.ndarray | None = None,
        existing_fig: dict | None = None,
    ):
        """
        Updates the histogram data for each frame.

        Args:
            n_intervals (int): Current frame number.

        Keyword Args:
            histograms (np.ndarray): The histogram data to update. If not provided, the
                sensor will be used to accumulate the histogram data.
        """
        if not self.blocking:
            if self.thread is None:
                self.thread = threading.Thread(target=self.run, daemon=True)
                self.thread.start()
            return dash.no_update

        self.num_updates += 1
        if n_intervals is None:
            return dash.no_update

        if histograms is None:
            acquired = self.lock.acquire(blocking=False)
            if acquired:
                try:
                    histograms = self.sensor.accumulate(1)
                finally:
                    self.lock.release()
            else:
                histograms = self.histograms
        self.histograms = histograms

        if existing_fig is None:
            existing_fig = self.existing_fig

        for idx, channel in enumerate(self.channel_mask):
            histogram = histograms[channel, self.min_bin : self.max_bin]
            xaxis_key = f"xaxis{idx + 1}" if idx > 0 else "xaxis"
            yaxis_key = f"yaxis{idx + 1}" if idx > 0 else "yaxis"

            bins = np.arange(self.min_bin, self.max_bin)

            # Update x and y for each channel
            existing_fig["data"][idx]["y"] = histogram.tolist()
            if len(existing_fig["data"][idx]["x"]) != len(bins):
                existing_fig["data"][idx]["x"] = bins.tolist()
                existing_fig["layout"][xaxis_key]["range"] = [
                    self.min_bin,
                    self.max_bin,
                ]

            if self.ylim is not None:
                existing_fig["layout"][yaxis_key]["range"] = [0, self.ylim]
            elif self.autoscale:
                existing_fig["layout"][yaxis_key]["autorange"] = True

        # Call user callback if provided
        if self.user_callback is not None:
            self.user_callback(self)

        return existing_fig
