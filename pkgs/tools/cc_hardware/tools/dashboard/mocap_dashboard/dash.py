import threading

import dash
import numpy as np
import plotly.graph_objects as go
from dash import dcc, html
from dash.dependencies import Input, Output, State

from cc_hardware.tools.dashboard.mocap_dashboard import (
    MotionCaptureDashboard,
    MotionCaptureDashboardConfig,
)
from cc_hardware.utils import config_wrapper


@config_wrapper
class DashMotionCaptureDashboardConfig(MotionCaptureDashboardConfig):
    host: str = "127.0.0.1"
    port: int = 8050


class DashMotionCaptureDashboard(
    MotionCaptureDashboard[DashMotionCaptureDashboardConfig]
):
    def setup(self):
        self.app = dash.Dash(__name__)

        self.blocking = False
        self.thread: threading.Thread = None
        self.data_store = {}
        self.trace_indices = {}

        self.figure = go.Figure()
        self.figure.update_layout(
            scene=dict(aspectmode="cube"), autosize=True, showlegend=False
        )

        self.app.layout = html.Div(
            [
                dcc.Graph(
                    id="live-update-graph",
                    figure=self.figure,
                    style={"height": "100vh", "width": "100vw"},
                ),
                dcc.Interval(
                    id="interval-component",
                    interval=1,
                    n_intervals=0,
                    max_intervals=self.config.num_frames,
                ),
            ],
        )

        @self.app.callback(
            Output("live-update-graph", "figure"),
            [Input("interval-component", "n_intervals")],
            [State("live-update-graph", "figure")],
        )
        def update_graph_live(n_intervals, fig):
            if n_intervals is None:
                return dash.no_update

            return self.update(n_intervals, fig=fig)

    def run(self):
        self.blocking = True
        self.app.run_server(
            debug=False,
            use_reloader=False,
            host=self.config.host,
            port=self.config.port,
        )

    def update(self, frame=-1, *, data=None, step=True, fig: dict | None = None):
        if not self.blocking:
            if self.thread is None:
                self.thread = threading.Thread(target=self.run, daemon=True)
                self.thread.start()
            return dash.no_update

        self.sensor.update()
        if data is None:
            data = self.sensor.accumulate()
        for name, (_, frame) in data.items():
            self.data_store[name] = frame

        if fig is None:
            fig = self.figure
        else:
            fig = go.Figure(fig)

        for name, mat in self.data_store.items():
            o = np.array([0, 0, 0, 1])
            x = np.array([1, 0, 0, 1])
            y = np.array([0, 1, 0, 1])
            z = np.array([0, 0, 1, 1])

            # Helper to transform and return Nx3 for easy plotting
            def transform_points(m, *points):
                stacked = np.column_stack(points)
                transformed = m @ stacked
                return transformed[:3].T  # shape -> (len(points), 3)

            xt = transform_points(mat, o, x)
            yt = transform_points(mat, o, y)
            zt = transform_points(mat, o, z)

            if name not in self.trace_indices:
                # Add new traces for x, y, z, label
                x_trace = len(fig.data)
                fig.add_trace(
                    go.Scatter3d(mode="lines", line=dict(color="red", width=5))
                )
                y_trace = len(fig.data)
                fig.add_trace(
                    go.Scatter3d(mode="lines", line=dict(color="green", width=5))
                )
                z_trace = len(fig.data)
                fig.add_trace(
                    go.Scatter3d(mode="lines", line=dict(color="blue", width=5))
                )
                label_trace = len(fig.data)
                fig.add_trace(
                    go.Scatter3d(mode="text", text=[name], textposition="top center")
                )

                self.trace_indices[name] = {
                    "x": x_trace,
                    "y": y_trace,
                    "z": z_trace,
                    "label": label_trace,
                }

            # Update existing traces
            xi = self.trace_indices[name]["x"]
            yi = self.trace_indices[name]["y"]
            zi = self.trace_indices[name]["z"]
            li = self.trace_indices[name]["label"]

            fig.data[xi].x = xt[:, 0]
            fig.data[xi].y = xt[:, 1]
            fig.data[xi].z = xt[:, 2]
            fig.data[yi].x = yt[:, 0]
            fig.data[yi].y = yt[:, 1]
            fig.data[yi].z = yt[:, 2]
            fig.data[zi].x = zt[:, 0]
            fig.data[zi].y = zt[:, 1]
            fig.data[zi].z = zt[:, 2]
            fig.data[li].x = [xt[0, 0]]
            fig.data[li].y = [xt[0, 1]]
            fig.data[li].z = [xt[0, 2]]

        # Call user callback if provided
        if self.config.user_callback is not None:
            self.config.user_callback(self)

        return fig

    @property
    def is_okay(self) -> bool:
        return True

    def close(self):
        if not hasattr(self, "app") or self.app is None:
            return
        self.blocking = False
        if self.thread is not None:
            self.thread.join()
        self.app = None
