import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import pyqtgraph.opengl as gl
import torch
from model import RegressionModel, RegressionModelSeparate
from PyQt5 import QtCore
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QApplication, QMainWindow

from cc_hardware.drivers.spads import SPADSensor, SPADSensorConfig
from cc_hardware.tools.cli import register_cli, run_cli
from cc_hardware.utils import get_logger
from cc_hardware.utils.manager import Manager


class Viz3D(QMainWindow):
    def __init__(self, radius_only: bool):
        super().__init__()
        self.radius_only = radius_only

        self.setWindowTitle("3D -like Arc Visualization")
        self.resize(800, 800)

        self.view = gl.GLViewWidget()
        self.view.setBackgroundColor("w")
        self.setCentralWidget(self.view)
        self.view.opts["distance"] = 5
        self.view.opts["elevation"] = 90
        self.view.opts["azimuth"] = 180
        self.view.opts["center"].setX(0 if radius_only else 1)

        self.arc_surface = None
        self.arc_line = None
        self.current_radius = 0.0
        self.current_angle = 0.0
        self.target_radius = 0.0
        self.target_angle = 0.0
        self.kp = 0.1

        self._add_center_cube()
        self._add_polar_grid()

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self._update_arc)
        self.timer.start(30)

        self.lock = mp.Lock()

    def _add_center_cube(self):
        vertices = np.array(
            [
                [-0.5, -0.5, -0.5],
                [0.5, -0.5, -0.5],
                [0.5, 0.5, -0.5],
                [-0.5, 0.5, -0.5],
                [-0.5, -0.5, 0.5],
                [0.5, -0.5, 0.5],
                [0.5, 0.5, 0.5],
                [-0.5, 0.5, 0.5],
            ]
        )
        faces = np.array(
            [
                [0, 1, 2],
                [0, 2, 3],
                [4, 5, 6],
                [4, 6, 7],
                [0, 1, 5],
                [0, 5, 4],
                [2, 3, 7],
                [2, 7, 6],
                [0, 3, 7],
                [0, 7, 4],
                [1, 2, 6],
                [1, 6, 5],
            ]
        )
        colors = np.array([[0.8, 0, 0, 1]] * len(faces))
        cube = gl.GLMeshItem(
            vertexes=vertices,
            faces=faces,
            faceColors=colors,
            smooth=False,
            shader="shaded",
            drawEdges=True,
        )
        self.view.addItem(cube)
        scale = 0.33
        cube.scale(scale * 2, scale, scale)

    def _add_polar_grid(self):
        theta = np.linspace(0, 2 * np.pi, 100)
        max_radius = 5
        for r in range(1, int(max_radius) + 1):
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            z = np.zeros_like(theta)
            self.view.addItem(
                gl.GLLinePlotItem(
                    pos=np.column_stack((x, y, z)),
                    color=(0.5, 0.5, 0.5, 0.5),
                    antialias=True,
                )
            )
            self.view.addItem(
                gl.GLTextItem(
                    text=f"{r}m",
                    color=(0, 0, 0),
                    pos=(r * np.cos(np.pi / 4), r * np.sin(np.pi / 4), 0),
                    font=QFont("Arial", 25),
                )
            )

        for angle in np.linspace(0, 2 * np.pi, 12, endpoint=False):
            x = [0, max_radius * np.cos(angle)]
            y = [0, max_radius * np.sin(angle)]
            z = [0, 0]
            self.view.addItem(
                gl.GLLinePlotItem(
                    pos=np.column_stack((x, y, z)),
                    color=(0.5, 0.5, 0.5, 0.5),
                    antialias=True,
                )
            )

    def update_arcs(self, arcs_data):
        with self.lock:
            if not hasattr(self, "history"):
                self.history = []

            # Append new data and keep the history size reasonable
            self.history.append(arcs_data[0])
            if len(self.history) > 100:
                self.history.pop(0)
            # Only compute covariance if we have >= 2 points
            if len(self.history) < 2:
                self.target_radius, self.target_angle = arcs_data[0]
                return

            # Compute mean and covariance
            data_np = np.array(self.history)
            mean = data_np.mean(axis=0)
            cov = np.cov(data_np, rowvar=False)
            if np.linalg.det(cov) < 1e-8:
                cov += np.eye(cov.shape[0]) * 1e-8

            # Mahalanobis distance to filter out outliers
            x = np.array(arcs_data[0])
            diff = x - mean
            md = diff @ np.linalg.inv(cov) @ diff
            threshold = 5.991  # ~95% for 2D (chi-square)

            if md < threshold:
                self.target_radius, self.target_angle = arcs_data[0]

            diff = self.target_angle - self.current_angle
            if diff > np.pi:
                self.target_angle -= 2 * np.pi
            elif diff < -np.pi:
                self.target_angle += 2 * np.pi

    def _update_arc(self):
        with self.lock:
            self.current_radius += self.kp * (self.target_radius - self.current_radius)
            self.current_angle += self.kp * (self.target_angle - self.current_angle)

        if self.arc_surface:
            self.view.removeItem(self.arc_surface)

        angle_deg = np.degrees(self.current_angle)
        angle_offset = 190 if self.radius_only else 22.5
        start_angle = angle_deg - angle_offset
        end_angle = angle_deg + angle_offset
        theta = np.radians(np.linspace(start_angle, end_angle, 100))
        width = 0.25
        current_radius = self.current_radius - width / 2
        x = current_radius * np.cos(theta)
        y = current_radius * np.sin(theta)
        z = np.zeros_like(theta)
        x_outer = (current_radius + width) * np.cos(theta)
        y_outer = (current_radius + width) * np.sin(theta)
        vertices = np.vstack(
            [
                np.stack([x, y, z], axis=1),
                np.stack([x_outer, y_outer, z], axis=1),
            ]
        ).reshape(-1, 3)
        faces = []
        for i in range(len(theta) - 1):
            faces.append([i, i + 1, len(theta) + i])
            faces.append([i + 1, len(theta) + i, len(theta) + i + 1])
        faces = np.array(faces)
        colors = np.array([[0, 0.3, 0.7, 1]] * len(faces))

        self.arc_surface = gl.GLMeshItem(
            vertexes=vertices,
            faces=faces,
            faceColors=colors,
            smooth=True,
            shader=None,
            drawEdges=True,
        )
        self.view.addItem(self.arc_surface)


def run_visualizer(pipe, radius_only: bool):
    app = QApplication([])
    win = Viz3D(radius_only)
    win.showFullScreen()
    while True:
        if pipe.poll():
            target_data = pipe.recv()
            win.update_arcs(target_data)
        app.processEvents()
        time.sleep(0.01)


@register_cli
def algo_nlos_trainer(
    pkl: Path,
    spad: SPADSensorConfig,
    model_path: Path | None = None,
    iter: int = 0,
    radius_only: bool = False,
    merge: bool = False,
    min_bin: int = 30,
    max_bin: int = 70,
    remove_ambient: bool = False,
):
    def setup(manager: Manager):
        nonlocal model_path
        sensor: SPADSensor = spad.create_instance()
        manager.add(sensor=sensor)

        if remove_ambient:
            histogram = sensor.accumulate()
            manager.add(ambient=histogram, primitive=True)
        else:
            manager.add(ambient=None)

        h, w = sensor.resolution
        if merge:
            h = w = 1
        num_bins = sensor.num_bins
        output_size = 1 if radius_only else 2
        model = RegressionModelSeparate(
            num_bins, (h, w), output_size, min_bin=min_bin, max_bin=max_bin
        )
        if model_path is None:
            model_path = pkl.with_name(pkl.stem + "_model.pth")
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.eval()
        manager.add(model=model)

        parent_conn, child_conn = mp.Pipe()
        p = mp.Process(
            target=run_visualizer, args=(child_conn, radius_only), daemon=True
        )
        p.start()
        manager.add(mp_process=p, mp_conn=parent_conn)

        time.sleep(1)
        sensor.reset(index=iter)

    def loop(
        frame: int,
        manager: Manager,
        sensor: SPADSensor,
        model: RegressionModel,
        mp_conn,
        mp_process,
        ambient: np.ndarray | None,
    ) -> bool:
        histogram = sensor.accumulate()
        if ambient is not None:
            histogram -= ambient
        histogram = torch.from_numpy(histogram)
        if merge:
            histogram = histogram.sum(dim=(0), keepdim=True)

        with torch.no_grad():
            prediction = model(histogram).numpy()
        if radius_only:
            prediction = np.array([[prediction[0][0], 0]])
        if mp_conn:
            mp_conn.send(tuple(prediction))
        get_logger().info(f"Prediction: {prediction}")
        time.sleep(0.1)
        return True

    def cleanup(manager: Manager, mp_process, **kwargs):
        if mp_process:
            mp_process.terminate()
            mp_process.join()

    with Manager() as manager:
        manager.run(setup=setup, loop=loop, cleanup=cleanup, iter=iter)


if __name__ == "__main__":
    run_cli(algo_nlos_trainer)
