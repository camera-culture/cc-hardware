import threading
import sys
from datetime import datetime
from pathlib import Path
from functools import partial
import multiprocessing

from PyQt6.QtCore import Qt, QTimer, QEvent
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout
from PyQt6.QtGui import QColor
from PyQt6 import QtGui
from PyQt6.QtWidgets import QWidget, QLabel
from PyQt6.QtGui import QPainter, QColor
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pyqtgraph.opengl import GLMeshItem, MeshData
import trimesh
import numpy as np
import torch
from torch import nn

from cc_hardware.drivers.spads import SPADSensor
from cc_hardware.drivers.stepper_motors.stepper_controller import SnakeStepperController
from cc_hardware.drivers.stepper_motors import StepperMotorSystem
from cc_hardware.utils import Manager, get_logger, register_cli, run_cli
from cc_hardware.algos.models import DeepLocation8

NOW = datetime.now()

BINARY = False
# OUTPUT_SMOOTHING = True
OUTPUT_MOMENTUM = 0
EXP_CAPTURE_SMOOTHING = False
ROLLING_MEANS_CAPTURE = True
ASYNC = True
CAPTURE_COUNT = 1

# ZERO_COUNT = 40

STDEV_FILTERING = False

MODEL_SAVE_PATH = Path(__file__).parent / "demo_model_1.mdl"

START_BIN = 0
END_BIN = 16
NUM_BINS = END_BIN - START_BIN
WIDTH = 8
HEIGHT = 8

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
print(f"Using {device} device")


class ModelWrapper:
    def __init__(self, model, queue):
        super().__init__()
        self.queue = queue

        if OUTPUT_MOMENTUM > 0:
            self.output = None

        if EXP_CAPTURE_SMOOTHING:
            self.exp_mean_capture = 0

        if ROLLING_MEANS_CAPTURE:
            self.last_captures = 0

        if STDEV_FILTERING:
            self.captures = np.empty((0, HEIGHT, WIDTH, NUM_BINS))

        if ASYNC:
            self.external_captures = np.empty((0, HEIGHT, WIDTH, NUM_BINS))

        self.model = model
        self.model.load_state_dict(torch.load(MODEL_SAVE_PATH, weights_only=True))
        self.model.eval()

    def run_model(self):
        if ASYNC:
            if len(self.external_captures) == 0:
                return
            hists = self.external_captures
        else:
            hists = self.get_capture(count=CAPTURE_COUNT)

        # if capture is filtered out, don't update the display
        if hists.shape[0] == 0:
            return

        hist = np.mean(hists, axis=0, keepdims=True)

        hist = torch.tensor(hist, dtype=torch.float32).to(device)

        if EXP_CAPTURE_SMOOTHING:
            if type(self.exp_mean_capture) == int:
                self.exp_mean_capture = hist
            else:
                self.exp_mean_capture = 0.8 * self.exp_mean_capture + 0.2 * hist
            hist = self.exp_mean_capture
        else:
            pass

        if ROLLING_MEANS_CAPTURE:
            if type(self.last_captures) == int:
                self.last_captures = hists[:5]
            else:
                self.last_captures = np.concatenate(
                    (self.last_captures, hists[:5]), axis=0
                )
                self.last_captures = self.last_captures[-5:]
            hist = self.last_captures.mean(axis=0, keepdims=True)
            hist = torch.tensor(hist, dtype=torch.float32).to(device)

        if ASYNC:
            self.last_captures = self.external_captures[-5:]
            hist = self.last_captures.mean(axis=0, keepdims=True)
            hist = torch.tensor(hist, dtype=torch.float32).to(device)

        # model_input = hist - self.zero_hist
        model_input = hist

        with torch.no_grad():
            output = self.model(model_input)

        output = output.cpu().numpy()
        output = output.squeeze()

        return output

    def process_output(self, output):
        if OUTPUT_MOMENTUM > 0:
            if self.output is None:
                self.output = output
            else:
                self.output = (
                    OUTPUT_MOMENTUM * self.output + (1 - OUTPUT_MOMENTUM) * output
                )
            output = self.output

        print(f"output: {output}")

        return torch.tensor(output).float().unsqueeze(1)

    # def update_zero_hist(self, count=ZERO_COUNT):
    #     if ASYNC:
    #         if self.external_captures.shape[0] < count:
    #             print(f"Found {self.external_captures.shape[0]} captures, waiting for {count}")
    #             return
    #         #     time.sleep(0.1)
    #         zero_hists = self.external_captures[-count:]
    #     else:
    #         zero_hists = self.get_capture(count=count)

    #     zero_hists = torch.tensor(zero_hists, dtype=torch.float32).to(device)
    #     if count > 1:
    #         zero_hist = torch.mean(zero_hists, dim=0, keepdim=True)
    #     else:
    #         zero_hist = zero_hists
    #     self.zero_hist = zero_hist

    def get_capture(self, count=1):
        hists = self.sensor.accumulate(count, average=False)
        hists = np.array(hists)
        hists = hists.reshape(count, HEIGHT, WIDTH, END_BIN)
        hists = np.moveaxis(hists, -1, 1)
        hists = hists[:, START_BIN:END_BIN, :, :]

        if STDEV_FILTERING:
            self.captures = np.concatenate((self.captures, hists), axis=0)
            if self.captures.shape[0] > 100:
                self.captures = self.captures[-100:]

            data = hists

            # Compute the mean and standard deviation for each position (depth, 4, 4) across all samples
            mean = self.captures.mean(axis=0)  # Shape: (depth, 4, 4)
            std = self.captures.std(axis=0)  # Shape: (depth, 4, 4)

            # Compute the threshold for values being within 3 standard deviations
            lower_bound = mean - 3 * std
            upper_bound = mean + 3 * std

            # Only consider the first n values along the depth-axis (shape: n x 4 x 4)
            n = 4
            data_to_check = data[:, :n, :, :]  # Shape: (4000, n, 4, 4)
            lower_bound_check = lower_bound[:n, :, :]  # Shape: (n, 4, 4)
            upper_bound_check = upper_bound[:n, :, :]  # Shape: (n, 4, 4)

            # Identify samples where all values in the first 3 indices along the depth-axis are within bounds
            valid_mask = np.all(
                (data_to_check >= lower_bound_check)
                & (data_to_check <= upper_bound_check),
                axis=(1, 2, 3),
            )

            # Apply the mask to filter the samples
            filtered_data = data[valid_mask]

            hists = filtered_data

        return hists

    def process_external_capture(self, hists, count=1):
        hists = np.array(hists)
        hists = hists.reshape(count, HEIGHT, WIDTH, END_BIN)
        # hists = np.moveaxis(hists, -1, 1)
        hists = hists[:, :, :, START_BIN:END_BIN]
        print("processing external capture: ", hists.shape)
        if STDEV_FILTERING:
            self.captures = np.concatenate((self.captures, hists), axis=0)
            if self.captures.shape[0] > 100:
                self.captures = self.captures[-100:]

            data = hists

            # Compute the mean and standard deviation for each position (depth, 4, 4) across all samples
            mean = self.captures.mean(axis=0)  # Shape: (depth, 4, 4)
            std = self.captures.std(axis=0)  # Shape: (depth, 4, 4)

            # Compute the threshold for values being within 3 standard deviations
            lower_bound = mean - 3 * std
            upper_bound = mean + 3 * std

            # Only consider the first n values along the depth-axis (shape: n x 4 x 4)
            n = 4
            data_to_check = data[:, :n, :, :]  # Shape: (4000, n, 4, 4)
            lower_bound_check = lower_bound[:n, :, :]  # Shape: (n, 4, 4)
            upper_bound_check = upper_bound[:n, :, :]  # Shape: (n, 4, 4)

            # Identify samples where all values in the first 3 indices along the depth-axis are within bounds
            valid_mask = np.all(
                (data_to_check >= lower_bound_check)
                & (data_to_check <= upper_bound_check),
                axis=(1, 2, 3),
            )

            # Apply the mask to filter the samples
            filtered_data = data[valid_mask]

            hists = filtered_data
        self.external_captures = np.concatenate((self.external_captures, hists), axis=0)
        return hists

    def external_capture_callback(self, hists):
        self.process_external_capture(hists)
        print("external captures: ", self.external_captures.shape)
        output = self.run_model()
        output = self.process_output(output)
        # gui.update_display(output)
        self.queue.put(("output", output))
        # if self.external_captures.shape[0] < ZERO_COUNT:
        #     return
        # if self.external_captures.shape[0] == ZERO_COUNT:
        #     self.update_zero_hist()
        # if self.external_captures.shape[0] > ZERO_COUNT:
        #     self.update_display()


class KalmanFilter:
    def __init__(self, state_dim, process_noise_var, measurement_noise_var):
        """
        Initialize the Kalman Filter.
        :param state_dim: Dimension of the state (2 in this case).
        :param process_noise_var: Process noise variance (assumed diagonal).
        :param measurement_noise_var: Measurement noise variance (assumed diagonal).
        """
        self.state_dim = state_dim
        self.x = torch.zeros(state_dim, 1)  # Initial state estimate
        self.P = torch.eye(state_dim)  # Initial state covariance

        self.F = torch.eye(
            state_dim
        )  # State transition matrix (identity for random walk)
        self.Q = torch.eye(state_dim) * process_noise_var  # Process noise covariance

        self.H = torch.eye(
            state_dim
        )  # Measurement function (NN directly estimates state)
        self.R = (
            torch.eye(state_dim) * measurement_noise_var
        )  # Measurement noise covariance

    def predict(self):
        """Predict the next state and uncertainty."""
        self.x = self.F @ self.x  # State prediction
        self.P = self.F @ self.P @ self.F.T + self.Q  # Covariance prediction

    def update(self, measurement):
        """
        Update the state using a new measurement.
        :param measurement: A PyTorch tensor of shape (state_dim, 1) from the neural network.
        """
        # Kalman gain
        S = self.H @ self.P @ self.H.T + self.R  # Innovation covariance
        K = self.P @ self.H.T @ torch.inverse(S)  # Kalman gain

        # State update
        y = measurement - self.H @ self.x  # Innovation
        self.x = self.x + K @ y  # Corrected state estimate

        # Covariance update
        I = torch.eye(self.state_dim)
        self.P = (I - K @ self.H) @ self.P

    def get_state(self):
        """Return the current state estimate."""
        return self.x.clone()


class KalmanWrapper(ModelWrapper):
    def __init__(self, model, queue):
        super().__init__(model, queue)
        self.kf = KalmanFilter(
            state_dim=2, process_noise_var=0.01, measurement_noise_var=0.01
        )

    def process_output(self, output):
        self.kf.predict()
        self.kf.update(torch.tensor(output).float().unsqueeze(1))
        return self.kf.get_state()


class MovingCircleWidget(QWidget):
    def __init__(self, flip_x=False, flip_y=False):
        super().__init__()
        self.setWindowTitle("Moving Target in Physical Space")
        self.scale_factor = 20
        self.true_width = 42
        self.true_height = 35

        self.rect_width = (
            self.true_width * self.scale_factor
        )  # Scale factor for visibility
        self.rect_height = self.true_height * self.scale_factor
        self.setGeometry(
            100, 100, self.rect_width, self.rect_height
        )  # Scaled for visibility
        self.circle_radius = 20  # White circle radius

        self.center_x = self.rect_width // 2
        self.center_y = self.rect_height // 2

        # QLabel for displaying coordinates
        self.coord_label = QLabel(self)
        self.coord_label.setStyleSheet(
            "color: white; background-color: black; padding: 5px;"
        )
        self.coord_label.move(10, 10)  # Position in the top-left corner
        self.coord_label.resize(150, 20)  # Size of the label

        self.flip_x = flip_x
        self.flip_y = flip_y

        self.output = (0, 0)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(0, 0, self.rect_width, self.rect_height, QColor("black"))

        # Compute circle position
        circle_x = self.output[1] * self.scale_factor
        circle_y = self.output[0] * self.scale_factor

        # bound x and y
        if circle_x < 0:
            circle_x = 0
        if circle_x > self.rect_width:
            circle_x = self.rect_width
        if circle_y < 0:
            circle_y = 0
        if circle_y > self.rect_height:
            circle_y = self.rect_height

        # flip x direction for better visualization
        if self.flip_x:
            circle_x = self.rect_width - circle_x

        # flip y direction for better visualization
        if self.flip_y:
            circle_y = self.rect_height - circle_y

        painter.setBrush(QColor("white"))
        painter.drawEllipse(
            int(circle_x - self.circle_radius),
            int(circle_y - self.circle_radius),
            self.circle_radius * 2,
            self.circle_radius * 2,
        )

        # Update coordinate label
        self.coord_label.setText(f"X: {int(self.output[0])}, Y: {int(self.output[1])}")

    def update_display(self, output):
        print("updating display")
        self.output = output
        self.repaint()


class HistogramWidget(QWidget):
    """
    Dashboard implementation using PyQtGraph for real-time visualization.
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        print("creating histogram widget")
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAutoFillBackground(False)

        # Create a layout and add a GraphicsLayoutWidget
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.plot_widget = pg.GraphicsLayoutWidget()
        layout.addWidget(self.plot_widget)

        # Set background transparent for the plot
        self.plot_widget.setBackground(None)

        self.setFixedSize(300, 200)
        self.setStyleSheet("background: transparent;")

        self._create_plots()

    def _create_plots(self):
        self.shared_y = True

        cols, rows = [1, 1]

        self.plots = []
        self.bars = []
        bins = np.arange(START_BIN, END_BIN)

        p: pg.PlotItem = self.plot_widget.addPlot()
        self.plots.append(p)
        y = np.zeros_like(bins)
        bg = self._create_bar_graph_item(bins, y)
        p.addItem(bg)
        self.bars.append(bg)
        p.setLabel("bottom", "Time")
        p.setLabel("left", "# of Photons")
        p.setXRange(START_BIN, END_BIN, padding=0)
        p.setTitle("Raw Sensor Data", size="16")

        # autoscale
        p.enableAutoRange(axis="y", enable=True)

    def run(self):
        """
        Executes the PyQtGraph dashboard application.

        Args:
            fullscreen (bool): Whether to display in fullscreen mode.
            headless (bool): Whether to run in headless mode.
            save (Path | None): If provided, save the output to this file.
        """

        self.timer = pg.QtCore.QTimer()
        self.timer.timeout.connect(partial(self.update, frame=-1, step=False))
        self.timer.start(10)

    def update(
        self,
        *,
        histograms: np.ndarray | None = None,
        step: bool = True,
    ):
        """
        Updates the histogram data in the plots.
        """

        histogram = histograms.mean(axis=0, keepdims=True)

        self.bars[0].setOpts(height=histogram)

    def _create_bar_graph_item(self, bins, y=None):
        y = np.zeros_like(bins) if y is None else y
        return pg.BarGraphItem(
            x=bins + 0.5, height=y, width=1.0, brush=QtGui.QColor(0, 100, 255, 100)
        )

    def paintEvent(self, event):
        # Paint semi-transparent white background with rounded corners
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        bg_color = QtGui.QColor(255, 255, 255, 100)  # White with alpha
        painter.setBrush(bg_color)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(self.rect(), 0, 0)


class DemoWindow(QWidget):
    def __init__(self, flip_x=False, flip_y=False, smoothing="EXTRAPOLATE", input_queue=None):
        super().__init__()
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        QApplication.instance().installEventFilter(self)

        self.setWindowTitle("NLOS Demo")

        # GL View
        self.layout = QVBoxLayout(self)
        self.view = gl.GLViewWidget()
        self.layout.addWidget(self.view)
        self.view.setBackgroundColor("#e5e5e5")
        self.resize(1200, 800)

        # Coordinate overlay
        self.coord_label = QLabel(self)
        self.coord_label.setStyleSheet(
            "QLabel { background-color : rgba(255, 255, 255, 200); color : black; padding: 4px; }"
        )
        self.coord_label.setAlignment(
            Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignLeft
        )
        self.coord_label.setFixedWidth(150)
        self.coord_label.setFixedHeight(30)
        self.coord_label.move(10, 10)  # Position in top-left corner
        self.coord_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

        # Make sure the label is always on top
        self.coord_label.raise_()

        # Center grid (midpoint of [0, 35] and [0, 42])
        center = QtGui.QVector3D(17.5, 21.0, 0)

        # Set camera position and angle
        self.view.opts["center"] = center
        self.view.setCameraPosition(
            elevation=30,  # tilt down 30 degrees
            azimuth=0,  # looking from positive Y toward center (i.e., front view)
            distance=80,  # tweak this to zoom in/out
        )

        grid_lines = self.create_custom_grid(
            35, 42, spacing=3.5, color=(0.5, 0.5, 0.5, 1), line_width=2
        )

        # Add to the scene
        for line in grid_lines:
            self.view.addItem(line)

        # Create arrow
        self.arrow_parts = self.load_arrow_mesh()
        self.arrow = self.arrow_parts[0]  # this mesh has only 1 part
        self.view.addItem(self.arrow)

        # plane = self.create_vertical_plane(width=50, height=30, distance=20, color=(0.7, 0.7, 0.7, 0.4))
        # self.view.addItem(plane)

        # create side wall
        plane = self.create_wall(
            width=50, height=30, distance=30, color=(0.7, 0.7, 0.7, 0.5)
        )
        self.view.addItem(plane)

        # 3D viewing parameters
        self.position = [0.0, 0.0]
        self.raw_output = [0.0, 0.0]  # used for coordinate display

        self.flip_x = flip_x
        self.flip_y = flip_y
        self.scale_factor = 1
        self.true_width = 35
        self.true_height = 42

        # Set up rendering timers for smooth 3d rendering
        self.frame_timer = QTimer(self)
        self.smoothing_option = smoothing
        if self.smoothing_option == "EXTRAPOLATE":
            self.frame_timer.timeout.connect(self.render_scene_smoothing)
        elif self.smoothing_option == "INTERPOLATE":
            self.frame_timer.timeout.connect(self.render_scene_interpolated)
        elif self.smoothing_option == "NONE":
            self.frame_timer.timeout.connect(self.render_scene)
        else:
            raise Exception("Invalid smoothing option")
        self.frame_timer.start(20)  # Rendering every ms
        self.interpolation_factor = 0.0  # To interpolate between positions

        # Initial state
        self.current_position = np.array([0.0, 0.0])  # Starting position
        self.last_position = np.array(
            [0.0, 0.0]
        )  # Last applied position (to be rendered)
        self.display_position = np.array(
            [0.0, 0.0]
        )  # interpolated position for smoother rendering
        self.last_frame_time = datetime.now()
        self.last_update_time = datetime.now()
        self.duration_last_update = 0  # seconds the last update took

        # Histogram
        self.histogram_display = HistogramWidget(self)
        self.histogram_display.setFixedSize(400, 300)
        self.histogram_display.setAttribute(
            Qt.WidgetAttribute.WA_TransparentForMouseEvents
        )
        self.histogram_display.raise_()
        # self.histogram_display.run()
        self.reposition_histogram_display()

        self.input_queue = input_queue
        self.user_has_input = False
        self.user_input_timer = QTimer(self)
        self.user_input_timer.timeout.connect(self.clear_input_queue)
        self.user_input_timer.start(500)

    def resizeEvent(self, event):
        self.reposition_histogram_display()
        super().resizeEvent(event)

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Type.KeyPress:
            return self.handle_key(event)
        return False

    def create_custom_grid(
        self,
        x_max,
        y_max,
        spacing=1.0,
        z=0.0,
        color=(0.5, 0.5, 0.5, 1.0),
        line_width=1.0,
    ):
        lines = []

        # Vertical lines (along Y)
        for x in np.arange(0, x_max + spacing, spacing):
            pts = np.array([[x, 0, z], [x, y_max, z]])
            line = gl.GLLinePlotItem(
                pos=pts, color=color, width=line_width, antialias=True
            )
            lines.append(line)

        # Horizontal lines (along X)
        for y in np.arange(0, y_max + spacing, spacing):
            pts = np.array([[0, y, z], [x_max, y, z]])
            line = gl.GLLinePlotItem(
                pos=pts, color=color, width=line_width, antialias=True
            )
            lines.append(line)

        return lines

    def create_vertical_plane(
        self, width=5, height=10, distance=10, color=(0.5, 0.5, 0.5, 0.2)
    ):
        # Define vertices for a rectangular plane
        vertices = np.array(
            [
                [0, -width / 2, -height / 2],  # Bottom-left
                [0, width / 2, -height / 2],  # Bottom-right
                [0, width / 2, height / 2],  # Top-right
                [0, -width / 2, height / 2],  # Top-left
            ]
        )

        # Define the faces (which triangles to form the rectangle)
        faces = np.array([[0, 1, 2], [0, 2, 3]])

        # Create the mesh item
        mesh_data = gl.MeshData(vertexes=vertices, faces=faces)
        plane = gl.GLMeshItem(
            meshdata=mesh_data, color=color, smooth=True, drawFaces=True
        )

        # Translate the plane to be in front of the camera
        plane.translate(
            17.5 + distance, 21.0, 10
        )  # Place the plane along Z-axis, in front of the camera

        plane.setGLOptions("translucent")

        return plane

    def create_wall(self, width=5, height=10, distance=10, color=(0.5, 0.5, 0.5, 0.2)):
        # Define vertices for a rectangular plane
        vertices = np.array(
            [
                [-width / 2, 0, -height / 2],  # Bottom-left
                [width / 2, 0, -height / 2],  # Bottom-right
                [width / 2, 0, height / 2],  # Top-right
                [-width / 2, 0, height / 2],  # Top-left
            ]
        )

        # Define the faces (which triangles to form the rectangle)
        faces = np.array([[0, 1, 2], [0, 2, 3]])

        # Create the mesh item
        mesh_data = gl.MeshData(vertexes=vertices, faces=faces)
        plane = gl.GLMeshItem(
            meshdata=mesh_data, color=color, smooth=True, drawFaces=True
        )

        # Translate the plane to be in front of the camera
        plane.translate(
            17.5 + 10, 21.0 + distance, 15
        )  # Place the plane along Z-axis, in front of the camera

        plane.setGLOptions("translucent")

        return plane

    def load_arrow_mesh(self, filename=Path(__file__).parent / "arrow.obj"):
        # Load the OBJ file using trimesh
        mesh = trimesh.load(filename, force="mesh")

        # Extract vertices and faces
        vertices = mesh.vertices
        faces = mesh.faces

        # Create MeshData from numpy arrays
        mesh_data = MeshData(vertexes=vertices, faces=faces)

        # Create the GLMeshItem
        arrow = GLMeshItem(
            meshdata=mesh_data, color=(0.7, 0.7, 0.7, 1), smooth=True, drawFaces=True
        )
        arrow.translate(0, 0, -10)  # Adjust if needed
        arrow.rotate(180, 1, 0, 0)
        arrow.scale(2.0, 2.0, 2.0)  # Optional: scale model

        return [arrow]

    def set_arrow_position(self, x, y):
        print(f"Setting new position: {x}, {y}")
        self.current_position = np.array([x, y])

    def set_arrow_position_interpolated(self, x, y):
        new_target = np.array([x, y])
        if not np.allclose(new_target, self.current_position):
            # Restart interpolation from the current interpolated position
            self.last_position = self.display_position.copy()
            self.interpolation_factor = 0.0
        self.current_position = new_target

    def set_arrow_position_smoothing(self, x, y):
        new_target = np.array([x, y])
        self.last_position = self.display_position.copy()
        self.current_position = new_target
        self.duration_last_update = (datetime.now() - self.last_update_time).total_seconds()
        self.last_update_time = datetime.now()

    def render_scene(self):
        app.processEvents()
        # Only update position in the scene at the rendering rate (not with each data update)
        if np.any(self.current_position != self.last_position):
            # self.arrow.setData(pos=np.array([[0, 0, 0], self.current_position]))  # Update arrow
            dx = self.current_position[0] - self.last_position[0]
            dy = self.current_position[1] - self.last_position[1]
            self.arrow.translate(dx, dy, 0)
            self.last_position = self.current_position
    def render_scene_interpolated(self):
        app.processEvents()
        if np.any(self.current_position != self.last_position):
            self.interpolation_factor += 1 / 50
            self.interpolation_factor = min(self.interpolation_factor, 1)
            t = self.interpolation_factor
            # cubic_factor = 3 * (t**2) - 2 * (t**3)  # cubic ease-in-out factor
            display_position = self.last_position + self.interpolation_factor * (
                self.current_position - self.last_position
            )
            dx = display_position[0] - self.display_position[0]
            dy = display_position[1] - self.display_position[1]
            self.arrow.translate(dx, dy, 0)
            self.display_position = display_position
            if self.interpolation_factor >= 1:
                self.last_position = self.current_position
                self.interpolation_factor = 0.0

    def render_scene_smoothing(self):
        app.processEvents()
        frame_time = datetime.now() - self.last_frame_time
        self.last_frame_time = datetime.now()
        # print(f"frame time: {frame_time}")

        if self.duration_last_update == 0:
            return
        object_velocity = (self.current_position - self.last_position) / self.duration_last_update

        # assume velocity is relatively constant
        frame_motion = frame_time.total_seconds() * object_velocity
        target_position = self.display_position + frame_motion
        dx = target_position[0] - self.display_position[0]
        dy = target_position[1] - self.display_position[1]
        self.arrow.translate(dx, dy, 0)
        self.display_position = target_position


    def update_coord_label(self):
        x, y = self.raw_output
        self.coord_label.setText(f"x: {x:.2f}, y: {y:.2f}")

    def update_display(self, output):
        self.raw_output = output.numpy().squeeze().tolist()
        self.update_coord_label()

        # Compute circle position
        arrow_pos_x = self.raw_output[0] * self.scale_factor
        arrow_pos_y = self.raw_output[1] * self.scale_factor

        # bound x and y
        if arrow_pos_x < 0:
            arrow_pos_x = 0
        if arrow_pos_x > self.true_width:
            arrow_pos_x = self.true_width
        if arrow_pos_y < 0:
            arrow_pos_y = 0
        if arrow_pos_y > self.true_height:
            arrow_pos_y = self.true_height

        # flip x direction for better visualization
        if self.flip_x:
            arrow_pos_x = self.true_width - arrow_pos_x

        # flip y direction for better visualization
        if self.flip_y:
            arrow_pos_y = self.true_height - arrow_pos_y
        
        if self.smoothing_option == "EXTRAPOLATE":
            self.set_arrow_position_smoothing(arrow_pos_x, arrow_pos_y)
        elif self.smoothing_option == "INTERPOLATE":
            self.set_arrow_position_interpolated(arrow_pos_x, arrow_pos_y)
        elif self.smoothing_option == "NONE":
            self.set_arrow_position(arrow_pos_x, arrow_pos_y)
        else:
            raise Exception(f"Invalid smoothing option")

    def update_histograms(self, hists):
        self.histogram_display.update(histograms=hists)

    def reposition_histogram_display(self):
        margin = 10
        w, h = self.width(), self.height()
        print(f"w: {w}, h: {h}")
        self.histogram_display.move(
            margin, h - self.histogram_display.height() - margin
        )

    def handle_key(self, ev: QtGui.QKeyEvent):
        if self.input_queue is not None:
            if ev.key() == Qt.Key.Key_W:
                self.input_queue.put("W")
                self.user_has_input = True
                return True
            elif ev.key() == Qt.Key.Key_A:
                self.input_queue.put("A")
                self.user_has_input = True
                return True
            elif ev.key() == Qt.Key.Key_S:
                self.input_queue.put("S")
                self.user_has_input = True
                return True
            elif ev.key() == Qt.Key.Key_D:
                self.input_queue.put("D")
                self.user_has_input = True
                return True
        return False
    
    def clear_input_queue(self):
        # clear input queue if user has not interacted this update
        if self.input_queue is not None:
            if not self.user_has_input:
                while not self.input_queue.empty():
                    self.input_queue.get()
            self.user_has_input = False

def demo(sensor, gantry, histogram_queue, input_queue):
    gantry_thread = None
    gantry_index = 0
    gantry_pos = {"x": 0, "y": 0}

    model = DeepLocation8().to(device)
    model_wrapper = KalmanWrapper(model, histogram_queue)

    def setup(manager: Manager):
        _sensor: SPADSensor = SPADSensor.create_from_config(sensor)
        manager.add(sensor=_sensor)

        gantry_controller = SnakeStepperController(
            [
                dict(name="x", range=(0, 32), samples=2),
                dict(name="y", range=(0, 32), samples=2),
            ]
        )
        manager.add(gantry=gantry(), controller=gantry_controller)

    def loop(
        frame: int,
        manager: Manager,
        sensor: SPADSensor,
        gantry: StepperMotorSystem,
        controller: SnakeStepperController,
    ):
        get_logger().info(f"Starting iter {frame}...")

        histograms = sensor.accumulate()
        print(f"shape: {histograms.shape}")
        model_wrapper.external_capture_callback(histograms)
        histogram_queue.put(histograms)

        nonlocal gantry_thread, gantry_index, gantry_pos, input_queue
        if gantry_thread is None or not gantry_thread.is_alive():
            # Read from input queue and move accordingly
            input_speed = 0.2
            max_x = 35
            max_y = 42
            input_items = []
            queue_has_items = True
            while queue_has_items:
                try:
                    input_item = input_queue.get_nowait()
                    input_items.append(input_item)
                except:
                    queue_has_items = False

            # combine input items into one transformation
            if len(input_items) == 0:
                return
            
            total_delta_x = 0
            total_delta_y = 0
            for input_item in input_items:
                if input_item == "W":
                    total_delta_x -= input_speed
                elif input_item == "A":
                    total_delta_y += input_speed
                elif input_item == "S":
                    total_delta_x += input_speed
                elif input_item == "D":
                    total_delta_y -= input_speed
            
            gantry_pos["x"] += total_delta_x
            gantry_pos["y"] += total_delta_y
            
            if gantry_pos["x"] < 0:
                gantry_pos["x"] = 0
            if gantry_pos["x"] > max_x:
                gantry_pos["x"] = max_x
            if gantry_pos["y"] < 0:
                gantry_pos["y"] = 0
            if gantry_pos["y"] > max_y:
                gantry_pos["y"] = max_y
            
            # pos = controller.get_position(gantry_index % controller.total_positions)
            gantry_thread = threading.Thread(
                target=gantry.move_to, args=(gantry_pos["x"], gantry_pos["y"])
            )
            gantry_thread.start()
            gantry_index += 1

    def cleanup(gantry: StepperMotorSystem, **kwargs):
        if gantry_thread is not None and gantry_thread.is_alive():
            gantry_thread.join()
        gantry.move_to(0, 0)
        gantry.close()

    with Manager() as manager:
        try:
            manager.run(setup=setup, loop=loop, cleanup=cleanup)
        except KeyboardInterrupt:
            cleanup(manager.components["gantry"])


# python location_test.py dashboard=PyQtGraphDashboardConfig sensor=VL53L8CHConfig4x4 sensor.port=/dev/cu.usbmodem1103 sensor.integration_time_ms=100 sensor.cnh_num_bins=48 sensor.cnh_subsample=1 sensor.cnh_start_bin=10
# python location_test.py dashboard=PyQtGraphDashboardConfig sensor=VL53L8CHConfig8x8 sensor.port=/dev/cu.usbmodem1103 sensor.integration_time_ms=100 sensor.cnh_num_bins=16 sensor.cnh_subsample=2 sensor.cnh_start_bin=12

if __name__ == "__main__":
    mp_manager = multiprocessing.Manager()
    histogram_queue = mp_manager.Queue()
    input_queue = mp_manager.Queue()

    # model / data parameters
    height = 8
    width = 8
    depth = 16

    app = QApplication(sys.argv)

    print("Creating window")
    gui = DemoWindow(flip_x=False, flip_y=True, smoothing="EXTRAPOLATE", input_queue=input_queue)
    gui.showFullScreen()

    from PyQt6.QtCore import QTimer

    def poll_histogram_queue():
        while not histogram_queue.empty():
            msg = histogram_queue.get()
            if isinstance(msg, tuple) and msg[0] == "output":
                gui.update_display(msg[1])
            else:
                gui.update_histograms(msg)

    poll_timer = QTimer()
    poll_timer.timeout.connect(poll_histogram_queue)
    poll_timer.start(50)

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--sensor-port", type=str, required=True)
    parser.add_argument("--gantry-port", type=str, required=True)
    args = parser.parse_args()
    from cc_hardware.drivers.spads.vl53l8ch import VL53L8CHConfig8x8

    sensor_config = VL53L8CHConfig8x8.create(
        port=args.sensor_port,
        integration_time_ms=100,
        cnh_num_bins=16,
        cnh_subsample=2,
        cnh_start_bin=24,
    )
    from cc_hardware.drivers.stepper_motors.telemetrix_stepper import (
        SingleDrive1AxisGantry,
    )

    gantry = SingleDrive1AxisGantry
    cli_process = multiprocessing.Process(
        target=demo,
        args=(
            sensor_config,
            partial(
                SingleDrive1AxisGantry,
                port=args.gantry_port,
                axes_kwargs=dict(speed=2**12),
            ),
            histogram_queue,
            input_queue,
        ),
    )
    cli_process.start()
    print("cli run")

    import signal

    def close(*args):
        app.quit()
        poll_timer.stop()
        cli_process.join()
        gui.close()

    signal.signal(signal.SIGINT, close)
    try:
        app.exec()
    except:
        close()