import sys
from datetime import datetime
import time

from PyQt6.QtCore import QSize, Qt, QTimer
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout
from PyQt6.QtGui import QColor, QPalette, QFont
from PyQt6.QtWidgets import QWidget, QHBoxLayout, QLabel

import numpy as np
import torch
from torch import nn

import matplotlib.pyplot as plt
import seaborn as sns


BINARY = False
# OUTPUT_SMOOTHING = True
OUTPUT_MOMENTUM = 0
EXP_CAPTURE_SMOOTHING = False
ROLLING_MEANS_CAPTURE = True
ASYNC = True
CAPTURE_COUNT = 1

# ZERO_COUNT = 40

STDEV_FILTERING = False

MODEL_SAVE_PATH = '../../models/robotics/demo/demo_model_1.mdl'

# START_BIN = 6
# END_BIN = 48
# NUM_BINS = END_BIN - START_BIN
# WIDTH = 4
# HEIGHT = 4

START_BIN = 0
END_BIN = 16
NUM_BINS = END_BIN - START_BIN
WIDTH = 8
HEIGHT = 8

class LocationCNN(nn.Module):
    def __init__(self):
        super(LocationCNN, self).__init__()
        # self.conv1 = nn.Conv2d(in_channels=(end_bin - start_bin), out_channels=16, kernel_size=3, padding=1)
        # self.batchnorm1 = nn.BatchNorm2d(16)
        # self.conv2 = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, padding=1)
        # self.batchnorm2 = nn.BatchNorm2d(64)

        out_channels = 4
        self.conv3d = nn.Conv3d(in_channels=1, out_channels=out_channels, kernel_size=(3, 3, 7), padding=(1, 1, 3))
        self.batchnorm3d = nn.BatchNorm3d(out_channels)

        self.fc1 = nn.Linear(out_channels * NUM_BINS * HEIGHT * WIDTH, 128)

        # self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        # self.fc1 = nn.Linear(64 * 4 * 4, 128)
        self.fc1_bn = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 2)  # 2 output dimensions (x, y)
        self.relu = nn.LeakyReLU(negative_slope=0.01)
        self.dropout = nn.Dropout(p=0.7)

    def forward(self, x):
        # # print(f'x shape at start: {x.shape}')
        # x = self.relu(self.conv1(x))
        # # print(f'x shape after conv1: {x.shape}')
        # x = self.batchnorm1(x)
        # # x = self.pool(x)
        # # print(f'x shape after pool1: {x.shape}')
        # x = self.relu(self.conv2(x))
        # # print(f'x shape after conv2: {x.shape}')
        # x = self.batchnorm2(x)

        x = self.relu(self.conv3d(x.unsqueeze(1)))
        x = self.batchnorm3d(x)

        # x = self.pool(x)
        # print(f'x shape after pool2: {x.shape}')
        x = torch.flatten(x, 1)
        # print(f'x shape after flatten: {x.shape}')
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc1_bn(x)
        x = self.fc2(x)
        return x


class DeepLocation(nn.Module):
    def __init__(self):
        super(DeepLocation, self).__init__()

        # in: (n, 4, 4, 42)
        self.conv_channels = 4
        self.conv_channels2 = 8
        self.conv3d = nn.Conv3d(in_channels=1, out_channels=self.conv_channels, kernel_size=(3, 3, 7), padding=(1, 1, 3))
        # (n, 4, 4, 4, 42)
        self.batchnorm3d = nn.BatchNorm3d(self.conv_channels)
        self.batchnorm3d2 = nn.BatchNorm3d(self.conv_channels2)
        # reshape to (n, 4, 4 x 4, 42)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        # (n, 4, 4, 4, 21)
        self.conv3d2 = nn.Conv3d(in_channels=self.conv_channels, out_channels=self.conv_channels2, kernel_size=(3, 3, 5), padding=(1, 1, 2))
        # (n, 8, 4, 4, 21)
        # reshape to (n, 8, 4 x 4, 7)
        self.pool2 = nn.MaxPool1d(kernel_size=3, stride=3, padding=0)
        # (n, 8, 4, 4, 7)

        self.fc1 = nn.Linear(self.conv_channels2 * 4 * 4 * 7, 128)
        # self.fc1 = nn.Linear(self.conv_channels * NUM_BINS * HEIGHT * WIDTH / 2 / 2 / 2, 128)

        self.fc1_bn = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 2)  # 2 output dimensions (x, y)
        self.relu = nn.LeakyReLU(negative_slope=0.01)
        self.dropout = nn.Dropout(p=0.7)

    def forward(self, x):
        x = self.relu(self.conv3d(x.unsqueeze(1)))
        x = self.batchnorm3d(x)
        # print(x.shape)
        x = torch.reshape(x, (x.shape[0], self.conv_channels * 4 * 4, NUM_BINS))
        x = self.pool1(x)
        # print(x.shape)
        x = torch.reshape(x, (x.shape[0], self.conv_channels, 4, 4, -1))
        x = self.relu(self.conv3d2(x))
        x = self.batchnorm3d2(x)
        x = torch.reshape(x, (x.shape[0], self.conv_channels2 * 4 * 4, -1))
        x = self.pool2(x)
        # print(x.shape)
        x = torch.reshape(x, (x.shape[0], self.conv_channels2, 4, 4, -1))

        x = torch.flatten(x, 1)
        # print(f'x shape after flatten: {x.shape}')
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc1_bn(x)
        x = self.fc2(x)
        return x



class DeepLocation8(nn.Module):
    def __init__(self):
        super(DeepLocation8, self).__init__()

        # in: (n, HEIGHT, WIDTH, 16)
        self.conv_channels = 4
        self.conv_channels2 = 8
        self.conv3d = nn.Conv3d(in_channels=1, out_channels=self.conv_channels, kernel_size=(3, 3, 7), padding=(1, 1, 3))
        # (n, 4, HEIGHT, WIDTH, 16)
        self.batchnorm3d = nn.BatchNorm3d(self.conv_channels)
        self.batchnorm3d2 = nn.BatchNorm3d(self.conv_channels2)
        # reshape to (n, 4, HEIGHT x WIDTH, 16)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        # (n, 4, HEIGHT, WIDTH, 8)
        self.conv3d2 = nn.Conv3d(in_channels=self.conv_channels, out_channels=self.conv_channels2, kernel_size=(3, 3, 5), padding=(1, 1, 2))
        # (n, 8, HEIGHT, WIDTH, 8)
        # reshape to (n, 8, HEIGHT x WIDTH, 8)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        # (n, 8, HEIGHT, WIDTH, 4)

        self.fc1 = nn.Linear(self.conv_channels2 * HEIGHT * WIDTH * 4, 128)
        # self.fc1 = nn.Linear(self.conv_channels * NUM_BINS * HEIGHT * WIDTH / 2 / 2 / 2, 128)

        self.fc1_bn = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 2)  # 2 output dimensions (x, y)
        self.relu = nn.LeakyReLU(negative_slope=0.01)
        self.dropout = nn.Dropout(p=0.7)

    def forward(self, x):
        x = self.relu(self.conv3d(x.unsqueeze(1)))
        x = self.batchnorm3d(x)
        # print(x.shape)
        x = torch.reshape(x, (x.shape[0], self.conv_channels * HEIGHT * WIDTH, NUM_BINS))
        x = self.pool1(x)
        # print(x.shape)
        x = torch.reshape(x, (x.shape[0], self.conv_channels, HEIGHT, WIDTH, -1))
        x = self.relu(self.conv3d2(x))
        x = self.batchnorm3d2(x)
        x = torch.reshape(x, (x.shape[0], self.conv_channels2 * HEIGHT * WIDTH, -1))
        x = self.pool2(x)
        # print(x.shape)
        x = torch.reshape(x, (x.shape[0], self.conv_channels2, HEIGHT, WIDTH, -1))

        x = torch.flatten(x, 1)
        # print(f'x shape after flatten: {x.shape}')
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc1_bn(x)
        x = self.fc2(x)
        return x



class ModelWrapper():
    def __init__(self, model):
        super().__init__()
        
        # if OUTPUT_SMOOTHING:
        # self.binary_heats = [0]
        # self.location_heats = [0 for _ in range(len(self.colors))]
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
        
        # self.sensor.flush_internal_buffer()

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
        # hist = np.max(hists, axis=0, keepdims=True)
        # hist = hists

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
                self.last_captures = np.concatenate((self.last_captures, hists[:5]), axis=0)
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
                self.output = OUTPUT_MOMENTUM * self.output + (1 - OUTPUT_MOMENTUM) * output
            output = self.output

        print(f'output: {output}')

        return output
    
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
            std = self.captures.std(axis=0)    # Shape: (depth, 4, 4)

            # Compute the threshold for values being within 3 standard deviations
            lower_bound = mean - 3 * std
            upper_bound = mean + 3 * std

            # Only consider the first n values along the depth-axis (shape: n x 4 x 4)
            n = 4
            data_to_check = data[:, :n, :, :]  # Shape: (4000, n, 4, 4)
            lower_bound_check = lower_bound[:n, :, :]  # Shape: (n, 4, 4)
            upper_bound_check = upper_bound[:n, :, :]  # Shape: (n, 4, 4)

            # Identify samples where all values in the first 3 indices along the depth-axis are within bounds
            valid_mask = np.all((data_to_check >= lower_bound_check) & (data_to_check <= upper_bound_check), axis=(1, 2, 3))

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
            std = self.captures.std(axis=0)    # Shape: (depth, 4, 4)

            # Compute the threshold for values being within 3 standard deviations
            lower_bound = mean - 3 * std
            upper_bound = mean + 3 * std

            # Only consider the first n values along the depth-axis (shape: n x 4 x 4)
            n = 4
            data_to_check = data[:, :n, :, :]  # Shape: (4000, n, 4, 4)
            lower_bound_check = lower_bound[:n, :, :]  # Shape: (n, 4, 4)
            upper_bound_check = upper_bound[:n, :, :]  # Shape: (n, 4, 4)

            # Identify samples where all values in the first 3 indices along the depth-axis are within bounds
            valid_mask = np.all((data_to_check >= lower_bound_check) & (data_to_check <= upper_bound_check), axis=(1, 2, 3))

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
        gui.update_display(output)
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

        self.F = torch.eye(state_dim)  # State transition matrix (identity for random walk)
        self.Q = torch.eye(state_dim) * process_noise_var  # Process noise covariance

        self.H = torch.eye(state_dim)  # Measurement function (NN directly estimates state)
        self.R = torch.eye(state_dim) * measurement_noise_var  # Measurement noise covariance

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
    def __init__(self, model):
        super().__init__(model)
        self.kf = KalmanFilter(state_dim=2, process_noise_var=0.01, measurement_noise_var=0.1)

    def process_output(self, output):
        self.kf.predict()
        self.kf.update(torch.tensor(output).float().unsqueeze(1))
        return self.kf.get_state()


import sys
import math
from PyQt6.QtWidgets import QApplication, QWidget
from PyQt6.QtGui import QPainter, QColor
from PyQt6.QtCore import QTimer

class MovingCircleWidget(QWidget):
    def __init__(self, flip_x=False, flip_y=False):
        super().__init__()
        self.setWindowTitle("Moving Target in Physical Space")
        self.scale_factor = 20
        self.true_width = 42
        self.true_height = 35
    
        self.rect_width = self.true_width * self.scale_factor  # Scale factor for visibility
        self.rect_height = self.true_height * self.scale_factor
        self.setGeometry(100, 100, self.rect_width, self.rect_height)  # Scaled for visibility
        self.circle_radius = 20  # White circle radius

        self.center_x = self.rect_width // 2
        self.center_y = self.rect_height // 2

        # QLabel for displaying coordinates
        self.coord_label = QLabel(self)
        self.coord_label.setStyleSheet("color: white; background-color: black; padding: 5px;")
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
        painter.drawEllipse(int(circle_x - self.circle_radius), 
                            int(circle_y - self.circle_radius), 
                            self.circle_radius * 2, 
                            self.circle_radius * 2)

        # Update coordinate label
        self.coord_label.setText(f"X: {int(self.output[0])}, Y: {int(self.output[1])}")
    
    def update_display(self, output):
        print("updating display")
        self.output = output
        self.repaint()


from datetime import datetime

from pathlib import Path

from cc_hardware.drivers.spads import SPADSensor, SPADSensorConfig
from cc_hardware.tools.dashboard import SPADDashboard, SPADDashboardConfig
from cc_hardware.utils import Manager, get_logger, register_cli, run_cli
from cc_hardware.utils.file_handlers import PklHandler

from cc_hardware.drivers.spads.vl53l8ch import VL53L8CHConfig4x4, VL53L8CHSharedConfig
from cc_hardware.tools.dashboard.spad_dashboard.pyqtgraph import PyQtGraphDashboardConfig

NOW = datetime.now()


@register_cli
def spad_dashboard2(
    sensor: SPADSensorConfig,
    dashboard: SPADDashboardConfig,
    # save_data: bool = True,
    # filename: Path | None = None,
    # logdir: Path = Path("logs") / NOW.strftime("%Y-%m-%d"),
):
    """Sets up and runs the SPAD dashboard.

    Args:
        sensor (SPADSensorConfig): Configuration object for the sensor.
        dashboard (SPADDashboardConfig): Configuration object for the dashboard.
    """

    def setup(manager: Manager):
        # if save_data:
        #     assert filename is not None, "Filename must be provided if saving data."
        #     assert (
        #         logdir / filename
        #     ).suffix == ".pkl", "Filename must have .pkl extension."
        #     assert not (
        #         logdir / filename
        #     ).exists(), "File already exists. Please provide a new filename."
        #     logdir.mkdir(exist_ok=True, parents=True)
        #     manager.add(writer=PklHandler(logdir / filename))

        # print(sensor)
        _sensor: SPADSensor = SPADSensor.create_from_config(sensor)
        manager.add(sensor=_sensor)

        _dashboard: SPADDashboard = dashboard.create_from_registry(
            config=dashboard, sensor=_sensor
        )
        _dashboard.setup()
        manager.add(dashboard=_dashboard)

    def loop(
        frame: int,
        manager: Manager,
        sensor: SPADSensor,
        dashboard: SPADDashboard,
        # writer: PklHandler | None = None,
    ):
        """Updates dashboard each frame.

        Args:
            frame (int): Current frame number.
            manager (Manager): Manager controlling the loop.
            sensor (SPADSensor): Sensor instance (unused here).
            dashboard (SPADDashboard): Dashboard instance to update.
        """
        get_logger().info(f"Starting iter {frame}...")

        histograms = sensor.accumulate()
        print(f"shape: {histograms.shape}")
        dashboard.update(frame, histograms=histograms)
        # print(f"shape: {histograms.shape}")
        model_wrapper.external_capture_callback(histograms)


        # if save_data:
        #     assert writer is not None
        #     writer.append(
        #         {
        #             "iter": iter,
        #             "histogram": histograms,
        #         }
        #     )

    with Manager() as manager:
        manager.run(setup=setup, loop=loop)

# python location_test.py dashboard=PyQtGraphDashboardConfig sensor=VL53L8CHConfig4x4 sensor.port=/dev/cu.usbmodem1103 sensor.integration_time_ms=100 sensor.cnh_num_bins=48 sensor.cnh_subsample=1 sensor.cnh_start_bin=10
# python location_test.py dashboard=PyQtGraphDashboardConfig sensor=VL53L8CHConfig8x8 sensor.port=/dev/cu.usbmodem1103 sensor.integration_time_ms=100 sensor.cnh_num_bins=16 sensor.cnh_subsample=2 sensor.cnh_start_bin=12


if __name__ == "__main__":

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    # model / data parameters
    height = 8
    width = 8
    depth = 16

    app = QApplication(sys.argv)

    print("Creating window")
    model = DeepLocation8().to(device)
    model_wrapper = KalmanWrapper(model)
    gui = MovingCircleWidget(flip_x=True, flip_y=False)
    gui.show()


    run_cli(spad_dashboard2)
    print("cli run")

    # app.exec()