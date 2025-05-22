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

from cc_hardware.algos.models import DeepLocation8


OUTPUT_MOMENTUM = 0.9

EXP_CAPTURE_SMOOTHING = False
ROLLING_MEANS_CAPTURE = True
CAPTURE_COUNT = 1
MODEL_SAVE_PATH = Path(__file__).parent / "display-box-2.mdl"
START_BIN = 0
END_BIN = 16

ZERO_COUNT = 10

STDEV_FILTERING = False

class CounterCNN(nn.Module):
    def __init__(self, out_channels=3):
        super(CounterCNN, self).__init__()
        # self.conv1 = nn.Conv2d(in_channels=(end_bin - start_bin), out_channels=16, kernel_size=3, padding=1)
        # self.batchnorm1 = nn.BatchNorm2d(16)
        # self.conv2 = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, padding=1)
        # self.batchnorm2 = nn.BatchNorm2d(64)

        conv_out_channels = 4
        self.conv3d = nn.Conv3d(in_channels=1, out_channels=conv_out_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.batchnorm3d = nn.BatchNorm3d(conv_out_channels)

        self.fc1 = nn.Linear(conv_out_channels * (END_BIN - START_BIN) * height * width, 128)

        # self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        # self.fc1 = nn.Linear(64 * 4 * 4, 128)
        self.fc1_bn = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, out_channels)  # Assuming 10 classes for the labels
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

class Color(QWidget):
    def __init__(self, color, text="initalizing..."):
        super().__init__()
        self.setAutoFillBackground(True)

        palette = self.palette()
        palette.setColor(QPalette.ColorRole.Window, QColor(color))
        self.setPalette(palette)

        # Create a QLabel with the specified text
        self.label = QLabel(text, self)
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setFont(QFont("Arial", 72))  # Customize font and size

        # Create a layout and add the label to center it
        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.setContentsMargins(0, 0, 0, 0)  # Optional: remove margins
        self.setLayout(layout)

    def set_color(self, color):
        palette = self.palette()
        palette.setColor(QPalette.ColorRole.Window, QColor(color))
        self.setPalette(palette)
    
    def set_text(self, text):
        self.label.setText(text)

# Subclass QMainWindow to customize your application's main window
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Demo")

        self.setFixedSize(QSize(1400, 600))

        layout = QHBoxLayout()

        self.colors = ["blue", "blue", "blue"]

        for i in range(3):
            layout.addWidget(Color(self.colors[i]))
        
        self.heats = [0 for _ in range(len(self.colors))]

        if EXP_CAPTURE_SMOOTHING:
            self.exp_mean_capture = 0

        if ROLLING_MEANS_CAPTURE:
            self.last_captures = 0

        self.external_captures = np.empty((0, height, width, END_BIN - START_BIN))

        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)
        
        self.model = DeepLocation8(height=4, width=4, num_bins=16, out_dims=3).to(device)
        self.model.load_state_dict(torch.load(MODEL_SAVE_PATH, weights_only=True))
        self.model.eval()

    def update_display(self):
        if len(self.external_captures) == 0:
            return
        hists = self.external_captures

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
                self.last_captures = np.concatenate((self.last_captures, hists[:5]), axis=0)
                self.last_captures = self.last_captures[-5:]
            hist = self.last_captures.mean(axis=0, keepdims=True)
            hist = torch.tensor(hist, dtype=torch.float32).to(device)

        self.last_captures = self.external_captures[-5:]
        hist = self.last_captures.mean(axis=0, keepdims=True)
        hist = torch.tensor(hist, dtype=torch.float32).to(device)
        
        model_input = hist
        with torch.no_grad():
            output = self.model(model_input)
            output = torch.sigmoid(output)


        output = output.squeeze(axis=0)
        print(output)
        output = output.flip(0)  # flip array for outward facing display

        self.heats = [OUTPUT_MOMENTUM * heat 
                        + (1-OUTPUT_MOMENTUM) * output[i] for i, heat in enumerate(self.heats)]
        output = self.heats

        strengths = np.array([0 for _ in self.colors])

        for i in range(len(self.colors)):
            if output[i] > 0.5:
                strengths[i] += 3
            elif output[i] > 0.3:
                strengths[i] += 2
            elif output[i] > 0.15:
                strengths[i] += 1
            else:
                strengths[i] += 0

        for i in range(len(self.colors)):
            if strengths[i] > 2.5:
                self.colors[i] = "red"
            elif strengths[i] > 1.5:
                self.colors[i] = "yellow"
            elif strengths[i] > 0.5:
                self.colors[i] = "green"
            else:
                self.colors[i] = "blue"

        for i in range(len(self.colors)):
            self.centralWidget().layout().itemAt(i).widget().set_color(self.colors[i])
            self.centralWidget().layout().itemAt(i).widget().set_text(f"{output[i]:.2f}")
            
    
    def update_zero_hist(self, count=ZERO_COUNT):
        if self.external_captures.shape[0] < count:
            print(f"Found {self.external_captures.shape[0]} captures, waiting for {count}")
            return
        #     time.sleep(0.1)
        zero_hists = self.external_captures[-count:]

        zero_hists = torch.tensor(zero_hists, dtype=torch.float32).to(device)
        if count > 1:
            zero_hist = torch.mean(zero_hists, dim=0, keepdim=True)
        else:
            zero_hist = zero_hists
        self.zero_hist = zero_hist

    def process_external_capture(self, hists, count=1):
        hists = np.array(hists)
        hists = hists.reshape(count, height, width, depth)
        hists = hists[:, :, :, START_BIN:END_BIN]
        print("processing external capture: ", hists.shape)
        self.external_captures = np.concatenate((self.external_captures, hists), axis=0)
        return hists
    
    def external_capture_callback(self, hists):
        self.process_external_capture(hists)
        print("external captures: ", self.external_captures.shape)
        # if self.external_captures.shape[0] < ZERO_COUNT:
        #     return
        # if self.external_captures.shape[0] == ZERO_COUNT:
        #     self.update_zero_hist()
        # if self.external_captures.shape[0] > ZERO_COUNT:
        #     self.update_display()
        self.update_display()


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
    save_data: bool = True,
    filename: Path | None = None,
    logdir: Path = Path("logs") / NOW.strftime("%Y-%m-%d"),
):
    """Sets up and runs the SPAD dashboard.

    Args:
        sensor (SPADSensorConfig): Configuration object for the sensor.
        dashboard (SPADDashboardConfig): Configuration object for the dashboard.
    """

    def setup(manager: Manager):
        if save_data:
            assert filename is not None, "Filename must be provided if saving data."
            assert (
                logdir / filename
            ).suffix == ".pkl", "Filename must have .pkl extension."
            assert not (
                logdir / filename
            ).exists(), "File already exists. Please provide a new filename."
            logdir.mkdir(exist_ok=True, parents=True)
            manager.add(writer=PklHandler(logdir / filename))

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
        writer: PklHandler | None = None,
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
        dashboard.update(frame, histograms=histograms)
        # print(f"shape: {histograms.shape}")
        window.external_capture_callback(histograms)


        if save_data:
            assert writer is not None
            writer.append(
                {
                    "iter": iter,
                    "histogram": histograms,
                }
            )

    with Manager() as manager:
        manager.run(setup=setup, loop=loop)

# SCRIPT TO RUN:
# python demo_new2.py dashboard=PyQtGraphDashboardConfig sensor=VL53L8CHConfig4x4 save_data=False sensor.cnh_start_bin=16 sensor.cnh_num_bins=16 sensor.cnh_subsample=3 sensor.integration_time_ms=100

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
    height = 4
    width = 4
    depth = 16

    app = QApplication(sys.argv)

    print("Creating window")
    window = MainWindow()
    window.show()
    print("Window created")

    run_cli(spad_dashboard2)
    print("cli run")

    # app.exec()