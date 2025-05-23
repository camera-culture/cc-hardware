import sys
from datetime import datetime

from PyQt6.QtCore import QSize, Qt, QTimer
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout
from PyQt6.QtGui import QColor, QPalette, QFont
from PyQt6.QtWidgets import QWidget, QHBoxLayout, QLabel

from cc_hardware.drivers.spads.vl53l8ch import VL53L8CHSensor
from cc_hardware.utils.plotting import histogram_gui

import numpy as np
import torch
from torch import nn

import matplotlib.pyplot as plt
import seaborn as sns


BINARY = False
# OUTPUT_SMOOTHING = True
BINARY_OUTPUT_MOMENTUM = 0.9
LOCATION_OUTPUT_MOMENTUM = 0.8
EXP_CAPTURE_SMOOTHING = False
ROLLING_MEANS_CAPTURE = True
CAPTURE_COUNT = 1
BINARY_MODEL_SAVE_PATH = 'models/display-box-18-rolling-means-binary.mdl'
LOCATION_MODEL_SAVE_PATH = 'models/display-box-18-rolling-means-location.mdl'
START_BIN = 4
END_BIN = 16

STDEV_FILTERING = True

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

        if BINARY:
            self.colors = ["blue"]
        else:
            self.colors = ["blue", "blue", "blue"]

        if BINARY:
            layout.addWidget(Color(self.colors[0]))
        else:
            for i in range(3):
                layout.addWidget(Color(self.colors[i]))
        
        # if OUTPUT_SMOOTHING:
        self.binary_heats = [0]
        self.location_heats = [0 for _ in range(len(self.colors))]

        if EXP_CAPTURE_SMOOTHING:
            self.exp_mean_capture = 0

        if ROLLING_MEANS_CAPTURE:
            self.last_captures = 0
        
        if STDEV_FILTERING:
            self.captures = np.empty((0, END_BIN - START_BIN, height, width))

        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_display)
        self.timer.start(200) # Update every 0.1 second

        # self.zero_timer = QTimer(self)
        # self.zero_timer.timeout.connect(self.update_zero_hist)
        # self.zero_timer.start(10000) # Update every 10 seconds

        self.sensor = VL53L8CHSensor(debug=False)

        # self.sensor.flush_internal_buffer()

        self.update_zero_hist(count=40)

        self.binary_model = CounterCNN(out_channels=1).to(device)
        self.location_model = CounterCNN(out_channels=3).to(device)
        self.binary_model.load_state_dict(torch.load(BINARY_MODEL_SAVE_PATH, weights_only=True))
        self.location_model.load_state_dict(torch.load(LOCATION_MODEL_SAVE_PATH, weights_only=True))

        self.binary_model.eval()
        self.location_model.eval()

    def update_display(self):
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
        
        model_input = hist - self.zero_hist
        with torch.no_grad():
            binary_output = self.binary_model(model_input)
            binary_output = torch.sigmoid(binary_output)
            # print(output)
            location_output = self.location_model(model_input)
            location_output = torch.sigmoid(location_output)

        # Update color based on the output
        # output = torch.round(output)
        binary_output = binary_output.cpu().numpy()
        binary_output = binary_output.squeeze(axis=0)
        binary_output = binary_output[::-1]  # flip array for outward facing display

        location_output = location_output.cpu().numpy()
        location_output = location_output.squeeze(axis=0)
        location_output = location_output[::-1]  # flip array for outward facing display

        # if OUTPUT_SMOOTHING:
        self.binary_heats = [BINARY_OUTPUT_MOMENTUM * heat 
                        + (1-BINARY_OUTPUT_MOMENTUM) * binary_output[i] for i, heat in enumerate(self.binary_heats)]
        binary_output = self.binary_heats
        self.location_heats = [LOCATION_OUTPUT_MOMENTUM * heat 
                        + (1-LOCATION_OUTPUT_MOMENTUM) * location_output[i] for i, heat in enumerate(self.location_heats)]
        location_output = self.location_heats 

        strengths = np.array([0 for _ in self.colors])

        for i in range(len(self.colors)):
            if location_output[i] > 0.5:
                strengths[i] += 3
            elif location_output[i] > 0.3:
                strengths[i] += 2
            elif location_output[i] > 0.15:
                strengths[i] += 1
            else:
                strengths[i] += 0

        if binary_output[0] < 0.4:
            strengths -= 3
        elif binary_output[0] < 0.5:
            strengths -= 2
        elif binary_output[0] < 0.6:
            strengths -= 1
        else:
            strengths -= 0
        
        for i in range(len(self.colors)):
            if strengths[i] > 2.5:
                self.colors[i] = "red"
            elif strengths[i] > 1.5:
                self.colors[i] = "orange"
            elif strengths[i] > 0.5:
                self.colors[i] = "yellow"
            elif strengths[i] > -0.5:
                self.colors[i] = "green"
            else:
                self.colors[i] = "blue"

        # if binary_output[0] > 0.5:
        #     for i in range(len(self.colors)):
        #         if location_output[i] > 0.5:
        #             self.colors[i] = "red"
        #         elif location_output[i] > 0.3:
        #             self.colors[i] = "yellow"
        #         elif location_output[i] > 0.15:
        #             self.colors[i] = "green"
        #         else:
        #             self.colors[i] = "blue"
        # else:
        #     for i in range(len(self.colors)):
        #         self.colors[i] = "blue"

        for i in range(len(self.colors)):
            self.centralWidget().layout().itemAt(i).widget().set_color(self.colors[i])
            self.centralWidget().layout().itemAt(i).widget().set_text(f"{binary_output[0]:.2f}\n{location_output[i]:.2f}")
            
    
    def update_zero_hist(self, count=40):
        zero_hists = self.get_capture(count=count)
        zero_hists = torch.tensor(zero_hists, dtype=torch.float32).to(device)
        if count > 1:
            zero_hist = torch.mean(zero_hists, dim=0, keepdim=True)
        else:
            zero_hist = zero_hists
        self.zero_hist = zero_hist

    def get_capture(self, count=1):
        hists = self.sensor.accumulate(count, average=False)
        hists = np.array(hists)
        hists = hists.reshape(count, height, width, depth)
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
depth = 24

app = QApplication(sys.argv)

window = MainWindow()
window.show()

app.exec()