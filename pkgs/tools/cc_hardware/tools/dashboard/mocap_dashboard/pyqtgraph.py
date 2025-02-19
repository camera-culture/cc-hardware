from functools import partial

import numpy as np
import pyqtgraph.opengl as gl
from pyqtgraph.Qt import QtCore, QtWidgets, QtGui

from cc_hardware.tools.dashboard.mocap_dashboard import (
    MotionCaptureDashboard,
    MotionCaptureDashboardConfig,
)
from cc_hardware.utils import config_wrapper
from cc_hardware.utils.transformations import TransformationMatrix


@config_wrapper
class PyQtGraphMotionCaptureDashboardConfig(MotionCaptureDashboardConfig):
    """
    Configuration class for the PyQtGraph 3D motion capture dashboard.
    E.g., can add more fields if needed.
    """

    fullscreen: bool = False


class GLFrame:
    def __init__(
        self,
        view: gl.GLViewWidget,
        name: str,
        *,
        width: int = 5,
        antialias: bool = True,
        alpha: float = 1,
        **kwargs,
    ):
        self.x = gl.GLLinePlotItem(
            pos=np.array([[0, 0, 0], [1, 0, 0]]),
            color=(1, 0, 0, alpha),
            width=width,
            antialias=antialias,
            **kwargs,
        )
        view.addItem(self.x)
        self.x_label = gl.GLTextItem(
            text="x",
            pos=self.x.pos[1],
            color=(0, 0, 0),
            font=QtGui.QFont("Arial", 15)
        )
        view.addItem(self.x_label)
        self.y = gl.GLLinePlotItem(
            pos=np.array([[0, 0, 0], [0, 1, 0]]),
            color=(0, 1, 0, alpha),
            width=width,
            antialias=antialias,
            **kwargs,
        )
        view.addItem(self.y)
        self.y_label = gl.GLTextItem(
            text="y",
            pos=self.y.pos[1],
            color=(0, 0, 0),
            font=QtGui.QFont("Arial", 15)
        )
        view.addItem(self.y_label)
        self.z = gl.GLLinePlotItem(
            pos=np.array([[0, 0, 0], [0, 0, 1]]),
            color=(0, 0, 1, alpha),
            width=width,
            antialias=antialias,
            **kwargs,
        )
        view.addItem(self.z)
        self.z_label = gl.GLTextItem(
            text="z",
            pos=self.z.pos[1],
            color=(0, 0, 0),
            font=QtGui.QFont("Arial", 15)
        )
        view.addItem(self.z_label)

        self.label = gl.GLTextItem(
            text=name,
            pos=(0.0, 0.0, 0.0),
            color=(0, 0, 0),
        )
        view.addItem(self.label)

    def __matmul__(self, mat: TransformationMatrix):
        transformed_x = mat @ np.array([[0, 0, 0, 1], [1, 0, 0, 1]]).T
        transformed_y = mat @ np.array([[0, 0, 0, 1], [0, 1, 0, 1]]).T
        transformed_z = mat @ np.array([[0, 0, 0, 1], [0, 0, 1, 1]]).T

        self.x.setData(
            pos=transformed_x[:3].T,
            color=self.x.color,
            width=self.x.width,
            antialias=self.x.antialias,
        )
        self.x_label.setData(
            text=self.x_label.text,
            pos=transformed_x[:3, 1].T,
            color=self.x_label.color,
            font=self.x_label.font,
        )
        self.y.setData(
            pos=transformed_y[:3].T,
            color=self.y.color,
            width=self.y.width,
            antialias=self.y.antialias,
        )
        self.y_label.setData(
            text=self.y_label.text,
            pos=transformed_y[:3, 1].T,
            color=self.y_label.color,
            font=self.y_label.font,
        )
        self.z.setData(
            pos=transformed_z[:3].T,
            color=self.z.color,
            width=self.z.width,
            antialias=self.z.antialias,
        )
        self.z_label.setData(
            text=self.z_label.text,
            pos=transformed_z[:3, 1].T,
            color=self.z_label.color,
            font=self.z_label.font,
        )

        self.label.setData(
            text=self.label.text,
            pos=transformed_x[:3, 0].T,
            color=self.label.color,
            font=self.label.font,
        )

    def __imatmul__(self, mat: TransformationMatrix):
        self.__matmul__(mat)
        return self


class DashboardWindow(QtWidgets.QWidget):
    """
    A QWidget that holds a 3D OpenGL view (pyqtgraph.opengl.GLViewWidget).
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        self.is_paused = False

    def init_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        self.view = gl.GLViewWidget()
        layout.addWidget(self.view)
        self.view.setBackgroundColor("w")
        self.view.opts["distance"] = 5

        # Add grid on 2d plane
        xy_grid = gl.GLGridItem(color=(0, 0, 0, 76.5))
        self.view.addItem(xy_grid)

        # Create axes
        self.origin_frame = GLFrame(self.view, "O", alpha=0.5)
        self.frames: dict[str, GLFrame] = {}

        # Add buttons
        self.button_layout = QtWidgets.QHBoxLayout()
        self.current_view_direction = None
        self.current_view_button = None
        self.prev_camera = None
        for direction in ["Orthographic X", "Orthographic -X", "Orthographic Y", "Orthographic -Y", "Orthographic Z", "Orthographic -Z"]:
            btn = QtWidgets.QPushButton(direction)
            btn.setCheckable(True)
            btn.clicked.connect(lambda checked, d=direction, b=btn: self.change_view(d, b))
            self.button_layout.addWidget(btn)
        layout.addLayout(self.button_layout)

    def update_frames(self, data: dict[str, tuple[float, TransformationMatrix]]):
        for name, (_, frame) in data.items():
            if name not in self.frames:
                self.frames[name] = GLFrame(self.view, name)
            frame @= np.array(
                [
                    [1, 0, 0, 0],
                    [0, -1, 0, 0],
                    [0, 0, -1, 0],
                    [0, 0, 0, 1],
                ]
            )
            self.frames[name] @= frame

    def change_view(self, direction, btn):
        if self.current_view_direction == direction:
            self.current_view_direction = None
            btn.setChecked(False)
            if self.prev_camera is not None:
                self.view.setCameraPosition(
                    azimuth=self.prev_camera["azimuth"],
                    elevation=self.prev_camera["elevation"],
                    distance=self.prev_camera["distance"],
                )
                self.view.opts["fov"] = self.prev_camera["fov"]
        else:
            if self.current_view_button and self.current_view_button != btn:
                self.current_view_button.setChecked(False)
            self.current_view_direction = direction
            self.current_view_button = btn
            pos_map = {
                "Orthographic X": dict(elevation=0, azimuth=0),
                "Orthographic -X": dict(elevation=0, azimuth=180),
                "Orthographic Y": dict(elevation=90, azimuth=0),
                "Orthographic -Y": dict(elevation=-90, azimuth=0),
                "Orthographic Z": dict(elevation=0, azimuth=90),
                "Orthographic -Z": dict(elevation=0, azimuth=-90),
            }
            pos = pos_map[direction]
            self.prev_camera = dict(azimuth=self.view.opts["azimuth"], elevation=self.view.opts["elevation"], distance=self.view.opts["distance"], fov=self.view.opts["fov"])
            self.view.setCameraPosition(**pos, distance=2000)
            self.view.opts["fov"] = 1

    def keyPressEvent(self, event):
        """
        Quit on Q or ESC.
        """
        if event.key() in [QtCore.Qt.Key.Key_Q, QtCore.Qt.Key.Key_Escape]:
            QtWidgets.QApplication.quit()
        elif event.key() == QtCore.Qt.Key.Key_Space:
            self.is_paused = not self.is_paused


class PyQtGraphMotionCaptureDashboard(
    MotionCaptureDashboard[PyQtGraphMotionCaptureDashboardConfig]
):
    """
    A 3D motion capture visualization dashboard using pyqtgraph's GLViewWidget.
    """

    def setup(self):
        self.app = QtWidgets.QApplication([])
        self.win = DashboardWindow()
        self.win.init_ui()

        if self.config.fullscreen:
            self.win.showFullScreen()
        else:
            self.win.show()

    def run(self):
        """
        Enter the Qt event loop.
        """
        # Timer to periodically update
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(partial(self.update, frame=-1, step=False))
        self.timer.start(50)

        self.app.exec()

    def update(
        self,
        frame: int,
        *,
        data: dict[str, TransformationMatrix] | None = None,
        step: bool = True,
    ):
        """
        Called periodically by the timer. We:
         1) Update the sensor,
         2) Accumulate the new transform,
         3) Send transform to the UI.
        """
        while self.win.is_paused:
            self.app.processEvents()

        self.sensor.update()
        if data is None:
            data = self.sensor.accumulate()
        self.win.update_frames(data)

        if step:
            self.app.processEvents()

    def close(self):
        """
        Clean up the Qt application when done.
        """
        QtWidgets.QApplication.quit()
        if hasattr(self, "win"):
            self.win.close()

        if hasattr(self, "app") and self.app is not None:
            self.app.quit()
            self.app = None

        if hasattr(self, "timer") and self.timer is not None:
            try:
                self.timer.stop()
                self.timer = None
            except:
                pass

    @property
    def is_okay(self) -> bool:
        """
        A simple property to check if the window is still open.
        """
        return not self.win.isHidden()
