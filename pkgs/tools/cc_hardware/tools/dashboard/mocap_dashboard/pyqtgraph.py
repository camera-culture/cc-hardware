from functools import partial

import numpy as np
import pyqtgraph.opengl as gl
from pyqtgraph.Qt import QtCore, QtWidgets

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
        **kwargs,
    ):
        self.x = gl.GLLinePlotItem(
            pos=np.array([[0, 0, 0], [1, 0, 0]]),
            color=(1, 0, 0, 1),
            width=width,
            antialias=antialias,
            **kwargs,
        )
        view.addItem(self.x)
        self.y = gl.GLLinePlotItem(
            pos=np.array([[0, 0, 0], [0, 1, 0]]),
            color=(0, 1, 0, 1),
            width=width,
            antialias=antialias,
            **kwargs,
        )
        view.addItem(self.y)
        self.z = gl.GLLinePlotItem(
            pos=np.array([[0, 0, 0], [0, 0, 1]]),
            color=(0, 0, 1, 1),
            width=width,
            antialias=antialias,
            **kwargs,
        )
        view.addItem(self.z)

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
        self.y.setData(
            pos=transformed_y[:3].T,
            color=self.y.color,
            width=self.y.width,
            antialias=self.y.antialias,
        )
        self.z.setData(
            pos=transformed_z[:3].T,
            color=self.z.color,
            width=self.z.width,
            antialias=self.z.antialias,
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
        self.origin_frame = GLFrame(self.view, "O")
        self.frames: dict[str, GLFrame] = {}

    def update_frames(self, data: dict[str, tuple[float, TransformationMatrix]]):
        for name, (_, frame) in data.items():
            if name not in self.frames:
                self.frames[name] = GLFrame(self.view, name)
            self.frames[name] @= frame

    def keyPressEvent(self, event):
        """
        Quit on Q or ESC.
        """
        if event.key() in [QtCore.Qt.Key.Key_Q, QtCore.Qt.Key.Key_Escape]:
            QtWidgets.QApplication.quit()


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

        # Timer to periodically update
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(partial(self.update, frame=-1, step=False))
        self.timer.start(50)

    def run(self):
        """
        Enter the Qt event loop.
        """
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
            self.timer.stop()
            self.timer = None

    @property
    def is_okay(self) -> bool:
        """
        A simple property to check if the window is still open.
        """
        return not self.win.isHidden()
