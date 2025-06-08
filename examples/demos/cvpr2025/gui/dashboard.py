import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets

from cc_hardware.tools.dashboard import Dashboard, DashboardConfig
from cc_hardware.utils import config_wrapper


@config_wrapper
class CVPR25DashboardConfig(DashboardConfig):
    x_range: tuple[float, float] = (-1.0, 1.0)
    y_range: tuple[float, float] = (-1.0, 1.0)
    point_size: float = 1.0


class CVPR25Dashboard(Dashboard[CVPR25DashboardConfig]):
    def setup(self):
        self.app = QtWidgets.QApplication([])
        self.win = QtWidgets.QMainWindow()
        self.plot = pg.PlotWidget()
        self.win.setCentralWidget(self.plot)
        self.scatter = pg.ScatterPlotItem(
            size=self.config.point_size, brush=pg.mkBrush(255, 0, 0, 255)
        )
        self.plot.addItem(self.scatter)
        self.plot.setXRange(*self.config.x_range)
        self.plot.setYRange(*self.config.y_range)
        self.win.show()

    def update(self, frame: int, positions: list[tuple[float, float]], **kwargs):
        x, y = zip(*positions)
        self.scatter.setData(x, y)
        self.app.processEvents()

    def run(self):
        self.app.exec()

    @property
    def is_okay(self) -> bool:
        return not self.win.isHidden()

    def close(self):
        QtWidgets.QApplication.quit()
        if hasattr(self, "win") and self.win is not None:
            self.win.close()
            self.win = None
        if hasattr(self, "app") and self.app is not None:
            self.app.quit()
            self.app = None
