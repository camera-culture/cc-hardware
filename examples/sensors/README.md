# Sensors

This folder contains examples of how to use various sensing interfaces defined within the {mod}`cc_hardware` package.

## `camera_viewer.py`

Simple example of how to visualize a camera feed.

```bash
python examples/sensors/camera_viewer.py camera=<CAMERA>
```

## `spad_dashboard_cli.py`

This demo shows how you can visualize SPAD data with a dashboard. There are currently a number of supported dashboards, and please refer to the {mod}`~cc_hardware.tools.dashboard` documentation for more information. Also, this exampel shows how to register an explicit callback with the dashboard. In this case, we use {meth}`~cc_hardware.tools.dashboard.SPADDashboard.update` explicitly, so the callback can actually just be called in the main loop. The callback becomes helpful when you use {meth}`~cc_hardware.tools.dashboard.SPADDashboard.run`, which is blocking.

You can call the example with the following command:

```bash
python examples/spad_dashboard/spad_dashboard_cli.py sensor=<SPADSensor> dashboard=<SPADDashboard>
```

You can also use wrappers to enable additional functionality, such using the {class}`~cc_hardware.drivers.spads.spad_wrappers.SPADWrapper` class to process spad data before returning it to the user. In the following case, we wrap a SPAD sensor with a {class}`~cc_hardware.drivers.spads.spad_wrappers.SPADMergeWrapper`, which merges neighboring pixels together.

```bash
python examples/spad_dashboard/spad_dashboard_cli.py sensor=SPADMergeWrapperConfig sensor/wrapped=<SPADSensor> sensor.merge_all=True dashboard=<SPADDashboard>
```

You can also group wrappers together, like the following:

```bash
python examples/spad_dashboard/spad_dashboard_cli.py sensor=SPADMergeWrapperConfig sensor/wrapped=SPADMovingAverageWrapperConfig sensor/wrapped/wrapped=<SPADSensor> sensor.merge_all=True sensor.wrapped.window_size=5 dashboard=<SPADDashboard>
```

## `spad_dashboard.py`

This is effectively the same demo, but can be run as a script.
