# SPAD Dashboard

## `spad_dashboard.py`

This demo shows how you can visualize SPAD data with a dashboard. There are currently a number of supported dashboards, and please refer to the {mod}`~cc_hardware.tools.dashboards` documentation for more information. Also, this exampel shows how to register an explicit callback with the dashboard. In this case, we use {meth}`~cc_hardware.tools.dashboards.SPADDashboard.update` explicitly, so the callback can actually just be called in the main loop. The callback becomes helpful when you use {meth}`~cc_hardware.tools.dashboards.SPADDashboard.run`, which is blocking.


## `spad_wrappers.py`

This file shows how you can use the {class}`~cc_hardware.drivers.spads.spad_wrappers.SPADWrapper` class to process spad data before returning it to the user. In the following case, we wrap a SPAD sensor with a {class}`~cc_hardware.drivers.spads.spad_wrappers.SPADMergeWrapper`, which merges neighboring pixels together.

```bash
python examples/spad_dashboard/spad_wrappers.py sensor=SPADMergeWrapper sensor/wrapped=<SPADSensor> sensor.merge_all=True dashboard=<SPADDashboard>
```

You can also group wrappers together, like the following:

```bash
python examples/spad_dashboard/spad_wrappers.py sensor=SPADMergeWrapper sensor/wrapped=SPADMovingAverageWrapper sensor/wrapped/wrapped=<SPADSensor> sensor.merge_all=True sensor.wrapped.window_size=5 dashboard=<SPADDashboard>
```
