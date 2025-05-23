# Modeling Capture

This contains scripts to capture SPAD data using the VL53L8CH SPAD sensor.

The file `spad_visualization_capture` contains scripts to capture SPAD data directly. If only one USB serial device is connected it will be used for data transfer. Otherwise, `sensor.port` must be provided as an additional parameter.

Example usage:

```bash
python spad_visualization_capture.py sensor=VL53L8CHConfig4x4 dashboard=PyQtGraphDashboardConfig save_data=False sensor.integration_time_ms=100 sensor.cnh_num_bins=16 sensor.cnh_subsample=2 sensor.cnh_start_bin=20
```

The file `spad_gantry_capture` contains scripts to capture SPAD data with gantry positioning. Both the sensor and gantry should be conencted via USB serial connections. `sensor_port` and `gantry_port` should be provided.

Example usage:

```bash
python spad_gantry_capture.py sensor=VL53L8CHConfig8x8 dashboard=PyQtGraphDashboardConfig sensor.integration_time_ms=100 sensor.cnh_num_bins=16 sensor.cnh_subsample=2 sensor.cnh_start_bin=0
```

Additional sensor parameters may be provided such as `sensor` for resolution configurations, `sensor.integration_time_ms`, `sensor.cnh_num_bins`, `sensor.cnh_subsample`, `sensor.cnh_start_bin`, etc. See `cc-hardware/pkgs/drivers/spads/vl53l8ch.py` for more configuration details.

Data is saved in the folder this script is run in, under `logs/Y-m-d/H-M-S/data.pkl` format. Data can be read from `pkl` file using the `HistogramDataset` module.

An example of loading SPAD data into a `HistogramDataset` and training a simple model are included in `training_example.ipynb`.
