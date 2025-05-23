# Classification Demo

This is the demo for the location classification in front of the display case. The current version is designed to perform without retro-reflective material.

You can run this demo using the following command:

```bash
python demo.py dashboard=PyQtGraphDashboardConfig sensor=VL53L8CHConfig4x4 save_data=False sensor.cnh_start_bin=16 sensor.cnh_num_bins=16 sensor.cnh_subsample=3 sensor.integration_time_ms=100
```

The environment may require additional packages to be installed to run properly:

```bash
pip install requirements.txt
```

The included model is at `display-box-2.mdl` and follows the `DeepLocation8` model using parameters: `height=4`, `width=4`, `num_bins=16`, `out_dims=3`.
