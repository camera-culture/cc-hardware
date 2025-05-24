# Localization Demo

This is the demo with a retro-reflective arrow moving on the table.
A version of this demo was used for MIT Media Lab Members Week in Spring 2025.

You can run this demo using the following command:

```bash
python demo.py --sensor-port=[SENSOR_PORT] --gantry-port=[GANTRY_PORT]
```

`[SENSOR_PORT]` and `[GANTRY_PORT]` must be replaced with their respective USB serial ports when connected.

There is an additional toggle-able parameter `--manual-gantry` that is boolean (default=`False`) that enables the gantry to be controlled with WASD keyboard controls. If set to `False`, the gantry will follow a predetermined loop automatically.

The environment may require additional packages to be installed to run properly:

```bash
pip install requirements.txt
```

The included model is at `demo_model_1.mdl` and follows the `DeepLocation8` model using defualt parameters.
