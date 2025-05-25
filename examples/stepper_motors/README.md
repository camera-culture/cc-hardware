# Stepper Motor Controller

## `stepper_motor_controller_cli.py`

```bash
python examples/stepper_motors/stepper_motor_controller_cli.py \
    controller=<CONTROLLER> \
    stepper_system=<GANTRY> \
    controller.axes.y.samples=2 # override the default value
```

## `stepper_system_w_spad.py`

```bash
python examples/stepper_motors/stepper_system_w_spad.py \
    controller=<CONTROLLER> \
    stepper_system=<GANTRY> \
    sensor=<SPAD> \
    dashboard=<DASHBOARD>
```
