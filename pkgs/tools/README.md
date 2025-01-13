# CC Hardware Tools

This package provides some tooling to interact with hardware devices. Primarily, it provides a single command line entrypoint which is accessible via the `cc_tools` command. Different subcommands can be registered, and can be accessed via the `cc_tools <subcommand>` command.

## Subcommands

### `dashboard`

This subcommand provides a dashboard for visualizing SPAD data. The available dashboards can be found at {mod}`~cc_hardware.tools.dashboards`.

### `jogger`

This subcommand provides a way to jog a stepper motor system. It uses {mod}`curses` to provide a simple interface for jogging the stepper motor.

### `tmf8828_flash` and `vl538lch_flash`

These subcommands provide a way to flash the TMF8828 and VL53LCH devices, respectively.

### `tmf8828_calibrate`

This subcommand provides a way to calibrate the TMF8828 device.

### `camera_viewer`

This subcommand provides a way to view the camera feed from any compatible {mod}`~cc_hardware.drivers.cameras` device.
