#!/usr/bin/env python3

import subprocess
from pathlib import Path
from glob import glob

# SET THESE VARIABLES: 
X_SAMPLES = 2
Y_SAMPLES = 2
OBJECT_NAME = "arrow"
SPAD_POSITION = [0.1, 0.4, 0.5]

# === Configuration Constants ===
DASHBOARD_FULLSCREEN = True
DASHBOARD_CONFIG = "PyQtGraphDashboardConfig"
GANTRY_CONFIG = "SingleDrive1AxisGantry"
GANTRY_AXES_KWARGS = {}
SENSOR_CONFIG = "TMF8828Config"

#set dev ports
def find_port(prefix: str) -> str:
    ports = glob("/dev/cu.*")
    matches = [p for p in ports if prefix in p]
    if not matches:
        raise RuntimeError(f"No serial port matching '{prefix}' found under /dev/cu.*")
    return sorted(matches)[0]
SENSOR_PORT = find_port("usbmodem")
GANTRY_PORT = find_port("usbserial")

CAPTURE_SCRIPT = Path(__file__).parent / "examples" / "spad_gantry_capture" / "spad_gantry_capture.py"

def build_command():
    spad_pos_str = "[" + ",".join(str(v) for v in SPAD_POSITION) + "]"

    cmd = [
        "python", str(CAPTURE_SCRIPT),
        f"sensor.port={SENSOR_PORT}",
        f"+gantry.port={GANTRY_PORT}",
        f"dashboard.fullscreen={str(DASHBOARD_FULLSCREEN).lower()}",
        f"dashboard={DASHBOARD_CONFIG}",
        f"gantry={GANTRY_CONFIG}",
        "+gantry.axes_kwargs={}",
        f"x_samples={X_SAMPLES}",
        f"y_samples={Y_SAMPLES}",
        f"sensor={SENSOR_CONFIG}",
        f"+object={OBJECT_NAME}",
        f"+spad_position={spad_pos_str}",
    ]
    return cmd


def main():
    cmd = build_command()
    print("Running:", " \\\" ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
