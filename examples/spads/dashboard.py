import threading
import time

from cc_hardware.drivers.spads import SPADSensor
from cc_hardware.drivers.spads.dashboard import SPADDashboard
from cc_hardware.utils.logger import get_logger

# Shared variable and lock
lock = threading.Lock()
num_bins_to_set = None


def handle_user_input():
    time.sleep(1)
    global num_bins_to_set
    while True:
        try:
            user_input = int(input("Enter the number of bins: "))
            with lock:
                num_bins_to_set = user_input
            print(f"Queued num_bins update to {user_input}")
        except ValueError:
            print("Invalid input. Please enter an integer.")


def my_callback(dashboard: SPADDashboard):
    global num_bins_to_set
    with lock:
        if num_bins_to_set is not None:
            dashboard.sensor.num_bins = num_bins_to_set
            num_bins_to_set = None


def main(sensor_name: str, dashboard_name: str):
    # Main application setup
    sensor = SPADSensor.create_from_registry(sensor_name)
    if not sensor.is_okay:
        get_logger().fatal("Failed to initialize sensor")
        return
    dashboard = SPADDashboard.create_from_registry(
        dashboard_name, sensor=sensor, user_callback=my_callback
    )

    # Start user input thread
    input_thread = threading.Thread(target=handle_user_input, daemon=True)
    input_thread.start()

    dashboard.run(fullscreen=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the SPAD sensor dashboard.")

    parser.add_argument(
        "--log-level", default="FATAL", help="The logging level to use."
    )
    parser.add_argument(
        "--spad", default="VL53L8CHSensor", help="The SPAD sensor to use."
    )
    parser.add_argument(
        "--dashboard", default="PyQtGraphDashboard", help="The dashboard to use."
    )

    args = parser.parse_args()

    # Set logger level
    get_logger(level=args.log_level)

    main(args.spad, args.dashboard)
