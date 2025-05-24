"""Camera drivers for the cc-hardware package."""

from cc_hardware.drivers.cameras.camera import Camera, CameraConfig

# Register the camera implementations
Camera.register("USBCamera", f"{__name__}.usb")
CameraConfig.register("USBCameraConfig", f"{__name__}.usb")
CameraConfig.register("USBCameraConfig", f"{__name__}.usb", "USBCamera")

Camera.register("FlirCamera", f"{__name__}.flir")
Camera.register("GrasshopperFlirCamera", f"{__name__}.flir")

Camera.register("PklCamera", f"{__name__}.pkl")

Camera.register("RealsenseCamera", f"{__name__}.realsense")
CameraConfig.register("RealsenseConfig", f"{__name__}.realsense")
CameraConfig.register("RealsenseConfig", f"{__name__}.realsense", "RealsenseCamera")

__all__ = [
    "Camera",
    "CameraConfig",
]
