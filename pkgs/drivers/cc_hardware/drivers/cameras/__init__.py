"""Camera drivers for the cc-hardware package."""

from cc_hardware.drivers.cameras.camera import Camera

# Register the camera implementations
Camera.register("FlirCamera", f"{__name__}.flir")
Camera.register("GrasshopperFlirCamera", f"{__name__}.flir")
Camera.register("PklCamera", f"{__name__}.pkl")
Camera.register("RealsenseCamera", f"{__name__}.realsense")

__all__ = [
    "Camera",
]
