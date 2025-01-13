# Stepper Gantry Collab

This demo shows how you can control a stepper gantry along with a SPAD camera. There are two examples which show two separate APIs. The first example (`v1`) uses {mod}`argparse` and demos a more explicit code instantiation. The second example (`v2`) uses the {func}`~cc_hardware.utils.register_cli` decorator and shows how to automatically generate a CLI/API for the code.
