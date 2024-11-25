"""Main application module.

Should import this `APP` and use it as a decorator to add commands.

Example:

.. code-block:: python

    from cc_hardware.tools.app import APP

    @APP.command()
    def my_command():
        ...
"""

import typer

APP = typer.Typer()
