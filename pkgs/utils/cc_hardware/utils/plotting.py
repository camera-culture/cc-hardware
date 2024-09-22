from cc_hardware.utils.logger import get_logger


def set_matplotlib_style(*, use_scienceplots: bool = True):
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme("paper", font_scale=1.5)
    sns.set_style("ticks")

    if use_scienceplots:
        try:
            import scienceplots  # noqa

            plt.style.use(["science", "nature"])
        except ImportError:
            get_logger().warning(
                "SciencePlots not found. Using default matplotlib style."
            )
