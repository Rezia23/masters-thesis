"""Background-set selection strategy."""


class BackgroundMethod:
    """Selection method (``"all"``, ``"kmeans"``, ``"kernel_density"``, ``"random"``) and optional integer parameter."""

    def __init__(self, method, param=None):
        self.method = method
        self.param = param