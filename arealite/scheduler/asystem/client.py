from arealite.api.scheduler_api import (
    SchedulerClient,
)

class AsystemClient(SchedulerClient):
    """
    SchedulerClient implementation for interacting with an Asystem server.
    """

    def __init__(
            self,
            expr_name: str,
            trial_name: str,
            endpoint: str,
    ):
        """
        Initializes the AsystemClient.

        Args:
            expr_name: Name of the experiment.
            trial_name: Name of the trial.
            asystem_server_url: Base URL of the Asystem server (e.g., "http://localhost:8081").
        """
        super().__init__(expr_name, trial_name)

        self.endpoint = endpoint
