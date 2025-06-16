import hydra

from arealite.api.cli_args import LLMServiceConfig
from arealite.api.llm_server_api import LLMServerFactory


@hydra.main(version_base=None, config_path="config", config_name="llm_server")
def main(args: LLMServiceConfig):
    """Main entry point for launching the LLM server."""
    server = LLMServerFactory.make_server(args)
    server.start()


if __name__ == "__main__":
    main()
