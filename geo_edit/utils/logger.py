import logging

import colorlog

_LOGGER_CONFIGURED = False


def setup_logger(name: str | None = None, level: int = logging.INFO) -> logging.Logger:
    global _LOGGER_CONFIGURED
    if not _LOGGER_CONFIGURED:
        handler = colorlog.StreamHandler()
        handler.setFormatter(
            colorlog.ColoredFormatter(
                "%(log_color)s%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%H:%M:%S",
                log_colors={
                    "DEBUG": "cyan",
                    "INFO": "white",
                    "WARNING": "yellow",
                    "ERROR": "red",
                    "CRITICAL": "bold_red",
                },
            )
        )

        root = logging.getLogger()
        root.handlers.clear()
        root.addHandler(handler)
        root.setLevel(level)
        _LOGGER_CONFIGURED = True
    else:
        logging.getLogger().setLevel(level)

    return logging.getLogger(name) if name else logging.getLogger()
