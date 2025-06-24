# Copyright 2025 Ant Group Inc.
# Copyright 2024 Wei Fu & Zhiyu Mei
# Licensed under the Apache License, Version 2.0 (the "License").

import logging.config
import os
from logging import WARNING, Logger, Manager, RootLogger
from typing import Literal, Optional

import colorlog

LOG_FORMAT = "%(asctime)s.%(msecs)03d %(name)s %(levelname)s: %(message)s"
DATE_FORMAT = "%Y%m%d-%H:%M:%S"
LOGLEVEL = logging.INFO

# NOTE: To use colorlog we should not call colorama.init() anywhere.
# The available color names are black, red, green, yellow, blue, purple, cyan and white
log_config = {
    "version": 1,
    "formatters": {
        "plain": {
            "()": colorlog.ColoredFormatter,
            "format": "%(log_color)s" + LOG_FORMAT,
            "datefmt": DATE_FORMAT,
            "log_colors": {
                "DEBUG": "white",
                "INFO": "white",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "bold_white,bg_red",
            },
        },
        "colored": {
            "()": colorlog.ColoredFormatter,
            "format": "%(log_color)s" + LOG_FORMAT,
            "datefmt": DATE_FORMAT,
            "log_colors": {
                "DEBUG": "blue",
                "INFO": "light_purple",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "bold_white,bg_red",
            },
        },
        "colored_system": {
            "()": colorlog.ColoredFormatter,
            "format": "%(log_color)s" + LOG_FORMAT,
            "datefmt": DATE_FORMAT,
            "log_colors": {
                "DEBUG": "blue",
                "INFO": "light_green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "bold_white,bg_red",
            },
        },
        "colored_benchmark": {
            "()": colorlog.ColoredFormatter,
            "format": "%(log_color)s" + LOG_FORMAT,
            "datefmt": DATE_FORMAT,
            "log_colors": {
                "DEBUG": "light_black",
                "INFO": "light_cyan",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "bold_white,bg_red",
            },
        },
    },
    "handlers": {
        "plainHandler": {
            "class": "logging.StreamHandler",
            "level": LOGLEVEL,
            "formatter": "plain",
            "stream": "ext://sys.stdout",
        },
        "benchmarkHandler": {
            "class": "logging.StreamHandler",
            "level": "DEBUG",
            "formatter": "colored_benchmark",
            "stream": "ext://sys.stdout",
        },
        "systemHandler": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "colored_system",
            "stream": "ext://sys.stdout",
        },
        "coloredHandler": {
            "class": "logging.StreamHandler",
            "level": LOGLEVEL,
            "formatter": "colored",
            "stream": "ext://sys.stdout",
        },
    },
    "loggers": {
        "plain": {
            "handlers": ["plainHandler"],
            "level": LOGLEVEL,
        },
        "benchmark": {
            "handlers": ["benchmarkHandler"],
            "level": "DEBUG",
        },
        "colored": {
            "handlers": ["coloredHandler"],
            "level": LOGLEVEL,
        },
        "system": {
            "handlers": ["systemHandler"],
            "level": LOGLEVEL,
        },
    },
    "disable_existing_loggers": True,
}


def getLogger(
    name: Optional[str] = None,
    type_: Optional[Literal["plain", "benchmark", "colored", "system"]] = None,
):
    # Fix the logging config automatically set by transformer_engine
    # by reset config everytime getLogger is called.
    root = RootLogger(WARNING)
    Logger.root = root
    Logger.manager = Manager(Logger.root)

    logging.config.dictConfig(log_config)

    if name is None:
        name = "plain"
    if type_ is None:
        type_ = "plain"
    assert type_ in ["plain", "benchmark", "colored", "system"]
    if name not in log_config["loggers"]:
        log_config["loggers"][name] = {
            "handlers": [f"{type_}Handler"],
            "level": LOGLEVEL,
        }
        logging.config.dictConfig(log_config)
    return logging.getLogger(name)


_LATEST_LOG_STEP = 0


def log_swanlab_wandb_tensorboard(data, step=None, summary_writer=None):
    # Logs data to SwanLab、 wandb、 TensorBoard.

    global _LATEST_LOG_STEP
    if step is None:
        step = _LATEST_LOG_STEP
    else:
        _LATEST_LOG_STEP = max(_LATEST_LOG_STEP, step)

    # swanlab
    import swanlab

    swanlab.log(data, step=step)

    # wandb
    import wandb

    wandb.log(data, step=step)

    # tensorboard
    if summary_writer is not None:
        for key, val in data.items():
            summary_writer.add_scalar(f"{key}", val, step)


if __name__ == "__main__":
    # The following serves as a color visualization test.
    # The available color names are black, red, green, yellow, blue, purple, cyan and white
    log_config = {
        "version": 1,
        "formatters": {
            "colored": {
                "()": colorlog.ColoredFormatter,
                "format": "%(log_color)s" + LOG_FORMAT,
                "datefmt": DATE_FORMAT,
                "log_colors": {
                    "DEBUG": "purple",
                    "INFO": "light_purple",
                    "WARNING": "yellow",
                    "ERROR": "red",
                    "CRITICAL": "bold_white,bg_red",
                },
            },
        },
        "handlers": {
            "coloredHandler": {
                "class": "logging.StreamHandler",
                "level": "DEBUG",
                "formatter": "colored",
                "stream": "ext://sys.stdout",
            },
        },
        "loggers": {
            "": {
                "handlers": ["coloredHandler"],
                "level": "DEBUG",
            },
        },
    }
    logging.config.dictConfig(log_config)
    logging.debug("This is a debug message")
    logging.info("This is an info message")
    logging.warning("This is a warning message")
    logging.error("This is an error message")
    logging.critical("This is a critical message")
