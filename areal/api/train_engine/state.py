from __future__ import annotations

import abc
from typing import TYPE_CHECKING

from areal.api.io_struct import SaveLoadMeta, WeightUpdateMeta

if TYPE_CHECKING:
    from areal.api.engine_api import InferenceEngine


class TrainEngineStateMixin(abc.ABC):
    @abc.abstractmethod
    def connect_engine(self, engine: InferenceEngine, meta: WeightUpdateMeta):
        """Connect to an inference engine for online training.

        Parameters
        ----------
        engine : InferenceEngine
            The inference engine to connect
        meta : WeightUpdateMeta
            Metadata containing information about the weight update
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def update_weights(self, meta: WeightUpdateMeta):
        """Update weights to the inference engine in a blocking manner.

        Parameters
        ----------
        meta : WeightUpdateMeta
            Metadata containing information about the weight update
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def set_version(self, version: int):
        """Set the current weight version in the training engine.

        Parameters
        ----------
        version : int
            The weight version number to set
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def get_version(self) -> int:
        """Get the current weight version in the training engine.

        Returns
        -------
        int
            The current weight version number
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def save(self, meta: SaveLoadMeta):
        """Save model weights and optimizer states for later use.

        Parameters
        ----------
        meta : SaveLoadMeta
            Metadata containing information about where and how to save
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def load(self, meta: SaveLoadMeta):
        """Load model weights and optimizer states from a file.

        Parameters
        ----------
        meta : SaveLoadMeta
            Metadata containing information about where and how to load
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def onload(self) -> None:
        """Load model parameters from CPU back to GPU.

        This method resumes model computation on GPU after a previous offload.
        Should be called before any forward/backward operations when the model
        has been offloaded.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def offload(self) -> None:
        """Offload model parameters from GPU to CPU to free GPU memory.

        This method is useful for memory-constrained scenarios where the model
        needs to be temporarily moved to CPU while other operations use the GPU.
        """
        raise NotImplementedError()
