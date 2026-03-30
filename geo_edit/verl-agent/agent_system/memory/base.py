# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import ABC, abstractmethod
from typing import List, Dict, Any


class BaseMemory(ABC):
    """
    Base class for memory management. Defines the interface for memory modules.
    """

    @abstractmethod
    def __len__(self):
        """Return the number of memory slots."""
        pass

    @abstractmethod
    def __getitem__(self, idx: int):
        """Access memory of specific environment index."""
        pass

    @abstractmethod
    def reset(self, batch_size: int):
        """
        Reset memory with given batch size.
        """
        pass

    @abstractmethod
    def store(self, record: Dict[str, List[Any]]):
        """
        Stores a new batch of records into memory.
        """
        pass

    @abstractmethod
    def fetch(self, step: int):
        """
        Fetches memory records at a specific time step across all environments.
        """
        pass