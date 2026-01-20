from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, List

class AbstractVLMTask(ABC):
    """VLM task base class"""
    
    def __init__(self, task_id: str):
        self.task_id = task_id
    
    @abstractmethod
    def validate(
        self,
        chat_history: List[Dict],
        last_observation: Any,
        full_history: List[Any]
    ) -> Tuple[float, bool, Dict[str, Any]]:
        """verify the task excution
        
        Args:
            chat_history: chat history
            last_observation: last observation
            full_history: full history
            
        Returns:
            reward: reward value
            done: whether finished
            info: additional information
        """
        pass
    
    @abstractmethod
    def get_info(self) -> Dict[str, Any]:
        """get current information"""
        pass
    
