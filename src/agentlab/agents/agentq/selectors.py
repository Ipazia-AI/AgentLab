import logging
from abc import ABC, abstractmethod
from typing import Any, List, Optional

logger = logging.getLogger(__name__)

class ActionSelector(ABC):
    @abstractmethod
    def select(self, root_node: Any) -> Optional[Any]:
        pass

class MaxVisitSelector(ActionSelector):
    def select(self, root_node):
        if not root_node.children:
            return None
        return max(root_node.children, key=lambda c: c.visits)

class AheadKSelector(ActionSelector):
    """
    Selects an action only if it is ahead of the second best by K.
    If no such action exists, it indicates that more sampling/search is needed.
    """
    def __init__(self, k: int = 1):
        self.k = k

    def select(self, root_node):
        if not root_node.children:
            return None
        
        # Sort children by visits (or value)
        sorted_children = sorted(root_node.children, key=lambda c: c.visits, reverse=True)
        
        if len(sorted_children) == 1:
            return sorted_children[0]
            
        best = sorted_children[0]
        second_best = sorted_children[1]
        
        if best.visits - second_best.visits >= self.k:
            logger.info(f"AheadKSelector: Winner found! {best.action} is {best.visits - second_best.visits} ahead of {second_best.action}")
            return best
        
        logger.info(f"AheadKSelector: No clear winner. Gap is {best.visits - second_best.visits}, need {self.k}.")
        return None # Signal that we need more search/sampling
