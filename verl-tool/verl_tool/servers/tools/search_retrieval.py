"""
Search Retrieval Tool for verl-tool - Compatible with Search-R1 functionality
"""
from .base import BaseTool, register_tool
import regex as re
import requests
from typing import Tuple, Dict, Any, List
import logging

logger = logging.getLogger(__name__)

@register_tool
class SearchRetrievalTool(BaseTool):
    tool_type = "search_retrieval"
    
    def __init__(self, num_workers=1, retriever_url="http://127.0.0.1:8000/retrieve", topk=3, **kwargs):
        super().__init__(num_workers)
        # Allow configuration from environment or kwargs
        import os
        self.retriever_url = kwargs.get('retriever_url', os.getenv('RETRIEVER_URL', retriever_url))
        self.topk = kwargs.get('topk', int(os.getenv('RETRIEVER_TOPK', str(topk))))
        logger.info(f"SearchRetrievalTool initialized with URL: {self.retriever_url}, topk: {self.topk}")
    
    def get_usage_inst(self):
        return "You can search for information by putting your query between <search> and </search> tags."
    
    def _parse_search_query(self, action: str) -> str:
        """
        Extract the search query from the action string.
        This is a helper function to parse the <search> tags.
        
        Args:
            action: Raw action string containing search query
            
        Returns:
            Extracted search query
        """
        # Priority logic moved from serve.py: prioritize search tool for <search> tags
        # This implements the original logic: if "</search>" in action and "search_retrieval" in self.tools
        if "</search>" in action:
            # Extract search query from <search>query</search> tags
            search_matches = re.findall(r"<search>(.*?)</search>", action, re.DOTALL)
            
            if len(search_matches) > 0:
                # Use the last search query if multiple are found
                query = search_matches[-1].strip()
                return query, True
        return "", False
    
    def _parse_answer_tags(self, action: str) -> Tuple[str, bool]:
        """
        Parse the action string to extract answer tags.
        This is a helper function to handle <answer> tags.
        
        Args:
            action: Raw action string containing answer tags
            
        Returns:
            Tuple containing the extracted answer and a validity flag
        """
        # Priority logic moved from serve.py: check for finish condition (</answer> tag)
        # This implements the original logic: if "</answer>" in action
        if "</answer>" in action:
            # Check for <answer> tags (Search-R1 style)
            answer_matches = re.findall(r"<answer>(.*?)</answer>", action, re.DOTALL)
            if len(answer_matches) > 0:
                final_answer = answer_matches[-1].strip()
                return final_answer, True
        return "", False
    
    def parse_action(self, action: str) -> Tuple[str, bool]:
        """
        Parse the raw action string to extract search queries.
        Implements the prioritization logic that was originally in serve.py lines 112-115.
        
        Args:
            action: Raw action string containing search query
            
        Returns:
            Tuple containing the extracted query and a validity flag
        """
        # Check for <search> tags first
        search_query, is_valid = self._parse_search_query(action)
        if is_valid:
            return search_query, True
        
        # If no search query found, check for <answer> tags
        answer, is_valid = self._parse_answer_tags(action)
        if is_valid:
            return answer, True
        
        # Default case - no valid action found
        return "", False
    
    def get_action_priority(self, action: str, extra_field: dict) -> int:
        """
        Get priority for handling this action. SearchRetrieval has high priority for <search> tags.
        This moves the tool identification logic from serve.py to the tool itself.
        
        Args:
            action: The raw action string
            extra_field: Extra fields associated with the action
        Returns:
            priority: Integer priority (-1 means cannot handle, higher numbers = higher priority)
        """
        # High priority for actions with </search> tags (original logic from serve.py line 112-115)
        if "</search>" in action:
            _, valid = self.parse_action(action)
            if valid:
                return 100  # High priority for search actions
        
        # Standard priority check
        _, valid = self.parse_action(action)
        return 0 if valid else -1
    
    def conduct_action(self, trajectory_id, action, extra_field):
        """
        Execute search query via retrieval service.
        
        Args:
            trajectory_id: ID for tracking the action
            action: Raw action string containing search query
            extra_field: Additional parameters
            
        Returns:
            Tuple containing observation, done flag, and validity flag
        """
        parsed_query, is_valid = self._parse_search_query(action)
        env = self.load_env(trajectory_id)
        
        if not is_valid:
            # try answer tags if no valid search query found
            parsed_query, is_valid = self._parse_answer_tags(action)
            if is_valid:
                observation = ""
                execution_result = ""
                done = True
                valid = False
            else:
                observation = ""
                execution_result = ""
                done = False
                valid = False
        else:
            try:
                # Call the retrieval service (same as Search-R1)
                search_results = self._batch_search([parsed_query])
                formatted_results = self._passages2string(search_results[0])
                
                # Format observation similar to Search-R1
                observation = f'\n\n<information>{formatted_results.strip()}</information>\n\n'
                execution_result = formatted_results
                done = False  # Search doesn't end the trajectory
                valid = True
                
            except Exception as e:
                logger.error(f"Search error for trajectory {trajectory_id}: {e}")
                execution_result = f"Search error: {str(e)}"
                observation = f'\n\n<information>Search temporarily unavailable</information>\n\n'
                done = False
                valid = False
        
        self.update_env(trajectory_id, env, parsed_query, is_valid, extra_field, execution_result)
        self.save_env(trajectory_id, env)
        
        return observation, done, valid
    
    def _batch_search(self, queries: List[str]) -> List[List[Dict]]:
        """
        Call the retrieval service with batch queries.
        Compatible with Search-R1's retrieval API.
        """
        payload = {
            "queries": queries,
            "topk": self.topk,
            "return_scores": True
        }
        
        try:
            response = requests.post(self.retriever_url, json=payload, timeout=30)
            response.raise_for_status()
            result = response.json()
            return result['result']
        except Exception as e:
            logger.error(f"Retrieval service error: {e}")
            # Return empty results on error
            return [[] for _ in queries]
    
    def _passages2string(self, retrieval_result: List[Dict]) -> str:
        """
        Format retrieval results into a readable string.
        Same format as Search-R1.
        """
        format_reference = ''
        for idx, doc_item in enumerate(retrieval_result):
            if 'document' in doc_item:
                content = doc_item['document']['contents']
            else:
                content = doc_item.get('contents', '')
            
            title = content.split("\n")[0] if content else "No title"
            text = "\n".join(content.split("\n")[1:]) if content else "No content"
            format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"

        return format_reference 