"""
Improved Ray Tool Manager - Cleaner, more robust distributed tool execution
"""
import ray
import asyncio
import logging
import time
import os
from typing import Dict, List, Optional, Tuple, Any, Union
from collections import defaultdict

from .tools import get_tool_cls, ALL_TOOLS, set_use_tqdm

logger = logging.getLogger(__name__)


# === RAY TOOL MANAGER ===
class RayToolManager:
    """Distributed tool manager using Ray for high-performance processing"""
    
    def __init__(
        self, 
        tool_types: Tuple[str], 
        config, 
        use_tqdm: bool = False, 
        done_if_invalid: bool = False
    ):
        """
        Initialize Ray-based tool manager.
        
        Args:
            tool_types: Types of tools to initialize
            config: Server configuration object
            use_tqdm: Whether to use progress bars
            done_if_invalid: Whether to mark invalid actions as done
        """
        self.tool_types = tool_types
        self.config = config
        self.use_tqdm = use_tqdm
        self.done_if_invalid = done_if_invalid
        self.tools: Dict[str, Any] = {}
        
        # Initialize Ray if needed
        self._ensure_ray_initialized()
        
        # Configure tqdm
        set_use_tqdm(use_tqdm)
        
        # Initialize tools
        self._initialize_tools()
        
        logger.info(f"Ray Tool Manager initialized with {len(self.tools)} tools")
    
    def _ensure_ray_initialized(self):
        """Ensure Ray is properly initialized"""
        if not ray.is_initialized():
            try:
                # Try to connect to existing cluster first
                ray.init(address="auto", ignore_reinit_error=True)
                logger.info("Connected to existing Ray cluster")
            except:
                # Fallback to local Ray
                ray.init(ignore_reinit_error=True)
                logger.info("Started local Ray cluster")
        else:
            logger.info("Ray already initialized")
    
    def _initialize_tools(self):
        """Initialize tools with proper error handling and dependency management"""
        # Ensure finish tool is processed last for dependencies
        ordered_tools = [t for t in self.tool_types if t != "finish"]
        
        initialized_tools = []
        failed_tools = []
        
        logger.info(f"Initializing Ray tools: {ordered_tools}")
        if self.config.workers_per_tool <= os.cpu_count() * 2:
            logger.warning(f"You are using ray with workers_per_tool={self.config.workers_per_tool} which is less than or equal to the 2 times of CPU cores  ({os.cpu_count() * 4}). This may lead to suboptimal performance due to resource contention. Consider increasing workers_per_tool if you see low gpu utilization during rollouts.")
        
        for tool_type in ordered_tools:
            try:
                tool_cls = get_tool_cls(tool_type)

                tool_instance = ray.remote(max_concurrency=self.config.workers_per_tool)(tool_cls).remote()
                self.tools[tool_type] = tool_instance
                initialized_tools.append(tool_type)
                logger.info(f"✓ Initialized Ray tool: {tool_type}")
                
            except Exception as e:
                failed_tools.append((tool_type, str(e)))
                logger.error(f"✗ Failed to initialize Ray tool {tool_type}: {e}")
                
        if "finish" not in self.tools:
            tool_instance = ray.remote(max_concurrency=self.config.workers_per_tool)(get_tool_cls("finish")).remote(
                num_workers=1,
                other_tools={t: self.tools[t] for t in initialized_tools if t in self.tools}
            )
            self.tools["finish"] = tool_instance
        
        self._log_tool_status()
        
        if failed_tools:
            logger.warning(f"Some Ray tools failed to initialize: {failed_tools}")
    
    def _log_tool_status(self):
        """Log comprehensive tool status"""
        logger.info("Ray Tool Status Summary:")
        for tool in ALL_TOOLS:
            if tool in self.tools:
                logger.info(f"  {tool}: 🟢 ACTIVE (Ray)")
            else:
                logger.info(f"  {tool}: ⚪ INACTIVE")
    
    def get_usage_instructions(self) -> str:
        """Generate usage instructions for available tools"""
        instructions = []
        for tool_type, tool in self.tools.items():
            if tool_type not in ["finish", "base"] and hasattr(tool, 'get_usage_inst'):
                try:
                    usage_inst = ray.get(tool.get_usage_inst.remote())
                    instructions.append(f"• {tool_type}: {usage_inst}")
                except Exception as e:
                    logger.warning(f"Could not get usage instructions for {tool_type}: {e}")
        
        if not instructions:
            return "No tools available with usage instructions."
            
        return "\n".join([
            "Available Ray tools:",
            *instructions
        ])
    
    def _identify_tool_for_action(self, action: str, extra_field: Dict[str, Any]) -> Optional[str]:
        """
        Identify which tool should process a given action.
        
        Args:
            action: The action string
            extra_field: Additional data for the action
            
        Returns:
            Tool type name or None if no tool matches
        """
        # Check for explicit finish signal
        if extra_field.get("finish", False):
            return "finish"
            
        # Single tool optimization
        if len(self.tools) == 1:
            return list(self.tools.keys())[0]
        
        # Try each tool (except special ones) to parse action
        standard_tools = [t for t in self.tools.keys() if t not in ["finish", "mcp_interface"]]
        
        for tool_type in standard_tools:
            try:
                tool = self.tools[tool_type]
                if hasattr(tool, 'parse_action'):
                    _, valid = ray.get(tool.parse_action.remote(action))
                    if valid:
                        return tool_type
            except Exception as e:
                logger.debug(f"Tool {tool_type} parse error: {e}")
                continue
        
        # Try MCP interface as fallback
        if "mcp_interface" in self.tools:
            try:
                _, valid = ray.get(self.tools["mcp_interface"].parse_action.remote(action))
                if valid:
                    return "mcp_interface"
            except Exception as e:
                logger.debug(f"MCP interface parse error: {e}")

        return None
    
    async def _identify_tool_types_batch(self, actions: List[str], extra_fields: List[Dict[str, Any]]) -> List[Optional[str]]:
        """
        Efficiently identify tool types for a batch of actions.
        
        Args:
            actions: List of action strings
            extra_fields: List of extra field dictionaries
            
        Returns:
            List of tool type names (or None for unmatched actions)
        """
        tool_types = []
        
        # Process in chunks to balance performance and responsiveness
        chunk_size = min(200, max(50, len(actions) // 4))
        
        for i in range(0, len(actions), chunk_size):
            chunk_end = min(i + chunk_size, len(actions))
            chunk_actions = actions[i:chunk_end]
            chunk_extra_fields = extra_fields[i:chunk_end]
            
            # Process chunk synchronously (tool identification is fast)
            chunk_tool_types = [
                self._identify_tool_for_action(action, extra_field)
                for action, extra_field in zip(chunk_actions, chunk_extra_fields)
            ]
            tool_types.extend(chunk_tool_types)
            
            # Yield control for large batches
            if len(actions) > 1000 and i % (chunk_size * 10) == 0:
                await asyncio.sleep(0.001)
        
        return tool_types
    
    def _group_actions_by_tool(
        self,
        tool_types: List[Optional[str]],
        trajectory_ids: List[str],
        actions: List[str],
        extra_fields: List[Dict[str, Any]]
    ) -> Dict[Optional[str], Tuple[List[int], List[str], List[str], List[Dict[str, Any]]]]:
        """
        Group actions by their assigned tool types for efficient batch processing.
        
        Returns:
            Dict mapping tool_type -> (indices, trajectory_ids, actions, extra_fields)
        """
        groups = defaultdict(lambda: ([], [], [], []))
        
        for i, tool_type in enumerate(tool_types):
            indices, traj_ids, acts, extras = groups[tool_type]
            indices.append(i)
            traj_ids.append(trajectory_ids[i])
            acts.append(actions[i])
            extras.append(extra_fields[i])
        
        return dict(groups)
    
    async def _process_tool_group_batch(
        self,
        tool_type: str,
        trajectory_ids: List[str],
        actions: List[str],
        extra_fields: List[Dict[str, Any]]
    ) -> Tuple[List[Any], List[bool], List[bool]]:
        """
        Process a group of actions for the same tool type using Ray batch execution.
        
        Args:
            tool_type: Type of tool to use
            trajectory_ids: List of trajectory IDs for this group
            actions: List of actions for this group
            extra_fields: List of extra fields for this group
            
        Returns:
            tuple: (observations, dones, valids) for this group
        """
        tool = self.tools[tool_type]
        
        # Use batch processing if available, otherwise individual Ray tasks
        if hasattr(tool, 'get_observations'):
            # Batch processing
            future = tool.get_observations.remote(trajectory_ids, actions, extra_fields)
            return await self._ray_get_async(future)
        else:
            # Individual processing with Ray parallelization
            futures = [
                tool.conduct_action.remote(tid, action, extra)
                for tid, action, extra in zip(trajectory_ids, actions, extra_fields)
            ]
            
            results = await self._ray_get_async(futures)
            
            # Unpack results
            observations, dones, valids = zip(*results) if results else ([], [], [])
            return list(observations), list(dones), list(valids)
    
    async def _ray_get_async(self, ray_futures):
        """
        Asynchronously wait for Ray futures to complete.
        
        Args:
            ray_futures: Single future or list of Ray futures
            
        Returns:
            Results from Ray futures
        """
        if isinstance(ray_futures, list):
            # Wait for all futures with proper async handling
            return await asyncio.get_event_loop().run_in_executor(
                None, lambda: ray.get(ray_futures)
            )
        else:
            # Single future
            return await asyncio.get_event_loop().run_in_executor(
                None, lambda: ray.get(ray_futures)
            )
    
    async def _handle_invalid_actions(
        self,
        trajectory_ids: List[str],
        actions: List[str],
        extra_fields: List[Dict[str, Any]]
    ) -> Tuple[List[Any], List[bool], List[bool]]:
        """
        Handle actions that couldn't be matched to any tool using Ray.
        
        Args:
            trajectory_ids: List of trajectory IDs for invalid actions
            actions: List of invalid actions
            extra_fields: List of extra fields for invalid actions
            
        Returns:
            tuple: (observations, dones, valids) for invalid actions
        """
        futures = [
            handle_invalid_action.remote(tid, action, extra, self.done_if_invalid)
            for tid, action, extra in zip(trajectory_ids, actions, extra_fields)
        ]
        
        results = await self._ray_get_async(futures)
        
        if results:
            observations, dones, valids = zip(*results)
            return list(observations), list(dones), list(valids)
        else:
            return [], [], []
    
    async def process_actions(
        self, 
        trajectory_ids: List[str], 
        actions: List[str], 
        extra_fields: List[Dict[str, Any]]
    ) -> Tuple[List[Union[str, dict]], List[bool], List[bool]]:
        """
        Process actions using Ray workers with optimized batch processing.
        
        Args:
            trajectory_ids: List of trajectory IDs
            actions: List of actions corresponding to each trajectory
            extra_fields: List of extra data for each action
            
        Returns:
            tuple: (observations, dones, valids) lists for all actions
        """
        start_time = time.time()
        num_actions = len(actions)
        
        logger.debug(f"Processing {num_actions} actions with Ray")
        
        # Identify tool types for all actions
        tool_types = await self._identify_tool_types_batch(actions, extra_fields)
        
        # Group actions by tool type for efficient batch processing
        tool_groups = self._group_actions_by_tool(tool_types, trajectory_ids, actions, extra_fields)
        
        # Initialize result containers
        observations = [None] * num_actions
        dones = [False] * num_actions
        valids = [False] * num_actions
        
        # Process each tool group concurrently
        processing_tasks = []
        
        for tool_type, (indices, group_traj_ids, group_actions, group_extras) in tool_groups.items():
            if tool_type is None:
                # Handle invalid actions
                task = self._handle_invalid_actions(group_traj_ids, group_actions, group_extras)
            else:
                # Process valid actions with appropriate tool
                task = self._process_tool_group_batch(tool_type, group_traj_ids, group_actions, group_extras)
            
            processing_tasks.append((tool_type, indices, task))
        
        # Execute all processing tasks concurrently
        await self._collect_results(processing_tasks, observations, dones, valids)
        
        # Validate all actions were processed
        none_count = observations.count(None)
        if none_count > 0:
            logger.error(f"{none_count} actions did not return observations")
            raise RuntimeError(f"Failed to process {none_count} actions")
        
        processing_time = (time.time() - start_time) * 1000
        logger.debug(f"Ray processed {num_actions} actions in {processing_time:.1f}ms")
        
        return observations, dones, valids
    
    async def _collect_results(
        self,
        processing_tasks: List[Tuple[Optional[str], List[int], Any]],
        observations: List[Any],
        dones: List[bool],
        valids: List[bool]
    ):
        """
        Collect results from all processing tasks and assign to correct positions.
        
        Args:
            processing_tasks: List of (tool_type, indices, task) tuples
            observations: Result list to populate
            dones: Result list to populate  
            valids: Result list to populate
        """
        for tool_type, indices, task in processing_tasks:
            try:
                # Await task results
                task_observations, task_dones, task_valids = await task
                
                # Validate result lengths
                if len(task_observations) != len(indices):
                    raise ValueError(f"Tool {tool_type} returned {len(task_observations)} results for {len(indices)} actions")
                
                # Assign results to correct positions
                for idx_pos, result_idx in enumerate(indices):
                    observations[result_idx] = task_observations[idx_pos]
                    dones[result_idx] = task_dones[idx_pos]
                    valids[result_idx] = task_valids[idx_pos]
                    
                logger.debug(f"✓ Tool {tool_type} processed {len(indices)} actions successfully")
                
            except Exception as e:
                logger.error(f"✗ Tool {tool_type} processing failed: {e}", exc_info=True)
                
                # Create error response for failed processing
                error_response = {
                    "obs": "",
                    "error": f"Ray tool processing failed: {str(e)}",
                    "tool_type": tool_type
                }
                
                # Assign error to all actions that were supposed to be processed by this tool
                for result_idx in indices:
                    observations[result_idx] = error_response
                    dones[result_idx] = True
                    valids[result_idx] = False
    
    def get_tool_stats(self) -> Dict[str, Any]:
        """Get statistics about Ray tools and cluster"""
        try:
            cluster_resources = ray.cluster_resources()
            node_stats = ray.nodes()
            
            return {
                "tools_count": len(self.tools),
                "tools_active": list(self.tools.keys()),
                "ray_cluster_resources": cluster_resources,
                "ray_nodes": len(node_stats),
                "ray_initialized": ray.is_initialized()
            }
        except Exception as e:
            logger.warning(f"Could not get Ray stats: {e}")
            return {
                "tools_count": len(self.tools),
                "tools_active": list(self.tools.keys()),
                "ray_error": str(e)
            }
    
    def cleanup(self):
        """Clean up Ray resources"""
        try:
            # Let Ray handle cleanup automatically
            # Don't call ray.shutdown() as it might affect other users
            logger.info("Ray Tool Manager cleanup completed")
        except Exception as e:
            logger.warning(f"Ray cleanup warning: {e}")


# === RAY REMOTE FUNCTIONS (UPDATED) ===
@ray.remote(num_cpus=0)
def handle_invalid_action(trajectory_id: str, action: str, extra_field: Dict[str, Any], done_if_invalid: bool):
    """Handle actions that don't match any tool with better error info"""
    observation = {
        "obs": "",
        "invalid_reason": "No valid tool found for action",
        "action_preview": action[:100] + "..." if len(action) > 100 else action,
        "trajectory_id": trajectory_id
    }
    return observation, done_if_invalid, False


# === PERFORMANCE MONITORING ===
class RayPerformanceMonitor:
    """Monitor Ray performance and provide insights"""
    
    def __init__(self):
        self.request_times = []
        self.batch_sizes = []
        self.start_time = time.time()
    
    def record_request(self, processing_time: float, batch_size: int):
        """Record performance metrics for a request"""
        self.request_times.append(processing_time)
        self.batch_sizes.append(batch_size)
        
        # Keep only recent data (last 1000 requests)
        if len(self.request_times) > 1000:
            self.request_times = self.request_times[-1000:]
            self.batch_sizes = self.batch_sizes[-1000:]
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary statistics"""
        if not self.request_times:
            return {"status": "no_data"}
        
        import statistics
        
        return {
            "requests_processed": len(self.request_times),
            "avg_processing_time_ms": statistics.mean(self.request_times),
            "median_processing_time_ms": statistics.median(self.request_times),
            "avg_batch_size": statistics.mean(self.batch_sizes),
            "uptime_seconds": time.time() - self.start_time,
            "requests_per_second": len(self.request_times) / max(1, time.time() - self.start_time)
        }


# === INTEGRATION HELPERS ===
def create_ray_tool_manager(tool_types: Tuple[str], config, **kwargs) -> RayToolManager:
    """Factory function to create Ray tool manager with proper validation"""
    
    # Validate Ray is available
    try:
        import ray
    except ImportError:
        raise RuntimeError("Ray is not installed. Install with: pip install ray")
    
    # Create and return manager
    return RayToolManager(tool_types, config, **kwargs)


# === USAGE EXAMPLE ===
def test_ray_performance():
    """Simple performance test for Ray tool manager"""
    import time
    
    # This would be called from your main server
    tool_types = ("base",)  # Example
    
    class MockConfig:
        workers_per_tool = 4
    
    manager = RayToolManager(tool_types, MockConfig())
    
    # Test single action
    start = time.time()
    result = asyncio.run(manager.process_actions(
        ["test_1"], 
        ["test action"], 
        [{}]
    ))
    single_time = time.time() - start
    
    # Test batch
    start = time.time()
    result = asyncio.run(manager.process_actions(
        [f"test_{i}" for i in range(100)],
        ["test action"] * 100,
        [{}] * 100
    ))
    batch_time = time.time() - start
    
    print(f"Single action: {single_time*1000:.1f}ms")
    print(f"100 actions: {batch_time*1000:.1f}ms ({batch_time/100*1000:.1f}ms per action)")
    
    manager.cleanup()


if __name__ == "__main__":
    test_ray_performance()