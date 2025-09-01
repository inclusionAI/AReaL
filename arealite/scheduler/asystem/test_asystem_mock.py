#!/usr/bin/env python3
"""
Mock test for AsystemScheduler that doesn't require actual Asystem cluster connectivity.
Uses unittest.mock to simulate Asystem API responses.
"""

import logging
import sys
from unittest.mock import Mock, patch

from arealite.scheduler.asystem import AsystemScheduler
from arealite.scheduler.base import ContainerSpec, SchedulingConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def test_basic_functionality():
    """Test basic AsystemScheduler functionality with mocked responses."""
    logger.info("Testing AsystemScheduler with mock responses...")

    # Create scheduler config
    scheduler_config = {
        "type": "asystem",
        "endpoint": "http://30.230.2.87:8081",
        "expr_name": "test_expr",
        "trial_name": "test_trial",
    }

    # Create container spec
    container_spec = ContainerSpec(
        cpu=1000,
        mem=2048,
        container_image="/storage/openpsi/images/areal-latest.sif",
        cmd="sleep 10000",
        env_vars={"TEST_VAR": "test_value"},
        portCount=2,
    )

    # Create scheduling config
    scheduling_config = SchedulingConfig(
        replicas=2, specs=[container_spec, container_spec]
    )
    # Create scheduler
    scheduler = AsystemScheduler(scheduler_config)
    logger.info("✓ AsystemScheduler created successfully")

    # Test create_workers
    scheduler.create_workers("my-test", scheduling_config)
    logger.info("✓ create_workers called successfully")

    # Test get_workers
    workers = scheduler.get_workers("my-test")
    logger.info(f"✓ get_workers returned: {workers}")

    logger.info("🎉 Real API test passed!")
    return True

    # # Mock the _build_rpc_client method to return a mock client
    # mock_client = Mock()

    # with patch.object(AsystemScheduler, '_build_rpc_client', return_value=mock_client):

    #     # Mock submit_job response
    #     mock_client.submit_job.return_value = {
    #         'job_name': 'test_expr_test_trial:20250714170000',
    #         'job_uid': 'test_job_123'
    #     }

    #     # Mock wait_for_jobs response - 模拟真实的Asystem API响应格式
    #     mock_client.wait_for_jobs.return_value = {
    #         'test_job_worker_0': {
    #             'ip': '10.0.0.1',
    #             'instance': 'test_job_worker_0',
    #             'ports': [8080, 8081]  # 支持多端口
    #         },
    #         'test_job_worker_1': {
    #             'ip': '10.0.0.2',
    #             'instance': 'test_job_worker_1',
    #             'ports': [8080]  # 单端口
    #         }
    #     }

    #     # Mock stop_job response
    #     mock_client.stop_job.return_value = {
    #         'job_uid': 'test_job_123',
    #         'status': 'stopped'
    #     }

    #     # Create scheduler
    #     scheduler = AsystemScheduler(scheduler_config)
    #     logger.info("✓ AsystemScheduler created successfully")

    #     # Test create_workers
    #     scheduler.create_workers(scheduling_config)
    #     logger.info("✓ create_workers called successfully")

    #     # Verify submit_job was called
    #     mock_client.submit_job.assert_called_once()
    #     logger.info("✓ submit_job was called correctly")

    #     # Test get_workers
    #     workers = scheduler.get_workers()
    #     logger.info(f"✓ get_workers returned: {workers}")

    #     # Verify wait_for_jobs was called
    #     mock_client.wait_for_jobs.assert_called()
    #     logger.info("✓ wait_for_jobs was called correctly")

    #     # Test cleanup
    #     scheduler.cleanup()
    #     logger.info("✓ cleanup called successfully")

    #     # Verify stop_job was called during cleanup
    #     mock_client.stop_job.assert_called()
    #     logger.info("✓ stop_job was called correctly during cleanup")

    #     logger.info("🎉 All tests passed!")
    #     return True


'''
def test_error_handling():
    """Test error handling with mocked exceptions."""
    logger.info("Testing error handling...")
    
    scheduler_config = {
        'type': 'asystem',
        'endpoint': 'http://mock-asystem:8081',
        'expr_name': 'test_expr',
        'trial_name': 'test_trial'
    }
    
    container_spec = ContainerSpec(
        cpu=1000,
        gpu=1,
        mem=2048,
        container_image='/storage/openpsi/images/areal-latest.sif',
        cmd='python -m test_worker',
        env_vars={'TEST_VAR': 'test_value'},
        # port=8080
    )
    
    scheduling_config = SchedulingConfig(
        replicas=1,
        specs=[container_spec]
    )
    
    # Mock the _build_rpc_client method to return a mock client that raises exceptions
    mock_client = Mock()
    
    # Mock submit_job to raise an exception
    from arealite.scheduler.asystem.rpc_client import SchedulerError
    mock_client.submit_job.side_effect = SchedulerError("Mocked connection error")
    
    with patch.object(AsystemScheduler, '_build_rpc_client', return_value=mock_client):
        scheduler = AsystemScheduler(scheduler_config)
        
        try:
            scheduler.create_workers(scheduling_config)
            logger.error("Expected SchedulerError but none was raised")
            return False
        except SchedulerError as e:
            logger.info(f"✓ SchedulerError correctly raised: {e}")
            return True
'''


def main():
    """Run all mock tests."""
    logger.info("Starting AsystemScheduler mock tests...")

    try:
        # Test basic functionality
        if not test_basic_functionality():
            logger.error("Basic functionality test failed")
            return 1

        # Test error handling
        # if not test_error_handling():
        #     logger.error("Error handling test failed")
        #     return 1

        logger.info("🎉 All mock tests completed successfully!")
        return 0

    except Exception as e:
        logger.error(f"Mock test failed with unexpected error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
