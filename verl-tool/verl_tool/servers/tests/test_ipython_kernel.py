import ray
import time
import threading
import concurrent.futures
from collections import defaultdict
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(threadName)-12s] %(levelname)-8s %(message)s'
)
logger = logging.getLogger(__name__)

# Import your functions
from verl_tool.servers.tools.utils.ipython_tool import (
    call_python_script_with_ipython,
    _get_actor,
    cleanup_all_kernels,
    get_kernel_stats
)


def test_1_sequential_same_request():
    """Test 1: Sequential calls with same request_id (should reuse actor)"""
    print("\n" + "="*80)
    print("TEST 1: Sequential calls with same request_id")
    print("="*80)
    
    cleanup_all_kernels()
    time.sleep(0.5)
    
    request_id = "seq_test"
    actor_ids = []
    
    for i in range(5):
        result, success = call_python_script_with_ipython(
            request_id, 
            f"import ray; print(ray.get_runtime_context().get_actor_id())"
        )
        logger.info(f"Call {i+1}: Result = {result}")
        actor_ids.append(result.strip())
        time.sleep(0.2)
    
    unique_actors = set(actor_ids)
    print(f"\nResults:")
    print(f"  Total calls: {len(actor_ids)}")
    print(f"  Unique actors: {len(unique_actors)}")
    print(f"  Actor IDs: {actor_ids}")
    
    if len(unique_actors) == 1:
        print("  ✅ PASS: Only one actor used")
    else:
        print(f"  ❌ FAIL: {len(unique_actors)} different actors created!")
    
    cleanup_all_kernels()
    return len(unique_actors) == 1


def test_2_concurrent_same_request():
    """Test 2: Concurrent calls with same request_id (should reuse actor)"""
    print("\n" + "="*80)
    print("TEST 2: Concurrent calls with same request_id")
    print("="*80)
    
    cleanup_all_kernels()
    time.sleep(0.5)
    
    request_id = "concurrent_test"
    num_threads = 10
    actor_ids = []
    
    def execute_and_get_actor_id(call_num):
        result, success = call_python_script_with_ipython(
            request_id,
            f"import ray; print(ray.get_runtime_context().get_actor_id())"
        )
        logger.info(f"Call {call_num}: Got actor_id = {result.strip()}")
        return result.strip()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(execute_and_get_actor_id, i) for i in range(num_threads)]
        actor_ids = [f.result() for f in concurrent.futures.as_completed(futures)]
    
    unique_actors = set(actor_ids)
    print(f"\nResults:")
    print(f"  Total calls: {len(actor_ids)}")
    print(f"  Unique actors: {len(unique_actors)}")
    
    if len(unique_actors) == 1:
        print("  ✅ PASS: Only one actor used")
    else:
        print(f"  ❌ FAIL: {len(unique_actors)} different actors created!")
        for i, aid in enumerate(actor_ids):
            print(f"    Call {i+1}: {aid}")
    
    cleanup_all_kernels()
    return len(unique_actors) == 1


def test_3_rapid_fire_same_request():
    """Test 3: Rapid-fire calls without delay (stress test)"""
    print("\n" + "="*80)
    print("TEST 3: Rapid-fire calls (no delays)")
    print("="*80)
    
    cleanup_all_kernels()
    time.sleep(0.5)
    
    request_id = "rapid_test"
    num_calls = 20
    actor_ids = []
    
    def execute_and_get_actor_id(call_num):
        result, success = call_python_script_with_ipython(
            request_id,
            f"import ray; print(ray.get_runtime_context().get_actor_id())"
        )
        return result.strip(), call_num
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_calls) as executor:
        futures = [executor.submit(execute_and_get_actor_id, i) for i in range(num_calls)]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]
    
    actor_ids = [r[0] for r in results]
    unique_actors = set(actor_ids)
    
    print(f"\nResults:")
    print(f"  Total calls: {len(actor_ids)}")
    print(f"  Unique actors: {len(unique_actors)}")
    
    if len(unique_actors) == 1:
        print("  ✅ PASS: Only one actor used")
    else:
        print(f"  ❌ FAIL: {len(unique_actors)} different actors created!")
        # Show which calls got which actors
        actor_to_calls = defaultdict(list)
        for aid, (_, call_num) in zip(actor_ids, results):
            actor_to_calls[aid].append(call_num)
        
        for aid, calls in actor_to_calls.items():
            print(f"    Actor {aid}: used by calls {calls}")
    
    cleanup_all_kernels()
    return len(unique_actors) == 1


def test_4_direct_get_actor():
    """Test 4: Direct _get_actor calls (bypassing the async wrapper)"""
    print("\n" + "="*80)
    print("TEST 4: Direct _get_actor calls")
    print("="*80)
    
    cleanup_all_kernels()
    time.sleep(0.5)
    
    request_id = "direct_test"
    num_threads = 10
    actor_handles = []
    
    def get_actor_directly(call_num):
        actor = _get_actor(request_id)
        actor_id = ray.get(actor.get_actor_id.remote())
        logger.info(f"Call {call_num}: Got actor_id = {actor_id}")
        return actor_id, actor
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(get_actor_directly, i) for i in range(num_threads)]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]
    
    actor_ids = [r[0] for r in results]
    unique_actors = set(actor_ids)
    
    print(f"\nResults:")
    print(f"  Total calls: {len(actor_ids)}")
    print(f"  Unique actors: {len(unique_actors)}")
    
    if len(unique_actors) == 1:
        print("  ✅ PASS: Only one actor used")
    else:
        print(f"  ❌ FAIL: {len(unique_actors)} different actors created!")
        for i, aid in enumerate(actor_ids):
            print(f"    Call {i+1}: {aid}")
    
    cleanup_all_kernels()
    return len(unique_actors) == 1


def test_6_state_persistence_across_calls():
    """Test 6: Verify state persists across multiple calls (same actor)"""
    print("\n" + "="*80)
    print("TEST 6: State persistence test")
    print("="*80)
    
    cleanup_all_kernels()
    time.sleep(0.5)
    
    request_id = "state_test"
    
    # Set a variable
    result1, success1 = call_python_script_with_ipython(request_id, "x = 42")
    print(f"Set x=42: success={success1}")
    
    # Read it back multiple times
    results = []
    for i in range(5):
        result, success = call_python_script_with_ipython(request_id, "print(x)")
        results.append((result.strip(), success))
        logger.info(f"Call {i+1}: x = {result.strip()}")
    
    # All should return "42"
    all_correct = all(r[0] == "42" and r[1] for r in results)
    
    print(f"\nResults: {results}")
    
    if all_correct:
        print("  ✅ PASS: State persisted correctly")
    else:
        print("  ❌ FAIL: State was not consistent!")
    
    cleanup_all_kernels()
    return all_correct


def test_7_interleaved_requests():
    """Test 7: Interleave calls to different request_ids"""
    print("\n" + "="*80)
    print("TEST 7: Interleaved requests (multiple request_ids)")
    print("="*80)
    
    cleanup_all_kernels()
    time.sleep(0.5)
    
    request_ids = ["req_A", "req_B", "req_C"]
    results = defaultdict(list)
    
    def execute_for_request(request_id, call_num):
        result, success = call_python_script_with_ipython(
            request_id,
            f"import ray; print(ray.get_runtime_context().get_actor_id())"
        )
        actor_id = result.strip()
        logger.info(f"Request {request_id}, Call {call_num}: actor_id = {actor_id}")
        return request_id, actor_id
    
    # Create interleaved calls
    calls = []
    for i in range(10):
        for req_id in request_ids:
            calls.append((req_id, i))
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:
        futures = [executor.submit(execute_for_request, req_id, call_num) 
                   for req_id, call_num in calls]
        
        for future in concurrent.futures.as_completed(futures):
            req_id, actor_id = future.result()
            results[req_id].append(actor_id)
    
    print(f"\nResults by request_id:")
    all_pass = True
    
    for req_id in request_ids:
        unique = set(results[req_id])
        print(f"  {req_id}: {len(results[req_id])} calls, {len(unique)} unique actors")
        
        if len(unique) == 1:
            print(f"    ✅ PASS")
        else:
            print(f"    ❌ FAIL: Multiple actors created!")
            print(f"       Actors: {unique}")
            all_pass = False
    
    cleanup_all_kernels()
    return all_pass


def run_all_threading_tests():
    """Run all tests and summarize results"""
    print("\n" + "#"*80)
    print("# RUNNING ALL TESTS TO REPRODUCE THE BUG")
    print("#"*80)
    
    tests = [
        ("Sequential same request", test_1_sequential_same_request),
        ("Concurrent same request", test_2_concurrent_same_request),
        ("Rapid-fire calls", test_3_rapid_fire_same_request),
        ("Direct _get_actor", test_4_direct_get_actor),
        ("State persistence", test_6_state_persistence_across_calls),
        ("Interleaved requests", test_7_interleaved_requests),
    ]
    
    results = {}
    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            logger.error(f"Test '{name}' crashed: {e}", exc_info=True)
            results[name] = False
        time.sleep(1)  # Brief pause between tests
    
    # Summary
    print("\n" + "#"*80)
    print("# TEST SUMMARY")
    print("#"*80)
    
    for name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{status}: {name}")
    
    passed_count = sum(1 for p in results.values() if p)
    total_count = len(results)
    
    print(f"\n{passed_count}/{total_count} tests passed")
    
    if passed_count == total_count:
        print("\n🎉 All tests passed! No bug detected.")
    else:
        print(f"\n⚠️  {total_count - passed_count} test(s) failed - bug reproduced!")


import ray
import time
import threading
import concurrent.futures
from collections import defaultdict
import logging


# ============================================================================
# RAY REMOTE FUNCTIONS - This is where the bug likely occurs
# ============================================================================

@ray.remote
def execute_code_remote(request_id: str, code: str):
    """
    Ray remote function that calls call_python_script_with_ipython.
    This simulates your actual use case: Ray function calling another Ray actor.
    """
    logger.info(f"Ray worker executing for request_id={request_id}")
    result, success = call_python_script_with_ipython(request_id, code)
    return result, success, request_id


@ray.remote
def get_actor_id_remote(request_id: str):
    """Get the actor ID through a ray remote function."""
    result, success = call_python_script_with_ipython(
        request_id,
        "import ray; print(ray.get_runtime_context().get_actor_id())"
    )
    actor_id = result.strip() if success else "ERROR"
    logger.info(f"Remote function got actor_id={actor_id} for request_id={request_id}")
    return actor_id


@ray.remote
def stress_test_remote(request_id: str, iteration: int):
    """Stress test: multiple remote functions hitting same request_id."""
    result, success = call_python_script_with_ipython(
        request_id,
        f"import ray; import os; print(ray.get_runtime_context().get_actor_id()); x = {iteration}"
    )
    return result.strip(), iteration


# ============================================================================
# TESTS - RAY UNDER RAY SCENARIOS
# ============================================================================

def test_8_ray_remote_sequential():
    """Test 8: Sequential ray.remote calls with same request_id"""
    print("\n" + "="*80)
    print("TEST 8: Sequential ray.remote calls (same request_id)")
    print("="*80)
    
    cleanup_all_kernels()
    time.sleep(0.5)
    
    request_id = "ray_seq_test"
    actor_ids = []
    
    for i in range(5):
        future = get_actor_id_remote.remote(request_id)
        actor_id = ray.get(future)
        actor_ids.append(actor_id)
        logger.info(f"Call {i+1}: actor_id = {actor_id}")
        time.sleep(0.2)
    
    unique_actors = set(actor_ids)
    print(f"\nResults:")
    print(f"  Total calls: {len(actor_ids)}")
    print(f"  Unique actors: {len(unique_actors)}")
    print(f"  Actor IDs: {actor_ids}")
    
    if len(unique_actors) == 1:
        print("  ✅ PASS: Only one actor used")
    else:
        print(f"  ❌ FAIL: {len(unique_actors)} different actors created!")
    
    cleanup_all_kernels()
    return len(unique_actors) == 1


def test_9_ray_remote_concurrent():
    """Test 9: Concurrent ray.remote calls with same request_id - THE KEY TEST"""
    print("\n" + "="*80)
    print("TEST 9: Concurrent ray.remote calls (same request_id) ⚠️ KEY TEST")
    print("="*80)
    
    cleanup_all_kernels()
    time.sleep(0.5)
    
    request_id = "ray_concurrent_test"
    num_calls = 10
    
    # Launch all remote calls at once (no waiting)
    futures = [get_actor_id_remote.remote(request_id) for _ in range(num_calls)]
    
    # Collect results
    actor_ids = ray.get(futures)
    
    unique_actors = set(actor_ids)
    print(f"\nResults:")
    print(f"  Total calls: {len(actor_ids)}")
    print(f"  Unique actors: {len(unique_actors)}")
    
    if len(unique_actors) == 1:
        print("  ✅ PASS: Only one actor used")
    else:
        print(f"  ❌ FAIL: {len(unique_actors)} different actors created!")
        for i, aid in enumerate(actor_ids):
            print(f"    Call {i+1}: {aid}")
    
    cleanup_all_kernels()
    return len(unique_actors) == 1


def test_10_ray_remote_rapid_fire():
    """Test 10: Rapid-fire ray.remote calls (stress test)"""
    print("\n" + "="*80)
    print("TEST 10: Rapid-fire ray.remote calls (stress test)")
    print("="*80)
    
    cleanup_all_kernels()
    time.sleep(0.5)
    
    request_id = "ray_rapid_test"
    num_calls = 20
    
    # Launch many calls simultaneously
    futures = [stress_test_remote.remote(request_id, i) for i in range(num_calls)]
    # futures = [get_actor_id_remote.remote(request_id) for _ in range(num_calls)]
    
    # Get all results
    results = ray.get(futures)
    # actor_ids = results
    actor_ids = [r[0] for r in results]
    iterations = [r[1] for r in results]
    
    unique_actors = set(actor_ids)
    print(f"\nResults:")
    print(f"  Total calls: {len(actor_ids)}")
    print(f"  Unique actors: {len(unique_actors)}")
    print(f"  Actor IDs: {actor_ids}")
    
    if len(unique_actors) == 1:
        print("  ✅ PASS: Only one actor used")
    else:
        print(f"  ❌ FAIL: {len(unique_actors)} different actors created!")
        # Group by actor
        actor_to_calls = defaultdict(list)
        for aid, iteration in zip(actor_ids, iterations):
            actor_to_calls[aid].append(iteration)
        
        for aid, calls in actor_to_calls.items():
            print(f"    Actor {aid}: handled iterations {sorted(calls)}")
    
    cleanup_all_kernels()
    return len(unique_actors) == 1


def test_11_ray_remote_with_wait():
    """Test 11: Ray remote calls with ray.wait (partial completion)"""
    print("\n" + "="*80)
    print("TEST 11: Ray remote with ray.wait (partial completion)")
    print("="*80)
    
    cleanup_all_kernels()
    time.sleep(0.5)
    
    request_id = "ray_wait_test"
    num_calls = 15
    
    futures = [get_actor_id_remote.remote(request_id) for _ in range(num_calls)]
    
    actor_ids = []
    remaining = futures
    
    while remaining:
        # Wait for at least 1 to complete
        ready, remaining = ray.wait(remaining, num_returns=1, timeout=10)
        
        for ref in ready:
            actor_id = ray.get(ref)
            actor_ids.append(actor_id)
            logger.info(f"Completed: actor_id = {actor_id}")
    
    unique_actors = set(actor_ids)
    print(f"\nResults:")
    print(f"  Total calls: {len(actor_ids)}")
    print(f"  Unique actors: {len(unique_actors)}")
    
    if len(unique_actors) == 1:
        print("  ✅ PASS: Only one actor used")
    else:
        print(f"  ❌ FAIL: {len(unique_actors)} different actors created!")
    
    cleanup_all_kernels()
    return len(unique_actors) == 1


def test_12_ray_remote_different_workers():
    """Test 12: Force execution on different Ray workers"""
    print("\n" + "="*80)
    print("TEST 12: Force different Ray workers")
    print("="*80)
    
    cleanup_all_kernels()
    time.sleep(0.5)
    
    request_id = "ray_worker_test"
    num_calls = 10
    
    # Use .options to potentially force different workers
    futures = []
    for i in range(num_calls):
        # Each call might land on a different worker
        future = get_actor_id_remote.options(num_cpus=0.1).remote(request_id)
        futures.append(future)
    
    actor_ids = ray.get(futures)
    
    unique_actors = set(actor_ids)
    print(f"\nResults:")
    print(f"  Total calls: {len(actor_ids)}")
    print(f"  Unique actors: {len(unique_actors)}")
    
    if len(unique_actors) == 1:
        print("  ✅ PASS: Only one actor used")
    else:
        print(f"  ❌ FAIL: {len(unique_actors)} different actors created!")
        for i, aid in enumerate(actor_ids):
            print(f"    Call {i+1}: {aid}")
    
    cleanup_all_kernels()
    return len(unique_actors) == 1


def test_13_mixed_direct_and_remote():
    """Test 13: Mix of direct calls and ray.remote calls"""
    print("\n" + "="*80)
    print("TEST 13: Mixed direct + ray.remote calls")
    print("="*80)
    
    cleanup_all_kernels()
    time.sleep(0.5)
    
    request_id = "mixed_test"
    actor_ids = []
    
    # Direct call
    result1, _ = call_python_script_with_ipython(
        request_id,
        "import ray; print(ray.get_runtime_context().get_actor_id())"
    )
    actor_ids.append(("direct", result1.strip()))
    
    # Remote calls
    futures = [get_actor_id_remote.remote(request_id) for _ in range(5)]
    remote_results = ray.get(futures)
    actor_ids.extend([("remote", aid) for aid in remote_results])
    
    # Another direct call
    result2, _ = call_python_script_with_ipython(
        request_id,
        "import ray; print(ray.get_runtime_context().get_actor_id())"
    )
    actor_ids.append(("direct", result2.strip()))
    
    print(f"\nResults:")
    for call_type, aid in actor_ids:
        print(f"  {call_type:6s}: {aid}")
    
    unique_actors = set(aid for _, aid in actor_ids)
    print(f"\nUnique actors: {len(unique_actors)}")
    
    if len(unique_actors) == 1:
        print("  ✅ PASS: Only one actor used")
    else:
        print(f"  ❌ FAIL: {len(unique_actors)} different actors created!")
    
    cleanup_all_kernels()
    return len(unique_actors) == 1


def test_14_nested_ray_calls():
    """Test 14: Deeply nested Ray remote calls"""
    print("\n" + "="*80)
    print("TEST 14: Nested Ray remote calls")
    print("="*80)
    
    @ray.remote
    def level_2_call(request_id: str, level: int):
        """Second level: calls the IPython executor"""
        result, success = call_python_script_with_ipython(
            request_id,
            f"import ray; print(ray.get_runtime_context().get_actor_id())"
        )
        return result.strip(), level
    
    @ray.remote
    def level_1_call(request_id: str, level: int):
        """First level: calls level_2"""
        future = level_2_call.remote(request_id, level)
        return ray.get(future)
    
    cleanup_all_kernels()
    time.sleep(0.5)
    
    request_id = "nested_test"
    num_calls = 8
    
    # Launch nested calls
    futures = [level_1_call.remote(request_id, i) for i in range(num_calls)]
    results = ray.get(futures)
    
    actor_ids = [r[0] for r in results]
    unique_actors = set(actor_ids)
    
    print(f"\nResults:")
    print(f"  Total calls: {len(actor_ids)}")
    print(f"  Unique actors: {len(unique_actors)}")
    
    if len(unique_actors) == 1:
        print("  ✅ PASS: Only one actor used")
    else:
        print(f"  ❌ FAIL: {len(unique_actors)} different actors created!")
        for i, aid in enumerate(actor_ids):
            print(f"    Call {i+1}: {aid}")
    
    cleanup_all_kernels()
    return len(unique_actors) == 1


def test_15_ray_state_persistence():
    """Test 15: State persistence through ray.remote calls"""
    print("\n" + "="*80)
    print("TEST 15: State persistence through ray.remote")
    print("="*80)
    
    @ray.remote
    def set_variable(request_id: str, var_name: str, value: int):
        result, success = call_python_script_with_ipython(
            request_id,
            f"{var_name} = {value}"
        )
        return success
    
    @ray.remote
    def get_variable(request_id: str, var_name: str):
        result, success = call_python_script_with_ipython(
            request_id,
            f"print({var_name})"
        )
        return result.strip(), success
    
    cleanup_all_kernels()
    time.sleep(0.5)
    
    request_id = "state_ray_test"
    
    # Set variable through ray remote
    success = ray.get(set_variable.remote(request_id, "test_var", 999))
    print(f"Set test_var=999: {success}")
    
    # Read it back multiple times through ray remote
    futures = [get_variable.remote(request_id, "test_var") for _ in range(5)]
    results = ray.get(futures)
    
    print(f"\nResults:")
    for i, (value, success) in enumerate(results):
        print(f"  Call {i+1}: value={value}, success={success}")
    
    all_correct = all(value == "999" and success for value, success in results)
    
    if all_correct:
        print("  ✅ PASS: State persisted correctly")
    else:
        print("  ❌ FAIL: State inconsistency detected!")
    
    cleanup_all_kernels()
    return all_correct


def run_ray_under_ray_tests():
    """Run all ray-under-ray tests"""
    print("\n" + "#"*80)
    print("# RAY-UNDER-RAY TESTS - THIS IS WHERE THE BUG SHOULD APPEAR")
    print("#"*80)
    
    tests = [
        ("Ray remote sequential", test_8_ray_remote_sequential),
        ("Ray remote concurrent ⚠️", test_9_ray_remote_concurrent),
        ("Ray remote rapid-fire", test_10_ray_remote_rapid_fire),
        ("Ray remote with wait", test_11_ray_remote_with_wait),
        ("Ray different workers", test_12_ray_remote_different_workers),
        ("Mixed direct + remote", test_13_mixed_direct_and_remote),
        ("Nested ray calls", test_14_nested_ray_calls),
        ("Ray state persistence", test_15_ray_state_persistence),
    ]
    
    results = {}
    for name, test_func in tests:
        try:
            print(f"\nRunning: {name}")
            results[name] = test_func()
        except Exception as e:
            logger.error(f"Test '{name}' crashed: {e}", exc_info=True)
            results[name] = False
        time.sleep(1)
    
    # Summary
    print("\n" + "#"*80)
    print("# RAY-UNDER-RAY TEST SUMMARY")
    print("#"*80)
    
    for name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{status}: {name}")
    
    passed_count = sum(1 for p in results.values() if p)
    total_count = len(results)
    
    print(f"\n{passed_count}/{total_count} tests passed")
    
    if passed_count == total_count:
        print("\n🎉 All tests passed! No bug detected in ray-under-ray scenario.")
    else:
        print(f"\n⚠️  {total_count - passed_count} test(s) failed - BUG REPRODUCED!")
        print("\nThe bug occurs when calling Ray actors from within Ray remote functions.")
        print("Each Ray worker process has its own copy of _actor_cache!")


if __name__ == "__main__":
    # Make sure Ray is initialized
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, include_dashboard=False)
    
    run_all_threading_tests()
    run_ray_under_ray_tests()
    
    ray.shutdown()
