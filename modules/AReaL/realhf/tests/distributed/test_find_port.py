import multiprocessing

from realhf.base import name_resolve, names, testing
from realhf.base.network import find_multiple_free_ports, gethostip


def _worker_process(result_queue, count, low, high, experiment, trial):
    """Helper function for multi-process testing."""
    ports = find_multiple_free_ports(
        count,
        low=low,
        high=high,
        experiment_name=experiment,
        trial_name=trial,
    )
    for port in ports:
        result_queue.put(port)


def test_find_free_port_multiprocess():
    """Test that multiple processes get different ports."""
    num_processes = 100
    experiment = "multi_port_test"
    trial = "trial1"

    testing.clear_name_resolve(experiment, trial)

    result_queue = multiprocessing.Queue()
    count = 2
    processes = []

    for _ in range(num_processes):
        p = multiprocessing.Process(
            target=_worker_process,
            args=(result_queue, count, 10000, 60000, experiment, trial),
        )
        processes.append(p)
        p.start()

    for p in processes:
        p.join()
        assert p.exitcode == 0

    results = []
    while not result_queue.empty():
        results.append(result_queue.get())

    assert len(results) == num_processes * count
    assert len(set(results)) == num_processes * count  # All ports are unique

    # Verify all ports are registered in name_resolve
    ports_name = names.used_ports(experiment, trial, gethostip())
    used_ports = list(map(int, name_resolve.get_subtree(ports_name)))
    assert set(results).issubset(set(used_ports))
