import logging
from areallite.scheduler.local import LocalScheduler
from areallite.scheduler.test.my_engine import MyEngine


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    sched = LocalScheduler({"type": "local"})
    workers = sched.create_workers({"num_workers": 3})
    worker_id, ip, port = workers[0]
    sched.wait_workers()
    engine_obj = MyEngine({"value": 42})
    assert sched.create_engine(worker_id, engine_obj, {"init": 1})
    result = sched.call(worker_id, "infer", 100)
    print("Result:", result)
    assert result == 142


if __name__ == "__main__":
    main()
