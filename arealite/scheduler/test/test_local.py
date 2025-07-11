import logging
from arealite.scheduler.local import LocalScheduler
from arealite.scheduler.test.my_engine import MyEngine


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    sched = LocalScheduler({"type": "local"})
    workers = sched.create_workers({"num_workers": 3})
    worker_id, ip, port = workers[0]
    sched.get_workers()
    engine_obj = MyEngine({"value": 24})
    assert sched.create_engine(worker_id, engine_obj, {"init": 1})
    result = sched.call_engine(worker_id, "infer", 100, 10)
    print("Result:", result)
    assert result == 1024


if __name__ == "__main__":
    main()
