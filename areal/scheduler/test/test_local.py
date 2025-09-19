import logging

from areal.scheduler.local import LocalScheduler
from areal.scheduler.test.my_engine import MyEngine


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    sched = LocalScheduler({"type": "local"})
    sched.create_workers("infer", {"num_workers": 1})

    workers = sched.get_workers("infer")
    engine_obj = MyEngine({"value": 24})
    assert sched.create_engine(workers[0].id, engine_obj, {"init": 1})
    result = sched.call_engine(workers[0].id, "infer", 100, 10)
    print("Result:", result)
    assert result == 1024


if __name__ == "__main__":
    main()
