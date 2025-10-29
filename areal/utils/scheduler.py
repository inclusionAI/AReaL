from areal.api.cli_args import SchedulingSpec
from areal.api.engine_api import Scheduling


def scheduling_specs_to_schedulings(specs: list[SchedulingSpec]) -> list[Scheduling]:
    result = []
    for spec in specs:
        sch = Scheduling(
            cpu=spec.cpu,
            gpu=spec.gpu,
            mem=spec.mem,
            port_count=spec.port_count,
            container_image=spec.image,
            type=spec.type,
            env_vars=spec.env_vars,
            cmd=spec.cmd,
        )
        result.append(sch)
    return result
