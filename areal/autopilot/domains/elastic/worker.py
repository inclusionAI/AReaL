# SPDX-License-Identifier: Apache-2.0

"""Candidate subprocess. Evaluation labels and reference objects stay in parent."""

import json
import os
import sys
import threading
import time
from dataclasses import asdict
from pathlib import Path

from areal.autopilot.domains.elastic.contracts import (
    forecast_from_dict,
    history_from_dict,
    observation_from_dict,
    plan_from_dict,
)
from areal.autopilot.domains.elastic.model_evaluator import HistoricalPrediction
from areal.autopilot.domains.elastic.model_execution import (
    historical_case_from_dict,
    historical_input_from_dict,
    joint_config_from_dict,
    joint_scenario_from_dict,
    prediction_fingerprint,
)
from areal.autopilot.domains.elastic.phase import (
    fixed_phase_simulator,
    phase_predictions,
)
from areal.autopilot.revision.execution import load_module
from areal.elastic.policy.basic.joint import (
    JointEvent,
    JointPrediction,
    JointSearchResult,
)
from areal.elastic.policy.basic.profile import workflow_from_profile
from areal.elastic.policy.basic.simulator import (
    SimulationResult,
)


def main():
    request = json.loads(Path(sys.argv[1]).read_text())
    incoming = os.fdopen(int(sys.argv[2]), "r")
    outgoing = os.fdopen(int(sys.argv[3]), "w", buffering=1)

    def send(value):
        outgoing.write(json.dumps(value, allow_nan=False) + "\n")

    rpc_lock = threading.RLock()

    def joint_rpc(index, operation, **payload):
        with rpc_lock:
            send({"type": "simulate", "case": index, "operation": operation, **payload})
            response = json.loads(incoming.readline())
        if "error" in response:
            raise ValueError(response["error"])
        return response

    class JointReferenceProxy:
        def __init__(self, index):
            self.index = index

        def evaluate(self, scenario, config):
            value = joint_rpc(
                self.index, "query", scenario=asdict(scenario), config=asdict(config)
            )["result"]
            return JointPrediction(
                **{
                    **value,
                    "config": joint_config_from_dict(value["config"]),
                    "events": tuple(JointEvent(**e) for e in value["events"]),
                }
            )

    class ReferenceProxy:
        def __init__(self, index, observation, forecast, history):
            self.index, self.observation, self.forecast, self.history = (
                index,
                observation,
                forecast,
                history,
            )

        def evaluate(self, observation, forecast, plan, history):
            if (observation, forecast, history) != (
                self.observation,
                self.forecast,
                self.history,
            ):
                raise ValueError("Optimizer changed the fixed evaluation workload")
            with rpc_lock:
                send(
                    {
                        "type": "simulate",
                        "case": self.index,
                        "plan": [asdict(x) for x in plan],
                    }
                )
                response = json.loads(incoming.readline())
            if "error" in response:
                raise ValueError(response["error"])
            return SimulationResult(**response["result"])

    try:
        module = load_module(Path(request["source_path"]))
        target = request["target"]
        if target == "joint_optimizer":
            configs = []
            for index, case in enumerate(request["cases"]):
                joint_rpc(index, "start")
                settings = {
                    **request["settings"],
                    "candidates": tuple(
                        joint_config_from_dict(c)
                        for c in request["settings"]["candidates"]
                    ),
                }
                candidate = module.build_joint_optimizer(settings)
                result = candidate.optimize(
                    joint_scenario_from_dict(case), JointReferenceProxy(index)
                )
                if not isinstance(result, JointSearchResult):
                    raise ValueError("Joint optimizer must return JointSearchResult")
                config = None if result.config is None else asdict(result.config)
                joint_rpc(index, "finish", config=config)
                configs.append(config)
            send({"type": "result", "configs": configs})
            return
        if target == "serial_model":
            from areal.autopilot.domains.elastic.serial import (
                build_serial_model,
                timing_predictions,
            )

            model = build_serial_model(
                Path(request["source_path"]).read_text(), request["profile"]
            )
            send(
                {"type": "result", "rows": timing_predictions(model, request["cases"])}
            )
            return
        if target == "model":
            candidate = module.build_model(
                tuple(
                    historical_case_from_dict(c)
                    for c in request["calibration"]["cases"]
                ),
                dict(request["settings"]),
            )
            if not callable(getattr(candidate, "predict", None)):
                raise ValueError("build_model must return a historical predictor")
            rows = []
            for case in request["cases"]:
                try:
                    prediction = candidate.predict(historical_input_from_dict(case))
                    if not isinstance(prediction, HistoricalPrediction):
                        raise ValueError("Candidate must return HistoricalPrediction")
                    rows.append(
                        {
                            "prediction": asdict(
                                HistoricalPrediction(
                                    prediction.values, prediction.reason
                                )
                            )
                        }
                    )
                except Exception as error:
                    rows.append({"error": f"{type(error).__name__}: {error}"})
            if all(not row.get("error") for row in rows):
                repeated = []
                for case in request["cases"]:
                    value = candidate.predict(historical_input_from_dict(case))
                    if not isinstance(value, HistoricalPrediction):
                        raise ValueError(
                            "Model predict must remain a pure frozen prediction"
                        )
                    repeated.append({"prediction": asdict(value)})
                if prediction_fingerprint(rows) != prediction_fingerprint(repeated):
                    raise ValueError(
                        "Model predict must be repeatable without updating fitted state"
                    )
            export = {}
            if "joint_cases" in request:
                joint = getattr(candidate, "joint", None)
                if not callable(getattr(joint, "evaluate", None)):
                    raise ValueError(
                        "Model factory must export predictor.joint.evaluate"
                    )

                def project():
                    values = []
                    for case in request["joint_cases"]:
                        value = joint.evaluate(
                            joint_scenario_from_dict(case["scenario"]),
                            joint_config_from_dict(case["config"]),
                        )
                        if not isinstance(value, JointPrediction):
                            raise ValueError("Joint export must return JointPrediction")
                        values.append(asdict(value))
                    return values

                values = project()
                if values != project():
                    raise ValueError(
                        "Joint export must remain a frozen repeatable model"
                    )
                after = [
                    {
                        "prediction": asdict(
                            candidate.predict(historical_input_from_dict(case))
                        )
                    }
                    for case in request["cases"]
                ]
                if prediction_fingerprint(after) != prediction_fingerprint(rows):
                    raise ValueError(
                        "Joint export changed fitted historical model state"
                    )
                export["joint_predictions"] = values
            send({"type": "result", "rows": rows, **export})
            return
        factory = getattr(module, f"build_{target}")
        candidate = factory(
            workflow_from_profile(request["profile"])
            if target == "simulator"
            else dict(request["settings"])
        )
        if target == "simulator" and request.get("phase_only"):
            candidate = fixed_phase_simulator(
                candidate, workflow_from_profile(request["profile"]), request["profile"]
            )
            send(
                {
                    "type": "result",
                    "rows": phase_predictions(candidate, request["cases"]),
                }
            )
            return
        rows = []
        for index, case in enumerate(request["cases"]):
            try:
                observation = observation_from_dict(case["observation"])
                history = history_from_dict(case["history"])
                forecast = forecast_from_dict(case["forecast"])
                started = time.monotonic()
                if target == "simulator":
                    result = candidate.evaluate(
                        observation, forecast, plan_from_dict(case["plan"]), history
                    )
                    rows.append({"prediction": asdict(result)})
                else:
                    proxy = ReferenceProxy(index, observation, forecast, history)
                    result = candidate.optimize(observation, proxy, history)
                    rows.append(
                        {
                            "plan": [asdict(x) for x in result.plan],
                            "decision": asdict(result.decision),
                            "seconds": time.monotonic() - started,
                        }
                    )
            except Exception as error:
                rows.append({"error": f"{type(error).__name__}: {error}"})
        send({"type": "result", "rows": rows})
    except BaseException as error:
        send({"type": "result", "error": f"{type(error).__name__}: {error}"})


if __name__ == "__main__":
    main()
