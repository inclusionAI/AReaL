# SPDX-License-Identifier: Apache-2.0

"""Run a bounded code-revision loop or its synthetic CPU demonstration."""

import argparse
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

from areal.autopilot.contracts import BatchWorkload, ElasticObservation, TrainingConfig
from areal.autopilot.elastic.policy.basic.simulator import (
    ConstantComponent,
    ForecastUpdate,
    HistorySnapshot,
    Measurement,
    PlanStep,
    WorkflowModel,
    WorkflowSimulator,
)
from areal.autopilot.revision import (
    CommandRevisionAgent,
    RevisionEvaluator,
    RevisionLoop,
)

AFFINE_SOURCE = """from dataclasses import replace
from areal.autopilot.elastic.policy.basic.simulator import Estimate, WorkflowSimulator

class AffineTrain:
    def __init__(self, base):
        self.base = base
    def estimate(self, context):
        rows = [x for x in (*self.base.seed, *context.history.measurements)
                if x.phase == "train" and x.world_size == context.config.world_size and x.sequence_tokens > 0]
        if len(rows) < 2:
            return self.base.estimate(context)
        mean_x = sum(x.sequence_tokens for x in rows) / len(rows)
        mean_y = sum(x.seconds for x in rows) / len(rows)
        denominator = sum((x.sequence_tokens - mean_x)**2 for x in rows)
        if not denominator:
            return self.base.estimate(context)
        slope = sum((x.sequence_tokens - mean_x)*(x.seconds - mean_y) for x in rows) / denominator
        return Estimate(max(0, mean_y + slope*(context.workload.sequence_tokens - mean_x)))

def build_simulator(workflow):
    return WorkflowSimulator(replace(workflow, train=AffineTrain(workflow.train)))
"""

CACHE_SOURCE = """from dataclasses import replace
from areal.autopilot.elastic.policy.basic.optimizer import BasicOptimizer

class CachedOptimizer:
    def __init__(self, settings):
        self.base = BasicOptimizer(**settings)
    def optimize(self, observation, simulator, history):
        cache = {}
        class CachedReference:
            def evaluate(self, obs, forecast, plan, hist):
                canonical = ((replace(plan[0], release_before=False), *plan[1:])
                             if not obs.training_active else plan)
                key = tuple((x.config.world_size, x.release_before) for x in canonical)
                if key not in cache:
                    cache[key] = simulator.evaluate(obs, forecast, canonical, hist)
                return cache[key]
        return self.base.optimize(observation, CachedReference(), history)

def build_optimizer(settings):
    return CachedOptimizer(settings)
"""


def demo_inputs(target):
    """Training tokens 100/400; simulator validation tokens 200/300 are held out."""
    phases = {"start": 4, "restore": 3, "checkpoint": 1, "publish": 1, "release": 1}
    records = []
    for count in range(1, 4):
        records.extend(
            Measurement(phase, count, seconds) for phase, seconds in phases.items()
        )
        records.extend(
            Measurement("train", count, 5 + tokens / (10 * count), tokens)
            for tokens in (100, 400)
        )
    profile = {
        "format_version": 1,
        "inference_gpus": 1,
        "compatibility": {"workflow": "synthetic-affine-cpu-demo-v1"},
        "measurements": [asdict(x) for x in records],
    }

    def obs(tokens=200, active=False):
        return ElasticObservation(
            "ACTIVE" if active else "SUSPENDED",
            1,
            1,
            active,
            True,
            0,
            True,
            False,
            True,
            10,
            training_config=TrainingConfig(2),
            next_batch=BatchWorkload(1, tokens // 2, tokens),
            max_train_gpus=3,
            has_checkpoint=True,
        )

    if target == "simulator":
        cases = []
        for count in range(1, 4):
            for tokens in (200, 300):
                observation = obs(tokens)
                workload = observation.next_batch
                forecast = (ForecastUpdate(workload, 0),)
                plan = (PlanStep(TrainingConfig(count)),)
                truth = WorkflowModel(
                    ConstantComponent(5 + tokens / (10 * count)),
                    **{key: ConstantComponent(value) for key, value in phases.items()},
                    inference_gpus=1,
                )
                measured = WorkflowSimulator(truth).evaluate(
                    observation, forecast, plan, HistorySnapshot()
                )
                cases.append(
                    {
                        "observation": asdict(observation),
                        "history": asdict(HistorySnapshot()),
                        "forecast": [asdict(x) for x in forecast],
                        "plan": [asdict(x) for x in plan],
                        "observed": {
                            "feasible": True,
                            "seconds": measured.seconds,
                            "total_gpu_seconds": measured.total_gpu_seconds,
                        },
                    }
                )
        invalid = dict(
            cases[0],
            plan=[asdict(PlanStep(TrainingConfig(4)))],
            observed={"feasible": False},
        )
        cases.append(invalid)
    else:
        history = HistorySnapshot((BatchWorkload(1, 150, 300, 20),))
        observations = [
            obs(200),
            obs(300),
            obs(200, True),
            replace(
                obs(active=True),
                batch_ready=False,
                next_batch=None,
                episode_active=True,
                episode_elapsed=5,
            ),
            replace(obs(active=True), pending_publication=True),
        ]
        cases = [
            {"observation": asdict(item), "history": asdict(history)}
            for item in observations
        ]
    return profile, cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=["simulator", "optimizer"])
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--cases", type=Path)
    parser.add_argument("--settings", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--demo", action="store_true")
    proposer = parser.add_mutually_exclusive_group()
    proposer.add_argument("--demo-agent", action="store_true")
    proposer.add_argument("--agent-command", nargs=argparse.REMAINDER)
    proposer.add_argument("--llm-model")
    parser.add_argument("--llm-base-url")
    parser.add_argument("--llm-api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--llm-timeout", type=float, default=180)
    parser.add_argument("--llm-max-output-tokens", type=int, default=8192)
    parser.add_argument(
        "--llm-response-format",
        choices=["json_schema", "json_object"],
        default="json_schema",
    )
    parser.add_argument("--llm-reasoning-effort")
    parser.add_argument(
        "--llm-token-parameter",
        choices=["max_completion_tokens", "max_tokens"],
        default="max_completion_tokens",
    )
    args = parser.parse_args()
    if args.demo_agent:
        request = json.load(sys.stdin)
        source = AFFINE_SOURCE if request["target"] == "simulator" else CACHE_SOURCE
        sys.stdout.write(
            json.dumps(
                {
                    "source": source,
                    "rationale": "Fit affine train costs from calibration"
                    if request["target"] == "simulator"
                    else "Reuse reference queries for equivalent suspended-state plans",
                }
            )
        )
        return
    if args.target is None or args.output is None:
        parser.error("--target and --output are required")
    if args.demo:
        profile, cases = demo_inputs(args.target)
    else:
        if (
            args.profile is None
            or args.cases is None
            or not (args.agent_command or args.llm_model)
        ):
            parser.error(
                "Supply --profile, --cases and either --llm-model or --agent-command; "
                "or use --demo"
            )
        profile, cases = (
            json.loads(args.profile.read_text()),
            json.loads(args.cases.read_text()),
        )
    if args.llm_model:
        command = [
            sys.executable,
            "-m",
            "areal.autopilot.revision",
            "--model",
            args.llm_model,
            "--api-key-env",
            args.llm_api_key_env,
            "--timeout",
            str(args.llm_timeout),
            "--max-output-tokens",
            str(args.llm_max_output_tokens),
            "--response-format",
            args.llm_response_format,
            "--token-parameter",
            args.llm_token_parameter,
        ]
        if args.llm_base_url:
            command.extend(["--base-url", args.llm_base_url])
        if args.llm_reasoning_effort:
            command.extend(["--reasoning-effort", args.llm_reasoning_effort])
        agent = CommandRevisionAgent(tuple(command), args.llm_timeout + 10)
    else:
        command = (
            tuple(args.agent_command)
            if args.agent_command
            else (sys.executable, "-m", "examples.autopilot.revising", "--demo-agent")
        )
        agent = CommandRevisionAgent(command)
    settings = None if args.settings is None else json.loads(args.settings.read_text())
    evaluator = RevisionEvaluator(args.target, profile, cases, settings=settings)
    loop = RevisionLoop(agent, evaluator, args.output)
    result = loop.run(max_rounds=args.max_rounds)
    if (
        args.demo
        and not (args.agent_command or args.llm_model)
        and result["active_source"] is None
    ):
        raise RuntimeError("Synthetic demo did not accept the expected improvement")


if __name__ == "__main__":
    main()
