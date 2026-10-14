"""Command-line entry point; collection failures retain successful evidence."""

import argparse
import json
import sys
from pathlib import Path

from .monitor._io import read_json
from .monitor.collector import collect, save
from .validator.checker import load_policy, read_document, validate_configuration


def _restart(args: argparse.Namespace) -> int:
    import math

    from .execution import LaunchSpec, RestartCoordinator, Timeouts
    from .execution.protocol import read_json as read_control

    runner = None
    try:
        if not math.isfinite(args.completion_timeout) or args.completion_timeout <= 0:
            raise ValueError("completion timeout must be positive and finite")
        value = read_control(args.launch)
        options = Timeouts(**value.pop("timeouts", {}))
        value["cwd"] = (args.launch.resolve().parent / value["cwd"]).resolve()
        spec = LaunchSpec(**value)
        with RestartCoordinator(spec, args.output, timeouts=options) as runner:
            runner.start()
            result = runner.restart()
            print(
                f"Restart: {result['status']}; segment={result['segment_id']}",
                flush=True,
            )
            runner.wait(timeout=args.completion_timeout)
            print(f"Training completed; artifacts: {runner.directory}")
    except Exception as error:
        # Detailed runtime errors stay in the private per-segment log.
        print(
            f"Restart failed ({type(error).__name__}); inspect {args.output}",
            file=sys.stderr,
        )
        if runner is not None and runner.retained_source is not None:
            source = runner.retained_source
            print(
                f"Source was not terminated (PID/PGID {source.process.pgid}); "
                f"inspect {source.directory / 'runtime.log'}",
                file=sys.stderr,
            )
        return 1
    return 0


def _validate(args: argparse.Namespace) -> int:
    try:
        policy = load_policy(args.policy)
        result = validate_configuration(
            read_document(args.baseline),
            read_document(args.candidate),
            policy,
            runtime_revision=args.runtime_revision,
            context=read_document(args.context) if args.context else None,
        )
        # Results contain paths and reasons, not the potentially sensitive values.
        serialized = (
            json.dumps(result.to_dict(), ensure_ascii=False, indent=2, allow_nan=False)
            + "\n"
        )
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as stream:
                stream.write(serialized)
            print(f"Validation result: {args.output.resolve()}")
            print(f"Validation: {result.status}; changed fields={len(result.changes)}")
        else:
            print(serialized, end="")
    except (ValueError, TypeError, KeyError, OSError) as exc:
        message = (
            str(exc)
            if isinstance(exc, ValueError) and not isinstance(exc, json.JSONDecodeError)
            else type(exc).__name__
        )
        print(f"Validator input/output error: {message}", file=sys.stderr)
        return 2
    return 0 if result.passed else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="autopilot-harness")
    commands = parser.add_subparsers(dest="command", required=True)
    monitor = commands.add_parser(
        "monitor", help="Capture Monitor context once; no GPU required"
    )
    monitor.add_argument(
        "--config", type=Path, required=True, help="JSON source/context configuration"
    )
    monitor.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New capture directory (must not exist)",
    )
    validator = commands.add_parser(
        "validate",
        help="Check configuration and transition constraints; no GPU required",
    )
    validator.add_argument(
        "--baseline",
        type=Path,
        required=True,
        help="Trusted full baseline configuration in JSON",
    )
    validator.add_argument(
        "--candidate",
        type=Path,
        required=True,
        help="Full proposed configuration in JSON",
    )
    validator.add_argument(
        "--policy",
        type=Path,
        required=True,
        help="Harness-owned validation policy in JSON",
    )
    validator.add_argument(
        "--runtime-revision",
        help="Trusted AReaL commit; missing/mismatched revision yields undetermined",
    )
    validator.add_argument(
        "--context",
        type=Path,
        help="Harness-owned model/resource/memory/transition evidence in JSON",
    )
    validator.add_argument(
        "--output", type=Path, help="New result JSON file; defaults to stdout"
    )
    restart = commands.add_parser(
        "restart",
        help="Launch a managed local job, restart once, then wait for completion",
    )
    restart.add_argument(
        "--launch",
        type=Path,
        required=True,
        help="Trusted launch JSON; cwd is relative to this file",
    )
    restart.add_argument("--output", type=Path, required=True, help="New run directory")
    restart.add_argument(
        "--completion-timeout",
        type=float,
        default=86400,
        help="Seconds to wait for final training completion",
    )
    args = parser.parse_args(argv)
    if args.command == "validate":
        return _validate(args)
    if args.command == "restart":
        return _restart(args)
    try:
        if args.output.exists():
            raise FileExistsError(
                "output directory already exists; choose a new capture directory"
            )
        config = read_json(args.config)
        if not isinstance(config, dict):
            raise ValueError("monitor config must be a JSON object")
        snapshot = collect(config, root=args.config.resolve().parent)
        path = save(snapshot, config, args.output)
    except (ValueError, TypeError, KeyError, OSError) as exc:
        # Avoid dumping config values or SDK responses containing credentials.
        message = (
            str(exc)
            if isinstance(exc, (ValueError, FileExistsError))
            and not isinstance(exc, json.JSONDecodeError)
            else type(exc).__name__
        )
        print(f"Monitor input/output error: {message}", file=sys.stderr)
        return 2
    sources = snapshot.runtime_observations["sources"]
    print(f"Monitor snapshot: {path.resolve()}")
    print(
        f"Four outputs saved; samples={len(snapshot.runtime_observations['samples'])}; "
        f"configuration={snapshot.current_configuration.status}"
    )
    for source in sources:
        print(f"{source['source']}: {source['status']} ({source['mode']})")
        for error in source["errors"]:
            print(f"  {error}", file=sys.stderr)
    return 1 if any(source["status"] != "ok" for source in sources) else 0


if __name__ == "__main__":
    raise SystemExit(main())
