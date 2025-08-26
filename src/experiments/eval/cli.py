import argparse
import json

from loguru import logger

from experiments.eval.analyze_results import analyze_results
from experiments.eval.run_eval import RunMode, make_configs, run_evals
from tau2.scripts.view_simulations import main as view_simulations_main
from tau2.utils.utils import DATA_DIR

DATA_EXP_DIR = DATA_DIR / "exp"

DEFAULT_LLM_SUPERVISOR = None
DEFAULT_LLM_USER = "gpt-4.1-2025-04-14"
DEFAULT_MAX_CONCURRENCY = 100
DEFAULT_NUM_TRIALS = 4
DEFAULT_SEED = 300
DEFAULT_MAX_STEPS = 200
DEFAULT_MAX_ERRORS = 10
DEFAULT_DOMAINS = ["retail", "airline", "telecom", "telecom-workflow"]
DEFAULT_MODES = [
    RunMode.DEFAULT.value,
    RunMode.ORACLE_PLAN.value,
    RunMode.NO_USER.value,
]
DEFAULT_LLM_AGENT_ARGS = {"temperature": 0.0}
DEFAULT_LLM_USER_ARGS = {"temperature": 0.0}
DEFAULT_LLM_SUPERVISOR_ARGS = {"temperature": 0.0}


def get_cli_parser() -> argparse.ArgumentParser:
    """
    Get the CLI parser with subparsers for run-evals and analyze-results commands.
    """
    parser = argparse.ArgumentParser(
        description="Run evaluations and analyze results for experiments."
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Run evals subparser
    run_parser = subparsers.add_parser("run-evals", help="Run evaluation experiments")
    run_parser.add_argument(
        "--exp-dir",
        type=str,
        required=True,
        help=f"Path to the experiment directory relative to {DATA_EXP_DIR}. This will be created if it doesn't exist.",
    )
    run_parser.add_argument(
        "--num-tasks",
        type=int,
        default=None,
        help="Number of tasks to run. Defaults to None.",
    )

    # Add hyperparameters arguments
    run_parser.add_argument(
        "--llms",
        type=str,
        nargs="+",
        required=True,
        help="List of LLMs to test (e.g. gpt-4.1-2025-04-14 claude-3-7-sonnet-20250219)",
    )
    run_parser.add_argument(
        "--domains",
        type=str,
        nargs="+",
        default=DEFAULT_DOMAINS,
        choices=DEFAULT_DOMAINS,
        help=f"List of domains to test. Default is {DEFAULT_DOMAINS}.",
    )
    run_parser.add_argument(
        "--modes",
        type=str,
        nargs="+",
        default=DEFAULT_MODES,
        choices=DEFAULT_MODES,
        help=f"List of modes to test. Default is {DEFAULT_MODES}.",
    )

    # Add experiment parameters
    run_parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED, help="Random seed for experiments"
    )
    run_parser.add_argument(
        "--max-steps",
        type=int,
        default=DEFAULT_MAX_STEPS,
        help="Maximum number of steps per trial",
    )
    run_parser.add_argument(
        "--max-errors",
        type=int,
        default=DEFAULT_MAX_ERRORS,
        help="Maximum number of errors allowed",
    )
    run_parser.add_argument(
        "--max-concurrency",
        type=int,
        default=DEFAULT_MAX_CONCURRENCY,
        help=f"Maximum number of concurrent simulations. Default is {DEFAULT_MAX_CONCURRENCY}.",
    )
    run_parser.add_argument(
        "--num-trials",
        type=int,
        default=DEFAULT_NUM_TRIALS,
        help=f"Number of trials per configuration. Default is {DEFAULT_NUM_TRIALS}.",
    )

    # LLM model for user simulator
    run_parser.add_argument(
        "--llm-user",
        type=str,
        default=DEFAULT_LLM_USER,
        help=f"LLM model to use for user simulator. Default is {DEFAULT_LLM_USER}.",
    )

    # LLM arguments for agent and user simulator.
    run_parser.add_argument(
        "--agent-llm-args",
        type=str,
        default=json.dumps(DEFAULT_LLM_AGENT_ARGS),
        help=f"JSON string of arguments for agent LLM. Default is {DEFAULT_LLM_AGENT_ARGS}.",
    )
    run_parser.add_argument(
        "--user-llm-args",
        type=str,
        default=json.dumps(DEFAULT_LLM_USER_ARGS),
        help=f"JSON string of arguments for user LLM. Default is {DEFAULT_LLM_USER_ARGS}.",
    )

    # Analyze results subparser
    analyze_parser = subparsers.add_parser(
        "analyze-results", help="Analyze experiment results"
    )
    analyze_parser.add_argument(
        "--exp-dir",
        type=str,
        required=True,
        help="Path to the experiment directory containing results to analyze.",
    )

    # View simulations subparser
    view_parser = subparsers.add_parser(
        "view", help="View simulation results interactively"
    )
    view_parser.add_argument(
        "--dir",
        type=str,
        default=None,
        help=f"Directory containing simulation files. Defaults to {DATA_DIR}/simulations if not specified.",
    )
    view_parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Specific simulation file to view (optional).",
    )
    view_parser.add_argument(
        "--only-failed",
        action="store_true",
        help="Only show failed simulations.",
    )
    view_parser.add_argument(
        "--only-all-failed",
        action="store_true",
        help="Only show tasks where all trials failed.",
    )

    return parser


def main():
    """
    Run the evaluations or analyze results based on the command.
    """
    parser = get_cli_parser()
    args = parser.parse_args()

    if args.command == "run-evals":
        # Convert relative path to absolute path using DATA_EXP_DIR
        exp_dir = DATA_EXP_DIR / args.exp_dir

        # Parse hyperparameters
        hyperparams = {
            "llm": args.llms,
            "domain": args.domains,
            "mode": [RunMode(mode) for mode in args.modes],
        }

        # Parse LLM arguments
        llm_agent_args = json.loads(args.agent_llm_args)
        llm_user_args = json.loads(args.user_llm_args)

        logger.info(
            f"Running experiment in {exp_dir} with num_tasks {args.num_tasks}..."
        )
        if exp_dir.exists():
            res = input(f"Experiment directory {exp_dir} already exists. Run it? (y/n)")
            if res.lower().strip() != "y":
                return
        exp_dir.mkdir(parents=True, exist_ok=True)
        configs = make_configs(
            hyperparams=hyperparams,
            llm_user=args.llm_user,
            llm_agent_args=llm_agent_args,
            llm_user_args=llm_user_args,
            seed=args.seed,
            max_steps=args.max_steps,
            max_errors=args.max_errors,
            max_concurrency=args.max_concurrency,
            num_trials=args.num_trials,
            num_tasks=args.num_tasks,
            exp_dir=exp_dir,
        )
        run_evals(configs)
        logger.info(f"Experiment in {exp_dir} completed.")
        analyze_results(exp_dir)

    elif args.command == "analyze-results":
        # Convert relative path to absolute path using DATA_DIR
        exp_dir = DATA_EXP_DIR / args.exp_dir
        analyze_results(exp_dir)

    elif args.command == "view":
        # Run the view simulations interactive tool
        view_simulations_main(
            sim_file=args.file,
            only_show_failed=args.only_failed,
            only_show_all_failed=args.only_all_failed,
            sim_dir=args.dir,
        )

    else:
        parser.print_help()
        return


if __name__ == "__main__":
    main()
