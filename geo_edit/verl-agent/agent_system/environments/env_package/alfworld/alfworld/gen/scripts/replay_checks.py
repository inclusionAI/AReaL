import os
import sys

import argparse
import json
import numpy as np
import shutil
import time
from alfworld.env.thor_env import ThorEnv
from alfworld.gen.utils.replay_json import replay_json


JSON_FILENAME = "traj_data.json"


def replay_check(args):
    env = ThorEnv()

    # replay certificate filenames
    replay_certificate_filenames = ["replay.certificate.%d" % idx for idx in range(args.num_replays)]

    # Clear existing failures in file recording.
    if args.failure_filename is not None:
        with open(args.failure_filename, 'w') as f:
            f.write('')

    continue_check = True
    total_checks, total_failures, crash_fails, unsat_fails, json_fails, nondet_fails = 0, 0, 0, 0, 0, 0
    errors = {}  # map from error strings to counts, to be shown after every failure.
    while continue_check:

        # Crawl the directory of trajectories and vet ones with no certificate.
        failure_list = []
        valid_dirs = []
        for dir_name, subdir_list, file_list in os.walk(args.data_path):
            if "trial_" in dir_name and (not "raw_images" in dir_name) and (not "pddl_states" in dir_name):
                json_file = os.path.join(dir_name, JSON_FILENAME)
                if not os.path.isfile(json_file):
                    continue

                # If we're just stripping certificates, do that and continue.
                if args.remove_certificates:
                    for cidx in range(args.num_replays):
                        certificate_file = os.path.join(dir_name, replay_certificate_filenames[cidx])
                        if os.path.isfile(certificate_file):
                            os.system("rm %s" % certificate_file)
                    continue

                valid_dirs.append(dir_name)

        np.random.shuffle(valid_dirs)
        for dir_name in valid_dirs:

            if not os.path.exists(dir_name):
                continue

            json_file = os.path.join(dir_name, JSON_FILENAME)
            if not os.path.isfile(json_file):
                continue

            cidx = 0
            certificate_file = os.path.join(dir_name, replay_certificate_filenames[cidx])
            already_checked = False
            while os.path.isfile(certificate_file):
                cidx += 1
                if cidx == args.num_replays:
                    already_checked = True
                    break
                certificate_file = os.path.join(dir_name, replay_certificate_filenames[cidx])
            if already_checked:
                continue

            if not os.path.isfile(certificate_file):
                total_checks += 1. / args.num_replays
                failed = False

                with open(json_file) as f:
                    print("check %d/%d for file '%s'" % (cidx + 1, args.num_replays, json_file))
                    try:
                        traj_data = json.load(f)
                        env.set_task(traj_data, args, reward_type='dense')
                    except json.decoder.JSONDecodeError:
                        failed = True
                        json_fails += 1

                if not failed:
                    steps_taken = None
                    try:
                        steps_taken = replay_json(env, json_file)
                    except Exception as e:
                        import traceback
                        traceback.print_exc()
                        failed = True
                        crash_fails += 1

                        if str(e) not in errors:
                            errors[str(e)] = 0
                        errors[str(e)] += 1
                        print("%%%%%%%%%%")
                        es = sum([errors[er] for er in errors])
                        print("\terrors (%d):" % es)
                        for er, v in sorted(errors.items(), key=lambda kv: kv[1], reverse=True):
                            # if v / es < 0.01:  # stop showing below 1% of errors.
                            #     break
                            print("\t(%.2f) (%d)\t%s" % (v / es, v, er))
                        print("%%%%%%%%%%")

                        if cidx > 1:
                            print("WARNING: replay that has succeeded before has failed at attempt %d"
                                  % cidx)
                            nondet_fails += 1

                    if steps_taken is not None:  # executed without crashing, so now we need to verify completion.
                        goal_satisfied = env.get_goal_satisfied()

                        if goal_satisfied:
                            with open(certificate_file, 'w') as f:
                                f.write('%d' % steps_taken)
                        else:
                            failed = True
                            unsat_fails += 1
                            print("Goal was not satisfied after execution!")

                if failed:
                    # Mark one failure and count the remainder of checks for this instance into the total.
                    total_failures += 1
                    total_checks += args.num_replays - ((cidx + 1) / float(args.num_replays))

                    failure_list.append(json_file)
                    if args.failure_filename is not None:
                        with open(args.failure_filename, 'a') as f:
                            f.write("%s\n" % json_file)
                    # If we're deleting bad trajectories, do that here.
                    if args.move_failed_trajectories is not None:
                        print("Relocating failed trajectory '%s' to '%s'" %
                              (dir_name, os.path.join(args.move_failed_trajectories)))
                        try:
                            shutil.move(dir_name, args.move_failed_trajectories)
                        except shutil.Error as e:
                            print("WARNING: failed to perform move; error follows; deleting instead")
                            print(repr(e))
                            shutil.rmtree(dir_name)
                    if args.remove_failed_trajectories:
                        print("Removing failed trajectory '%s'" % dir_name)
                        shutil.rmtree(dir_name)

                print("-------------------------")
                print("Success Rate: %.2f/%.2f = %.3f" %
                      (total_checks - total_failures, total_checks,
                       float(total_checks - total_failures) / float(total_checks)))
                if total_failures > 0:
                    print("Non-deterministic failure: %d/%d = %.3f" % (nondet_fails, total_failures,
                                                                       float(nondet_fails) / total_failures))
                    print("Failures by crash: %d/%d = %.3f" % (crash_fails, total_failures,
                                                               float(crash_fails) / total_failures))
                    print("Failures by unsatisfied: %d/%d = %.3f" % (unsat_fails, total_failures,
                                                                     float(unsat_fails) / total_failures))
                    print("Failures by json decode error: %d/%d = %.3f" % (json_fails, total_failures,
                                                                           float(json_fails) / total_failures))
                print("-------------------------")

        if not args.in_parallel:
            continue_check = False
        else:
            time.sleep(60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default="dataset/2.1.0",
                        help="where to look for the generated data")
    parser.add_argument("--failure_filename", type=str, required=False,
                        help="where to write failed trajectory dirs as strings, if anywhere")
    parser.add_argument("--remove_failed_trajectories", dest='remove_failed_trajectories', action='store_true',
                        help="delete trajectory trials if they fail replay")
    parser.add_argument("--move_failed_trajectories", type=str, required=False,
                        help="if given, relocate failed trajectories to this directory")
    parser.add_argument("--remove_certificates", dest='remove_certificates', action='store_true',
                        help="instead of vetting trajectories, remove all vetting certificates")
    parser.add_argument("--in_parallel", dest='in_parallel', action='store_true',
                        help="whether to run this script with parallel generation scripts in mind")
    parser.add_argument('--reward_config', default='../models/config/rewards.json')
    parser.add_argument('--num_replays', type=int, default=2)

    args = parser.parse_args()
    replay_check(args)