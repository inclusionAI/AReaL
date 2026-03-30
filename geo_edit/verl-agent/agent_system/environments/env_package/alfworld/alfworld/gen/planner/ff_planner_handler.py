import pdb
import ast
import multiprocessing
import re
import shlex
import subprocess
import time

import alfworld.gen.constants as constants
from alfworld.gen.utils import game_util
from alfworld.gen.utils import py_util

DEBUG = False

CAPS_ACTION_TO_PLAN_ACTION = {
    'GOTOLOCATION': 'GotoLocation',
    'SCAN': 'Scan',
    'OPENOBJECT': 'OpenObject',
    'CLOSEOBJECT': 'CloseObject',
    'PICKUPOBJECT': 'PickupObject',
    'PICKUPOBJECTINRECEPTACLE1': 'PickupObjectInReceptacle',
    'PICKUPOBJECTINRECEPTACLE2': 'PickupObjectInReceptacle',
    'PICKUPRECEPTACLEOBJECTINRECEPTACLE1': 'PickupObjectInReceptacle',
    'PICKUPRECEPTACLEOBJECTINRECEPTACLE2': 'PickupObjectInReceptacle',
    'PICKUPOBJECTINOBJECT1': 'PickupObjectInObject',
    'PICKUPOBJECTINOBJECT2': 'PickupObjectInObject',
    'PUTOBJECTINRECEPTACLE1': 'PutObjectInReceptacle',
    'PUTOBJECTINRECEPTACLE2': 'PutObjectInReceptacle',
    'PUTOBJECTINRECEPTACLEOBJECT1': 'PutObjectInReceptacleObject',
    'PUTOBJECTINRECEPTACLEOBJECT2': 'PutObjectInReceptacleObject',
    'PUTRECEPTACLEOBJECTINRECEPTACLE1': 'PutReceptacleObjectInReceptacle',
    'PUTRECEPTACLEOBJECTINRECEPTACLE2': 'PutReceptacleObjectInReceptacle',
    'PICKUPOBJECTNORECEPTACLE': 'PickupObjectNoReceptacle',
    'PUTOBJECT': 'PutObject',
    'CLEANOBJECT': 'CleanObject',
    'HEATOBJECT': 'HeatObject',
    'TOGGLEOBJECT': 'ToggleObject',
    'COOLOBJECT': 'CoolObject',
    'SLICEOBJECT': 'SliceObject',
    'REACH-GOAL': 'End'
}

LOWER_TO_FULL = {name.lower(): name for name in constants.OBJECTS}


def lower_to_full(input_str):
    arr = input_str.split('|')
    new_arr = []
    for item in arr:
        if item in LOWER_TO_FULL:
            new_arr.append(LOWER_TO_FULL[item])
        else:
            new_arr.append(item)
    return '|'.join(new_arr)



def parse_action_arg(action_arg):
    action_arg = action_arg.lower()
    action_arg = py_util.multireplace(action_arg,
                                      {'_minus_': '-',
                                       '-': '#',
                                       '_bar_': '|',
                                       '_plus_': '+',
                                       '_dot_': '.',
                                       '_comma_': ','})
    action_arg = lower_to_full(action_arg)
    return action_arg


def parse_line(line):
    line = re.sub(r'^\s*step|\d+:\s*', '', line)
    line = line.strip()
    line_args = line.split(' ')
    if line_args[0] not in CAPS_ACTION_TO_PLAN_ACTION:
        return None
    action = CAPS_ACTION_TO_PLAN_ACTION[line_args[0]]
    if action == 'End':
        return {'action': 'End', 'value': 1}
    action_dict = {'action': action}
    line_args = line_args[1:]  # Remove action name from line_args

    if action in {'GotoLocation', 'Scan'}:
        action_arg = line_args[2].lower()
        action_arg = py_util.multireplace(action_arg,
                                          {'_minus_': '-',
                                           '-': '#',
                                           '_bar_': '|',
                                           '_plus_': '+',
                                           '_dot_': '.',
                                           '_comma_': ','})
        action_dict['location'] = action_arg
    elif action in {'OpenObject', 'CloseObject', 'ToggleObject'}:
        action_dict['objectId'] = parse_action_arg(line_args[2])
        action_dict['receptacleObjectId'] = parse_action_arg(line_args[2])
    elif action in {'HeatObject', 'CoolObject'}:
        action_dict['receptacleObjectId'] = parse_action_arg(line_args[2])
    elif action in {'PickupObjectInReceptacle', 'PickupObjectNoReceptacle'}:
        action_dict['action'] = 'PickupObject'
        action_dict['objectId'] = parse_action_arg(line_args[2])
        if action == 'PickupObjectInReceptacle':
            action_dict['receptacleObjectId'] = parse_action_arg(line_args[3])
    elif action in {'SliceObject'}:
        action_dict['objectId'] = parse_action_arg(line_args[2])
    elif action in {'CleanObject'}:
        action_dict['objectId'] = parse_action_arg(line_args[3])
        action_dict['receptacleObjectId'] = parse_action_arg(line_args[2])
    elif action in {'PutObjectInReceptacle',
                    'PutObjectInReceptacleObject',
                    'PutReceptacleObjectInReceptacle'}:
        action_dict['action'] = 'PutObject'
        action_dict['objectId'] = parse_action_arg(line_args[3])
        action_dict['receptacleObjectId'] = parse_action_arg(line_args[4])
    elif action in {'PickupObjectInObject'}:
        action_dict['action'] = 'PickupObject'


    return action_dict


def parse_plan(lines):
    plan = []
    for line in lines:
        action_dict = parse_line(line)
        if action_dict is not None:
            plan.append(action_dict)
    return plan


def parse_plan_from_file(self, path):
    lines = [line for line in open(path)]
    return self.parse_plan(lines)


def get_plan_from_file(args):
    domain, filepath, solver_type = args

    start_t = time.time()
    try:
        command = ('ff_planner/ff '
                   '-o %s '
                   '-s %d '
                   '-f %s ' % (domain, solver_type, filepath))
        if DEBUG:
            print(command)
        planner_output = subprocess.check_output(shlex.split(command), timeout=30)
    except subprocess.CalledProcessError as error:
        # Plan is done
        output_str = error.output.decode('utf-8')
        if DEBUG:
            print('output', output_str)
        if ('goal can be simplified to FALSE' in output_str or
                "won't get here: simplify, non logical" in output_str):
            return [{'action': 'End', 'value': 0}]
        elif 'goal can be simplified to TRUE' in output_str:
            return [{'action': 'End', 'value': 1}]
        elif len(output_str) == 0:
            # Usually indicates segfault with ffplanner
            # This happens when the goal needs an object that hasn't been seen yet like
            # Q: "is there an egg in the garbage can," but no garbage can has been seen.
            print('Empty plan')
            print('Seg Fault')
            return [{'action': 'End', 'value': 0}]
        else:
            print('problem', filepath)
            print(output_str)
            print('Empty plan')
            return [{'action': 'End', 'value': 0}]
    except subprocess.TimeoutExpired:
        print('timeout solver', solver_type, 'problem', filepath)
        print('Empty plan')
        return ['timeout', {'action': 'End', 'value': 0}]
    unparsed_plan = planner_output.decode('utf-8').split('\n')
    if DEBUG:
        print('unparsed', '\n'.join(unparsed_plan))
    parsed_plan = parse_plan(unparsed_plan)
    if constants.DEBUG:
        print('planned %s in %.5f, plan length %d solver type %d' % (
            filepath, time.time() - start_t, len(parsed_plan), solver_type))
    if len(parsed_plan) == 0:
        parsed_plan = [{'action': 'End', 'value': 1}]
    return parsed_plan


# Example of how to call ff
# /path/to/Metric-FF-v2.1/ff -o planner/domains/Question_domain.pddl -f planner/exists_problem.pddl
def get_plan_async(args):
    domain, problem_id, solver_type = args
    filepath = '%s/planner/generated_problems/problem_%s.pddl' % (constants.LOG_FILE, problem_id)
    return get_plan_from_file((domain, filepath, solver_type))


class PlanParser(object):
    def __init__(self, domain_file_path):
        self.domain = domain_file_path
        self.problem_id = -1
        self.process_pool = multiprocessing.Pool(3)
        #from multiprocessing.pool import ThreadPool
        #self.process_pool = ThreadPool(3)

    def get_plan(self):
        parsed_plans = self.process_pool.map(get_plan_async, zip([self.domain] * 3, [self.problem_id] * 3, range(3, 6)))
        return self.find_best_plan(parsed_plans)

    def get_plan_from_file(self, domain_path, filepath):
        parsed_plans = self.process_pool.map(get_plan_from_file, zip([domain_path] * 3, [filepath] * 3, range(3, 6)))
        return self.find_best_plan(parsed_plans)

    # Unncessary, planner should be optimal. But the planner produces some weird actions
    def clean_plan(self, plan):
        cleaned_plan = list()
        for i in range(len(plan)-1):
            if not (plan[i]['action'] == 'GotoLocation' and plan[i+1]['action'] == 'GotoLocation'):
                cleaned_plan.append(plan[i])
        cleaned_plan.append(plan[len(plan)-1])
        return cleaned_plan

    def find_best_plan(self, parsed_plans):

        if all([parsed_plan[0] == 'timeout' for parsed_plan in parsed_plans]):
            parsed_plan = parsed_plans[0][1:]
        else:
            parsed_plans = [self.clean_plan(parsed_plan) for parsed_plan in parsed_plans if parsed_plan[0] != 'timeout']
            parsed_plan = min(parsed_plans, key=len)

        if constants.DEBUG:
            print('plan\n' + '\n'.join(['%03d: %s' % (pp, game_util.get_action_str(pl))
                                        for pp, pl in enumerate(parsed_plan)]))
        else:
            print('plan\n' + '\n'.join(['%03d: %s' % (pp, game_util.get_action_str(pl))
                                        for pp, pl in enumerate(parsed_plan)]))
        return parsed_plan


class SinglePlanParser(PlanParser):
    def get_plan(self):
        parsed_plan = get_plan_async([self.domain, self.problem_id, 3])
        return parsed_plan

    def get_plan_from_file(self, domain_path, filepath):
        parsed_plan = get_plan_from_file([domain_path, filepath, 3])
        return parsed_plan


if __name__ == '__main__':
    import sys

    DEBUG = constants.DEBUG
    parser = PlanParser('planner/domains/PutTaskExtended_domain.pddl')
    parser.problem_id = sys.argv[1]
    result_plan = parser.get_plan()
    print('plan\n' + '\n'.join(['%03d: %s' % (pp, game_util.get_action_str(pl)) for pp, pl in enumerate(result_plan)]))
