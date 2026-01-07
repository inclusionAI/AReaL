import subprocess
import re

import numpy as np


class CPPSolver:
    solver_path: str = (
        "/storage/openpsi/users/wanghuaijie.whj/jwhj/AReaL-multitask-agent/multitask_agent/gem_train/minesweeper/dfs"
    )

    def __init__(self, solver_path: str | None = None):
        if solver_path is not None:
            self.solver_path = solver_path

    def parse(self, n: int, m: int, obs: str):
        board = np.zeros((n, m))
        tokens = obs.split()
        tokens = tokens[m:]
        for i in range(n):
            for j in range(m):
                tmp = tokens[i * (m + 1) + j + 1]
                if tmp == "." or tmp == "F":
                    board[i, j] = -1
                else:
                    board[i, j] = int(tmp)
        return board

    def solve(self, n: int, m: int, obs: str):
        matching = re.search(
            r"Here is the current board layout:\n(.*)Enter your guess", obs, re.S
        )
        board_str = matching.group(1)
        board = self.parse(n, m, board_str)
        if np.all(board == -1):
            return np.zeros((n, m))
        input_str = ("{} {}\n" "{}").format(n, m, board_str)
        result = subprocess.run(
            [self.solver_path],
            input=input_str,
            capture_output=True,
            text=True,
        )
        tmp = result.stdout.split()
        probs: np.ndarray = np.zeros((n, m))
        for i in range(n):
            for j in range(m):
                # if tmp[i * m + j] != "-" and tmp[i * m + j] != "?":
                #     probs[i, j] = float(tmp[i * m + j])
                # else:
                #     probs[i, j] = float("nan")
                if tmp[i * m + j] == "-":
                    probs[i, j] = float("nan")
                elif tmp[i * m + j] == "?":
                    probs[i, j] = 0.5
                else:
                    probs[i, j] = float(tmp[i * m + j])
        return probs

    def act(self, n: int, m: int, obs: str):
        probs = self.solve(n, m, obs)
        min_prob = float("inf")
        choices = []
        for i in range(n):
            for j in range(m):
                if not np.isnan(probs[i, j]):
                    if probs[i, j] < min_prob:
                        min_prob = probs[i, j]
                        choices = [(i, j)]
                    elif probs[i, j] == min_prob:
                        choices.append((i, j))
        x, y = choices[np.random.choice(list(range(len(choices))))]
        return f"<think>\n</think>\n\n\\boxed{{reveal {x} {y}}}"

    def compute_reward(
        self, n: int, m: int, obs: str, action: str, debug: bool = False
    ):
        probs = self.solve(n, m, obs)

        action_search_pattern = re.compile(
            r"\\boxed{([a-zA-Z]+)\s(\d+)\s(\d+)}"
        )  # e.g. \\boxed{reveal 3 2}
        matches = list(action_search_pattern.finditer(action))
        clean_action = matches[-1] if matches else None
        try:
            action_type = clean_action.group(1).lower() if clean_action else None
            row = int(clean_action.group(2)) if clean_action else None
            col = int(clean_action.group(3)) if clean_action else None
        except Exception:
            action_type, row, col = None, None, None

        if (
            action_type == "flag"
            or row is None
            or row >= n
            or col is None
            or col >= m
            or np.isnan(probs[row, col])
        ):
            if debug:
                print(f"debug======{action_type} {row} {col} {obs}\n{probs}")
            return 0

        optimal_prob = np.min([x for x in probs.flatten().tolist() if not np.isnan(x)])
        if optimal_prob >= 1.0:
            print(f"debug====={optimal_prob} {obs}\n{probs}")

        # return np.log(1 - probs[row, col] + 1e-4) - np.log((1 - optimal_prob) * 0.8)
        return optimal_prob - probs[row, col]


if __name__ == "__main__":
    solver = CPPSolver()
    __import__("pdb").set_trace()
    solver.solve(
        8,
        8,
        """At turn 24, you successfully revealed cell (6, 0).Here is the current board layout:
    0  1  2  3  4  5  6  7                                                                                                                                                                                                     
 0  0  0  0  1  1  3  .  2                                                                                                                                                                                                     
 1  1  1  1  1  .  3  .  2                                                                                                                                                                                                     
 2  2  .  1  1  1  2  1  1                                                                                                                                                                                                     
 3  .  4  2  1  0  0  0  0                                                                                                                                                                                                     
 4  .  4  .  1  0  0  0  0
 5  3  .  3  2  2  2  1  0
 6  2  .  3  2  .  .  1  0
 7  .  1  .  .  .  2  1  0

Enter your guess.""",
    )
