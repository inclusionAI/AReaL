import random
import re
from typing import Dict, List, Tuple

from realhf.api.core.env_api import EnvironmentService, register_environment
from realhf.base import logging

logger = logging.getLogger("WerewolfEnv")
LOG_DEBUG = False

def logutil(msg):
    if LOG_DEBUG:
        logger.info(msg)
    else:
        logger.debug(msg)

class WerewolfEnv(EnvironmentService):
    """A simple multi-turn environment for the werewolf game.

    Parameters allow customising the number of players for each role.  The
    environment will automatically create unique player names such as
    ``player1`` or ``player2`` when more than one player exists for a
    role.
    """

    def __init__(
        self,
        repeat_rules: bool = True,
        num_villagers: int = 2,
        num_witches: int = 1,
        num_foreseers: int = 1,
        num_werewolves: int = 2,
        num_hunters: int = 1,
    ):
        self.num_players = {
            "villager": num_villagers,
            "witch": num_witches,
            "foreseer": num_foreseers,
            "werewolf": num_werewolves,
            "hunter": num_hunters,
        }
        assert (num_villagers != 0) and (num_werewolves != 0), "The number of villagers and/or werewolves cannot be 0."

        self.roles: List[str] = []
        self.role_type: Dict[str, str] = {}
        self.agent_player = "player1"
        self.agent_role = "villager"
        self.turn_order: List[str] = []
        self.current_agent_idx = 0
        self.phase_player_list: List[str] = []  # Records the players who shall act in current phase
        self.phase_actions: Dict[str, str] = {}  # Records the actions performed in one phase
        self.hunter_player = None  # Records who is the hunter

        self.alive: Dict[str, bool] = {}
        self.round = 0
        self.phase = "night"  # night, discussion, day or hunter
        self.witch_heal = True
        self.witch_poison = True
        self.await_hunter = False
        self.player_memory: Dict[str, List[str]] = {}
        self.repeat_rules = repeat_rules
        self.rules = (
            "You are playing the werewolf game."
            "Roles in the game are villager, witch, foreseer, werewolf, and hunter. "
            "Werewolves kill at night. The witch can heal or poison once each every game. "
            "The foreseer checks one role each night. The hunter shoots one player "
            "when killed. Players vote during the day to eliminate one suspect. "
            "Villagers win if all werewolves die; werewolves win if they kill "
            "everyone else."
        )
        self.role_prompts = {
            "villager": "You are a villager. Find out the werewolves and vote them out.",
            "werewolf": "You are a werewolf. Work with the other werewolves to kill the rest.",
            "witch": "You are the witch. You may heal or poison during the night. You can do each action only once in a single game.",
            "foreseer": "You are the foreseer. Each night you can check one player's role.",
            "hunter": "You are the hunter. When killed, you may shoot one player.",
        }
        self.guide = (
            "Decide your action for this round. Format your response in the following format: "
            "\"<think>your reasoning</think> <answer>action + target</answer>\""
            # " Example: <think>I am a werewolf. I suspect player0 is a villager, I shall kill him to win.</think> <answer>kill player0</answer>"
            "Your valid actions are: {actions}. Perform one action only. Do not say anything else."
        )
        # Stats for RL training
        self.stats: Dict[str, int] = {
            "vill_wins": 0,
            "were_wins": 0,
            "werewolf_kills": 0,
            "werewolf_correct_kills": 0,
            "villager_correct_votes": 0,
            "villager_wrong_votes": 0,
            "witch_heals": 0,
            "witch_correct_heals": 0,
            "witch_poisons": 0,
            "witch_correct_poisons": 0,
            "hunter_shots": 0,
            "hunter_correct_shots": 0,
        }

        logutil(f"Game initialized with roles: {self.num_players}.")

    def _setup_players(self) -> None:
        """Create player names and role mapping based on ``self.num_players``."""
        self.roles = []
        self.role_type = {}
        idx = 1
        for role, num in self.num_players.items():
            for _ in range(num):
                name = f"player{idx}"
                idx += 1
                self.roles.append(name)
                self.role_type[name] = role

                logutil(f"Initialized a player {name} with role {role}.")

    async def reset(self, seed=None, options=None):
        if seed is not None:
            random.seed(seed)
        self._setup_players()
        self.alive = {r: True for r in self.roles}
        self.turn_order = self.roles[:]
        self.phase_actions = {}
        self.hunter_player = None
        self.player_memory = {p: [] for p in self.roles}
        self.stats: Dict[str, int] = {
            "vill_wins": 0,
            "were_wins": 0,
            "werewolf_kills": 0,
            "werewolf_correct_kills": 0,
            "villager_correct_votes": 0,
            "villager_wrong_votes": 0,
            "witch_heals": 0,
            "witch_correct_heals": 0,
            "witch_poisons": 0,
            "witch_correct_poisons": 0,
            "hunter_shots": 0,
            "hunter_correct_shots": 0,
        }

        # game always starts at night with werewolves acting first
        self.phase = "night"
        self.phase_player_list = self._phase_players("night")
        self.current_agent_idx = 0
        self.agent_player = self.phase_player_list[0]
        self.agent_role = self.role_type[self.agent_player]

        logutil(f"Game reset with alive players {self.alive}, current agent is {self.agent_player} ({self.agent_role}).")

        self.round = 1
        self.phase = "night"
        self.witch_heal = True
        self.witch_poison = True
        self.await_hunter = False
        guide = self.guide.format(actions=', '.join(self._get_valid_actions()))
        role_prompt = self.role_prompts.get(self.agent_role, "")
        obs = (
            f"{self.rules} You are the {self.agent_player} ({self.agent_role}). {role_prompt} {guide}"
            f" Game start. Night {self.round}. Alive players: {', '.join(self.roles)}."
        )
        return obs, {}

    def _format_reward(self, text: str) -> float:
        has_think = "<think>" in text and "</think>" in text
        m = re.findall(r"<answer>(.*?)</answer>", text, re.DOTALL)
        if len(m) >= 1:
            return 0.1 if has_think else 0.05
        return 0.05 if has_think else 0.0

    def _action_reward(self, role: str, action: str) -> List[float]:
        """Assign rewards for meaningful actions."""
        reward = [0.0, 0.0]
        if not action:
            return reward

        if action.startswith("vote "):
            target = action.split("vote ")[1].strip()
            if self.role_type.get(target) == "werewolf" and role == "villager":
                reward[0] += 1.0
            elif self.role_type.get(target) != "werewolf" and role == "werewolf":
                reward[1] += 0.5

        if action.startswith("poison ") and role == "witch":
            target = action.split("poison ")[1].strip()
            if self.role_type.get(target) == "werewolf":
                reward[0] += 1.0
            else:
                reward[0] -= 1.0

        if action.startswith("save ") and role == "witch":
            target = action.split("save ")[1].strip()
            if self.role_type.get(target) != "werewolf":
                reward[0] += 0.7
            else:
                reward[0] -= 1.0

        if action.startswith("kill ") and role == "werewolf":
            target = action.split("kill ")[1].strip()
            if self.role_type.get(target) != "werewolf":
                reward[1] += 1.0
            else:
                reward[1] -= 0.5

        if action.startswith("check ") and role == "foreseer":
            target = action.split("check ")[1].strip()
            if self.role_type.get(target) == "werewolf":
                reward[0] += 1.0

        if action.startswith("shoot ") and role == "hunter":
            target = action.split("shoot ")[1].strip()
            if self.role_type.get(target) == "werewolf":
                reward[0] += 1.0
            else:
                reward[0] -= 0.7

        return reward

    def _add_memory(self, player: str, text:str) -> None:
        if player not in self.player_memory:
            self.player_memory[player] = []
        self.player_memory[player].append(text)

    def _check_win(self):
        werewolves_alive = [p for p in self.roles if self.role_type[p] == "werewolf" and self.alive[p]]
        others_alive = [p for p in self.roles if self.role_type[p] != "werewolf" and self.alive[p]]
        if not werewolves_alive:
            logutil(f"Confirm villagers win.")
            return "villagers"
        if not others_alive:
            logutil(f"Confirm werewolves win.")
            return "werewolf"
        return None

    def _apply_kill(self, target: str, reason: str = "killed") -> str:
        msg = ""
        if target and self.alive.get(target, False):
            self.alive[target] = False
            msg = f"{target} died."
            logutil(f"Player {target} is {reason}.")

            if self.role_type.get(target) == "hunter":
                logutil(f"{target} is a hunter. Now the hunter will shoot another player.")
                self.await_hunter = True
                self.hunter_player = target
        return msg

    def _alive_list(self) -> List[str]:
        return [r for r in self.roles if self.alive[r]]

    def _phase_players(self, phase: str) -> List[str]:
        """Return the list of players who act in the given phase in turn order."""
        alive = self._alive_list()
        if phase == "night":
            players = []
            for p in self.turn_order:
                if p in alive and self.role_type[p] == "werewolf":
                    players.append(p)
            for p in self.turn_order:
                if p in alive and self.role_type[p] in {"witch", "foreseer"}:
                    players.append(p)
            return players
        if phase == "hunter":
            return [self.hunter_player] if self.hunter_player else []
        # discussion or day
        return [p for p in self.turn_order if p in alive]

    def _next_agent(self) -> None:
        """Advance ``self.agent_player`` to the next player who should act."""
        if not self.phase_player_list:
            self.agent_player = None
            self.agent_role = None
            return

        idx = len(self.phase_actions)
        while idx < len(self.phase_player_list):
            candidate = self.phase_player_list[idx]
            role = self.role_type[candidate]
            if not (
                self.phase == "night"
                and role == "witch"
                and not self.witch_heal
                and not self.witch_poison
            ):
                self.agent_player = candidate
                self.agent_role = role
                return
            # skip witch with no potions
            self.phase_actions[candidate] = "wait"
            idx += 1

        self.agent_player = None
        self.agent_role = None

    def _get_valid_actions(self) -> List[str]:
        actions = []
        alive = self.alive.get(self.agent_player, False)
        all_players = self.roles
        players = self._alive_list()
        if self.phase == "night":
            if self.agent_role == "werewolf" and alive:
                actions.extend(
                    [f"kill {p}" for p in players if p != self.agent_player and self.role_type[p] != "werewolf"]
                )
            if self.agent_role == "witch" and alive:
                if self.witch_poison:
                    actions.extend([f"poison {p}" for p in players if p != self.agent_player])
                if self.witch_heal:
                    actions.extend([f"save {p}" for p in all_players if not self.alive[p]])
            if self.agent_role == "foreseer" and alive:
                actions.extend([f"check {p}" for p in players if p != self.agent_player])
            if not actions:
                actions.append("wait")
        elif self.phase == "discussion":
            actions.append("say 'text'")
        elif self.phase == "day":
            actions.extend([f"vote {p}" for p in players if p != self.agent_player])
        elif self.phase == "hunter":
            actions.extend([f"shoot {p}" for p in players if p != self.agent_player])

        return actions

    def _discussion_phase(self, actions: Dict[str, str], players: List[str]) -> str:
        msg = ""
        for i, p in enumerate(players):
            content = actions.get(p, "nothing")
            if content.startswith("say "):
                content = content[4:]
            if i == 0:
                msg += f"{p} says: {content}."
            else:
                msg += f"{p} says {content}."
        return msg

    async def _change_phase(self, last_role: str):
        """Process allocated actions and change to next pahse"""
        info = ""
        reward = [0.0, 0.0]
        done = False
        winner = ""

        players = self.phase_player_list
        actions = self.phase_actions

        if self.phase == "night":
            info += self._night_phase(actions, players)
            self.phase = "discussion"
            info += f"Night ends. Alive players: {', '.join(self._alive_list())}. Discussion begins."
        elif self.phase == "discussion":
            info += self._discussion_phase(actions, players)
            self.phase = "day"
            info += f"Discussion over. Please vote."
        elif self.phase == "day":
            info += self._day_phase(actions, players)
            winner = self._check_win()
            if winner:
                done = True
                if winner == "werewolf":
                    self.stats["were_wins"] += 1
                    reward[1] += 3.0
                else:
                    self.stats["vill_wins"] += 1
                    reward[0] += 3.0
                info += f"Game over. {winner} win."
            else:
                # Check if hunter shall act
                if self.await_hunter:
                    self.phase = "hunter"
                    info += "The hunter is dead. He shall choose a player to shoot."
                else:
                    self.round += 1
                    self.phase = "night"
                    info += f"Day ends. Night {self.round}."
        elif self.phase == "hunter":
            shoot_act = actions.get(players[0], "") if players else ""
            if shoot_act.startswith("shoot "):
                self.stats["hunter_shots"] += 1 
                target = shoot_act.split("shoot ")[1].strip()
                if self.alive.get(target, False) and self.role_type[target] == "werewolf":
                    self.stats["hunter_correct_shots"] += 1
                info += self._apply_kill(target, "shot")
            self.await_hunter = False
            winner = self._check_win()
            if winner:
                done = True
                if winner == "werewolf":
                    self.stats["were_wins"] += 1
                    reward[1] += 3.0
                else:
                    self.stats["vill_wins"] += 1
                    reward[0] += 3.0
                info += f"Game over. {winner} win."
            else:
                self.phase = "night"
                self.round += 1
                info += f"Day ends. Night {self.round}."
        
        if done:
            logger.warning(f"Game successfully ends with winner {winner}!")
        else:
            self.phase_actions = {}
            self.phase_player_list = self._phase_players(self.phase)
            if self.phase_player_list:
                self.current_agent_idx = 0
                self.agent_player = self.phase_player_list[0]
                self.agent_role = self.role_type[self.agent_player]
            else:
                logger.error(f"Game ends abruptly with no winners and no players." , exc_info=True)
                done = True

        logutil(info)
        return info, reward, done

    async def step(self, action: Tuple[str, List[str]]):
        qid, acts = action
        text = acts[0] if isinstance(acts, list) and acts else ""
        reward = [self._format_reward(text), self._format_reward(text)] # Villager, werewolves
        m = re.findall(r"<answer>(.*?)</answer>", text, re.DOTALL)
        ans = m[-1].strip().lower() if m else ""

        # record action for current agent
        # logutil(f"{self.agent_player} ({self.agent_role}) sends response: {(text[-100:] + "(Truncated)") if len(text) > 100 else text}, parsed act is: {ans}.")
        if ans.startswith("say ") or ans in self._get_valid_actions():
            self.phase_actions[self.agent_player] = ans
            reward = [reward[i] + self._action_reward(self.agent_role, ans)[i] for i in range(2)]
        elif self.phase == "discussion" and ans:
            # Treat free-form text as valid speech
            self.phase_actions[self.agent_player] = f"say {ans}"
        else:
            logutil(f"{self.agent_player} ({self.agent_role}) sended an invalid action, skipping this turn.")
            self.phase_actions[self.agent_player] = f"wait"
        info = ""
        done = False

        # check if phase finished
        if len(self.phase_actions) >= len(self.phase_player_list):
            info, extra_reward, done = await self._change_phase(self.agent_role)
            reward = [reward[i] + extra_reward[i] for i in range(2)]
        else:
            info = f"It is now phase {self.phase} round {self.round}."
            self._next_agent()

        guide = self.guide.format(actions=', '.join(self._get_valid_actions()))
        role_prompt = self.role_prompts.get(self.agent_role, "")
        memory = "; ".join(self.player_memory.get(self.agent_player, []))

        obs = ""
        if self.repeat_rules:
            obs += f"{self.rules} You are {self.agent_player} ({self.agent_role}). {role_prompt}"
        if memory:
            obs += f" You remember: {memory}"
        obs += f" {info} {guide}"
        
        return obs, reward, done, False, {}

    def _night_phase(self, actions: Dict[str, str], players: List[str]) -> str:
        msg = ""

        kill_target = None
        for p in players:
            if self.role_type[p] == "werewolf" and actions.get(p, "").startswith("kill "):
                target = actions.get(p, "").split("kill ")[1].strip()
                if target != p and self.role_type.get(target) != "werewolf" and self.alive.get(target, False):
                    kill_target = target
                    break

        if kill_target is None and any(self.alive[p] for p in players if self.role_type[p] == "werewolf"):
            candidates = [r for r in players if self.role_type[r] != "werewolf" and self.alive[r]]
            if candidates:
                kill_target = random.choice(candidates)

        for p in players:
            if self.role_type[p] == "witch":
                act = actions.get(p, "")
                if act.startswith("poison ") and self.witch_poison:
                    target = act.split("poison ")[1].strip()
                    if target in self.alive and self.alive[target]:
                        msg += self._apply_kill(target, "poisoned")
                        self.witch_poison = False # The witch can only poison once per game.
                        self.stats["witch_poisons"] += 1
                        self._add_memory(p, f"You poisoned {target}.")
                        if self.role_type[target] == "werewolf":
                            self.stats["witch_correct_poisons"] += 1
                elif act.startswith("save ") and self.witch_heal:
                    heal_target = act.split("save ")[1].strip()
                    self.witch_heal = False
                    self.stats["witch_heals"] += 1
                    if heal_target in self.alive:
                        if not self.alive[heal_target]:
                            self.alive[heal_target] = True
                            msg += f"Witch used heal on {heal_target}."
                            self._add_memory(p, f"You healed {heal_target}.")
                            if self.role_type[heal_target] != "werewolf":
                                self.stats["witch_correct_heals"] += 1
                        else:
                            self._add_memory(p, f"You used heal potion on {heal_target}, to no effect.")
                            msg += "Witch used heal potion, to no effect."

        if kill_target:
            self.stats["werewolf_kills"] += 1
            if self.role_type[kill_target] != "werewolf":
                self.stats["werewolf_correct_kills"] += 1
            msg += self._apply_kill(kill_target, "killed")
            for p in players:
                if self.role_type[p] == "werewolf" and actions.get(p, "").startswith("kill "):
                    target = actions.get(p, "").split("kill ")[1].strip()
                    if target == kill_target:
                        self._add_memory(p, f"You killed {kill_target}.")
                elif self.role_type[p] == "werewolf":
                    self._add_memory(p, f"{kill_target} is killed by another werewolf.")
                else:
                    self._add_memory(p, f"{kill_target} is killed during the night.")

        for p in players:
            if self.role_type[p] == "foreseer" and actions.get(p, "").startswith("check "):
                target = actions.get(p, "").split("check ")[1].strip()
                if target in self.alive:
                    role = self.role_type.get(target)
                    if role == "werewolf":
                        msg += f"{target} is werewolf. "
                    else:
                        msg += f"{target} is not werewolf. "
                    self._add_memory(p, f"You checked {target}, who is a {role}.")

        logutil(msg)
        return msg

    def get_stats(self) -> Dict[str, int]:
        """Returns a copy of the current game stats"""
        return dict(self.stats)

    def _day_phase(self, actions: Dict[str, str], players: List[str]) -> str:
        msg = ""
        votes = []
        for p in players:
            act = actions.get(p, "").lower()
            if act.startswith("vote "):
                vote_target = act.split("vote ")[1].strip()
            else:
                choices = [x for x in players if x != p]
                vote_target = random.choice(choices) if choices else p
            votes.append(vote_target)
            if self.role_type.get(p) == "villager":
                if self.role_type.get(vote_target) == "werewolf":
                    self.stats["villager_correct_votes"] += 1
                else:
                    self.stats["villager_wrong_votes"] += 1

        tallies = {}
        for v in votes:
            tallies[v] = tallies.get(v, 0) + 1
        
        if len(tallies) > 0:
            target = max(tallies.items(), key=lambda x: x[1])[0]
            msg += self._apply_kill(target, "voted out")
            for p in players:
                if p != target:
                    self._add_memory(p, f"{target} is voted out in day {self.round}.")
        else:
            msg += f"No player is voted out in day {self.round} since there are no valid votes."

        logutil(msg)
        return msg

register_environment("werewolf_env", WerewolfEnv)