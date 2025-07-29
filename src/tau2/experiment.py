import threading
import time
from queue import Empty, Queue
from random import choice

DELAY = 0.01


class Environment:
    def __init__(self, name: str):
        self.name = name
        self.actions = {"get_number": "1234", "get_balance": "1234"}
        self.num_turns = 0
        self.obs_0 = "1"

    def step(self, action: str) -> str:
        self.num_turns += 1
        time.sleep(DELAY)
        if self.num_turns == 4:
            observation = "done"
        else:
            observation = choice(self.actions[action])
        return observation

    def reset(self):
        self.num_turns = 0
        return self.obs_0


class Agent:
    def __init__(self, name: str):
        self.name = name
        self.policy = {
            "1": "get_number",
            "2": "get_balance",
            "3": "get_number",
            "4": "get_balance",
        }

    def predict(self, observation: str) -> str:
        time.sleep(DELAY)
        if observation in self.policy:
            return self.policy[observation]
        else:
            raise ValueError(f"Invalid observation: {observation}")


def run(agent: Agent, environment: Environment):
    obj = environment.reset()
    while True:
        print(f"Environment: {obj}")
        action = agent.predict(obj)
        print(f"Agent: {action}")
        obj = environment.step(action)
        if obj == "done":
            break
    return


def run_threaded(agent: Agent, environment: Environment):
    """
    Threaded version of run where agent and environment run on different threads.
    Uses queues for communication between threads.
    """
    # Queues for communication between threads
    action_queue = Queue()  # Agent -> Environment
    observation_queue = Queue()  # Environment -> Agent
    done_event = threading.Event()

    def agent_thread():
        """Agent thread that processes observations and produces actions"""
        try:
            while not done_event.is_set():
                # Wait for observation from environment
                try:
                    observation = observation_queue.get(timeout=1.0)
                    if observation == "done":
                        break

                    # Agent predicts action
                    action = agent.predict(observation)
                    print(f"Agent: {action}")

                    # Send action to environment
                    action_queue.put(action)

                except Empty:
                    continue
        except Exception as e:
            print(f"Agent thread error: {e}")
            done_event.set()

    def environment_thread():
        """Environment thread that processes actions and produces observations"""
        try:
            # Initialize environment
            obj = environment.reset()
            print(f"Environment: {obj}")

            # Send initial observation to agent
            observation_queue.put(obj)

            while not done_event.is_set():
                # Wait for action from agent
                try:
                    action = action_queue.get(timeout=1.0)

                    # Environment steps
                    obj = environment.step(action)
                    print(f"Environment: {obj}")

                    # Send observation back to agent
                    observation_queue.put(obj)

                    if obj == "done":
                        break

                except Empty:
                    continue
        except Exception as e:
            print(f"Environment thread error: {e}")
            done_event.set()

    # Create and start threads
    agent_thread_obj = threading.Thread(target=agent_thread, name="AgentThread")
    env_thread_obj = threading.Thread(
        target=environment_thread, name="EnvironmentThread"
    )

    agent_thread_obj.start()
    env_thread_obj.start()

    # Wait for both threads to complete
    agent_thread_obj.join()
    env_thread_obj.join()

    print("Threaded run completed")


if __name__ == "__main__":
    agent = Agent("agent")
    environment = Environment("environment")

    print("=== Original Run ===")
    run(agent, environment)

    print("\n=== Threaded Run ===")
    run_threaded(agent, environment)
