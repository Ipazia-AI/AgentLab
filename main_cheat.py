import random
from time import sleep

from browsergym.core.env import BrowserEnv
from browsergym.experiments.benchmark import DEFAULT_BENCHMARKS
from browsergym.workarena import ATOMIC_TASKS, LIST_TASKS, NAVIGATION_TASKS
from browsergym.workarena.tasks.navigation import AllMenuTask

# random.shuffle(LIST_TASKS)
# for task in LIST_TASKS:

print("Task:", AllMenuTask)

# Instantiate a new environment
env = BrowserEnv(task_entrypoint=AllMenuTask, headless=False)
env.reset(seed=769)

# Cheat functions use Playwright to automatically solve the task
env.chat.add_message(role="assistant", msg="On it. Please wait...")
cheat_messages = []
env.task.cheat(env.page, cheat_messages)

# Send cheat messages to chat
for cheat_msg in cheat_messages:
    env.chat.add_message(role=cheat_msg["role"], msg=cheat_msg["message"])

# Post solution to chat
env.chat.add_message(role="assistant", msg="I'm done!")

# Validate the solution
reward, stop, message, info = env.task.validate(env.page, cheat_messages)
if reward == 1:
    env.chat.add_message(role="user", msg="Yes, that works. Thanks!")
else:
    env.chat.add_message(role="user", msg=f"No, that doesn't work. {info.get('message', '')}")

sleep(3)
env.close()
