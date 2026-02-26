import json
import random
import zipfile
from pathlib import Path
from time import sleep

from browsergym.core.env import BrowserEnv
from browsergym.experiments.benchmark import DEFAULT_BENCHMARKS
from browsergym.workarena import ATOMIC_TASKS, LIST_TASKS, NAVIGATION_TASKS
from browsergym.workarena.tasks.navigation import AllMenuTask

# random.shuffle(LIST_TASKS)
# for task in LIST_TASKS:

TRACE_DIR = Path("cheat_traces")
TRACE_DIR.mkdir(exist_ok=True)

print("Task:", AllMenuTask)

# Instantiate a new environment
env = BrowserEnv(task_entrypoint=AllMenuTask, headless=False)
env.reset(seed=769)

# Start Playwright tracing before the cheat
env.context.tracing.start(screenshots=True, snapshots=True, sources=True)

# Cheat functions use Playwright to automatically solve the task
env.chat.add_message(role="assistant", msg="On it. Please wait...")
cheat_messages = []
env.task.cheat(env.page, cheat_messages)

# Stop tracing and save the trace file
trace_path = TRACE_DIR / "cheat_trace.zip"
env.context.tracing.stop(path=str(trace_path))
print(f"\nTrace saved to: {trace_path}")

# Extract and display the actions from the trace
def extract_actions_from_trace(trace_zip_path: Path) -> list[dict]:
    """Parse a Playwright trace zip and return a list of user actions."""
    actions = []
    with zipfile.ZipFile(trace_zip_path, "r") as zf:
        for name in zf.namelist():
            if not name.endswith(".trace"):
                continue
            for line in zf.read(name).decode("utf-8").splitlines():
                entry = json.loads(line)
                if entry.get("type") == "action":
                    call = entry.get("callMetadata", {})
                    actions.append(
                        {
                            "api": call.get("apiName", ""),
                            "params": call.get("params", {}),
                            "selector": call.get("params", {}).get("selector", ""),
                            "wallTime": call.get("wallTime", 0),
                        }
                    )
    return actions

actions = extract_actions_from_trace(trace_path)
print(f"\n{'='*60}")
print(f"Cheat actions ({len(actions)} total):")
print(f"{'='*60}")
for i, action in enumerate(actions, 1):
    api = action["api"]
    selector = action["selector"]
    params = {k: v for k, v in action["params"].items() if k != "selector"}
    parts = [f"  {i}. {api}"]
    if selector:
        parts.append(f"     selector: {selector}")
    if params:
        parts.append(f"     params:   {params}")
    print("\n".join(parts))

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

print(f"\nReward: {reward}")

sleep(3)
env.close()
