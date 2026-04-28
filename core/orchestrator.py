from modules.observe import observe
from modules.plan import plan   
from modules.execute import execute
from modules.verify import verify
from modules.update import update


def run_cycle(state):

    state = observe(state)

    state = plan(state)

    state = execute(state)

    state = verify(state)

    state = update(state)

    return state


def run_fixed(state):
    # evaluation mode (no planner)
    state = execute(state)
    state = verify(state)
    state = update(state)
    return state