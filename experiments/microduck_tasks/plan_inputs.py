"""One observable request contract for teacher collection and deployed skills."""

import copy
import hashlib
from pathlib import Path

from .plan_protocol import CATALOG, VERSION, entities, validate_input
from .skill_vlm import SkillInput

BOUNDARY_CONTRACT = "microduck-skill-boundary-1"
LEGACY_CONTRACT = "microduck-primitive-relabel-1"


def contract_digest():
    root = Path(__file__).parent
    names = ("plan_inputs.py", "plan_tasks.py", "selection_plan.py")
    return hashlib.sha256(b"\0".join((root / name).read_bytes() for name in names)).hexdigest()


def request(instruction, observation, images, task_id, started, outcomes, committed=()):
    state = {key: observation[key] for key in ("yaw", "stationary", "tilt_deg")}
    state.update(
        entities=entities(observation),
        observation_id=observation["frame_id"],
        task_elapsed_s=observation["time_s"] - started,
        available_skills=CATALOG,
        schema_version=VERSION,
    )
    context = dict(task_id=task_id, committed_steps=list(committed), recent_outcomes=outcomes[-6:])
    validate_input(state, context)
    return SkillInput(instruction, images, copy.deepcopy(state), copy.deepcopy(context))


def feedback(step, outcome, target):
    """Report the submitted semantic skill, including early exits and refusals."""
    return copy.deepcopy(dict(step=step, outcome=outcome, target=target))
