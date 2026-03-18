# llm_tasks/flowchart_dot.py

"""
DOT flowchart code generation.
Two approaches:
1. From transcript (audio pipeline) — uses meeting_compact
2. From PDD steps (video pipeline) — deterministic step-to-node mapping

Enhanced classification for decisions, loops, and parallel paths.
All labels enforced to max configured words.
"""

import re
import time
from typing import Dict, List, Optional, Tuple

from core.gemini_client import gemini_client
from core.config import config
from core.utils import timed, filter_conversation_steps
from llm_tasks.system_prompts import get_system_prompt
from llm_tasks.requirements import get_interface_requirements
from llm_tasks.entity_extraction import extract_entities_and_project


# ============================================================
# Step Classification (Enhanced)
# ============================================================

DECISION_KEYWORDS = [
    'if ', 'whether', 'check if', 'verify if', 'determine if',
    'validate if', 'confirm if', 'eligible', 'meets criteria',
    'qualifies', 'is valid', 'is active', 'is inactive',
    'found', 'not found', 'exists', 'does not exist',
    'successful', 'failed', 'pass', 'fail', 'approved', 'rejected',
    'otherwise', 'condition', 'based on', 'depending on',
    'disabled', 'enabled', 'blank', 'empty', 'missing',
]

LOOP_KEYWORDS = [
    'each', 'every', 'all ', 'iterate', 'repeat', 'loop',
    'next record', 'next item', 'next entry', 'next user',
    'for all', 'processes each', 'validates each',
    'for every', 'across all', 'one by one',
    'repeats step', 'repeats the',
]

END_PHASE_KEYWORDS = [
    'final report', 'final status', 'execution status',
    'completion', 'summary report', 'audit log',
    'logs out', 'closes', 'terminates', 'ends',
    'shares the', 'saves the updated', 'saves report',
]

DATA_OPERATION_KEYWORDS = [
    'filter', 'remove duplicate', 'extract', 'export',
    'download', 'consolidate', 'clean', 'merge',
    'apply filter', 'remove blank', 'deduplicate',
    'open file', 'open csv', 'open excel',
]


def classify_steps(steps: List[str]) -> List[Dict]:
    """Classify each step as PROCESS, DECISION, LOOP, DATA_OP, or END_PHASE."""
    classified = []

    for i, step in enumerate(steps):
        step_lower = step.lower()
        step_type = "PROCESS"

        if any(kw in step_lower for kw in DECISION_KEYWORDS):
            step_type = "DECISION"
        elif any(kw in step_lower for kw in LOOP_KEYWORDS):
            step_type = "LOOP"
        elif any(kw in step_lower for kw in END_PHASE_KEYWORDS):
            step_type = "END_PHASE"
        elif any(kw in step_lower for kw in DATA_OPERATION_KEYWORDS):
            step_type = "DATA_OP"

        short_label = _shorten_label(step)
        classified.append({
            "index": i + 1,
            "text": step,
            "type": step_type,
            "short_label": short_label
        })

    type_counts = {}
    for c in classified:
        type_counts[c["type"]] = type_counts.get(c["type"], 0) + 1
    print(f"    [Classify] {len(classified)} steps: {type_counts}")

    return classified


def _shorten_label(text: str, max_words: int = None) -> str:
    """Shorten step text to a flowchart-friendly label."""
    max_words = max_words or config.flowchart.max_label_words

    text = re.sub(
        r'^(the system|the automation|the bot|the solution|the process|it)\s+',
        '', text, flags=re.IGNORECASE
    ).strip()

    text = re.sub(
        r'\b(the|a|an|to|of|for|in|on|at|by|with|using|based|upon|into)\b',
        ' ', text, flags=re.IGNORECASE
    )
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.rstrip('.')

    if text:
        text = text[0].upper() + text[1:]

    words = text.split()
    if len(words) > max_words:
        text = ' '.join(words[:max_words])

    return text


# ============================================================
# Audio Pipeline: From Transcript
# ============================================================

def generate_dot_and_apps(
    process_steps: List[str],
    transcript: str,
    entities: Dict = None
) -> Tuple[str, List[Dict]]:
    """Generate DOT flowchart code and interface requirements from transcript."""
    start = time.time()
    if entities is None:
        entities, _ = extract_entities_and_project(transcript)
    dot_code = _generate_dot_from_steps_list(process_steps)
    apps = get_interface_requirements(transcript, entities)
    timed("DOT+Apps", start)
    return dot_code, apps


def _generate_dot_from_steps_list(steps: List[str]) -> str:
    """Generate DOT code from a list of step strings."""
    if not steps:
        return _deterministic_dot([{
            "index": 1, "text": "Process",
            "type": "PROCESS", "short_label": "Process"
        }])

    filtered = filter_conversation_steps(steps)
    classified = classify_steps(filtered[:20])

    if not classified:
        return _deterministic_dot([{
            "index": 1, "text": "Process",
            "type": "PROCESS", "short_label": "Process"
        }])

    return _deterministic_dot(classified)


# ============================================================
# Video Pipeline: From PDD Steps (dicts)
# ============================================================

def generate_flowchart_dot_from_steps(
    pdd_steps: List[Dict],
    project_name: str
) -> str:
    """Generate DOT flowchart code from PDD step dicts."""
    start = time.time()

    if not pdd_steps:
        return ""

    descriptions = [s.get("description", "") for s in pdd_steps]
    classified = classify_steps(descriptions[:25])

    if not classified:
        timed("Flowchart", start)
        return ""

    num_steps = len(classified)
    if num_steps > 20:
        rankdir = "LR"
        size_hint = 'size="12,8"'
    else:
        rankdir = "TB"
        size_hint = 'size="8,10"'

    dot_code = _deterministic_dot(classified, rankdir=rankdir, size_hint=size_hint)
    timed("Flowchart", start)
    return dot_code


# ============================================================
# Deterministic DOT Generation (Enhanced)
# ============================================================

def _deterministic_dot(
    classified: List[Dict],
    rankdir: str = "TB",
    size_hint: str = 'size="8,10"'
) -> str:
    """Generate DOT code deterministically from classified steps.
    Enhanced to handle decisions with proper Yes/No branching
    and loops that connect back."""
    lines = [
        'digraph ProcessFlow {',
        f'    rankdir={rankdir};',
        f'    {size_hint};',
        '    dpi=300;',
        '    node [fontname="Arial", fontsize=10, style=filled];',
        '    edge [fontname="Arial", fontsize=9];',
        '',
        '    Start [label="Start", shape=oval, fillcolor=lightgreen];'
    ]

    nodes = []  # List of (node_id, node_type)
    step_counter = 0
    decision_counter = 0

    for c in classified:
        step_type = c["type"]
        label = c["short_label"].replace('"', '\\"')

        # Wrap long labels
        if len(label) > 22:
            words = label.split()
            mid = len(words) // 2
            label = ' '.join(words[:mid]) + '\\n' + ' '.join(words[mid:])

        if step_type == "DECISION":
            decision_counter += 1
            node_id = f'Decision{decision_counter}'
            question_label = _make_question(label)
            lines.append(
                f'    {node_id} [label="{question_label}", '
                f'shape=diamond, fillcolor=gold];'
            )
            nodes.append((node_id, "DECISION"))

        elif step_type == "LOOP":
            step_counter += 1
            node_id = f'Step{step_counter}'
            lines.append(
                f'    {node_id} [label="{label}", '
                f'shape=box, fillcolor=lightblue];'
            )
            nodes.append((node_id, "LOOP"))

            # Add loop-back decision
            decision_counter += 1
            loop_decision_id = f'LoopCheck{decision_counter}'
            lines.append(
                f'    {loop_decision_id} [label="More Items?", '
                f'shape=diamond, fillcolor=gold];'
            )
            nodes.append((loop_decision_id, "LOOP_DECISION"))

        elif step_type == "DATA_OP":
            step_counter += 1
            node_id = f'Step{step_counter}'
            lines.append(
                f'    {node_id} [label="{label}", '
                f'shape=box, fillcolor=lightskyblue];'
            )
            nodes.append((node_id, "DATA_OP"))

        else:
            step_counter += 1
            node_id = f'Step{step_counter}'
            lines.append(
                f'    {node_id} [label="{label}", '
                f'shape=box, fillcolor=lightblue];'
            )
            nodes.append((node_id, "PROCESS"))

    lines.append('    End [label="End", shape=oval, fillcolor=lightcoral];')
    lines.append('')

    # Build edges
    all_node_ids = ['Start'] + [n[0] for n in nodes] + ['End']

    i = 0
    while i < len(all_node_ids) - 1:
        current = all_node_ids[i]
        next_node = all_node_ids[i + 1]

        # Find the type of current node
        current_type = None
        for nid, ntype in nodes:
            if nid == current:
                current_type = ntype
                break

        if current_type == "LOOP_DECISION":
            # Find the loop body (previous LOOP node)
            loop_body = all_node_ids[i - 1] if i > 0 else next_node
            lines.append(f'    {current} -> {loop_body} [label="Yes"];')
            lines.append(f'    {current} -> {next_node} [label="No"];')

        elif current_type == "DECISION":
            # Yes goes to next, No skips to the one after or End
            lines.append(f'    {current} -> {next_node} [label="Yes"];')
            skip_to = all_node_ids[i + 2] if i + 2 < len(all_node_ids) else 'End'
            lines.append(f'    {current} -> {skip_to} [label="No"];')

        else:
            lines.append(f'    {current} -> {next_node};')

        i += 1

    lines.append('}')
    return '\n'.join(lines)


def _make_question(label: str) -> str:
    """Convert a statement label into a short question for decision diamonds."""
    label = label.rstrip('.').rstrip('?')

    lower = label.lower()
    question_map = {
        'valid': 'Valid?', 'eligible': 'Eligible?', 'active': 'Active?',
        'found': 'Found?', 'exist': 'Exists?', 'success': 'Successful?',
        'fail': 'Failed?', 'match': 'Matches?', 'approv': 'Approved?',
        'complet': 'Complete?', 'available': 'Available?',
        'disabled': 'Account Disabled?', 'inactive': 'Account Inactive?',
        'blank': 'Data Present?', 'empty': 'Data Present?',
        'missing': 'Data Found?',
    }

    for keyword, question in question_map.items():
        if keyword in lower:
            return question

    # Shorten to max 4 words + "?"
    words = label.split()
    if len(words) > 4:
        label = ' '.join(words[:4])
    return label + '?'