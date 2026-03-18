# llm_tasks/__init__.py

"""
LLM tasks module.
All Gemini-powered text and vision generation tasks.
Shared by both audio and video pipelines.
"""

from llm_tasks.system_prompts import (
    PDD_SYSTEM_PROMPT,
    BRD_SYSTEM_PROMPT,
    VISION_SYSTEM_PROMPT,
    get_system_prompt,
)
from llm_tasks.entity_extraction import extract_entities_and_project
from llm_tasks.document_sections import generate_all_sections_parallel
from llm_tasks.process_steps import extract_process_steps, get_detailed_process_steps
from llm_tasks.requirements import (
    get_input_requirements,
    get_interface_requirements,
    get_exception_handling,
)
from llm_tasks.flowchart_dot import (
    generate_dot_and_apps,
    generate_flowchart_dot_from_steps,
)
from llm_tasks.vision_describer import analyze_transitions_smart, identify_application
from llm_tasks.step_synthesizer import synthesize_pdd_steps
from llm_tasks.meeting_compact import (
    generate_pdd_bundle_batch,
    generate_dot_from_transcript,
)
