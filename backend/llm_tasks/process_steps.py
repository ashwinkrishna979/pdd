# llm_tasks/process_steps.py

"""
Process step extraction from transcript.
Used as FALLBACK only — primary extraction is via meeting_compact.py.
"""

import time
from typing import Dict, List, Optional

from core.gemini_client import gemini_client
from core.config import config
from core.utils import (
    timed, safe_sample, build_entity_hint,
    filter_conversation_steps, parse_numbered_steps,
    deduplicate_steps, enforce_tone, redact_pii_text
)
from llm_tasks.system_prompts import get_system_prompt, TONE_RULES


def extract_process_steps(
    transcript: str,
    entities: Optional[Dict] = None
) -> List[str]:
    """
    Extract automation process steps from transcript.
    Fallback method — prefer generate_doc_bundle_from_transcript().
    """
    start = time.time()

    if entities is None:
        from llm_tasks.entity_extraction import extract_entities_and_project
        entities, _ = extract_entities_and_project(transcript)

    entity_hint = build_entity_hint(entities)
    sample = safe_sample(transcript, max_len=config.llm.max_sample_text)

    prompt = f"""You are a senior Business Analyst extracting process steps from a meeting transcript.

{entity_hint}

YOUR TASK:
Extract 8-15 HIGH-LEVEL AUTOMATION STEPS describing what the automated system does end-to-end.

RULES:
- Each step starts with an action verb: Connect, Navigate, Extract, Validate, Process, Generate, Update, Log.
- Write in imperative tone. Do NOT start with "The system...".
- Use ONLY application/system names from the transcript.
- NEVER include meeting conversations, scheduling, or coordination activities.
- NEVER include personal names, emails, or phone numbers.
{TONE_RULES}

EXAMPLES OF CORRECT STEPS:
1. Connect to the target application using authorized credentials.
2. Navigate to the data management module.
3. Extract records from the source database based on configured criteria.
4. Validate each record against the defined business rules.
5. Process eligible records and update their status.

TRANSCRIPT:
{sample}

OUTPUT (numbered list only):
1."""

    response = gemini_client.generate(
        prompt=prompt,
        system_prompt=get_system_prompt(),
        call_name="ProcessSteps_Fallback"
    )

    all_steps = []
    if response:
        if not response.strip().startswith("1"):
            response = "1. " + response
        all_steps = parse_numbered_steps(response)
        all_steps = filter_conversation_steps(all_steps)
        all_steps = [redact_pii_text(s) for s in all_steps]

    if len(all_steps) < 3:
        all_steps = _template_steps(entities)

    unique = deduplicate_steps(all_steps)
    if len(unique) > 20:
        mid = len(unique) // 2
        unique = unique[:8] + unique[mid-2:mid+2] + unique[-8:]
    if not unique:
        unique = _template_steps(entities)

    timed(f"Steps ({len(unique)})", start)
    return unique


def get_detailed_process_steps(
    transcript: str,
    project_name: str,
    entity_hint: str
) -> List[Dict]:
    """
    Extract detailed step-by-step process for Section 2.4.
    Fallback method — prefer generate_doc_bundle_from_transcript().
    """
    start = time.time()
    sample = safe_sample(transcript, max_len=config.llm.max_sample_text)

    prompt = f"""You are a senior Business Analyst writing detailed process steps for a PDD Section 2.4.

Project: "{project_name}". {entity_hint}

YOUR TASK:
Write 15-25 DETAILED SCREEN-LEVEL STEPS describing specific system actions.

Each step must describe ONE specific action the system performs:
- Logging into applications (with credential handling)
- Navigating to specific pages, tabs, or modules
- Clicking buttons, selecting menu items
- Entering data in specific fields
- Downloading, exporting, or saving files
- Processing, filtering, or validating records
- Generating reports or summaries
- Handling errors or exceptions

RULES:
- Each step is 1-2 sentences describing a specific system action.
- Start each with an action verb (e.g., Log in, Navigate, Click, Enter). Do NOT start with "The system...".
- Be specific about UI elements: "Click the 'Submit' button", not "Submit".
- NEVER include meeting discussions, conversations, or coordination.
- NEVER include personal names, emails, or phone numbers.

TRANSCRIPT:
{sample}

OUTPUT (numbered list only):
1."""

    response = gemini_client.generate(
        prompt=prompt,
        system_prompt=get_system_prompt(),
        call_name="DetailedSteps_Fallback"
    )
    timed("Detailed Steps", start)

    detailed = []
    if response:
        if not response.strip().startswith("1"):
            response = "1. " + response
        parsed = parse_numbered_steps(response)
        parsed = filter_conversation_steps(parsed)
        for i, step in enumerate(parsed):
            step = enforce_tone(step)
            step = redact_pii_text(step)
            detailed.append({
                "number": f"2.4.{i+1}",
                "description": step
            })

    if not detailed:
        detailed = [
            {"number": "2.4.1", "description": "Open the target application and navigate to the login page."},
            {"number": "2.4.2", "description": "Enter the configured credentials and authenticate."},
            {"number": "2.4.3", "description": "Navigate to the relevant processing module."},
            {"number": "2.4.4", "description": "Extract the required data from the source."},
            {"number": "2.4.5", "description": "Validate each record against the defined business rules."},
            {"number": "2.4.6", "description": "Process eligible records and update their status."},
            {"number": "2.4.7", "description": "Generate a summary report of processed records."},
            {"number": "2.4.8", "description": "Log execution details and close the application."},
        ]
    return detailed


def _template_steps(entities: Dict) -> List[str]:
    """Generate template steps when LLM fails."""
    apps = (
        ', '.join(entities.get('applications', []))
        or 'the target application'
    )
    return [
        f"Connect to {apps} using authorized credentials.",
        f"Navigate to the relevant processing module within {apps}.",
        "Extract the required data from the configured source.",
        "Filter records based on the defined business criteria.",
        "Validate each record against the established rules.",
        f"Perform the required processing actions in {apps}.",
        "Capture and record the updated status for each processed item.",
        "Generate a comprehensive execution report.",
        "Update the process status and log all execution details.",
    ]