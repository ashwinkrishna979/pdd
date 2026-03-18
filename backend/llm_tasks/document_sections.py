# llm_tasks/document_sections.py

"""
Document section generation — used ONLY by the video pipeline now.
Audio pipeline uses the consolidated meeting_compact.py call.
Includes parallel generation for video pipeline sections.
Prompting enhanced to look for loops, conditionals, and validation logic.
"""

import re
import time
import concurrent.futures
from typing import Dict, List, Any

from core.gemini_client import gemini_client
from core.config import config
from core.utils import timed, safe_sample, enforce_tone, redact_pii_text
from llm_tasks.system_prompts import get_system_prompt, PDD_SYSTEM_PROMPT, TONE_RULES


def _sanitize_section_output(text: str) -> str:
    """Remove instruction echoes and apply tone + redaction."""
    if not text:
        return ""
    patterns = [
        r'^Write\s+\d+-\d+\s+.*?(?=\n|$)',
        r'^Do\s+NOT\s+.*?(?=\n|$)',
        r'^INSTRUCTIONS?:.*?(?=\n\n|$)',
        r'^OUTPUT:?\s*',
        r'^SECTION\s*\d+[:\s]*',
        r'^Sure[,!.]?\s*',
        r'^Certainly[,!.]?\s*',
        r'^Here\s+(?:is|are)\s+.*?(?=\n\n|$)',
    ]
    cleaned = text
    for pattern in patterns:
        cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE | re.MULTILINE)
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    cleaned = enforce_tone(cleaned)
    cleaned = redact_pii_text(cleaned)
    return cleaned.strip()


_VIDEO_SECTION_PROMPT_BASE = f"""You are a senior Business Analyst creating a Process Definition Document.

{TONE_RULES}

CRITICAL: Output ONLY the requested section content. No headers, no instructions, no meta-commentary."""


def _generate_purpose_video(project_name: str, app_name: str, step_summaries: List[str]) -> str:
    steps_text = "\n".join(f"- {s[:100]}" for s in step_summaries[:20])
    prompt = f"""Write the "Purpose of this Document" section for a Process Definition Document.
Project: "{project_name}" | Application: "{app_name}"
Key steps observed:
{steps_text}

REQUIREMENTS:
- Write 2-3 substantive paragraphs (150-250 words total).
- Paragraph 1: Define what this document covers — objectives, scope, and requirements.
- Paragraph 2: Identify intended audience — developers, QA engineers, stakeholders.
- Paragraph 3: Describe the scope — which applications, validations, and data flows the document addresses.
- Write in third person, present tense, active voice.
- NEVER mention screenshots, recordings, or video analysis.
Output ONLY the section content."""

    response = gemini_client.generate(
        prompt=prompt, system_prompt=_VIDEO_SECTION_PROMPT_BASE, temperature=0.3, call_name="DocumentPurpose_Video"
    )
    return _sanitize_section_output(response) if response else ""


def _generate_overview_video(project_name: str, app_name: str, step_summaries: List[str]) -> Dict[str, str]:
    steps_text = "\n".join(f"- {s[:100]}" for s in step_summaries[:15])
    prompt = f"""Write two sections for a Process Definition Document.
Project: "{project_name}" | Application: "{app_name}"
Process steps observed:
{steps_text}

=== OVERVIEW ===
Write 1 opening paragraph stating the primary business objective.
Then list 5-7 bullet points (using •) describing specific outcomes. Focus on validations, automated data extractions, conditionals, and iterative processing.
Each bullet must start with "The system..."

=== JUSTIFICATION ===
Write 1 opening paragraph about operational value.
Then list 5-7 numbered items, each with a title and description.
Example: "1. Reduced Processing Time — The system completes the process rapidly..."

Output both sections with the === headers. Third person, present tense only."""

    response = gemini_client.generate(
        prompt=prompt, system_prompt=_VIDEO_SECTION_PROMPT_BASE, temperature=0.3, call_name="OverviewJustification_Video"
    )

    result = {"overview": "", "justification": ""}
    if response:
        overview_match = re.search(r'===\s*OVERVIEW\s*===\s*(.*?)(?====\s*JUSTIFICATION|$)', response, re.DOTALL | re.IGNORECASE)
        justification_match = re.search(r'===\s*JUSTIFICATION\s*===\s*(.*?)$', response, re.DOTALL | re.IGNORECASE)
        if overview_match:
            result["overview"] = _sanitize_section_output(overview_match.group(1))
        if justification_match:
            result["justification"] = _sanitize_section_output(justification_match.group(1))

    return result


def _generate_as_is_video(project_name: str, app_name: str, step_summaries: List[str]) -> str:
    steps_text = "\n".join(f"- {s[:100]}" for s in step_summaries[:15])
    prompt = f"""Write the "As Is" (current manual process) section for a Process Definition Document.
Project: "{project_name}" | Application: "{app_name}"
Automated steps:
{steps_text}

Describe how this process is CURRENTLY performed MANUALLY before automation.
REQUIREMENTS:
- Write 3-4 paragraphs (200-300 words).
- Highlight the manual effort of filtering data, checking rules/conditions, and repeating tasks for each record/user.
- End with "Business Challenges:" followed by 4-6 bullet points listing specific problems (e.g., Risk of human error during manual validation).
Output ONLY the section content."""

    response = gemini_client.generate(
        prompt=prompt, system_prompt=_VIDEO_SECTION_PROMPT_BASE, temperature=0.3, call_name="AsIsProcess_Video"
    )
    return _sanitize_section_output(response) if response else ""


def _generate_to_be_video(project_name: str, app_name: str, step_summaries: List[str]) -> str:
    steps_text = "\n".join(f"- {s[:100]}" for s in step_summaries[:20])
    prompt = f"""Write the "To Be" (automated process) section for a Process Definition Document.
Project: "{project_name}" | Application: "{app_name}"
Automated steps:
{steps_text}

Describe the AUTOMATED process as if it is already operational.
REQUIREMENTS:
- Write 3-4 paragraphs (200-300 words).
- Paragraph 1: Describe the initialization and secure application access.
- Paragraph 2: Explicitly mention how the system extracts data, filters/cleans it, and iterates through records.
- Paragraph 3: Explicitly describe validation logic and conditional paths (e.g., "If validation fails, the system...").
- Paragraph 4: Describe post-action reporting, capturing counts, and updating tracking sheets.
- Write as present-tense facts: "The system connects...", "The automation validates..."
Output ONLY the section content."""

    response = gemini_client.generate(
        prompt=prompt, system_prompt=_VIDEO_SECTION_PROMPT_BASE, temperature=0.3, call_name="ToBeProcess_Video"
    )
    return _sanitize_section_output(response) if response else ""


def _generate_prerequisites_video(project_name: str, app_name: str, vision_descriptions: List[str]) -> List[Dict]:
    desc_sample = "\n".join(safe_sample(d, 100) for d in vision_descriptions[:10])
    prompt = f"""List the INPUT REQUIREMENTS for this automation.
Project: "{project_name}" | Application: "{app_name}"
Screens observed:
{desc_sample}

List each input parameter the automation requires to execute (e.g., URLs, SSO Credentials, Validation Scripts, Shared Paths, Configuration Lists).
FORMAT (one per line, use | separator):
Parameter Name | Description of what it is and why the automation needs it

List 5-10 inputs. Output ONLY the list. NEVER include actual credential values."""

    response = gemini_client.generate(
        prompt=prompt, system_prompt=_VIDEO_SECTION_PROMPT_BASE, temperature=0.3, call_name="Prerequisites_Video"
    )

    inputs = []
    if response:
        for line in response.split('\n'):
            line = line.strip()
            if '|' in line:
                parts = line.split('|', 1)
                param = re.sub(r'^\d+[\.\)]\s*', '', parts[0]).strip()
                desc = parts[1].strip() if len(parts) > 1 else ""
                if param and len(param) > 2:
                    inputs.append({"parameter": redact_pii_text(param), "description": redact_pii_text(desc)})

    return inputs if inputs else [
        {"parameter": "Admin Portal Credentials", "description": "Authorized credentials required to access the portals."},
        {"parameter": "Target Application URL", "description": f"The web address for accessing {app_name}."},
    ]


def _generate_exceptions_video(project_name: str, app_name: str, step_descriptions: List[str]) -> List[Dict]:
    steps_sample = "\n".join(f"- {s[:100]}" for s in step_descriptions[:15])
    prompt = f"""List exception handling scenarios for this automation.
Project: "{project_name}" | Application: "{app_name}"
Process steps:
{steps_sample}

For each potential failure scenario, describe the exception and the system's handling action.
FORMAT (one per line, use | separator):
Exception Scenario | Handling Action (what the system does)

Include items like: Login Failure, Portal Timeout, Missing Export Data, Validation Script Error, File Access Denied, Duplicate Records.
List 6-10 exceptions. Output ONLY the list."""

    response = gemini_client.generate(
        prompt=prompt, system_prompt=_VIDEO_SECTION_PROMPT_BASE, temperature=0.3, call_name="ExceptionHandling_Video"
    )

    exceptions = []
    if response:
        for line in response.split('\n'):
            line = line.strip()
            if '|' in line:
                parts = line.split('|', 1)
                exc = re.sub(r'^\d+[\.\)]\s*', '', parts[0]).strip()
                handling = parts[1].strip() if len(parts) > 1 else ""
                if exc and len(exc) > 5:
                    exceptions.append({"exception": exc, "handling": handling})

    return exceptions if exceptions else [
        {"exception": "Admin Portal Login Failure", "handling": "The system logs the error and retries the connection three times."},
        {"exception": "User Validation Failure", "handling": "The system stops processing the current user and moves to the next record."},
    ]


def _generate_interfaces_video(app_name: str, vision_descriptions: List[str]) -> List[Dict]:
    desc_sample = "\n".join(safe_sample(d, 100) for d in vision_descriptions[:8])
    prompt = f"""List the INTERFACE REQUIREMENTS (applications and systems) for this automation.
Application: "{app_name}"
Screens observed:
{desc_sample}

List each application or system the automation interacts with (e.g., Target Portal, Active Directory, Excel, Shared Drive).
FORMAT (one per line, use | separator):
Application/System Name | Purpose of interaction

List 3-6 interfaces. Output ONLY the list."""

    response = gemini_client.generate(
        prompt=prompt, system_prompt=_VIDEO_SECTION_PROMPT_BASE, temperature=0.3, call_name="InterfaceReqs_Video"
    )

    interfaces = []
    if response:
        for line in response.split('\n'):
            line = line.strip()
            if '|' in line:
                parts = line.split('|', 1)
                app = re.sub(r'^\d+[\.\)]\s*', '', parts[0]).strip()
                purpose = parts[1].strip() if len(parts) > 1 else ""
                if app and len(app) > 2:
                    interfaces.append({"application": app, "purpose": purpose})

    return interfaces if interfaces else [
        {"application": app_name or "Target Portal", "purpose": "Primary portal for process execution."},
        {"application": "Excel", "purpose": "Used for processing data and maintaining the tracking report."}
    ]


def generate_all_sections_parallel(
    project_name: str,
    app_name: str,
    step_descriptions: List[str],
    vision_descriptions: List[str]
) -> Dict[str, Any]:
    """Generate all PDD sections in parallel using thread pool."""
    start = time.time()
    results = {}

    tasks = {
        "purpose": lambda: _generate_purpose_video(project_name, app_name, step_descriptions),
        "overview_justification": lambda: _generate_overview_video(project_name, app_name, step_descriptions),
        "as_is": lambda: _generate_as_is_video(project_name, app_name, step_descriptions),
        "to_be": lambda: _generate_to_be_video(project_name, app_name, step_descriptions),
        "prerequisites": lambda: _generate_prerequisites_video(project_name, app_name, vision_descriptions),
        "exceptions": lambda: _generate_exceptions_video(project_name, app_name, step_descriptions),
        "interfaces": lambda: _generate_interfaces_video(app_name, vision_descriptions),
    }

    workers = min(config.llm.max_workers, len(tasks))
    print(f"    [Sections] Generating {len(tasks)} sections ({workers} parallel workers)...")

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_name = {executor.submit(fn): name for name, fn in tasks.items()}
        for future in concurrent.futures.as_completed(future_to_name):
            name = future_to_name[future]
            try:
                results[name] = future.result()
                print(f"    [Sections] {name}")
            except Exception as e:
                print(f"    [Sections] {name}: {e}")
                results[name] = None

    timed(f"All sections ({len(results)})", start)
    return results