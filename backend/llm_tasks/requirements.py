# llm_tasks/requirements.py

"""
Requirements extraction from transcript.
Used as FALLBACK — primary extraction is via meeting_compact.py.
"""

import re
import time
from typing import Dict, List

from core.gemini_client import gemini_client
from core.config import config
from core.utils import timed, safe_sample, redact_pii_text
from llm_tasks.system_prompts import get_system_prompt, TONE_RULES


def get_input_requirements(
    transcript: str,
    project_name: str,
    entity_hint: str
) -> List[Dict]:
    """Extract input requirements from transcript."""
    start = time.time()
    sample = safe_sample(transcript, max_len=config.llm.max_sample_small)

    prompt = f"""You are a senior Business Analyst identifying input requirements for an automation project.

Project: "{project_name}". {entity_hint}

YOUR TASK:
List 5-10 input parameters the automation requires to execute.

Consider:
- Application credentials (username, password — never actual values)
- Source data files or database connections
- Application URLs or endpoints
- Configuration parameters (thresholds, filters, date ranges)
- Output file paths or destinations

FORMAT (one per line, use | separator):
INPUT: Parameter Name | DESCRIPTION: What it is and why the automation needs it

RULES:
- NEVER include actual credential values, personal names, or email addresses.
- Use ONLY application names from the transcript.

TRANSCRIPT:
{sample}

OUTPUT:"""

    response = gemini_client.generate(
        prompt=prompt,
        system_prompt=get_system_prompt(),
        call_name="InputRequirements_Fallback"
    )
    timed("Inputs", start)

    inputs = []
    if response:
        for line in response.split('\n'):
            line = line.strip()
            if '|' in line:
                parts = line.split('|')
                param, desc = "", ""
                for part in parts:
                    part = part.strip()
                    if part.upper().startswith('INPUT'):
                        param = re.sub(
                            r'^INPUT[S]?\s*[:]\s*', '', part,
                            flags=re.IGNORECASE
                        ).strip()
                    elif part.upper().startswith('DESC'):
                        desc = re.sub(
                            r'^DESCRIPTION\s*[:]\s*', '', part,
                            flags=re.IGNORECASE
                        ).strip()
                if param:
                    inputs.append({
                        "parameter": redact_pii_text(param),
                        "description": redact_pii_text(desc) or "Required for automation execution."
                    })

    if not inputs:
        inputs = [
            {"parameter": "Application Credentials",
             "description": "Authorized username and password for secure application access."},
            {"parameter": "Source Data Location",
             "description": "File path or database connection for source data retrieval."},
            {"parameter": "Application URL",
             "description": "Web address of the target application portal."},
        ]
    return inputs


def get_interface_requirements(
    transcript: str,
    entities: Dict
) -> List[Dict]:
    """Extract interface/application requirements from transcript."""
    start = time.time()
    sample = safe_sample(transcript, max_len=config.llm.max_sample_small)
    apps_hint = ""
    if entities.get('applications'):
        apps_hint = (
            f"Applications from transcript: "
            f"{', '.join(entities['applications'])}"
        )

    prompt = f"""You are a senior Business Analyst identifying interface requirements.

{apps_hint}

YOUR TASK:
List all applications and systems the automation interacts with.
ONLY include applications explicitly mentioned in the transcript.

FORMAT (one per line, use | separator):
APP: Application Name | INTERFACE: Web/Desktop/API/Database | PURPOSE: Why the automation uses it

TRANSCRIPT:
{sample}

OUTPUT:"""

    response = gemini_client.generate(
        prompt=prompt,
        system_prompt=get_system_prompt(),
        call_name="InterfaceRequirements_Fallback"
    )
    timed("Interfaces", start)

    apps = []
    if response:
        for line in response.split('\n'):
            line = line.strip()
            if '|' in line and 'APP' in line.upper():
                app = {
                    "application": "", "interface": "",
                    "url": "", "purpose": ""
                }
                parts = line.split('|')
                for part in parts:
                    part = part.strip()
                    p_up = part.upper()
                    if p_up.startswith('APP'):
                        app["application"] = re.sub(
                            r'^APP(?:LICATION)?:\s*', '', part,
                            flags=re.IGNORECASE
                        ).strip()
                    elif p_up.startswith('INTERFACE'):
                        app["interface"] = part.split(':', 1)[-1].strip()
                    elif p_up.startswith('PURPOSE'):
                        app["purpose"] = part.split(':', 1)[-1].strip()
                if app["application"]:
                    apps.append(app)

    if not apps:
        apps = [{
            "application": "Target Application",
            "interface": "Web/Desktop",
            "url": "",
            "purpose": "Primary application for process automation"
        }]
    return apps


def get_exception_handling(
    transcript: str,
    project_name: str,
    entity_hint: str
) -> List[Dict]:
    """Generate exception handling scenarios from transcript."""
    start = time.time()
    sample = safe_sample(transcript, max_len=config.llm.max_sample_small)

    prompt = f"""You are a senior Business Analyst defining exception handling for an automation project.

Project: "{project_name}". {entity_hint}

YOUR TASK:
List 6-10 exception scenarios and how the system handles each.

Consider: login failures, missing data, records not found, processing errors,
validation failures, timeout errors, network issues, application crashes.

FORMAT (one per line, use | separator):
EXCEPTION: Scenario title | HANDLING: What the system does (in third person, present tense)

Example:
EXCEPTION: Application Login Failure | HANDLING: The system retries login up to 3 times. If authentication fails, the system stops execution, captures a screenshot, and sends an error notification.

TRANSCRIPT:
{sample}

OUTPUT:"""

    response = gemini_client.generate(
        prompt=prompt,
        system_prompt=get_system_prompt(),
        call_name="ExceptionHandling_Fallback"
    )
    timed("Exceptions", start)

    exceptions = []
    if response:
        for line in response.split('\n'):
            line = line.strip()
            if '|' in line and 'EXCEPTION' in line.upper():
                parts = line.split('|')
                exc = {"exception": "", "handling": ""}
                for part in parts:
                    part = part.strip()
                    if part.upper().startswith('EXCEPTION'):
                        exc["exception"] = re.sub(
                            r'^EXCEPTION:\s*', '', part,
                            flags=re.IGNORECASE
                        ).strip()
                    elif part.upper().startswith('HANDLING'):
                        exc["handling"] = re.sub(
                            r'^HANDLING:\s*', '', part,
                            flags=re.IGNORECASE
                        ).strip()
                if exc["exception"]:
                    exceptions.append(exc)

    if not exceptions:
        exceptions = [
            {"exception": "Application Login Failure",
             "handling": "The system retries login up to 3 times. If still failing, the system stops execution and sends an error notification."},
            {"exception": "Missing or Invalid Input Data",
             "handling": "The system logs the validation failure, flags the affected record, and continues processing remaining items."},
            {"exception": "Record Not Found",
             "handling": "The system logs the missing record details, skips the entry, and continues with the next record."},
            {"exception": "Application Timeout",
             "handling": "The system waits for the configured timeout period, retries the operation, and logs the timeout event."},
            {"exception": "System Exception",
             "handling": "The system captures error details, saves a screenshot, logs the exception, and terminates gracefully."},
        ]
    return exceptions