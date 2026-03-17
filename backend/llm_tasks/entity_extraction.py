# llm_tasks/entity_extraction.py

"""
Entity extraction and project name detection from transcript.
Used by the audio pipeline to identify key entities.
Kept as standalone for backward compatibility — the consolidated
bundle call in meeting_compact.py also extracts entities.
"""

import time
from typing import Dict, List, Tuple

from core.gemini_client import gemini_client
from core.config import config
from core.utils import timed, safe_sample, verify_entities_against_transcript
from llm_tasks.system_prompts import get_system_prompt


def extract_entities_and_project(
    transcript: str
) -> Tuple[Dict[str, List[str]], str]:
    """
    Extract named entities and project name from transcript.
    Returns:
        Tuple of (entities dict, project name string)
    """
    start = time.time()
    sample = safe_sample(transcript, max_len=config.llm.max_sample_entity)

    prompt = f"""You are a senior Business Analyst. Extract factual information from this meeting transcript.

YOUR TASK:
Identify all named entities explicitly mentioned in the text.

STRICT RULES:
- ONLY extract names EXPLICITLY mentioned in the text.
- Do NOT guess, infer, or correct names. Write them exactly as they appear.
- If no names exist for a category, write "None".
- For the project name: if not explicitly stated, create a short descriptive name (max 6 words) from the main process discussed.
- NEVER include personal names, email addresses, or phone numbers.

OUTPUT FORMAT (exactly as shown, no other text):
COMPANIES: name1, name2
APPLICATIONS: name1, name2
SYSTEMS: name1, name2
DEPARTMENTS: name1, name2
PROJECT_NAME: Short Descriptive Name

TRANSCRIPT:
{sample}"""

    response = gemini_client.generate(
        prompt=prompt,
        system_prompt=get_system_prompt(),
        call_name="EntityExtraction"
    )

    entities = {
        "companies": [],
        "applications": [],
        "systems": [],
        "departments": [],
        "processes": []
    }
    project_name = "Process Automation Project"

    if response:
        for line in response.split('\n'):
            line = line.strip()
            if ':' not in line:
                continue
            key, _, value = line.partition(':')
            key = key.strip().upper()
            items = [
                x.strip() for x in value.split(',')
                if x.strip() and x.strip().lower() not in
                ['none', 'n/a', '', 'not mentioned', 'none mentioned']
            ]
            if 'COMPAN' in key:
                entities["companies"] = items
            elif 'APPLIC' in key:
                entities["applications"] = items
            elif 'SYSTEM' in key:
                entities["systems"] = items
            elif 'DEPART' in key:
                entities["departments"] = items
            elif 'PROJECT' in key:
                name = value.strip().strip('"\'')
                if name and name.lower() not in ['none', 'n/a', 'not mentioned']:
                    project_name = ' '.join(name.split()[:7])

    entities = verify_entities_against_transcript(entities, transcript)
    timed("Entities+Project", start)
    return entities, project_name