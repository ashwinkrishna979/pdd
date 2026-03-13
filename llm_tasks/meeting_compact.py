# llm_tasks/meeting_compact.py

"""
Compact meeting transcript → PDD bundle extraction.

THREE consolidated LLM calls:
  Call 1: Document sections (purpose, overview, justification, as-is, to-be)
  Call 2: Process data (steps, detailed steps, inputs, interfaces, exceptions)
  Call 3: Step refinement — decomposes coarse steps into granular sub-steps
          with conditionals, loops, data operations, and validation logic
"""

from __future__ import annotations

import json
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from core.gemini_client import gemini_client
from core.config import config
from core.utils import safe_sample, timed, enforce_tone, redact_pii_text
from llm_tasks.system_prompts import get_system_prompt, TONE_RULES


# ============================================================
# JSON extraction helpers
# ============================================================


def _repair_json(text: str) -> str:
    """
    Attempt to repair common JSON issues from LLM output.
    """
    if not text:
        return text

    text = text.strip("\ufeff\u200b\u200c\u200d")
    text = re.sub(r"(?<=[{,\[])\s*'([^']+)'\s*:", r' "\1":', text)
    text = re.sub(r":\s*'([^']*)'(?=\s*[,}\]])", r': "\1"', text)
    text = re.sub(r"(?<=[{,])\s*(\w+)\s*:", r' "\1":', text)
    text = re.sub(r",\s*}", "}", text)
    text = re.sub(r",\s*\]", "]", text)
    text = re.sub(r'"\s*\n\s*"(?=\w+"\s*:)', '",\n"', text)
    text = re.sub(r'}\s*\n\s*"(?=\w+"\s*:)', '},\n"', text)
    text = re.sub(r']\s*\n\s*"(?=\w+"\s*:)', '],\n"', text)
    text = _escape_newlines_in_strings(text)
    text = _fix_inner_quotes(text)

    open_braces = text.count("{") - text.count("}")
    open_brackets = text.count("[") - text.count("]")

    if open_braces > 0 or open_brackets > 0:
        in_string = False
        i = 0
        while i < len(text):
            ch = text[i]
            if ch == "\\" and in_string:
                i += 2
                continue
            if ch == '"':
                in_string = not in_string
            i += 1
        if in_string:
            text = text + '"'
        text = re.sub(r",\s*$", "", text.rstrip())
        for _ in range(max(0, open_brackets)):
            text = text.rstrip().rstrip(",") + "]"
        for _ in range(max(0, open_braces)):
            text = text.rstrip().rstrip(",") + "}"

    return text


def _escape_newlines_in_strings(text: str) -> str:
    result = []
    in_string = False
    i = 0
    while i < len(text):
        ch = text[i]
        if ch == "\\" and in_string and i + 1 < len(text):
            result.append(ch)
            result.append(text[i + 1])
            i += 2
            continue
        if ch == '"':
            in_string = not in_string
            result.append(ch)
            i += 1
            continue
        if ch == "\n" and in_string:
            result.append("\\n")
            i += 1
            continue
        if ch == "\r" and in_string:
            i += 1
            continue
        if ch == "\t" and in_string:
            result.append("\\t")
            i += 1
            continue
        result.append(ch)
        i += 1
    return "".join(result)


def _fix_inner_quotes(text: str) -> str:
    try:
        json.loads(text)
        return text
    except json.JSONDecodeError:
        pass

    result = []
    i = 0
    in_string = False

    while i < len(text):
        ch = text[i]
        if ch == "\\" and in_string and i + 1 < len(text):
            result.append(ch)
            result.append(text[i + 1])
            i += 2
            continue
        if ch == '"':
            if not in_string:
                in_string = True
                result.append(ch)
                i += 1
                continue
            else:
                after = text[i + 1 :].lstrip() if i + 1 < len(text) else ""
                is_closing = False
                if not after:
                    is_closing = True
                elif after[0] in (",", "}", "]", ":"):
                    is_closing = True
                elif re.match(r"^[,}\]:]", after):
                    is_closing = True
                if is_closing:
                    in_string = False
                    result.append(ch)
                else:
                    result.append('\\"')
                i += 1
                continue
        result.append(ch)
        i += 1
    return "".join(result)


def _extract_json_object(text: str) -> Optional[str]:
    if not text:
        return None
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s*```\s*$", "", text, flags=re.MULTILINE)
    text = text.strip()

    if text.startswith("{"):
        try:
            json.loads(text)
            return text
        except json.JSONDecodeError:
            pass

    json_text = _extract_balanced_braces(text)
    if json_text:
        try:
            json.loads(json_text)
            return json_text
        except json.JSONDecodeError:
            pass
        repaired = _repair_json(json_text)
        try:
            json.loads(repaired)
            return repaired
        except json.JSONDecodeError:
            pass
        aggressive = _aggressive_json_repair(json_text)
        if aggressive:
            try:
                json.loads(aggressive)
                return aggressive
            except json.JSONDecodeError:
                pass
        truncated = _truncate_to_valid_json(repaired)
        if truncated:
            return truncated

    m = re.search(r"(\{.*\})", text, re.DOTALL)
    if m:
        candidate = m.group(1).strip()
        repaired = _repair_json(candidate)
        try:
            json.loads(repaired)
            return repaired
        except json.JSONDecodeError:
            aggressive = _aggressive_json_repair(candidate)
            if aggressive:
                try:
                    json.loads(aggressive)
                    return aggressive
                except json.JSONDecodeError:
                    pass
    return None


def _aggressive_json_repair(text: str) -> Optional[str]:
    if not text:
        return None
    key_pattern = re.compile(r'"(\w+)"\s*:\s*', re.DOTALL)
    keys_found = [(m.group(1), m.start(), m.end()) for m in key_pattern.finditer(text)]
    if not keys_found:
        return None

    extracted = {}
    for idx, (key, key_start, val_start) in enumerate(keys_found):
        if idx + 1 < len(keys_found):
            next_key_start = keys_found[idx + 1][1]
            raw_value = (
                text[val_start:next_key_start].strip().rstrip().rstrip(",").strip()
            )
        else:
            raw_value = (
                text[val_start:].strip().rstrip().rstrip("}").rstrip(",").strip()
            )
        parsed_value = _parse_raw_value(raw_value, key)
        if parsed_value is not None:
            extracted[key] = parsed_value

    if not extracted:
        return None
    try:
        result = json.dumps(extracted, ensure_ascii=False)
        json.loads(result)
        return result
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


def _parse_raw_value(raw: str, key: str) -> Any:
    if not raw:
        return "" if key not in ("entities",) else {}
    raw = raw.strip()
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        pass
    if raw.startswith("["):
        return _parse_raw_array(raw)
    if raw.startswith("{"):
        repaired = _repair_json(raw)
        try:
            return json.loads(repaired)
        except (json.JSONDecodeError, ValueError):
            return {}
    if raw.startswith('"'):
        return _parse_raw_string(raw)
    return raw.strip('"').strip()


def _parse_raw_array(raw: str) -> List:
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        pass
    repaired = _repair_json(raw)
    try:
        return json.loads(repaired)
    except (json.JSONDecodeError, ValueError):
        pass
    items = []
    if "{" in raw:
        obj_pattern = re.compile(r"\{([^{}]*)\}")
        for m in obj_pattern.finditer(raw):
            obj_text = "{" + m.group(1) + "}"
            repaired_obj = _repair_json(obj_text)
            try:
                items.append(json.loads(repaired_obj))
            except (json.JSONDecodeError, ValueError):
                kv_pattern = re.compile(r'"(\w+)"\s*:\s*"([^"]*)"')
                obj_dict = {}
                for kv in kv_pattern.finditer(m.group(1)):
                    obj_dict[kv.group(1)] = kv.group(2)
                if obj_dict:
                    items.append(obj_dict)
    else:
        str_pattern = re.compile(r'"((?:[^"\\]|\\.)*)"')
        for m in str_pattern.finditer(raw):
            val = m.group(1).strip()
            if val and len(val) > 3:
                items.append(val)
    return items


def _parse_raw_string(raw: str) -> str:
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        pass
    if raw.startswith('"') and raw.endswith('"'):
        inner = raw[1:-1]
    elif raw.startswith('"'):
        inner = raw[1:]
    else:
        inner = raw
    inner = inner.replace("\\", "\\\\")
    inner = inner.replace("\n", "\\n")
    inner = inner.replace("\r", "")
    inner = inner.replace("\t", "\\t")
    inner = inner.replace('"', '\\"')
    try:
        return json.loads(f'"{inner}"')
    except (json.JSONDecodeError, ValueError):
        return inner.replace("\\n", "\n").replace('\\"', '"').replace("\\\\", "\\")


def _extract_balanced_braces(text: str) -> Optional[str]:
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    i = start
    while i < len(text):
        ch = text[i]
        if ch == "\\" and in_string:
            i += 2
            continue
        if ch == '"':
            in_string = not in_string
        elif not in_string:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
        i += 1
    if depth > 0:
        fragment = text[start:]
        fragment = fragment.rstrip().rstrip(",")
        fragment += "}" * depth
        return fragment
    return None


def _truncate_to_valid_json(text: str) -> Optional[str]:
    if not text:
        return None
    for end_pattern in [
        r',\s*"[^"]*"\s*:\s*(?:"(?:[^"\\]|\\.)*$)',
        r',\s*"[^"]*"\s*:\s*\[(?:[^\]]*$)',
        r',\s*"[^"]*"\s*:\s*\{(?:[^}]*$)',
        r',\s*"[^"]*"\s*:\s*(?:"[^"]*"|[\[\{]).*$',
        r',\s*"[^"]*"\s*:.*$',
    ]:
        truncated = re.sub(end_pattern, "", text, flags=re.DOTALL)
        if truncated != text and len(truncated) > 10:
            repaired = _repair_json(truncated)
            try:
                json.loads(repaired)
                return repaired
            except json.JSONDecodeError:
                continue
    comma_positions = _find_top_level_commas(text)
    for pos in reversed(comma_positions):
        candidate = text[:pos].rstrip()
        repaired = _repair_json(candidate)
        try:
            json.loads(repaired)
            return repaired
        except json.JSONDecodeError:
            continue
    return None


def _find_top_level_commas(text: str) -> List[int]:
    positions = []
    depth = 0
    in_string = False
    i = 0
    while i < len(text):
        ch = text[i]
        if ch == "\\" and in_string:
            i += 2
            continue
        if ch == '"':
            in_string = not in_string
        elif not in_string:
            if ch in ("{", "["):
                depth += 1
            elif ch in ("}", "]"):
                depth -= 1
            elif ch == "," and depth == 1:
                positions.append(i)
        i += 1
    return positions


def _coerce_list_str(x: Any) -> List[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return [str(i).strip() for i in x if str(i).strip()]
    if isinstance(x, str):
        lines = [l.strip("•*- \t") for l in x.splitlines() if l.strip()]
        return [l for l in lines if len(l) > 5]
    return [str(x).strip()] if str(x).strip() else []


def _coerce_list_dict(x: Any, keys: Tuple[str, str]) -> List[Dict[str, str]]:
    k1, k2 = keys
    out: List[Dict[str, str]] = []
    if isinstance(x, list):
        for row in x:
            if isinstance(row, dict):
                v1 = str(row.get(k1, "")).strip()
                v2 = str(row.get(k2, "")).strip()
                if v1:
                    out.append({k1: v1, k2: v2})
            elif isinstance(row, str) and "|" in row:
                a, b = row.split("|", 1)
                a, b = a.strip(), b.strip()
                if a:
                    out.append({k1: a, k2: b})
    elif isinstance(x, str):
        for line in x.splitlines():
            if "|" in line:
                a, b = line.split("|", 1)
                a, b = a.strip(), b.strip()
                if a:
                    out.append({k1: a, k2: b})
    return out


def _apply_tone_and_redaction(text: str) -> str:
    if not text:
        return text
    text = enforce_tone(text)
    text = redact_pii_text(text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"__([^_]+)__", r"\1", text)
    return text


def _strip_markdown(text: str) -> str:
    if not text:
        return text
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"__([^_]+)__", r"\1", text)
    text = re.sub(r"\*([^*]+)\*", r"\1", text)
    text = re.sub(r"_([^_]+)_", r"\1", text)
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^[-*]\s+", "- ", text, flags=re.MULTILINE)
    return text


# ============================================================
# Call 1: Document Sections (unchanged logic, same as before)
# ============================================================


def _generate_document_sections(
    transcript: str, project_name_hint: Optional[str] = None
) -> Dict[str, Any]:
    """
    LLM Call 1: Extract project name + all narrative document sections.
    """
    sample = safe_sample(transcript, max_len=config.llm.max_sample_text)
    doc_type = config.document.document_type
    doc_full = config.document.document_type_full

    project_hint = (
        f'Project name hint: "{project_name_hint}"'
        if project_name_hint
        else "Derive a short descriptive project name (max 6 words) from the main process discussed."
    )

    prompt = f"""You are a senior Business Analyst creating a {doc_full} ({doc_type}).

{project_hint}

TASK: Produce the following sections as a JSON object.

SECTION REQUIREMENTS:

1. "project_name": Short descriptive name (max 6 words).

2. "entities": Object with arrays: "companies", "applications", "systems", "departments".
   ONLY names explicitly mentioned in the transcript. Use empty array if none found.

3. "purpose": 2-3 paragraphs (150-250 words).
   - What this document defines (objectives, scope, requirements).
   - Intended audience (developers, QA, business stakeholders).
   - Scope of the automation being documented.

4. "overview": 1 paragraph + 5-7 bullet points.
   - Opening paragraph stating the primary business objective.
   - Bullet points listing specific outcomes. Each starts with "The system".

5. "justification": 1 paragraph + 5-7 numbered benefits.
   - Opening paragraph about operational value.
   - Numbered items with a title and description for each benefit.
   DO NOT use markdown bold (no ** markers). Use plain text only.

6. "as_is": 3-4 paragraphs (200-300 words).
   - Current manual process description.
   - Pain points, inefficiencies, risks.
   - End with "Business Challenges:" followed by 4-6 bullet points.

7. "to_be": 3-4 paragraphs (200-300 words).
   - Future automated process, written in present tense as if operational.
   - Cover trigger, data handling, validation, processing, reporting, logging.

CRITICAL JSON RULES:
- Use ONLY names from the transcript. Never invent names.
- NEVER mention the meeting, transcript, recording, or speakers.
- NEVER include personal names, emails, or phone numbers.
- Write in THIRD PERSON, PRESENT TENSE, ACTIVE VOICE.
- DO NOT use markdown formatting (no **, no ##, no __).
- IMPORTANT: All string values MUST be on a SINGLE LINE. Use \\n for paragraph breaks within strings.
- IMPORTANT: Escape all double quotes inside string values with backslash: \\"
- Output STRICT valid JSON only. No markdown fences, no commentary before or after.

JSON FORMAT:
{{
  "project_name": "string",
  "entities": {{"companies": [], "applications": [], "systems": [], "departments": []}},
  "purpose": "paragraph 1\\n\\nparagraph 2\\n\\nparagraph 3",
  "overview": "paragraph\\n\\n- bullet 1\\n- bullet 2",
  "justification": "paragraph\\n\\n1. benefit title - description\\n2. benefit title - description",
  "as_is": "paragraph 1\\n\\nparagraph 2\\n\\nBusiness Challenges:\\n- challenge 1\\n- challenge 2",
  "to_be": "paragraph 1\\n\\nparagraph 2\\n\\nparagraph 3"
}}

TRANSCRIPT:
{sample}
"""

    resp = gemini_client.generate(
        prompt=prompt,
        system_prompt=get_system_prompt(),
        temperature=0.3,
        max_output_tokens=config.llm.max_output_tokens,
        call_name="DocBundle_Sections",
        max_retries=3,
    )

    result = {
        "project_name": project_name_hint or "Process Automation Project",
        "entities": {
            "companies": [],
            "applications": [],
            "systems": [],
            "departments": [],
        },
        "document": {
            "purpose": "",
            "overview": "",
            "justification": "",
            "as_is": "",
            "to_be": "",
        },
    }

    json_text = _extract_json_object(resp or "")
    if json_text:
        try:
            data = json.loads(json_text)
            pn = str(data.get("project_name", "")).strip()
            if pn and len(pn) > 3:
                result["project_name"] = pn
            ent = data.get("entities", {})
            if isinstance(ent, dict):
                for key in ["companies", "applications", "systems", "departments"]:
                    items = ent.get(key, [])
                    if isinstance(items, list):
                        result["entities"][key] = [
                            str(i).strip() for i in items if str(i).strip()
                        ]
            for key in ["purpose", "overview", "justification", "as_is", "to_be"]:
                val = str(data.get(key, "")).strip()
                if val:
                    val = _strip_markdown(val)
                    val = _apply_tone_and_redaction(val)
                    result["document"][key] = val
            print(
                f"    [DocBundle_Sections] Parsed successfully: "
                f"{sum(1 for v in result['document'].values() if v)}/5 sections"
            )
        except json.JSONDecodeError as e:
            print(f"    [DocBundle_Sections] JSON parse failed: {e}")
            result["document"] = _fallback_parse_sections(resp or "")
    else:
        print("    [DocBundle_Sections] No JSON found, using fallback parser")
        result["document"] = _fallback_parse_sections(resp or "")

    # Ensure all sections have content — use generic templates
    _ensure_section_defaults(result)

    return result


def _ensure_section_defaults(result: Dict):
    """Fill empty sections with generic templates based on project name."""
    pn = result["project_name"]
    doc = result["document"]
    doc_full_name = config.document.document_type_full

    if not doc["purpose"] or len(doc["purpose"]) < 50:
        doc["purpose"] = (
            f"This {doc_full_name} defines the objectives, scope, and detailed "
            f"requirements for the {pn} automation initiative. The document serves "
            f"as the primary reference for development teams, quality assurance personnel, "
            f"and business stakeholders involved in the design, implementation, and "
            f"validation of the automated solution.\n\n"
            f"The scope encompasses the complete end-to-end process flow, including "
            f"system interactions, data handling procedures, validation rules, exception "
            f"handling scenarios, and expected outputs."
        )

    if not doc["overview"] or len(doc["overview"]) < 50:
        doc["overview"] = (
            f"The primary objective of the {pn} automation is to streamline "
            f"and standardize the existing manual process.\n\n"
            f"- The system automates repetitive manual tasks to reduce processing time.\n"
            f"- The system enforces standardized validation rules across all records.\n"
            f"- The system generates comprehensive audit trails for compliance tracking.\n"
            f"- The system reduces human error through automated data handling.\n"
            f"- The system provides real-time status reporting and exception notifications."
        )

    if not doc["justification"] or len(doc["justification"]) < 50:
        doc["justification"] = (
            f"Automating the {pn} process delivers measurable operational value.\n\n"
            f"1. Reduced Processing Time - The system completes the process in minutes.\n"
            f"2. Improved Accuracy - Eliminates manual data entry errors.\n"
            f"3. Enhanced Compliance - Maintains detailed audit logs.\n"
            f"4. Consistent Execution - Applies identical business rules to every record.\n"
            f"5. Scalability - Handles increased volume without additional resources."
        )

    if not doc["as_is"] or len(doc["as_is"]) < 50:
        doc["as_is"] = (
            f"The current {pn} process relies on manual execution by trained operators.\n\n"
            f"Business Challenges:\n"
            f"- High processing time due to manual handling.\n"
            f"- Risk of human error in data entry and validation.\n"
            f"- Inconsistent application of business rules.\n"
            f"- Limited audit trail for compliance verification."
        )

    if not doc["to_be"] or len(doc["to_be"]) < 50:
        doc["to_be"] = (
            f"The {pn} automation executes the end-to-end process through a structured "
            f"sequence of system actions. The automation initiates upon a configured "
            f"trigger and processes data according to defined business rules.\n\n"
            f"Upon completion, the system generates a comprehensive execution report."
        )


def _fallback_parse_sections(raw_text: str) -> Dict[str, str]:
    """Parse sections from non-JSON LLM response as fallback."""
    sections = {
        "purpose": "",
        "overview": "",
        "justification": "",
        "as_is": "",
        "to_be": "",
    }
    if not raw_text:
        return sections
    raw_text = _strip_markdown(raw_text)

    section_patterns = {
        "purpose": [
            r'(?:"?purpose"?\s*[:=]\s*"?)(.*?)(?="?\s*,?\s*"?(?:overview|justification|as.is|to.be|entities)"?\s*[:=]|\Z)',
            r"(?:purpose|document purpose)[:\s]*(.*?)(?=overview|justification|as.is|to.be|\Z)",
        ],
        "overview": [
            r'(?:"?overview"?\s*[:=]\s*"?)(.*?)(?="?\s*,?\s*"?(?:justification|as.is|to.be)"?\s*[:=]|\Z)',
        ],
        "justification": [
            r'(?:"?justification"?\s*[:=]\s*"?)(.*?)(?="?\s*,?\s*"?(?:as.is|to.be)"?\s*[:=]|\Z)',
        ],
        "as_is": [
            r'(?:"?as.is"?\s*[:=]\s*"?)(.*?)(?="?\s*,?\s*"?(?:to.be)"?\s*[:=]|\Z)',
        ],
        "to_be": [
            r'(?:"?to.be"?\s*[:=]\s*"?)(.*?)(?="?\s*}|\Z)',
        ],
    }

    for key, patterns in section_patterns.items():
        for pattern in patterns:
            m = re.search(pattern, raw_text, re.DOTALL | re.IGNORECASE)
            if m and len(m.group(1).strip()) > 30:
                val = m.group(1).strip().strip('"').strip()
                val = val.replace("\\n", "\n")
                sections[key] = _apply_tone_and_redaction(val)
                break

    found = sum(1 for v in sections.values() if v)
    if found > 0:
        print(f"    [DocBundle_Sections] Fallback parser recovered {found}/5 sections")
    return sections


# ============================================================
# Call 2: Process Data (steps, requirements) — IMPROVED
# ============================================================


def _generate_process_data(
    transcript: str, project_name: str, entities: Dict
) -> Dict[str, Any]:
    """
    LLM Call 2: Extract process steps, detailed steps, and all requirements tables.
    Enhanced prompts to capture sub-steps, conditionals, loops, data operations.
    """
    sample = safe_sample(transcript, max_len=config.llm.max_sample_text)
    min_steps = config.llm.min_process_steps
    max_steps = config.llm.max_process_steps
    min_detailed = config.llm.min_detailed_steps
    max_detailed = config.llm.max_detailed_steps

    apps_hint = ""
    if entities.get("applications"):
        apps_hint = f"Applications mentioned: {', '.join(entities['applications'])}"

    prompt = f"""You are a senior Business Analyst extracting automation process data from a meeting transcript.

Project: "{project_name}"
{apps_hint}

TASK: Extract ALL process steps and requirements. Be EXHAUSTIVE — capture every action, decision, validation, loop, and data operation discussed.

OUTPUT STRICT JSON with this exact structure:

{{
  "process_steps": [
    "High-level step description ({min_steps}-{max_steps} items)"
  ],
  "detailed_steps": [
    "Granular screen-level action description ({min_detailed}-{max_detailed} items)"
  ],
  "input_requirements": [
    {{"parameter": "Parameter Name", "description": "What it is and why needed"}}
  ],
  "interface_requirements": [
    {{"application": "App Name", "purpose": "Why the automation uses it"}}
  ],
  "exception_handling": [
    {{"exception": "Error scenario", "handling": "What the system does"}}
  ]
}}

CRITICAL STEP EXTRACTION RULES:

For process_steps ({min_steps}-{max_steps} items):
- Each step starts with an action verb: Access, Navigate, Export, Filter, Validate, Execute, Capture, Update, Generate, Repeat
- Include steps for: login/authentication, navigation, data export, data cleaning/filtering, validation/verification, conditional actions, iterative/repeated actions, reporting, logout
- If the process is performed for MULTIPLE systems/portals, include separate steps for each
- Include data manipulation steps: filtering, deduplication, extraction of specific columns
- Include validation/decision steps: checking status, verifying conditions
- Include reporting/logging steps: capturing counts, updating status, generating reports

For detailed_steps ({min_detailed}-{max_detailed} items):
- These are screen-level actions for PDD Section 2.4
- Each describes ONE specific UI action or system operation
- MUST include ALL of these action types when discussed in the transcript:
  a) Login and authentication steps (entering credentials, SSO, MFA)
  b) Navigation steps (clicking tabs, selecting menus, choosing options from dropdowns)
  c) Data export/download steps (clicking export, selecting format, downloading files)
  d) Data manipulation steps (opening files, applying filters, removing blanks, removing duplicates, extracting columns)
  e) Validation steps (checking each record against a system, querying status)
  f) Conditional/decision steps (if account is disabled then remove license, if data is blank then skip)
  g) Action execution steps (removing/revoking/assigning items, clicking buttons)
  h) Iteration/loop steps (repeating for each team/contract/user, processing next record)
  i) Post-action capture steps (recording counts, noting remaining items, capturing status)
  j) Reporting steps (updating tracking files, recording success/failure, saving reports)
  k) Sharing/distribution steps (saving to shared paths, sending reports)
  l) Logout/cleanup steps
- Write in imperative tone. Start each step with an action verb (e.g., Log in, Navigate, Click, Select).
- Be SPECIFIC: mention exact UI elements, button names, column names, dropdown values when discussed
- For loops/iterations: explicitly state "Repeat steps X through Y for each..."
- For conditionals: explicitly state "If [condition], [action]. Otherwise, [alternative]."

input_requirements (5-10 items):
- Parameters needed: credentials, URLs, file paths, scripts, configuration values.
- NEVER include actual passwords or personal data.

interface_requirements (3-8 items):
- Applications and systems the automation interacts with.
- ONLY names from the transcript.

exception_handling (6-10 items):
- Error scenarios with system response.
- Include: login failure, timeout, missing data, validation error, system crash, script error, file access denied.
- Handling must be in third person: "The system retries...", "The system logs..."

CRITICAL JSON RULES:
- Use ONLY application names from the transcript.
- NEVER mention the meeting, transcript, or speakers.
- NEVER include personal names, emails, or phone numbers.
- Write in imperative tone, present tense, active voice.
- DO NOT use markdown formatting.
- Keep all string values on a SINGLE LINE.
- Escape any double quotes inside strings with backslash: \\"
- Output STRICT valid JSON only.

TRANSCRIPT:
{sample}
"""

    resp = gemini_client.generate(
        prompt=prompt,
        system_prompt=get_system_prompt(),
        temperature=0.3,
        max_output_tokens=config.llm.max_output_tokens,
        call_name="DocBundle_ProcessData",
        max_retries=3,
    )

    result = {
        "process_steps": [],
        "detailed_steps": [],
        "input_requirements": [],
        "interface_requirements": [],
        "exception_handling": [],
    }

    json_text = _extract_json_object(resp or "")
    if json_text:
        try:
            data = json.loads(json_text)
            result["process_steps"] = _coerce_list_str(data.get("process_steps"))
            result["detailed_steps"] = _coerce_list_str(data.get("detailed_steps"))
            result["input_requirements"] = _coerce_list_dict(
                data.get("input_requirements"), ("parameter", "description")
            )
            result["interface_requirements"] = _coerce_list_dict(
                data.get("interface_requirements"), ("application", "purpose")
            )
            result["exception_handling"] = _coerce_list_dict(
                data.get("exception_handling"), ("exception", "handling")
            )
            print(
                f"    [DocBundle_ProcessData] Parsed: "
                f"{len(result['process_steps'])} steps, "
                f"{len(result['detailed_steps'])} detailed, "
                f"{len(result['input_requirements'])} inputs, "
                f"{len(result['interface_requirements'])} interfaces, "
                f"{len(result['exception_handling'])} exceptions"
            )
        except json.JSONDecodeError as e:
            print(f"    [DocBundle_ProcessData] JSON parse failed: {e}")
            result = _fallback_parse_process_data(resp or "")
    else:
        print("    [DocBundle_ProcessData] No JSON found, using fallback parser")
        result = _fallback_parse_process_data(resp or "")

    # Redact PII
    result["process_steps"] = [redact_pii_text(s) for s in result["process_steps"]]
    result["detailed_steps"] = [redact_pii_text(s) for s in result["detailed_steps"]]

    # Apply fallbacks for empty sections
    _ensure_process_data_defaults(result, entities)

    return result


def _ensure_process_data_defaults(result: Dict, entities: Dict):
    """Fill empty process data with generic templates."""
    apps = ", ".join(entities.get("applications", [])) or "the target application"

    if not result["process_steps"]:
        result["process_steps"] = [
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
        print("    [DocBundle_ProcessData] Using fallback process steps")

    if not result["detailed_steps"]:
        result["detailed_steps"] = [
            {
                "action": f"Open {apps} and navigate to the login page.",
                "ui_target": "Login page",
            },
            {
                "action": "Enter the configured credentials and authenticate.",
                "ui_target": "Sign in button",
            },
            {
                "action": "Navigate to the main processing module.",
                "ui_target": "Main menu",
            },
            {
                "action": "Select the appropriate data source or report.",
                "ui_target": "Data source dropdown",
            },
            {
                "action": "Apply the configured filters and criteria.",
                "ui_target": "Filter button",
            },
            {
                "action": "Extract the matching records from the source.",
                "ui_target": "Export button",
            },
            {
                "action": "Validate each record against the defined business rules.",
                "ui_target": "Validation screen",
            },
            {
                "action": "Flag records that fail validation checks.",
                "ui_target": "Flag button",
            },
            {
                "action": "Process each validated record according to the workflow.",
                "ui_target": "Process button",
            },
            {
                "action": "Update the record status after processing.",
                "ui_target": "Status dropdown",
            },
            {
                "action": "Capture processing results and timestamps.",
                "ui_target": "Results table",
            },
            {
                "action": "Generate a summary report of all processed records.",
                "ui_target": "Report generator",
            },
            {
                "action": "Export the report to the configured output location.",
                "ui_target": "Export report button",
            },
            {
                "action": "Log all execution details for audit purposes.",
                "ui_target": "Log viewer",
            },
            {
                "action": f"Close {apps} and terminate the session.",
                "ui_target": "Logout button",
            },
        ]
        print("    [DocBundle_ProcessData] Using fallback detailed steps")

    if not result["input_requirements"]:
        result["input_requirements"] = [
            {"parameter": "Application URL", "description": f"Web address of {apps}."},
            {
                "parameter": "User Credentials",
                "description": "Username and password for secure access.",
            },
            {
                "parameter": "Input Data Source",
                "description": "File path or database connection for source data.",
            },
            {
                "parameter": "Processing Criteria",
                "description": "Business rules and filters for record selection.",
            },
            {
                "parameter": "Output Location",
                "description": "File path or destination for generated reports.",
            },
        ]

    if not result["interface_requirements"]:
        result["interface_requirements"] = [
            {
                "application": "Target Application",
                "purpose": "Primary application for process execution.",
            },
        ]

    if not result["exception_handling"]:
        result["exception_handling"] = [
            {
                "exception": "Application Login Failure",
                "handling": "The system retries login up to 3 times. If authentication fails, the system stops and sends a notification.",
            },
            {
                "exception": "Element Not Found",
                "handling": "The system waits up to 30 seconds. If not found, the system captures a screenshot and logs the error.",
            },
            {
                "exception": "Data Validation Error",
                "handling": "The system flags the invalid record, logs details, and continues processing remaining items.",
            },
            {
                "exception": "Application Timeout",
                "handling": "The system retries the operation after a configured wait period and logs the timeout event.",
            },
            {
                "exception": "System Exception",
                "handling": "The system captures error details, saves a diagnostic screenshot, and terminates gracefully.",
            },
        ]


def _fallback_parse_process_data(raw_text: str) -> Dict[str, Any]:
    """Parse process data from non-JSON response."""
    result = {
        "process_steps": [],
        "detailed_steps": [],
        "input_requirements": [],
        "interface_requirements": [],
        "exception_handling": [],
    }
    if not raw_text:
        return result
    raw_text = _strip_markdown(raw_text)
    lines = raw_text.split("\n")
    current_section = None

    for line in lines:
        line = line.strip()
        lower = line.lower()
        if "process_step" in lower or "process step" in lower:
            current_section = "process_steps"
            continue
        elif "detailed_step" in lower or "detailed step" in lower:
            current_section = "detailed_steps"
            continue
        elif "input_req" in lower or "input req" in lower or "input_param" in lower:
            current_section = "input_requirements"
            continue
        elif "interface_req" in lower or "interface req" in lower:
            current_section = "interface_requirements"
            continue
        elif "exception" in lower and ("handl" in lower or ":" in line):
            current_section = "exception_handling"
            continue
        if not line or not current_section:
            continue
        cleaned = re.sub(r"^[\d]+[\.\)]\s*", "", line).strip()
        cleaned = re.sub(r"^[-•*]\s*", "", cleaned).strip()
        cleaned = cleaned.strip('"')
        if len(cleaned) < 10:
            continue
        if current_section in ("process_steps", "detailed_steps"):
            result[current_section].append(cleaned)
        elif current_section == "input_requirements":
            if "|" in cleaned:
                parts = cleaned.split("|", 1)
                result[current_section].append(
                    {
                        "parameter": parts[0].strip(),
                        "description": parts[1].strip() if len(parts) > 1 else "",
                    }
                )
        elif current_section == "interface_requirements":
            if "|" in cleaned:
                parts = cleaned.split("|", 1)
                result[current_section].append(
                    {
                        "application": parts[0].strip(),
                        "purpose": parts[1].strip() if len(parts) > 1 else "",
                    }
                )
        elif current_section == "exception_handling":
            if "|" in cleaned:
                parts = cleaned.split("|", 1)
                result[current_section].append(
                    {
                        "exception": parts[0].strip(),
                        "handling": parts[1].strip() if len(parts) > 1 else "",
                    }
                )

    found_sections = sum(1 for v in result.values() if v)
    if found_sections > 0:
        print(
            f"    [DocBundle_ProcessData] Fallback parser recovered {found_sections}/5 sections"
        )
    return result


# ============================================================
# Call 3: Step Refinement (NEW)
# ============================================================

# (Find the _refine_detailed_steps function in llm_tasks/meeting_compact.py and replace it)


def _refine_detailed_steps(
    transcript: str, project_name: str, coarse_steps: List[str], entities: Dict
) -> List[str]:
    """
    LLM Call 3: Expand coarse detailed steps into granular, EXACT UI sub-steps.
    Decomposes each step that implies multiple actions into explicit sub-steps.
    """
    if not coarse_steps:
        return coarse_steps

    if not config.llm.enable_step_refinement:
        print("    [StepRefine] Step refinement disabled")
        return coarse_steps

    min_target = config.llm.min_refined_steps
    max_target = config.llm.max_refined_steps

    sample = safe_sample(transcript, max_len=config.llm.max_sample_text)

    apps_hint = ""
    if entities.get("applications"):
        apps_hint = f"Applications: {', '.join(entities['applications'])}"

    steps_block = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(coarse_steps))

    prompt = f"""You are a strict, robotic Technical Writer refining process steps for a Process Definition Document (PDD).

Project: "{project_name}"
{apps_hint}

Below are coarse process steps extracted from a transcript. DECOMPOSE them into highly granular, mechanical UI steps.

CURRENT COARSE STEPS:
{steps_block}

CRITICAL MECHANICAL RULES (NO EXCEPTIONS):
1. ABSOLUTELY NO NARRATIVE FLUFF. Do NOT write "in the primary navigation menu" or "to complete the process". Just state the action.
2. COMBINE related navigation and selections using breadcrumbs (`->`) and commas. 
   Format: "Go to [Menu] -> [SubMenu], select [Dropdown] -> [Value], then click [Button]."
3. Quote EXACT text for buttons, tabs, dropdowns, and checkboxes mentioned in the transcript.
4. Do NOT skip intermediate file operations. Use: "Click 'Download' and open the file."
5. Every step MUST start with an action verb (e.g., Log in, Navigate, Click, Select). Do NOT start with "The system...".

=== STRICT FORMATTING EXAMPLES ===
BAD (Fluff): "The system navigates to the user management in the primary menu."
GOOD (Mechanical): "Navigate to the 'User Management' tab."

BAD (Too wordy): "The system clicks the team selection dropdown menu to view the list and selects the group."
GOOD (Mechanical): "Go to User Management -> By User, select Team -> 'Regional Group A', then click 'Export'."

BAD (Abstract): "The system filters the file."
GOOD (Mechanical): "Open the downloaded file, under 'Categories' select 'Users', then click 'Export'."

BAD (Narrative): "The system clicks download to get the file."
GOOD (Mechanical): "Click 'Download' and open the file."
============================

REFERENCE TRANSCRIPT (for context and exact UI wording):
{sample[:8000]}

OUTPUT: Numbered list of refined steps only. Target {min_target}-{max_target} steps.
1."""

    resp = gemini_client.generate(
        prompt=prompt,
        system_prompt=get_system_prompt(),
        temperature=0.1,  # Extremely low temperature to force strict adherence to format
        max_output_tokens=config.llm.max_output_tokens,
        call_name="DocBundle_StepRefine",
        max_retries=3,
    )

    refined = []
    if resp:
        if not resp.strip().startswith("1"):
            resp = "1. " + resp

        from core.utils import (
            parse_numbered_steps,
            filter_conversation_steps,
            deduplicate_steps,
        )

        parsed = parse_numbered_steps(resp)
        parsed = filter_conversation_steps(parsed)
        parsed = [redact_pii_text(s) for s in parsed]
        parsed = deduplicate_steps(parsed)

        if len(parsed) >= len(coarse_steps):
            refined = parsed
            print(
                f"    [StepRefine] Refined {len(coarse_steps)} -> {len(refined)} exact UI steps"
            )
        else:
            print(
                f"    [StepRefine] Refinement produced fewer steps ({len(parsed)}), keeping originals"
            )
            refined = coarse_steps
    else:
        print("    [StepRefine] No response, keeping original steps")
        refined = coarse_steps

    return refined


# ============================================================
# Public API
# ============================================================


def generate_pdd_bundle_batch(
    transcript: str,
    image_paths: List[str],
    audio_path: Optional[str] = None,
    project_name_hint: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Single batch LLM call to extract ALL PDD content.
    Takes transcript, keyframes, and optional audio to generate:
    - Document narrative sections
    - Process steps and requirements
    - Granular UI steps (Menu -> Submenu format)
    """
    start = time.time()
    sample = safe_sample(transcript, max_len=config.llm.max_sample_text)
    doc_type = config.document.document_type
    doc_full = config.document.document_type_full

    project_hint = (
        f'Project name hint: "{project_name_hint}"'
        if project_name_hint
        else "Derive a short descriptive project name (max 6 words) from the main process discussed."
    )

    prompt = f"""You are a senior Business Analyst creating a {doc_full} ({doc_type}).
You are provided with keyframe screenshots from the process execution, the spoken transcript, and potentially the audio file.

{project_hint}

TASK: Synthesize the full Process Definition Document as a STRICT JSON object.

JSON STRUCTURE REQUIRED:
{{
  "project_name": "Short name",
  "entities": {{"companies": [], "applications": [], "systems": [], "departments": []}},
  "document": {{
    "purpose": "Comprehensive 3-4 paragraph explanation of WHY this process automation is needed, including current pain points and business drivers.",
    "overview": "Detailed 2-3 paragraph description of WHAT the automation does, WHO is involved, and WHICH systems are used. Include specific system names, user roles, and data flows.",
    "justification": "Detailed breakdown of expected benefits with specific numbers where possible. Include time savings, error reduction, and compliance improvements.",
    "as_is": "Highly detailed step-by-step walkthrough of the CURRENT manual process. For each step, describe WHO does WHAT, WHICH screen/field is used, and WHAT data is handled. End with a numbered list of business challenges.",
    "to_be": "Highly detailed step-by-step description of the AUTOMATED process. For each step, specify WHICH button to click, WHICH menu to navigate, WHICH data fields are used, and HOW the automation handles exceptions."
  }},
  "process": {{
    "process_steps": [
      "Detailed actionable step 1: Describe exactly what action is taken (e.g., 'Login to Autodesk Portal using SSO credentials')",
      "Detailed actionable step 2: Describe exactly what action is taken (e.g., 'Navigate to User Management -> By User, select team from dropdown')"
    ],
    "detailed_steps": [
      {{
        "action": "Granular screen-level action with EXACT button names and navigation paths. Format with newlines for complex actions.",
        "ui_target": "Exact UI element for querying (e.g. 'Export button' or 'Al Futtaim dropdown')"
      }}
    ]
  }},
  "requirements": {{
    "input_requirements": [
      {{"parameter": "Param", "description": "Detailed description of what this input is, where it comes from, and format expected."}},
      {{"parameter": "Application URL", "description": "Web address of the target application (e.g., https://autodesk.example.com)."}},
      {{"parameter": "User Credentials", "description": "SSO credentials with appropriate permissions to access user management functions."}}
    ],
    "interface_requirements": [
      {{"application": "App Name", "purpose": "Detailed description of how this application is used in the process."}}
    ],
    "exception_handling": [
      {{"exception": "Error Scenario", "handling": "Detailed step-by-step handling action."}}
    ]
  }}
}}

CRITICAL RULES FOR DOCUMENT SECTIONS:
1. PURPOSE: Write as a compelling business case. Explain WHY this automation is critical. Reference specific pain points from the transcript/screenshots.
2. OVERVIEW: Be specific about systems, users, and data. Don't say "the system" - say EXACTLY which application, screen, or module.
3. AS-IS: Write a chronological walkthrough of the MANUAL process. Each sentence should tell the reader exactly what happens. Include specific UI elements, data values, and decisions.
4. TO-BE: Write from the automation's perspective. Every sentence should describe what the bot clicks, navigates to, or data it processes. Be specific about button names, menu paths, checkbox states.

CRITICAL RULES FOR process_steps (High-Level Logic):
- These are the main automation actions - write them as actionable instructions someone could follow
- Start each with an action verb (e.g., "Log in to...", "Navigate to...", "Export...")
- Include specific application names, menu paths, and data elements
- DO NOT summarize - be specific about what happens at each major phase

CRITICAL RULES FOR detailed_steps (Screen-level Actions):
1. ABSOLUTELY NO NARRATIVE FLUFF. Do NOT write "in the primary navigation menu".
2. COMBINE related navigation using breadcrumbs (`->`). Use commas and newlines for complex actions.
   Format: "Go to [Menu] -> [SubMenu], \nselect [Dropdown] -> [Value], \nthen click [Button]."
3. STRICT ANTI-HALLUCINATION: Quote EXACT text for buttons, tabs, dropdowns, and checkboxes VISIBLE in the screenshots or mentioned in the transcript. NEVER invent or use example names like 'Alpha team'. If you don't know the exact name, use a generic identifier like 'the target team'.
4. STRICT CHRONOLOGICAL ORDER: Follow the exact chronological sequence of actions (e.g., selecting a column must happen before clicking an action button).
5. CONDITIONALS: If a conditional pop-up occurs, state it clearly: "If a Warning popup appears -> Select 'Continue', Then click 'Remove'."
6. Every step MUST start with an action verb (e.g., Log in, Navigate, Click, Select, Go to). Do NOT start with "The system...".
7. The `ui_target` field MUST be a hyper-focused, 3-5 word description of the PRIMARY button or element being interacted with in this step (e.g., "Export button", "User Management tab", "Categories Users checkbox"). This field is used for visual bounding-box search.

TRANSCRIPT:
{sample}
"""

    print("    [DocBundle_Batch] Sending unified generation call...")
    resp = gemini_client.generate(
        prompt=prompt,
        system_prompt=get_system_prompt(),
        image_paths=image_paths,
        audio_path=audio_path,
        temperature=0.2,
        max_output_tokens=config.llm.max_output_tokens,
        call_name="DocBundle_Batch",
        max_retries=3,
    )

    result = {
        "project_name": project_name_hint or "Process Automation Project",
        "entities": {
            "companies": [],
            "applications": [],
            "systems": [],
            "departments": [],
        },
        "document": {
            "purpose": "",
            "overview": "",
            "justification": "",
            "as_is": "",
            "to_be": "",
        },
        "process": {"process_steps": [], "detailed_steps": []},
        "requirements": {
            "input_requirements": [],
            "interface_requirements": [],
            "exception_handling": [],
        },
    }

    json_text = _extract_json_object(resp or "")
    if json_text:
        try:
            data = json.loads(json_text)

            # Map sections
            pn = str(data.get("project_name", "")).strip()
            if pn:
                result["project_name"] = pn

            ent = data.get("entities", {})
            if isinstance(ent, dict):
                for key in ["companies", "applications", "systems", "departments"]:
                    result["entities"][key] = _coerce_list_str(ent.get(key, []))

            doc = data.get("document", {})
            for key in ["purpose", "overview", "justification", "as_is", "to_be"]:
                val = _strip_markdown(str(doc.get(key, "")).strip())
                result["document"][key] = _apply_tone_and_redaction(val)

            proc = data.get("process", {})
            result["process"]["process_steps"] = [
                redact_pii_text(s) for s in _coerce_list_str(proc.get("process_steps"))
            ]

            detailed_raw = proc.get("detailed_steps", [])
            parsed_detailed = []
            if isinstance(detailed_raw, list):
                for step in detailed_raw:
                    if isinstance(step, dict):
                        action = str(step.get("action", "")).strip()
                        target = str(step.get("ui_target", "")).strip()
                        if action:
                            parsed_detailed.append(
                                {"action": redact_pii_text(action), "ui_target": target}
                            )
                    elif isinstance(step, str):
                        parsed_detailed.append(
                            {"action": redact_pii_text(step), "ui_target": step}
                        )
            result["process"]["detailed_steps"] = parsed_detailed

            reqs = data.get("requirements", {})
            result["requirements"]["input_requirements"] = _coerce_list_dict(
                reqs.get("input_requirements"), ("parameter", "description")
            )
            result["requirements"]["interface_requirements"] = _coerce_list_dict(
                reqs.get("interface_requirements"), ("application", "purpose")
            )
            result["requirements"]["exception_handling"] = _coerce_list_dict(
                reqs.get("exception_handling"), ("exception", "handling")
            )

            print(f"    [DocBundle_Batch] Parsed successfully")
        except json.JSONDecodeError as e:
            print(f"    [DocBundle_Batch] JSON parse failed: {e}")
    else:
        print("    [DocBundle_Batch] No JSON found in response")

    _ensure_section_defaults(result)

    temp_result = {
        "process_steps": result["process"]["process_steps"],
        "detailed_steps": result["process"]["detailed_steps"],
        "input_requirements": result["requirements"]["input_requirements"],
        "interface_requirements": result["requirements"]["interface_requirements"],
        "exception_handling": result["requirements"]["exception_handling"],
    }
    _ensure_process_data_defaults(temp_result, result["entities"])
    result["process"]["process_steps"] = temp_result["process_steps"]
    result["process"]["detailed_steps"] = temp_result["detailed_steps"]
    result["requirements"]["input_requirements"] = temp_result["input_requirements"]
    result["requirements"]["interface_requirements"] = temp_result[
        "interface_requirements"
    ]
    result["requirements"]["exception_handling"] = temp_result["exception_handling"]

    timed("DocBundle_Batch", start)
    return result


def generate_dot_from_transcript(
    transcript: str, project_name: str, process_steps: Optional[List[str]] = None
) -> str:
    """
    Single LLM call to generate DOT flowchart code.
    Enhanced to include decision diamonds, loops, and parallel paths.
    """
    start = time.time()
    sample = safe_sample(transcript, max_len=config.llm.max_sample_text)

    steps_block = ""
    if process_steps:
        steps_block = "PROCESS STEPS (use these as nodes):\n" + "\n".join(
            f"{i + 1}. {s[:100]}" for i, s in enumerate(process_steps[:20])
        )

    max_words = config.flowchart.max_label_words

    prompt = f"""Generate a Graphviz DOT flowchart for this automation process.

Process: "{project_name}"

{steps_block}

STRICT FORMATTING RULES:
1. Output ONLY valid DOT code. No markdown, no explanation, no code fences.
2. Must start with: digraph ProcessFlow {{
3. Must include Start (oval, green) and End (oval, red) nodes.
4. Use rankdir=TB.
5. Every node label MUST be MAX {max_words} WORDS. Short verb+object phrases only.
   GOOD: "Login to Portal", "Export User List", "Validate in AD", "Remove License"
   BAD: "The system validates each record against the defined criteria"
6. Use diamond shape for DECISION NODES where the process branches based on a condition.
   Decision nodes should have Yes/No or True/False edge labels.
   Examples of decisions: "Account Disabled?", "Valid User?", "More Items?", "Export Success?"
7. Use box shape for process steps.
8. Colors: process=lightblue, decision=gold, start=lightgreen, end=lightcoral.
9. All nodes must have style=filled.
10. Include LOOP STRUCTURES: when a set of steps repeats for each item, add a decision diamond that loops back.
    Example: after processing a user, add "More Users?" diamond that goes back to the processing step (Yes) or forward (No).
11. Include PARALLEL PATHS: if the process handles multiple systems (e.g., two portals), show the flow going through one then the other.
12. Both branches of a decision should eventually reconnect to the main flow or lead to End.
13. CRITICAL: Every single node MUST be connected to the graph with at least one incoming and one outgoing edge (except Start and End). Do NOT leave any node floating or disconnected. Ensure all steps are linked via arrows (`->`).

TRANSCRIPT (context only):
{sample}
"""

    resp = gemini_client.generate(
        prompt=prompt,
        system_prompt=get_system_prompt(),
        temperature=0.2,
        call_name="DOT_FromTranscript",
        max_retries=3,
    )

    dot = (resp or "").strip()
    dot = re.sub(r"^```(?:dot|graphviz)?\s*", "", dot)
    dot = re.sub(r"\s*```$", "", dot)
    dot = dot.strip()

    m = re.search(r"(digraph\s+\w*\s*\{.*\})", dot, re.DOTALL)
    if m:
        dot = m.group(1).strip()

    dot = _enforce_short_labels(dot, max_words)

    timed("DOT_FromTranscript", start)
    return dot


def _enforce_short_labels(dot_code: str, max_words: int = 6) -> str:
    """Post-process DOT code to enforce short labels."""
    if not dot_code:
        return dot_code

    def _shorten(match):
        prefix = match.group(1)
        label = match.group(2)
        suffix = match.group(3)

        label = re.sub(
            r"\b(the|a|an|to|of|for|in|on|at|by|with|and|is|are|was|were|been|being)\b",
            " ",
            label,
            flags=re.IGNORECASE,
        )
        label = re.sub(r"\s+", " ", label).strip()

        words = label.split()
        if len(words) > max_words:
            label = " ".join(words[:max_words])

        if label:
            label = label[0].upper() + label[1:]

        return f'{prefix}"{label}"{suffix}'

    result = re.sub(r'(label\s*=\s*)"([^"]+)"(\s*[,\]])', _shorten, dot_code)
    return result
