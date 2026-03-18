# pipeline/common.py

"""
Shared pipeline logic used by both audio and video pipelines.
Document assembly, flowchart generation, file saving.
"""

import os
import shutil
import time
from typing import Optional, List, Dict

from core.config import config
from core.gemini_client import gemini_client
from core.token_tracker import TokenTracker
from document.flowchart_renderer import generate_flowchart_from_dot
from document.pdd_generator import PDDGenerator


def save_persistent_document(doc_path: str, project_name: str) -> str:
    """Copy generated document to persistent outputs/ directory."""
    d = config.paths.output_dir
    os.makedirs(d, exist_ok=True)
    safe = "".join(
        c for c in project_name if c.isalnum() or c in (' ', '-', '_')
    ).strip()[:50] or "PDD"
    doc_type = config.document.document_type
    path = os.path.join(d, f"{safe}_{doc_type}.docx")
    c = 1
    base = path
    while os.path.exists(path):
        path = f"{base.rsplit('.', 1)[0]}_{c}.docx"
        c += 1
    shutil.copy2(doc_path, path)
    print(f"  📁 Saved: {path}")
    return path


def save_dot_code(dot_code: str, project_name: str, output_dir: str):
    """Save DOT code to both output_dir and persistent outputs/."""
    if not dot_code:
        return
    d = config.paths.output_dir
    os.makedirs(d, exist_ok=True)
    safe = "".join(
        c for c in project_name if c.isalnum() or c in (' ', '-', '_')
    ).strip()[:50] or "flowchart"
    path = os.path.join(d, f"{safe}_flowchart.dot")
    c = 1
    base = path
    while os.path.exists(path):
        path = f"{base.rsplit('.', 1)[0]}_{c}.dot"
        c += 1
    with open(path, 'w', encoding='utf-8') as f:
        f.write(dot_code)
    print(f"  📁 DOT: {path}")
    # Also save locally
    local = os.path.join(output_dir, f"{safe}_flowchart.dot")
    if local != path:
        os.makedirs(output_dir, exist_ok=True)
        with open(local, 'w', encoding='utf-8') as f:
            f.write(dot_code)


def save_flowchart_persistent(
    flowchart_path: str, dot_code: str, project_name: str
):
    """Save flowchart SVG/PNG and DOT to persistent outputs/."""
    d = config.paths.output_dir
    os.makedirs(d, exist_ok=True)
    safe = "".join(
        c for c in project_name if c.isalnum() or c in (' ', '-', '_')
    ).strip()[:50] or "flowchart"

    if flowchart_path and os.path.exists(flowchart_path):
        # Determine extension from the actual file
        _, ext = os.path.splitext(flowchart_path)
        ext = ext or '.svg'
        dest = os.path.join(d, f"{safe}_flowchart{ext}")
        c = 1
        while os.path.exists(dest):
            dest = os.path.join(d, f"{safe}_flowchart_{c}{ext}")
            c += 1
        shutil.copy2(flowchart_path, dest)
        print(f"  📁 Flowchart: {dest}")

    if dot_code:
        dot_dest = os.path.join(d, f"{safe}_flowchart.dot")
        c = 1
        while os.path.exists(dot_dest):
            dot_dest = os.path.join(d, f"{safe}_flowchart_{c}.dot")
            c += 1
        with open(dot_dest, 'w', encoding='utf-8') as f:
            f.write(dot_code)
        print(f"  📁 Flowchart DOT: {dot_dest}")


def generate_flowchart(
    dot_code: str, output_dir: str, project_name: str
) -> str:
    """Generate flowchart SVG from DOT code and save persistently."""
    if not dot_code:
        return ""
    fc_output = os.path.join(output_dir, "flowchart")
    result = generate_flowchart_from_dot(dot_code, fc_output)
    if result:
        save_flowchart_persistent(result, dot_code, project_name)
        return result
    else:
        save_flowchart_persistent("", dot_code, project_name)
        return ""


def build_document(
    project_name: str,
    output_dir: str,
    purpose: str = "",
    overview: str = "",
    justification: str = "",
    as_is: str = "",
    to_be: str = "",
    process_steps: list = None,
    input_requirements: list = None,
    detailed_steps: list = None,
    interface_requirements: list = None,
    exception_handling: list = None,
    flowchart_path: str = "",
    app_name: str = "",
) -> str:
    """
    Build the final PDD/BRD DOCX document.
    Used by both audio and video pipelines.
    """
    safe = "".join(
        c for c in project_name if c.isalnum() or c in (' ', '-', '_')
    ).strip()[:50] or "PDD"
    doc_type = config.document.document_type
    doc_path = os.path.join(output_dir, f"{safe}_{doc_type}.docx")

    gen = PDDGenerator()
    gen.generate(
        project_name=project_name,
        app_name=app_name,
        document_purpose=purpose,
        overview=overview,
        justification=justification,
        as_is=as_is,
        to_be=to_be,
        process_steps=process_steps,
        input_requirements=input_requirements,
        detailed_steps=detailed_steps,
        interface_requirements=interface_requirements,
        exception_handling=exception_handling,
        flowchart_path=flowchart_path,
        output_path=doc_path,
    )
    return doc_path


def print_pipeline_header(
    pipeline_name: str,
    video_path: str = "",
    project_name: str = "",
    extra_info: Dict[str, str] = None
):
    """Print formatted pipeline header."""
    print("=" * 65)
    print(f"PDD Agent — {pipeline_name}")
    if video_path:
        print(f"  Video: {video_path}")
    if project_name:
        print(f"  Project: {project_name}")
    print(f"  Document: {config.document.document_type_full}")
    print(f"  Model: {config.gemini.text_model}")
    if extra_info:
        for k, v in extra_info.items():
            print(f"  {k}: {v}")
    print("=" * 65)


def print_pipeline_footer(
    doc_path: str,
    project_name: str,
    stats: Dict[str, any],
    total_time: float
):
    """Print formatted pipeline footer with stats."""
    print(f"\n{'='*65}")
    print(f"✓ PDD Complete!")
    print(f"  Document: {doc_path}")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print(f"  Time: {total_time/60:.1f} minutes")
    print(f"{'='*65}")