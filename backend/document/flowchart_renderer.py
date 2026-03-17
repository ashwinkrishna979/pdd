# document/flowchart_renderer.py

"""
Flowchart generation from DOT code.
Renders LLM-generated DOT code using Graphviz.
Outputs SVG with gradient fills, no borders, orthogonal arrows, Segoe UI font.
"""

import re
import os
from typing import Dict, List, Optional

try:
    from graphviz import Source, Digraph
    GRAPHVIZ_AVAILABLE = True
except ImportError:
    GRAPHVIZ_AVAILABLE = False
    print("    [Flowchart] graphviz not installed. Install: pip install graphviz")


# ============================================================
# Color Theme Constants
# ============================================================

THEME = {
    # Diamond (decision) - Pink/Magenta gradient with white text
    "decision_fill": "#d80073",
    "decision_fill_light": "#f179a3",
    "decision_text": "white",

    # Rectangle (process) - Blue gradient with white text
    "process_fill": "#1ba1e2",
    "process_fill_light": "#8abbe7",
    "process_text": "white",

    # Start/End - Grey rounded rectangle, white text, no border
    "terminal_fill": "#c9c9c9",
    "terminal_text": "white",

    # Parallelogram (input/output) - Orange gradient with white text
    "parallelogram_fill": "#f09609",
    "parallelogram_fill_light": "#fabf6c",
    "parallelogram_text": "white",

    # Arrows - Blue with blue text on top
    "arrow_color": "#1ba1e2",
    "arrow_text": "#1ba1e2",

    # General
    "font_name": "Segoe UI",
    "font_size": "10",
}

# ============================================================
# SVG Gradient Injection
# ============================================================

SVG_GRADIENT_DEFS = f"""
  <linearGradient id="gradProcess" x1="0%" y1="0%" x2="0%" y2="100%">
    <stop offset="0%" style="stop-color:{THEME['process_fill_light']};stop-opacity:1" />
    <stop offset="100%" style="stop-color:{THEME['process_fill']};stop-opacity:1" />
  </linearGradient>
  <linearGradient id="gradDecision" x1="0%" y1="0%" x2="0%" y2="100%">
    <stop offset="0%" style="stop-color:{THEME['decision_fill_light']};stop-opacity:1" />
    <stop offset="100%" style="stop-color:{THEME['decision_fill']};stop-opacity:1" />
  </linearGradient>
  <linearGradient id="gradParallelogram" x1="0%" y1="0%" x2="0%" y2="100%">
    <stop offset="0%" style="stop-color:{THEME['parallelogram_fill_light']};stop-opacity:1" />
    <stop offset="100%" style="stop-color:{THEME['parallelogram_fill']};stop-opacity:1" />
  </linearGradient>
"""


def _inject_svg_gradients(svg_content: str) -> str:
    """
    Post-process SVG output to:
    1. Inject gradient definitions into <defs>.
    2. Replace flat fill colors with gradient references.
    3. Remove all borders (stroke) from shapes.
    """
    if not svg_content:
        return svg_content

    if '<defs>' in svg_content:
        svg_content = svg_content.replace(
            '<defs>',
            f'<defs>{SVG_GRADIENT_DEFS}',
            1
        )
    elif '</svg>' in svg_content:
        svg_match = re.search(r'(<svg[^>]*>)', svg_content)
        if svg_match:
            insert_pos = svg_match.end()
            svg_content = (
                svg_content[:insert_pos] +
                f'\n<defs>{SVG_GRADIENT_DEFS}</defs>\n' +
                svg_content[insert_pos:]
            )

    svg_content = _replace_fill_with_gradient(
        svg_content, THEME["process_fill"], "url(#gradProcess)"
    )
    svg_content = _replace_fill_with_gradient(
        svg_content, THEME["process_fill_light"], "url(#gradProcess)"
    )
    svg_content = _replace_fill_with_gradient(
        svg_content, THEME["decision_fill"], "url(#gradDecision)"
    )
    svg_content = _replace_fill_with_gradient(
        svg_content, THEME["decision_fill_light"], "url(#gradDecision)"
    )
    svg_content = _replace_fill_with_gradient(
        svg_content, THEME["parallelogram_fill"], "url(#gradParallelogram)"
    )
    svg_content = _replace_fill_with_gradient(
        svg_content, THEME["parallelogram_fill_light"], "url(#gradParallelogram)"
    )

    svg_content = _remove_shape_borders(svg_content)

    return svg_content


def _replace_fill_with_gradient(svg_content: str, hex_color: str, gradient_ref: str) -> str:
    """Replace a hex fill color with a gradient reference in SVG."""
    hex_lower = hex_color.lower()
    hex_upper = hex_color.upper()

    svg_content = svg_content.replace(f'fill="{hex_lower}"', f'fill="{gradient_ref}"')
    svg_content = svg_content.replace(f'fill="{hex_upper}"', f'fill="{gradient_ref}"')
    svg_content = svg_content.replace(f'fill="{hex_color}"', f'fill="{gradient_ref}"')

    svg_content = re.sub(
        rf'fill:\s*{re.escape(hex_color)}',
        f'fill:{gradient_ref}',
        svg_content,
        flags=re.IGNORECASE
    )

    return svg_content


def _remove_shape_borders(svg_content: str) -> str:
    """
    Remove stroke (border) from filled shapes in SVG.
    Preserves stroke on edges/arrows.
    """
    lines = svg_content.split('\n')
    result_lines = []
    i = 0

    while i < len(lines):
        line = lines[i]

        is_shape = bool(re.search(r'<(polygon|ellipse|rect|path)\s', line))

        has_gradient = 'url(#grad' in line
        has_theme_fill = any(c.lower() in line.lower() for c in [
            THEME["process_fill"], THEME["decision_fill"],
            THEME["parallelogram_fill"], THEME["terminal_fill"]
        ])

        is_edge = False
        context_start = max(0, i - 5)
        context = '\n'.join(lines[context_start:i + 1])
        if 'class="edge"' in context or 'marker-end' in line:
            is_edge = True

        if is_shape and (has_gradient or has_theme_fill) and not is_edge:
            line = re.sub(r'\bstroke="[^"]*"', 'stroke="none"', line)
            line = re.sub(r'\bstroke-width="[^"]*"', 'stroke-width="0"', line)
            line = re.sub(r'stroke:\s*[^;"]+', 'stroke:none', line)
            line = re.sub(r'stroke-width:\s*[^;"]+', 'stroke-width:0', line)

        result_lines.append(line)
        i += 1

    return '\n'.join(result_lines)


# ============================================================
# DOT Code Styling
# ============================================================

def _strip_attr(attrs: str, attr_name: str) -> str:
    """
    Safely remove a single DOT attribute from an attribute string.
    Handles: attr=value, attr="value", attr="multi,value"
    """
    # Pattern 1: attr="quoted value" followed by optional comma
    result = re.sub(
        rf'\b{attr_name}\s*=\s*"[^"]*"\s*,?\s*',
        '', attrs
    )
    # Pattern 2: attr=unquoted_value followed by optional comma
    result = re.sub(
        rf'\b{attr_name}\s*=\s*[^",\]\s]+\s*,?\s*',
        '', result
    )
    return result


def _clean_attrs(attrs: str) -> str:
    """Clean up attribute string after removals."""
    attrs = re.sub(r',\s*,', ',', attrs)
    attrs = re.sub(r'^\s*,\s*', '', attrs)
    attrs = re.sub(r'\s*,\s*$', '', attrs)
    attrs = attrs.strip()
    return attrs


def _is_terminal_node(node_name: str, attrs: str) -> bool:
    """Check if a node is a start/end terminal node by name or label."""
    name_lower = node_name.lower()
    
    # Check node name
    if any(kw in name_lower for kw in ('start', 'end', 'begin', 'finish', 'stop', 'terminate')):
        return True
    
    # Check label
    label_match = re.search(r'label\s*=\s*"([^"]*)"', attrs, re.IGNORECASE)
    if label_match:
        label_text = label_match.group(1).lower()
        if any(kw in label_text for kw in ('start', 'end', 'begin', 'finish', 'stop', 'terminate')):
            return True
    
    return False


def _inject_theme_style(dot_code: str) -> str:
    """
    Inject the custom visual theme into DOT code.
    Enforces orthogonal arrows and Segoe UI font.
    """
    if not dot_code or not dot_code.strip():
        return dot_code

    match = re.search(r'(digraph\s+\w*\s*\{)', dot_code)
    if not match:
        return dot_code

    font = THEME['font_name']

    # Edge label positioning: decorate=true puts label on the edge line
    # headlabel/taillabel with labeldistance controls position
    style_block = f"""
    dpi=150;
    size="8,10";
    bgcolor="white";
    splines=ortho;
    nodesep=0.8;
    ranksep=1.0;

    node [fontname="{font}", fontsize={THEME['font_size']}];

    edge [fontname="{font}", fontsize=9, color="{THEME['arrow_color']}",
          fontcolor="{THEME['arrow_text']}", arrowsize=0.8, penwidth=1.5,
          decorate=false, labelfloat=true, labeldistance=1.5];
"""

    insert_pos = match.end()
    styled = dot_code[:insert_pos] + style_block + dot_code[insert_pos:]

    styled = _apply_node_themes(styled)
    styled = _apply_edge_themes(styled)

    return styled


def _apply_node_themes(dot_code: str) -> str:
    """Apply color themes to different node shapes."""

    # 1. Diamond nodes (decisions) - but check if it's actually a terminal
    dot_code = re.sub(
        r'(\w+)\s*\[([^\]]*\bshape\s*=\s*diamond\b[^\]]*)\]',
        lambda m: _restyle_node(m, "diamond"),
        dot_code,
        flags=re.IGNORECASE
    )

    # 2. Oval/ellipse nodes (start/end) -> rounded rectangle
    dot_code = re.sub(
        r'(\w+)\s*\[([^\]]*\bshape\s*=\s*(?:oval|ellipse)\b[^\]]*)\]',
        lambda m: _restyle_node(m, "terminal"),
        dot_code,
        flags=re.IGNORECASE
    )

    # 3. Parallelogram nodes
    dot_code = re.sub(
        r'(\w+)\s*\[([^\]]*\bshape\s*=\s*parallelogram\b[^\]]*)\]',
        lambda m: _restyle_node(m, "parallelogram"),
        dot_code,
        flags=re.IGNORECASE
    )

    # 4. Box/rectangle nodes - check if terminal first
    dot_code = re.sub(
        r'(\w+)\s*\[([^\]]*\bshape\s*=\s*(?:box|rect|rectangle)\b[^\]]*)\]',
        lambda m: _restyle_node(m, "terminal" if _is_terminal_node(m.group(1), m.group(2)) else "box"),
        dot_code,
        flags=re.IGNORECASE
    )

    # 5. Circle/doublecircle nodes
    dot_code = re.sub(
        r'(\w+)\s*\[([^\]]*\bshape\s*=\s*(?:circle|doublecircle)\b[^\]]*)\]',
        lambda m: _restyle_node(m, "circle"),
        dot_code,
        flags=re.IGNORECASE
    )

    # 6. Nodes without explicit shape but with label
    dot_code = re.sub(
        r'(\w+)\s*\[([^\]]*\blabel\s*=\s*"[^"]*"[^\]]*)\]',
        lambda m: _restyle_node_if_no_shape(m),
        dot_code,
        flags=re.IGNORECASE
    )

    return dot_code


def _restyle_node(match, node_type: str) -> str:
    """Restyle a node with the appropriate theme colors."""
    node_name = match.group(1)
    attrs = match.group(2)

    # Check if this should be a terminal node regardless of original shape
    if node_type != "terminal" and _is_terminal_node(node_name, attrs):
        node_type = "terminal"

    # Strip all visual attributes safely
    for attr in ['fillcolor', 'color', 'fontcolor', 'style', 'penwidth', 'fontname', 'shape']:
        attrs = _strip_attr(attrs, attr)
    attrs = _clean_attrs(attrs)

    font = THEME["font_name"]

    if node_type == "diamond":
        new_style = (
            f'shape=diamond, style=filled, '
            f'fillcolor="{THEME["decision_fill"]}", '
            f'fontcolor="{THEME["decision_text"]}", '
            f'color="transparent", penwidth=0, fontname="{font}"'
        )
    elif node_type == "terminal":
        # Start/End - grey rounded rectangle, white text, no border
        new_style = (
            f'shape=box, style="filled,rounded", '
            f'fillcolor="{THEME["terminal_fill"]}", '
            f'fontcolor="{THEME["terminal_text"]}", '
            f'color="transparent", penwidth=0, fontname="{font}"'
        )
    elif node_type == "parallelogram":
        new_style = (
            f'shape=parallelogram, style=filled, '
            f'fillcolor="{THEME["parallelogram_fill"]}", '
            f'fontcolor="{THEME["parallelogram_text"]}", '
            f'color="transparent", penwidth=0, fontname="{font}"'
        )
    elif node_type == "circle":
        new_style = (
            f'shape=circle, style=filled, '
            f'fillcolor="{THEME["process_fill"]}", '
            f'fontcolor="{THEME["process_text"]}", '
            f'color="transparent", penwidth=0, fontname="{font}"'
        )
    else:
        # Default box/process - blue
        new_style = (
            f'shape=box, style=filled, '
            f'fillcolor="{THEME["process_fill"]}", '
            f'fontcolor="{THEME["process_text"]}", '
            f'color="transparent", penwidth=0, fontname="{font}"'
        )

    if attrs:
        result = f'{node_name} [{attrs}, {new_style}]'
    else:
        result = f'{node_name} [{new_style}]'

    return result


def _restyle_node_if_no_shape(match) -> str:
    """Restyle node only if it doesn't already have a shape defined."""
    node_name = match.group(1)
    attrs = match.group(2)

    if re.search(r'\bshape\s*=', attrs, re.IGNORECASE):
        return match.group(0)

    if THEME["process_fill"] in attrs or THEME["decision_fill"] in attrs or THEME["terminal_fill"] in attrs:
        return match.group(0)

    if node_name.lower() in ('node', 'edge', 'graph'):
        return match.group(0)

    # Check if terminal node
    if _is_terminal_node(node_name, attrs):
        return _restyle_node(match, "terminal")

    return _restyle_node(match, "box")


def _apply_edge_themes(dot_code: str) -> str:
    """Apply theme colors to edges/arrows."""

    # Edges with existing attributes
    dot_code = re.sub(
        r'(\w+\s*->\s*\w+)\s*\[([^\]]*)\]',
        lambda m: _restyle_edge(m),
        dot_code
    )

    # Edges without attributes
    dot_code = re.sub(
        r'(\w+\s*->\s*\w+)\s*;',
        lambda m: (
            f'{m.group(1)} [color="{THEME["arrow_color"]}", '
            f'fontcolor="{THEME["arrow_text"]}", '
            f'penwidth=1.5, fontname="{THEME["font_name"]}"];'
        ),
        dot_code
    )

    return dot_code


def _restyle_edge(match) -> str:
    """Restyle an edge with theme colors and proper label positioning."""
    edge_def = match.group(1)
    attrs = match.group(2)

    # Strip visual attributes but keep label
    for attr in ['color', 'fontcolor', 'penwidth', 'style', 'fontname', 'decorate', 'labelfloat', 'labeldistance', 'labelangle']:
        attrs = _strip_attr(attrs, attr)
    attrs = _clean_attrs(attrs)

    # Label positioning for edge labels to appear on top/center of arrow
    new_style = (
        f'color="{THEME["arrow_color"]}", '
        f'fontcolor="{THEME["arrow_text"]}", '
        f'penwidth=1.5, fontname="{THEME["font_name"]}", '
        f'decorate=false, labelfloat=true'
    )

    if attrs:
        return f'{edge_def} [{attrs}, {new_style}]'
    else:
        return f'{edge_def} [{new_style}]'


# ============================================================
# DOT Validation and Fixing
# ============================================================

def _validate_dot_code(dot_code: str) -> bool:
    """Basic validation that DOT code is structurally sound."""
    if not dot_code:
        return False
    if 'digraph' not in dot_code:
        return False

    open_count = dot_code.count('{')
    close_count = dot_code.count('}')
    if open_count != close_count:
        print(f"    [Flowchart] Warning: Unbalanced braces ({open_count} open, {close_count} close)")
        return False

    if '->' not in dot_code:
        print("    [Flowchart] Warning: No edges found in DOT code")
        return False

    return True


def _fix_common_dot_issues(dot_code: str) -> str:
    """Fix common issues in LLM-generated DOT code."""
    open_count = dot_code.count('{')
    close_count = dot_code.count('}')
    if open_count > close_count:
        dot_code = dot_code + '\n}' * (open_count - close_count)
    elif close_count > open_count:
        for _ in range(close_count - open_count):
            last_brace = dot_code.rfind('}')
            if last_brace != -1:
                dot_code = dot_code[:last_brace] + dot_code[last_brace + 1:]

    dot_code = re.sub(
        r'(end\s*\[.*?)style\s*=\s*invis(.*?\])',
        r'\1style=filled\2',
        dot_code,
        flags=re.IGNORECASE
    )

    dot_code = re.sub(r'\bsplines\s*=\s*\w+\s*;?\s*', '', dot_code)

    return dot_code


# ============================================================
# Rendering
# ============================================================

def render_dot_direct(dot_code: str, output_path: str, fmt: str = 'svg') -> Optional[str]:
    """
    Render DOT code directly using graphviz.Source.
    Default output format is SVG for gradient support.
    """
    if not GRAPHVIZ_AVAILABLE:
        print("    [Flowchart] Graphviz not available")
        return None

    try:
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        src = Source(dot_code, format=fmt)
        rendered_path = src.render(output_path, cleanup=True)

        if fmt == 'svg' and rendered_path and os.path.exists(rendered_path):
            try:
                with open(rendered_path, 'r', encoding='utf-8') as f:
                    svg_content = f.read()

                svg_content = _inject_svg_gradients(svg_content)

                with open(rendered_path, 'w', encoding='utf-8') as f:
                    f.write(svg_content)

                print(f"    [Flowchart] Rendered + gradients applied: {rendered_path}")
            except Exception as e:
                print(f"    [Flowchart] Gradient injection warning: {e}")
                print(f"    [Flowchart] Rendered (without gradients): {rendered_path}")
        else:
            print(f"    [Flowchart] Rendered: {rendered_path}")

        return rendered_path

    except Exception as e:
        print(f"    [Flowchart] Render failed: {e}")
        return None


def _try_render_with_fallback_splines(dot_code: str, output_path: str) -> Optional[str]:
    """
    Try rendering with ortho splines first.
    Falls back to polyline then line if ortho fails.
    """
    result = render_dot_direct(dot_code, output_path)
    if result:
        return result

    print("    [Flowchart] ortho failed, trying polyline...")
    polyline_code = dot_code.replace('splines=ortho', 'splines=polyline')
    result = render_dot_direct(polyline_code, output_path)
    if result:
        return result

    print("    [Flowchart] polyline failed, trying line...")
    line_code = dot_code.replace('splines=polyline', 'splines=line')
    result = render_dot_direct(line_code, output_path)
    if result:
        return result

    print("    [Flowchart] line failed, removing splines constraint...")
    no_splines_code = re.sub(r'splines\s*=\s*\w+\s*;?\s*', '', dot_code)
    return render_dot_direct(no_splines_code, output_path)


def generate_flowchart_from_dot(
    dot_code: str,
    output_path: str = "flowchart"
) -> Optional[str]:
    """
    Generate flowchart SVG from DOT code.

    Pipeline:
    1. Fix common issues
    2. Inject theme styling
    3. Render as SVG with spline fallbacks
    4. Post-process SVG for gradients
    5. Fall back to simplified rendering if needed
    """
    if not dot_code or not dot_code.strip():
        print("    [Flowchart] No DOT code provided")
        return None

    if not GRAPHVIZ_AVAILABLE:
        print("    [Flowchart] Graphviz not available, cannot render flowchart")
        return None

    print(f"    [Flowchart] Processing {len(dot_code)} chars of DOT code...")

    # Step 1: Fix common issues
    fixed_code = _fix_common_dot_issues(dot_code)

    # Step 2: Inject theme style
    styled_code = _inject_theme_style(fixed_code)

    # Debug: dump styled DOT to file for inspection
    _debug_dump_dot(styled_code, output_path, "styled")

    # Step 3: Validate
    if not _validate_dot_code(styled_code):
        print("    [Flowchart] Validation failed, attempting render anyway...")

    # Step 4: Render SVG with spline fallbacks
    result = _try_render_with_fallback_splines(styled_code, output_path)
    if result:
        return result

    # Step 5: Try with theme on original code
    print("    [Flowchart] Retrying with theme on original code...")
    styled_original = _inject_theme_style(dot_code)
    _debug_dump_dot(styled_original, output_path, "styled_original")
    result = _try_render_with_fallback_splines(styled_original, output_path)
    if result:
        return result

    # Step 6: Try raw DOT code (no theme)
    print("    [Flowchart] Retrying with raw DOT code...")
    result = render_dot_direct(dot_code, output_path)
    if result:
        return result

    # Step 7: Fallback builder
    print("    [Flowchart] All renders failed, using fallback...")
    return _fallback_render(dot_code, output_path)


def _debug_dump_dot(dot_code: str, output_path: str, suffix: str):
    """Dump DOT code to a debug file for inspection."""
    try:
        output_dir = os.path.dirname(output_path) or "."
        os.makedirs(output_dir, exist_ok=True)
        debug_path = os.path.join(output_dir, f"flowchart_debug_{suffix}.dot")
        with open(debug_path, 'w', encoding='utf-8') as f:
            f.write(dot_code)
        print(f"    [Flowchart] Debug DOT saved: {debug_path}")
    except Exception:
        pass


def _fallback_render(dot_code: str, output_path: str) -> Optional[str]:
    """Last-resort fallback: parse what we can and build a simple flowchart."""
    if not GRAPHVIZ_AVAILABLE:
        return None

    try:
        data = _extract_flowchart_data_simple(dot_code)

        if not data['steps']:
            print("    [Flowchart] Fallback: No steps found")
            return None

        font = THEME['font_name']

        dot = Digraph(format='svg')
        dot.attr(rankdir='TB', size='8,10', dpi='150', bgcolor='white',
                 splines='ortho', nodesep='0.8', ranksep='1.0')
        dot.attr('node', fontname=font, fontsize=THEME['font_size'])
        dot.attr('edge', fontname=font, fontsize='9',
                 color=THEME['arrow_color'], fontcolor=THEME['arrow_text'],
                 penwidth='1.5', decorate='false', labelfloat='true')

        for step in data['steps']:
            nid = step['id']
            label = step['label']
            shape = step.get('shape', 'box')

            ll = label.lower()
            nid_lower = nid.lower()
            
            # Check if this is a terminal node (start/end)
            is_terminal = any(kw in ll for kw in ('start', 'begin', 'end', 'stop', 'finish', 'terminate'))
            is_terminal = is_terminal or any(kw in nid_lower for kw in ('start', 'begin', 'end', 'stop', 'finish', 'terminate'))
            
            if is_terminal:
                # Start/End - grey rounded rectangle, white text, no border
                dot.node(nid, label=label, shape='box',
                         style='filled,rounded', fillcolor=THEME['terminal_fill'],
                         color='transparent', fontcolor=THEME['terminal_text'],
                         penwidth='0', fontname=font)
            elif shape == 'diamond':
                dot.node(nid, label=label, shape='diamond',
                         style='filled', fillcolor=THEME['decision_fill'],
                         color='transparent', fontcolor=THEME['decision_text'],
                         penwidth='0', fontname=font)
            elif shape == 'parallelogram':
                dot.node(nid, label=label, shape='parallelogram',
                         style='filled', fillcolor=THEME['parallelogram_fill'],
                         color='transparent', fontcolor=THEME['parallelogram_text'],
                         penwidth='0', fontname=font)
            elif shape in ('circle', 'doublecircle'):
                dot.node(nid, label=label, shape=shape,
                         style='filled', fillcolor=THEME['process_fill'],
                         color='transparent', fontcolor=THEME['process_text'],
                         penwidth='0', fontname=font)
            else:
                dot.node(nid, label=label, shape='box',
                         style='filled', fillcolor=THEME['process_fill'],
                         color='transparent', fontcolor=THEME['process_text'],
                         penwidth='0', fontname=font)

        for conn in data['connections']:
            if conn.get('label'):
                dot.edge(conn['from'], conn['to'], label=conn['label'])
            else:
                dot.edge(conn['from'], conn['to'])

        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        try:
            dot.render(output_path, cleanup=True)
        except Exception:
            print("    [Flowchart] Fallback ortho failed, trying polyline...")
            dot.graph_attr['splines'] = 'polyline'
            try:
                dot.render(output_path, cleanup=True)
            except Exception:
                print("    [Flowchart] Fallback polyline failed, removing splines...")
                del dot.graph_attr['splines']
                dot.render(output_path, cleanup=True)

        result_path = f"{output_path}.svg"
        if os.path.exists(result_path):
            try:
                with open(result_path, 'r', encoding='utf-8') as f:
                    svg_content = f.read()
                svg_content = _inject_svg_gradients(svg_content)
                with open(result_path, 'w', encoding='utf-8') as f:
                    f.write(svg_content)
            except Exception as e:
                print(f"    [Flowchart] Fallback gradient injection warning: {e}")

        print(f"    [Flowchart] Fallback render: {result_path}")
        return result_path

    except Exception as e:
        print(f"    [Flowchart] Fallback render failed: {e}")
        return None


def _extract_flowchart_data_simple(dot_code: str) -> Dict[str, list]:
    """Simple extraction for fallback only."""
    steps = []
    connections = []
    step_ids = set()

    node_pattern = re.compile(
        r'(\w+)\s*\[\s*label\s*=\s*"([^"]+)"(?:.*?shape\s*=\s*(\w+))?.*?\]',
        re.IGNORECASE
    )

    group_pattern = re.compile(
        r'node\s*\[.*?shape\s*=\s*(\w+).*?\]\s+([\w\s]+);',
        re.IGNORECASE
    )

    edge_pattern = re.compile(
        r'(\w+)\s*->\s*(\w+)(?:\s*\[.*?label\s*=\s*"([^"]*)".*?\])?',
        re.IGNORECASE
    )

    node_shapes = {}
    for match in group_pattern.finditer(dot_code):
        shape = match.group(1).lower()
        ids = match.group(2).strip().split()
        for nid in ids:
            nid = nid.strip().rstrip(';')
            if nid and nid not in ('node', 'edge', 'graph', 'digraph', 'subgraph'):
                node_shapes[nid] = shape

    for match in node_pattern.finditer(dot_code):
        nid, label, shape = match.groups()
        if nid in ('node', 'edge', 'graph', 'digraph', 'subgraph', 'rank'):
            continue
        if nid not in step_ids:
            final_shape = (shape or node_shapes.get(nid, 'box')).lower()
            steps.append({
                "id": nid,
                "label": label.replace("\\n", "\n"),
                "shape": final_shape
            })
            step_ids.add(nid)

    for nid, shape in node_shapes.items():
        if nid not in step_ids:
            label = nid.replace('_', ' ').title()
            steps.append({"id": nid, "label": label, "shape": shape})
            step_ids.add(nid)

    for match in edge_pattern.finditer(dot_code):
        from_node, to_node, _ = match.groups()
        for nid in (from_node, to_node):
            if nid not in step_ids and nid not in ('node', 'edge', 'graph', 'subgraph'):
                shape = node_shapes.get(nid, 'box')
                label = nid.replace('_', ' ').title()
                steps.append({"id": nid, "label": label, "shape": shape})
                step_ids.add(nid)

    for match in edge_pattern.finditer(dot_code):
        from_node, to_node, label = match.groups()
        conn = {"from": from_node, "to": to_node}
        if label:
            conn["label"] = label
        connections.append(conn)

    return {"steps": steps, "connections": connections}


# ============================================================
# Legacy API (backward compatibility)
# ============================================================

def fix_dot_code(dot_code: str) -> str:
    """Legacy: minimal fixes."""
    return _inject_theme_style(_fix_common_dot_issues(dot_code))


def extract_flowchart_data(dot_code: str) -> Dict[str, list]:
    """Legacy: uses improved parser."""
    return _extract_flowchart_data_simple(dot_code)