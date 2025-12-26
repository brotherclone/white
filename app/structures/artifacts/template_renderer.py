"""
HTML Template Renderer for Chain Artifacts

This module provides utilities to render HTML templates with data substitution.
Templates use JavaScript template literal syntax: ${variable_name}
"""

import re
from pathlib import Path
from typing import Any, Dict


class HTMLTemplateRenderer:
    """Renders HTML templates with variable substitution."""

    def __init__(self, template_path: str | Path):
        """
        Initialize the renderer with a template file.

        Args:
            template_path: Path to the HTML template file
        """
        self.template_path = Path(template_path)
        if not self.template_path.exists():
            raise FileNotFoundError(f"Template not found: {template_path}")

        with open(self.template_path, "r", encoding="utf-8") as f:
            self.template_content = f.read()

    def render(self, data: Dict[str, Any]) -> str:
        """
        Render the template with the provided data.

        Args:
            data: Dictionary of variable names to values for substitution

        Returns:
            Rendered HTML string

        Notes:
            - Template variables use ${variable_name} syntax
            - Supports conditional rendering: ${variable ? 'value' : ''}
            - Array mapping: ${array.map(item => `<span>${item}</span>`).join('')}
        """
        rendered = self.template_content

        # Handle array mapping (e.g., tags.map(...).join(''))
        # This is a simplified version - for production you'd want a full JS parser
        array_pattern = r"\$\{(\w+)\.map\((.*?)\)\.join\((.*?)\)\}"

        def replace_array_map(match):
            array_name = match.group(1)
            map_expression = match.group(2)
            join_string = match.group(3).strip("'\"")

            if array_name in data and isinstance(data[array_name], list):
                # Extract the template from the map expression
                # This is a simple regex to get content between backticks or quotes
                template_match = re.search(r'[`\'"](.+?)[`\'"]', map_expression)
                if template_match:
                    item_template = template_match.group(1)
                    # Replace 'item' or other variable names in the template
                    # For simplicity, we'll just replace ${item} with each array element
                    result_parts = []
                    for item in data[array_name]:
                        item_rendered = item_template
                        # Replace ${item} with the actual value
                        item_rendered = re.sub(r"\$\{item\}", str(item), item_rendered)
                        # Also support ${tag} for tags array
                        item_rendered = re.sub(r"\$\{tag\}", str(item), item_rendered)
                        result_parts.append(item_rendered)
                    return join_string.join(result_parts)
            return ""

        rendered = re.sub(array_pattern, replace_array_map, rendered)

        # Handle logical OR (||) operator: ${var || 'default'}
        or_pattern = r"\$\{(\w+)\s*\|\|\s*([^}]+)\}"

        def replace_or(match):
            var_name = match.group(1)
            default_value = match.group(2).strip()

            # If variable exists and is truthy, use it; otherwise use default
            if var_name in data and data[var_name]:
                return str(data[var_name])
            else:
                return self._process_expression(default_value, data)

        rendered = re.sub(or_pattern, replace_or, rendered)

        # Handle ternary operators with complex expressions
        # Use a more careful pattern that handles nested content
        # This handles multi-line ternaries with backtick strings
        def find_and_replace_ternaries(text):
            # Find ${var ? ... : ...} patterns, carefully balancing nested braces and quotes
            pos = 0
            result = []

            while pos < len(text):
                # Look for ${
                start = text.find("${", pos)
                if start == -1:
                    result.append(text[pos:])
                    break

                # Add text before ${
                result.append(text[pos:start])

                # Find the matching }
                depth = 1
                i = start + 2
                in_string = None
                while i < len(text) and depth > 0:
                    if text[i] in ('"', "'", "`") and (i == 0 or text[i - 1] != "\\"):
                        if in_string == text[i]:
                            in_string = None
                        elif in_string is None:
                            in_string = text[i]
                    elif not in_string:
                        if text[i] == "{":
                            depth += 1
                        elif text[i] == "}":
                            depth -= 1
                    i += 1

                if depth != 0:
                    # Unmatched braces, just append and continue
                    result.append(text[start:i])
                    pos = i
                    continue

                # Extract the full ${...} expression
                expr = text[start + 2 : i - 1]

                # Check if it's a ternary
                if "?" in expr and ":" in expr:
                    # Find the ? and : at the right nesting level
                    parts = self._split_ternary(expr)
                    if parts:
                        var_name, true_val, false_val = parts
                        val = data.get(var_name)
                        # Check if value is falsy (including string "None" from Pydantic serialization)
                        is_falsy = val in (
                            None,
                            "",
                            0,
                            False,
                            "None",
                            "null",
                            "undefined",
                        )
                        if var_name in data and not is_falsy:
                            result.append(self._process_expression(true_val, data))
                        else:
                            result.append(self._process_expression(false_val, data))
                        pos = i
                        continue

                # Not a ternary, put it back
                result.append(text[start:i])
                pos = i

            return "".join(result)

        rendered = find_and_replace_ternaries(rendered)

        # Handle simple variable substitution
        rendered = self._simple_substitute(rendered, data)

        return rendered

    def _split_ternary(self, expr: str):
        """
        Split a ternary expression into (var_name, true_value, false_value).
        Handles nested quotes and braces properly.
        """
        # Find the variable name before ?
        question_pos = -1
        in_string = None
        depth = 0

        for i, char in enumerate(expr):
            if char in ('"', "'", "`") and (i == 0 or expr[i - 1] != "\\"):
                if in_string == char:
                    in_string = None
                elif in_string is None:
                    in_string = char
            elif not in_string:
                if char in ("{", "(", "["):
                    depth += 1
                elif char in ("}", ")", "]"):
                    depth -= 1
                elif char == "?" and depth == 0:
                    question_pos = i
                    break

        if question_pos == -1:
            return None

        var_name = expr[:question_pos].strip()

        # Find the : separator between true and false values
        colon_pos = -1
        in_string = None
        depth = 0

        for i in range(question_pos + 1, len(expr)):
            char = expr[i]
            if char in ('"', "'", "`") and (i == 0 or expr[i - 1] != "\\"):
                if in_string == char:
                    in_string = None
                elif in_string is None:
                    in_string = char
            elif not in_string:
                if char in ("{", "(", "["):
                    depth += 1
                elif char in ("}", ")", "]"):
                    depth -= 1
                elif char == ":" and depth == 0:
                    colon_pos = i
                    break

        if colon_pos == -1:
            return None

        true_value = expr[question_pos + 1 : colon_pos].strip()
        false_value = expr[colon_pos + 1 :].strip()

        return (var_name, true_value, false_value)

    def _process_expression(self, expr: str, data: Dict[str, Any]) -> str:
        """
        Process a JavaScript-like expression (handles string concat, template strings, etc).

        Args:
            expr: Expression to process (e.g., "' (' + var + ')'")
            data: Dictionary of variable values

        Returns:
            Processed expression result
        """
        # Remove surrounding quotes if it's a simple string
        expr = expr.strip()

        # Handle empty strings
        if expr in ("''", '""', "``"):
            return ""

        # Handle template literals with backticks
        if expr.startswith("`") and expr.endswith("`"):
            # Extract content between backticks and substitute variables
            template_content = expr[1:-1]
            return self._simple_substitute(template_content, data)

        # Handle string concatenation with +
        if "+" in expr:
            parts = []
            # Split by + but be careful with quotes
            concat_parts = re.split(r"\s*\+\s*", expr)
            for part in concat_parts:
                part = part.strip()
                # Check if it's a quoted string
                if (part.startswith("'") and part.endswith("'")) or (
                    part.startswith('"') and part.endswith('"')
                ):
                    parts.append(part[1:-1])  # Remove quotes
                # Check if it's a variable
                elif part in data:
                    val = data[part]
                    parts.append(str(val) if val is not None else "")
                # Check if it's a template string
                elif part.startswith("`") and part.endswith("`"):
                    parts.append(self._process_expression(part, data))
            return "".join(parts)

        # Handle simple quoted strings
        if (expr.startswith("'") and expr.endswith("'")) or (
            expr.startswith('"') and expr.endswith('"')
        ):
            return expr[1:-1]

        # Otherwise treat as variable reference and substitute
        return self._simple_substitute(expr, data)

    def _simple_substitute(self, text: str, data: Dict[str, Any]) -> str:
        """
        Perform simple ${variable} substitution.

        Args:
            text: Text containing ${variable} placeholders
            data: Dictionary of variable values

        Returns:
            Text with variables substituted
        """
        pattern = r"\$\{(\w+(?:\.\w+)*)\}"

        def replace_var(match):
            var_path = match.group(1)
            # Support dot notation (e.g., ${object.property})
            parts = var_path.split(".")
            value = data
            try:
                for part in parts:
                    value = (
                        value[part] if isinstance(value, dict) else getattr(value, part)
                    )
                return str(value) if value is not None else ""
            except (KeyError, AttributeError):
                # Variable not found, return empty string
                return ""

        return re.sub(pattern, replace_var, text)

    def render_with_model(self, model: Any) -> str:
        """
        Render the template using a Pydantic model.

        Args:
            model: Pydantic model instance with data for template

        Returns:
            Rendered HTML string
        """
        if hasattr(model, "model_dump"):
            data = model.model_dump()
        elif hasattr(model, "dict"):
            data = model.dict()
        else:
            raise ValueError(
                "Model must be a Pydantic model with model_dump() or dict()"
            )

        return self.render(data)


def get_template_path(template_name: str) -> Path:
    """
    Get the full path to a template file.

    Args:
        template_name: Name of the template (e.g., 'card_catalog', 'character_sheet', 'quantum_tape')

    Returns:
        Path to the template file
    """
    base_dir = Path(__file__).parent / "templates"
    return base_dir / f"{template_name}.html"
