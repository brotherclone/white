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

        # Handle ternary operators (e.g., ${var ? 'yes' : 'no'})
        ternary_pattern = r"\$\{(\w+)\s*\?\s*(.+?)\s*:\s*(.+?)\}"

        def replace_ternary(match):
            var_name = match.group(1)
            true_value = match.group(2).strip("'\"")
            false_value = match.group(3).strip("'\"")

            # Check if the variable exists and is truthy
            if var_name in data and data[var_name]:
                # Handle nested template strings in the true value
                if "${" in true_value:
                    return self._simple_substitute(true_value, data)
                return true_value
            else:
                if "${" in false_value:
                    return self._simple_substitute(false_value, data)
                return false_value

        rendered = re.sub(ternary_pattern, replace_ternary, rendered)

        # Handle simple variable substitution
        rendered = self._simple_substitute(rendered, data)

        return rendered

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
