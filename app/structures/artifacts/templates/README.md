# HTML Chain Artifact Templates

This directory contains HTML templates for rendering chain artifacts with custom styling.

## Available Templates

### 1. Card Catalog (`card_catalog.html`)
Library card catalog style for documenting forbidden/dangerous books.

**Visual Style:** Vintage library catalog drawer with aged paper card
**Use Cases:**
- Documenting mysterious books
- Creating artifact inventories
- Blue Album "forbidden knowledge" references

**Template Variables:**
- `danger_level` - 1-5 rating
- `title`, `subtitle` - Book identification
- `author`, `author_credentials` - Author info
- `year`, `publisher`, `edition` - Publication details
- `abstract` - Book description
- `notable_quote` - Memorable quote
- `suppression_history` - Censorship info
- `tags` - Category tags (array)
- `catalog_number` - Reference ID

### 2. Character Sheet (`character_sheet.html`)
Pulsar Palace character dossier with retro-futuristic CRT aesthetic.

**Visual Style:** 80s computer terminal with RGB color scheme
**Use Cases:**
- Pulsar Palace RPG characters
- Yellow Agent character documentation
- Game run character tracking

**Template Variables:**
- `portrait_image_url` - Character portrait
- `disposition`, `profession` - Character traits
- `background_place`, `background_time` - Origin
- `on_current/max`, `off_current/max` - Stats
- `frequency_attunement` - Attunement level 0-100
- `current_location` - Where they are
- `inventory` - Items (array, up to 9 slots)
- `reality_anchor` - Reality status

### 3. Quantum Tape (`quantum_tape.html`)
Cassette tape label showing "taped over" memories/timelines.

**Visual Style:** Vintage cassette tape with A-Side/B-Side structure
**Use Cases:**
- Blue Album biographical exploration
- Memory revision documentation
- Quantum superposition of timelines

**Template Variables:**
- `original_date`, `original_title` - A-Side (original memory)
- `tapeover_date`, `tapeover_title` - B-Side (revised memory)
- `subject_name`, `age_during` - Subject info
- `location` - Where event occurred
- `year_documented` - When documented
- `catalog_number` - Reference ID

## Usage

### Basic Usage

```python
from app.structures.artifacts.html_artifacts import CardCatalogArtifact

# Create an artifact
artifact = CardCatalogArtifact(
    thread_id="my_thread",
    base_path="./output",
    title="The Necronomicon",
    author="Abdul Alhazred",
    danger_level=5,
    # ... other fields
)

# Save as HTML
artifact.save_file()
```

### With Pydantic Model

The artifacts are Pydantic models, so you can:

```python
# Load from dict/JSON
data = {
    "thread_id": "thread_001",
    "title": "My Book",
    # ...
}
artifact = CardCatalogArtifact(**data)

# Export to dict
artifact_dict = artifact.model_dump()

# Use in prompts
prompt_text = artifact.for_prompt()
```

### Template Rendering

Under the hood, the `HTMLTemplateRenderer` handles:
- Variable substitution using `${variable}` syntax
- Conditional rendering with ternary operators
- Array mapping for lists
- Nested object property access

## Adding New Templates

1. Create HTML template in this directory
2. Use `${variable_name}` for substitution
3. Create artifact class in `html_artifacts.py`
4. Add example in `examples/html_artifact_examples.py`

## Template Syntax

### Simple Variables
```html
<div>${title}</div>
```

### Conditional Rendering
```html
${notable_quote ? `<div class="quote">${notable_quote}</div>` : ''}
```

### Array Mapping
```html
${tags.map(tag => `<span class="tag">${tag}</span>`).join('')}
```

### Nested Properties
```html
${character.disposition}
```

## File Naming

Artifacts are saved with this naming pattern:
```
{artifact_id}_{color_code}_{artifact_name}.html
```

Example: `abc123_b_quantum_tape.html`

Color codes:
- `b` - Blue (biographical)
- `y` - Yellow (character sheets)
- `r` - Red (dangerous content)
- etc.

## Output Locations

HTML files are saved to:
```
{base_path}/{thread_id}/html/{artifact_id}_{color}_{name}.html
```

Example:
```
./chain_artifacts/thread_001/html/abc123_b_quantum_tape.html
```
