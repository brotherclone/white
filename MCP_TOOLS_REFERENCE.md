# MCP Server Quick Reference

## Todoist Server - Available Tools

### `get_earthly_frames_project_sections`
Get all sections in The Earthly Frames Todoist project.

**Parameters:**
- `project_id` (optional): Defaults to EF project

**Example:**
```
List all sections in my Earthly Frames project
```

### `create_sigil_charging_task`
Create a task for human to charge a sigil.

**Parameters:**
- `sigil_description`: Description of the sigil glyph
- `charging_instructions`: Ritual instructions
- `song_title`: Title of the song
- `section_name` (optional): Defaults to "Black Agent - Sigil Work"

**Example:**
```
Create a sigil charging task:
- sigil: Three concentric circles with lightning bolt
- instructions: Hold sigil while listening to song 3 times
- song: "Dover Temporal Horror"
```

### `list_pending_black_agent_tasks`
List all incomplete tasks in Black Agent sections.

**Parameters:**
- `section_name` (optional): Defaults to "Black Agent - Sigil Work"

**Example:**
```
What Black Agent tasks are pending?
```

### `create_todoist_task_for_human_earthly_frame`
Generic task creation for any purpose.

**Parameters:**
- `content`: Task title
- `project_id` (optional): Defaults to EF project
- `section_id` (optional): Specific section
- `description` (optional): Longer description
- `priority` (optional): 1-4, defaults to 1

**Example:**
```
Create a task "Review album artwork" with priority 3 in the Earthly Frames project
```

---

## Orange Mythos Server - Available Tools

### `add_story_to_corpus`
Add a mythologizable news story with automatic scoring.

**Parameters:**
- `headline`: Story headline
- `date`: Publication date (YYYY-MM-DD)
- `source`: Newspaper name
- `text`: Full article text
- `location`: Specific NJ location
- `tags`: Category tags (e.g., ["rock_bands", "youth_crime"])

**Example:**
```
Add this story to the mythology corpus:
headline: "Teen Band Vanishes After Dover Show"
date: 1984-06-15
source: "Daily Record"
location: "Dover, NJ"
tags: ["rock_bands", "disappearances"]
text: [full article text]
```

### `insert_symbolic_object`
Insert a symbolic object into a story for mythologization.

**Parameters:**
- `story_id`: ID of story to mythologize
- `object_category`: One of:
  - `CIRCULAR_TIME` - Clocks, loops, temporal anomalies
  - `INFORMATION_ARTIFACTS` - Recordings, transmissions, tapes
  - `LIMINAL_OBJECTS` - Doorways, thresholds, boundaries
  - `PSYCHOGEOGRAPHIC` - Maps, coordinates, impossible locations
- `custom_object` (optional): Custom object description

**Example:**
```
Insert a symbolic object into story xyz123:
- category: INFORMATION_ARTIFACTS
- object: A cassette tape labeled "182 BPM Transmission"
```

### `gonzo_rewrite`
Rewrite a mythologized story in gonzo journalism style.

**Parameters:**
- `story_id`: ID of mythologized story
- `perspective` (optional): One of:
  - `journalist` - Embedded reporter
  - `witness` - First-hand witness
  - `investigator` - Private detective
  - `participant` - Part of the story
- `intensity` (optional): 1-5, where:
  - 1 = Subtle first-person observer
  - 2 = Embedded journalist with suspicion
  - 3 = Active participant, reality slippery
  - 4 = Deep paranoia, conspiracy emerging
  - 5 = Full Hunter S. Thompson madness

**Example:**
```
Rewrite story xyz123 in gonzo style:
- perspective: investigator
- intensity: 4
```

### `search_corpus`
Search the mythology corpus.

**Parameters:**
- `tags` (optional): Required tags
- `min_score` (optional): Minimum score (0.0-1.0), defaults to 0.5
- `location` (optional): Specific NJ location
- `start_date` (optional): Date range start
- `end_date` (optional): Date range end
- `needs_mythologizing` (optional): Skip already mythologized, defaults to true

**Example:**
```
Search for stories:
- tags: ["rock_bands", "youth_crime"]
- location: "Dover"
- between: 1980-1990
- min score: 0.7
```

### `get_corpus_stats`
Get statistics about the mythology corpus.

**No parameters**

**Example:**
```
Show me corpus statistics
```

### `export_corpus_json`
Export entire corpus as JSON.

**Parameters:**
- `filename` (optional): Output filename

**Example:**
```
Export the corpus to "sussex_mythology.json"
```

---

## Common Workflows

### Black Agent Sigil Workflow
1. Check pending tasks: `list_pending_black_agent_tasks`
2. Complete sigil charging ritual (in real life)
3. Mark task complete in Todoist
4. Resume: `python run_white_agent.py resume`

### Orange Mythos Workflow
1. Search corpus: `search_corpus` with your criteria
2. Choose a story and insert symbolic object: `insert_symbolic_object`
3. Rewrite in gonzo style: `gonzo_rewrite` with desired intensity
4. Export results: `export_corpus_json`

---

## Troubleshooting

**Servers not showing up?**
- Check Claude Desktop logs: `~/Library/Logs/Claude/mcp*.log`
- Restart Claude Desktop completely
- Verify config: `cat ~/Library/Application\ Support/Claude/claude_desktop_config.json`

**Todoist errors?**
- Verify API token is correct
- Check project ID matches: 6CrfWqXrxppjhqMJ

**Mythos errors?**
- Verify Anthropic API key is set
- Check corpus directory exists and has corpus.parquet

