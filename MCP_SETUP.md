# MCP Server Configuration for Claude Desktop

This project includes two MCP (Model Context Protocol) servers for use with Claude Desktop:

1. **earthly-frames-todoist** - Todoist integration for The Earthly Frames project management
2. **orange-mythos** - Sussex County mythology corpus and gonzo journalism rewriter

## Installation

### 1. Create your configuration file

Copy the template and fill in your API keys:

```bash
cp claude_desktop_config.template.json claude_desktop_config.json
```

Then edit `claude_desktop_config.json` and replace:
- `YOUR_TODOIST_API_TOKEN_HERE` with your Todoist API token
- `YOUR_ANTHROPIC_API_KEY_HERE` with your Anthropic API key

**Or** use the values from your `.env` file if you already have them set up.

### 2. Copy the configuration to Claude Desktop

The configuration file needs to be placed in Claude Desktop's config directory:

**macOS:**
```bash
cp claude_desktop_config.json ~/Library/Application\ Support/Claude/claude_desktop_config.json
```

**Or manually copy the contents** of `claude_desktop_config.json` to:
- macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
- Windows: `%APPDATA%\Claude\claude_desktop_config.json`
- Linux: `~/.config/Claude/claude_desktop_config.json`


### 3. Restart Claude Desktop

After copying the config, completely quit and restart Claude Desktop.

## Available Tools

### Todoist Server Tools

- `get_earthly_frames_project_sections` - List all sections in the EF project
- `create_sigil_charging_task` - Create a task for human sigil charging ritual
- `list_pending_black_agent_tasks` - List incomplete Black Agent tasks
- `create_todoist_task_for_human_earthly_frame` - Generic task creation

### Orange Mythos Server Tools

- `add_story_to_corpus` - Add a mythologizable news story with automatic scoring
- `insert_symbolic_object` - Insert symbolic objects into stories for mythologization
- `gonzo_rewrite` - Rewrite stories in gonzo journalism style (Hunter S. Thompson method)
- `search_corpus` - Search the mythology corpus by tags, dates, locations
- `get_corpus_stats` - Get statistics about the corpus
- `export_corpus_json` - Export corpus for human inspection

## Testing the Servers

### Test Todoist Server

From the project directory:
```bash
uv run python app/reference/mcp/todoist/main.py
```

This should start the MCP server in stdio mode.

### Test Orange Mythos Server

```bash
uv run python app/reference/mcp/rows_bud/orange_mythos_server.py
```

You should see:
```
🌹 Orange Mythos MCP Server (Polars Edition)
   Sussex County mythologizer - 182 BPM transmission
   Corpus: /Volumes/LucidNonsense/White/app/reference/mcp/rows_bud
   Stories loaded: [number]
```

## Troubleshooting

### MCP servers not showing up in Claude Desktop

1. Check the Claude Desktop logs:
   - macOS: `~/Library/Logs/Claude/mcp*.log`
   
2. Verify the paths are correct in the config file

3. Make sure `uv` is in your PATH:
   ```bash
   which uv
   ```

4. Test the servers manually as shown above

### Todoist "No module named" errors

The todoist server no longer depends on `todoist-api-python`. It uses the direct REST API with `requests`. Make sure your dependencies are installed:

```bash
uv sync
```

### Orange Mythos API errors

Make sure your `ANTHROPIC_API_KEY` environment variable is set correctly:

```bash
echo $ANTHROPIC_API_KEY
```

## Development

To modify the MCP servers:

1. Edit the server files:
   - Todoist: `app/reference/mcp/todoist/main.py`
   - Orange Mythos: `app/reference/mcp/rows_bud/orange_mythos_server.py`

2. Test your changes locally using `uv run python <server_file>`

3. Restart Claude Desktop to pick up changes

## Security Note

The `claude_desktop_config.json` file in this repository contains the Todoist API token. **Do not commit this to a public repository.** It's included here for local development only.

Consider adding it to `.gitignore`:
```bash
echo "claude_desktop_config.json" >> .gitignore
```

