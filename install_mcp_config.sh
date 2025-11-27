#!/bin/bash
# Install MCP configuration to Claude Desktop

CONFIG_FILE="claude_desktop_config.json"
TEMPLATE_FILE="claude_desktop_config.template.json"

# Detect OS
if [[ "$OSTYPE" == "darwin"* ]]; then
    CLAUDE_CONFIG_DIR="$HOME/Library/Application Support/Claude"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    CLAUDE_CONFIG_DIR="$HOME/.config/Claude"
elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    CLAUDE_CONFIG_DIR="$APPDATA/Claude"
else
    echo "❌ Unsupported OS: $OSTYPE"
    exit 1
fi

CLAUDE_CONFIG_FILE="$CLAUDE_CONFIG_DIR/claude_desktop_config.json"

echo "🔧 MCP Configuration Installer"
echo "================================"
echo ""

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "⚠️  $CONFIG_FILE not found!"
    echo ""
    if [ -f "$TEMPLATE_FILE" ]; then
        echo "📝 Creating from template..."
        cp "$TEMPLATE_FILE" "$CONFIG_FILE"
        echo ""
        echo "⚠️  Please edit $CONFIG_FILE and add your API keys:"
        echo "   - TODOIST_API_TOKEN"
        echo "   - ANTHROPIC_API_KEY"
        echo ""
        echo "Then run this script again."
        exit 1
    else
        echo "❌ Template file $TEMPLATE_FILE not found!"
        exit 1
    fi
fi

# Check if API keys are still placeholders
if grep -q "YOUR_.*_API.*_HERE" "$CONFIG_FILE"; then
    echo "⚠️  API keys not configured!"
    echo ""
    echo "Please edit $CONFIG_FILE and replace:"
    echo "   - YOUR_TODOIST_API_TOKEN_HERE"
    echo "   - YOUR_ANTHROPIC_API_KEY_HERE"
    echo ""
    echo "With your actual API keys."
    exit 1
fi

# Create Claude config directory if it doesn't exist
mkdir -p "$CLAUDE_CONFIG_DIR"

# Backup existing config if it exists
if [ -f "$CLAUDE_CONFIG_FILE" ]; then
    BACKUP_FILE="${CLAUDE_CONFIG_FILE}.backup.$(date +%Y%m%d_%H%M%S)"
    echo "📦 Backing up existing config to:"
    echo "   $BACKUP_FILE"
    cp "$CLAUDE_CONFIG_FILE" "$BACKUP_FILE"
    echo ""
fi

# Copy config
echo "📋 Copying configuration to Claude Desktop..."
cp "$CONFIG_FILE" "$CLAUDE_CONFIG_FILE"

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Configuration installed successfully!"
    echo ""
    echo "📍 Config location: $CLAUDE_CONFIG_FILE"
    echo ""
    echo "Next steps:"
    echo "1. Completely quit Claude Desktop"
    echo "2. Restart Claude Desktop"
    echo "3. Look for 🔨 icon in chat to confirm MCP servers are connected"
    echo ""
    echo "Installed MCP servers:"
    echo "  • earthly-frames-todoist - Todoist integration"
    echo "  • orange-mythos - Sussex County mythology corpus"
else
    echo ""
    echo "❌ Failed to copy configuration"
    exit 1
fi

