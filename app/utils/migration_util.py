import importlib
import os
from datetime import datetime
from typing import List, Optional

import alembic.config
from alembic import command
from alembic.script import ScriptDirectory

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MIGRATIONS_DIR = os.path.join(BASE_DIR, "migrations")


def init_migrations():
    """Initialize the migration environment if it doesn't exist"""
    if not os.path.exists(MIGRATIONS_DIR):
        alembic_cfg = alembic.config.Config()
        alembic_cfg.set_main_option("script_location", MIGRATIONS_DIR)
        alembic_cfg.set_main_option("sqlalchemy.url", "sqlite:///./app.db")
        command.init(alembic_cfg, directory=MIGRATIONS_DIR)
        print("Migration environment initialized")


def create_migration(name: str):
    """Create a new migration file with up and down methods"""
    init_migrations()
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    versions_dir = os.path.join(MIGRATIONS_DIR, "versions")
    os.makedirs(versions_dir, exist_ok=True)  # Ensure versions directory exists
    filename = os.path.join(versions_dir, f"{timestamp}_{name}.py")
    with open(filename, "w") as f:
        f.write(f'''"""
Migration: {name}
Created: {datetime.now().isoformat()}
"""
from alembic import op
import sqlalchemy as sa

# revision identifiers used by Alembic
revision = '{timestamp}'
down_revision = None
branch_labels = None
depends_on = None

def upgrade():
    # Implement up() logic
    up()

def downgrade():
    # Implement down() logic
    down()

def up():
    """Apply migration changes"""
    # Write your forward migration code here
    # Example: op.create_table('users', ...)
    pass

def down():
    """Revert migration changes"""
    # Write your rollback migration code here  
    # Example: op.drop_table('users')
    pass
''')
    print(f"Created migration: {filename}")
    return filename


def run_migrations(direction: str = "up"):
    """Run all pending migrations up or down"""
    alembic_cfg = alembic.config.Config()
    alembic_cfg.set_main_option("script_location", MIGRATIONS_DIR)

    if direction == "up":
        command.upgrade(alembic_cfg, "head")
        print("Migrations applied successfully")
    elif direction == "down":
        command.downgrade(alembic_cfg, "-1")
        print("Migration reverted successfully")