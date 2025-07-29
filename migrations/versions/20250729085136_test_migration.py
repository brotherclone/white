"""
Migration: test_migration
Created: 2025-07-29T08:51:36.231465
"""
from alembic import op
import sqlalchemy as sa

# revision identifiers used by Alembic
revision = '20250729085136'
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
