"""
Migration: create_artists_table
Created: 2025-07-12T16:12:11.123939
"""
from alembic import op
import sqlalchemy as sa

# revision identifiers used by Alembic
revision = '20250712161211'
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
    """Create Artists table"""
    op.create_table(
        'artists',
        sa.Column('id', sa.Integer, primary_key=True, autoincrement=True),
        sa.Column('discogs_id', sa.Integer, unique=True, nullable=False),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('profile', sa.Text, nullable=True),
        sa.Column('created_at', sa.DateTime, server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime, server_default=sa.func.now(), onupdate=sa.func.now(), nullable=False)
    )
    print("Created 'artists' table successfully.")


def down():
    """Revert migration changes"""
    op.drop_table('artists')
    pass
