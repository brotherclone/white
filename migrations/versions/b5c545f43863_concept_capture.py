"""concept capture

Revision ID: b5c545f43863
Revises: 20250712161211
Create Date: 2025-08-05 10:56:23.806231

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = 'b5c545f43863'
down_revision: Union[str, Sequence[str], None] = '20250712161211'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    up()


def downgrade() -> None:
    """Downgrade schema."""
    down()


def up() -> None:
    """Create ConceptCapture table."""
    op.create_table(
        'concept_capture',
        sa.Column('id', sa.Integer, primary_key=True, autoincrement=True),
        sa.Column('concept', sa.Text, nullable=True),
        sa.Column('created_at', sa.DateTime, server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime, server_default=sa.func.now(), onupdate=sa.func.now(), nullable=False)
    )
    print("Created 'concept_capture' table successfully.")


def down() -> None:
    """Revert migration changes."""
    op.drop_table('concept_capture')
    print("Dropped 'concept_capture' table successfully.")
