# app/cli/migration_commands.py
import argparse
import os
import sys

# Add project root to path to ensure imports work correctly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from app.utils.migration_util import create_migration, run_migrations


def main():
    parser = argparse.ArgumentParser(description="Database migration tool")
    subparsers = parser.add_subparsers(dest="command", help="Migration commands")

    # Create migration command
    create_parser = subparsers.add_parser("create", help="Create a new migration")
    create_parser.add_argument("name", help="Name of the migration (e.g., create_users_table)")

    # Run migrations command
    migrate_parser = subparsers.add_parser("migrate", help="Run migrations")
    migrate_parser.add_argument("direction", choices=["up", "down"],
                                help="'up' to apply migration, 'down' to rollback")

    args = parser.parse_args()

    if args.command == "create":
        create_migration(args.name)
    elif args.command == "migrate":
        run_migrations(args.direction)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()