import unittest
from unittest.mock import patch
from app.utils.migration_util import init_migrations, create_migration, run_migrations

class TestMigrationUtil(unittest.TestCase):
    @patch('app.utils.migration_util.os.path.exists', return_value=False)
    @patch('app.utils.migration_util.command.init')
    def test_init_migrations(self, mock_init, mock_exists):
            init_migrations()
            mock_init.assert_called_once()

    def test_create_migration(self):
        migration_name = "test_migration"
        migration_path = create_migration(migration_name)
        self.assertIsNotNone(migration_path)
        self.assertIn(migration_name, migration_path)

    @patch('app.utils.migration_util.command.upgrade')
    def test_run_migrations(self,mock_upgrade):
        run_migrations('up')
        mock_upgrade.assert_called_once()