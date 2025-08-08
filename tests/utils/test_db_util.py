import unittest
from app.utils.db_util import get_db_connection, close_db_connection

class TestDBUtil(unittest.TestCase):
    def test_get_db_connection(self):
        conn = get_db_connection()
        self.assertIsNotNone(conn)
        self.assertTrue(conn.is_connected())

    def test_close_db_connection(self):
        conn = get_db_connection()
        close_db_connection(conn)
        self.assertFalse(conn.is_connected())