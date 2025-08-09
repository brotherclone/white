import unittest
import os
import tempfile
import pandas as pd
from unittest.mock import patch, MagicMock

from app.agents.BaseRainbowAgent import BaseRainbowAgent
from app.enums.agent_state import AgentState


class TestBaseRainbowAgent(unittest.TestCase):
    """Test suite for the BaseRainbowAgent class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.agent = BaseRainbowAgent()

        # Create a temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()

        # Create a sample DataFrame for testing
        self.test_data = pd.DataFrame({
            'song_title': ['Test Song 1', 'Test Song 2'],
            'artist': ['Test Artist 1', 'Test Artist 2'],
            'lyrics': ['Test lyrics 1', 'Test lyrics 2'],
            'genre': ['Rock', 'Pop']
        })

        # Create a test parquet file
        self.test_file = os.path.join(self.temp_dir.name, 'test_data.parquet')
        self.test_data.to_parquet(self.test_file)

    def tearDown(self):
        """Clean up after each test method."""
        self.temp_dir.cleanup()

    def test_initialization(self):
        """Test that the agent initializes correctly."""
        self.assertEqual(self.agent.agent_state, AgentState.IDLE)
        self.assertIsNone(self.agent.training_data)
        self.assertEqual(self.agent.data_frames, [])
        self.assertIsNotNone(self.agent.device)
        self.assertIn(self.agent.device, ['cuda', 'cpu'])

    def test_load_training_data(self):
        """Test loading training data from a parquet file."""
        success, fail = self.agent.load_training_data(self.test_file)

        self.assertEqual(success, 1)
        self.assertEqual(fail, 0)
        self.assertEqual(self.agent.agent_state, AgentState.READY)
        self.assertEqual(len(self.agent.data_frames), 1)
        self.assertEqual(len(self.agent.training_data), 2)

    def test_load_nonexistent_file(self):
        """Test loading training data from a nonexistent file."""
        nonexistent_file = os.path.join(self.temp_dir.name, 'nonexistent.parquet')
        success, fail = self.agent.load_training_data(nonexistent_file)

        self.assertEqual(success, 0)
        self.assertEqual(fail, 1)
        self.assertEqual(self.agent.agent_state, AgentState.ERROR)

    @patch('app.agents.BaseRainbowAgent.HuggingFaceEmbeddings')
    @patch('app.agents.BaseRainbowAgent.FAISS')
    def test_create_vector_store(self, mock_faiss, mock_embeddings):
        """Test creating a vector store from training data."""
        # Set up mocks
        mock_embeddings_instance = MagicMock()
        mock_embeddings.return_value = mock_embeddings_instance

        # Load training data
        self.agent.load_training_data(self.test_file)

        # Create vector store
        result = self.agent.create_vector_store('lyrics', ['song_title', 'artist', 'genre'])

        # Verify results
        self.assertTrue(result)
        mock_faiss.from_texts.assert_called_once()
        args, kwargs = mock_faiss.from_texts.call_args
        self.assertEqual(len(args[0]), 2)  # Two text entries
        self.assertEqual(len(kwargs['metadatas']), 2)  # Two metadata entries

    def test_process(self):
        """Test the process method."""
        # Test processing when agent is in IDLE state
        result = self.agent.process()
        self.assertEqual(result['status'], 'success')

        # Test processing when agent is in ERROR state
        self.agent.agent_state = AgentState.ERROR
        result = self.agent.process()
        self.assertIn('error', result)

    def test_get_status(self):
        """Test getting agent status."""
        status = self.agent.get_status()

        self.assertEqual(status['agent_type'], 'BaseRainbowAgent')
        self.assertEqual(status['state'], 'IDLE')
        self.assertEqual(status['training_data_count'], 0)
        self.assertFalse(status['vector_store_ready'])

    def test_cleanup(self):
        """Test cleaning up agent resources."""
        # Create a mock vector store
        self.agent.vector_store = MagicMock()

        # Run cleanup
        self.agent.cleanup()

        # Check that vector store was cleared
        self.assertIsNone(self.agent.vector_store)
        self.assertEqual(self.agent.agent_state, AgentState.IDLE)


if __name__ == '__main__':
    unittest.main()
