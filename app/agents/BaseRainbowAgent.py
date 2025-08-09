import pandas as pd
import torch
import os
import logging
from datetime import datetime
from typing import Any, List, Dict, Optional, Union, Tuple

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from pydantic import BaseModel, ConfigDict, field_validator
from app.enums.agent_state import AgentState

# Configure logging
logger = logging.getLogger(__name__)

class BaseRainbowAgent(BaseModel):
    """
    Base class for Rainbow Agents that process music training data.

    This class provides common functionality for all Rainbow agents,
    including training data loading, vector store creation, and
    basic processing methods.

    Attributes:
        analyzer_name: Name of the analyzer model to use
        processor_name: Name of the processor model to use
        tokenizer_name: Name of the tokenizer model to use
        generator_name: Name of the generator model to use
        llm_model_name: Name of the LLM model to use
        training_data: Combined training data from all data frames
        data_frames: List of loaded data frames
        embeddings: Embeddings model for vector store
        vector_store: FAISS vector store for similarity searches
        device: Device to use for processing (cuda or cpu)
        agent_state: Current state of the agent
        initialization_time: When the agent was last initialized
    """
    analyzer_name: Optional[str] = None
    processor_name: Optional[str] = None
    tokenizer_name: Optional[str] = None
    generator_name: Optional[str] = None
    llm_model_name: Optional[str] = None
    training_data: Optional[pd.DataFrame] = None
    data_frames: List[pd.DataFrame] = []
    embeddings: Optional[Any] = None
    vector_store: Optional[Any] = None
    device: Optional[str] = None
    agent_state: AgentState = AgentState.IDLE
    initialization_time: Optional[datetime] = None

    # Allow arbitrary types for pytorch/langchain objects
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, **data):
        """
        Initialize the Rainbow Agent.

        Args:
            **data: Agent configuration parameters
        """
        super().__init__(**data)
        # Set device based on GPU availability
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Agent initialized using device: {self.device}")

        # Set initial state
        self.agent_state = AgentState.IDLE
        self.data_frames = []
        self.embeddings = None
        self.vector_store = None
        self.initialization_time = datetime.now()

    def initialize(self) -> bool:
        """
        Initialize the agent with required models and data.

        This method should be overridden by subclasses to load specific models.

        Returns:
            bool: True if initialization was successful, False otherwise
        """
        try:
            self.agent_state = AgentState.INITIALIZING
            logger.info(f"Initializing {self.__class__.__name__}")
            self.initialization_time = datetime.now()

            # Actual initialization should be implemented in subclasses

            self.agent_state = AgentState.READY
            logger.info(f"{self.__class__.__name__} successfully initialized")
            return True
        except Exception as e:
            self.agent_state = AgentState.ERROR
            logger.error(f"Failed to initialize {self.__class__.__name__}: {e}", exc_info=True)
            return False

    def process(self) -> Dict[str, Any]:
        """
        Process data with the agent.

        This method should be overridden by subclasses to implement specific processing logic.

        Returns:
            dict: Results of the processing
        """
        if self.agent_state not in [AgentState.READY, AgentState.IDLE]:
            logger.warning(f"Cannot process: agent in state {self.agent_state}")
            return {"error": f"Agent not ready. Current state: {self.agent_state}"}

        try:
            self.agent_state = AgentState.PROCESSING
            # Actual processing should be implemented in subclasses
            self.agent_state = AgentState.READY
            return {"status": "success", "message": "Processing completed"}
        except Exception as e:
            self.agent_state = AgentState.ERROR
            logger.error(f"Processing error: {e}", exc_info=True)
            return {"error": str(e)}

    def load_training_data(self, training_data_path: str) -> Tuple[int, int]:
        """
        Load training data from parquet files.

        Args:
            training_data_path: Path to a parquet file or directory containing parquet files

        Returns:
            tuple: (Number of files loaded successfully, Number of files that failed to load)
        """
        logger.info(f"Loading training data from {training_data_path}")
        self.agent_state = AgentState.LOADING

        # Find parquet files
        if os.path.isdir(training_data_path):
            parquet_files = [os.path.join(training_data_path, f) for f in os.listdir(training_data_path)
                             if f.endswith('.parquet')]
        else:
            parquet_files = [training_data_path]

        if not parquet_files:
            logger.warning(f"No parquet files found at {training_data_path}")
            self.agent_state = AgentState.IDLE
            return 0, 0

        # Load data frames
        success_count = 0
        fail_count = 0
        self.data_frames = []

        for file in parquet_files:
            try:
                logger.debug(f"Loading {file}")
                df = pd.read_parquet(file)
                self.data_frames.append(df)
                success_count += 1
            except Exception as e:
                logger.error(f"Error loading {file}: {e}")
                fail_count += 1

        # Concatenate data frames if any were loaded successfully
        if self.data_frames:
            self.training_data = pd.concat(self.data_frames, ignore_index=True)
            logger.info(f"Loaded {len(self.training_data)} training samples from {success_count} files")
            self._process_agent_specific_data()
        else:
            logger.warning("No valid data frames found in the provided training data")

        self.agent_state = AgentState.READY if success_count > 0 else AgentState.ERROR
        return success_count, fail_count

    def _process_agent_specific_data(self) -> None:
        """
        Process agent-specific data after loading training data.

        This method should be overridden by subclasses to implement
        specific data processing logic.

        Returns:
            None
        """
        if not self.data_frames or all(df.empty for df in self.data_frames):
            logger.warning("No data frames to process")
            return

        # Subclasses should implement their specific data processing here
        logger.debug("Base agent data processing completed")

    def create_vector_store(self, text_field: str, metadata_fields: List[str],
                           embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2") -> bool:
        """
        Create a vector store from the training data.

        Args:
            text_field: Name of the field containing text to embed
            metadata_fields: List of fields to include as metadata
            embedding_model: Name of the HuggingFace embedding model to use

        Returns:
            bool: True if vector store was created successfully, False otherwise
        """
        if self.training_data is None or self.training_data.empty:
            logger.warning("No training data available to create vector store")
            return False

        try:
            logger.info(f"Creating vector store using {embedding_model}")
            self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

            # Get text and metadata
            texts = self.training_data[text_field].fillna("").tolist()
            metadatas = []
            for _, row in self.training_data.iterrows():
                metadata_item = {field: str(row.get(field, "")) for field in metadata_fields}
                metadatas.append(metadata_item)

            # Create vector store
            self.vector_store = FAISS.from_texts(texts, self.embeddings, metadatas=metadatas)
            logger.info(f"Vector store created with {len(texts)} entries")
            return True

        except Exception as e:
            logger.error(f"Error creating vector store: {e}", exc_info=True)
            return False

    def get_status(self) -> Dict[str, Any]:
        """
        Get the current status of the agent.

        Returns:
            dict: Status information
        """
        return {
            "agent_type": self.__class__.__name__,
            "state": self.agent_state.name,
            "device": self.device,
            "training_data_count": len(self.training_data) if self.training_data is not None else 0,
            "vector_store_ready": self.vector_store is not None,
            "initialization_time": self.initialization_time.isoformat() if self.initialization_time else None
        }

    def cleanup(self) -> None:
        """
        Clean up resources used by the agent.

        Returns:
            None
        """
        logger.info(f"Cleaning up {self.__class__.__name__}")
        self.vector_store = None
        # Force garbage collection for large models
        import gc
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()
        self.agent_state = AgentState.IDLE
