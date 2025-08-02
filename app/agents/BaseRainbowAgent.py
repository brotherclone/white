import pandas as pd
import torch
import os

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from pydantic import BaseModel, ConfigDict
from typing import Any, Optional
from app.enums.agent_state import AgentState

class BaseRainbowAgent(BaseModel):
    analyzer_name: Optional[str] = None
    processor_name: Optional[str] = None
    tokenizer_name: Optional[str] = None
    generator_name: Optional[str] = None
    llm_model_name: Optional[str] = None
    training_data: Any = None
    data_frames: Any = None
    embeddings: Any = None
    vector_store: Any = None
    device: str =None
    agent_state: Optional[AgentState] = None
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, **data):
        super().__init__(**data)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.agent_state = AgentState.IDLE
        self.data_frames = []
        self.embeddings = None
        self.vector_store = None

    def initialize(self):
        pass

    def process(self):
        pass

    def load_training_data(self, training_data_path: str) -> None:
        if os.path.isdir(training_data_path):
            parquet_files = [os.path.join(training_data_path, f) for f in os.listdir(training_data_path)
                             if f.endswith('.parquet')]
        else:
            parquet_files = [training_data_path]

        for file in parquet_files:
            try:
                df = pd.read_parquet(file)
                self.data_frames.append(df)
            except Exception as e:
                print(f"Error loading {file}: {e}")

        if self.data_frames:
            self.training_data = pd.concat(self.data_frames, ignore_index=True)
            print(f"Loaded {len(self.training_data)} training samples")
            self._process_agent_specific_data()
        else:
            print("No valid data frames found in the provided training data.")

    def _process_agent_specific_data(self)-> None :
        if self.data_frames is not None and not self.data_frames.empty:
            pass
        else:
            print("No data frames to process.")

    def create_vector_store(self, text_field:str, metadata_fields: list[str]) -> None:
        if self.training_data is None:
            print("No training data available to create vector store.")
            return
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        texts = self.training_data[text_field].fillna("").tolist()
        metadatas = []
        for _, row in self.training_data.iterrows():
            metadata_item = {field: row.get(field, "") for field in metadata_fields}
            metadatas.append(metadata_item)

        self.vector_store = FAISS.from_texts(texts, self.embeddings, metadatas=metadatas)
