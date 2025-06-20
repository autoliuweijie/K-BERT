"""Shared pytest fixtures and configuration for K-BERT tests."""

import os
import json
import tempfile
import shutil
from pathlib import Path
from typing import Generator, Dict, Any
import pytest
import torch


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


@pytest.fixture
def mock_config() -> Dict[str, Any]:
    """Provide a mock configuration dictionary for testing."""
    return {
        "vocab_path": "test_vocab.txt",
        "pretrained_model_path": None,
        "output_model_path": "test_model.bin",
        "config_path": "test_config.json",
        "train_path": "test_train.tsv",
        "dev_path": "test_dev.tsv",
        "test_path": "test_test.tsv",
        "embedding": "bert",
        "encoder": "bert",
        "target": "bert",
        "seq_length": 128,
        "batch_size": 32,
        "learning_rate": 2e-5,
        "warmup": 0.1,
        "epochs_num": 3,
        "seed": 42,
        "device": "cpu",
        "world_size": 1,
        "rank": 0,
    }


@pytest.fixture
def sample_vocab_file(temp_dir: Path) -> Path:
    """Create a sample vocabulary file for testing."""
    vocab_path = temp_dir / "vocab.txt"
    vocab_content = [
        "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]",
        "hello", "world", "test", "example", "token"
    ]
    vocab_path.write_text("\n".join(vocab_content))
    return vocab_path


@pytest.fixture
def sample_model_config(temp_dir: Path) -> Path:
    """Create a sample model configuration file."""
    config_path = temp_dir / "config.json"
    config = {
        "vocab_size": 10000,
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "intermediate_size": 3072,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "attention_probs_dropout_prob": 0.1,
        "max_position_embeddings": 512,
        "type_vocab_size": 2,
        "initializer_range": 0.02,
        "layer_norm_eps": 1e-12,
    }
    config_path.write_text(json.dumps(config, indent=2))
    return config_path


@pytest.fixture
def sample_knowledge_graph(temp_dir: Path) -> Path:
    """Create a sample knowledge graph file for testing."""
    kg_path = temp_dir / "test_kg.spo"
    kg_content = [
        "entity1\trelation1\tentity2",
        "entity2\trelation2\tentity3",
        "entity3\trelation3\tentity4",
    ]
    kg_path.write_text("\n".join(kg_content))
    return kg_path


@pytest.fixture
def sample_dataset(temp_dir: Path) -> Path:
    """Create a sample dataset file for testing."""
    dataset_path = temp_dir / "test_data.tsv"
    dataset_content = [
        "1\tThis is a positive example.",
        "0\tThis is a negative example.",
        "1\tAnother positive test case.",
    ]
    dataset_path.write_text("\n".join(dataset_content))
    return dataset_path


@pytest.fixture
def mock_model():
    """Create a mock PyTorch model for testing."""
    class MockModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(768, 2)
            
        def forward(self, x):
            return self.linear(x)
    
    return MockModel()


@pytest.fixture
def device():
    """Get the appropriate device for testing (CPU)."""
    return torch.device("cpu")


@pytest.fixture(autouse=True)
def reset_random_seeds():
    """Reset random seeds before each test for reproducibility."""
    import random
    
    random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    # Only set numpy seed if numpy is available
    try:
        import numpy as np
        np.random.seed(42)
    except ImportError:
        pass


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer for testing."""
    class MockTokenizer:
        def __init__(self):
            self.vocab = {
                "[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, "[MASK]": 4,
                "hello": 5, "world": 6, "test": 7, "example": 8, "token": 9
            }
            self.inv_vocab = {v: k for k, v in self.vocab.items()}
            
        def tokenize(self, text):
            return text.lower().split()
            
        def convert_tokens_to_ids(self, tokens):
            return [self.vocab.get(token, self.vocab["[UNK]"]) for token in tokens]
            
        def convert_ids_to_tokens(self, ids):
            return [self.inv_vocab.get(id, "[UNK]") for id in ids]
    
    return MockTokenizer()


@pytest.fixture
def capture_logs():
    """Capture log messages during tests."""
    import logging
    
    class LogCapture:
        def __init__(self):
            self.records = []
            
        def __enter__(self):
            self.handler = logging.Handler()
            self.handler.emit = lambda record: self.records.append(record)
            logging.root.addHandler(self.handler)
            return self
            
        def __exit__(self, *args):
            logging.root.removeHandler(self.handler)
            
        def get_messages(self, level=None):
            if level:
                return [r.getMessage() for r in self.records if r.levelno >= level]
            return [r.getMessage() for r in self.records]
    
    return LogCapture()


@pytest.fixture
def env_setup(monkeypatch):
    """Setup environment variables for testing."""
    test_env = {
        "CUDA_VISIBLE_DEVICES": "",
        "MASTER_ADDR": "localhost",
        "MASTER_PORT": "12345",
        "WORLD_SIZE": "1",
        "RANK": "0",
    }
    
    for key, value in test_env.items():
        monkeypatch.setenv(key, value)
    
    return test_env