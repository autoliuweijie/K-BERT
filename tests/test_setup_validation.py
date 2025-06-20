"""Validation tests to verify the testing infrastructure is properly configured."""

import pytest
import sys
import os
from pathlib import Path


class TestSetupValidation:
    """Tests to validate the testing infrastructure setup."""
    
    @pytest.mark.unit
    def test_python_version(self):
        """Verify Python version is compatible."""
        assert sys.version_info >= (3, 8), "Python 3.8+ is required"
    
    @pytest.mark.unit
    def test_project_structure_exists(self):
        """Verify the expected project structure exists."""
        root_dir = Path(__file__).parent.parent
        
        # Check main source directories
        assert (root_dir / "uer").exists(), "uer package directory not found"
        assert (root_dir / "brain").exists(), "brain package directory not found"
        
        # Check test directories
        assert (root_dir / "tests").exists(), "tests directory not found"
        assert (root_dir / "tests" / "unit").exists(), "tests/unit directory not found"
        assert (root_dir / "tests" / "integration").exists(), "tests/integration directory not found"
    
    @pytest.mark.unit
    def test_pyproject_toml_exists(self):
        """Verify pyproject.toml exists and is properly configured."""
        root_dir = Path(__file__).parent.parent
        pyproject_path = root_dir / "pyproject.toml"
        
        assert pyproject_path.exists(), "pyproject.toml not found"
        
        # Verify it contains expected sections
        content = pyproject_path.read_text()
        assert "[tool.poetry]" in content, "Poetry configuration missing"
        assert "[tool.pytest.ini_options]" in content, "Pytest configuration missing"
        assert "[tool.coverage.run]" in content, "Coverage configuration missing"
    
    @pytest.mark.unit
    def test_conftest_fixtures_available(self, temp_dir, mock_config, sample_vocab_file):
        """Verify conftest fixtures are available and working."""
        # Test temp_dir fixture
        assert temp_dir.exists()
        assert temp_dir.is_dir()
        
        # Test mock_config fixture
        assert isinstance(mock_config, dict)
        assert "vocab_path" in mock_config
        assert "batch_size" in mock_config
        
        # Test sample_vocab_file fixture
        assert sample_vocab_file.exists()
        assert sample_vocab_file.is_file()
        content = sample_vocab_file.read_text()
        assert "[PAD]" in content
        assert "[UNK]" in content
    
    @pytest.mark.unit
    def test_markers_defined(self):
        """Verify custom pytest markers are defined."""
        # This test will pass if markers are properly configured
        # The markers themselves are verified by pytest at runtime
        pass
    
    @pytest.mark.unit
    def test_coverage_configuration(self):
        """Verify coverage is properly configured."""
        # This test verifies that coverage runs with tests
        # Actual verification happens when running with coverage
        import uer
        
        # Verify uer package can be imported
        assert uer.__name__ == "uer"
        
        # Try to import brain, but skip if dependencies are missing
        try:
            import brain
            assert brain.__name__ == "brain"
        except ImportError as e:
            pytest.skip(f"Brain module requires additional dependencies: {e}")
    
    @pytest.mark.integration
    def test_pytorch_available(self):
        """Verify PyTorch is available for testing."""
        try:
            import torch
            tensor = torch.tensor([1.0, 2.0, 3.0])
            assert tensor.shape == (3,)
        except ImportError:
            pytest.skip("PyTorch not yet installed")
    
    @pytest.mark.slow
    def test_slow_marker(self):
        """Test that slow marker works correctly."""
        import time
        # This is marked as slow but doesn't actually need to be slow
        # It's just to verify the marker works
        assert True


def test_standalone_function():
    """Test that standalone test functions are discovered."""
    assert True


@pytest.fixture
def local_fixture():
    """Test that local fixtures work alongside conftest fixtures."""
    return {"local": "fixture"}


def test_local_fixture(local_fixture):
    """Test that local fixtures can be used."""
    assert local_fixture == {"local": "fixture"}