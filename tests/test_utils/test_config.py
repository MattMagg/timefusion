"""
Tests for the configuration utilities.
"""

import pytest
import os
import json
import yaml
import tempfile
from timefusion.utils.config import Config, get_default_config, get_config_schema


def test_config_init():
    """Test Config initialization."""
    # Test with no config
    config = Config()
    assert config.config == {}

    # Test with config
    config = Config({"key": "value"})
    assert config.config == {"key": "value"}


def test_config_get_set():
    """Test Config get and set methods."""
    config = Config()

    # Test get with default
    assert config.get("key", "default") == "default"

    # Test set and get
    config.set("key", "value")
    assert config.get("key") == "value"

    # Test update
    config.update({"key2": "value2"})
    assert config.get("key") == "value"
    assert config.get("key2") == "value2"


def test_config_nested_get_set():
    """Test Config nested get and set methods."""
    config = Config()

    # Test nested set
    config.set("section.subsection.key", "value")

    # Test nested get
    assert config.get("section.subsection.key") == "value"
    assert config.get("section.subsection.nonexistent", "default") == "default"

    # Test nested structure
    assert config.config == {"section": {"subsection": {"key": "value"}}}


def test_config_load_save():
    """Test Config load and save methods."""
    config = Config({"key": "value"})

    # Create a temporary JSON file
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        json_path = f.name

    # Create a temporary YAML file
    with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
        yaml_path = f.name

    try:
        # Test JSON save and load
        config.save_to_file(json_path)
        assert os.path.exists(json_path)

        new_config = Config()
        new_config.load_from_file(json_path)
        assert new_config.config == {"key": "value"}

        # Test YAML save and load
        config.save_to_file(yaml_path)
        assert os.path.exists(yaml_path)

        new_config = Config()
        new_config.load_from_file(yaml_path)
        assert new_config.config == {"key": "value"}

        # Test unsupported format
        with pytest.raises(ValueError):
            config.save_to_file("config.unsupported")

        with pytest.raises(ValueError):
            config.load_from_file("config.unsupported")

        # Test file not found
        with pytest.raises(FileNotFoundError):
            config.load_from_file("nonexistent.json")
    finally:
        # Clean up
        if os.path.exists(json_path):
            os.remove(json_path)
        if os.path.exists(yaml_path):
            os.remove(yaml_path)


def test_config_update():
    """Test Config update method."""
    config = Config({
        "section1": {
            "key1": "value1",
            "subsection": {
                "key2": "value2"
            }
        }
    })

    # Test deep update
    config.update({
        "section1": {
            "key1": "new_value1",
            "subsection": {
                "key3": "value3"
            }
        },
        "section2": {
            "key4": "value4"
        }
    })

    # Check updated values
    assert config.get("section1.key1") == "new_value1"
    assert config.get("section1.subsection.key2") == "value2"
    assert config.get("section1.subsection.key3") == "value3"
    assert config.get("section2.key4") == "value4"


def test_config_validate():
    """Test Config validate method."""
    config = Config({
        "preprocessing": {
            "cleaner": {
                "outlier_method": "z-score",
                "outlier_threshold": 3.0
            }
        }
    })

    # Create a simple schema
    schema = {
        "type": "object",
        "properties": {
            "preprocessing": {
                "type": "object",
                "properties": {
                    "cleaner": {
                        "type": "object",
                        "properties": {
                            "outlier_method": {"type": "string", "enum": ["z-score", "iqr"]},
                            "outlier_threshold": {"type": "number", "minimum": 0}
                        }
                    }
                }
            }
        }
    }

    # Test valid configuration
    assert config.validate(schema) == True

    # Test invalid configuration
    config.set("preprocessing.cleaner.outlier_method", "invalid")
    assert config.validate(schema) == False


def test_config_from_env(monkeypatch):
    """Test Config from_env method."""
    # Set environment variables
    monkeypatch.setenv("TIMEFUSION_SECTION_KEY", "value")
    monkeypatch.setenv("TIMEFUSION_NESTED__KEY", "nested_value")
    monkeypatch.setenv("TIMEFUSION_JSON_VALUE", '{"key": "value"}')

    # Create config and load from environment
    config = Config()
    config.from_env()

    # Check values
    assert config.get("section.key") == "value"
    assert config.get("nested.key") == "nested_value"
    assert config.get("json_value") == {"key": "value"}


def test_get_default_config():
    """Test get_default_config function."""
    config = get_default_config()
    assert isinstance(config, Config)
    assert "preprocessing" in config.config
    assert "models" in config.config
    assert "evaluation" in config.config
    assert "logging" in config.config
    assert "visualization" in config.config
    assert "hpo" in config.config


def test_get_config_schema():
    """Test get_config_schema function."""
    schema = get_config_schema()
    assert isinstance(schema, dict)
    assert schema["type"] == "object"
    assert "properties" in schema
    assert "preprocessing" in schema["properties"]
    assert "models" in schema["properties"]
    assert "evaluation" in schema["properties"]
    assert "logging" in schema["properties"]
    assert "visualization" in schema["properties"]
    assert "hpo" in schema["properties"]
