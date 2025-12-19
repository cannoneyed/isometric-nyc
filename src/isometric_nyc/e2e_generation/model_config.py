"""
Model configuration for the generation app.

Loads model configurations from app_config.json and provides
utilities for selecting and using different AI models.
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class ModelConfig:
  """Configuration for a single AI model."""

  name: str
  model_id: str
  api_key_env: str  # Environment variable name for the API key
  endpoint: str = "https://hub.oxen.ai/api/images/edit"
  num_inference_steps: int = 28
  model_type: str = "oxen"  # "oxen" for Oxen API, "local" for local inference
  use_base64: bool = False  # Use base64 encoding for local inference (faster)

  @property
  def api_key(self) -> str | None:
    """Get the API key from environment variables."""
    return os.getenv(self.api_key_env) if self.api_key_env else None

  @property
  def is_local(self) -> bool:
    """Check if this model uses local inference."""
    return self.model_type == "local"

  def to_dict(self) -> dict[str, Any]:
    """Convert to dictionary for JSON serialization (without API key)."""
    return {
      "name": self.name,
      "model_id": self.model_id,
      "endpoint": self.endpoint,
      "num_inference_steps": self.num_inference_steps,
      "model_type": self.model_type,
      "use_base64": self.use_base64,
    }


@dataclass
class AppConfig:
  """Full application configuration."""

  models: list[ModelConfig]
  default_model_id: str | None = None

  def get_model(self, model_id: str) -> ModelConfig | None:
    """Get a model configuration by its ID."""
    for model in self.models:
      if model.model_id == model_id:
        return model
    return None

  def get_default_model(self) -> ModelConfig | None:
    """Get the default model, or the first model if no default is set."""
    if self.default_model_id:
      return self.get_model(self.default_model_id)
    return self.models[0] if self.models else None

  def to_dict(self) -> dict[str, Any]:
    """Convert to dictionary for JSON serialization."""
    return {
      "models": [m.to_dict() for m in self.models],
      "default_model_id": self.default_model_id,
    }


def load_app_config(config_path: Path | None = None) -> AppConfig:
  """
  Load the application configuration from app_config.json.

  Args:
    config_path: Path to the config file. If None, looks in the generation dir.

  Returns:
    AppConfig object with model configurations

  If the config file doesn't exist, returns a default configuration
  with the legacy Oxen models.
  """
  if config_path is None:
    config_path = Path(__file__).parent / "app_config.json"

  if not config_path.exists():
    # Return default configuration with legacy models
    return get_default_config()

  with open(config_path) as f:
    data = json.load(f)

  models = []
  for model_data in data.get("models", []):
    models.append(
      ModelConfig(
        name=model_data["name"],
        model_id=model_data["model_id"],
        api_key_env=model_data.get("api_key_env", ""),
        endpoint=model_data.get("endpoint", "https://hub.oxen.ai/api/images/edit"),
        num_inference_steps=model_data.get("num_inference_steps", 28),
        model_type=model_data.get("model_type", "oxen"),
        use_base64=model_data.get("use_base64", False),
      )
    )

  return AppConfig(
    models=models,
    default_model_id=data.get("default_model_id"),
  )


def get_default_config() -> AppConfig:
  """
  Get the default configuration with legacy Oxen models.

  This is used when no app_config.json exists.
  """
  return AppConfig(
    models=[
      ModelConfig(
        name="Omni Water v1",
        model_id="cannoneyed-quiet-green-lamprey",
        api_key_env="OXEN_OMNI_v04_WATER_API_KEY",
      ),
      ModelConfig(
        name="Omni Water v2",
        model_id="cannoneyed-rural-rose-dingo",
        api_key_env="OXEN_OMNI_WATER_V2_API_KEY",
      ),
      ModelConfig(
        name="Omni (Original)",
        model_id="cannoneyed-gentle-gold-antlion",
        api_key_env="OXEN_OMNI_API_KEY",
      ),
    ],
    default_model_id="cannoneyed-quiet-green-lamprey",
  )


def save_app_config(config: AppConfig, config_path: Path | None = None) -> None:
  """
  Save the application configuration to app_config.json.

  Note: This does NOT save API keys - those should remain in environment variables.
  """
  if config_path is None:
    config_path = Path(__file__).parent / "app_config.json"

  data = {
    "models": [],
    "default_model_id": config.default_model_id,
  }

  for model in config.models:
    data["models"].append(
      {
        "name": model.name,
        "model_id": model.model_id,
        "api_key_env": model.api_key_env,
        "endpoint": model.endpoint,
        "num_inference_steps": model.num_inference_steps,
        "model_type": model.model_type,
        "use_base64": model.use_base64,
      }
    )

  with open(config_path, "w") as f:
    json.dump(data, f, indent=2)
