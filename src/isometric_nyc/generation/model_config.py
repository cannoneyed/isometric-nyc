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
  endpoint_env: str | None = (
    None  # Environment variable name for the endpoint (overrides endpoint if set)
  )
  num_inference_steps: int = 28
  model_type: str = (
    "oxen"  # "oxen" for Oxen API, "url" or "local" for URL-based inference
  )
  use_base64: bool = (
    True  # Use base64 encoding for URL inference (faster, default True)
  )
  is_water_mask: bool = (
    False  # If True, save output to water_mask column instead of generation
  )
  is_dark_mode: bool = (
    False  # If True, save output to dark_mode column instead of generation
  )
  # Optional preprocessing parameters applied to render images in the template
  desaturation: float | None = (
    None  # 0.0-1.0, amount to desaturate (0=no change, 1=grayscale)
  )
  gamma_shift: float | None = (
    None  # Gamma adjustment (1.0=no change, <1=darker, >1=brighter)
  )
  noise: float | None = None  # 0.0-1.0, amount of noise to add
  # Optional default prompt for this model (used if no user prompt is provided)
  prompt: str | None = None

  @property
  def api_key(self) -> str | None:
    """Get the API key from environment variables."""
    return os.getenv(self.api_key_env) if self.api_key_env else None

  @property
  def resolved_endpoint(self) -> str:
    """Get the endpoint, checking environment variable first if endpoint_env is set."""
    if self.endpoint_env:
      env_endpoint = os.getenv(self.endpoint_env)
      if env_endpoint:
        return env_endpoint
    return self.endpoint

  @property
  def is_local(self) -> bool:
    """Check if this model uses URL-based inference (local server or remote URL)."""
    return self.model_type in ("local", "url")

  def to_dict(self) -> dict[str, Any]:
    """Convert to dictionary for JSON serialization (without API key)."""
    result = {
      "name": self.name,
      "model_id": self.model_id,
      "endpoint": self.endpoint,
      "num_inference_steps": self.num_inference_steps,
      "model_type": self.model_type,
      "use_base64": self.use_base64,
    }
    # Include is_water_mask if set
    if self.is_water_mask:
      result["is_water_mask"] = self.is_water_mask
    # Include is_dark_mode if set
    if self.is_dark_mode:
      result["is_dark_mode"] = self.is_dark_mode
    # Include preprocessing params if set
    if self.desaturation is not None:
      result["desaturation"] = self.desaturation
    if self.gamma_shift is not None:
      result["gamma_shift"] = self.gamma_shift
    if self.noise is not None:
      result["noise"] = self.noise
    if self.prompt is not None:
      result["prompt"] = self.prompt
    return result


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
        endpoint_env=model_data.get("endpoint_env"),
        num_inference_steps=model_data.get("num_inference_steps", 28),
        model_type=model_data.get("model_type", "oxen"),
        use_base64=model_data.get("use_base64", True),
        is_water_mask=model_data.get("is_water_mask", False),
        is_dark_mode=model_data.get("is_dark_mode", False),
        desaturation=model_data.get("desaturation"),
        gamma_shift=model_data.get("gamma_shift"),
        noise=model_data.get("noise"),
        prompt=model_data.get("prompt"),
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
        name="Omni",
        model_id="cannoneyed-quiet-green-lamprey",
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
    model_dict = {
      "name": model.name,
      "model_id": model.model_id,
      "api_key_env": model.api_key_env,
      "endpoint": model.endpoint,
      "num_inference_steps": model.num_inference_steps,
      "model_type": model.model_type,
      "use_base64": model.use_base64,
    }
    # Include is_water_mask if set
    if model.is_water_mask:
      model_dict["is_water_mask"] = model.is_water_mask
    # Include is_dark_mode if set
    if model.is_dark_mode:
      model_dict["is_dark_mode"] = model.is_dark_mode
    # Include preprocessing params if set
    if model.desaturation is not None:
      model_dict["desaturation"] = model.desaturation
    if model.gamma_shift is not None:
      model_dict["gamma_shift"] = model.gamma_shift
    if model.noise is not None:
      model_dict["noise"] = model.noise
    if model.prompt is not None:
      model_dict["prompt"] = model.prompt

    data["models"].append(model_dict)

  with open(config_path, "w") as f:
    json.dump(data, f, indent=2)
