"""
Unified model interface for CEI benchmark evaluation.

Provides a consistent API for evaluating scenarios across different model types:
- OpenAI API models (GPT-4, GPT-5-mini, etc.)
- Anthropic API models (Claude 3, Claude Haiku 4.5, etc.)
- Google API models (Gemini 2.0 Flash, etc.)
- xAI API models (Grok 4.1 Fast, etc.)
- Fireworks API models (Kimi, Qwen, DeepSeek, MiniMax, etc.)
- Together API models (Llama, Mistral, Gemma, etc.)
- HuggingFace local models (Llama, Mistral, etc.)
- Baseline systems (sentiment, lexicon)

Example:
    >>> model = get_model("gpt-5-mini", provider="openai")
    >>> output = model.predict(prompt, return_confidence=True)
    >>> print(output.subtype_prediction, output.confidence)
"""

from __future__ import annotations

import json
import logging
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Supported model types."""

    OPENAI_API = "openai"
    ANTHROPIC_API = "anthropic"
    GOOGLE_API = "google"
    XAI_API = "xai"
    FIREWORKS_API = "fireworks"
    TOGETHER_API = "together"
    HUGGINGFACE_LOCAL = "huggingface"
    HUGGINGFACE_INFERENCE = "hf_inference"
    BASELINE_SENTIMENT = "baseline_sentiment"
    BASELINE_LEXICON = "baseline_lexicon"
    MOCK = "mock"


@dataclass
class ModelConfig:
    """Configuration for a specific model."""

    model_type: ModelType
    model_id: str
    api_key_env: Optional[str] = None
    max_tokens: int = 512
    temperature: float = 0.0
    supports_hidden_states: bool = False
    supports_logits: bool = False
    hidden_dim: Optional[int] = None
    n_layers: Optional[int] = None
    is_thinking_model: bool = False  # For models that output reasoning tokens

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "model_type": self.model_type.value,
            "model_id": self.model_id,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "supports_hidden_states": self.supports_hidden_states,
            "supports_logits": self.supports_logits,
            "hidden_dim": self.hidden_dim,
            "n_layers": self.n_layers,
            "is_thinking_model": self.is_thinking_model,
        }


# Static model registry with common configurations
MODEL_REGISTRY: Dict[str, ModelConfig] = {
    # OpenAI models
    "gpt-4": ModelConfig(
        model_type=ModelType.OPENAI_API,
        model_id="gpt-4-0125-preview",
        api_key_env="OPENAI_API_KEY",
        supports_logits=True,
    ),
    "gpt-4o": ModelConfig(
        model_type=ModelType.OPENAI_API,
        model_id="gpt-4o",
        api_key_env="OPENAI_API_KEY",
        supports_logits=True,
    ),
    "gpt-5-mini": ModelConfig(
        model_type=ModelType.OPENAI_API,
        model_id="gpt-5-mini",
        api_key_env="OPENAI_API_KEY",
        supports_logits=True,
        is_thinking_model=True,
    ),
    # Anthropic models
    "claude-3-opus": ModelConfig(
        model_type=ModelType.ANTHROPIC_API,
        model_id="claude-3-opus-20240229",
        api_key_env="ANTHROPIC_API_KEY",
    ),
    "claude-3.5-sonnet": ModelConfig(
        model_type=ModelType.ANTHROPIC_API,
        model_id="claude-3-5-sonnet-20241022",
        api_key_env="ANTHROPIC_API_KEY",
    ),
    "claude-haiku-4-5": ModelConfig(
        model_type=ModelType.ANTHROPIC_API,
        model_id="claude-haiku-4-5",
        api_key_env="ANTHROPIC_API_KEY",
    ),
    # Google models
    "gemini-2.0-flash": ModelConfig(
        model_type=ModelType.GOOGLE_API,
        model_id="gemini-2.0-flash",
        api_key_env="GOOGLE_API_KEY",
    ),
    # xAI models
    "grok-4-1-fast": ModelConfig(
        model_type=ModelType.XAI_API,
        model_id="grok-4-1-fast",
        api_key_env="XAI_API_KEY",
    ),
    # Baselines
    "baseline-sentiment": ModelConfig(
        model_type=ModelType.BASELINE_SENTIMENT,
        model_id="vader",
    ),
    "mock": ModelConfig(
        model_type=ModelType.MOCK,
        model_id="mock",
        supports_hidden_states=True,
        supports_logits=True,
        hidden_dim=128,
        n_layers=4,
    ),
}


@dataclass
class ModelOutput:
    """Output from a model prediction."""

    subtype_prediction: str
    emotion_prediction: str
    vad_prediction: Tuple[float, float, float]
    confidence: Optional[float] = None
    raw_response: Optional[str] = None
    logits: Optional[np.ndarray] = None
    hidden_states: Optional[Dict[int, np.ndarray]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "subtype_prediction": self.subtype_prediction,
            "emotion_prediction": self.emotion_prediction,
            "vad_prediction": list(self.vad_prediction),
            "confidence": self.confidence,
            "raw_response": self.raw_response,
            "metadata": self.metadata,
        }
        if self.logits is not None:
            result["has_logits"] = True
        if self.hidden_states is not None:
            result["has_hidden_states"] = True
            result["hidden_state_layers"] = list(self.hidden_states.keys())
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelOutput":
        """Create from dictionary."""
        return cls(
            subtype_prediction=data["subtype_prediction"],
            emotion_prediction=data["emotion_prediction"],
            vad_prediction=tuple(data["vad_prediction"]),
            confidence=data.get("confidence"),
            raw_response=data.get("raw_response"),
            metadata=data.get("metadata", {}),
        )


def parse_json_response(response: str) -> Dict[str, Any]:
    """Parse JSON response from model, handling code fences."""
    # Strip code fences if present
    cleaned = response.strip()
    code_fence_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', cleaned, re.DOTALL)
    if code_fence_match:
        cleaned = code_fence_match.group(1).strip()

    # Try to extract JSON object
    json_match = re.search(r'\{[^{}]*\}', cleaned, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group())
            return {
                "subtype": data.get("subtype", data.get("emotion", "unknown")),
                "emotion": data.get("emotion", "unknown"),
                "valence": float(data.get("valence", 0.0)),
                "arousal": float(data.get("arousal", 0.0)),
                "dominance": float(data.get("dominance", 0.0)),
                "confidence": data.get("confidence"),
            }
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

    # Fallback
    return {
        "subtype": "unknown",
        "emotion": "unknown",
        "valence": 0.0,
        "arousal": 0.0,
        "dominance": 0.0,
        "confidence": None,
    }


class ModelInterface(ABC):
    """Abstract base class for model interfaces."""

    def __init__(self, config: ModelConfig) -> None:
        """Initialize with model configuration."""
        self.config = config
        self.model_id = config.model_id

    @abstractmethod
    def predict(
        self,
        prompt: str,
        return_logits: bool = False,
        return_confidence: bool = True,
        return_hidden_states: bool = False,
    ) -> ModelOutput:
        """Generate prediction for a prompt."""
        ...

    def predict_batch(
        self,
        prompts: List[str],
        return_logits: bool = False,
        return_confidence: bool = True,
        return_hidden_states: bool = False,
    ) -> List[ModelOutput]:
        """Generate predictions for multiple prompts."""
        return [
            self.predict(p, return_logits, return_confidence, return_hidden_states)
            for p in prompts
        ]

    @property
    def supports_hidden_states(self) -> bool:
        return self.config.supports_hidden_states

    @property
    def supports_logits(self) -> bool:
        return self.config.supports_logits


class MockModel(ModelInterface):
    """Mock model for testing purposes."""

    SUBTYPES = ["sarcasm-irony", "mixed-signals", "strategic-politeness", "passive-aggression", "deflection-misdirection"]
    EMOTIONS = ["joy", "trust", "fear", "surprise", "sadness", "disgust", "anger", "anticipation"]

    def __init__(self, config: ModelConfig, seed: int = 42) -> None:
        super().__init__(config)
        self.rng = np.random.default_rng(seed)

    def predict(
        self,
        prompt: str,
        return_logits: bool = False,
        return_confidence: bool = True,
        return_hidden_states: bool = False,
    ) -> ModelOutput:
        prompt_hash = hash(prompt) % 1000
        self.rng = np.random.default_rng(prompt_hash)

        return ModelOutput(
            subtype_prediction=self.rng.choice(self.SUBTYPES),
            emotion_prediction=self.rng.choice(self.EMOTIONS),
            vad_prediction=tuple(self.rng.uniform(-1, 1, 3).tolist()),
            confidence=self.rng.uniform(0.5, 1.0) if return_confidence else None,
            raw_response=f"Mock response for: {prompt[:50]}...",
            logits=self.rng.standard_normal(len(self.SUBTYPES)) if return_logits else None,
            hidden_states={i: self.rng.standard_normal(self.config.hidden_dim or 128) for i in range(self.config.n_layers or 4)} if return_hidden_states else None,
            metadata={"model": "mock", "seed": prompt_hash},
        )


class OpenAIModel(ModelInterface):
    """OpenAI API model interface."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        self._client = None

    @property
    def client(self):
        if self._client is None:
            import openai
            self._client = openai.OpenAI()
        return self._client

    def predict(
        self,
        prompt: str,
        return_logits: bool = False,
        return_confidence: bool = True,
        return_hidden_states: bool = False,
    ) -> ModelOutput:
        kwargs = {
            "model": self.config.model_id,
            "messages": [{"role": "user", "content": prompt}],
        }

        # Handle thinking models (gpt-5-mini, o1, o3)
        if self.config.is_thinking_model:
            kwargs["max_completion_tokens"] = self.config.max_tokens
            kwargs["reasoning_effort"] = "minimal"
        else:
            kwargs["max_tokens"] = self.config.max_tokens
            kwargs["temperature"] = self.config.temperature
            if return_logits:
                kwargs["logprobs"] = True

        response = self.client.chat.completions.create(**kwargs)
        raw_response = response.choices[0].message.content
        parsed = parse_json_response(raw_response)

        return ModelOutput(
            subtype_prediction=parsed["subtype"],
            emotion_prediction=parsed["emotion"],
            vad_prediction=(parsed["valence"], parsed["arousal"], parsed["dominance"]),
            confidence=parsed.get("confidence"),
            raw_response=raw_response,
            metadata={"model": self.config.model_id, "usage": dict(response.usage) if response.usage else {}},
        )


class AnthropicModel(ModelInterface):
    """Anthropic API model interface."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        self._client = None

    @property
    def client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic()
        return self._client

    def predict(
        self,
        prompt: str,
        return_logits: bool = False,
        return_confidence: bool = True,
        return_hidden_states: bool = False,
    ) -> ModelOutput:
        response = self.client.messages.create(
            model=self.config.model_id,
            max_tokens=self.config.max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )

        raw_response = response.content[0].text
        parsed = parse_json_response(raw_response)

        return ModelOutput(
            subtype_prediction=parsed["subtype"],
            emotion_prediction=parsed["emotion"],
            vad_prediction=(parsed["valence"], parsed["arousal"], parsed["dominance"]),
            confidence=parsed.get("confidence"),
            raw_response=raw_response,
            metadata={
                "model": self.config.model_id,
                "usage": {"input_tokens": response.usage.input_tokens, "output_tokens": response.usage.output_tokens},
            },
        )


class GoogleModel(ModelInterface):
    """Google Gemini API model interface using the new google-genai SDK."""

    # Timeout for API calls (seconds)
    API_TIMEOUT = 60

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from google import genai
            api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY or GEMINI_API_KEY environment variable required")
            # Configure client with timeout via httpx_client
            self._client = genai.Client(
                api_key=api_key,
                http_options={"timeout": self.API_TIMEOUT},
            )
        return self._client

    def predict(
        self,
        prompt: str,
        return_logits: bool = False,
        return_confidence: bool = True,
        return_hidden_states: bool = False,
    ) -> ModelOutput:
        import concurrent.futures
        from google.genai import types

        def _call_api():
            return self.client.models.generate_content(
                model=self.config.model_id,
                contents=prompt,
                config=types.GenerateContentConfig(
                    max_output_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                ),
            )

        # Use thread pool with timeout to prevent hanging
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_call_api)
            try:
                response = future.result(timeout=self.API_TIMEOUT)
            except concurrent.futures.TimeoutError:
                raise TimeoutError(f"Gemini API call timed out after {self.API_TIMEOUT}s")

        raw_response = response.text
        parsed = parse_json_response(raw_response)

        return ModelOutput(
            subtype_prediction=parsed["subtype"],
            emotion_prediction=parsed["emotion"],
            vad_prediction=(parsed["valence"], parsed["arousal"], parsed["dominance"]),
            confidence=parsed.get("confidence"),
            raw_response=raw_response,
            metadata={"model": self.config.model_id},
        )


class XAIModel(ModelInterface):
    """xAI (Grok) API model interface using OpenAI-compatible endpoint."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        self._client = None

    @property
    def client(self):
        if self._client is None:
            import openai
            api_key = os.environ.get("XAI_API_KEY")
            if not api_key:
                raise ValueError("XAI_API_KEY environment variable required")
            self._client = openai.OpenAI(
                api_key=api_key,
                base_url="https://api.x.ai/v1",
            )
        return self._client

    def predict(
        self,
        prompt: str,
        return_logits: bool = False,
        return_confidence: bool = True,
        return_hidden_states: bool = False,
    ) -> ModelOutput:
        response = self.client.chat.completions.create(
            model=self.config.model_id,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
        )

        raw_response = response.choices[0].message.content
        parsed = parse_json_response(raw_response)

        return ModelOutput(
            subtype_prediction=parsed["subtype"],
            emotion_prediction=parsed["emotion"],
            vad_prediction=(parsed["valence"], parsed["arousal"], parsed["dominance"]),
            confidence=parsed.get("confidence"),
            raw_response=raw_response,
            metadata={"model": self.config.model_id, "usage": dict(response.usage) if response.usage else {}},
        )


class FireworksModel(ModelInterface):
    """Fireworks AI API model interface using OpenAI-compatible endpoint."""

    # Thinking models that may output reasoning before JSON
    THINKING_MODELS = ["minimax", "deepseek"]

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        self._client = None

    @property
    def client(self):
        if self._client is None:
            import openai
            api_key = os.environ.get("FIREWORKS_API_KEY")
            if not api_key:
                raise ValueError("FIREWORKS_API_KEY environment variable required")
            self._client = openai.OpenAI(
                api_key=api_key,
                base_url="https://api.fireworks.ai/inference/v1",
            )
        return self._client

    def predict(
        self,
        prompt: str,
        return_logits: bool = False,
        return_confidence: bool = True,
        return_hidden_states: bool = False,
    ) -> ModelOutput:
        # Check if thinking model (needs more tokens for reasoning)
        is_thinking = any(t in self.config.model_id.lower() for t in self.THINKING_MODELS)
        max_tokens = 2048 if is_thinking else self.config.max_tokens

        response = self.client.chat.completions.create(
            model=self.config.model_id,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=self.config.temperature,
        )

        raw_response = response.choices[0].message.content
        parsed = parse_json_response(raw_response)

        return ModelOutput(
            subtype_prediction=parsed["subtype"],
            emotion_prediction=parsed["emotion"],
            vad_prediction=(parsed["valence"], parsed["arousal"], parsed["dominance"]),
            confidence=parsed.get("confidence"),
            raw_response=raw_response,
            metadata={"model": self.config.model_id, "usage": dict(response.usage) if response.usage else {}},
        )


class TogetherModel(ModelInterface):
    """Together AI API model interface using OpenAI-compatible endpoint."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        self._client = None

    @property
    def client(self):
        if self._client is None:
            import openai
            api_key = os.environ.get("TOGETHER_API_KEY")
            if not api_key:
                raise ValueError("TOGETHER_API_KEY environment variable required")
            self._client = openai.OpenAI(
                api_key=api_key,
                base_url="https://api.together.xyz/v1",
            )
        return self._client

    def predict(
        self,
        prompt: str,
        return_logits: bool = False,
        return_confidence: bool = True,
        return_hidden_states: bool = False,
    ) -> ModelOutput:
        response = self.client.chat.completions.create(
            model=self.config.model_id,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
        )

        raw_response = response.choices[0].message.content
        parsed = parse_json_response(raw_response)

        return ModelOutput(
            subtype_prediction=parsed["subtype"],
            emotion_prediction=parsed["emotion"],
            vad_prediction=(parsed["valence"], parsed["arousal"], parsed["dominance"]),
            confidence=parsed.get("confidence"),
            raw_response=raw_response,
            metadata={"model": self.config.model_id, "usage": dict(response.usage) if response.usage else {}},
        )


class BaselineSentimentModel(ModelInterface):
    """Baseline model using VADER sentiment analysis."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        self._analyzer = None

    @property
    def analyzer(self):
        if self._analyzer is None:
            from nltk.sentiment.vader import SentimentIntensityAnalyzer
            import nltk
            try:
                self._analyzer = SentimentIntensityAnalyzer()
            except LookupError:
                nltk.download("vader_lexicon", quiet=True)
                self._analyzer = SentimentIntensityAnalyzer()
        return self._analyzer

    def predict(
        self,
        prompt: str,
        return_logits: bool = False,
        return_confidence: bool = True,
        return_hidden_states: bool = False,
    ) -> ModelOutput:
        utterance_match = re.search(r'"([^"]+)"', prompt)
        utterance = utterance_match.group(1) if utterance_match else prompt

        scores = self.analyzer.polarity_scores(utterance)
        compound = scores["compound"]

        if compound >= 0.5:
            emotion = "joy"
        elif compound >= 0.1:
            emotion = "trust"
        elif compound <= -0.5:
            emotion = "anger"
        elif compound <= -0.1:
            emotion = "sadness"
        else:
            emotion = "anticipation"

        return ModelOutput(
            subtype_prediction="sarcasm-irony",
            emotion_prediction=emotion,
            vad_prediction=(compound, abs(compound), 0.0),
            confidence=abs(compound) if return_confidence else None,
            raw_response=str(scores),
            metadata={"model": "vader", "scores": scores},
        )


def get_model(model_id: str, provider: Optional[str] = None, **kwargs) -> ModelInterface:
    """
    Factory function to get a model interface.

    Args:
        model_id: Model identifier (e.g., "gpt-5-mini", "accounts/fireworks/models/kimi-k2")
        provider: Optional provider hint (openai, anthropic, google, xai, fireworks, together)
        **kwargs: Additional arguments for model constructor

    Returns:
        ModelInterface instance
    """
    # Check static registry first
    if model_id in MODEL_REGISTRY:
        config = MODEL_REGISTRY[model_id]
        return _create_model_from_config(config, **kwargs)

    # Infer provider from model_id if not specified
    if provider is None:
        provider = _infer_provider(model_id)

    # Create dynamic config
    model_type = _provider_to_model_type(provider)
    config = ModelConfig(
        model_type=model_type,
        model_id=model_id,
        max_tokens=kwargs.pop("max_tokens", 512),
        temperature=kwargs.pop("temperature", 0.0),
        is_thinking_model=any(t in model_id.lower() for t in ["minimax", "deepseek", "gpt-5", "o1-", "o3-"]),
    )

    return _create_model_from_config(config, **kwargs)


def _infer_provider(model_id: str) -> str:
    """Infer provider from model ID."""
    model_lower = model_id.lower()

    if "gpt" in model_lower or "o1-" in model_lower or "o3-" in model_lower:
        return "openai"
    elif "claude" in model_lower:
        return "anthropic"
    elif "gemini" in model_lower:
        return "google"
    elif "grok" in model_lower:
        return "xai"
    elif "accounts/fireworks" in model_lower or any(x in model_lower for x in ["kimi", "qwen", "deepseek", "minimax", "glm"]):
        return "fireworks"
    elif any(x in model_lower for x in ["meta-llama", "mistralai", "google/gemma"]):
        return "together"
    else:
        return "mock"


def _provider_to_model_type(provider: str) -> ModelType:
    """Convert provider string to ModelType."""
    mapping = {
        "openai": ModelType.OPENAI_API,
        "anthropic": ModelType.ANTHROPIC_API,
        "google": ModelType.GOOGLE_API,
        "xai": ModelType.XAI_API,
        "fireworks": ModelType.FIREWORKS_API,
        "together": ModelType.TOGETHER_API,
        "mock": ModelType.MOCK,
    }
    return mapping.get(provider, ModelType.MOCK)


def _create_model_from_config(config: ModelConfig, **kwargs) -> ModelInterface:
    """Create model instance from config."""
    if config.model_type == ModelType.MOCK:
        return MockModel(config, **kwargs)
    elif config.model_type == ModelType.OPENAI_API:
        return OpenAIModel(config)
    elif config.model_type == ModelType.ANTHROPIC_API:
        return AnthropicModel(config)
    elif config.model_type == ModelType.GOOGLE_API:
        return GoogleModel(config)
    elif config.model_type == ModelType.XAI_API:
        return XAIModel(config)
    elif config.model_type == ModelType.FIREWORKS_API:
        return FireworksModel(config)
    elif config.model_type == ModelType.TOGETHER_API:
        return TogetherModel(config)
    elif config.model_type == ModelType.BASELINE_SENTIMENT:
        return BaselineSentimentModel(config)
    elif config.model_type == ModelType.HUGGINGFACE_LOCAL:
        raise NotImplementedError("HuggingFace local models require GPU setup")
    else:
        raise ValueError(f"Unsupported model type: {config.model_type}")


def list_available_models() -> List[str]:
    """Return list of available model names in static registry."""
    return sorted(MODEL_REGISTRY.keys())


def get_model_config(model_name: str) -> ModelConfig:
    """Get configuration for a specific model."""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}")
    return MODEL_REGISTRY[model_name]
