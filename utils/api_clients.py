import os
import time
import random
from dotenv import load_dotenv

from typing import Dict, List, Callable, Tuple, Any

import openai
import anthropic
from google import genai
from together import Together

from utils.misc.logging_setup import logger

from utils.paths.base import DEFAULT_PATHS
from utils.misc.variables import MODEL_CONFIGS, PROVIDER_CONFIGS


class APIClientManager:
    """Manages multiple API clients for different LLM providers"""

    def __init__(self, models: str = "all"):
        """
        Args:
            * models (str): Comma-separated list of model names to use OR
                category of models ('all', 'all-excl-dummy', 'paid', 'dummy')
                (default: 'all')
        """
        self.models: str = models
        self.clients: Dict[str, Callable] = {}
    
    def init_clients(self) -> bool:
        """
        Initialize API clients for all providers

        Raises:
            * Exception if any client fails to initialize
        """

        # Load API keys
        api_keys = self._load_api_keys()

        # Get the providers needed for the selected models
        model_names, providers = self.get_models_info()

        # Ensure that we have correct API keys for the selected providers
        self._check_api_keys_present(providers=providers, api_keys=api_keys)
        
        # Initialize clients for each provider that's needed
        try:
            for provider in set(providers):
                if not PROVIDER_CONFIGS[provider]["requires_client"]:
                    continue  # Skip dummy
                
                # Initialize the client
                api_key = api_keys[provider]
                self.clients[provider] = self._init_provider_client(
                    provider=provider, api_key=api_key
                )
                logger.info(f"Initialized {provider} client")
        except Exception as e:
            logger.error_status(
                f"Failed to initialize clients: {str(e)}", exc_info=True
            )
            raise
        
        return model_names, providers

    
    def _init_provider_client(self, provider: str, api_key: str) -> Any:
        """
        Initialize a client for a specific provider

        Args:
            * provider: Provider name
            * api_key: API key for the provider

        Returns:
            Initialized client object
        """
        initializers = {
            "openai": lambda: setattr(openai, 'api_key', api_key) or openai,
            "claude": lambda: anthropic.Anthropic(api_key=api_key),
            "gemini": lambda: genai.Client(api_key=api_key),
            "together": lambda: Together(api_key=api_key),
        }
        
        if provider not in initializers:
            raise ValueError(f"No initializer found for provider: {provider}")
        
        return initializers[provider]()

    def get_models_info(self)-> Tuple[List[str], List[str]]:
        """
        Get model names and providers

        Returns:
            * Tuple of model names and their providers
        """

        if self.models == "all":
            names = list(MODEL_CONFIGS.keys())
        elif self.models == "all-excl-dummy":
            names = [
                name
                for name, config in MODEL_CONFIGS.items()
                if config["category"] != "dummy"
            ]
        elif self.models in ["paid", "dummy"]:
            names = [
                name
                for name, config in MODEL_CONFIGS.items()
                if config["category"] == self.models
            ]
        else:
            # Comma-separated list of model names
            names = [model.strip() for model in self.models.split(",")]
        
        # Provider always retrieved the same, irrespective of category
        providers = [
            MODEL_CONFIGS[name]["provider"] for name in names
        ]
        
        return names, providers


    def _load_api_keys(self) -> Dict[str, str]:
        """
        Load API keys from environment variables or .env file

        Returns:
            * api_keys (dict): Dictionary of API keys
                {
                    "openai": "openai_api_key",
                    "claude": "claude_api_key",
                    "gemini": "gemini_api_key",
                    "together": "togetherai_api_key",
                }
        """

        # Load from .env file
        env_file = DEFAULT_PATHS["env_file"]
        try:
            load_dotenv(env_file)
            logger.info(f"Loaded environment variables from {env_file}")
        except Exception as e:
            logger.error_status(
                f"Error loading {env_file}: {str(e)}", exc_info=True
            )
            raise

        # Get the API keys from environment variables using the provider config
        api_keys = {
            provider: os.environ.get(config["env_key"])
            for provider, config in PROVIDER_CONFIGS.items()
            if config["env_key"] is not None
        }

        return api_keys

    def _check_api_keys_present(
        self, providers: List[str], api_keys: Dict[str, str]
    ) -> bool:
        """
        Check if all required API keys are present in environment variables

        Args:
            * providers (list): List of provider names
            * api_keys (dict): Dictionary of API keys

        Returns:
            * Exception if any required API key is missing
        """

        # For each provider, check if the API key is present
        for provider in set(providers):
            provider_config = PROVIDER_CONFIGS.get(provider)
            
            # Only check for API key if the provider requires a client
            if (
                provider_config["requires_client"]
                and not api_keys.get(provider)
            ):
                logger.error(f"Missing API key for provider: {provider}")
                raise ValueError(f"Missing API key for provider: {provider}")

    def query_model(self, model_name: str, prompt: str) -> str:
        """
        Query a model by name

        Args:
            * model_name (str): Name of the model to query
            * prompt (str): Prompt to send to the model

        Returns:
            * response (str): Model response or error message
        """

        # Get the model config, as well as provider and model_id
        config = MODEL_CONFIGS[model_name]
        provider = config["provider"]
        model_id = config["model_id"]

        try:
            return self._query_provider(
                provider=provider, model_id=model_id, prompt=prompt
            )
        except Exception as e:
            error_msg = f"Error querying {model_name}: {str(e)}"
            logger.error_status(error_msg, exc_info=True)
            raise

    def retry_with_backoff(
        self,
        func: Callable,
        *args,
        max_retries: int = 5,
        initial_delay: int = 2,
    ):
        """
        Retry a function with exponential backoff
        Args:
            * func: Function to retry
            * *args: Arguments to pass to func
            * max_retries: Maximum number of retry attempts
            * initial_delay: Initial backoff delay in seconds
        Returns:
            * The result from the function
        """
        for attempt in range(1, max_retries + 1):
            try:
                return func(*args)
            except Exception as e:
                if attempt >= max_retries:
                    logger.error_status(
                        f"Maximum retries ({max_retries}) exceeded. "
                        f"Last error: {str(e)}",
                        exc_info=True,
                    )
                    raise

                # Add jitter to avoid thundering herd
                delay = initial_delay * (2 ** (attempt - 1)) + random.uniform(
                    0, 1
                )
                logger.warning_status(
                    f"API error: {str(e)}. Retrying in {delay:.1f}s "
                    f"(attempt {attempt}/{max_retries})..."
                )
                time.sleep(delay)

    def _query_provider(
        self, provider: str, model_id: str, prompt: str
    ) -> str:
        """
        Route to the appropriate provider with retry logic

        Args:
            * provider (str): Provider name
                ('openai', 'claude', 'gemini', 'together', 'dummy')
            * model_id (str): Model ID for the provider
            * prompt (str): Prompt to send to the model

        Returns:
            * response (str): Model response
        """

        # Map providers to their query functions
        provider_methods = {
            "dummy": lambda: self._query_dummy(prompt),
            "openai": lambda: self.retry_with_backoff(
                self._query_openai, model_id, prompt
            ),
            "claude": lambda: self.retry_with_backoff(
                self._query_claude, model_id, prompt
            ),
            "gemini": lambda: self.retry_with_backoff(
                self._query_gemini, model_id, prompt
            ),
            "together": lambda: self.retry_with_backoff(
                self._query_together,
                model_id,
                prompt,
                max_retries=7,
                initial_delay=5,
            ),
        }

        if provider not in provider_methods:
            raise ValueError(f"Unknown provider: {provider}")

        return provider_methods[provider]()

    def _query_openai(self, model_id: str, prompt: str) -> str:
        """
        Query OpenAI models

        Args:
            * model_id (str): OpenAI model ID
            * prompt (str): Prompt to send to the model

        Returns:
            * response (str): Model response
        """
        client = self.clients["openai"]
        response = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content.strip()

    def _query_claude(self, model_id: str, prompt: str) -> str:
        """
        Query Claude models

        Args:
            * model_id (str): Claude model ID
            * prompt (str): Prompt to send to the model

        Returns:
            * response (str): Model response
        """
        client = self.clients["claude"]
        response = client.messages.create(
            model=model_id,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip()

    def _query_gemini(self, model_id: str, prompt: str) -> str:
        """
        Query Gemini models

        Args:
            * model_id (str): Gemini model ID
            * prompt (str): Prompt to send to the model

        Returns:
            * response (str): Model response
        """
        client = self.clients["gemini"]
        response = client.models.generate_content(
            model=model_id, contents=prompt
        )
        return response.text

    def _query_together(self, model_id: str, prompt: str) -> str:
        """
        Query TogetherAI models

        Args:
            * model_id (str): TogetherAI model ID
            * prompt (str): Prompt to send to the model

        Returns:
            * response (str): Model response
        """
        client = self.clients["together"]
        response = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content

    def _query_dummy(self, prompt: str) -> str:
        """
        Query dummy model for testing without API usage

        Args:
            * prompt (str): Prompt to send to the model

        Returns:
            * response (str): Dummy response
        """
        if "probability" in prompt.lower():
            return f"{random.uniform(0.1, 0.9):.2f}"
        elif "top 5 functions" in prompt.lower():
            functions = [
                "sensory processing",
                "motor control",
                "memory formation",
                "emotional regulation",
                "language processing",
                "attention control",
                "decision making",
                "spatial navigation",
                "auditory processing",
                "visual processing",
                "pain perception",
                "reward processing",
                "fear response",
                "learning",
                "cognition",
            ]
            selected = random.sample(functions, 5)
            return f"[{', '.join(selected)}]"
        else:
            return "This is a dummy response for testing purposes"

    def get_embeddings(self, text: str, model: str) -> List[float]:
        """
        Get embeddings for text using OpenAI or return dummy embeddings

        Args:
            * text: Text to embed
            * model: Model name (if "dummy", return random embeddings)

        Returns:
            * Embedding vector as a list of floats
        """

        # Return a random 3073-dim vector (OpenAI's dimension) for dummy
        if model == "dummy":
            return [random.uniform(-1, 1) for _ in range(3073)]

        return self.retry_with_backoff(self._get_openai_embeddings, text)

    def _get_openai_embeddings(self, text: str) -> List[float]:
        """
        Get embeddings from OpenAI

        Args:
            * text: Text to embed

        Returns:
            * Embedding vector as a list of floats
        """
        client = self.clients["openai"]
        response = client.embeddings.create(
            input=text, model="text-embedding-3-large"
        )
        return response.data[0].embedding
