import os
import time
import random
from dotenv import load_dotenv

from typing import Dict, List, Callable

import openai
import anthropic
from google import genai
from together import Together

from utils.misc.logging_setup import logger

from utils.paths.base import DEFAULT_PATHS
from utils.misc.variables import MODEL_CONFIGS


class APIClientManager:
    """Manages multiple API clients for different LLM providers"""

    def __init__(self):
        self.clients: Dict[str, Callable] = {}
        self.client_names: List[str] = []
        self.free_models: List[str] = []
        self.paid_models: List[str] = []
        self.all_models: List[str] = []
        self.initialized = False

    def load_api_keys(self) -> Dict[str, str]:
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

        # Get the API keys from environment variables
        api_keys = {
            "openai": os.environ.get("OPENAI_API_KEY"),
            "claude": os.environ.get("CLAUDE_API_KEY"),
            "gemini": os.environ.get("GEMINI_API_KEY"),
            "together": os.environ.get("TOGETHERAI_API_KEY"),
        }

        # Log which keys were found
        for provider, key in api_keys.items():
            if key:
                logger.info(f"Found API key for {provider}")
            else:
                logger.error(f"No API key found for {provider}")
                raise

        return api_keys

    def init_clients(self) -> bool:
        """
        Initialize API clients for all providers

        Raises:
            * Exception if any client fails to initialize
        """

        # Load API keys
        api_keys = self.load_api_keys()
        if not api_keys:
            logger.error("No API keys found. Cannot initialize clients")
            raise

        try:
            # OpenAI
            openai.api_key = api_keys["openai"]
            self.clients["openai"] = openai
            logger.info("Initialized openai client")

            # Claude
            client = anthropic.Anthropic(api_key=api_keys["claude"])
            self.clients["claude"] = client
            logger.info("Initialized claude client")

            # Gemini
            self.clients["gemini"] = genai.Client(api_key=api_keys["gemini"])
            logger.info("Initialized gemini client")

            # Together
            self.clients["together"] = Together(api_key=api_keys["together"])
            logger.info("Initialized together client")

        except Exception as e:
            logger.error_status(
                f"Failed to initialize clients: {str(e)}", exc_info=True
            )
            raise

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
        retries = 0
        while True:
            try:
                return func(*args)
            except Exception as e:
                retries += 1
                if retries > max_retries:
                    logger.error_status(
                        (
                            f"Maximum retries ({max_retries}) exceeded. "
                            f"Last error: {str(e)}"
                        ),
                        exc_info=True,
                    )
                    raise

                # Add jitter to avoid thundering herd
                delay = initial_delay * (2 ** (retries - 1)) + random.uniform(
                    0, 1
                )
                logger.warning_status(
                    f"API error: {str(e)}. Retrying in {delay:.1f}s "
                    f"(attempt {retries}/{max_retries})..."
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

        if provider == "dummy":
            return self._query_dummy(prompt)
        elif provider == "openai":
            return self.retry_with_backoff(
                self._query_openai, model_id, prompt
            )
        elif provider == "claude":
            return self.retry_with_backoff(
                self._query_claude, model_id, prompt
            )
        elif provider == "gemini":
            return self.retry_with_backoff(
                self._query_gemini, model_id, prompt
            )
        elif provider == "together":
            return self.retry_with_backoff(
                self._query_together,
                model_id,
                prompt,
                max_retries=7,
                initial_delay=5,
            )
        else:
            raise ValueError(f"Unknown provider: {provider}")

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
            return "This is a dummy response for testing purposes."

    def get_embeddings(self, text: str, model: str) -> List[float]:
        """
        Get embeddings for text using OpenAI or return dummy embeddings

        Args:
            * text: Text to embed
            * model: Model name (if "dummy", return random embeddings)

        Returns:
            * Embedding vector as a list of floats
        """
        # Return a random 3073-dimensional vector (OpenAI's dimension) for
        # dummy model, else use OpenAI embeddings
        return (
            [random.uniform(-1, 1) for _ in range(3073)]
            if model == "dummy"
            else self.retry_with_backoff(self._get_openai_embeddings, text)
        )

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

    @classmethod
    def get_models_by_category(self, category: str = "all") -> List[str]:
        """
        Get model names by category

        Args:
            * category (str): Category of models to retrieve
                ('all', 'all-excl-dummy', 'paid', 'free', 'dummy', or
                    comma-separated list of model names)

        Returns:
            * models (list): List of model names
        """

        if category == "all":
            return list(MODEL_CONFIGS.keys())
        elif category == "all-excl-dummy":
            return [
                name
                for name, config in MODEL_CONFIGS.items()
                if config["category"] != "dummy"
            ]
        elif category in ["paid", "free", "dummy"]:
            return [
                name
                for name, config in MODEL_CONFIGS.items()
                if config["category"] == category
            ]
        else:
            # Comma-separated list of model names
            return [model.strip() for model in category.split(",")]
