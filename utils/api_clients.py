import os
import time
import torch
import random

from typing import Dict, List, Callable, Any

import openai

from huggingface_hub import model_info
from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError
from peft import PeftModel
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.misc.logging_setup import logger
from utils.misc.variables import (
    OPENROUTER_BASE_URL,
    BRAINGPT_CONFIG,
    LOCAL_MODELS,
    EMBEDDING_DIMS,
)


class APIClientManager:
    """Manages API clients for OpenRouter, BrainGPT, and Dummy providers"""

    def __init__(
        self,
        models: str = "dummy",
        embedding_provider: str = None,
        max_tokens: int = 256,
    ):
        """
        Initialize API clients for all needed providers

        Args:
            * models (str): Comma-separated list of OpenRouter model IDs,
                'braingpt', or 'dummy' (default: 'dummy')
            * embedding_provider (str | None): 'openai' or 'local', or None to
                skip embedding init (only needed for top-functions)
                'openai' uses text-embedding-3-large (requires OPENAI_API_KEY)
                'local' uses BAAI/bge-large-en-v1.5 via sentence-transformers
            * max_tokens (int): Maximum tokens to generate per response
                (default: 256)

        Raises:
            * Exception if any client fails to initialize
        """
        self.clients: Dict[str, Any] = {}
        self.model_names: List[str] = [m.strip() for m in models.split(",")]
        self.embedding_provider = embedding_provider
        self.max_tokens = max_tokens

        try:
            if any(m not in LOCAL_MODELS for m in self.model_names):
                self._init_openrouter()

            if embedding_provider == "openai":
                self._init_openai_embeddings()
            elif embedding_provider == "local":
                self._init_local_embeddings()

            if "braingpt" in self.model_names:
                self._validate_braingpt_access()
                self._init_braingpt()

        except Exception as e:
            logger.error_status(
                f"Failed to initialize clients: {str(e)}", exc_info=True
            )
            raise

    def query_model(self, model_name: str, prompt: str) -> str:
        """
        Query a model by name

        Args:
            * model_name (str): OpenRouter model ID, 'braingpt', or 'dummy'
            * prompt (str): Prompt to send to the model

        Returns:
            * response (str): Model response
        """
        try:
            if model_name == "dummy":
                return self._query_dummy(prompt=prompt)
            elif model_name == "braingpt":
                return self._query_braingpt(prompt=prompt)
            else:
                return self.retry_with_backoff(
                    self._query_openrouter, model_name, prompt
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

    # -------------------------------------------------------------------------
    # Initialisation helpers
    # -------------------------------------------------------------------------

    def _init_openrouter(self):
        """
        Validate OPENROUTER_API_KEY and create the OpenRouter client. Uses the
        OpenAI SDK with a base_url override
        """
        openrouter_key = os.environ.get("OPENROUTER_API_KEY")
        if not openrouter_key:
            raise ValueError(
                "Missing OPENROUTER_API_KEY in .env file. "
                "Get one at https://openrouter.ai/keys"
            )
        self.clients["openrouter"] = openai.OpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=openrouter_key,
        )
        logger.info("Initialized OpenRouter client")

    def _init_openai_embeddings(self):
        """
        Validate OPENAI_API_KEY and create the OpenAI embeddings client.

        Note: embeddings cannot be routed through OpenRouter - a direct
        OpenAI key is required (only used for top-functions command, and can be
        bypassed with --embedding-provider local)
        """
        openai_key = os.environ.get("OPENAI_API_KEY")
        if not openai_key:
            raise ValueError(
                "Missing OPENAI_API_KEY in .env file. "
                "This is separate from your OpenRouter key and is only needed "
                "for top-functions (embeddings). "
                "Get one at https://platform.openai.com/account/api-keys. "
                "Alternatively, use --embedding-provider local."
            )
        self.clients["openai_embeddings"] = openai.OpenAI(api_key=openai_key)
        logger.info("Initialized OpenAI embeddings client")

    def _init_local_embeddings(self):
        """
        Load BAAI/bge-large-en-v1.5 via sentence-transformers.
        Weights are downloaded on first use and cached locally in
        ~/.cache/huggingface/hub/ by the huggingface_hub library
        """
        self.clients["local_embeddings"] = SentenceTransformer(
            "BAAI/bge-large-en-v1.5"
        )
        logger.info(
            "Initialized local embeddings model (BAAI/bge-large-en-v1.5)"
        )

    def _validate_braingpt_access(self):
        """
        Validate HF_TOKEN exists and has access to the gated BrainGPT base
        model on Hugging Face. Must be called before _init_braingpt()
        """
        # Get HF token
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            raise ValueError(
                "BrainGPT requires HF_TOKEN in .env file. "
                "Get one at https://huggingface.co/settings/tokens"
            )
        # Check access to base model by fetching model info (fast-ish)
        try:
            model_info(BRAINGPT_CONFIG["base_model_id"], token=hf_token)
        except GatedRepoError:
            raise ValueError(
                f"HF_TOKEN does not have access to "
                f"{BRAINGPT_CONFIG['base_model_id']}. Request access at "
                "https://huggingface.co/meta-llama/Llama-2-7b-chat-hf"
            )
        except RepositoryNotFoundError:
            raise ValueError(
                f"Model {BRAINGPT_CONFIG['base_model_id']} not found on "
                "Hugging Face. Check BRAINGPT_CONFIG in variables.py"
            )
        logger.info("BrainGPT: HF access verified")

    def _init_braingpt(self):
        """
        Load BrainGPT model and tokenizer (Llama-2 base + LoRA adapter)
        Stores them in self.clients["braingpt"] as a dict
        """
        # Get HF token, as well as model IDs from config
        hf_token = os.environ.get("HF_TOKEN")
        base_model_id = BRAINGPT_CONFIG["base_model_id"]
        adapter_id = BRAINGPT_CONFIG["adapter_id"]

        # Load base model, then apply LoRA adapter. Use 16-bit precision and
        # auto device mapping for efficiency
        model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            token=hf_token,
        )
        model = PeftModel.from_pretrained(model, adapter_id)
        model.eval()

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_id, token=hf_token
        )

        # Store in clients dict
        self.clients["braingpt"] = {
            "model": model,
            "tokenizer": tokenizer,
        }
        logger.info("Initialized BrainGPT model")

    # -------------------------------------------------------------------------
    # Query helpers
    # -------------------------------------------------------------------------

    def _query_openrouter(self, model_id: str, prompt: str) -> str:
        """
        Query any model via OpenRouter

        Args:
            * model_id (str): OpenRouter model ID (e.g., 'openai/gpt-4o-mini')
            * prompt (str): Prompt to send to the model

        Returns:
            * response (str): Model response
        """
        client = self.clients["openrouter"]
        response = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            seed=42,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content.strip()

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

    def _query_braingpt(self, prompt: str) -> str:
        """
        Query BrainGPT (Llama-2 + LoRA adapter) locally

        Args:
            * prompt (str): Prompt to send to the model

        Returns:
            * response (str): Model response
        """

        # Get model and tokenizer from clients dict
        model = self.clients["braingpt"]["model"]
        tokenizer = self.clients["braingpt"]["tokenizer"]

        # Tokenize prompt and move to model device
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # Generate responses
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                do_sample=False,
            )

        # Decode only the newly generated tokens (skip the prompt tokens)
        token_start = inputs["input_ids"].shape[1]
        response = tokenizer.decode(
            outputs[0][token_start:],
            skip_special_tokens=True,
        )
        return response.strip()

    # -------------------------------------------------------------------------
    # Embedding helpers
    # -------------------------------------------------------------------------

    def get_embeddings(self, text: str, model: str) -> List[float]:
        """
        Get embeddings for text using the configured embedding provider,
        or return dummy embeddings in test mode.

        Args:
            * text: Text to embed
            * model: Model name (if "dummy", return random embeddings)

        Returns:
            * Embedding vector as a list of floats
        """

        # Return random embeddings for dummy model
        if model == "dummy":
            return [
                random.uniform(-1, 1)
                for _ in range(EMBEDDING_DIMS[self.embedding_provider])
            ]

        # Get embeddings from the local provider
        if self.embedding_provider == "local":
            return self._get_local_embeddings(text=text)

        # Get embeddings from OpenAI with retry logic
        return self.retry_with_backoff(self._get_openai_embeddings, text)

    def _get_local_embeddings(self, text: str) -> List[float]:
        """
        Get embeddings from the local BAAI/bge-large-en-v1.5 model

        Args:
            * text: Text to embed

        Returns:
            * Embedding vector as a list of floats
        """
        return self.clients["local_embeddings"].encode(
            text, normalize_embeddings=True
        ).tolist()

    def _get_openai_embeddings(self, text: str) -> List[float]:
        """
        Get embeddings from OpenAI's text-embedding-3-large model

        Args:
            * text: Text to embed

        Returns:
            * Embedding vector as a list of floats
        """
        client = self.clients["openai_embeddings"]
        response = client.embeddings.create(
            input=text, model="text-embedding-3-large"
        )
        return response.data[0].embedding
