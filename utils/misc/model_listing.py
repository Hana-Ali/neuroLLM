import os

import requests

from utils.misc.logging_setup import logger
from utils.misc.variables import OPENROUTER_BASE_URL


def list_available_models():
    """
    Fetch and display available models from OpenRouter with pricing
    """

    # Get API key from environment
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        logger.error(
            "OPENROUTER_API_KEY not found in .env file. "
            "Get one at https://openrouter.ai/keys"
        )
        return

    # Fetch models from OpenRouter
    response = requests.get(
        f"{OPENROUTER_BASE_URL}/models",
        headers={"Authorization": f"Bearer {api_key}"},
    )

    # Check for errors, then parse models
    response.raise_for_status()
    models = response.json().get("data", [])

    # Sort by model ID (groups by provider)
    models.sort(key=lambda m: m.get("id", ""))

    # Print header
    print(
        f"{'Model ID':<50} {'Name':<40} {'Context':<10} "
        f"{'Prompt $/1M':<14} {'Completion $/1M'}"
    )
    print("-" * 140)

    # Iterate through models
    for model in models:

        # Get relevant info with defaults
        model_id = model.get("id", "")
        name = model.get("name", "")[:38]
        context = model.get("context_length", "?")
        pricing = model.get("pricing", {})

        # Get prompt and completion costs, defaulting to "0" if not available
        prompt_cost = pricing.get("prompt", "0")
        completion_cost = pricing.get("completion", "0")

        # Convert per-token to per-million-tokens
        try:
            prompt_per_m = f"${float(prompt_cost) * 1_000_000:.2f}"
        except (ValueError, TypeError):
            prompt_per_m = "N/A"
        try:
            completion_per_m = f"${float(completion_cost) * 1_000_000:.2f}"
        except (ValueError, TypeError):
            completion_per_m = "N/A"

        print(
            f"{model_id:<50} {name:<40} {str(context):<10} "
            f"{prompt_per_m:<14} {completion_per_m}"
        )

    print(f"\nTotal: {len(models)} models available")
    print(
        "\nUsage: --models 'openai/gpt-4o-mini,anthropic/claude-3.5-sonnet'"
    )
