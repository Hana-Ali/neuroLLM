import os

import requests

from utils.misc.logging_setup import logger
from utils.misc.variables import OPENROUTER_BASE_URL


def _get_costs(model: dict) -> tuple[float, float]:
    """
    Return (prompt_cost_per_m, completion_cost_per_m) for a model Costs are
    converted from per-token to per million tokens

    Args:
        * model (dict): Model dictionary from OpenRouter API

    Returns:
        * tuple[float, float]: (prompt_cost_per_m, completion_cost_per_m)
    """
    pricing = model.get("pricing", {})
    try:
        prompt = float(pricing.get("prompt", "0")) * 1_000_000
    except (ValueError, TypeError):
        prompt = float("inf")
    try:
        completion = float(pricing.get("completion", "0")) * 1_000_000
    except (ValueError, TypeError):
        completion = float("inf")
    return prompt, completion


def _format_cost(value: float) -> str:
    return "free" if value == 0.0 else f"${value:.2f}"


def _print_table(models: list):
    """
    Print the standard model table with columns: Model ID, Name, Context
    Length, Prompt Cost/1M, Completion Cost/1M

    Args:
        * models (list): List of model dictionaries from OpenRouter API
    """
    print(
        f"{'Model ID':<50} {'Name':<40} {'Context':<10} "
        f"{'Prompt $/1M':<14} {'Completion $/1M'}"
    )
    print("-" * 140)
    for model in models:
        model_id = model.get("id", "")
        name = model.get("name", "")[:38]
        context = model.get("context_length", "?")
        prompt, completion = _get_costs(model)
        print(
            f"{model_id:<50} {name:<40} {str(context):<10} "
            f"{_format_cost(prompt):<14} {_format_cost(completion)}"
        )


def list_available_models(filter: str = "all"):
    """
    Fetch and display available models from OpenRouter with pricing

    Args:
        * filter (str): 'all' shows every chat model, 'free' shows only
            zero-cost models, 'paid' shows the 3 cheapest paid models per
            provider sorted by combined prompt + completion cost
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

    # Keep only chat models (output modality must be text)
    models = [
        m for m in models
        if m.get("architecture", {}).get("modality", "").endswith("->text")
    ]

    # Display only free models
    if filter == "free":
        models = [
            m for m in models
            if sum(_get_costs(model=m)) == 0.0
        ]
        models.sort(key=lambda m: m.get("id", ""))
        print(f"\n=== Free models ({len(models)} total) ===\n")
        _print_table(models=models)

    # Display only paid models, showing top 3 cheapest per provider
    elif filter == "paid":
        models = [m for m in models if sum(_get_costs(model=m)) > 0.0]

        # Group by provider (part before the first '/')
        by_provider: dict[str, list] = {}
        for m in models:
            provider = m.get("id", "").split("/")[0]
            by_provider.setdefault(provider, []).append(m)

        # Sort each provider's models by combined cost, keep top 3
        print("\n=== Cheapest 3 paid models per provider ===\n")
        for provider in sorted(by_provider):
            cheapest = sorted(
                by_provider[provider], key=lambda m: sum(_get_costs(model=m))
            )[:3]
            print(f"--- {provider} ---")
            _print_table(models=cheapest)
            print()

    else:  # all
        models.sort(key=lambda m: m.get("id", ""))
        print(f"\n=== All chat models ({len(models)} total) ===\n")
        _print_table(models=models)

    print("\nUsage: --models 'openai/gpt-4o-mini,anthropic/claude-3.5-sonnet'")
