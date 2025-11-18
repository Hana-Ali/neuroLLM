DEFAULT_FUNCTION_TEMPLATE = """
    You are an expert in neuroscience literature analysis. Your task is to
    list out the top 5 functions that region region **{region}**
    {hemisphere_part} **{species}** brain is involved in.

    These functions should be based on a simulated review of neuroscience
    literature, reflecting how frequently these functions are associated with
    this brain region across **peer-reviewed** studies, textbooks, and
    reputable sources.

    ### Guidelines
    1. Consider only **{species}**-specific neuroscience literature. Do **not**
    use data from other species.
    2. **DO NOT** provide explanations, citations, or any extra text—**return
    only the function names**.
    3. Return your results as a list: [function_1, function_2, function_3,
    function_4, function_5]
    4. **DO NOT** repeat functions in your list.
    5. List out the functions **only** for the specified {hemisphere_part}
    hemisphere.

    ### Expected Output Format
    [function_1, function_2, function_3, function_4, function_5]
    """

DEFAULT_PROBABILITY_TEMPLATE = """
    You are an expert in neuroscience literature analysis. Your task is to
    estimate the probability that the brain region region **{region}**
    {hemisphere_part} **{species}** brain is involved in **
    {function}**.

    These functions should be based on a simulated review of neuroscience
    literature, reflecting how frequently these functions are associated with
    this brain region across **peer-reviewed** studies, textbooks, and
    reputable sources.

    ### Guidelines
    1. Consider only **{species}**-specific neuroscience literature. Do
    **not** use data from other species.
    2. The probability should be a **single decimal number** between **0 and
    1**.
    3. This number should **approximate the relative frequency** with which
    this function is linked to the given brain region in literature.
    4. **DO NOT** provide explanations, citations, or any extra text—**return
    only the probability value**.

    ### Expected Output Format
    0.XX
    """


DEFAULT_TEMPLATES = {
    "functions": DEFAULT_FUNCTION_TEMPLATE,
    "probabilities": DEFAULT_PROBABILITY_TEMPLATE,
}


MODEL_CONFIGS = {
    "openai": {
        "provider": "openai",
        "model_id": "gpt-4o-mini",
        "category": "paid",
        "rate_limits": {"calls_per_minute": 20, "concurrent": 5},
    },
    "claude": {
        "provider": "claude",
        "model_id": "claude-3-7-sonnet-latest",
        "category": "paid",
        "rate_limits": {"calls_per_minute": 15, "concurrent": 3},
    },
    "gemini": {
        "provider": "gemini",
        "model_id": "gemini-2.0-flash",
        "category": "paid",
        "rate_limits": {"calls_per_minute": 20, "concurrent": 5},
    },
    "qwen": {
        "provider": "together",
        "model_id": "Qwen/Qwen2.5-7B-Instruct-Turbo",
        "category": "paid",
        "rate_limits": {"calls_per_minute": 10, "concurrent": 2},
    },
    "mistral": {
        "provider": "together",
        "model_id": "mistralai/Mistral-7B-Instruct-v0.3",
        "category": "paid",
        "rate_limits": {"calls_per_minute": 10, "concurrent": 2},
    },
    "llama": {
        "provider": "together",
        "model_id": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "category": "free",
        "rate_limits": {"calls_per_minute": 10, "concurrent": 2},
    },
    "deepseek": {
        "provider": "together",
        "model_id": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        "category": "free",
        "rate_limits": {"calls_per_minute": 10, "concurrent": 2},
    },
    "dummy": {
        "provider": "dummy",
        "model_id": "dummy-model-for-testing",
        "category": "dummy",
        "rate_limits": {"calls_per_minute": 1000, "concurrent": 100},
    },
}

DEFAULT_FUNCTIONS = [
    "spatial cognition",
    "rationality",
    "creativity",
    "metacognition",
    "consciousness",
    "anaesthesia",
    "coma",
]
