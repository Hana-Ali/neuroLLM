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


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

BRAINGPT_CONFIG = {
    "base_model_id": "meta-llama/Llama-2-7b-chat-hf",
    "adapter_id": "BrainGPT/BrainGPT-7B-v0.1",
}

LOCAL_MODELS = {"braingpt", "dummy"}

EMBEDDING_DIMS = {"openai": 3072, "local": 1024}

DEFAULT_FUNCTIONS = [
    "spatial cognition",
    "rationality",
    "creativity",
    "metacognition",
    "consciousness",
    "anaesthesia",
    "coma",
]
