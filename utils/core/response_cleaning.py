import re
from typing import List


def clean_functions_response(response: str) -> List[str]:
    """
    Clean an LLM response to extract the list of top 5 functions

    Args:
        * response: Raw response string from the model

    Returns:
        * List of up to 5 function names
    """
    # Remove thinking tags
    response = re.sub(
        r"<think>.*?</think>", "", response, flags=re.DOTALL
    ).strip()

    # Remove introductory text like "Region X is involved in:"
    response = re.sub(
        pattern=(
            r"^\s*.*?(is involved in|is primarily involved in|"
            r"here are the top 5 main functions associated with these regions|"
            r"functions of).*?:\s*"
        ),
        repl="",
        string=response,
        flags=re.IGNORECASE | re.DOTALL,
    )

    # Extract from brackets if present
    bracket_match = re.search(r"\[(.*?)\]", response)
    if bracket_match:
        response = bracket_match.group(1)

    # Remove numbering and function labels
    response = re.sub(r"(?:\d+\.\s*|Function\s*\d+:\s*)", "", response)

    # Split by commas and clean
    functions = [
        f.strip().strip("\"'")
        for f in response.split(",")
        if f.strip()
        and f.strip().lower() not in {"unknown", "unclear", "n/a", ""}
    ]

    # Assert we have at least 5 functions
    assert len(functions) >= 5, "Less than 5 functions found"

    # Return up to 5 functions
    return functions[:5]


def clean_probability_response(response: str) -> float:
    """
    Extract a decimal probability from the LLM response, supporting
    negative values

    Args:
        * response: Raw response string from the model

    Returns:
        * Probability as float in [0, 1]
    """

    # Remove thinking tags
    response = re.sub(
        r"<think>.*?</think>", "", response, flags=re.DOTALL
    ).strip()

    # Find all numbers (including negative decimals)
    numbers = re.findall(r"-?\d*\.?\d+", response)

    for num_str in numbers:
        try:
            value = float(num_str)
            # Return first value in valid probability range
            if -1.0 <= value <= 1.0:
                return value
        except ValueError:
            continue

    # If no valid probability found, return default
    return 0.0
