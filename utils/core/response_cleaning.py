import re
from typing import List, Tuple


def _strip_thinking_tags(response: str) -> str:
    """
    Remove <think>...</think> tags from LLM responses

    Args:
        * response: Raw response string from the model

    Returns:
        * Cleaned response string without <think> tags
    """
    return re.sub(
        r"<think>.*?</think>", "", response, flags=re.DOTALL
    ).strip()


def clean_functions_response(response: str) -> List[str]:
    """
    Clean an LLM response to extract the list of top 5 functions

    Args:
        * response: Raw response string from the model

    Returns:
        * List of up to 5 function names
    """
    response = _strip_thinking_tags(response=response)

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
    response = _strip_thinking_tags(response=response)

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


def split_justified_response(response: str) -> Tuple[str, str]:
    """
    Split a justified LLM response into the answer and justification
    parts using " | " as the separator

    Args:
        * response: Raw response string from the model

    Returns:
        * Tuple of (answer_part, justification)
    """
    cleaned = _strip_thinking_tags(response=response)

    if " | " in cleaned:
        parts = cleaned.split(" | ", 1)
        return parts[0].strip(), parts[1].strip()
    return cleaned, ""


def clean_ranking_response(response: str) -> int:
    """
    Extract a ranking (1 or 2) from the LLM response

    Args:
        * response: Raw response string from the model

    Returns:
        * Ranking as int (1 or 2), or 0 if parsing fails
    """
    response = _strip_thinking_tags(response=response)

    # Find the first 1 or 2
    match = re.search(r"\b([12])\b", response)
    if match:
        return int(match.group(1))

    return 0
