import json
from typing import List, Dict, Tuple, Optional

from utils.misc.logging_setup import logger
from utils.misc.variables import DEFAULT_FUNCTIONS


def save_functions(
    functions: List[str], groups: Optional[Dict[str, List[str]]] = None
) -> None:
    """
    Save functions to a JSON file, with optional function groups.
    If groups is None, preserves existing groups.

    Args:
        functions: List of functions to save
        groups: Optional dictionary mapping group names to lists of functions
    """
    # If groups is None, load existing groups
    if groups is None:
        try:
            _, existing_groups = load_functions()
            groups = existing_groups
        except Exception:
            # If loading fails, use empty groups
            groups = {}

    data = {"functions": functions, "groups": groups}

    with open("functions.json", "w") as f:
        json.dump(data, f, indent=2)

    logger.processing(
        f"Saved {len(functions)} functions and {len(data['groups'])} "
        "groups to functions.json"
    )


def load_functions() -> Tuple[List[str], Dict[str, List[str]]]:
    """
    Load functions and function groups from functions.json

    Returns:
        Tuple containing (list of functions, dictionary of function groups)
    """
    try:
        with open("functions.json", "r") as f:
            data = json.load(f)

        functions = data.get("functions", [])
        groups = data.get("groups", {})

        logger.processing(
            f"Loaded {len(functions)} functions and {len(groups)} "
            "groups from functions.json"
        )
        return functions, groups
    except Exception as e:
        logger.warning_status(
            f"Error loading functions.json: {str(e)}. Using default functions"
        )
        default_groups = {}
        save_functions(DEFAULT_FUNCTIONS, default_groups)
        return DEFAULT_FUNCTIONS, default_groups


def load_function_group(group_name: str) -> List[str]:
    """
    Load functions from a specific group

    Args:
        group_name: Name of function group to load

    Returns:
        List of functions from the group or empty list if group not found
    """
    _, groups = load_functions()
    if group_name in groups:
        return groups[group_name]
    return None
