import os
import json

from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Tuple, Union

from utils.misc.logging_setup import logger
from utils.prompts import generate_prompt
from utils.core.response_cleaning import split_justified_response


def json_file_has_key(path: str, key: str) -> bool:
    """
    Check if a JSON file exists and contains a key

    Args:
        * path: Path to the JSON file
        * key: Key to check for in the JSON data

    Returns:
        * True if file exists and contains the key, False otherwise
    """
    if not os.path.exists(path):
        return False
    try:
        with open(path, "r") as f:
            data = json.load(f)
        return key in data
    except (json.JSONDecodeError, OSError):
        return False


def split_if_justified(
    config: SimpleNamespace, response: str,
) -> Tuple[str, str]:
    """
    Split response into answer + justification if needed

    Args:
        * config: Config object with 'justify' attribute
        * response: Raw model response

    Returns:
        * Tuple of (answer, justification). Justification is None if not
            justified
    """
    return (
        split_justified_response(response=response)
        if config.justify
        else (response, None)
    )


def process_task(
    config: SimpleNamespace,
    *,
    model: str,
    skip_check: Callable[[], bool],
    trial_complete: Callable[[Union[int, str]], bool],
    load_trial: Callable[[Union[int, str]], Tuple[Dict[str, Any], Any]],
    prompt_kwargs: Dict[str, Any],
    log_label: str,
    process_response: Callable[[str, str], Dict[str, Any]],
    save_result: Callable[..., None],
    save_final: Callable[[List[Dict[str, Any]], List[str]], None],
) -> None:
    """
    Generic task processor. Skips existing trials, queries only missing ones,
    then always recomputes final/summary results from all trials to ensure
    consistency

    Args:
        * config: Analysis configuration
        * model: Model name
        * skip_check: () -> bool, return True to skip entirely
        * trial_complete: (trial) -> bool, per-trial check
        * load_trial: (trial) -> (result_dict, justification)
        * prompt_kwargs: extra kwargs for generate_prompt
        * log_label: string for log messages
        * process_response: (answer, response) -> dict
        * save_result: (result, trial, justification) -> None
        * save_final: (trial_results, justifications) -> None
    """
    if skip_check():
        return

    prompt = None
    trial_results = []
    trial_justifications = []

    # Loop over trials, skipping existing ones and loading results if present
    for trial in range(config.retest):

        # Skip existing trials
        if trial_complete(trial):
            logger.info(
                f"Loading existing {log_label}"
                + (
                    f" trial {trial}"
                    if config.retest > 1
                    else ""
                )
            )
            # Load result and justification (if applicable) for this trial
            result, justification = load_trial(trial)
            trial_results.append(result)
            if justification:
                trial_justifications.append(
                    justification
                )
            continue

        # Generate prompt lazily
        if prompt is None:
            prompt = generate_prompt(
                species=config.species,
                atlas_name=config.atlas_name,
                template_name=config.prompt_template_name,
                justify=config.justify,
                save_to_results=True,
                **prompt_kwargs,
            )

        logger.processing(
            f"Querying {log_label}"
            + (
                f" trial {trial}"
                if config.retest > 1
                else ""
            )
        )

        # Query model and split justification if needed
        response = config.client_manager.query_model(
            model_name=model, prompt=prompt,
            temperature=config.temperature,
        )
        answer, justification = (
            split_if_justified(config=config, response=response)
        )

        # Process response and append justification if applicable
        result = process_response(answer, response)
        trial_results.append(result)
        if justification:
            trial_justifications.append(justification)

        # Save trial result
        save_result(
            result, trial=trial,
            justification=justification,
        )

    # After all trials, recompute final results from all trial results to
    # ensure consistency (e.g., averaging probabilities, etc)
    save_final(trial_results, trial_justifications)
