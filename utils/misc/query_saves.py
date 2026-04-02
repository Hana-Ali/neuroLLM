import os
import json
import pandas as pd
import time
from typing import Dict, Any, Callable

from utils.misc.logging_setup import logger

from utils.paths.query import QueryPathConstructor
from utils.paths.embeddings import EmbeddingsPathConstructor


def _locked_json_write(filepath: str, update_fn: Callable):
    """
    Safely read-modify-write a JSON file under a lock

    Args:
        * filepath: Path to JSON file
        * update_fn: callable(existing_data) -> updated_data

    Raises:
        * Exception if lock cannot be obtained after 3 attempts
    """

    lock_file = f"{filepath}.lock"

    # Acquire lock (max 3 attempts)
    for attempt in range(3):
        try:
            # Create lock file (fails if already exists)
            with open(lock_file, "x") as lock:
                lock.write(str(os.getpid()))
            break
        except FileExistsError:
            if attempt < 2:
                time.sleep(0.1 * (attempt + 1))
            else:
                logger.error_status(
                    f"Could not get lock for {filepath} "
                    "after 3 attempts",
                    exc_info=True,
                )
                raise

    # Read-modify-write under lock
    try:
        data = {}
        if os.path.exists(filepath):
            try:
                with open(filepath, "r") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError):
                data = {}
        data = update_fn(data)
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
    finally:
        try:
            os.remove(lock_file)
        except OSError:
            pass


def _save_json_update(filepath: str, new_data: Dict[str, Any]):
    """
    Safely update a JSON file with new data (shallow merge)

    Args:
        * filepath: Path to JSON file
        * new_data: Dictionary of data to add/update
    """
    def _update(data):
        data.update(new_data)
        return data
    _locked_json_write(filepath=filepath, update_fn=_update)


def _save_json_deep_merge(
    filepath: str, model: str, new_data: Dict[str, Any],
):
    """
    Deep-merge new_data into the model's entry in a JSON file.
    For dict-valued keys, merges rather than replaces.

    Args:
        * filepath: Path to JSON file
        * model: Model key to merge under
        * new_data: Data to merge into data[model]
    """
    def _merge(data):
        if model not in data:
            data[model] = {}
        for key, value in new_data.items():
            if (
                key in data[model]
                and isinstance(data[model][key], dict)
                and isinstance(value, dict)
            ):
                data[model][key].update(value)
            else:
                data[model][key] = value
        return data
    _locked_json_write(filepath=filepath, update_fn=_merge)


def _save_function_results(
    model: str,
    config: Dict[str, Any],
    region: str,
    response: str,
    embedding: list,
    functions: list,
    hemisphere: str,
    trial="final",
    analysis_type: str = "functions",
    justification: str = None,
):
    """
    Save all function analysis results

    Args:
        * model: Model name
        * config: Configuration object
        * region: Brain region name
        * response: Raw model response
        * embedding: Embedding vector
        * functions: Cleaned list of functions
        * hemisphere: Hemisphere string for directory naming
        * trial: int for a specific trial, or "final"
        * analysis_type: Type of analysis ("functions")
        * justification: Justification text (None = not justified)
    """
    query = QueryPathConstructor(
        model=model, species=config.species,
        atlas_name=config.atlas_name,
        analysis_type=analysis_type, hemisphere=hemisphere,
        template_name=config.prompt_template_name,
    )
    emb = EmbeddingsPathConstructor(
        model=model, species=config.species,
        atlas_name=config.atlas_name,
        analysis_type=analysis_type, hemisphere=hemisphere,
        template_name=config.prompt_template_name,
    )

    # # Raw response
    # query_path = query.construct_query_region_path(
    #     region=region, trial=trial,
    # )
    # os.makedirs(os.path.dirname(query_path), exist_ok=True)
    # _save_json_update(
    #     filepath=query_path, new_data={model: response},
    # )

    # Cleaned functions
    clean_path = query.construct_query_cleaned_region_path(
        region=region, trial=trial,
    )
    os.makedirs(os.path.dirname(clean_path), exist_ok=True)
    _save_json_update(
        filepath=clean_path, new_data={model: functions},
    )

    # Embedding
    emb_path = emb.construct_embeddings_region_path(
        region=region, trial=trial,
    )
    os.makedirs(os.path.dirname(emb_path), exist_ok=True)
    df = pd.DataFrame([embedding])
    df.columns = [f"dim_{i}" for i in range(len(embedding))]
    df.index = [region]
    df.to_csv(emb_path)

    # Justification
    if justification:
        just_path = query.construct_query_justification_region_path(
            region=region, trial=trial,
        )
        os.makedirs(os.path.dirname(just_path), exist_ok=True)
        _save_json_update(
            filepath=just_path, new_data={model: justification},
        )


def _save_probability_results(
    region: str,
    hemisphere: str,
    function: str,
    model: str,
    config: Dict[str, Any],
    probability: float,
    trial="final",
    analysis_type: str = "probabilities",
    justification: str = None,
):
    """
    Save probability analysis results

    Args:
        * region: Brain region name
        * hemisphere: Hemisphere string for directory naming
        * function: Brain function queried
        * model: Model name
        * config: Configuration object
        * probability: Cleaned probability result
        * trial: int for a specific trial, or "final"
        * analysis_type: Type of analysis ("probabilities")
        * justification: Justification text (None = not justified)
    """
    query = QueryPathConstructor(
        model=model, species=config.species,
        atlas_name=config.atlas_name,
        analysis_type=analysis_type, hemisphere=hemisphere,
        template_name=config.prompt_template_name,
    )

    query_path = query.construct_query_region_path(
        region=region, trial=trial,
    )
    os.makedirs(os.path.dirname(query_path), exist_ok=True)
    _save_json_update(
        filepath=query_path,
        new_data={function: {model: probability}},
    )

    if justification:
        just_path = query.construct_query_justification_region_path(
            region=region, trial=trial,
        )
        os.makedirs(os.path.dirname(just_path), exist_ok=True)
        _save_json_update(
            filepath=just_path,
            new_data={function: {model: justification}},
        )


def _save_ranking_results(
    region_1: str,
    region_2: str,
    hemisphere: str,
    function: str,
    model: str,
    config: Dict[str, Any],
    ranking: int,
    trial="final",
    analysis_type: str = "rankings",
    justification: str = None,
):
    """
    Save ranking analysis results

    Args:
        * region_1: First brain region name
        * region_2: Second brain region name
        * hemisphere: Hemisphere string for directory naming
        * function: Brain function queried
        * model: Model name
        * config: Configuration object
        * ranking: Ranking result (1 or 2)
        * trial: int for a specific trial, or "final"
        * analysis_type: Type of analysis ("rankings")
        * justification: Justification text (None = not justified)
    """
    query = QueryPathConstructor(
        model=model, species=config.species,
        atlas_name=config.atlas_name,
        analysis_type=analysis_type, hemisphere=hemisphere,
        template_name=config.prompt_template_name,
    )

    path = query.construct_query_pair_path(
        region_1=region_1, region_2=region_2, trial=trial,
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    _save_json_update(
        filepath=path,
        new_data={function: {model: ranking}},
    )

    if justification:
        just_path = query.construct_query_pair_justification_path(
            region_1=region_1, region_2=region_2, trial=trial,
        )
        os.makedirs(os.path.dirname(just_path), exist_ok=True)
        _save_json_update(
            filepath=just_path,
            new_data={function: {model: justification}},
        )


def _save_retest_summary(
    region: str,
    model: str,
    config: Dict[str, Any],
    analysis_type: str,
    hemisphere: str,
    summary_data: Dict[str, Any],
):
    """
    Save retest summary data (only for retest > 1)

    Args:
        * region: Brain region name (or "r1_vs_r2" for rankings)
        * model: Model name
        * config: Configuration object
        * analysis_type: Type of analysis
        * hemisphere: Hemisphere string
        * summary_data: Summary data dict to save
    """
    query = QueryPathConstructor(
        model=model, species=config.species,
        atlas_name=config.atlas_name,
        analysis_type=analysis_type, hemisphere=hemisphere,
        template_name=config.prompt_template_name,
    )
    path = query.construct_query_retest_summary_path(
        region=region,
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    _save_json_deep_merge(
        filepath=path, model=model, new_data=summary_data,
    )
