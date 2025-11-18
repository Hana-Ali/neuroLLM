import os
import json
import pandas as pd
from time import time
from typing import Dict, Any

from utils.misc.logging_setup import logger

from utils.paths.query import QueryPathConstructor
from utils.paths.embeddings import EmbeddingsPathConstructor


def _save_json_update(filepath: str, new_data: Dict[str, Any]):
    """
    Safely update a JSON file with new data, using a lock file to prevent
    concurrent write issues

    Args:
        * filepath: Path to JSON file
        * new_data: Dictionary of data to add/update in the JSON file

    Raises:
        * Exception if lock cannot be obtained after 3 attempts
    """

    lock_file = f"{filepath}.lock"

    # Try to get lock (max 3 attempts)
    for attempt in range(3):
        try:
            # Create lock file (fails if already exists)
            with open(lock_file, "x") as lock:
                lock.write(str(os.getpid()))

            try:
                # We have the lock - now update the file
                if os.path.exists(filepath):
                    with open(filepath, "r") as f:
                        data = json.load(f)
                else:
                    data = {}
                data.update(new_data)
                with open(filepath, "w") as f:
                    json.dump(data, f, indent=2)
                break  # Success!

            finally:
                # Always remove lock
                try:
                    os.remove(lock_file)
                except OSError:
                    pass

        except FileExistsError:
            # Someone else has the lock, wait and retry
            if attempt < 2:
                time.sleep(0.1 * (attempt + 1))
            else:
                logger.error_status(
                    f"Could not get lock for {filepath} after 3 attempts",
                    exc_info=True,
                )
                raise


def _save_function_results(
    model: str,
    config: Dict[str, Any],
    region: str,
    response: str,
    embedding: str,
    functions: str,
    hemisphere: str,
    analysis_type: str = "functions",
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
        * analysis_type: Type of analysis ("functions")
    """

    # Get the query region path for raw response
    query_region_path = QueryPathConstructor.construct_query_region_path(
        model=model,
        region=region,
        species=config.species,
        atlas_name=config.atlas_name,
        analysis_type=analysis_type,
        hemisphere=hemisphere,
        template_name=config.prompt_template_name,
    )
    os.makedirs(os.path.dirname(query_region_path), exist_ok=True)
    _save_json_update(filepath=query_region_path, new_data={model: response})

    # Cleaned functions
    clean_reg_path = QueryPathConstructor.construct_query_cleaned_region_path(
        model=model,
        region=region,
        species=config.species,
        atlas_name=config.atlas_name,
        analysis_type=analysis_type,
        hemisphere=hemisphere,
        template_name=config.prompt_template_name,
    )
    os.makedirs(os.path.dirname(clean_reg_path), exist_ok=True)
    _save_json_update(filepath=clean_reg_path, new_data={model: functions})

    # Embedding
    emb_reg_path = EmbeddingsPathConstructor.construct_embeddings_region_path(
        model=model,
        region=region,
        species=config.species,
        atlas_name=config.atlas_name,
        hemisphere=hemisphere,
        template_name=config.prompt_template_name,
        analysis_type=analysis_type,
    )
    os.makedirs(os.path.dirname(emb_reg_path), exist_ok=True)
    df = pd.DataFrame([embedding])  # Single row
    df.columns = [f"dim_{i}" for i in range(len(embedding))]
    df.index = [region]
    df.to_csv(emb_reg_path)


def _save_probability_results(
    region: str,
    hemisphere: str,
    function: str,
    model: str,
    config: Dict[str, Any],
    probability: str,
    analysis_type: str = "probabilities",
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
        * analysis_type: Type of analysis ("probabilities")
    """
    # Get the query region path
    query_region_path = QueryPathConstructor.construct_query_region_path(
        model=model,
        region=region,
        species=config.species,
        atlas_name=config.atlas_name,
        analysis_type=analysis_type,
        hemisphere=hemisphere,
        template_name=config.prompt_template_name,
    )
    os.makedirs(os.path.dirname(query_region_path), exist_ok=True)
    _save_json_update(
        filepath=query_region_path, new_data={function: {model: probability}}
    )
