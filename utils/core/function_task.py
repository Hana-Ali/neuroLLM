import os
import json
import numpy as np
import pandas as pd

from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple, Union

from utils.misc.logging_setup import logger
from utils.misc.query_saves import (
    _save_function_results,
    _save_retest_summary,
)
from utils.core.response_cleaning import clean_functions_response
from utils.core.task_processor import process_task, json_file_has_key

from utils.paths.query import QueryPathConstructor
from utils.paths.embeddings import EmbeddingsPathConstructor

from utils.core.retest_averaging import average_function_trials


def run_function_task(
    config: SimpleNamespace,
    region: str,
    hemisphere: Optional[str],
    model: str,
) -> None:
    """
    Function analysis task

    Args:
        * config: Config object with necessary attributes
        * region: Brain region name
        * hemisphere: Hemisphere ('left', 'right', or None')
        * model: Model name to use for querying
    """
    analysis_type = "top-functions"

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

    def _trial_done(trial: Union[int, str]) -> bool:
        """
        Check if a single trial is complete by verifying the presence of
        results in the JSON file

        Args:
            * trial: Trial number (int) or 'final' for final averaged result

        Returns:
            * True if the trial is complete, False otherwise
        """
        # Check cleaned response JSON contains the model key
        clean = query.construct_query_cleaned_region_path(
            region=region, trial=trial,
        )
        if not json_file_has_key(path=clean, key=model):
            return False

        # Check for embedding CSV file
        emb_path = emb.construct_embeddings_region_path(
            region=region, trial=trial,
        )
        if not os.path.exists(emb_path):
            return False

        # Check for justification if needed
        if config.justify:
            just = (
                query
                .construct_query_justification_region_path(
                    region=region, trial=trial,
                )
            )
            if not json_file_has_key(path=just, key=model):
                return False

        # All required files and keys are present, trial is done
        return True

    def skip_check() -> bool:
        """
        Check if all trials and final result are complete, and skip if so

        Returns:
            * True if all trials and final result are complete, False otherwise
        """
        # Check all retest trials
        for t in range(config.retest):
            if not _trial_done(t):
                return False

        # Check final averaged result
        if not _trial_done("final"):
            return False

        # All trials and final result are done, we can skip
        logger.info(f"Skipping {region} ({model}) - already done")
        return True

    def load_trial(
        trial: Union[int, str],
    ) -> Tuple[Dict[str, Any], Optional[str]]:
        """
        Load the cleaned response, embedding, and justification (if needed)
        for a single trial

        Args:
            * trial: Trial number (int) or 'final' for final averaged result

        Returns:
            * Tuple of (trial data dict, justification string or None)
        """
        # Load cleaned response from JSON file
        clean_path = (
            query.construct_query_cleaned_region_path(
                region=region, trial=trial,
            )
        )
        with open(clean_path) as f:
            data = json.load(f)
        cleaned = data[model]

        # Load embedding from CSV file
        emb_path = emb.construct_embeddings_region_path(
            region=region, trial=trial,
        )
        df = pd.read_csv(emb_path, index_col=0)
        embedding = df.iloc[0].tolist()

        # Load per-function embeddings if available
        pf_path = (
            emb.construct_per_function_embeddings_region_path(
                region=region, trial=trial,
            )
        )
        per_function_embeddings = None
        if os.path.exists(pf_path):
            pf_df = pd.read_csv(pf_path, index_col=0)
            per_function_embeddings = pf_df.values.tolist()

        # Load justification if needed
        justification = None
        if config.justify:
            just_path = (
                query
                .construct_query_justification_region_path(
                    region=region, trial=trial,
                )
            )
            if os.path.exists(just_path):
                with open(just_path) as f:
                    jdata = json.load(f)
                justification = jdata.get(model)

        # Return the trial data and justification (if any)
        return {
            "cleaned": cleaned,
            "embedding": embedding,
            "per_function_embeddings": per_function_embeddings,
            "response": str(cleaned),
        }, justification

    def process_response(
        answer: str, response: str,
    ) -> Dict[str, Any]:
        """
        Process the raw model response to extract and clean the list of
        functions, compute embeddings, and prepare data for saving

        Args:
            * answer: Raw model response containing the list of functions
            * response: Full raw response

        Returns:
            * Dict containing cleaned functions, combined embedding,
                per-function embeddings, and original response
        """
        cleaned = clean_functions_response(response=answer)
        # Per-function embeddings via single batch call
        per_function_embeddings = (
            config.client_manager.get_embeddings_batch(
                texts=cleaned, model=model
            )
        )
        # Combined embedding = mean of per-function vectors
        combined_embedding = np.mean(
            per_function_embeddings, axis=0
        ).tolist()
        return {
            "cleaned": cleaned,
            "embedding": combined_embedding,
            "per_function_embeddings": per_function_embeddings,
            "response": response,
        }

    def save_result(
        result: Dict[str, Any],
        trial: Union[int, str],
        justification: Optional[str] = None,
    ) -> None:
        """
        Save the cleaned functions, embedding, and justification (if any) for
        a single trial

        Args:
            * result: Dict containing cleaned functions, embedding,
                per-function embeddings, and original response
            * trial: Trial number (int) or 'final' for final averaged result
            * justification: Justification string to save (if any)
        """
        _save_function_results(
            model=model, config=config,
            region=region,
            response=result["response"],
            embedding=result["embedding"],
            functions=result["cleaned"],
            hemisphere=hemisphere,
            trial=trial,
            analysis_type=analysis_type,
            justification=justification,
            per_function_embeddings=result.get("per_function_embeddings"),
        )

    def save_final(
        results: List[Dict[str, Any]],
        justifications: List[str],
    ) -> None:
        """
        Save the final averaged result by computing consensus functions and
        mean embedding across all trials, and save a summary of retest results
        and justifications if applicable

        Args:
            * results: List of dicts containing cleaned functions, embeddings,
                per-function embeddings, and original responses from each trial
            * justifications: List of justification strings from each trial
        """
        # Get cleaned functions, embeddings, and per-function embeddings
        cleaned = [r["cleaned"] for r in results]
        embeddings = [r["embedding"] for r in results]
        per_func_embs = [
            r["per_function_embeddings"] for r in results
            if r.get("per_function_embeddings") is not None
        ]

        # Average the results across trials to get consensus functions and mean
        # embedding, then save final result and summary
        consensus_threshold = config.consensus_threshold
        avg = average_function_trials(
            trial_cleaned=cleaned,
            trial_embeddings=embeddings,
            trial_per_function_embeddings=(
                per_func_embs if per_func_embs else None
            ),
            consensus_threshold=consensus_threshold,
        )
        _save_function_results(
            model=model, config=config,
            region=region,
            response=str(avg["consensus_functions"]),
            embedding=avg["mean_embedding"],
            functions=avg["consensus_functions"],
            hemisphere=hemisphere,
            trial="final",
            analysis_type=analysis_type,
        )

        # If retesting was done, also save a summary of the retest results
        if config.retest > 1:
            summary = {
                "num_trials": config.retest,
                "consistency_score": avg[
                    "consistency_score"
                ],
                "function_frequencies": avg[
                    "function_frequencies"
                ],
                "consensus_functions": avg[
                    "consensus_functions"
                ],
                "trials": cleaned,
            }
            if justifications:
                summary["justifications"] = justifications
            _save_retest_summary(
                region=region, model=model,
                config=config, analysis_type=analysis_type,
                hemisphere=hemisphere,
                summary_data=summary,
            )

    # Run the task processor with the defined functions
    process_task(
        config,
        model=model,
        skip_check=skip_check,
        trial_complete=_trial_done,
        load_trial=load_trial,
        prompt_kwargs={
            "prompt_type": analysis_type,
            "region_name": region,
            "hemisphere": hemisphere,
        },
        log_label=f"{region} ({model})",
        process_response=process_response,
        save_result=save_result,
        save_final=save_final,
    )
