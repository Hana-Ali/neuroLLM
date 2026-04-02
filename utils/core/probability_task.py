import os
import json

from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple, Union

from utils.misc.logging_setup import logger
from utils.misc.query_saves import (
    _save_probability_results,
    _save_retest_summary,
)
from utils.core.response_cleaning import clean_probability_response
from utils.core.task_processor import (
    process_task,
    json_file_has_key,
)

from utils.paths.query import QueryPathConstructor

from utils.core.retest_averaging import average_probability_trials


def run_probability_task(
    config: SimpleNamespace,
    region: str,
    hemisphere: Optional[str],
    function: str,
    model: str,
) -> None:
    """
    Probability analysis task

    Args:
        * config: Config object with necessary attributes
        * region: Brain region name
        * hemisphere: Hemisphere ('left', 'right', or None)
        * function: Function name to query
        * model: Model name to use for querying
    """
    analysis_type = "query-functions"

    query = QueryPathConstructor(
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
        # Create the path to the JSON file for this trial
        path = query.construct_query_region_path(region=region, trial=trial)

        # Check if the JSON file exists and contains the function key
        if not json_file_has_key(path=path, key=function):
            return False

        # Check for justification if needed
        if config.justify:
            just = (
                query
                .construct_query_justification_region_path(
                    region=region, trial=trial,
                )
            )
            if not json_file_has_key(path=just, key=function):
                return False

        # Result and justification (if needed) are present, trial is done
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
        logger.info(f"Skipping {region}/{function} ({model}) - already done")
        return True

    def load_trial(
        trial: Union[int, str],
    ) -> Tuple[Dict[str, Any], Optional[str]]:
        """
        Load the probability result (and justification if needed) for a single
        trial

        Args:
            * trial: Trial number (int) or 'final' for final averaged result

        Returns:
            * Tuple of (result dict, justification string or None)
        """
        # Load probability result from JSON file
        path = query.construct_query_region_path(region=region, trial=trial)
        with open(path) as f:
            data = json.load(f)
        prob = data[function][model]

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
                justification = (
                    jdata.get(function, {}).get(model)
                )

        # Return the probability result and justification (if any)
        return {"probability": prob}, justification

    def process_response(answer: str, _response: str) -> Dict[str, float]:
        """
        Process the raw model response to extract the probability value and
        clean it

        Args:
            * answer: Raw model response containing the probability
            * _response: Full raw response

        Returns:
            * Dict containing the cleaned probability value
        """
        return {"probability": clean_probability_response(answer)}

    def save_result(
        result: Dict[str, float],
        trial: Union[int, str],
        justification: Optional[str] = None,
    ) -> None:
        """
        Save the probability result (and justification if needed) for a single
        trial

        Args:
            * result: Dict containing the probability value to save
            * trial: Trial number (int) or 'final' for final averaged result
            * justification: Justification string to save (if any)
        """
        _save_probability_results(
            region=region, hemisphere=hemisphere,
            function=function, model=model,
            config=config,
            probability=result["probability"],
            trial=trial,
            analysis_type=analysis_type,
            justification=justification,
        )

    def save_final(
        results: List[Dict[str, float]],
        justifications: List[str],
    ) -> None:
        """
        Save the final averaged probability result across all retest trials,
        along with a summary of the retest results and justifications if
        applicable

        Args:
            * results: List of dicts containing prob values from each trial
            * justifications: List of justification strings from each trial
        """
        # Average the probability values across trials and save final result
        probs = [r["probability"] for r in results]
        avg = average_probability_trials(probabilities=probs)
        _save_probability_results(
            region=region, hemisphere=hemisphere,
            function=function, model=model,
            config=config,
            probability=avg["mean"],
            trial="final",
            analysis_type=analysis_type,
        )

        # If retesting was done, also save a summary of the retest results
        if config.retest > 1:
            func_summary = {
                "mean": avg["mean"],
                "std": avg["std"],
                "min": avg["min"],
                "max": avg["max"],
                "trials": probs,
            }
            if justifications:
                func_summary["justifications"] = justifications
            summary = {
                "num_trials": config.retest,
                "functions": {function: func_summary},
            }
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
            "function": function,
        },
        log_label=f"{region} - {function} ({model})",
        process_response=process_response,
        save_result=save_result,
        save_final=save_final,
    )
