import os
import json
from typing import Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils.misc.logging_setup import logger

from utils.misc.query_saves import (
    _save_function_results,
    _save_probability_results,
)
from utils.misc.atlas import load_regions_for_species
from utils.prompts import generate_prompt
from utils.core.response_cleaning import (
    clean_functions_response,
    clean_probability_response,
)
from utils.core.aggregation import aggregate_results
from utils.core.visualisation import create_visualisations

from utils.paths.base import BasePathConstructor
from utils.paths.query import QueryPathConstructor
from utils.paths.embeddings import EmbeddingsPathConstructor


class BrainAnalyser:
    """Brain Region Analyser"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        if not config.regions:
            config.regions = load_regions_for_species(
                species=config.species,
                atlas_name=config.atlas_name,
            )

    def analyze_functions(self) -> bool:
        """
        Given a list of brain regions, prompt an LLM to identify the likeliest
        functions associated with each region, then embed the results to find a
        similarity score for each vector pair
        """
        logger.info(f"Analyzing functions for {self.config.species}")
        logger.info(
            f"Processing {len(self.config.regions)} regions with "
            f"{len(self.config.models)} models"
        )

        success_count = self._process_regions(analysis_type="functions")
        if success_count < len(self.config.regions):
            logger.error_status(
                "Some regions failed to process. Check logs for details. "
                "Exiting post-processing"
            )
            raise

        self._run_post_processing(analysis_type="functions")

    def analyze_probabilities(self) -> bool:
        """
        Given a list of brain regions and functions, prompt an LLM to determine
        the probability that each function is associated with each region
        """
        if not self.config.functions:
            logger.error(
                "Functions must be specified for probability analysis",
                exc_info=True,
            )
            raise

        logger.info(f"Analyzing probabilities for {self.config.species}")
        logger.info(f"Functions: {', '.join(self.config.functions)}")

        success_count = self._process_regions(analysis_type="probabilities")
        if success_count < len(self.config.regions):
            logger.error_status(
                "Some regions failed to process. Check logs for details. "
                "Exiting post-processing"
            )
            raise

        self._run_post_processing(analysis_type="probabilities")

    def _process_regions(self, analysis_type: str) -> bool:
        """
        Process all regions in parallel

        Args:
            * analysis_type: "functions" or "probabilities"
        """
        logger.info(f"Running in parallel ({self.config.workers} workers)")

        with ThreadPoolExecutor(max_workers=self.config.workers) as executor:
            # Submit all region processing tasks
            future_to_region = {
                executor.submit(
                    self._process_single_region, region, analysis_type
                ): region
                for region in self.config.regions
            }

            # Process results as they complete
            success_count = 0
            for future in as_completed(future_to_region):
                region = future_to_region[future]
                try:
                    future.result()
                    success_count += 1
                    logger.processing(
                        f"Completed {self.config.species} - {region}"
                    )
                except Exception as e:
                    logger.error_status(f"Failed {region}: {e}")

        logger.info(
            f"Completed {success_count}/{len(self.config.regions)} regions"
        )
        return success_count

    def _process_single_region(self, region: str, analysis_type: str) -> None:
        """
        Process one region for all hemispheres and models

        Args:
            * region: Brain region name
            * analysis_type: "functions" or "probabilities"
        """

        hemispheres = (
            ["left", "right"] if self.config.separate_hemispheres else [None]
        )

        for hemisphere in hemispheres:
            if analysis_type == "functions":
                for model in self.config.models:
                    self._process_function_task(
                        region=region,
                        hemisphere=hemisphere,
                        model=model,
                        analysis_type=analysis_type,
                    )
            else:  # probabilities
                for function in self.config.functions:
                    for model in self.config.models:
                        self._process_probability_task(
                            region=region,
                            hemisphere=hemisphere,
                            function=function,
                            model=model,
                            analysis_type=analysis_type,
                        )

    def _get_existing_response(
        self,
        region: str,
        hemisphere: str,
        model: str,
        analysis_type: str,
        function: str = None,
    ):
        """
        Load and return an existing raw LLM response if one exists for this
        exact configuration. Returns None if no response is found

        For functions: returns the raw response string
        For probabilities: returns the stored result for the given function

        Args:
            * region: Brain region name
            * hemisphere: Hemisphere string
            * model: Model name
            * analysis_type: "functions" or "probabilities"
            * function: Brain function (only for probabilities)

        Returns:
            * The existing response, or None if not found
        """

        # Get the path for the raw response
        result_path = QueryPathConstructor.construct_query_region_path(
            model=model,
            region=region,
            species=self.config.species,
            atlas_name=self.config.atlas_name,
            analysis_type=analysis_type,
            hemisphere=hemisphere,
            template_name=self.config.prompt_template_name,
        )

        # If the file doesn't exist, return None
        if not os.path.exists(result_path):
            return None

        # If it does exist, try to load it and return the relevant part
        try:
            with open(result_path, "r") as f:
                existing = json.load(f)
        except (json.JSONDecodeError, OSError):
            return None

        # For functions, return raw response string. For probabilities, return
        # the stored result for the given function
        if analysis_type == "functions":
            return list(existing.values())[0] if existing else None
        else:
            return existing.get(function)

    def _process_function_task(
        self,
        region: str,
        hemisphere: str,
        model: str,
        analysis_type: str = "functions",
    ):
        """
        Process a single function analysis task
            1. Query model (skip if raw response exists)
            2. Clean response
            3. Get embedding (skip if embedding exists)
            4. Save results

        Args:
            * region: Brain region name
            * hemisphere: Hemisphere string for directory naming
            * model: Model name
        """

        # Check for existing response from LLM
        existing_response = self._get_existing_response(
            region=region,
            hemisphere=hemisphere,
            model=model,
            analysis_type=analysis_type,
        )
        # Check for existing embedding
        embedding_path = (
            EmbeddingsPathConstructor.construct_embeddings_region_path(
                model=model,
                region=region,
                species=self.config.species,
                atlas_name=self.config.atlas_name,
                analysis_type=analysis_type,
                hemisphere=hemisphere,
                template_name=self.config.prompt_template_name,
            )
        )

        # If both exist, skip processing this region/model combo
        if existing_response is not None and os.path.exists(embedding_path):
            logger.info(f"Skipping {region} ({model}) - already done")
            return

        # 1. Get response (load existing or query LLM)
        if existing_response is not None:
            logger.info(
                f"Loading existing response for {region} ({model})"
            )
            response = existing_response
        else:
            logger.processing(f"Querying {region} ({model})")
            prompt = generate_prompt(
                prompt_type=analysis_type,
                region_name=region,
                species=self.config.species,
                atlas_name=self.config.atlas_name,
                hemisphere=hemisphere,
                template_name=self.config.prompt_template_name,
                save_to_results=True,
            )
            response = self.config.client_manager.query_model(
                model_name=model, prompt=prompt
            )

        # 2. Clean response
        cleaned_functions = clean_functions_response(response=response)

        # 3. Get embedding
        embedding = self.config.client_manager.get_embeddings(
            text=", ".join(cleaned_functions), model=model
        )

        # 4. Save results
        _save_function_results(
            model=model,
            config=self.config,
            region=region,
            response=response,
            embedding=embedding,
            functions=cleaned_functions,
            hemisphere=hemisphere,
            analysis_type=analysis_type,
        )

    def _process_probability_task(
        self,
        region: str,
        hemisphere: str,
        function: str,
        model: str,
        analysis_type: str = "probabilities",
    ):
        """
        Process a single probability analysis task
            1. Generate prompt
            2. Query model
            3. Clean response
            4. Save results

        Args:
            * region: Brain region name
            * hemisphere: Hemisphere string for directory naming
            * hemisphere_value: "left", "right", or "both"
            * function: Brain function to query
            * model: Model name
            * analysis_type: Type of analysis ("probabilities")
        """

        # Check for existing response from LLM - if exists, skip
        if self._get_existing_response(
            region=region,
            hemisphere=hemisphere,
            model=model,
            analysis_type=analysis_type,
            function=function,
        ) is not None:
            logger.info(
                f"Skipping {region}/{function} ({model}) - already done"
            )
            return

        logger.processing(
            f"Querying {region} - {function} ({model})"
        )

        # 1. Generate prompt
        prompt = generate_prompt(
            prompt_type=analysis_type,
            region_name=region,
            species=self.config.species,
            atlas_name=self.config.atlas_name,
            hemisphere=hemisphere,
            template_name=self.config.prompt_template_name,
            function=function,
            save_to_results=True,
        )

        # 2. Query model
        response = self.config.client_manager.query_model(
            model_name=model, prompt=prompt
        )

        # 3. Clean response
        probability = clean_probability_response(response)

        # 4. Save results
        _save_probability_results(
            region=region,
            hemisphere=hemisphere,
            function=function,
            model=model,
            config=self.config,
            probability=probability,
            analysis_type=analysis_type,
        )

    def _run_post_processing(self, analysis_type: str):
        """
        Run aggregation, visualization, and cleanup
        Args:
            * analysis_type: "functions" or "probabilities"
        """
        logger.info("Aggregating results...")
        aggregate_results(config=self.config, analysis_type=analysis_type)

        if not self.config.skip_visualization:
            logger.info("Creating visualizations...")
            create_visualisations(
                config=self.config, analysis_type=analysis_type
            )

        if self.config.skip_raw_saving:
            self._cleanup_raw_data()

        logger.success(f"{analysis_type.title()} analysis complete!")

    def _cleanup_raw_data(self):
        """
        Clean up raw data directory
        """
        try:
            BasePathConstructor.cleanup_raw_dir()
        except Exception as e:
            logger.error_status(
                f"Could not clean up raw data: {e}", exc_info=True
            )
            raise
