from typing import Dict, Any
from itertools import product
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils.misc.logging_setup import logger
from utils.core.aggregation import aggregate_results
from utils.core.visualisation import create_visualisations
from utils.core.function_task import run_function_task
from utils.core.probability_task import run_probability_task
from utils.core.ranking_task import run_ranking_task

from utils.paths.base import BasePathConstructor


class BrainAnalyser:
    """Brain Region Analyser"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def analyze_functions(self):
        """
        Identify the likeliest functions per region, embed results,
        and compute similarity scores
        """
        logger.info(f"Analyzing functions for {self.config.species}")
        logger.info(
            f"Processing {len(self.config.regions)} regions with "
            f"{len(self.config.models)} models"
        )

        success_count = self._process_regions(analysis_type="top-functions")
        if success_count < len(self.config.regions):
            logger.error_status(
                "Some regions failed to process. Check logs for details. "
                "Exiting post-processing"
            )
            raise RuntimeError("Incomplete region processing")

        self._run_post_processing(analysis_type="top-functions")

    def analyze_probabilities(self):
        """
        Estimate the probability each function is associated with
        each region
        """
        if not self.config.functions:
            logger.error(
                "Functions must be specified for probability analysis",
                exc_info=True,
            )
            raise ValueError("No functions specified")

        logger.info(f"Analyzing probabilities for {self.config.species}")
        logger.info(f"Functions: {', '.join(self.config.functions)}")

        success_count = self._process_regions(analysis_type="query-functions")
        if success_count < len(self.config.regions):
            logger.error_status(
                "Some regions failed to process. Check logs for details. "
                "Exiting post-processing"
            )
            raise RuntimeError("Incomplete region processing")

        self._run_post_processing(analysis_type="query-functions")

    def analyze_rankings(self):
        """
        Rank pairs of regions by relevance to each function
        """
        if not self.config.functions:
            raise ValueError(
                "Functions must be specified for ranking analysis"
            )
        if not self.config.pairs:
            raise ValueError(
                "Pairs must be specified for ranking analysis"
            )

        logger.info(
            f"Analyzing rankings for {self.config.species}"
        )
        logger.info(
            f"Functions: {', '.join(self.config.functions)}"
        )
        logger.info(f"Pairs: {len(self.config.pairs)}")

        success_count = self._process_pairs()
        total = len(self.config.pairs)
        if success_count < total:
            logger.error_status(
                "Some pairs failed to process. Check logs for details. "
                "Exiting post-processing"
            )
            raise RuntimeError("Incomplete pair processing")

        self._run_post_processing(analysis_type="rankings")

    def _process_regions(self, analysis_type: str) -> int:
        """
        Process all regions in parallel

        Args:
            * analysis_type: "top-functions" or "query-functions"
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

    def _process_pairs(self) -> int:
        """Process all pairs in parallel"""
        logger.info(
            f"Running in parallel ({self.config.workers} workers)"
        )
        hemispheres = (
            ["left", "right"]
            if self.config.separate_hemispheres
            else [None]
        )

        with ThreadPoolExecutor(
            max_workers=self.config.workers
        ) as executor:
            future_to_task = {}
            for pair, function, hemisphere, model in product(
                self.config.pairs,
                self.config.functions,
                hemispheres,
                self.config.models,
            ):
                future = executor.submit(
                    run_ranking_task,
                    config=self.config,
                    region_1=pair[0],
                    region_2=pair[1],
                    function=function,
                    model=model,
                    hemisphere=hemisphere,
                )
                future_to_task[future] = (pair, function, model)

            success_count = 0
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    future.result()
                    success_count += 1
                except Exception as e:
                    logger.error_status(f"Failed {task}: {e}")

        total = (
            len(self.config.pairs)
            * len(self.config.functions)
            * len(hemispheres)
            * len(self.config.models)
        )
        logger.info(f"Completed {success_count}/{total} tasks")
        return len(self.config.pairs) if success_count == total else 0

    def _process_single_region(
        self, region: str, analysis_type: str
    ) -> None:
        """
        Process one region for all hemispheres and models

        Args:
            * region: Brain region name
            * analysis_type: "top-functions" or "query-functions"
        """
        hemispheres = (
            ["left", "right"] if self.config.separate_hemispheres else [None]
        )

        if analysis_type == "top-functions":
            for hemisphere, model in product(
                hemispheres, self.config.models
            ):
                run_function_task(
                    config=self.config,
                    region=region,
                    hemisphere=hemisphere,
                    model=model,
                )
        else:  # probabilities
            for hemisphere, function, model in product(
                hemispheres, self.config.functions, self.config.models
            ):
                run_probability_task(
                    config=self.config,
                    region=region,
                    hemisphere=hemisphere,
                    function=function,
                    model=model,
                )

    def _run_post_processing(self, analysis_type: str):
        """
        Run aggregation, visualization, and cleanup

        Args:
            * analysis_type: "top-functions", "query-functions", or "rankings"
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
        """Clean up raw data directory"""
        try:
            BasePathConstructor.cleanup_raw_dir()
        except Exception as e:
            logger.error_status(
                f"Could not clean up raw data: {e}", exc_info=True
            )
            raise
