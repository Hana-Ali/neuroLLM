import os
import json
import pandas as pd
from typing import Dict, Any

from utils.misc.logging_setup import logger

from utils.paths.query import QueryPathConstructor
from utils.paths.embeddings import EmbeddingsPathConstructor
from utils.paths.aggregation import AggregatedResultsPathConstructor


def aggregate_function_results(
    config: Dict[str, Any], analysis_type: str = "functions"
):
    """
    Function aggregation. Creates:
    - results/aggregated/functions/{species}/{atlas}/{model}/{template}/
        {hemisphere}/all_responses.json
    - results/aggregated/functions/{species}/{atlas}/{model}/{template}/
        {hemisphere}/all_embeddings.csv

    Args:
        * config: Analysis configuration
        * analysis_type: "functions" or "probabilities"
    """
    hemispheres = ["left", "right"] if config.separate_hemispheres else [None]

    for hemisphere in hemispheres:
        for model in config.models:
            # Collect all function results for this model/hemisphere
            all_responses = {}
            embedding_dfs = []

            for region in config.regions:

                # Get the function response path
                res_path = (
                    QueryPathConstructor.construct_query_cleaned_region_path(
                        model=model,
                        region=region,
                        species=config.species,
                        atlas_name=config.atlas_name,
                        analysis_type=analysis_type,
                        hemisphere=hemisphere,
                        template_name=config.prompt_template_name,
                    )
                )
                if not os.path.exists(res_path):
                    logger.error_status(
                        f"No query results file found: {res_path}",
                        exc_info=True,
                    )
                    raise

                # Load function response
                with open(res_path) as f:
                    data = json.load(f)
                    if model in data:
                        all_responses[region] = data[model]

                # Get the embedding path
                emb_path = (
                    EmbeddingsPathConstructor.construct_embeddings_region_path(
                        model=model,
                        region=region,
                        species=config.species,
                        atlas_name=config.atlas_name,
                        hemisphere=hemisphere,
                        template_name=config.prompt_template_name,
                        analysis_type=analysis_type,
                    )
                )
                if not os.path.exists(emb_path):
                    logger.error_status(
                        f"No embeddings file found: {emb_path}", exc_info=True
                    )
                    raise

                # Load embedding and set region as index
                df = pd.read_csv(emb_path, index_col=0)
                embedding_dfs.append(df)

            # Get aggregated paths
            agg_path = AggregatedResultsPathConstructor.construct_aggregated_query_results_path(
                model=model,
                species=config.species,
                atlas_name=config.atlas_name,
                analysis_type=analysis_type,
                hemisphere=hemisphere,
                template_name=config.prompt_template_name,
            )
            aggemb_path = AggregatedResultsPathConstructor.construct_aggregated_embeddings_path(
                model=model,
                species=config.species,
                atlas_name=config.atlas_name,
                analysis_type=analysis_type,
                hemisphere=hemisphere,
                template_name=config.prompt_template_name,
            )
            os.makedirs(os.path.dirname(agg_path), exist_ok=True)
            os.makedirs(os.path.dirname(aggemb_path), exist_ok=True)

            # Save query responses
            with open(agg_path, "w") as f:
                json.dump(all_responses, f, indent=2)
            logger.processing(
                f"Saved {len(all_responses)} function responses for "
                f"{model}/{hemisphere if hemisphere else 'no_separation'}"
            )

            # Save embeddings as CSV
            all_embeddings_df = pd.concat(embedding_dfs)
            all_embeddings_df.to_csv(aggemb_path)
            logger.processing(
                f"Saved {len(embedding_dfs)} embeddings for "
                f"{model}/{hemisphere if hemisphere else 'no_separation'}"
            )


def aggregate_probability_results(
    config: Dict[str, Any], analysis_type: str = "probabilities"
):
    """
    Probability aggregation. Creates:
    - results/aggregated/probabilities/{species}/{atlas}/{model}/{template}/
        {hemisphere}/probability_distribution.csv
    - results/aggregated/probabilities/{species}/{atlas}/{model}/{template}/
        {hemisphere}/{function}/probabilities.csv

    Args:
        * config: Analysis configuration
        * analysis_type: "functions" or "probabilities"
    """
    hemispheres = ["left", "right"] if config.separate_hemispheres else [None]

    for hemisphere in hemispheres:
        for model in config.models:
            # Collect probability data
            all_data = {}

            for region in config.regions:

                # Get the function response path
                res_path = QueryPathConstructor.construct_query_region_path(
                    model=model,
                    region=region,
                    species=config.species,
                    atlas_name=config.atlas_name,
                    analysis_type=analysis_type,
                    hemisphere=hemisphere,
                    template_name=config.prompt_template_name,
                )
                if not os.path.exists(res_path):
                    logger.error_status(
                        f"No query results file found: {res_path}",
                        exc_info=True,
                    )
                    raise

                # Load function response
                with open(res_path) as f:
                    region_data = json.load(f)
                    all_data[region] = region_data

            # Create overview DataFrame
            df_data = []
            for region in config.regions:
                if region in all_data:
                    row = {"Region": region}
                    for function in config.functions:
                        # Cleaner probability extraction
                        prob_value = (
                            all_data[region].get(function, {}).get(model)
                        )
                        row[function] = prob_value
                    df_data.append(row)

            df = pd.DataFrame(df_data).set_index("Region")

            # Get aggregated paths
            agg_path = AggregatedResultsPathConstructor.construct_aggregated_query_results_path(
                model=model,
                species=config.species,
                atlas_name=config.atlas_name,
                analysis_type=analysis_type,
                hemisphere=hemisphere,
                template_name=config.prompt_template_name,
                extension="csv",
            )
            os.makedirs(os.path.dirname(agg_path), exist_ok=True)
            df.to_csv(agg_path)
            logger.processing(
                f"Saved probability overview for {model}/"
                f"{hemisphere if hemisphere else 'no_separation'}"
            )

            # Save individual function files
            for function in config.functions:
                path = AggregatedResultsPathConstructor.construct_individual_function_prob_path(
                    model=model,
                    species=config.species,
                    atlas_name=config.atlas_name,
                    analysis_type=analysis_type,
                    hemisphere=hemisphere,
                    function=function,
                    template_name=config.prompt_template_name,
                )
                os.makedirs(os.path.dirname(path), exist_ok=True)

                # Extract function probabilities and drop NaNs
                func_df = (
                    df[[function]]
                    .dropna()
                    .rename(columns={function: "Probability"})
                )

                # Save if not empty
                if not func_df.empty:
                    func_df.to_csv(path)
                    logger.processing(
                        f"Saved {len(func_df)} probabilities for {function}"
                    )


def aggregate_results(config: Dict[str, Any], analysis_type: str):
    """
    Aggregate results based on analysis type

    Args:
        * config: Analysis configuration
        * analysis_type: "functions" or "probabilities"
    """
    if analysis_type == "functions":
        aggregate_function_results(config=config, analysis_type=analysis_type)
    elif analysis_type == "probabilities":
        aggregate_probability_results(
            config=config, analysis_type=analysis_type
        )
    else:
        raise ValueError(f"Unknown analysis type: {analysis_type}")
    logger.success("Aggregation completed successfully!")
