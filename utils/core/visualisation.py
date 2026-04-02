import os
import pandas as pd
import seaborn as sns
from typing import Dict, Any
import matplotlib.pyplot as plt

from sklearn.metrics.pairwise import cosine_similarity

from utils.misc.logging_setup import logger

from utils.paths.aggregation import AggregatedResultsPathConstructor
from utils.paths.visualisation import VisualizationPathConstructor


def create_similarity_visualizations(
    config: Dict[str, Any], analysis_type: str = "top-functions"
):
    """
    Create similarity matrix visualizations

    Reads from:
    - results/aggregated/functions/{species}/{atlas}/{model}/{template}/
        {hemisphere}/all_embeddings.csv
    Creates:
    - results/visualizations/similarities/{species}/{atlas}/{model}/{template}/
        {hemisphere}/similarity_matrix.png
    """
    hemispheres = ["left", "right"] if config.separate_hemispheres else [None]

    for hemisphere in hemispheres:
        for model in config.models:
            common = dict(
                model=model, species=config.species,
                atlas_name=config.atlas_name,
                analysis_type=analysis_type,
                hemisphere=hemisphere,
                template_name=config.prompt_template_name,
            )
            agg = AggregatedResultsPathConstructor(**common)
            viz = VisualizationPathConstructor(**common)

            # Load aggregated embeddings
            emb_path = agg.construct_aggregated_embeddings_path()
            if not os.path.exists(emb_path):
                logger.error_status(
                    f"No embeddings found: {emb_path}", exc_info=True
                )
                raise FileNotFoundError(
                    f"No embeddings found: {emb_path}"
                )

            # Load and process embeddings
            df = pd.read_csv(emb_path, index_col=0)

            # Get embedding vectors
            regions = df.index.tolist()
            embeddings = df.values

            # Compute similarity matrix
            similarity_matrix = cosine_similarity(embeddings)

            # Create similarity DataFrame
            sim_df = pd.DataFrame(
                similarity_matrix, index=regions, columns=regions
            )

            # Get visualisation path
            path = viz.construct_visualisations_similarity_path(
                extension="csv",
            )
            os.makedirs(os.path.dirname(path), exist_ok=True)
            sim_df.to_csv(path)

            # Create visualization
            plt.figure(figsize=(20, 16))
            sns.heatmap(
                sim_df,
                annot=True,
                fmt=".2f",
                cmap="magma",
                square=True,
                xticklabels=True,
                yticklabels=True,
                annot_kws={"size": 6},
                cbar_kws={"shrink": 0.5},
            )
            plt.xticks(rotation=90, fontsize=8)
            plt.yticks(rotation=0, fontsize=8)

            # Title
            hemi_text = (
                hemisphere.capitalize()
                if hemisphere is not None
                else "No Hemisphere Separation"
            )
            atlas_text = f" ({config.atlas_name})" if config.atlas_name else ""
            plt.title(
                (
                    f"Region Similarity - {model} - {config.species}"
                    f"{atlas_text} - {hemi_text}"
                ),
                fontsize=12,
            )

            # Save plot
            visualisation_path = path.replace("csv", "png")
            plt.savefig(
                visualisation_path,
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()

            logger.processing(
                f"Created similarity visualization: {model}/"
                f"{hemisphere if hemisphere else 'no_separation'}"
            )


def create_probability_visualizations(
    config: Dict[str, Any], analysis_type: str = "query-functions"
):
    """
    Create probability heatmap visualizations

    Reads from:
    - results/aggregated/probabilities/{species}/{atlas}/{model}/{template}/
        {hemisphere}/probability_distribution.csv
    Creates:
    - results/visualizations/probabilities/{species}/{atlas}/{model}/
        {template}/{hemisphere}/heatmap.png
    """
    hemispheres = ["left", "right"] if config.separate_hemispheres else [None]

    for hemisphere in hemispheres:
        for model in config.models:
            common = dict(
                model=model, species=config.species,
                atlas_name=config.atlas_name,
                analysis_type=analysis_type,
                hemisphere=hemisphere,
                template_name=config.prompt_template_name,
            )
            agg = AggregatedResultsPathConstructor(**common)
            viz = VisualizationPathConstructor(**common)

            # Load aggregated probabilities
            agg_path = agg.construct_aggregated_query_results_path(
                extension="csv",
            )
            if not os.path.exists(agg_path):
                logger.error_status(
                    f"No probabilities found: {agg_path}", exc_info=True
                )
                raise FileNotFoundError(
                    f"No probabilities found: {agg_path}"
                )

            # Load data
            df = pd.read_csv(agg_path, index_col=0)
            df = df.fillna(0.0)

            # Create output directory
            path = viz.construct_visualisations_probability_path(
                extension="png",
            )
            os.makedirs(os.path.dirname(path), exist_ok=True)

            # Create main heatmap
            plt.figure(
                figsize=(len(config.functions) * 0.8 + 3, len(df) * 0.3 + 3)
            )

            # Check if we have negative values
            has_negative = (df.values < 0).any()

            if has_negative:
                # Use diverging colormap for negative values
                sns.heatmap(
                    df,
                    annot=True,
                    fmt=".2f",
                    cmap="RdBu_r",  # Red for negative, blue for positive
                    center=0,
                    vmin=-1,
                    vmax=1,
                    xticklabels=True,
                    yticklabels=True,
                    annot_kws={"size": 7},
                    cbar_kws={"shrink": 0.5},
                )
            else:
                # Standard colormap for positive values
                sns.heatmap(
                    df,
                    annot=True,
                    fmt=".2f",
                    cmap="magma",
                    vmin=0,
                    vmax=1,
                    xticklabels=True,
                    yticklabels=True,
                    annot_kws={"size": 7},
                    cbar_kws={"shrink": 0.5},
                )

            plt.xticks(rotation=45, fontsize=9)
            plt.yticks(rotation=0, fontsize=8)

            # Title
            hemi_text = (
                hemisphere.capitalize()
                if hemisphere is not None
                else "No Hemisphere Separation"
            )
            atlas_text = f" ({config.atlas_name})" if config.atlas_name else ""
            plt.title(
                f"Probabilities - {model} - {hemi_text}{atlas_text}",
                fontsize=12,
            )

            # Save main heatmap
            plt.savefig(path, dpi=300, bbox_inches="tight")
            plt.close()

            # Create individual function plots
            for function in config.functions:
                if function not in df.columns:
                    continue

                func_data = df[function].dropna()
                if func_data.empty:
                    continue

                # Sort by probability
                func_data = func_data.sort_values(ascending=True)

                plt.figure(figsize=(10, len(func_data) * 0.3 + 2))

                # Color bars based on positive/negative values
                if has_negative:
                    colors = [
                        "red" if val < 0 else "blue"
                        for val in func_data.values
                    ]
                    plt.barh(func_data.index, func_data.values, color=colors)
                    plt.axvline(x=0, color="black", linestyle="-", alpha=0.3)
                    plt.xlim(-1, 1)
                else:
                    plt.barh(func_data.index, func_data.values, color="blue")
                    plt.xlim(0, 1)

                plt.title(f"{function} - {model} - {hemi_text}")
                plt.xlabel("Probability")
                plt.tight_layout()

                # Get function visualisation path
                path = viz.construct_visualisations_function_path(
                    function=function, extension="png",
                )
                os.makedirs(os.path.dirname(path), exist_ok=True)

                # Save function plot
                plt.savefig(
                    path,
                    dpi=300,
                    bbox_inches="tight",
                )
                plt.close()

            logger.processing(
                f"Created probability visualizations: {model}/"
                f"{hemisphere if hemisphere else 'no_separation'}"
            )


def create_ranking_visualizations(
    config: Dict[str, Any], analysis_type: str = "rankings",
):
    """
    Create one comparison chart per pair. Each chart is a diverging horizontal
    bar chart with one row per function: bars extend left for region_1, right
    for region_2

    Reads from:
    - results/aggregated/rankings/{species}/{atlas}/{model}/{template}/
        {hemisphere}/{region_1}_vs_{region_2}/results.csv

    Creates:
    - results/visualizations/rankings/{species}/{atlas}/{model}/{template}/
        {hemisphere}/{region_1}_vs_{region_2}/comparison.png
    """
    hemispheres = (
        ["left", "right"]
        if config.separate_hemispheres
        else [None]
    )

    for hemisphere in hemispheres:
        for model in config.models:
            common = dict(
                model=model, species=config.species,
                atlas_name=config.atlas_name,
                analysis_type=analysis_type,
                hemisphere=hemisphere,
                template_name=config.prompt_template_name,
            )
            agg = AggregatedResultsPathConstructor(**common)
            viz = VisualizationPathConstructor(**common)

            for r1, r2 in config.pairs:
                agg_path = agg.construct_aggregated_pair_results_path(
                    region_1=r1, region_2=r2,
                )
                if not os.path.exists(agg_path):
                    logger.warning_status(
                        f"No ranking data for "
                        f"{r1} vs {r2}: {agg_path}"
                    )
                    continue

                df = pd.read_csv(agg_path, index_col=0)

                # Build diverging values: -1 for r1, +1 for r2
                functions = df.index.tolist()
                values = [
                    -1 if row["Winner"] == r1 else 1
                    for _, row in df.iterrows()
                ]
                colors = [
                    "#4C72B0" if v < 0 else "#DD8452"
                    for v in values
                ]

                fig_h = max(
                    len(functions) * 0.6 + 1.5, 3
                )
                fig, ax = plt.subplots(
                    figsize=(8, fig_h)
                )
                ax.barh(
                    functions, values,
                    color=colors, height=0.6,
                )

                ax.set_xlim(-1.5, 1.5)
                ax.set_xticks([])
                ax.axvline(
                    x=0, color="black",
                    linewidth=0.8,
                )

                # Label regions on each side
                ax.text(
                    -0.75, len(functions),
                    r1.replace("_", " "),
                    ha="center", va="bottom",
                    fontsize=11, fontweight="bold",
                    color="#4C72B0",
                )
                ax.text(
                    0.75, len(functions),
                    r2.replace("_", " "),
                    ha="center", va="bottom",
                    fontsize=11, fontweight="bold",
                    color="#DD8452",
                )

                ax.set_yticks(range(len(functions)))
                ax.set_yticklabels(functions, fontsize=9)
                ax.invert_yaxis()

                hemi_text = (
                    hemisphere.capitalize()
                    if hemisphere
                    else "No Hemisphere Separation"
                )
                atlas_text = (
                    f" ({config.atlas_name})"
                    if config.atlas_name
                    else ""
                )
                ax.set_title(
                    f"{r1} vs {r2} - {model} - "
                    f"{hemi_text}{atlas_text}",
                    fontsize=12,
                )

                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.spines["bottom"].set_visible(False)

                viz_path = viz.construct_visualisations_ranking_pair_path(
                    region_1=r1, region_2=r2,
                )
                os.makedirs(
                    os.path.dirname(viz_path),
                    exist_ok=True,
                )
                fig.savefig(
                    viz_path, dpi=300,
                    bbox_inches="tight",
                )
                plt.close(fig)

                logger.processing(
                    f"Created comparison chart: "
                    f"{r1} vs {r2} ({model})"
                )


def create_consistency_visualizations(
    config: Dict[str, Any],
    analysis_type: str = "top-functions",
):
    """
    Create retest consistency bar charts (functions only)
    """
    hemispheres = (
        ["left", "right"]
        if config.separate_hemispheres
        else [None]
    )

    for hemisphere in hemispheres:
        for model in config.models:
            common = dict(
                model=model, species=config.species,
                atlas_name=config.atlas_name,
                analysis_type=analysis_type,
                hemisphere=hemisphere,
                template_name=config.prompt_template_name,
            )
            agg = AggregatedResultsPathConstructor(**common)
            viz = VisualizationPathConstructor(**common)

            stats_path = agg.construct_aggregated_retest_stats_path(
                filename="consistency_scores.csv",
            )
            if not os.path.exists(stats_path):
                continue

            df = pd.read_csv(stats_path, index_col=0)
            df = df.sort_values(
                "consistency_score", ascending=True
            )

            viz_path = viz.construct_visualisations_consistency_path()
            os.makedirs(
                os.path.dirname(viz_path), exist_ok=True
            )

            plt.figure(
                figsize=(10, len(df) * 0.3 + 2)
            )
            plt.barh(
                df.index,
                df["consistency_score"],
                color="teal",
            )
            plt.xlim(0, 1)

            hemi_text = (
                hemisphere.capitalize()
                if hemisphere
                else "No Hemisphere Separation"
            )
            plt.title(
                f"Retest Consistency - {model} - "
                f"{hemi_text}"
            )
            plt.xlabel("Consistency Score")
            plt.tight_layout()
            plt.savefig(
                viz_path, dpi=300, bbox_inches="tight"
            )
            plt.close()

            logger.processing(
                f"Created consistency visualization: "
                f"{model}/"
                f"{hemisphere or 'no_separation'}"
            )


def create_visualisations(
    config: Dict[str, Any],
    analysis_type: str = "query-functions",
):
    """
    Create all visualisations based on analysis type

    Args:
        * config: Analysis configuration dictionary
        * analysis_type: "top-functions", "query-functions", or
            "rankings"
    """
    if config.skip_visualization:
        logger.processing(
            "Skipping visualizations as per configuration"
        )
        return

    if analysis_type == "top-functions":
        create_similarity_visualizations(
            config=config, analysis_type=analysis_type
        )
    elif analysis_type == "query-functions":
        create_probability_visualizations(
            config=config, analysis_type=analysis_type
        )
    elif analysis_type == "rankings":
        create_ranking_visualizations(
            config=config, analysis_type=analysis_type
        )
    else:
        logger.error_status(
            f"Unknown analysis type for visualization: "
            f"{analysis_type}",
            exc_info=True,
        )
        raise ValueError(
            f"Unknown analysis type: {analysis_type}"
        )

    # Retest consistency charts for functions
    if (
        getattr(config, "retest", 1) > 1
        and analysis_type == "top-functions"
    ):
        create_consistency_visualizations(
            config=config, analysis_type=analysis_type
        )
