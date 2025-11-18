from utils.paths.base import BasePathConstructor


class AggregatedResultsPathConstructor(BasePathConstructor):
    """Handles aggregated results path construction"""

    @classmethod
    def construct_aggregated_query_results_dir(
        cls,
        model: str,
        species: str,
        atlas_name: str,
        analysis_type: str,
        hemisphere: str = None,
        template_name: str = "default",
    ):
        """
        Construct path for saving aggregated query results

        Args:
            * model: Model used to generate the analysis
            * species: Species used for the analysis
            * atlas_name: Name of the atlas
            * analysis_type: Type of analysis ("function" or "probability")
            * hemisphere: Hemisphere used, whether separated or not
            * template_name: Name of the template chosen (default: "default")

        Returns:
            * Path to the aggregated query results dir
        """
        hemisphere = cls.get_hemisphere_path(hemisphere=hemisphere)
        base_dir = f"results/aggregated/{analysis_type}"

        return (
            f"{base_dir}/{species}/{atlas_name}/{model}/"
            f"{template_name}/{hemisphere}"
        )

    @classmethod
    def construct_aggregated_query_results_path(
        cls,
        model: str,
        species: str,
        atlas_name: str,
        analysis_type: str,
        hemisphere: str = None,
        template_name: str = "default",
        extension: str = "json",
    ):
        """
        Construct path for saving an aggregated query

        Args:
            * model: Model used to generate the analysis
            * species: Species used for the analysis
            * atlas_name: Name of the atlas
            * analysis_type: Type of analysis ("function" or "probability")
            * hemisphere: Hemisphere used, whether separated or not
            * template_name: Name of the template chosen (default: "default")

        Returns:
            * Path to the aggregated json region query
        """
        aggregated_dir = cls.construct_aggregated_query_results_dir(
            model=model,
            species=species,
            atlas_name=atlas_name,
            analysis_type=analysis_type,
            hemisphere=hemisphere,
            template_name=template_name,
        )
        return f"{aggregated_dir}/probability_distribution.{extension}"

    @classmethod
    def construct_individual_function_prob_path(
        cls,
        model: str,
        function: str,
        species: str,
        atlas_name: str,
        analysis_type: str,
        hemisphere: str = None,
        template_name: str = "default",
    ):
        """
        Construct path for saving an individual function probabilities CSV

        Args:
            * function: Function name
            * model: Model used to generate the analysis
            * species: Species used for the analysis
            * atlas_name: Name of the atlas
            * analysis_type: Type of analysis ("function" or "probability")
            * hemisphere: Hemisphere used, whether separated or not
            * template_name: Name of the template chosen (default: "default")

        Returns:
            * Path to the individual function probabilities CSV file
        """
        aggregated_dir = cls.construct_aggregated_query_results_dir(
            model=model,
            species=species,
            atlas_name=atlas_name,
            analysis_type=analysis_type,
            hemisphere=hemisphere,
            template_name=template_name,
        )
        func_dir = f"{aggregated_dir}/{function.replace(' ', '_')}"
        return f"{func_dir}/probabilities.csv"

    @classmethod
    def construct_aggregated_embeddings_path(
        cls,
        model: str,
        species: str,
        atlas_name: str,
        analysis_type: str,
        hemisphere: str = None,
        template_name: str = "default",
    ):
        """
        Construct path for saving an aggregated embeddings CSV

        Args:
            * model: Model used to generate the analysis
            * species: Species used for the analysis
            * atlas_name: Name of the atlas
            * analysis_type: Type of analysis ("function" or "probability")
            * hemisphere: Hemisphere used, whether separated or not
            * template_name: Name of the template chosen (default: "default")

        Returns:
            * Path to the aggregated embeddings CSV file
        """
        aggregated_dir = cls.construct_aggregated_query_results_dir(
            model=model,
            species=species,
            atlas_name=atlas_name,
            analysis_type=analysis_type,
            hemisphere=hemisphere,
            template_name=template_name,
        )
        return f"{aggregated_dir}/all_embeddings.csv"
