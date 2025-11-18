from utils.paths.base import BasePathConstructor


class QueryPathConstructor(BasePathConstructor):
    """Handles query results path construction"""

    @classmethod
    def construct_query_results_dir(
        cls,
        model: str,
        species: str,
        atlas_name: str,
        analysis_type: str,
        hemisphere: str = None,
        template_name: str = "default",
    ):
        """
        Construct path for saving query results

        Args:
            * model: Model used to generate the analysis
            * species: Species used for the analysis
            * atlas_name: Name of the atlas
            * analysis_type: Type of analysis ("function" or "probability")
            * hemisphere: Hemisphere used, whether separated or not
            * template_name: Name of the template chosen (default: "default")

        Returns:
            * Path to the raw query results dir
        """
        hemisphere = cls.get_hemisphere_path(hemisphere=hemisphere)
        raw_dir = cls.get_raw_results_dir()

        return (
            f"{raw_dir}/{analysis_type}/{species}/{atlas_name}/{model}/"
            f"{template_name}/{hemisphere}"
        )

    @classmethod
    def construct_query_region_path(
        cls,
        model: str,
        region: str,
        species: str,
        atlas_name: str,
        analysis_type: str,
        hemisphere: str = None,
        template_name: str = "default",
    ):
        """
        Construct path for saving a region query

        Args:
            * region: Brain region name
            * model: Model used to generate the analysis
            * species: Species used for the analysis
            * atlas_name: Name of the atlas
            * analysis_type: Type of analysis ("function" or "probability")
            * hemisphere: Hemisphere used, whether separated or not
            * template_name: Name of the template chosen (default: "default")

        Returns:
            * Path to the json region query
        """
        query_dir = cls.construct_query_results_dir(
            model=model,
            species=species,
            atlas_name=atlas_name,
            analysis_type=analysis_type,
            hemisphere=hemisphere,
            template_name=template_name,
        )
        return f"{query_dir}/{region}.json"

    @classmethod
    def construct_query_cleaned_results_dir(
        cls,
        model: str,
        species: str,
        atlas_name: str,
        analysis_type: str,
        hemisphere: str = None,
        template_name: str = "default",
    ):
        """
        Construct path for saving cleaned query results

        Args:
            * model: Model used to generate the analysis
            * species: Species used for the analysis
            * atlas_name: Name of the atlas
            * analysis_type: Type of analysis ("function" or "probability")
            * hemisphere: Hemisphere used, whether separated or not
            * template_name: Name of the template chosen (default: "default")

        Returns:
            * Path to the cleaned query results dir
        """
        query_dir = cls.construct_query_results_dir(
            model=model,
            species=species,
            atlas_name=atlas_name,
            analysis_type=analysis_type,
            hemisphere=hemisphere,
            template_name=template_name,
        )
        return f"{query_dir}/cleaned"

    @classmethod
    def construct_query_cleaned_region_path(
        cls,
        model: str,
        region: str,
        species: str,
        atlas_name: str,
        analysis_type: str,
        hemisphere: str = None,
        template_name: str = "default",
    ):
        """
        Construct path for saving a cleaned region query

        Args:
            * region: Brain region name
            * model: Model used to generate the analysis
            * species: Species used for the analysis
            * atlas_name: Name of the atlas
            * analysis_type: Type of analysis ("function" or "probability")
            * hemisphere: Hemisphere used, whether separated or not
            * template_name: Name of the template chosen (default: "default")

        Returns:
            * Path to the cleaned json region query
        """
        query_dir = cls.construct_query_cleaned_results_dir(
            model=model,
            species=species,
            atlas_name=atlas_name,
            analysis_type=analysis_type,
            hemisphere=hemisphere,
            template_name=template_name,
        )
        return f"{query_dir}/{region}/{model}.json"

    @classmethod
    def construct_query_combined_cleaned_results_dir(
        cls,
        model: str,
        species: str,
        atlas_name: str,
        analysis_type: str,
        hemisphere: str = None,
        template_name: str = "default",
    ):
        """
        Construct path for saving combined cleaned query results

        Args:
            * model: Model used to generate the analysis
            * species: Species used for the analysis
            * atlas_name: Name of the atlas
            * analysis_type: Type of analysis ("function" or "probability")
            * hemisphere: Hemisphere used, whether separated or not
            * template_name: Name of the template chosen (default: "default")

        Returns:
            * Path to the combined cleaned query results dir
        """
        query_dir = cls.construct_query_cleaned_results_dir(
            model=model,
            species=species,
            atlas_name=atlas_name,
            analysis_type=analysis_type,
            hemisphere=hemisphere,
            template_name=template_name,
        )
        return f"{query_dir}/combined"
