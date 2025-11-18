from utils.paths.base import BasePathConstructor


class EmbeddingsPathConstructor(BasePathConstructor):
    """Handles embeddings path construction"""

    @classmethod
    def construct_embeddings_dir(
        cls,
        model: str,
        species: str,
        atlas_name: str,
        analysis_type: str,
        hemisphere: str = None,
        template_name: str = "default",
    ):
        """
        Construct path for saving embeddings

        Args:
            * model: Model used to generate the analysis
            * species: Species used for the analysis
            * atlas_name: Name of the atlas
            * analysis_type: Type of analysis ("function" or "probability")
            * hemisphere: Hemisphere used, whether separated or not
            * template_name: Name of the template chosen (default: "default")

        Returns:
            * Path to the embeddings dir
        """
        hemisphere = cls.get_hemisphere_path(hemisphere=hemisphere)
        base_dir = "results/embeddings"

        return (
            f"{base_dir}/{analysis_type}/{species}/{atlas_name}/{model}/"
            f"{template_name}/{hemisphere}"
        )

    @classmethod
    def construct_embeddings_region_path(
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
        Construct path for saving a region embedding

        Args:
            * region: Brain region name
            * model: Model used to generate the analysis
            * species: Species used for the analysis
            * atlas_name: Name of the atlas
            * analysis_type: Type of analysis ("function" or "probability")
            * hemisphere: Hemisphere used, whether separated or not
            * template_name: Name of the template chosen (default: "default")

        Returns:
            * Path to the region embedding file
        """
        embeddings_dir = cls.construct_embeddings_dir(
            model=model,
            species=species,
            atlas_name=atlas_name,
            analysis_type=analysis_type,
            hemisphere=hemisphere,
            template_name=template_name,
        )
        return f"{embeddings_dir}/{region}.csv"
