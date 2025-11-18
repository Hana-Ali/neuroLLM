from utils.paths.base import BasePathConstructor


class VisualizationPathConstructor(BasePathConstructor):
    """Handles visualization path construction"""

    @staticmethod
    def get_visualisations_dir():
        """
        Get visualizations directory

        Returns:
            * Path to visualizations directory
        """
        return "results/visualizations"

    @classmethod
    def construct_visualisations_similarity_path(
        cls,
        model: str,
        species: str,
        atlas_name: str,
        hemisphere: str = None,
        template_name: str = "default",
        extension: str = "png",
    ):
        """
        Construct path for saving a similarity matrix visualization

        Args:
            * model: Model used to generate the analysis
            * species: Species used for the analysis
            * atlas_name: Name of the atlas
            * hemisphere: Hemisphere used, whether separated or not
            * template_name: Name of the template chosen (default: "default")
            * extension: File extension for the visualization (default: "png")

        Returns:
            * Path to the similarity matrix visualization file
        """
        hemisphere = cls.get_hemisphere_path(hemisphere=hemisphere)
        viz_dir = cls.get_visualisations_dir()
        base_dir = f"{viz_dir}/similarities"

        full_dir = (
            f"{base_dir}/{species}/{atlas_name}/{model}/"
            f"{template_name}/{hemisphere}"
        )
        return f"{full_dir}/similarity_matrix.{extension}"

    @classmethod
    def construct_visualisations_probability_path(
        cls,
        model: str,
        species: str,
        atlas_name: str,
        hemisphere: str = None,
        template_name: str = "default",
        extension: str = "png",
    ):
        """
        Construct path for saving a probability distribution visualization

        Args:
            * model: Model used to generate the analysis
            * species: Species used for the analysis
            * atlas_name: Name of the atlas
            * hemisphere: Hemisphere used, whether separated or not
            * template_name: Name of the template chosen (default: "default")
            * extension: File extension for the visualization (default: "png")

        Returns:
            * Path to the probability distribution visualization file
        """
        hemisphere = cls.get_hemisphere_path(hemisphere=hemisphere)
        viz_dir = cls.get_visualisations_dir()
        base_dir = f"{viz_dir}/probabilities"

        full_dir = (
            f"{base_dir}/{species}/{atlas_name}/{model}/"
            f"{template_name}/{hemisphere}"
        )
        return f"{full_dir}/heatmap.{extension}"

    @classmethod
    def construct_visualisations_function_path(
        cls,
        model: str,
        function: str,
        species: str,
        atlas_name: str,
        hemisphere: str = None,
        template_name: str = "default",
        extension: str = "png",
    ):
        """
        Construct path for saving a probability distribution visualization for
        a specific function

        Args:
            * function: Function name
            * model: Model used to generate the analysis
            * species: Species used for the analysis
            * atlas_name: Name of the atlas
            * hemisphere: Hemisphere used, whether separated or not
            * template_name: Name of the template chosen (default: "default")
            * extension: File extension for the visualization (default: "png")

        Returns:
            * Path to the probability distribution visualization file for the
                specific function
        """
        hemisphere = cls.get_hemisphere_path(hemisphere=hemisphere)
        viz_dir = cls.get_visualisations_dir()
        base_dir = f"{viz_dir}/probabilities"

        full_dir = (
            f"{base_dir}/{species}/{atlas_name}/{model}/"
            f"{template_name}/{hemisphere}/{function.replace(' ', '_')}"
        )
        return f"{full_dir}/heatmap.{extension}"
