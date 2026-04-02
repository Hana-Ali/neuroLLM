from utils.paths.base import BasePathConstructor


class VisualizationPathConstructor(BasePathConstructor):
    """Handles visualization path construction"""

    @property
    def visualisations_dir(self):
        return "results/visualizations"

    @property
    def base_dir(self):
        return (
            f"{self.visualisations_dir}/{self.analysis_type}/{self.species}/"
            f"{self._atlas_segment}/{self.model}/{self.template_name}/"
            f"{self._hemisphere_segment}"
        )

    def construct_visualisations_similarity_path(
        self, extension: str = "png",
    ):
        """
        Construct path for saving a similarity matrix visualization

        Args:
            * extension: File extension (default: "png")

        Returns:
            * Path to the similarity matrix visualization file
        """
        return f"{self.base_dir}/similarity_matrix.{extension}"

    def construct_visualisations_probability_path(
        self, extension: str = "png",
    ):
        """
        Construct path for saving a probability distribution
        visualization

        Args:
            * extension: File extension (default: "png")

        Returns:
            * Path to the probability distribution visualization
        """
        return f"{self.base_dir}/heatmap.{extension}"

    def construct_visualisations_function_path(
        self, function: str, extension: str = "png",
    ):
        """
        Construct path for saving a probability distribution
        visualization for a specific function

        Args:
            * function: Function name
            * extension: File extension (default: "png")

        Returns:
            * Path to the function probability visualization
        """
        return f"{self.base_dir}/heatmap.{extension}"

    def construct_visualisations_ranking_pair_path(
        self,
        region_1: str,
        region_2: str,
        extension: str = "png",
    ):
        """
        Construct path for a per-pair ranking comparison chart

        Args:
            * region_1: First brain region name
            * region_2: Second brain region name
            * extension: File extension (default: "png")

        Returns:
            * Path to the pair comparison visualization file
        """
        return (
            f"{self.base_dir}/{region_1}_vs_{region_2}/comparison.{extension}"
        )

    def construct_visualisations_consistency_path(
        self, extension: str = "png",
    ):
        """
        Construct path for retest consistency visualization

        Args:
            * extension: File extension (default: "png")

        Returns:
            * Path to the consistency chart file
        """
        return f"{self.base_dir}/consistency_scores.{extension}"
