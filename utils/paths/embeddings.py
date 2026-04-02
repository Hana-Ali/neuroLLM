from utils.paths.base import BasePathConstructor
from utils.paths.query import QueryPathConstructor


class EmbeddingsPathConstructor(BasePathConstructor):
    """Handles embeddings path construction"""

    def construct_embeddings_dir(self):
        """
        Construct base directory for embeddings

        Returns:
            * Path to the embeddings dir
        """
        base_dir = "results/embeddings"

        return (
            f"{base_dir}/{self.analysis_type}/{self.species}/"
            f"{self._atlas_segment}/{self.model}/"
            f"{self.template_name}/{self._hemisphere_segment}"
        )

    def construct_embeddings_region_path(
        self, region: str, trial="final",
    ):
        """
        Construct path for a region embedding file
        (combined embedding: single row, one vector per region)

        Args:
            * region: Brain region name
            * trial: int for a specific trial, or "final"

        Returns:
            * Path to the region embedding csv file
        """
        embeddings_dir = self.construct_embeddings_dir()
        segment = QueryPathConstructor._trial_segment(trial)
        return f"{embeddings_dir}/{segment}/{region}.csv"

    def construct_per_function_embeddings_region_path(
        self, region: str, trial="final",
    ):
        """
        Construct path for per-function embeddings file
        (multi-row CSV: one row per function, indexed by function name)

        Args:
            * region: Brain region name
            * trial: int for a specific trial, or "final"

        Returns:
            * Path to the per-function embeddings csv file
        """
        embeddings_dir = self.construct_embeddings_dir()
        segment = QueryPathConstructor._trial_segment(trial)
        return f"{embeddings_dir}/{segment}/{region}_per_function.csv"
