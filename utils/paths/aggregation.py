from utils.paths.base import BasePathConstructor


class AggregatedResultsPathConstructor(BasePathConstructor):
    """Handles aggregated results path construction"""

    @property
    def aggregated_query_results_dir(self):
        """
        Construct path for saving aggregated query results

        Returns:
            * Path to the aggregated query results dir
        """
        base_dir = f"results/aggregated/{self.analysis_type}"

        return (
            f"{base_dir}/{self.species}/{self._atlas_segment}/"
            f"{self.model}/{self.template_name}/"
            f"{self._hemisphere_segment}"
        )

    def construct_aggregated_query_results_path(
        self, extension: str = "json",
    ):
        """
        Construct path for saving an aggregated query

        Args:
            * extension: File extension (default: "json")

        Returns:
            * Path to the aggregated json region query
        """
        aggregated_dir = self.aggregated_query_results_dir
        return f"{aggregated_dir}/results_distribution.{extension}"

    def construct_individual_function_prob_path(
        self, function: str,
    ):
        """
        Construct path for saving an individual function probabilities CSV

        Args:
            * function: Function name

        Returns:
            * Path to the individual function probabilities CSV file
        """
        aggregated_dir = self.aggregated_query_results_dir
        func_dir = (
            f"{aggregated_dir}/{function.replace(' ', '_')}"
        )
        return f"{func_dir}/probabilities.csv"

    def construct_aggregated_embeddings_path(self):
        """
        Construct path for saving an aggregated embeddings CSV

        Returns:
            * Path to the aggregated embeddings CSV file
        """
        aggregated_dir = self.aggregated_query_results_dir
        return f"{aggregated_dir}/all_embeddings.csv"

    def construct_aggregated_justification_path(self):
        """
        Construct path for saving aggregated justifications

        Returns:
            * Path to the justification distribution JSON
        """
        aggregated_dir = self.aggregated_query_results_dir
        return f"{aggregated_dir}/justification_distribution.json"

    def construct_aggregated_pair_results_path(
        self, region_1: str, region_2: str,
    ):
        """
        Construct path for saving aggregated results for a single pair

        Args:
            * region_1: First brain region name
            * region_2: Second brain region name

        Returns:
            * Path to the pair results CSV file
        """
        aggregated_dir = self.aggregated_query_results_dir
        pair_dir = f"{region_1}_vs_{region_2}"
        return f"{aggregated_dir}/{pair_dir}/results.csv"

    def construct_aggregated_retest_stats_path(
        self, filename: str = "retest_statistics.csv",
    ):
        """
        Construct path for aggregated retest statistics

        Args:
            * filename: Output filename (default: "retest_statistics.csv")

        Returns:
            * Path to the retest statistics CSV
        """
        aggregated_dir = self.aggregated_query_results_dir
        return f"{aggregated_dir}/{filename}"
