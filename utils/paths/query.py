from utils.paths.base import BasePathConstructor


class QueryPathConstructor(BasePathConstructor):
    """Handles query results path construction"""

    @staticmethod
    def _trial_segment(trial):
        """
        Return the path segment for a trial or final result.

        Args:
            * trial: int for a specific trial (0, 1, 2, ...),
                or "final" for the canonical/averaged result

        Returns:
            * "trial_0", "trial_1", ..., or "final"
        """
        return (
            "final"
            if trial == "final"
            else f"trial_{trial}"
        )

    def construct_query_results_dir(self):
        """
        Construct base directory for query results

        Returns:
            * Path to the raw query results dir
        """
        raw_dir = self.get_raw_results_dir()

        return (
            f"{raw_dir}/{self.analysis_type}/{self.species}/"
            f"{self._atlas_segment}/{self.model}/"
            f"{self.template_name}/{self._hemisphere_segment}"
        )

    def construct_query_cleaned_results_dir(self):
        """
        Construct path for saving cleaned query results

        Returns:
            * Path to the cleaned query results dir
        """
        query_dir = self.construct_query_results_dir()
        return f"{query_dir}/cleaned"

    def construct_query_trials_dir(self, trial="final"):
        """
        Construct directory for a specific trial or final query results

        Args:
            * trial: int for a specific trial, or "final"

        Returns:
            * Path to the trial-specific query results dir
        """
        query_dir = self.construct_query_results_dir()
        segment = self._trial_segment(trial=trial)
        return f"{query_dir}/{segment}"

    def construct_query_justifications_dir(self, trial="final"):
        """
        Construct directory for justifications of a specific trial or final
        query results

        Args:
            * trial: int for a specific trial, or "final"

        Returns:
            * Path to the trial-specific justifications dir
        """
        query_dir = self.construct_query_results_dir()
        segment = self._trial_segment(trial=trial)
        return f"{query_dir}/justifications/{segment}"

    def construct_query_region_path(
        self, region: str, trial="final",
    ):
        """
        Construct path for a region query result

        Args:
            * region: Brain region name
            * trial: int for a specific trial, or "final"

        Returns:
            * Path to the region json file
        """
        return f"{self.construct_query_trials_dir(trial=trial)}/{region}.json"

    def construct_query_cleaned_region_path(
        self, region: str, trial="final",
    ):
        """
        Construct path for a cleaned region query result

        Args:
            * region: Brain region name
            * trial: int for a specific trial, or "final"

        Returns:
            * Path to the cleaned json region query
        """
        return (
            f"{self.construct_query_cleaned_results_dir()}/"
            f"{self._trial_segment(trial=trial)}/{region}/"
            f"{self.model}.json"
        )

    def construct_query_justification_region_path(
        self, region: str, trial="final",
    ):
        """
        Construct path for a region justification file

        Args:
            * region: Brain region name
            * trial: int for a specific trial, or "final"

        Returns:
            * Path to the justification json file
        """
        return (
            f"{self.construct_query_justifications_dir(trial=trial)}/"
            f"{region}.json"
        )

    def construct_query_pair_path(
        self,
        region_1: str,
        region_2: str,
        trial="final",
    ):
        """
        Construct path for a ranking pair query result

        Args:
            * region_1: First brain region name
            * region_2: Second brain region name
            * trial: int for a specific trial, or "final"

        Returns:
            * Path to the pair ranking json file
        """
        pair = f"{region_1}_vs_{region_2}"
        return f"{self.construct_query_trials_dir(trial=trial)}/{pair}.json"

    def construct_query_pair_justification_path(
        self,
        region_1: str,
        region_2: str,
        trial="final",
    ):
        """
        Construct path for a pair justification file

        Args:
            * region_1: First brain region name
            * region_2: Second brain region name
            * trial: int for a specific trial, or "final"

        Returns:
            * Path to the pair justification json file
        """
        pair = f"{region_1}_vs_{region_2}"
        return (
            f"{self.construct_query_justifications_dir(trial=trial)}/"
            f"{pair}.json"
        )

    def construct_query_retest_summary_path(
        self, region: str,
    ):
        """
        Construct path for saving retest summary

        Args:
            * region: Brain region name

        Returns:
            * Path to the retest summary json file
        """
        query_dir = self.construct_query_results_dir()
        return f"{query_dir}/retest_summary/{region}.json"
