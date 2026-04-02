import datetime
from utils.paths.base import BasePathConstructor, DEFAULT_PATHS


class PromptPathConstructor(BasePathConstructor):
    """Handles prompt and template-related path construction"""

    @staticmethod
    def get_prompt_dir(prompt_type: str = None):
        """
        Get directory for prompt type

        Args:
            * prompt_type: Type of analysis

        Returns:
            * Path to the prompt directory, according to analysis type
        """
        prompt_type = prompt_type if prompt_type else "base"
        return DEFAULT_PATHS["prompts"][prompt_type]

    @classmethod
    def get_results_prompt_dir(cls, prompt_type: str):
        """
        Get results directory for prompt type

        Args:
            * prompt_type: Type of analysis

        Returns:
            * Path to the results prompt directory
        """
        return f"results/{cls.get_prompt_dir(prompt_type=prompt_type)}"

    @classmethod
    def construct_template_path(
        cls, prompt_type: str, template_name: str = "default"
    ):
        """
        Construct path to default prompt template

        Args:
            * prompt_type: Type of analysis
            * template_name: Name of the template chosen (default: "default")

        Returns:
            * Path to the chosen prompt template
        """
        prompt_dir = cls.get_prompt_dir(prompt_type=prompt_type)
        return f"{prompt_dir}/{template_name}.txt"

    def construct_results_prompt_path(
        self, prompt_type: str,
    ):
        """
        Construct path for saving generated prompts to results

        Args:
            * prompt_type: Type of analysis

        Returns:
            * Path to the generated prompt
        """
        timestamp = datetime.datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )
        results_dir = self.get_results_prompt_dir(
            prompt_type=prompt_type
        )
        return (
            f"{results_dir}/{self.species}/"
            f"{self._atlas_segment}/{self.template_name}/"
            f"{self._hemisphere_segment}/prompt_{timestamp}.txt"
        )
