import datetime
from utils.paths.base import BasePathConstructor, DEFAULT_PATHS


class PromptPathConstructor(BasePathConstructor):
    """Handles prompt and template-related path construction"""

    @staticmethod
    def get_prompt_dir(prompt_type: str = None):
        """
        Get directory for prompt type

        Args:
            * prompt_type: Type of analysis ("functions" or "probabilities")

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
            * prompt_type: Type of analysis ("functions" or "probabilities")

        Returns:
            * Path to the results prompt directory, according to analysis type
        """
        return f"results/{cls.get_prompt_dir(prompt_type=prompt_type)}"

    @classmethod
    def construct_template_path(
        cls, prompt_type: str, template_name: str = "default"
    ):
        """
        Construct path to default prompt template

        Args:
            * prompt_type: Type of analysis ("functions" or "probabilities")
            * template_name: Name of the template chosen (default: "default")

        Returns:
            * Path to the chosen prompt template
        """
        prompt_dir = cls.get_prompt_dir(prompt_type=prompt_type)
        return f"{prompt_dir}/{template_name}.txt"

    @classmethod
    def construct_results_prompt_path(
        cls,
        species: str,
        atlas_name: str,
        prompt_type: str,
        hemisphere: str = None,
        template_name: str = "default",
    ):
        """
        Construct path for saving generated prompts to results

        Args:
            * species: Species used for the analysis
            * atlas_name: Name of the atlas
            * prompt_type: Type of analysis ("functions" or "probabilities")
            * hemisphere: Hemisphere used, whether separated or not
            * template_name: Name of the template chosen (default: "default")

        Returns:
            * Path to the generated prompt
        """
        hemisphere = cls.get_hemisphere_path(hemisphere=hemisphere)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        results_dir = cls.get_results_prompt_dir(prompt_type=prompt_type)
        return (
            f"{results_dir}/{species}/{atlas_name}/{template_name}/"
            f"{hemisphere}/prompt_{timestamp}.txt"
        )
