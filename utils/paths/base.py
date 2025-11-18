from abc import ABC

DEFAULT_PATHS = {
    "prompts": {
        "base": "prompts",
        "functions": "prompts/functions",
        "probabilities": "prompts/probabilities",
    },
    "atlas": "./atlases",
    "env_file": ".env",
}


class BasePathConstructor(ABC):
    """Base class for all path constructors"""

    @staticmethod
    def get_raw_results_dir():
        """
        Get raw results directory

        Returns:
            * Path to raw results directory
        """
        return "results/raw"

    @staticmethod
    def get_hemisphere_path(hemisphere: str = None):
        """
        Construct hemisphere path based on provided value

        Args:
            * hemisphere: Hemisphere used, whether separated or not

        Returns:
            * Path segment for hemisphere
        """
        return f"separation/{hemisphere}" if hemisphere else "no_separation"

    @staticmethod
    def cleanup_raw_dir():
        """
        Cleanup raw results directory

        Returns:
            * None
        """
        import shutil
        import os

        raw_dir = BasePathConstructor.get_raw_results_dir()
        if os.path.exists(raw_dir):
            shutil.rmtree(raw_dir)
