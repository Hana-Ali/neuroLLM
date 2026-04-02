from abc import ABC

DEFAULT_PATHS = {
    "prompts": {
        "base": "prompts",
        "top-functions": "prompts/functions",
        "query-functions": "prompts/probabilities",
        "rankings": "prompts/rankings",
    },
    "atlas": "./atlases",
    "env_file": ".env",
}


class BasePathConstructor(ABC):
    """Base class for all path constructors"""

    def __init__(
        self,
        model: str = None,
        species: str = None,
        atlas_name: str = None,
        analysis_type: str = None,
        hemisphere: str = None,
        template_name: str = "default",
    ):
        self.model = model
        self.species = species
        self.atlas_name = atlas_name
        self.analysis_type = analysis_type
        self.hemisphere = hemisphere
        self.template_name = template_name

        # Pre-compute common path segments
        self._hemisphere_segment = (
            f"separation/{hemisphere}" if hemisphere else "no_separation"
        )
        self._atlas_segment = atlas_name if atlas_name else "no_atlas"

    @staticmethod
    def get_raw_results_dir():
        """
        Get raw results directory

        Returns:
            * Path to raw results directory
        """
        return "results/raw"

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
