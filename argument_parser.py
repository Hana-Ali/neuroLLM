import argparse


def parse_args():
    """Argument parser for brain analysis script"""
    parser = argparse.ArgumentParser(description="Brain Analysis with LLMs")

    # Main command - what to do
    parser.add_argument(
        "command", choices=["functions", "probabilities", "test"]
    )

    # Essential options
    parser.add_argument(
        "--species", default="human", choices=["human", "macaque", "mouse"]
    )
    parser.add_argument(
        "--regions",
        default=None,
        help="Comma-separated brain regions (default: all regions in atlas)",
    )
    parser.add_argument(
        "--models",
        default="dummy",
        help="'paid', 'dummy', 'all', or model names",
    )

    # Functional options
    parser.add_argument(
        "--functions",
        help="Comma-separated functions (for probabilities command)",
    )
    parser.add_argument(
        "--function-group",
        help="Predefined function group to load (for probabilities command)",
    )
    parser.add_argument("--atlas-name", help="Atlas name", required=True)

    # Prompt options
    parser.add_argument(
        "--prompt-template-name",
        default="default",
        help="Prompt template name",
    )
    parser.add_argument(
        "--separate-hemispheres",
        action="store_true",
        help="Skip hemisphere separation",
    )

    # Skipping options
    parser.add_argument(
        "--skip-visualization",
        action="store_true",
        help="Skip visualizations",
    )
    parser.add_argument(
        "--skip-raw-saving",
        action="store_true",
        help="Clean up raw data files",
    )

    # Others
    parser.add_argument("--workers", type=int, default=4)

    return parser.parse_args()
