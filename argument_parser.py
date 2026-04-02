import argparse


def parse_args():
    """Argument parser for brain analysis script"""
    parser = argparse.ArgumentParser(description="Brain Analysis with LLMs")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Shared arguments for analysis commands
    analysis_parent = argparse.ArgumentParser(add_help=False)
    analysis_parent.add_argument(
        "--species",
        default=None,
        choices=["human", "macaque", "mouse"],
        help="Species to analyze (required)",
    )
    analysis_parent.add_argument(
        "--regions",
        default=None,
        help="Comma-separated brain regions (default: all regions in atlas)",
    )
    analysis_parent.add_argument(
        "--models",
        default="dummy",
        help="Comma-separated OpenRouter model IDs (e.g., "
        "'openai/gpt-4o-mini,anthropic/claude-3.5-sonnet'), "
        "'braingpt', or 'dummy'. Use 'list-models' command to see "
        "available models",
    )
    analysis_parent.add_argument(
        "--prompt-template-name",
        default="default",
        help="Prompt template name",
    )
    analysis_parent.add_argument(
        "--separate-hemispheres",
        action="store_true",
        help="Process left and right hemispheres separately",
    )
    analysis_parent.add_argument(
        "--skip-visualization",
        action="store_true",
        help="Skip visualizations",
    )
    analysis_parent.add_argument(
        "--skip-raw-saving",
        action="store_true",
        help="Clean up raw data files",
    )
    analysis_parent.add_argument("--workers", type=int, default=4)
    analysis_parent.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help=(
            "Maximum tokens per response (default: 512 with --justify, 256 "
            "otherwise"
        ),
    )
    analysis_parent.add_argument(
        "--justify",
        action="store_true",
        help="Ask model to provide a justification alongside its answer",
    )
    analysis_parent.add_argument(
        "--retest",
        type=int,
        default=1,
        help=(
            "Number of times to repeat each query and average results "
            "(default: 1, no retest)"
        ),
    )
    analysis_parent.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for model querying (default 0.0)",
    )

    # List models
    list_models_parser = subparsers.add_parser(
        "list-models", help="List available models from OpenRouter"
    )
    list_models_parser.add_argument(
        "--filter",
        default="all",
        choices=["all", "free", "paid"],
        help=(
            "Filter models by pricing: 'all' (default), 'free', or 'paid'"
            "(sorted cheapest first per provider)"
        ),
    )

    # Top functions - inherits shared arguments
    top_functions_parser = subparsers.add_parser(
        "top-functions",
        parents=[analysis_parent],
        help="Run top functions analysis",
    )
    top_functions_parser.add_argument(
        "--atlas-name",
        default=None,
        help="Atlas name (required when --regions is not provided)",
    )
    top_functions_parser.add_argument(
        "--embedding-provider",
        default="openai",
        choices=["openai", "local"],
        help="Embedding provider for top-functions analysis. "
        "'openai' uses text-embedding-3-large (requires OPENAI_API_KEY). "
        "'local' uses BAAI/bge-large-en-v1.5 via sentence-transformers "
        "(no API key needed, runs on your machine)",
    )
    top_functions_parser.add_argument(
        "--consensus-threshold",
        type=float,
        default=0.80,
        help=(
            "Cosine similarity threshold for semantic clustering when "
            "computing consensus functions across retests (default: 0.80)"
        ),
    )

    # Query functions - inherits shared arguments and adds function selection
    query_parser = subparsers.add_parser(
        "query-functions",
        parents=[analysis_parent],
        help="Run query functions analysis",
    )
    query_parser.add_argument(
        "--atlas-name",
        default=None,
        help="Atlas name (required when --regions is not provided)",
    )
    query_parser.add_argument(
        "--functions",
        help="Comma-separated functions",
    )
    query_parser.add_argument(
        "--function-group",
        help="Predefined function group to load",
    )

    # Rank pairs - inherits shared arguments and adds pair selection
    rank_parser = subparsers.add_parser(
        "rank-pairs",
        parents=[analysis_parent],
        help="Rank which of two brain regions is more relevant "
        "to a function",
    )
    rank_parser.add_argument(
        "--atlas-name",
        default=None,
        help=(
            "Atlas name (required when --regions and --pairs are not provided)"
        ),
    )
    rank_parser.add_argument(
        "--functions",
        help="Comma-separated functions to rank pairs for",
    )
    rank_parser.add_argument(
        "--function-group",
        help="Predefined function group to load",
    )
    rank_parser.add_argument(
        "--pairs",
        help="Region pairs as 'region1:region2,region3:region4'. "
        "If omitted, generates all unique pairs from atlas",
    )

    test_parser = subparsers.add_parser(
        "test",
        parents=[analysis_parent],
        help="Run test workflow",
    )
    test_parser.add_argument(
        "--atlas-name",
        default=None,
        help="Atlas name (required when --regions is not provided)",
    )

    return parser.parse_args()
