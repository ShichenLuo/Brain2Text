"""Small command-line entry point for discovering the research pipeline."""

from argparse import ArgumentParser

from .paths import PROJECT_ROOT


def main() -> None:
    parser = ArgumentParser(description="Brain2Text project utilities")
    parser.add_argument(
        "--show-root",
        action="store_true",
        help="Print the detected project root.",
    )
    args = parser.parse_args()
    if args.show_root:
        print(PROJECT_ROOT)
    else:
        parser.print_help()