import click
from pathlib import Path

@click.command()
@click.option(
    "--config",
    "-c",
    help="Path to YAML configuration file for capture",
    required=True,
    type=click.types.Path(file_okay=True, exists=True, dir_okay=False, readable=True, path_type=Path)
)
def cli(config: Path):
    pass