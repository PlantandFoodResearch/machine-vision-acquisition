import click
import importlib
import logging
from typing import Optional
from pathlib import Path


@click.command()
@click.option(
    "--model",
    "-m",
    help="The python class object to export as a schema. Must derive from pydantic.BaseModel",
    required=True,
    default="machine_vision_acquisition_python.models.Config",
)
@click.option(
    "--output",
    "-o",
    "output_path",
    help="Path to JSON configuration file for capture",
    required=False,
    type=click.types.Path(
        file_okay=True, dir_okay=False, writable=True, path_type=Path
    ),
)
def cli(model: str, output_path: Optional[Path]):
    try:
        parts = model.rsplit(".", 1)
        if len(parts) != 2:
            raise ValueError(f"Could not split {model} into package and class")
        library = parts[0]
        target_class_str = parts[1]
        logging.info(f"Attempting import of {target_class_str} from {library}")
        module = importlib.import_module(library)
        target_class = getattr(module, target_class_str)

        # Construct default output name if not given
        if not output_path:
            output_path = Path.cwd() / f"{target_class_str.lower()}.schema.json"

        schema_text = target_class.schema_json(indent=2)
        output_path.write_text(schema_text)
    except ImportError as exc:
        logging.exception(f"Failed to import {model}")
        raise


if __name__ == "__main__":
    cli()
