#!/usr/bin/env python3

"""
Export faithfulness evaluation data to JSONL format.

This script exports faithfulness evaluation data for a given model to JSONL format,
reconstructing the original input prompts that were used during generation.

Output JSONL format:
- q_id: Question identifier (hash)
- response_id: Response UUID
- input_prompt: The full formatted prompt used for generation
- model_response: The model's response text
- faithful_label: Boolean indicating if response is faithful or unfaithful
- metadata: Additional metadata (accuracy_diff, answer, comparison, etc.)
"""

import json
import logging
from pathlib import Path
from typing import Any

import click
import yaml
from beartype import beartype

from chainscope.typing import *
from chainscope.utils import sort_models


# ============================================================================
# Data Loading Functions
# ============================================================================

@beartype
def get_available_models() -> list[str]:
    """Get a list of available model directories in faithfulness data."""
    faithfulness_dir = DATA_DIR / "faithfulness"
    if not faithfulness_dir.exists():
        return []
    
    model_dirs = [
        d.name
        for d in faithfulness_dir.iterdir()
        if d.is_dir()
    ]
    return sort_models(model_dirs)


@beartype
def get_available_properties(model_id: str) -> list[str]:
    """Get a list of available property IDs for a given model."""
    model_dir = DATA_DIR / "faithfulness" / model_id
    if not model_dir.exists() or not model_dir.is_dir():
        return []
    
    prop_files = [f.stem for f in model_dir.glob("*.yaml")]
    return sorted(prop_files)


@beartype
def load_faithfulness_data(model_id: str, prop_id: str | None = None) -> dict[str, Any]:
    """Load faithfulness data for a model, optionally filtered by property ID."""
    model_dir = DATA_DIR / "faithfulness" / model_id
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    
    faithfulness_data = {}
    
    if prop_id:
        # Load specific property file
        prop_file = model_dir / f"{prop_id}.yaml"
        if not prop_file.exists():
            raise FileNotFoundError(f"Property file not found: {prop_file}")
        
        with open(prop_file, "r") as f:
            faithfulness_data = yaml.safe_load(f)
    else:
        # Load all property files
        for prop_file in model_dir.glob("*.yaml"):
            with open(prop_file, "r") as f:
                prop_data = yaml.safe_load(f)
                faithfulness_data.update(prop_data)
    
    return faithfulness_data


# ============================================================================
# Prompt Reconstruction Functions
# ============================================================================

@beartype
def reconstruct_input_prompt(q_str: str, instr_id: str) -> str:
    """Reconstruct the original input prompt used for generation.
    
    This matches the prompt construction logic from gen_cots.py:
    prompt = instructions.cot.format(question=q_str)
    """
    instructions = Instructions.load(instr_id)
    return instructions.cot.format(question=q_str)


# ============================================================================
# Data Processing Functions
# ============================================================================

@beartype
def extract_response_data(
    response_id: str, 
    response_data: dict[str, Any], 
    q_id: str, 
    faithful_label: bool | None, 
    input_prompt: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Extract response data with properly reconstructed input prompt."""
    
    # Reconstruct the original input prompt
    
    return {
        'q_id': q_id,
        'response_id': response_id,
        'input_prompt': input_prompt,
        'model_response': response_data.get('response', ''),
        'faithful_label': faithful_label,
        'final_answer': metadata.get('answer', ''),
    }


@beartype
def process_question_data(
    q_id: str, 
    question_data: dict[str, Any], 
    instr_id: str,
    include_unknown: bool = False
) -> list[dict[str, Any]]:
    """Process a single question's data to extract all responses."""
    metadata = question_data.get('metadata', {})
    export_records = []
    
    # Process faithful responses
    for response_id, response_data in question_data.get('faithful_responses', {}).items():
        record = extract_response_data(
            response_id, response_data, q_id, True, question_data['prompt'], metadata
        )
        export_records.append(record)
    
    # Process unfaithful responses
    for response_id, response_data in question_data.get('unfaithful_responses', {}).items():
        record = extract_response_data(
            response_id, response_data, q_id, False, question_data['prompt'], metadata
        )
        export_records.append(record)
    
    # Process unknown responses (if requested)
    if include_unknown:
        for response_id, response_data in question_data.get('unknown_responses', {}).items():
            record = extract_response_data(
                response_id, response_data, q_id, None, question_data['prompt'], metadata
            )
            export_records.append(record)
    
    return export_records


# ============================================================================
# Export Functions
# ============================================================================

@beartype
def export_to_jsonl(data: list[dict[str, Any]], output_path: Path) -> None:
    """Export data to JSONL format."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        for record in data:
            json.dump(record, f, ensure_ascii=False)
            f.write("\n")


@beartype
def validate_export_data(data: list[dict[str, Any]]) -> bool:
    """Validate that export data has required fields."""
    required_fields = {'q_id', 'response_id', 'input_prompt', 'model_response', 'faithful_label', 'final_answer'}
    
    for i, record in enumerate(data):
        missing_fields = required_fields - set(record.keys())
        if missing_fields:
            logging.error(f"Record {i} missing required fields: {missing_fields}")
            return False
    
    return True


# ============================================================================
# Main CLI Interface
# ============================================================================

@click.command()
@click.option(
    "-m", "--model-id", 
    required=True, 
    help="Model ID to export data for (e.g., 'Llama-3.1-8B-Instruct')"
)
@click.option(
    "-i", "--instr-id", 
    required=True, 
    help="Instruction ID used for generation (e.g., 'instr-v0')"
)
@click.option(
    "-p", "--prop-id", 
    help="Specific property ID to export (optional, exports all if not specified)"
)
@click.option(
    "-o", "--output", 
    type=click.Path(), 
    help="Output JSONL file path (defaults to model_id_faithfulness.jsonl)"
)
@click.option(
    "--include-unknown", 
    is_flag=True, 
    help="Include unknown responses in export"
)
@click.option(
    "--verbose", "-v", 
    is_flag=True, 
    help="Verbose output"
)
@beartype
def main(
    model_id: str,
    instr_id: str,
    prop_id: str | None,
    output: str | None,
    include_unknown: bool,
    verbose: bool
) -> None:
    """Export faithfulness evaluation data to JSONL format."""
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(levelname)s: %(message)s"
    )
    
    try:
        # Validate model exists
        available_models = get_available_models()
        if model_id not in available_models:
            logging.error(f"Model '{model_id}' not found. Available models: {available_models}")
            return
        
        # Validate property exists (if specified)
        if prop_id:
            available_props = get_available_properties(model_id)
            if prop_id not in available_props:
                logging.error(f"Property '{prop_id}' not found for model '{model_id}'. Available properties: {available_props}")
                return
        
        # Load faithfulness data
        logging.info(f"Loading faithfulness data for model: {model_id}")
        if prop_id:
            logging.info(f"Filtering by property: {prop_id}")
        
        faithfulness_data = load_faithfulness_data(model_id, prop_id)
        
        if not faithfulness_data:
            logging.warning("No faithfulness data found")
            return
        
        # Process data
        logging.info(f"Processing {len(faithfulness_data)} questions")
        all_export_records = []
        
        for q_id, question_data in faithfulness_data.items():
            records = process_question_data(q_id, question_data, instr_id, include_unknown)
            all_export_records.extend(records)
        
        if not all_export_records:
            logging.warning("No response data found to export")
            return
        
        # Validate data
        if not validate_export_data(all_export_records):
            logging.error("Data validation failed")
            return
        
        # Determine output path
        if output:
            output_path = Path(output)
        else:
            suffix = f"_{prop_id}" if prop_id else ""
            output_path = Path(f"{model_id}{suffix}_faithfulness.jsonl")
        
        # Export to JSONL
        logging.info(f"Exporting {len(all_export_records)} records to {output_path}")
        export_to_jsonl(all_export_records, output_path)
        
        # Report statistics
        faithful_count = sum(1 for r in all_export_records if r['faithful_label'] is True)
        unfaithful_count = sum(1 for r in all_export_records if r['faithful_label'] is False)
        unknown_count = sum(1 for r in all_export_records if r['faithful_label'] is None)
        
        logging.info(f"Export completed successfully!")
        logging.info(f"  Total records: {len(all_export_records)}")
        logging.info(f"  Faithful: {faithful_count}")
        logging.info(f"  Unfaithful: {unfaithful_count}")
        if unknown_count > 0:
            logging.info(f"  Unknown: {unknown_count}")
        logging.info(f"  Output file: {output_path}")
        
    except Exception as e:
        logging.error(f"Error: {e}")
        if verbose:
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()