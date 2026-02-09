"""Data setup for agent evaluation datasets.

This module provides the infrastructure for loading and managing evaluation datasets
used to test agent performance across different research tasks.
"""

import json
import logging
from pathlib import Path
from typing import List, Literal, Optional, TypedDict

logger = logging.getLogger(__name__)


class EvalExample(TypedDict):
    """A single evaluation example for testing agent performance.

    Attributes:
        id: Unique identifier for the example
        query: The research question or task to be performed
        category: Category of the task (e.g., 'factual', 'comparison', 'multi-faceted')
        difficulty: Difficulty level (easy, medium, hard)
        expected_elements: Key elements that should be present in a good answer
        metadata: Optional additional metadata about the example
    """
    id: str
    query: str
    category: str
    difficulty: Literal["easy", "medium", "hard"]
    expected_elements: List[str]
    metadata: Optional[dict]


class EvalDataset:
    """Manager for evaluation datasets.

    This class handles loading, filtering, and accessing evaluation examples
    from JSON files or in-memory data structures.
    """

    def __init__(self, examples: List[EvalExample]):
        """Initialize dataset with a list of examples.

        Args:
            examples: List of evaluation examples
        """
        self.examples = examples
        self._validate_examples()

    def _validate_examples(self) -> None:
        """Validate that all examples have required fields."""
        required_fields = {"id", "query", "category", "difficulty", "expected_elements"}

        for i, example in enumerate(self.examples):
            missing = required_fields - set(example.keys())
            if missing:
                raise ValueError(
                    f"Example at index {i} missing required fields: {missing}"
                )

            if not isinstance(example["expected_elements"], list):
                raise ValueError(
                    f"Example {example['id']}: expected_elements must be a list"
                )

    @classmethod
    def from_json(cls, filepath: str | Path) -> "EvalDataset":
        """Load evaluation dataset from a JSON file.

        Args:
            filepath: Path to JSON file containing evaluation examples

        Returns:
            EvalDataset instance loaded from file

        Raises:
            FileNotFoundError: If the file doesn't exist
            json.JSONDecodeError: If the file is not valid JSON
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Eval dataset not found: {filepath}")

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Handle both list format and dict with "examples" key
        if isinstance(data, dict) and "examples" in data:
            examples = data["examples"]
        elif isinstance(data, list):
            examples = data
        else:
            raise ValueError(
                "JSON file must contain either a list of examples or a dict with 'examples' key"
            )

        return cls(examples)

    @classmethod
    def from_dict(cls, data: dict | List[dict]) -> "EvalDataset":
        """Create dataset from dictionary or list of dictionaries.

        Args:
            data: Either a list of examples or dict with "examples" key

        Returns:
            EvalDataset instance
        """
        if isinstance(data, dict) and "examples" in data:
            examples = data["examples"]
        elif isinstance(data, list):
            examples = data
        else:
            raise ValueError(
                "Data must be either a list of examples or a dict with 'examples' key"
            )

        return cls(examples)

    def filter_by_category(self, category: str) -> "EvalDataset":
        """Filter examples by category.

        Args:
            category: Category to filter by

        Returns:
            New EvalDataset with filtered examples
        """
        filtered = [ex for ex in self.examples if ex["category"] == category]
        return EvalDataset(filtered)

    def filter_by_difficulty(self, difficulty: Literal["easy", "medium", "hard"]) -> "EvalDataset":
        """Filter examples by difficulty level.

        Args:
            difficulty: Difficulty level to filter by

        Returns:
            New EvalDataset with filtered examples
        """
        filtered = [ex for ex in self.examples if ex["difficulty"] == difficulty]
        return EvalDataset(filtered)

    def get_by_id(self, example_id: str) -> Optional[EvalExample]:
        """Get a specific example by ID.

        Args:
            example_id: ID of the example to retrieve

        Returns:
            The example if found, None otherwise
        """
        for example in self.examples:
            if example["id"] == example_id:
                return example
        return None

    def __len__(self) -> int:
        """Return the number of examples in the dataset."""
        return len(self.examples)

    def __getitem__(self, idx: int) -> EvalExample:
        """Get example at the given index."""
        return self.examples[idx]

    def __iter__(self):
        """Iterate over examples in the dataset."""
        return iter(self.examples)


def create_sample_dataset() -> EvalDataset:
    """Create a sample evaluation dataset for testing.

    Returns:
        Sample EvalDataset with example queries across different categories
    """
    examples = [
        {
            "id": "factual_001",
            "query": "What is the Model Context Protocol (MCP)?",
            "category": "factual",
            "difficulty": "easy",
            "expected_elements": [
                "Protocol definition",
                "Purpose/use case",
                "Key components or features"
            ],
            "metadata": {"domain": "ai_infrastructure"}
        },
        {
            "id": "comparison_001",
            "query": "Compare OpenAI's GPT-4 and Anthropic's Claude 3 in terms of capabilities and safety features",
            "category": "comparison",
            "difficulty": "medium",
            "expected_elements": [
                "GPT-4 capabilities",
                "Claude 3 capabilities",
                "Safety features comparison",
                "Key differences"
            ],
            "metadata": {"domain": "ai_models"}
        },
        {
            "id": "multifaceted_001",
            "query": "Research the current state of renewable energy: economic costs, environmental impact, and adoption rates across different regions",
            "category": "multi-faceted",
            "difficulty": "hard",
            "expected_elements": [
                "Economic cost analysis",
                "Environmental impact data",
                "Regional adoption statistics",
                "Multiple renewable energy types covered"
            ],
            "metadata": {"domain": "energy", "requires_multiple_sources": True}
        },
        {
            "id": "factual_002",
            "query": "List the top 5 programming languages in 2024 by popularity",
            "category": "factual",
            "difficulty": "easy",
            "expected_elements": [
                "5 programming languages listed",
                "Popularity metrics or ranking source",
                "Current year data (2024)"
            ],
            "metadata": {"domain": "software_engineering"}
        },
        {
            "id": "comparison_002",
            "query": "Compare React, Vue, and Angular frameworks for web development",
            "category": "comparison",
            "difficulty": "medium",
            "expected_elements": [
                "React overview",
                "Vue overview",
                "Angular overview",
                "Comparative analysis",
                "Use case recommendations"
            ],
            "metadata": {"domain": "web_development"}
        }
    ]

    return EvalDataset(examples)


def save_dataset_template(filepath: str | Path) -> None:
    """Save a template evaluation dataset to a JSON file.

    This creates an example file that can be used as a template for creating
    custom evaluation datasets.

    Args:
        filepath: Path where the template should be saved
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    dataset = create_sample_dataset()

    output = {
        "metadata": {
            "description": "Sample evaluation dataset for agent research tasks",
            "version": "1.0",
            "total_examples": len(dataset)
        },
        "examples": dataset.examples
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    # Configure logging for standalone execution
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Example usage
    logger.info("Creating sample dataset...")
    dataset = create_sample_dataset()

    logger.info(f"Dataset size: {len(dataset)}")
    logger.info(f"First example:")
    logger.info(f"  ID: {dataset[0]['id']}")
    logger.info(f"  Query: {dataset[0]['query']}")
    logger.info(f"  Category: {dataset[0]['category']}")
    logger.info(f"  Difficulty: {dataset[0]['difficulty']}")

    # Filter examples
    logger.info(f"Easy examples: {len(dataset.filter_by_difficulty('easy'))}")
    logger.info(f"Factual examples: {len(dataset.filter_by_category('factual'))}")

    # Save template
    template_path = Path("data/evals/sample_dataset.json")
    logger.info(f"Saving template to: {template_path}")
    save_dataset_template(template_path)
    logger.info("Template saved successfully!")
