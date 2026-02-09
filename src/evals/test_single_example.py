"""Quick test of a single evaluation example."""

import logging

from src.evals.data_setup import EvalDataset
from src.evals.test_agent import test_agent_on_example

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load dataset
dataset = EvalDataset.from_json("data/evals/sample_dataset.json")

# Get first easy example
easy_examples = dataset.filter_by_difficulty('easy')
example = easy_examples[0]

logger.info(f"Testing example: {example['id']}")
logger.info(f"Query: {example['query']}")

# Run test
result = test_agent_on_example(example, verbose=True)

# Log summary
logger.info("=" * 80)
logger.info("SUMMARY")
logger.info("=" * 80)
logger.info(f"Example ID: {result.example_id}")
logger.info(f"Execution time: {result.execution_time:.2f}s")
logger.info(f"Files created: {len(result.files_created)}")
logger.info(f"Files: {', '.join(result.files_created)}")
logger.info("Expected elements:")
for elem in example['expected_elements']:
    logger.info(f"  - {elem}")
