"""Test agent performance using evaluation datasets.

This module runs the full research agent against evaluation examples
and collects results for analysis.
"""

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from src.evals.data_setup import EvalDataset, EvalExample
from src.full_agent import run_research_query

logger = logging.getLogger(__name__)


class AgentTestResult:
    """Container for a single agent test run result.

    Attributes:
        example_id: ID of the evaluation example
        query: The original query
        response: Agent's final response
        files_created: List of files created during execution
        todos_final: Final TODO list state
        execution_time: Time taken in seconds
        metadata: Additional metadata from the example
    """

    def __init__(
        self,
        example_id: str,
        query: str,
        response: str,
        files_created: List[str],
        todos_final: List[Dict],
        execution_time: float,
        metadata: Dict[str, Any],
    ):
        self.example_id = example_id
        self.query = query
        self.response = response
        self.files_created = files_created
        self.todos_final = todos_final
        self.execution_time = execution_time
        self.metadata = metadata

    def __repr__(self) -> str:
        return (
            f"AgentTestResult(id={self.example_id}, "
            f"files={len(self.files_created)}, "
            f"time={self.execution_time:.2f}s)"
        )


def test_agent_on_example(example: EvalExample, verbose: bool = True) -> AgentTestResult:
    """Run the agent on a single evaluation example.

    Args:
        example: The evaluation example to test
        verbose: Whether to log progress information

    Returns:
        AgentTestResult containing the test outcome
    """
    if verbose:
        logger.info("=" * 80)
        logger.info(f"Testing: {example['id']}")
        logger.info(f"Query: {example['query']}")
        logger.info(f"Category: {example['category']} | Difficulty: {example['difficulty']}")
        logger.info("=" * 80)

    # Run the agent and measure execution time
    start_time = time.time()
    result = run_research_query(example['query'])
    execution_time = time.time() - start_time

    # Extract results
    final_message = result['messages'][-1]
    response = final_message.content if hasattr(final_message, 'content') else str(final_message)
    files_created = list(result.get('files', {}).keys())
    todos_final = result.get('todos', [])

    if verbose:
        logger.info("=" * 80)
        logger.info(f"COMPLETED in {execution_time:.2f}s")
        logger.info(f"Files created: {len(files_created)}")
        logger.info("=" * 80)
        logger.info("RESPONSE:")
        logger.info(response)
        logger.info("=" * 80)

        if files_created:
            logger.info("Files created:")
            for filename in files_created:
                logger.info(f"  - {filename}")

    return AgentTestResult(
        example_id=example['id'],
        query=example['query'],
        response=response,
        files_created=files_created,
        todos_final=todos_final,
        execution_time=execution_time,
        metadata=example.get('metadata', {}),
    )


def test_agent_on_dataset(
    dataset: EvalDataset,
    max_examples: int = None,
    verbose: bool = True,
) -> List[AgentTestResult]:
    """Run the agent on multiple evaluation examples.

    Args:
        dataset: EvalDataset to test against
        max_examples: Maximum number of examples to test (None for all)
        verbose: Whether to log progress information

    Returns:
        List of AgentTestResult objects
    """
    examples = dataset.examples[:max_examples] if max_examples else dataset.examples
    results = []

    logger.info("#" * 80)
    logger.info(f"# Testing agent on {len(examples)} examples")
    logger.info(f"# Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("#" * 80)

    for i, example in enumerate(examples, 1):
        if verbose:
            logger.info(f"[{i}/{len(examples)}] Processing {example['id']}...")

        try:
            result = test_agent_on_example(example, verbose=verbose)
            results.append(result)
        except Exception as e:
            logger.error(f"ERROR testing {example['id']}: {e}")
            # Create a failed result
            results.append(AgentTestResult(
                example_id=example['id'],
                query=example['query'],
                response=f"ERROR: {str(e)}",
                files_created=[],
                todos_final=[],
                execution_time=0.0,
                metadata=example.get('metadata', {}),
            ))

    logger.info("#" * 80)
    logger.info(f"# Testing completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"# Total tests: {len(results)}")
    logger.info(f"# Success: {sum(1 for r in results if not r.response.startswith('ERROR'))}")
    logger.info(f"# Failed: {sum(1 for r in results if r.response.startswith('ERROR'))}")
    logger.info("#" * 80)

    return results


def print_summary(results: List[AgentTestResult]) -> None:
    """Log a summary of test results.

    Args:
        results: List of AgentTestResult objects
    """
    logger.info("=" * 80)
    logger.info("TEST SUMMARY")
    logger.info("=" * 80)

    total = len(results)
    successful = [r for r in results if not r.response.startswith('ERROR')]
    failed = [r for r in results if r.response.startswith('ERROR')]

    logger.info(f"Total tests: {total}")
    logger.info(f"Successful: {len(successful)}")
    logger.info(f"Failed: {len(failed)}")

    if successful:
        avg_time = sum(r.execution_time for r in successful) / len(successful)
        avg_files = sum(len(r.files_created) for r in successful) / len(successful)

        logger.info(f"Average execution time: {avg_time:.2f}s")
        logger.info(f"Average files created: {avg_files:.1f}")

        logger.info("Execution times:")
        for result in successful:
            logger.info(f"  {result.example_id}: {result.execution_time:.2f}s ({len(result.files_created)} files)")

    if failed:
        logger.info("Failed tests:")
        for result in failed:
            logger.error(f"  ❌ {result.example_id}")


if __name__ == "__main__":
    # Configure logging for standalone execution
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Load the sample dataset
    dataset_path = Path("data/evals/sample_dataset.json")

    logger.info(f"Loading dataset from: {dataset_path}")
    dataset = EvalDataset.from_json(dataset_path)

    logger.info(f"Loaded {len(dataset)} examples")

    # Test on easy examples only for quick testing
    easy_dataset = dataset.filter_by_difficulty('easy')
    logger.info(f"Testing on {len(easy_dataset)} easy examples...")

    # Run tests
    results = test_agent_on_dataset(easy_dataset, verbose=True)

    # Print summary
    print_summary(results)
