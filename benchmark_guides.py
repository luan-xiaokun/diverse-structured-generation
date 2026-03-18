"""
Benchmark script for diverse guide implementations.

This script benchmarks the performance of different regex guide implementations,
including Python-based and Rust-based versions. It supports multiple iterations,
cache avoidance mechanisms, and provides detailed statistical analysis of results.
"""

import argparse
import os
import statistics
import sys
import time
import types
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
from unittest.mock import MagicMock

# Add src to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

# --- Mocks for missing dependencies ---

# Since torch is already installed, we don't need to mock it completely
# Just mock outlines which is not installed

# Mock outlines
mock_outlines = MagicMock()
sys.modules["outlines"] = mock_outlines
sys.modules["outlines.generate"] = MagicMock()
sys.modules["outlines.generate.api"] = MagicMock()
sys.modules["outlines.models"] = MagicMock()
sys.modules["outlines.models.tokenizer"] = MagicMock()
sys.modules["outlines.models.transformers"] = MagicMock()
sys.modules["outlines.processors"] = MagicMock()
sys.modules["outlines.processors.structured"] = MagicMock()
sys.modules["outlines.samplers"] = MagicMock()

# Mock outlines_core
mock_outlines_core = MagicMock()
sys.modules["outlines_core"] = mock_outlines_core
sys.modules["outlines_core.fsm"] = MagicMock()
sys.modules["outlines_core.fsm.guide"] = MagicMock()


# Define base classes that are inherited
class MockGuide:
    """Mock guide base class."""

    def __init__(self, *args, **kwargs):
        pass

    def copy(self):
        return self


class MockGuideLogitsProcessor:
    """Mock guide logits processor."""

    def __init__(self, tokenizer, guide):
        self.tokenizer = tokenizer
        self.guide = guide
        self._guide_states = {}
        self._seq_start_idx = None


class MockSequenceGeneratorAdapter:
    """Mock sequence generator adapter."""

    def __init__(self, model, logits_processor, sampler):
        pass


sys.modules["outlines_core.fsm.guide"].Guide = MockGuide
sys.modules["outlines_core.fsm.guide"].Generate = MagicMock()
sys.modules["outlines_core.fsm.guide"].Write = MagicMock()
sys.modules["outlines.processors.structured"].GuideLogitsProcessor = (
    MockGuideLogitsProcessor
)
sys.modules["outlines.generate.api"].SequenceGeneratorAdapter = (
    MockSequenceGeneratorAdapter
)

# --- End Mocks ---


# Mock Tokenizer
class MockTokenizer:
    """Mock tokenizer for testing purposes."""

    def __init__(self):
        self.eos_token_id = 0
        self.vocabulary = {
            "<eos>": 0,
            "a": 1,
            "b": 2,
            "c": 3,
            "1": 4,
            "2": 5,
            "3": 6,
            '"': 7,
            ":": 8,
            "{": 9,
            "}": 10,
            "name": 11,
            "value": 12,
            ",": 13,
            " ": 14,
            "\n": 15,
        }
        self.special_tokens = {"<eos>"}
        self.all_special_tokens = ["<eos>"]
        self.inverse_vocab = {v: k for k, v in self.vocabulary.items()}

    def convert_token_to_string(self, token: str) -> str:
        return token

    def convert_tokens_to_string(self, tokens: list) -> str:
        """Convert a list of tokens to a string."""
        return "".join(tokens)

    def get_vocab(self):
        """Return vocabulary dictionary mapping token to id."""
        return self.vocabulary


# Import guides
try:
    import diverse_guide
    import diverse_guide_rust
except ImportError as e:
    print(f"Error importing guides: {e}")
    print(
        "Make sure you are running this script from the project root or have set up the environment correctly."
    )
    sys.exit(1)


@dataclass
class BenchmarkResult:
    """Data class to store benchmark results for a single iteration."""

    init_time: float
    allowed_time: float
    next_state_time: float
    is_final_time: float
    seq_time: float
    update_time: float
    total_time: float


@dataclass
class StatisticalSummary:
    """Data class to store statistical summary of multiple benchmark runs."""

    mean: float
    std: float
    min: float
    max: float
    median: float


def calculate_statistics(values: List[float]) -> StatisticalSummary:
    """Calculate statistical summary from a list of values."""
    if not values:
        return StatisticalSummary(0.0, 0.0, 0.0, 0.0, 0.0)

    return StatisticalSummary(
        mean=statistics.mean(values),
        std=statistics.stdev(values) if len(values) > 1 else 0.0,
        min=min(values),
        max=max(values),
        median=statistics.median(values),
    )


def format_time(value: float) -> str:
    """Format time value in milliseconds with appropriate precision."""
    if value < 0.001:
        return f"{value * 1000000:.2f} μs"
    elif value < 1.0:
        return f"{value * 1000:.2f} ms"
    else:
        return f"{value:.4f} s"


def format_summary(summary: StatisticalSummary) -> str:
    """Format statistical summary as a string."""
    return (
        f"Mean: {format_time(summary.mean)}, "
        f"Std: {format_time(summary.std)}, "
        f"Min: {format_time(summary.min)}, "
        f"Max: {format_time(summary.max)}, "
        f"Median: {format_time(summary.median)}"
    )


def print_separator(char: str = "=", length: int = 80):
    """Print a separator line."""
    print(char * length)


def print_header(text: str):
    """Print a formatted header."""
    print_separator()
    print(f"\n{text}\n")
    print_separator()


def benchmark_method(name: str, func, *args) -> Tuple[Any, float]:
    """
    Benchmark a method call.

    Args:
        name: Name of the method being benchmarked
        func: Function to benchmark
        *args: Arguments to pass to the function

    Returns:
        Tuple of (result, time_in_milliseconds)
    """
    start = time.perf_counter()
    result = func(*args)
    end = time.perf_counter()
    return result, (end - start) * 1000  # Convert to milliseconds


def run_single_iteration(
    impl_name: str, GuideClass, regex: str, tokens: List[int], tokenizer
) -> BenchmarkResult:
    """
    Run a single benchmark iteration.

    Args:
        impl_name: Name of the implementation
        GuideClass: The guide class to benchmark
        regex: Regular expression pattern
        tokens: List of token IDs to process
        tokenizer: Tokenizer instance

    Returns:
        BenchmarkResult containing timing information
    """
    # Initialization
    start_init = time.perf_counter()
    guide = GuideClass(regex, tokenizer)
    end_init = time.perf_counter()
    init_time = (end_init - start_init) * 1000

    current_state = guide.initial_state
    total_allowed_time = 0.0
    total_next_state_time = 0.0
    total_is_final_time = 0.0
    total_seq_time = 0.0
    total_update_time = 0.0

    for token_id in tokens:
        # 1. get_allowed_token_ids
        if hasattr(guide.dfa, "get_allowed_token_ids"):
            _, t = benchmark_method(
                "get_allowed_token_ids",
                guide.dfa.get_allowed_token_ids,
                current_state,
            )
            total_allowed_time += t

        # 2. is_final_state
        _, t = benchmark_method("is_final_state", guide.is_final_state, current_state)
        total_is_final_time += t

        # 3. get_next_token_state
        next_state, t = benchmark_method(
            "get_next_token_state",
            guide.get_next_state,
            current_state,
            token_id,
        )
        total_next_state_time += t

        # 4. get_byte_state_sequence / get_state_sequence_from_token_id
        if hasattr(guide, "get_state_sequence_from_token_id"):
            _, t = benchmark_method(
                "get_state_sequence_from_token_id",
                guide.get_state_sequence_from_token_id,
                current_state,
                token_id,
            )
            total_seq_time += t
        elif hasattr(guide.dfa, "get_byte_state_sequence"):
            _, t = benchmark_method(
                "get_byte_state_sequence",
                guide.dfa.get_byte_state_sequence,
                current_state,
                token_id,
            )
            total_seq_time += t

        # 5. update_local_state_counter
        if hasattr(guide.dfa, "update_local_state_counter"):
            _, t = benchmark_method(
                "update_local_state_counter",
                guide.dfa.update_local_state_counter,
                current_state,
                token_id,
            )
            total_update_time += t

        current_state = next_state

    total_time = (
        init_time
        + total_allowed_time
        + total_next_state_time
        + total_is_final_time
        + total_seq_time
        + total_update_time
    )

    return BenchmarkResult(
        init_time=init_time,
        allowed_time=total_allowed_time,
        next_state_time=total_next_state_time,
        is_final_time=total_is_final_time,
        seq_time=total_seq_time,
        update_time=total_update_time,
        total_time=total_time,
    )


def run_benchmark_with_iterations(
    impl_name: str,
    GuideClass,
    regex: str,
    tokens: List[int],
    tokenizer,
    num_iterations: int = 100,
    use_cold_restart: bool = False,
) -> List[BenchmarkResult]:
    """
    Run benchmark with multiple iterations.

    Args:
        impl_name: Name of the implementation
        GuideClass: The guide class to benchmark
        regex: Regular expression pattern
        tokens: List of token IDs to process
        tokenizer: Tokenizer instance
        num_iterations: Number of iterations to run
        use_cold_restart: Whether to use cold restart between iterations

    Returns:
        List of BenchmarkResult objects
    """
    results = []

    for i in range(num_iterations):
        if use_cold_restart and i > 0:
            # Force Python to release memory and reset Rust backend
            # This is a best-effort approach to avoid caching
            import gc

            gc.collect()

        result = run_single_iteration(impl_name, GuideClass, regex, tokens, tokenizer)
        results.append(result)

    return results


def print_detailed_results(
    impl_name: str, results: List[BenchmarkResult], show_all_iterations: bool = False
):
    """
    Print detailed benchmark results with statistics.

    Args:
        impl_name: Name of the implementation
        results: List of benchmark results
        show_all_iterations: Whether to show all iteration results
    """
    print_header(f"{impl_name} - Detailed Results")

    # Extract timing data
    init_times = [r.init_time for r in results]
    allowed_times = [r.allowed_time for r in results]
    next_state_times = [r.next_state_time for r in results]
    is_final_times = [r.is_final_time for r in results]
    seq_times = [r.seq_time for r in results]
    update_times = [r.update_time for r in results]
    total_times = [r.total_time for r in results]

    # Calculate statistics
    print("\nStatistical Summary:")
    print_separator("-")

    print(f"\nInitialization Time:")
    print(f"  {format_summary(calculate_statistics(init_times))}")

    print(f"\nget_allowed_token_ids Time:")
    print(f"  {format_summary(calculate_statistics(allowed_times))}")

    print(f"\nget_next_token_state Time:")
    print(f"  {format_summary(calculate_statistics(next_state_times))}")

    print(f"\nis_final_state Time:")
    print(f"  {format_summary(calculate_statistics(is_final_times))}")

    print(f"\nSequence Retrieval Time:")
    print(f"  {format_summary(calculate_statistics(seq_times))}")

    print(f"\nupdate_local_state_counter Time:")
    print(f"  {format_summary(calculate_statistics(update_times))}")

    print(f"\nTotal Time:")
    print(f"  {format_summary(calculate_statistics(total_times))}")

    if show_all_iterations and len(results) <= 20:
        print("\n\nIndividual Iteration Results:")
        print_separator("-")
        for i, result in enumerate(results, 1):
            print(f"\nIteration {i}:")
            print(f"  Init: {result.init_time:.4f} ms")
            print(f"  Allowed: {result.allowed_time:.4f} ms")
            print(f"  Next State: {result.next_state_time:.4f} ms")
            print(f"  Is Final: {result.is_final_time:.4f} ms")
            print(f"  Sequence: {result.seq_time:.4f} ms")
            print(f"  Update: {result.update_time:.4f} ms")
            print(f"  Total: {result.total_time:.4f} ms")


def print_comparison_table(all_results: Dict[str, List[BenchmarkResult]]):
    """
    Print a comparison table of all implementations.

    Args:
        all_results: Dictionary mapping implementation names to their results
    """
    print_header("Implementation Comparison")

    # Calculate mean times for each implementation
    impl_names = list(all_results.keys())

    print("\nAverage Execution Times (ms):")
    print_separator("-")
    print(
        f"{'Implementation':<40} {'Init':>12} {'Allowed':>12} {'Next':>12} {'Seq':>12} {'Total':>12}"
    )
    print_separator("-")

    for impl_name in impl_names:
        results = all_results[impl_name]
        init_avg = statistics.mean([r.init_time for r in results])
        allowed_avg = statistics.mean([r.allowed_time for r in results])
        next_avg = statistics.mean([r.next_state_time for r in results])
        seq_avg = statistics.mean([r.seq_time for r in results])
        total_avg = statistics.mean([r.total_time for r in results])

        print(
            f"{impl_name:<40} {init_avg:>12.4f} {allowed_avg:>12.4f} {next_avg:>12.4f} {seq_avg:>12.4f} {total_avg:>12.4f}"
        )

    print_separator()

    # Calculate speedup
    if len(impl_names) == 2:
        print("\nSpeedup Analysis:")
        print_separator("-")
        base_name = impl_names[0]
        other_name = impl_names[1]

        base_total = statistics.mean([r.total_time for r in all_results[base_name]])
        other_total = statistics.mean([r.total_time for r in all_results[other_name]])

        speedup = base_total / other_total if other_total > 0 else 0
        print(f"{other_name} is {speedup:.2f}x faster than {base_name}")
        print_separator()


def verify_correctness(
    impl_name: str,
    GuideClass,
    regex: str,
    tokens: List[int],
    tokenizer,
    reference_state_history: List[int] = None,
) -> bool:
    """
    Verify correctness of an implementation.

    Args:
        impl_name: Name of the implementation
        GuideClass: The guide class to verify
        regex: Regular expression pattern
        tokens: List of token IDs to process
        tokenizer: Tokenizer instance
        reference_state_history: Reference state history to compare against

    Returns:
        True if correct, False otherwise
    """
    try:
        guide = GuideClass(regex, tokenizer)
        current_state = guide.initial_state
        state_history = [current_state]

        for token_id in tokens:
            current_state = guide.get_next_state(current_state, token_id)
            state_history.append(current_state)

        if reference_state_history is not None:
            if state_history == reference_state_history:
                return True
            else:
                print(f"  State history MISMATCH for {impl_name}")
                print(f"  Expected: {reference_state_history}")
                print(f"  Got: {state_history}")
                return False

        return True
    except Exception as e:
        print(f"  Error during correctness check for {impl_name}: {e}")
        return False


def run_benchmark(
    num_iterations: int = 100,
    use_cold_restart: bool = False,
    show_all_iterations: bool = False,
    regex_pattern: str = None,
    token_sequence: str = None,
):
    """
    Run the complete benchmark suite.

    Args:
        num_iterations: Number of iterations per test case
        use_cold_restart: Whether to use cold restart between iterations
        show_all_iterations: Whether to show all iteration results
        regex_pattern: Optional specific regex pattern to test
        token_sequence: Optional specific token sequence to test
    """
    tokenizer = MockTokenizer()

    # Test cases: (regex, sequence of token_ids to simulate)
    test_cases = [
        (r"[1-3]+", [4, 5, 6, 4, 0], "Simple digit pattern"),
        (
            r"\{ \"name\": \"[a-c]+\" \}",
            [9, 14, 7, 11, 7, 8, 14, 7, 1, 2, 3, 7, 14, 10, 0],
            "JSON-like pattern with name",
        ),
        (r"[a-c][1-3]+[a-c]", [1, 4, 5, 6, 2, 0], "Mixed alphanumeric pattern"),
        (r"(a|b|c)+", [1, 2, 3, 1, 2, 0], "Alternation pattern"),
    ]

    # Override test cases if specific pattern is provided
    if regex_pattern is not None:
        if token_sequence is not None:
            tokens = [int(x.strip()) for x in token_sequence.split(",")]
        else:
            tokens = [4, 5, 6, 4, 0]  # Default token sequence
        test_cases = [(regex_pattern, tokens, "Custom test case")]

    # Implementations to test
    implementations = [
        (
            "diverse_guide_rust (Rust/DiverseGuideDFA)",
            diverse_guide_rust.DiverseRegexGuide,
        ),
        ("diverse_guide (Python/MinDivDFA)", diverse_guide.DiverseRegexGuide),
    ]

    print_header(f"Benchmark Suite - {num_iterations} Iterations per Test")
    print(f"Cold Restart: {'Enabled' if use_cold_restart else 'Disabled'}")
    print(f"Test Cases: {len(test_cases)}")

    for regex, tokens, description in test_cases:
        print_header(f"Testing: {description}")
        print(f"Regex: {regex}")
        print(f"Token Sequence: {tokens}")
        print(f"Tokens: {len(tokens)} tokens to process")

        all_results = {}
        reference_state_history = None

        # First, verify correctness
        print("\nCorrectness Verification:")
        print_separator("-")
        for impl_name, GuideClass in implementations:
            is_correct = verify_correctness(
                impl_name, GuideClass, regex, tokens, tokenizer, reference_state_history
            )
            if is_correct and reference_state_history is None:
                # Use first implementation as reference
                guide = GuideClass(regex, tokenizer)
                current_state = guide.initial_state
                state_history = [current_state]
                for token_id in tokens:
                    current_state = guide.get_next_state(current_state, token_id)
                    state_history.append(current_state)
                reference_state_history = state_history

        # Run benchmarks
        print("\nRunning Benchmarks...")
        print_separator("-")

        for impl_name, GuideClass in implementations:
            print(f"\nBenchmarking {impl_name}...")
            results = run_benchmark_with_iterations(
                impl_name,
                GuideClass,
                regex,
                tokens,
                tokenizer,
                num_iterations,
                use_cold_restart,
            )
            all_results[impl_name] = results
            print(f"Completed {len(results)} iterations")

        # Print detailed results
        for impl_name, results in all_results.items():
            print_detailed_results(impl_name, results, show_all_iterations)

        # Print comparison
        print_comparison_table(all_results)

    print_header("Benchmark Complete")


def main():
    """Main entry point for the benchmark script."""
    parser = argparse.ArgumentParser(
        description="Benchmark diverse guide implementations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run default benchmark with 100 iterations
  python benchmark_guides.py
  
  # Run with 1000 iterations and cold restart
  python benchmark_guides.py --iterations 1000 --cold-restart
  
  # Run single test case with specific regex
  python benchmark_guides.py --regex "[a-z]+" --tokens "1,2,3,4,0"
  
  # Show all iteration results
  python benchmark_guides.py --show-all
        """,
    )

    parser.add_argument(
        "-i",
        "--iterations",
        type=int,
        default=100,
        help="Number of iterations per test case (default: 100)",
    )

    parser.add_argument(
        "-c",
        "--cold-restart",
        action="store_true",
        help="Use cold restart between iterations to avoid caching",
    )

    parser.add_argument(
        "-s",
        "--show-all",
        action="store_true",
        help="Show results for all individual iterations (only for <= 20 iterations)",
    )

    parser.add_argument(
        "-r", "--regex", type=str, default=None, help="Specific regex pattern to test"
    )

    parser.add_argument(
        "-t",
        "--tokens",
        type=str,
        default=None,
        help="Comma-separated token sequence to test (e.g., '1,2,3,4,0')",
    )

    args = parser.parse_args()

    run_benchmark(
        num_iterations=args.iterations,
        use_cold_restart=args.cold_restart,
        show_all_iterations=args.show_all,
        regex_pattern=args.regex,
        token_sequence=args.tokens,
    )


if __name__ == "__main__":
    main()
