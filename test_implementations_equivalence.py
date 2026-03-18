"""
Comprehensive test suite to verify equivalence between two DiverseGuide implementations.

This tests:
1. diverse_guide.guide_rust.DiverseRegexGuide (uses DiverseGuideDFA)
2. diverse_guide.DiverseRegexGuide (uses MinDivDFA)
"""

import random
import sys
from typing import List, Tuple
from collections import defaultdict

# Add src to sys.path
sys.path.append("src")

from diverse_guide.guide_rust import DiverseRegexGuide as DiverseGuideRust
from diverse_guide import DiverseRegexGuide as DiverseGuidePython


# Mock Tokenizer (same as benchmark_guides.py)
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
            "x": 16,
            "y": 17,
            "z": 18,
            "0": 19,
            "4": 20,
            "5": 21,
            "6": 22,
            "7": 23,
            "8": 24,
            "9": 25,
            "A": 26,
            "B": 27,
            "C": 28,
            "D": 29,
            "E": 30,
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


# Test regex patterns (diverse and representative)
TEST_PATTERNS = [
    # Simple patterns
    r"[a-z]+",
    r"[0-9]+",
    r"[A-Z]+",
    # Digit patterns
    r"[1-3]+",
    r"[0-9]+",
    r"[a-c0-9]+",
    # Character classes
    r"[a-zA-Z]+",
    r"[a-z0-9]+",
    r"[A-Za-z0-9_]+",
    # Repetition
    r"a+",
    r"a*b+",
    r"a?b+",
    # Alternation
    r"(a|b|c)+",
    r"(cat|dog|mouse)+",
    r"(yes|no|maybe)+",
    # Grouping
    r"(ab)+",
    r"(abc)+",
    r"[a-b]{3}",
    # Complex patterns
    r"[a-z]{2,5}",
    r"[0-9]{2,4}",
    r"[A-Z]{1,3}",
    # Mixed patterns
    r"[a-c][1-3]+[a-c]",
    r"[A-Z][0-9]+[a-z]",
    r"[a-z]+[0-9]+",
    # JSON-like patterns
    r'\{ "name": "[a-c]+" \}',
    r'\{ "value": "[0-9]+" \}',
    # Email-like
    r"[a-z]+@[a-z]+\.[a-z]{2,3}",
    # Phone-like
    r"[0-9]{3}-[0-9]{4}",
    # More complex
    r"[a-z]+([0-9]+[a-z]+)*",
    r"([a-z]+|[0-9]+)+",
]


def generate_random_tokens(
    tokenizer, length: int, allowed_tokens: set = None, include_eos: bool = True
) -> List[int]:
    """Generate random tokens for testing.

    Args:
        tokenizer: The tokenizer
        length: Length of token sequence
        allowed_tokens: Set of allowed token IDs (if None, use all tokens)
        include_eos: Whether to include EOS token
    """
    if allowed_tokens:
        # Only generate from allowed tokens
        token_pool = list(allowed_tokens)
    else:
        # Use all tokens from vocabulary
        token_pool = list(tokenizer.vocabulary.values())

    if not token_pool:
        return []

    tokens = [random.choice(token_pool) for _ in range(length)]
    if include_eos and random.random() < 0.2:
        tokens.append(tokenizer.eos_token_id)
    return tokens


def test_initial_state(guide_rust, guide_python, regex: str) -> bool:
    """Test if initial states match."""
    rust_initial = guide_rust.initial_state
    python_initial = guide_python.initial_state
    match = rust_initial == python_initial
    return match, f"Rust: {rust_initial}, Python: {python_initial}"


def test_get_next_state(
    guide_rust, guide_python, state: int, token_id: int
) -> Tuple[bool, str]:
    """Test if get_next_state produces same result."""
    rust_next = guide_rust.get_next_state(state, token_id)
    python_next = guide_python.get_next_state(state, token_id)
    match = rust_next == python_next
    return match, f"Rust: {rust_next}, Python: {python_next}"


def test_is_final_state(guide_rust, guide_python, state: int) -> Tuple[bool, str]:
    """Test if is_final_state produces same result."""
    rust_final = guide_rust.is_final_state(state)
    python_final = guide_python.is_final_state(state)
    match = rust_final == python_final
    return match, f"Rust: {rust_final}, Python: {python_final}"


def test_get_allowed_tokens(guide_rust, guide_python, state: int) -> Tuple[bool, str]:
    """Test if get_next_instruction/get_allowed_tokens produces same result."""
    rust_tokens = guide_rust.get_next_instruction(state)
    python_tokens = guide_python.get_next_instruction(state)

    # Convert python Generate/Write to token list
    if python_tokens is not None and hasattr(python_tokens, "tokens"):
        python_token_list = (
            python_tokens.tokens.tolist()
            if hasattr(python_tokens.tokens, "tolist")
            else list(python_tokens.tokens)
        )
    else:
        python_token_list = python_tokens if isinstance(python_tokens, list) else None

    # Handle None case
    if rust_tokens is None:
        rust_set = None
    else:
        rust_set = set(rust_tokens)

    if python_token_list is None:
        python_set = None
    else:
        python_set = set(python_token_list)

    match = rust_set == python_set
    return match, f"Rust: {rust_tokens}, Python: {python_token_list}"


def test_state_sequence(
    guide_rust, guide_python, state: int, token_id: int, tokenizer
) -> Tuple[bool, str]:
    """Test if state sequences match."""
    # Rust version: get_byte_state_sequence returns [state, byte1_state, ..., final_state]
    if hasattr(guide_rust.dfa, "get_byte_state_sequence"):
        rust_seq = guide_rust.dfa.get_byte_state_sequence(state, token_id)
        # Remove starting state so both sequences contain only the visited intermediate states
        if rust_seq and len(rust_seq) > 0:
            rust_seq = rust_seq[1:]
    else:
        rust_seq = None

    # Python version: get_state_sequence_from_token_id returns [state, byte1_state, ..., final_state]
    if hasattr(guide_python, "get_state_sequence_from_token_id"):
        python_seq = guide_python.get_state_sequence_from_token_id(state, token_id)
        # Convert to list if tensor
        if python_seq is not None and hasattr(python_seq, "tolist"):
            python_seq = python_seq.tolist()
        # Remove starting state (same normalization as Rust)
        if python_seq and len(python_seq) > 0:
            python_seq = python_seq[1:]
    else:
        python_seq = None

    match = rust_seq == python_seq
    return match, f"Rust: {rust_seq}, Python: {python_seq}"


def run_comprehensive_test(
    num_patterns: int = None,
    num_sequences_per_pattern: int = 10,
    sequence_length: int = 20,
) -> dict:
    """Run comprehensive equivalence tests."""

    patterns_to_test = TEST_PATTERNS[:num_patterns] if num_patterns else TEST_PATTERNS

    results = {
        "total_tests": 0,
        "passed": 0,
        "failed": 0,
        "pattern_results": [],
        "test_details": {
            "initial_state": defaultdict(list),
            "get_next_state": defaultdict(list),
            "is_final_state": defaultdict(list),
            "get_allowed_tokens": defaultdict(list),
            "state_sequence": defaultdict(list),
        },
        "failures": [],
    }

    tokenizer = MockTokenizer()

    for regex in patterns_to_test:
        print(f"\n{'='*80}")
        print(f"Testing pattern: {regex}")
        print("=" * 80)

        pattern_result = {
            "regex": regex,
            "total_tests": 0,
            "passed": 0,
            "failed": 0,
            "failures": [],
        }

        try:
            # Create guides
            guide_rust = DiverseGuideRust(regex, tokenizer)
            guide_python = DiverseGuidePython(regex, tokenizer)

            # Test 1: Initial state
            match, detail = test_initial_state(guide_rust, guide_python, regex)
            results["test_details"]["initial_state"][regex].append((match, detail))
            pattern_result["total_tests"] += 1
            results["total_tests"] += 1
            if match:
                pattern_result["passed"] += 1
                results["passed"] += 1
                print(f"✓ Initial state: PASS")
            else:
                pattern_result["failed"] += 1
                results["failed"] += 1
                pattern_result["failures"].append(("initial_state", detail))
                results["failures"].append((regex, "initial_state", detail))
                print(f"✗ Initial state: FAIL - {detail}")

            # Test 2: Random sequences
            for seq_idx in range(num_sequences_per_pattern):
                print(f"\n  Sequence {seq_idx + 1}/{num_sequences_per_pattern}")

                state_rust = guide_rust.initial_state
                state_python = guide_python.initial_state

                for step in range(sequence_length):
                    # Get allowed tokens from the CURRENT state at each step
                    rust_allowed = guide_rust.get_next_instruction(state_rust)
                    if rust_allowed is None:
                        rust_allowed_set = {tokenizer.eos_token_id}
                    else:
                        rust_allowed_set = set(rust_allowed)

                    if not rust_allowed_set:
                        break

                    token_id = random.choice(list(rust_allowed_set))

                    # Stop if EOS is chosen
                    if token_id == tokenizer.eos_token_id:
                        break
                    # Test get_next_state
                    match, detail = test_get_next_state(
                        guide_rust, guide_python, state_rust, token_id
                    )
                    results["test_details"]["get_next_state"][regex].append(
                        (match, detail)
                    )
                    pattern_result["total_tests"] += 1
                    results["total_tests"] += 1

                    if match:
                        pattern_result["passed"] += 1
                        results["passed"] += 1
                    else:
                        pattern_result["failed"] += 1
                        results["failed"] += 1
                        pattern_result["failures"].append(
                            ("get_next_state", f"step={step}, {detail}")
                        )
                        results["failures"].append(
                            (regex, f"get_next_state step={step}", detail)
                        )
                        print(f"    ✗ Step {step} (token={token_id}): FAIL - {detail}")
                        # Stop testing this sequence if they diverge
                        break

                    if not match:
                        break

                    # Test is_final_state
                    match, detail = test_is_final_state(
                        guide_rust, guide_python, state_rust
                    )
                    results["test_details"]["is_final_state"][regex].append(
                        (match, detail)
                    )
                    pattern_result["total_tests"] += 1
                    results["total_tests"] += 1

                    if match:
                        pattern_result["passed"] += 1
                        results["passed"] += 1
                    else:
                        pattern_result["failed"] += 1
                        results["failed"] += 1
                        pattern_result["failures"].append(
                            ("is_final_state", f"step={step}, {detail}")
                        )
                        results["failures"].append(
                            (regex, f"is_final_state step={step}", detail)
                        )
                        print(f"    ✗ Step {step} is_final: FAIL - {detail}")
                        break

                    # Test get_allowed_tokens
                    match, detail = test_get_allowed_tokens(
                        guide_rust, guide_python, state_rust
                    )
                    results["test_details"]["get_allowed_tokens"][regex].append(
                        (match, detail)
                    )
                    pattern_result["total_tests"] += 1
                    results["total_tests"] += 1

                    if match:
                        pattern_result["passed"] += 1
                        results["passed"] += 1
                    else:
                        pattern_result["failed"] += 1
                        results["failed"] += 1
                        pattern_result["failures"].append(
                            ("get_allowed_tokens", f"step={step}, {detail}")
                        )
                        results["failures"].append(
                            (regex, f"get_allowed_tokens step={step}", detail)
                        )
                        print(f"    ✗ Step {step} allowed tokens: FAIL - {detail}")
                        break

                    # Test state sequence
                    match, detail = test_state_sequence(
                        guide_rust, guide_python, state_rust, token_id, tokenizer
                    )
                    results["test_details"]["state_sequence"][regex].append(
                        (match, detail)
                    )
                    pattern_result["total_tests"] += 1
                    results["total_tests"] += 1

                    if match:
                        pattern_result["passed"] += 1
                        results["passed"] += 1
                    else:
                        pattern_result["failed"] += 1
                        results["failed"] += 1
                        pattern_result["failures"].append(
                            ("state_sequence", f"step={step}, {detail}")
                        )
                        results["failures"].append(
                            (regex, f"state_sequence step={step}", detail)
                        )
                        print(f"    ✗ Step {step} state sequence: FAIL - {detail}")
                        break

                    # Update states
                    state_rust = guide_rust.get_next_state(state_rust, token_id)
                    state_python = guide_python.get_next_state(state_python, token_id)

                    if state_rust != state_python:
                        print(
                            f"    ✗ States diverged at step {step}: Rust={state_rust}, Python={state_python}"
                        )
                        break

                if state_rust == state_python:
                    print(f"    ✓ Sequence {seq_idx + 1}: {step + 1} steps, all passed")

        except Exception as e:
            error_msg = f"Exception testing pattern {regex}: {e}"
            print(f"✗ ERROR: {error_msg}")
            pattern_result["failed"] += 1
            pattern_result["failures"].append(("exception", str(e)))
            results["failures"].append((regex, "exception", str(e)))

        results["pattern_results"].append(pattern_result)

    return results


def print_summary(results: dict):
    """Print detailed summary of test results."""
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    print(f"\nOverall Results:")
    print(f"  Total tests: {results['total_tests']}")
    print(
        f"  Passed: {results['passed']} ({results['passed']/results['total_tests']*100:.2f}%)"
    )
    print(
        f"  Failed: {results['failed']} ({results['failed']/results['total_tests']*100:.2f}%)"
    )

    print(f"\nPer-Pattern Results:")
    print(f"{'Pattern':<50} {'Total':>8} {'Passed':>8} {'Failed':>8} {'Pass %':>8}")
    print("-" * 82)

    for pattern_result in results["pattern_results"]:
        # Handle missing 'total_tests' key
        total = pattern_result.get("total_tests", 0)
        passed = pattern_result.get("passed", 0)
        failed = pattern_result.get("failed", 0)
        pass_pct = passed / total * 100 if total > 0 else 0
        pattern_short = (
            pattern_result["regex"][:47] + "..."
            if len(pattern_result["regex"]) > 47
            else pattern_result["regex"]
        )
        print(
            f"{pattern_short:<50} {total:>8} {passed:>8} {failed:>8} {pass_pct:>7.1f}%"
        )

    print("\n" + "=" * 80)
    print("PER TEST TYPE BREAKDOWN")
    print("=" * 80)

    for test_type in [
        "initial_state",
        "get_next_state",
        "is_final_state",
        "get_allowed_tokens",
        "state_sequence",
    ]:
        all_results = [
            item
            for tests in results["test_details"][test_type].values()
            for item in tests
        ]
        total = len(all_results)
        passed = sum(1 for match, _ in all_results if match)
        failed = total - passed
        pass_pct = passed / total * 100 if total > 0 else 0

        print(f"\n{test_type}:")
        print(f"  Total: {total}")
        print(f"  Passed: {passed} ({pass_pct:.2f}%)")
        print(f"  Failed: {failed} ({100-pass_pct:.2f}%)")

    if results["failures"]:
        print("\n" + "=" * 80)
        print(f"FAILURE DETAILS ({len(results['failures'])} failures)")
        print("=" * 80)

        for i, (regex, test_type, detail) in enumerate(results["failures"][:20], 1):
            print(f"\n{i}. Pattern: {regex}")
            print(f"   Test: {test_type}")
            print(f"   Detail: {detail}")

        if len(results["failures"]) > 20:
            print(f"\n... and {len(results['failures']) - 20} more failures")

    print("\n" + "=" * 80)
    if results["failed"] == 0:
        print("✓ ALL TESTS PASSED - Implementations are EQUIVALENT")
    else:
        print("✗ SOME TESTS FAILED - Implementations are NOT equivalent")
    print("=" * 80)


def main():
    """Main test runner."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Test equivalence between DiverseGuide implementations"
    )
    parser.add_argument(
        "-p",
        "--patterns",
        type=int,
        default=None,
        help="Number of patterns to test (default: all)",
    )
    parser.add_argument(
        "-s",
        "--sequences",
        type=int,
        default=10,
        help="Number of sequences per pattern (default: 10)",
    )
    parser.add_argument(
        "-l",
        "--length",
        type=int,
        default=20,
        help="Length of each sequence (default: 20)",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        print(f"Random seed set to: {args.seed}")

    print("=" * 80)
    print("DIVERSE GUIDE IMPLEMENTATION EQUIVALENCE TEST")
    print("=" * 80)
    print(f"\nConfig:")
    print(f"  Patterns to test: {args.patterns or 'all'}")
    print(f"  Sequences per pattern: {args.sequences}")
    print(f"  Sequence length: {args.length}")
    print(f"  Random seed: {args.seed or 'random'}")

    results = run_comprehensive_test(
        num_patterns=args.patterns,
        num_sequences_per_pattern=args.sequences,
        sequence_length=args.length,
    )

    print_summary(results)

    # Exit with appropriate code
    sys.exit(0 if results["failed"] == 0 else 1)


if __name__ == "__main__":
    main()
