"""Main benchmark runner."""

import json
import time
import argparse
from pathlib import Path

from adbench.problems import PROBLEMS, get_problems_by_category, get_problems_by_level
from adbench.problem import Problem
from adbench.extract import extract_code
from adbench.sandbox import execute_code
from adbench.llm import query_model


def run_problem(problem: Problem, provider: str, model: str | None, verbose: bool = False) -> dict:
    """Run a single problem: query LLM, extract code, execute, grade."""
    result = {
        "id": problem.id,
        "category": problem.category,
        "difficulty": problem.difficulty,
    }

    # Query LLM
    t0 = time.time()
    try:
        response = query_model(problem.prompt, provider=provider, model=model)
        result["response_time"] = time.time() - t0
        result["raw_response"] = response
    except Exception as e:
        result["error"] = f"LLM query failed: {e}"
        result["score"] = 0.0
        return result

    # Extract code
    code = extract_code(response)
    if code is None:
        result["error"] = "Could not extract 'solve' function from response"
        result["score"] = 0.0
        if verbose:
            print(f"  [FAIL] {problem.id}: no code extracted")
        return result
    result["extracted_code"] = code

    # Execute code
    solve_fn, exec_error = execute_code(code, timeout=30)
    if solve_fn is None:
        result["error"] = exec_error
        result["score"] = 0.0
        if verbose:
            print(f"  [FAIL] {problem.id}: {exec_error}")
        return result

    # Grade
    grade_result = problem.grade(solve_fn)
    result.update(grade_result)

    if verbose:
        status = "PASS" if grade_result["score"] == 1.0 else "PARTIAL" if grade_result["score"] == 0.5 else "FAIL"
        print(f"  [{status}] {problem.id}: {grade_result['n_passed']}/{grade_result['n_total']} test cases")

    return result


def run_benchmark(
    provider: str = "anthropic",
    model: str | None = None,
    categories: list[str] | None = None,
    levels: list[int] | None = None,
    problem_ids: list[str] | None = None,
    output_file: str | None = None,
    verbose: bool = True,
) -> dict:
    """Run the full benchmark or a subset."""
    # Select problems
    if problem_ids:
        problems = [PROBLEMS[pid] for pid in problem_ids if pid in PROBLEMS]
    else:
        problems = list(PROBLEMS.values())
        if categories:
            problems = [p for p in problems if p.category in categories]
        if levels:
            problems = [p for p in problems if p.difficulty in levels]

    if not problems:
        print("No problems matched the filters.")
        return {}

    print(f"Running {len(problems)} problems with {provider}" + (f" ({model})" if model else ""))
    print("=" * 60)

    results = []
    for i, problem in enumerate(problems):
        print(f"[{i+1}/{len(problems)}] {problem.id} (L{problem.difficulty}, {problem.category})")
        result = run_problem(problem, provider, model, verbose=verbose)
        results.append(result)

    # Summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    total_score = sum(r.get("score", 0) for r in results)
    max_score = len(results)
    print(f"Overall: {total_score:.1f} / {max_score} ({100*total_score/max_score:.1f}%)")

    # By category
    cat_scores = {}
    for r in results:
        cat = r["category"]
        if cat not in cat_scores:
            cat_scores[cat] = {"total": 0, "count": 0}
        cat_scores[cat]["total"] += r.get("score", 0)
        cat_scores[cat]["count"] += 1

    print("\nBy category:")
    for cat, s in sorted(cat_scores.items()):
        pct = 100 * s["total"] / s["count"]
        print(f"  {cat:25s}: {s['total']:.1f}/{s['count']} ({pct:.0f}%)")

    # By difficulty
    level_scores = {}
    for r in results:
        lvl = r["difficulty"]
        if lvl not in level_scores:
            level_scores[lvl] = {"total": 0, "count": 0}
        level_scores[lvl]["total"] += r.get("score", 0)
        level_scores[lvl]["count"] += 1

    print("\nBy difficulty:")
    for lvl in sorted(level_scores):
        s = level_scores[lvl]
        pct = 100 * s["total"] / s["count"]
        print(f"  Level {lvl}: {s['total']:.1f}/{s['count']} ({pct:.0f}%)")

    # Save results
    output = {
        "provider": provider,
        "model": model,
        "n_problems": len(results),
        "total_score": total_score,
        "max_score": max_score,
        "accuracy": total_score / max_score if max_score > 0 else 0,
        "by_category": cat_scores,
        "by_difficulty": level_scores,
        "results": results,
    }

    if output_file:
        # Strip raw responses for a cleaner saved version (they can be huge)
        save_output = json.loads(json.dumps(output, default=str))
        Path(output_file).write_text(json.dumps(save_output, indent=2, default=str))
        print(f"\nResults saved to {output_file}")

    return output


def run_dry(verbose: bool = True) -> dict:
    """Dry run: test all reference solutions against their own test cases."""
    print(f"Dry run: testing {len(PROBLEMS)} reference solutions")
    print("=" * 60)

    failures = []
    for pid, problem in PROBLEMS.items():
        grade = problem.grade(problem.reference)
        status = "OK" if grade["score"] == 1.0 else "PARTIAL" if grade["score"] == 0.5 else "FAIL"
        if verbose or status != "OK":
            print(f"  [{status}] {pid}: {grade['n_passed']}/{grade['n_total']}")
        if grade["score"] < 1.0:
            failures.append((pid, grade))
            if verbose:
                for d in grade["details"]:
                    if not d["passed"]:
                        print(f"    inputs={d['inputs']}, expected={d['expected']}, got={d['computed']}, err={d['error']}")

    print(f"\n{len(PROBLEMS) - len(failures)}/{len(PROBLEMS)} reference solutions passed all test cases.")
    if failures:
        print(f"Failures: {[f[0] for f in failures]}")

    return {"n_total": len(PROBLEMS), "n_passed": len(PROBLEMS) - len(failures), "failures": failures}


def main():
    parser = argparse.ArgumentParser(description="AD-Bench: Automatic Differentiation Benchmark")
    parser.add_argument("--provider", default="anthropic", choices=["anthropic", "openai"],
                        help="LLM provider")
    parser.add_argument("--model", default=None, help="Model name (default: provider default)")
    parser.add_argument("--category", nargs="*", help="Filter by category")
    parser.add_argument("--level", nargs="*", type=int, help="Filter by difficulty level")
    parser.add_argument("--problems", nargs="*", help="Specific problem IDs to run")
    parser.add_argument("--output", "-o", default=None, help="Output JSON file")
    parser.add_argument("--dry-run", action="store_true", help="Test reference solutions only")
    parser.add_argument("--list", action="store_true", help="List all problems")
    parser.add_argument("--quiet", action="store_true", help="Less verbose output")

    args = parser.parse_args()

    if args.list:
        print(f"{'ID':40s} {'Category':20s} {'Level':6s}")
        print("-" * 70)
        for pid, p in sorted(PROBLEMS.items()):
            print(f"{pid:40s} {p.category:20s} L{p.difficulty}")
        print(f"\nTotal: {len(PROBLEMS)} problems")
        return

    if args.dry_run:
        run_dry(verbose=not args.quiet)
        return

    run_benchmark(
        provider=args.provider,
        model=args.model,
        categories=args.category,
        levels=args.level,
        problem_ids=args.problems,
        output_file=args.output,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
