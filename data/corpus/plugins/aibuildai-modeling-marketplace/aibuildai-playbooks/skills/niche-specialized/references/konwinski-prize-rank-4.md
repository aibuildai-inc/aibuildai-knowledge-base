# 4th Place Solution

Competition: konwinski-prize
Rank: #4
Source: https://www.kaggle.com/c/konwinski-prize/writeups/4th-place-solution

I would like to express my sincere appreciation to the Kaggle team and the competition organizers for creating such a highly engaging and intellectually stimulating challenge. I am also deeply grateful to @huikang for their invaluable contributions through the [Starter Notebook - Select-Patch-Verify](https://www.kaggle.com/code/huikang/starter-notebook-select-patch-verify) and the [LB calculator](https://www.kaggle.com/competitions/konwinski-prize/discussion/557148). Their work provided a strong foundation for my approach and significantly enhanced the overall experience of the competition.

This competition presented a unique and complex challenge: balancing model accuracy with the validity of generated patches, akin to walking a tightrope with limited visibility. As model performance improved, the risk of generating patches that *appeared* correct but were fundamentally flawed increased.

## TL;DR

Building on the foundational work from @huikang’s [Starter Notebook - Select-Patch-Verify](https://www.kaggle.com/code/huikang/starter-notebook-select-patch-verify), I explored two distinct strategies in my final submission, each with its own advantages and trade-offs:

| Strategy                                                                                                               | Private LB Score                                | Public LB Score                                |
| ---------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------- | ---------------------------------------------- |
| [Select-Patch-Verify-Test (Winning)](https://www.kaggle.com/code/genxxsky/4th-place-solution-select-patch-verify-test) | `0.016571` <br> (4 correct, 2 wrong, 114 skip)  | `-0.000097` <br> (1 correct, 1 wrong, 69 skip) |
| [Select-Patch-Verify-Choose](https://www.kaggle.com/code/genxxsky/kprice-select-patch-verify-choose)                   | `-0.008429` <br> (2 correct, 3 wrong, 115 skip) | `-0.000094` <br> (2 correct, 2 wrong, 67 skip) |

[Pipeline of Select-Patch-Verify-Test]

- **Select-Patch-Verify-Test (Winning Submission)**:
  - Prioritizes **efficiency** by reducing the number of patch attempts and thinking time.
  - Utilizes **pytest code generation and testing** for each candidate patch.
  - Performs better in cases where the issue is relatively simple.
  - [Code](https://www.kaggle.com/code/genxxsky/4th-place-solution-select-patch-verify-test)

- **Select-Patch-Verify-Choose**:
  - Takes a more **exploratory** approach.
  - Uses more tokens and attempts to generate and verify multiple patches.
  - Selects the best option based on a multi-dimensional evaluation.
  - Involves higher computational overhead and requires further optimization.
  - Shows better performance in more complex issue, but may take up a lot of time in some issues.
  - [Code](https://www.kaggle.com/code/genxxsky/kprice-select-patch-verify-choose)

### Key Steps in the Pipeline

- **Select**: Search for relevant code snippets using **tree-sitter**.
- **Patch**: Generate diff patches based on the selected code.
- **Verify**: Validate patches using **LLM self-check**.
- **Test**: Generate and execute **pytest** code for testing.
- **Choose**: Compare multiple patch options using a multi-dimensional evaluation.

## Key Improvements

### Enhanced Code Contextualization

By leveraging **tree-sitter** for function- and class-level code analysis, I was able to retrieve richer contextual information for the model. This improved the model’s understanding of the code around a specific line of interest.

Here is a sample of the code context retrieved for line 4778:

```
[file name]: astroid/nodes/node_classes.py
[terms searched]:
line(4778)
string_to_search(formatted = format(value.value, format_spec.value))
string_to_search(if value is not None:)
[file relevant content begin]

Match #1, lines 4708 to 4791:
  4708|class JoinedStr(NodeNG):
  4709|    """Represents a list of string expressions to be joined.
  4710|
  4711|    >>> import astroid
  4712|    >>> node = astroid.extract_node('f"Format {type_}"')
  4713|    >>> node
  4714|    <JoinedStr l.1 at 0x7f23b2e4ed30>
  4715|    """
  ...|
  4719|    def __init__(
  4720|        self,
  4721|        lineno: int | None = None,
  4722|        col_offset: int | None = None,
  4723|        parent: NodeNG | None = None,
  4724|        *,
  4725|        end_lineno: int | None = None,
  4726|        end_col_offset: int | None = None,
  4727|    ) -> None:
  ...|
  4773|    @classmethod
  4774|    def _infer_from_values(
  4775|        cls, nodes: list[NodeNG], context: InferenceContext | None = None, **kwargs: Any
  4776|    ) -> Generator[InferenceResult, None, InferenceErrorInfo | None]:
  4777|        if len(nodes) == 1:
  4778|            yield from nodes[0]._infer(context, **kwargs)
  4779|            return
  4780|        uninferable_already_generated = False
  4781|        for prefix in nodes[0]._infer(context, **kwargs):
  4782|            for suffix in cls._infer_from_values(nodes[1:], context, **kwargs):
  4783|                result = ""
  4784|                for node in (prefix, suffix):
  4785|                    if isinstance(node, Const):
  4786|                        result += str(node.value)
  4787|                        continue
  4788|                    result += MISSING_VALUE
  4789|                if MISSING_VALUE in result:
  4790|                    if not uninferable_already_generated:
  4791|                        uninferable_already_generated = True

[file relevant content end]
```

### Automated Test Script Generation

I enabled the model to generate a **pytest script** containing two test functions based on the problem statement:

- `test_before_patch`: Should pass before the issue is fixed and fail after the fix.
- `test_after_patch`: Should fail before the fix and pass after the fix.

Here is a sample of the generated test script:

```python
import pytest
from astroid.nodes.node_classes import FormattedValue, Const


def test_before_patch():
    # Test that without the patch, trying to format a None value raises a TypeError
    value_node = Const(None)
    format_spec_node = Const('.1f')
    formatted_value = FormattedValue(value=value_node, format_spec=format_spec_node)
    try:
        list(formatted_value.infer())
    except TypeError:
        # Test passes if TypeError is raised (before the patch)
        pass
    else:
        # Test fails if no TypeError is raised (after the patch)
        pytest.fail("Expected TypeError was not raised")


def test_after_patch():
    # Test that with the patch, formatting a None value returns None without errors
    value_node = Const(None)
    format_spec_node = Const('.1f')
    formatted_value = FormattedValue(value=value_node, format_spec=format_spec_node)
    inferred = list(formatted_value.infer())
    # After the patch, the result should be a single Const with value None
    assert len(inferred) == 1
    assert inferred[0].value is None
```

---

## Additional Improvements

- **Efficiency Enhancements**: Implemented **parallel processing**, **token compression**, **caching**, and **early stopping** to improve pipeline efficiency.
- **Confidence Assessment**: Rapid evaluation of model confidence allowed for the prioritization of high-quality patches, improving the final output.
- **Rule-Based Diff Fixes**: Adjustments to diffs using rule-based methods had a random impact on final scores.
- **Code Analysis Techniques**: Methods such as multi-method code lookup, multi-step analysis, and cross-attempt code sharing did not yield meaningful gains.
- **Prompt and Parameter Variations**: Varying prompts or parameters between attempts, or extending patch generation, did not consistently improve performance.
- **Patch Validation Methods**: Techniques like **AST-based checking** or reusing existing pytest functions showed minimal impact on performance.

---

## Reflections and Lessons Learned

This competition marked my first **Kaggle gold medal** — a significant milestone that has greatly motivated me. However, there are several areas for improvement. Time constraints limited the exploration of promising ideas, and a major code refactoring during optimization led to inconsistent application of improvements across both strategies. Many changes lacked thorough **ablation studies**, which I recognize as a critical area for improvement in future workflows.

Moving forward, I am committed to refining these approaches, applying the insights gained to future challenges, and adopting a more systematic and rigorous development process to maximize performance and innovation.
