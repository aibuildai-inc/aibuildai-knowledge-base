# 5th Place Solution

Competition: konwinski-prize
Rank: #5
Source: https://www.kaggle.com/c/konwinski-prize/writeups/5th-place-solution

Firstly, thanks to Kaggle and Andy for hosting this fun competition. Congratulations to all the participants.

My solution is based on [huikang notebook](https://www.kaggle.com/code/huikang/starter-notebook-select-patch-verify). I very much appreciate his contribution to this competition.

My notebook [here](https://www.kaggle.com/code/manhnguyen315/5th-place-solution).

I have 2 major changes from the public notebook:
- New parse module `get_selection_query_x`(regex base) over old `get_selection_query`(LLM base) to parse error traceback and identify exactly files and code snippets that need editing.
- Use git conflict markers(GCM) format to generate a fix patch and create a diff patch via `git diff .` instead of directly generating a diff patch - same [top 1](https://www.kaggle.com/competitions/konwinski-prize/discussion/568884) approach.



# Skip non error traceback issues
In my tests, the correct location to modify can only be identified if there is an error traceback in the issue. Therefore, I decided to skip all sentences without an error traceback to speed up testing. In addition, based on my public LB exploration, the number of sentences with error traceback is 6 < n < 11. The fact that Eduardo solved 9 sentences on public LB somewhat strengthens this assumption.

**If the issue has a traceback, go to the next step `get_selection_query_x`:**

# Context retrieval

`get_selection_query` in public notebook initially generates LLM-based keyword searches, a problem that arises when LLM suggests a phrase that does not exist in the repo or a phrase that is too simple, leading to too many noisy code snippets.

I built `get_selection_query_x` (regex base) to extract and map the error traceback to the file and error location in the repo. The module consists of 2 main components:
- identify the file path where the error appears in the traceback and map the path to the corresponding location in the new repo
- Extract the entire subtraceback for each file and line location where the error appears

The module is compatible with the error reporting style of Python 3.10, 3.11 and 3.12, ensuring the correct and minimal search for the location that needs to be fixed, if the file or error is not identified -> use the old get_selection_query(LLM base)

Example:

```
problem = """
TypeError: unsupported format string passed to NoneType.__format__
Regression in #2459

### Steps to reproduce
a.py:
\```py
class A:
    def __init__(self):
        self._magnitude = None

    def name(self) -> str | None:
        if self._magnitude:
            return f"M {self._magnitude:.1f}"
\```
\```
pylint a.py
\```
### Current behavior
\```
  File "/Users/jwalls/release/lib/python3.12/site-packages/astroid/nodes/node_classes.py", line 4778, in _infer_from_values
    yield from nodes[0]._infer(context, **kwargs)
  File "/Users/jwalls/release/lib/python3.12/site-packages/astroid/nodes/node_classes.py", line 4695, in _infer
    formatted = format(value.value, format_spec.value)
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
TypeError: unsupported format string passed to NoneType.__format__
\```
"""

print(get_selection_query_x(problem))
# {'/kaggle/working/q1/repo/astroid/nodes/node_classes.py': ['    yield from nodes[0]._infer(context, **kwargs)', '    formatted = format(value.value, format_spec.value)']}
```

# Generate fix patch

My approach uses git conflict markers(GCM) format to generate a fix patch instead of directly generating a diff patch, which makes most of the patches valid. I generate 8 queries for each issue use DeepSeek-R1-Distill-Qwen-14B.

GCM prompt:

```
We are currently solving the following issue within our repository. Here is the issue text:
--- BEGIN ISSUE ---
{problem_statement}
--- END ISSUE ---

Below are some code segments, each from a relevant file. One or more of these files may contain bugs.

--- BEGIN FILE ---
\```
{file_content_string}
\```
--- END FILE ---

Please first localize the bug based on the issue statement, and then generate *SEARCH/REPLACE* edits to fix the issue.

Every *SEARCH/REPLACE* edit must use this format:
1. The file path
2. The start of search block: <<<<<<< SEARCH
3. A contiguous chunk of lines to search for in the existing source code
4. The dividing line: =======
5. The lines to replace into the source code
6. The end of the replace block: >>>>>>> REPLACE

Here is an example:

\```python
### mathweb/flask/app.py
<<<<<<< SEARCH
from flask import Flask
=======
import math
from flask import Flask
>>>>>>> REPLACE
\```

Please note that the *SEARCH/REPLACE* edit REQUIRES PROPER INDENTATION. If you would like to add the line '        print(x)', you must fully write that out, with all those spaces before the code!
Wrap each *SEARCH/REPLACE* edit in a code block as shown in the example above. If you have multiple *SEARCH/REPLACE* edits, use a separate code block for each one.
```

**Create diff patch:** Search/replace is extracted, then make direct edits to the repo(I created a fake repo) then generate a valid diff patch using the git diff command.

Create a fake repo -> Apply all edit commands -> Commit -> run !git diff .

# Evaluate:

I simply chose the first valid patch to submit. My second submission which included all the steps was randomly broken forcing me to create a simpler submission.

# In summary:

The approach is based on a powerful search module and creating a solution using the git conflict markers format. **No run test, no evaluation, don't need run env setup.**

# What doesn't work:

- check and fix the number of diff patches by counting the number of lines added and removed -> no improvement

- Larger model: DeepSeek-R1-Distill-Qwen-32B did not improve in my tests, I haven't tested many other models yet as I joined the contest quite late.

# Acknowledgement:

- @huikang for his great contributions to the contest.
- Many forum discussions.
