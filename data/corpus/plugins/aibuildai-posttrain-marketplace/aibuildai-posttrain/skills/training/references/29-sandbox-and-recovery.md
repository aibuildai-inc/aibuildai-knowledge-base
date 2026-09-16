# Sandbox, repeatable builds, and failure recovery

Read in step 5 before executing any generated code, and in step 11 for operational failures.

On code tasks, the training data and the reward both run code that the model wrote.

Suppose a run executes candidate code straight inside the task directory early on, and the candidate's file operations wipe `data/`. After that, the run:

- rebuilds the data in a deterministic way;
- runs every untrusted execution with `cwd=TemporaryDirectory()`;
- sets a timeout;
- captures stdout and stderr;
- does not inherit the task directory.

### The smallest sandbox

```text
temporary working directory
subprocess timeout
restricted environment variables
no task-directory write access
bounded stdout/stderr
process-group kill
no inherited secrets
deterministic rebuild script
```

You must also handle:

- a full root disk;
- where the Hugging Face cache sits;
- a zombie vLLM process;
- GPU memory that was never freed;
- a checkpoint written only half way;
- a pip install that breaks the environment;
- a background job that failed while the run did not notice.

This kind of operations skill decides directly whether a complex RFT or GRPO pipeline can finish at all. The methodology skill's cards on RFT and GRPO carry the full methods; this card keeps only what the workflow needs.
