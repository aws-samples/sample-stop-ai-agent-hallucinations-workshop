# Workshop Utilities

This directory contains repository-wide utilities and sorts after the numbered
workshop labs in a normal directory listing.

## Run the notebooks

`run_notebooks.py` executes the workshop notebooks with `nbconvert` and reports
pass, fail, or skip for each notebook. While a notebook runs, it reports each
code cell's source preview and elapsed time, followed by the cell's captured
text output. Run it from the repository root:

```bash
uv run setup/run_notebooks.py                 # Labs 00-05; 06-08 are skipped
uv run setup/run_notebooks.py --labs 4        # One lab
uv run setup/run_notebooks.py --labs 2,3,4    # A list
uv run setup/run_notebooks.py --labs 2-5      # A range
uv run setup/run_notebooks.py --list          # Show the notebook registry
```

The runner is a PEP 723 script. `uv` creates and caches its shared environment,
including `nbconvert` and all notebook dependencies, on the first run. No
top-level virtual environment or separate installation step is required.

The original notebook files are never modified. Executed notebooks are written
to a temporary directory and removed after the run. To retain them for
inspection, use:

```bash
uv run setup/run_notebooks.py --labs 4 --keep-output
```

Retained notebooks are stored under `setup/notebook-output/`, which is ignored
by Git.

### AWS side effects

The default command runs labs 00 through 05. The remaining notebooks require
explicit flags because they change AWS resources:

```bash
# Deploy or update the AgentCore resources in labs 06 and 07
uv run setup/run_notebooks.py --labs 6,7 --include-deploy

# Delete the tagged workshop resources in lab 08
uv run setup/run_notebooks.py --labs 8 --include-cleanup
```

The runner treats an uncaught cell error as a failure and exits nonzero when
any selected notebook fails. A clean execution does not validate narrative or
model-quality claims unless the notebook contains assertions for those claims.

### Interactive notebook output

The runner protects source notebooks because it executes in-memory copies.
When notebooks are run interactively in an IDE, Git output stripping is still
handled by the repository's `nbstripout` filter. Register it once after cloning
as described in the root README.
