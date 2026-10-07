# Suicide notes as structured final communications: analysis code

This folder is the code-only GitHub upload package. It contains analysis scripts,
reference code, synthetic tests and dependency/configuration templates. It does
not contain study data, annotations, model weights, manuscript documents,
generated features, results, figures, or execution logs.

Upload only the contents of `github_upload`. The adjacent `manuscript_analysis`
folder is the local research workspace and must stay off GitHub. Existing raw
inputs elsewhere on the computer remain outside this package.

## Local setup

Use Python 3.10 or newer and R. Dependency lists are in `environment/`.
Copy `config/paths.example.json` to `config/paths.json`,
`config/paths.rebuilt.example.json` to `config/paths.rebuilt.json`, and
`config/runtime.example.json` to `config/runtime.json`. Edit the ignored copies
to point to your local inputs and absolute interpreter paths. The example input
paths use a separate sibling `private_data/inputs` folder; the actual files must
be supplied locally. No private input is distributed with this repository.

`python run.py list` lists stages. `python run.py check` checks input presence
and syntax. Run stages individually; demographic stages 01–05 and text stages
10–13, then 17, have dependencies in that order. `--inputs rebuilt` selects new
primary features and separate outputs. The default saved profile expects local
saved study snapshots. It does not silently supply them.

The launcher creates ignored output directories. Keep local data, caches, model
weights, and generated outputs out of commits. Git exclusions do not apply to
GitHub's browser file uploader: upload this clean snapshot before running it,
or upload only the tracked code/configuration templates afterward.

## Verification and limitations

`python -m pytest tests/` uses synthetic inputs; no registry records are needed.
The reference manifest verifies only the public reference-code copies. Those
copies have personal filesystem paths replaced with placeholders; the original
research files remain unchanged in the local workspace.

Primary floor/remainder segmentation and robustness midpoint bins are separate.
Current and older saved classifier cohorts differ. Complete reproduction of
every submitted manuscript value has not been established. The detailed local
audit and manuscript comparisons are retained in the research workspace and are
not part of this public package. Independent full-study weak-label validation
and fresh NLP feature extraction still need verification.

## Before uploading

Run `python tools/check_upload_contents.py` on this snapshot. It checks an
explicit file inventory and rejects additional files, including extensionless
datasets. Run it again after adding or generating files. No upload, Git commit,
or remote repository is created by these scripts.
