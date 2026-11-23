# Manuscript code release

Suggested repository name: `pFLEDL` under `WahabMam` (not created or verified available).

## Remaining release work

- Add the exact paper title, authors, and citation.
- Restore CIFAR-100/Tiny-ImageNet partition-generation instructions and scripts needed for the paper.
- Preserve upstream attribution and supply the full applicable GPL license text.
- Freeze and document the source/settings for every manuscript ablation; the working code is still changing.
- Make notebook paths relative and bundle a deliberately selected set of timing CSVs for analysis reproduction.
- Validate installation and a small run in a clean environment, then record dependencies.
- Review the publication file list; generated datasets, virtual environments, raw logs, and checkpoints are excluded by default.
- Create the GitHub repository, upload the reviewed files, and verify public access before adding availability claims.
- Tag the tested paper revision so the manuscript points to a stable version.

## Wording after public release

Manuscript (replace URL with the actual public repository):

```latex
The source code and reproduction instructions are publicly available at
\url{https://github.com/WahabMam/pFLEDL}.
```

Reviewer response (use only after the stated materials are actually released):

> We have made the implementation publicly available and added the repository link to the revised manuscript. The repository provides installation instructions, dataset preparation details, and commands for reproducing the experiments, including the three-seed evaluation protocol.

Do not claim that scripts, archived data, or analysis reproduction are included until they have been added and checked.
