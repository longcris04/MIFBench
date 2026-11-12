"""mifkit -- the shared benchmark pipeline for multi-modal medical image fusion.

One training loop, one data pipeline and one scorer for every model in the
benchmark.  Each model directory keeps its authors' code untouched and adds a
``mif_adapter.py`` that tells the pipeline how to build, train and run it.

    python -m mifkit.cli.train --model DATFuse --pair CT-MRI --fold 1
    python -m mifkit.cli.run_fold --fold 1                 # every model x pair of one fold
    python -m mifkit.cli.summary                           # cross-fold tables

See ``README.md`` at the repository root.
"""
