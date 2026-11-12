"""Train one model on one fold of one pair; fuse and score the test split after every epoch.

    python -m mifkit.cli.train --model DATFuse --pair PET-MRI --fold 1
    python -m mifkit.cli.train --model DATFuse --pair CT-MRI --fold 2 --lr 3e-3 \\
        --out models/DATFuse/mif_runs_lr3e-3/CT-MRI/fold2

Run with --help --model <MODEL> to see the model's own options.
"""

from ..pipeline import parse_args, train


def main(argv=None):
    args, adapter = parse_args(argv)
    train(args, adapter)


if __name__ == "__main__":
    main()
