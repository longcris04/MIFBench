"""Split every pair into 4 folds (25 % test each).

The benchmark's splits are shipped in splits/ (seed 42); this rebuilds them from data/.

    python -m mifkit.cli.make_folds                 # all pairs, seed 42
    python -m mifkit.cli.make_folds --pairs CT-MRI --seed 7

Writes splits/<PAIR>.json.  See mifkit/folds.py for the stratification.
"""

import argparse
from collections import Counter

from ..folds import DEFAULT_SEED, case_of, make_folds
from ..paths import N_FOLDS, PAIRS


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pairs", nargs="*", default=list(PAIRS), choices=list(PAIRS))
    parser.add_argument("--folds", type=int, default=N_FOLDS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args(argv)

    for pair in args.pairs:
        payload = make_folds(pair, args.folds, args.seed)
        print(f"{pair}: {payload['n_images']} images, {len(payload['cases'])} case groups, seed {args.seed}")
        for fold, split in payload["folds"].items():
            n_test, n_train = len(split["test"]), len(split["train"])
            per_case = Counter(case_of(s) or "3-digit" for s in split["test"])
            print(f"  fold {fold}: train {n_train:3d}  test {n_test:3d} "
                  f"({100 * n_test / (n_test + n_train):.1f} %)  test per case: "
                  + ", ".join(f"{c}:{n}" for c, n in sorted(per_case.items())))


if __name__ == "__main__":
    main()
