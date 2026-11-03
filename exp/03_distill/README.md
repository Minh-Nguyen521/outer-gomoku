# Method 3: Distillation from KataGomo (teacher labels)

Reuse the existing pipeline but pointed at the HARVEST data (fresh,
Outer-Open-legal, diverse openings) instead of the old contaminated set
described in docs/value-head.md:

  1. `sample_positions.py` -- extract positions from harvest jsonl into
     the format label_positions.py consumes (`--in`).
  2. `label_positions.py` (existing tool) -- KataGomo evaluates each
     position: full 225-way policy + value, no search.
  3. `train.py` (wrapper over tools/train_policy.py) -- distil.

Teacher labels >> one-hot self labels: the policy DISTRIBUTION carries
ranked alternatives, so the same data volume teaches far more.

    ./run.sh                       # full pipeline attempt
    ./eval.sh                      # net vs SlowRenju ladder engine

Requires KataGomo built (third_party/KataGomo/kg-run) and its model/config
paths -- see tools/label_positions.py --help if paths differ on your box.
