"""Baseline model: predictions that ignore which teams are playing.

    uniform_baseline   - pure chance: 1/3 home win, 1/3 draw, 1/3 away win
    freq_baseline      - historical shares of H/D/A in the TRAINING seasons

Every other model should beat these.
"""
import numpy as np

from helper import CLASSES, LABEL, load_data, save_results, split

# Load the matches and split them by season
df, features = load_data("compact")
train, valid, test = split(df)
print(f"Matches: train={len(train)}  valid={len(valid)}  test={len(test)}\n")

# Uniform baseline: 1/3, 1/3, 1/3 for every match
uniform = np.array([1 / 3, 1 / 3, 1 / 3])

# Frequency baseline: share of each outcome in the training seasons only
freq = train[LABEL].value_counts(normalize=True).reindex(CLASSES, fill_value=0).values
print("Training-season outcome shares:")
for c, p in zip(CLASSES, freq):
    print(f"  {c}: {p:.1%}")
print()

# Give every match in a split the same probabilities, then score and save
for split_name, data in [("valid", valid), ("test", test)]:
    for model_name, p in [("uniform_baseline", uniform), ("freq_baseline", freq)]:
        probs = np.tile(p, (len(data), 1))   # repeat the row once per match
        save_results(model_name, split_name, data, probs)