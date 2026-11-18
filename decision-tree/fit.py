"""Clinic no-shows: should the clinic overbook this slot?

Run from this folder:  python fit.py
Fits a classification tree on clinic_appointments.csv (400 rows, 70/30 split
stratified on no-shows). Saves tree_v1.png.
"""

import matplotlib

matplotlib.use("Agg")  # save charts to files instead of opening a window
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

FEATURES = ["lead_time_days", "prior_no_shows", "age", "distance_miles",
            "sms_reminder", "new_patient", "appointment_hour"]
DEPTH = 3  # deep enough to combine a few causes, shallow enough to explain to a manager

# Dollar value of each cell (from the brief), per appointment
VALUE = {"TN": 0, "FP": -60, "FN": -120, "TP": 0}
APPOINTMENTS_PER_YEAR = 20_000  # ASSUMPTION: ~80 appointments a day x 250 clinic days

# ---------------------------------------------------------------- 1. Split
df = pd.read_csv("clinic_appointments.csv")
X = df[FEATURES]
y = (df["no_show"] == "no-show").astype(int)  # 1 = no-show, 0 = checked in
print(f"df.shape {df.shape}   no-shows {y.sum()} ({y.mean():.1%})")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
print(f"Train {len(X_train)} rows ({y_train.mean():.1%} no-show)   "
      f"Test {len(X_test)} rows ({y_test.mean():.1%} no-show, {y_test.sum()} no-shows)")

# ---------------------------------------------------------------- 2. Fit
v1 = DecisionTreeClassifier(max_depth=DEPTH, random_state=42).fit(X_train, y_train)
print(f"\n=== v1 tree (depth {DEPTH}, {v1.get_n_leaves()} leaves), training rows ===")
print(export_text(v1, feature_names=FEATURES, show_weights=True))
fig = plt.figure(figsize=(14, 7))
plot_tree(v1, feature_names=FEATURES, class_names=["shows", "no-show"], filled=True, fontsize=9)
fig.savefig("tree_v1.png", dpi=200, bbox_inches="tight")


# ---------------------------------------------------------------- 3. Evaluate
def cells(actual, predicted):
    tn, fp, fn, tp = confusion_matrix(actual, predicted, labels=[0, 1]).ravel()
    return {"TN": tn, "FP": fp, "FN": fn, "TP": tp}


def report(name, actual, predicted):
    c = cells(actual, predicted)
    dollars = sum(VALUE[k] * c[k] for k in c)
    return {
        "model": name, **c,
        "accuracy": accuracy_score(actual, predicted),
        "precision (no-show)": precision_score(actual, predicted, zero_division=0),
        "recall (no-show)": recall_score(actual, predicted),
        "total value $": dollars,
        "value per appt $": dollars / len(actual),
    }


baseline_pred = [0] * len(y_test)  # most common class: everyone shows, hold every slot
results = pd.DataFrame([
    report("baseline (hold every slot)", y_test, baseline_pred),
    report("v1 tree", y_test, v1.predict(X_test)),
])
pd.set_option("display.width", 200)
print("=== Test appointments: baseline vs v1 ===")
print(results.round(3).to_string(index=False))
print(f"\nv1 accuracy: training {accuracy_score(y_train, v1.predict(X_train)):.3f}   "
      f"test {accuracy_score(y_test, v1.predict(X_test)):.3f}")


# ---------------------------------------------------------------- 4. Decide
def leaf_path(appointment_df):
    """The questions this appointment answers on its way down the tree."""
    t = v1.tree_
    node, steps = 0, []
    while t.children_left[node] != -1:
        name, cut = FEATURES[t.feature[node]], t.threshold[node]
        value = appointment_df.iloc[0][name]
        go_left = value <= cut
        steps.append(f"{name} = {value} {'<=' if go_left else '>'} {cut:.2f}")
        node = t.children_left[node] if go_left else t.children_right[node]
    shows, no_shows = (t.value[node][0] * t.n_node_samples[node]).round().astype(int)
    return node, steps, shows, no_shows


def should_overbook(appointment):
    """Rule: overbook if v1 predicts a no-show, hold if it predicts a show."""
    row = pd.DataFrame([appointment])[FEATURES]
    return "overbook" if v1.predict(row)[0] == 1 else "hold"


upcoming = {
    "D": dict(lead_time_days=21, prior_no_shows=1, age=27, distance_miles=9.5,
              sms_reminder=0, new_patient=1, appointment_hour=15),
    "E": dict(lead_time_days=3, prior_no_shows=0, age=61, distance_miles=2.0,
              sms_reminder=1, new_patient=0, appointment_hour=9),
}
print("\n=== Decisions for upcoming appointments ===")
for name, appt in upcoming.items():
    node, steps, shows, no_shows = leaf_path(pd.DataFrame([appt])[FEATURES])
    print(f"{name}: {' -> '.join(steps)}")
    print(f"   leaf {node}: {no_shows} no-shows of {shows + no_shows} training appointments "
          f"({no_shows / (shows + no_shows):.0%})  ->  {should_overbook(appt).upper()}")

# ---------------------------------------------------------------- 5. Value
print(f"\n=== Value (assumed {APPOINTMENTS_PER_YEAR:,} appointments a year) ===")
for _, r in results.iterrows():
    print(f"{r['model']}: {r['total value $']:,.0f} $ on {len(y_test)} test appointments, "
          f"{r['value per appt $']:,.2f} $ each, {r['value per appt $'] * APPOINTMENTS_PER_YEAR:,.0f} $ a year")
saved = (results.loc[1, "value per appt $"] - results.loc[0, "value per appt $"]) * APPOINTMENTS_PER_YEAR
print(f"v1 vs baseline: {saved:,.0f} $ a year")
