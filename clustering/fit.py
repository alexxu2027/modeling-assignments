"""Gym member segments: who should get the $10 monthly credit?

Run from this folder:  python fit.py
Fits k-means on gym_members.csv (330 members, 70/30 split; scaling fitted on
training members only). Saves elbow.png.
"""

import matplotlib

matplotlib.use("Agg")  # save charts to files instead of opening a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Everything that describes how a member uses the gym and who they are.
# member_id is left out: it is a label, not a trait.
FEATURES = ["age", "visits_per_month", "avg_visit_minutes", "classes_per_month",
            "months_member", "monthly_extras_spend"]
K = 4  # chosen from the elbow plot (see the printed spreads)

# Finance's assumptions, applied member by member
CREDIT_COST = 10
SPEND_CUTOFF = 20  # members spending this much or more would spend it anyway
NET_IF_LOW_SPENDER = 5  # +$25 spend x 60% margin = $15, minus the $10 credit
NET_IF_HIGH_SPENDER = -10  # pure giveaway

pd.set_option("display.width", 200)

# ---------------------------------------------------------------- 1. Split
df = pd.read_csv("gym_members.csv")
print(f"df.shape {df.shape}   average extras spend ${df['monthly_extras_spend'].mean():.2f}")
train, test = train_test_split(df, test_size=0.3, random_state=42)
print(f"Train {len(train)} members   Test {len(test)} members")

scaler = StandardScaler().fit(train[FEATURES])  # fitted on training members only
Z_train = scaler.transform(train[FEATURES])
Z_test = scaler.transform(test[FEATURES])

# ---------------------------------------------------------------- 2-3. Elbow, train and test
models, rows = {}, []
for k in range(1, 9):
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(Z_train)
    models[k] = km
    # test spread: assign each test member to its nearest TRAINING center, no refit
    nearest = km.cluster_centers_[km.predict(Z_test)]
    test_spread = float(((Z_test - nearest) ** 2).sum())
    rows.append({"k": k, "train spread": km.inertia_, "test spread": test_spread})
elbow = pd.DataFrame(rows)
elbow["train drop %"] = (-elbow["train spread"].pct_change() * 100)
elbow["test drop %"] = (-elbow["test spread"].pct_change() * 100)
print("\n=== Within-group spread by k (sum of squared distances, scaled units) ===")
print(elbow.round(1).to_string(index=False))

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(elbow["k"], elbow["train spread"], "o-", color="#1f4e79", label=f"training ({len(train)} members)")
ax.plot(elbow["k"], elbow["test spread"], "s--", color="#c55a11", label=f"test ({len(test)} members, training centers)")
ax.axvline(K, color="#999999", lw=0.8)
ax.set_xlabel("Number of groups (k)")
ax.set_ylabel("Within-group spread")
ax.set_title("Elbow plot")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig("elbow.png", dpi=200)

k1, kK = elbow.set_index("k").loc[1, "test spread"], elbow.set_index("k").loc[K, "test spread"]
print(f"\nTest spread: k=1 (baseline) {k1:.1f}   k={K} (v1) {kK:.1f}   ({1 - kK / k1:.0%} lower)")

# ---------------------------------------------------------------- 2. Describe the groups
v1 = models[K]
centers = pd.DataFrame(scaler.inverse_transform(v1.cluster_centers_), columns=FEATURES)
centers["members (train)"] = np.bincount(v1.labels_, minlength=K)
print(f"\n=== v1: {K} group centers in original units (training members) ===")
print(centers.round(1).to_string())


# ---------------------------------------------------------------- 4. Decide
def member_net(spend):
    return np.where(spend < SPEND_CUTOFF, NET_IF_LOW_SPENDER, NET_IF_HIGH_SPENDER)


# A segment gets the credit if, on the training members, giving it to everyone
# in that segment nets the gym money under finance's assumptions.
train_seg = v1.labels_
seg_net = pd.Series(member_net(train["monthly_extras_spend"].to_numpy())).groupby(train_seg).sum()
GETS_CREDIT = {seg: bool(seg_net[seg] > 0) for seg in range(K)}

# Names written from the centers printed above
# (segment numbers are fixed by random_state=42; re-check these if K or the features change)
NAMES = {
    0: "New and casual",  # youngest, newest, ~3 visits, ~$9 extras
    1: "Long-time seniors",  # ~64, members ~5 years, steady visits, ~$17 extras
    2: "Solo grinders",  # most visits and longest workouts, almost no classes
    3: "Class-loving spenders",  # ~7 classes a month, ~$47 extras
}


def segment_name(seg):
    return NAMES.get(seg, f"segment {seg}")


def segment_and_credit(member):
    """Takes one member's features, returns (segment number, name, gets the credit?)."""
    z = scaler.transform(pd.DataFrame([member])[FEATURES])
    seg = int(v1.predict(z)[0])
    return seg, segment_name(seg), GETS_CREDIT[seg]


new_member = dict(age=30, visits_per_month=4, avg_visit_minutes=42, classes_per_month=1,
                  months_member=5, monthly_extras_spend=10)
seg, name, credit = segment_and_credit(new_member)
z_new = scaler.transform(pd.DataFrame([new_member])[FEATURES])
dists = np.linalg.norm(v1.cluster_centers_ - z_new, axis=1)
print("\n=== New member ===")
print(f"Distances to centers: {np.round(dists, 2).tolist()}")
print(f"Segment {seg} ({name}) -> {'gets the $10 credit' if credit else 'no credit'}")

# ---------------------------------------------------------------- 5. Value, all 330 members
df["segment"] = v1.predict(scaler.transform(df[FEATURES]))
df["net if credited"] = member_net(df["monthly_extras_spend"].to_numpy())
df["credited (v1)"] = df["segment"].map(GETS_CREDIT)

table = df.groupby("segment").agg(
    members=("member_id", "size"),
    age=("age", "mean"),
    visits=("visits_per_month", "mean"),
    minutes=("avg_visit_minutes", "mean"),
    classes=("classes_per_month", "mean"),
    months=("months_member", "mean"),
    extras_spend=("monthly_extras_spend", "mean"),
    share_under_20=("monthly_extras_spend", lambda s: (s < SPEND_CUTOFF).mean()),
    net_if_all_credited=("net if credited", "sum"),
)
table["name"] = [segment_name(s) for s in table.index]
table["gets credit"] = [GETS_CREDIT[s] for s in table.index]
print("\n=== Segments, all 330 members ===")
print(table.round(1).to_string())

baseline_net = df["net if credited"].sum()
v1_net = df.loc[df["credited (v1)"], "net if credited"].sum()
print("\n=== Monthly net from the credit ===")
print(f"Baseline (credit to all {len(df)}): ${baseline_net:,.0f}   "
      f"cost ${CREDIT_COST * len(df):,}")
print(f"v1 (credit to {int(df['credited (v1)'].sum())} members in chosen segments): ${v1_net:,.0f}   "
      f"cost ${CREDIT_COST * int(df['credited (v1)'].sum()):,}")
print(f"Difference: ${v1_net - baseline_net:,.0f} a month, ${(v1_net - baseline_net) * 12:,.0f} a year")
