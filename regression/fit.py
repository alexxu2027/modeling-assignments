"""Used car prices: should the dealer buy this car at auction?

Run from this folder:  python fit.py
Fits on used_car_sales.csv (240 cars, 70/30 split). Tests the buy/pass rule
on auction_lots.csv (40 cars never used for fitting). Saves pred_vs_actual.png.
"""

import matplotlib

matplotlib.use("Agg")  # save charts to files instead of opening a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split

READY_COST = 1500  # inspection, reconditioning, detailing, lot time
ALL_FEATURES = ["Mileage_K", "Age_Years", "Horsepower", "Accident", "Num_Owners"]

pd.set_option("display.width", 200)
pd.set_option("display.float_format", "{:,.2f}".format)


def prepare(df):
    """Turn Accident yes/no into 1/0 so the regression can use it."""
    df = df.copy()
    df["Accident"] = (df["Accident"] == "yes").astype(int)
    return df


def fit_ols(train, features):
    return sm.OLS(train["Sale_Price"], sm.add_constant(train[features])).fit()


def predict(model, df, features):
    return model.predict(sm.add_constant(df[features], has_constant="add"))


def r2(actual, predicted):
    return 1 - ((actual - predicted) ** 2).sum() / ((actual - actual.mean()) ** 2).sum()


def rmse(actual, predicted):
    return float(np.sqrt(((actual - predicted) ** 2).mean()))


# ---------------------------------------------------------------- 1. Split
cars = prepare(pd.read_csv("used_car_sales.csv"))
train, test = train_test_split(cars, test_size=0.3, random_state=42)
print(f"Cars: {len(cars)}   train: {len(train)}   test: {len(test)}")

# ---------------------------------------------------------------- 2. Fit
full = fit_ols(train, ALL_FEATURES)
print("\n=== Model with all five features (training cars) ===")
print(full.summary().tables[1])
print(f"Adjusted R-squared: {full.rsquared_adj:.4f}")

# Drop the feature with the highest p-value if it is not significant (p > 0.05)
pvalues = full.pvalues.drop("const")
weakest = pvalues.idxmax()
if pvalues[weakest] > 0.05:
    FEATURES = [f for f in ALL_FEATURES if f != weakest]
    reduced = fit_ols(train, FEATURES)
    print(f"\n=== Model without {weakest} (p = {pvalues[weakest]:.3f}) ===")
    print(reduced.summary().tables[1])
    print(f"Adjusted R-squared: {reduced.rsquared_adj:.4f}  (all five: {full.rsquared_adj:.4f})")
    v1 = reduced if reduced.rsquared_adj >= full.rsquared_adj else full
    if v1 is full:
        FEATURES = ALL_FEATURES
else:
    FEATURES, v1 = ALL_FEATURES, full
print(f"\nv1 uses: {FEATURES}   adjusted R-squared {v1.rsquared_adj:.4f}")

# ---------------------------------------------------------------- 3. Test fit
average_price = train["Sale_Price"].mean()
actual = test["Sale_Price"]
baseline_pred = pd.Series(average_price, index=test.index)
v1_pred = predict(v1, test, FEATURES)
fit_table = pd.DataFrame({
    "model": ["baseline (training average)", "v1 regression"],
    "test R2": [r2(actual, baseline_pred), r2(actual, v1_pred)],
    "typical error $ (RMSE)": [rmse(actual, baseline_pred), rmse(actual, v1_pred)],
})
print(f"\n=== Test cars: baseline vs v1 (baseline predicts ${average_price:,.0f}) ===")
print(fit_table.to_string(index=False))

fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(actual, v1_pred, s=18, color="#1f4e79", label="v1 regression")
ax.axhline(average_price, color="#999999", ls="--", label=f"baseline: ${average_price:,.0f}")
lims = [actual.min() - 1000, actual.max() + 1000]
ax.plot(lims, lims, color="black", lw=0.8, label="perfect prediction")
ax.set_xlabel("Actual sale price ($)")
ax.set_ylabel("Predicted sale price ($)")
ax.set_title("Test cars: predicted vs actual")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig("pred_vs_actual.png", dpi=200)


# ---------------------------------------------------------------- 4. Decide
def should_buy(car, auction_price):
    """Dealer's rule: buy if auction price + $1,500 is below the predicted sale price."""
    row = prepare(pd.DataFrame([car]))
    predicted = float(predict(v1, row, FEATURES).iloc[0])
    return ("buy" if auction_price + READY_COST < predicted else "pass"), predicted


def baseline_should_buy(auction_price):
    return "buy" if auction_price + READY_COST < average_price else "pass"


lots = pd.read_csv("auction_lots.csv")
lots["v1_predicted"] = [should_buy(r[ALL_FEATURES].to_dict(), r["Auction_Price"])[1] for _, r in lots.iterrows()]
lots["v1_decision"] = [should_buy(r[ALL_FEATURES].to_dict(), r["Auction_Price"])[0] for _, r in lots.iterrows()]
lots["baseline_decision"] = lots["Auction_Price"].map(baseline_should_buy)
lots["profit_if_bought"] = lots["Later_Sale_Price"] - lots["Auction_Price"] - READY_COST

summary = []
for name, col in [("baseline", "baseline_decision"), ("v1", "v1_decision")]:
    bought = lots[lots[col] == "buy"]
    summary.append({
        "rule": name,
        "cars bought": len(bought),
        "money-losing cars bought": int((bought["profit_if_bought"] < 0).sum()),
        "total profit $": bought["profit_if_bought"].sum(),
    })
summary = pd.DataFrame(summary)
print("\n=== 40 auction lots: baseline vs v1 ===")
print(summary.to_string(index=False))
print("\nLot by lot:")
print(lots[["Lot", "Auction_Price", "v1_predicted", "Later_Sale_Price", "baseline_decision",
            "v1_decision", "profit_if_bought"]].to_string(index=False))
gain = summary.loc[1, "total profit $"] - summary.loc[0, "total profit $"]
print(f"\nv1 vs baseline at this auction: ${gain:,.0f} more profit")
