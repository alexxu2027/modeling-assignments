# Modeling assignments

Three individual model assignments. Each folder holds its data, the brief, and one
plain Python script, `fit.py`, that reproduces every number on the slides.

| Folder | Model | Question | Data |
| --- | --- | --- | --- |
| `regression/` | Linear regression | Should the used-car dealer buy this car at auction? | `used_car_sales.csv` (fit), `auction_lots.csv` (decision test) |
| `decision-tree/` | Classification tree | Should the clinic overbook this slot? | `clinic_appointments.csv` |
| `clustering/` | k-means | Which gym members get the $10 credit? | `gym_members.csv` |

## Run

```
python -m venv venv
venv\Scripts\activate        # Mac: source venv/bin/activate
pip install -r requirements.txt
cd regression && python fit.py
```

Every split and model uses `random_state=42` and `test_size=0.3`, so the numbers are the same on every run.
Charts are saved next to each script (`pred_vs_actual.png`, `tree_v1.png`, `elbow.png`).
