from pathlib import Path
import fastf1

root = Path(__file__).resolve().parent
(root / "cache").mkdir(exist_ok=True)
(root / "data").mkdir(exist_ok=True)

fastf1.Cache.enable_cache(str(root / "cache"))

session = fastf1.get_session(2024, "Bahrain", "R")
session.load()

laps = session.laps[
    session.laps["Driver"].isin(["LEC", "SAI"])
]

output = root / "data" / "bahrain_2024_ferrari_laps.csv"
laps.to_csv(output, index=False)

print(f"Saved {len(laps)} lap records to {output}")