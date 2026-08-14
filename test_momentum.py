"""
Momentum test suite.

    python test_momentum.py

Loads your real gymap.py with Streamlit and Postgres stubbed out, then exercises
the pure calculation functions. No database, no network, no Streamlit server —
it runs in about a second, so there's no excuse not to run it before a commit.

HOW IT WORKS
    gymap.py connects to Supabase at import time and calls st.* everywhere, so it
    can't just be imported. This file installs fake `streamlit` and `psycopg2`
    modules into sys.modules first, then executes gymap.py up to the "UI" banner
    comment — everything above it is definitions.

    That means it tests the code you actually ship, not a copy that can drift.

WHAT IT COVERS
    The pure functions, which is where silent bugs hide: 1RM maths, per-hand load
    doubling, exercise-name normalisation, set summaries, streak counting, muscle
    volume attribution, calorie estimation, and target derivation.

    Both bugs you hit earlier — sets numbered from 2, and the "Use meal total"
    no-op — were in this category.

IF A TEST FAILS
    Read the FAIL line: it prints what it expected and what it got. The test is
    not automatically right; if you changed behaviour deliberately, update the
    test.
"""

import sys
import types
from datetime import date, timedelta

# ---------------------------------------------------------------
# STUBS — installed before gymap.py is loaded
# ---------------------------------------------------------------

_QUERY_RESULTS = {}      # substring of SQL -> (cols, rows)


class _FakeCursor:
    description = []

    def execute(self, q, params=()):
        self._q = q
        self._cols, self._rows = [], []
        for needle, (cols, rows) in _QUERY_RESULTS.items():
            if needle in q:
                self._cols, self._rows = cols, rows
                break
        self.description = [(c,) for c in self._cols]

    def fetchall(self):
        return self._rows

    def fetchone(self):
        return self._rows[0] if self._rows else None

    def close(self):
        pass


class _FakeConn:
    closed = 0
    autocommit = True

    def cursor(self):
        return _FakeCursor()


def _install_stubs():
    st = types.ModuleType("streamlit")

    class _SessionState(dict):
        def __getattr__(self, k):
            try:
                return self[k]
            except KeyError:
                raise AttributeError(k)

        def __setattr__(self, k, v):
            self[k] = v

    def _passthrough(*a, **kw):
        # Supports both @st.cache_data and @st.cache_data(ttl=60)
        if len(a) == 1 and callable(a[0]) and not kw:
            fn = a[0]
            fn.clear = lambda: None
            return fn

        def deco(fn):
            fn.clear = lambda: None
            return fn
        return deco

    st.cache_data = _passthrough
    st.cache_resource = _passthrough
    st.secrets = {"connections": {"postgres": {"url": "postgresql://stub"}}}
    st.session_state = _SessionState()
    for name in ("write", "markdown", "caption", "title", "header", "subheader",
                 "info", "warning", "error", "success", "metric", "image",
                 "dataframe", "progress", "line_chart", "plotly_chart",
                 "set_page_config", "rerun", "download_button", "divider"):
        setattr(st, name, lambda *a, **kw: None)
    st.columns = lambda spec, **kw: [types.SimpleNamespace(
        metric=lambda *a, **kw: None, caption=lambda *a, **kw: None,
        markdown=lambda *a, **kw: None, __enter__=lambda s: s,
        __exit__=lambda *a: False) for _ in (spec if isinstance(spec, (list, tuple))
                                             else range(spec))]
    sys.modules["streamlit"] = st

    comp = types.ModuleType("streamlit.components.v1")
    comp.html = lambda *a, **kw: None
    sys.modules["streamlit.components.v1"] = comp
    sys.modules["streamlit.components"] = types.ModuleType("streamlit.components")

    pg = types.ModuleType("psycopg2")
    pg.connect = lambda *a, **kw: _FakeConn()
    pg.Binary = lambda b: b
    pg.InterfaceError = type("InterfaceError", (Exception,), {})
    pg.OperationalError = type("OperationalError", (Exception,), {})
    sys.modules["psycopg2"] = pg

    px = types.ModuleType("plotly.express")
    px.line = lambda *a, **kw: types.SimpleNamespace(update_layout=lambda **kw: None)
    sys.modules["plotly"] = types.ModuleType("plotly")
    sys.modules["plotly.express"] = px

    req = types.ModuleType("requests")
    req.post = lambda *a, **kw: None
    sys.modules["requests"] = req

    return st


ST = _install_stubs()

UI_BANNER = ("# ---------------------------------------------------------------\n"
             "# UI\n"
             "# ---------------------------------------------------------------")

with open("gymap.py", encoding="utf-8") as fh:
    source = fh.read()

if UI_BANNER not in source:
    print("ERROR: couldn't find the '# UI' banner comment in gymap.py.")
    print("The tests load everything above it. If you renamed that comment,")
    print("update UI_BANNER in this file to match.")
    sys.exit(1)

G = {"__name__": "gymap_under_test"}
exec(compile(source[:source.index(UI_BANNER)], "gymap.py", "exec"), G)


# ---------------------------------------------------------------
# HARNESS
# ---------------------------------------------------------------
FAILURES = []
COUNT = 0


def check(label, got, want):
    global COUNT
    COUNT += 1
    ok = got == want
    print(("  PASS  " if ok else "  FAIL  ") + label)
    if not ok:
        print("          expected: %r" % (want,))
        print("          got:      %r" % (got,))
        FAILURES.append(label)


def check_close(label, got, want, tol=0.05):
    global COUNT
    COUNT += 1
    ok = abs(got - want) <= tol
    print(("  PASS  " if ok else "  FAIL  ") + label)
    if not ok:
        print("          expected: %r (+/- %s)" % (want, tol))
        print("          got:      %r" % (got,))
        FAILURES.append(label)


def check_true(label, cond, detail=""):
    global COUNT
    COUNT += 1
    print(("  PASS  " if cond else "  FAIL  ") + label)
    if not cond:
        if detail:
            print("          " + str(detail))
        FAILURES.append(label)


def section(name):
    print("\n--- %s ---" % name)


# ---------------------------------------------------------------
# TESTS
# ---------------------------------------------------------------
section("estimated_1rm (Epley)")
e1rm = G["estimated_1rm"]
check_close("100kg x 1 is 100kg", e1rm(100, 1), 103.33)
check_close("80kg x 8", e1rm(80, 8), 101.33)
check_close("60kg x 10", e1rm(60, 10), 80.0)
check("zero weight returns 0", e1rm(0, 8), 0.0)
check("zero reps returns 0", e1rm(80, 0), 0.0)
check("None weight returns 0", e1rm(None, 8), 0.0)
check("negative reps returns 0", e1rm(80, -3), 0.0)
check_true("more reps at same weight scores higher", e1rm(80, 10) > e1rm(80, 8))

section("effective_load (per-hand doubling)")
load = G["effective_load"]
check("22kg dumbbells = 44kg total", load(22, True), 44.0)
check("100kg barbell stays 100kg", load(100, False), 100.0)
check("None weight is 0", load(None, True), 0.0)
check("zero stays zero", load(0, True), 0.0)

section("base_exercise_name (strips set/rep notation)")
base = G["base_exercise_name"]
check("'Bench Press 4x6-8'", base("Bench Press 4x6-8"), "Bench Press")
check("'Incline DB Press 3x10'", base("Incline DB Press 3x10"), "Incline DB Press")
check("no notation is untouched", base("Face Pulls"), "Face Pulls")
check("spacing variants", base("Squat 5 x 5"), "Squat")
check("does not eat legitimate names", base("T-Bar Row"), "T-Bar Row")

section("_normalise_exercise_name")
norm = G["_normalise_exercise_name"]
check("case insensitive", norm("PULL-UPS"), norm("Pull Ups"))
check("punctuation ignored", norm("Chin-Up"), "chin up")
check("set notation stripped", norm("Bench Press 4x6-8"), "bench press")
check_true("different lifts stay different", norm("Squat") != norm("Deadlift"))

section("format_set_summary")
fmt = G["format_set_summary"]
barbell = [{"weight_kg": 80, "reps": 8, "duration_min": 0, "distance_km": 0},
           {"weight_kg": 80, "reps": 7, "duration_min": 0, "distance_km": 0}]
check("barbell sets", fmt(barbell), "80kg × 8, 80kg × 7")
check("per-hand notation", fmt([barbell[0]], per_side=True), "80kg ×2 × 8")
check("bodyweight sets",
      fmt([{"weight_kg": 0, "reps": 10, "duration_min": 0, "distance_km": 0}]),
      "BW × 10")
check("cardio entry",
      fmt([{"weight_kg": 0, "reps": 0, "duration_min": 25, "distance_km": 3.2}]),
      "25 min · 3.2 km")
check("empty sets give empty string", fmt([]), "")
check("zero-rep set is skipped",
      fmt([{"weight_kg": 80, "reps": 0, "duration_min": 0, "distance_km": 0}]), "")

section("compute_targets_from_profile")
targets = G["compute_targets_from_profile"]
prof = {"height_cm": 180, "weight_kg": 88, "age": 27, "sex": "Male",
        "activity_level": "Moderate (3-5 days/week)", "goal": "Lose weight"}
r = targets(prof)
check_true("returns a dict", isinstance(r, dict))
check_true("cut sits below maintenance", r["calories_min"] < r["calories_max"] < 3500,
           r)
check_true("protein scales with bodyweight", 140 < r["protein_min"] < 200, r["protein_min"])
check_true("min below max on every key",
           all(r[k + "_min"] <= r[k + "_max"] for k in
               ("calories", "protein", "water", "steps")), r)
check("missing height returns None", targets({**prof, "height_cm": None}), None)
check("missing age returns None", targets({**prof, "age": None}), None)
gain = targets({**prof, "goal": "Gain muscle"})
check_true("bulk targets exceed cut targets",
           gain["calories_min"] > r["calories_min"],
           (gain["calories_min"], r["calories_min"]))

section("muscle_regions_for (volume attribution)")
regions = G["muscle_regions_for"]
bench = regions("Bench Press")
check_true("bench hits chest", "Chest" in bench, bench)
check_true("bench hits triceps", "Triceps" in bench, bench)
check_true("bench hits shoulders", "Shoulders" in bench, bench)
check_true("bench does NOT hit legs", "Quads" not in bench, bench)
check("set notation still resolves", regions("Bench Press 4x6-8"), bench)
check_true("deadlift hits posterior chain",
           {"Hamstrings", "Glutes", "Back"} <= set(regions("Deadlift")),
           regions("Deadlift"))
check("unknown exercise returns empty", regions("Underwater Basket Weaving"), [])

section("estimate_exercise_calories")
_QUERY_RESULTS.clear()
G["get_sets"] = lambda d, e: [
    {"set_number": 1, "reps": 8, "weight_kg": 80.0, "duration_min": 0, "distance_km": 0},
    {"set_number": 2, "reps": 8, "weight_kg": 80.0, "duration_min": 0, "distance_km": 0}]
G["get_exercise_pref"] = lambda e: {"log_type": "weight_reps", "per_side": False}
kcal, mins = G["estimate_exercise_calories"]("2026-08-13", "Bench Press", 88.0)
check_true("lifting burn is plausible, not tiny", 5 < kcal < 60, kcal)
check_true("duration derived from reps and rest", 2 < mins < 8, mins)

G["get_sets"] = lambda d, e: [
    {"set_number": 1, "reps": 0, "weight_kg": 0, "duration_min": 30, "distance_km": 3.0}]
kcal_walk, mins_walk = G["estimate_exercise_calories"]("2026-08-13", "30-45 min walk", 88.0)
check("walk uses logged minutes", mins_walk, 30.0)
check_true("30 min walk burns more than 2 bench sets", kcal_walk > kcal,
           (kcal_walk, kcal))
kcal_incline, _ = G["estimate_exercise_calories"](
    "2026-08-13", "20 min incline walk", 88.0)
G["get_sets"] = lambda d, e: []
check("no sets means no calories", G["estimate_exercise_calories"]("x", "y", 88.0),
      (0.0, 0.0))

section("weight prediction guards")
pred = G["compute_weight_prediction"]


def _weights(entries):
    import pandas as pd
    return pd.DataFrame([
        {"log_date": (date.today() - timedelta(days=d)).isoformat(), "weight_kg": w}
        for d, w in entries])


G["get_range_df"] = lambda a, b: _weights([(14, 92.0), (0, 88.0)])
check("2 entries is not enough", pred(), None)

G["get_range_df"] = lambda a, b: _weights(
    [(6, 90.0), (4, 89.5), (2, 89.0), (0, 88.5)])
check("span under 14 days is not enough", pred(), None)

G["get_range_df"] = lambda a, b: _weights(
    [(56 - i * 7, 92.0 - i * 0.5) for i in range(9)])
r = pred()
check_true("steady cut produces a projection", r is not None)
check_true("slope is sane", -0.7 < r["slope_per_week"] < -0.3, r["slope_per_week"])
check_true("not clamped on a normal cut", not r["clamped"])
check_true("30-day projection is plausible", 84 < r["pred_30"] < 88, r["pred_30"])

G["get_range_df"] = lambda a, b: _weights(
    [(21, 100.0), (14, 96.0), (7, 92.0), (0, 88.0)])
r = pred()
check_true("crash-diet slope triggers the clamp", r["clamped"])
check_true("clamped projection stays in the real world", 70 < r["pred_90"] < 90,
           r["pred_90"])

G["get_range_df"] = lambda a, b: _weights(
    [(365, 200.0), (240, 150.0), (120, 100.0), (0, 60.0)])
check_true("never projects below the 30kg floor", pred()["pred_90"] >= 30.0)

section("compute_streak_stats")
import pandas as _pd


def _active(days_ago_list):
    return _pd.DataFrame([{"log_date": (date.today() - timedelta(days=d)).isoformat()}
                          for d in days_ago_list])


G["get_active_dates_df"] = lambda: _pd.DataFrame()
check("no data gives zeros", G["compute_streak_stats"](), (0, 0, 0))

G["get_active_dates_df"] = lambda: _active([0, 1, 2])
streak, longest, total = G["compute_streak_stats"]()
check("3 consecutive days ending today", (streak, longest, total), (3, 3, 3))

G["get_active_dates_df"] = lambda: _active([1, 2, 3])
streak, longest, total = G["compute_streak_stats"]()
check_true("streak survives not having logged yet today", streak == 3,
           (streak, longest, total))

G["get_active_dates_df"] = lambda: _active([0, 1, 5, 6, 7, 8])
streak, longest, total = G["compute_streak_stats"]()
check("current streak counts only the recent run", streak, 2)
check("longest streak finds the earlier run", longest, 4)
check("total counts every logged day", total, 6)

section("get_earned_badges")
badges = G["get_earned_badges"]
check("no badges at zero", badges(0, 0), [])
check_true("3-day streak earns one", any("3-Day" in b for b in badges(3, 3)),
           badges(3, 3))
check_true("30-day streak earns all lower tiers", len(badges(30, 30)) >= 4,
           badges(30, 30))
check_true("days-logged badges are separate",
           any("10 Days Logged" in b for b in badges(0, 10)), badges(0, 10))

section("local_today (timezone)")
check_true("local_today exists", "local_today" in G)
check_true("returns a date", isinstance(G["local_today"](), date))
check_true("within a day of the server clock",
           abs((G["local_today"]() - date.today()).days) <= 1)
if G.get("_APP_TZ") is None:
    print("  NOTE  zoneinfo unavailable here — falling back to the server clock.")
    print("        Run `pip install tzdata` on Windows to get the real fix.")

# ---------------------------------------------------------------
print("\n" + "=" * 58)
if FAILURES:
    print("%d of %d checks FAILED:" % (len(FAILURES), COUNT))
    for f in FAILURES:
        print("   - " + f)
    sys.exit(1)
print("All %d checks passed." % COUNT)
