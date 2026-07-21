import streamlit as st
import psycopg2
import pandas as pd
from datetime import date, timedelta

# ---------------------------------------------------------------
# CONFIG - your plan, baked in
# ---------------------------------------------------------------
TARGETS = {
    "calories_min": 2200, "calories_max": 2500,
    "protein_min": 180, "protein_max": 220,
    "water_min": 3.0, "water_max": 4.0,   # litres
    "steps_min": 10000, "steps_max": 15000,
}

MEALS = ["Breakfast", "Lunch", "Snack", "Dinner"]

GYM_SPLIT = {
    "Monday": {"label": "Push", "exercises": [
        "Bench Press 4x6-8", "Incline DB Press 3x10", "Shoulder Press 3x10",
        "Lateral Raise 4x15", "Cable Fly 3x15", "Triceps 4x12", "15 min incline walk"
    ]},
    "Tuesday": {"label": "Active Recovery", "exercises": [
        "45 min walk or football"
    ]},
    "Wednesday": {"label": "Pull", "exercises": [
        "Deadlift 3x5", "Lat Pulldown 4x10", "Rows", "Face Pulls",
        "Curls", "20 min incline walk"
    ]},
    "Thursday": {"label": "Legs", "exercises": [
        "Squat", "RDL", "Leg Press", "Lunges", "Leg Curl", "Calves", "20 min incline walk"
    ]},
    "Friday": {"label": "Upper", "exercises": [
        "Incline Bench", "Pull Ups/Pulldown", "Chest Press", "Row",
        "Shoulders", "Arms", "Cardio"
    ]},
    "Saturday": {"label": "Conditioning", "exercises": [
        "30-45 min walk", "Abs", "Stretch"
    ]},
    "Sunday": {"label": "Rest", "exercises": [
        "Rest and 10k steps"
    ]},
}

WEEKDAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# ---------------------------------------------------------------
# DATABASE (Postgres via Supabase)
# ---------------------------------------------------------------
@st.cache_resource
def get_conn():
    conn = psycopg2.connect(st.secrets["connections"]["postgres"]["url"])
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS daily_log (
            log_date TEXT PRIMARY KEY,
            protein_g REAL DEFAULT 0,
            water_l REAL DEFAULT 0,
            steps INTEGER DEFAULT 0,
            weight_kg REAL,
            notes TEXT
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS meal_checks (
            log_date TEXT,
            meal TEXT,
            done INTEGER DEFAULT 0,
            PRIMARY KEY (log_date, meal)
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS meal_details (
            log_date TEXT,
            meal TEXT,
            note TEXT DEFAULT '',
            calories REAL DEFAULT 0,
            protein_g REAL DEFAULT 0,
            PRIMARY KEY (log_date, meal)
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS exercise_checks (
            log_date TEXT,
            exercise TEXT,
            done INTEGER DEFAULT 0,
            PRIMARY KEY (log_date, exercise)
        )
    """)
    cur.close()
    return conn

conn = get_conn()

def run(query, params=()):
    """Execute a write query."""
    cur = conn.cursor()
    cur.execute(query, params)
    cur.close()

def fetch(query, params=()):
    """Execute a read query and return raw rows + column names."""
    cur = conn.cursor()
    cur.execute(query, params)
    cols = [d[0] for d in cur.description]
    rows = cur.fetchall()
    cur.close()
    return cols, rows

def get_daily_row(log_date):
    cols, rows = fetch("SELECT * FROM daily_log WHERE log_date = %s", (log_date,))
    if not rows:
        run("INSERT INTO daily_log (log_date) VALUES (%s)", (log_date,))
        return {"log_date": log_date, "protein_g": 0, "water_l": 0, "steps": 0, "weight_kg": None, "notes": ""}
    return dict(zip(cols, rows[0]))

def save_daily_row(log_date, protein_g, water_l, steps, weight_kg, notes):
    run("""
        UPDATE daily_log SET protein_g=%s, water_l=%s, steps=%s, weight_kg=%s, notes=%s
        WHERE log_date=%s
    """, (protein_g, water_l, steps, weight_kg, notes, log_date))

def get_meal_checks(log_date):
    cols, rows = fetch("SELECT meal, done FROM meal_checks WHERE log_date=%s", (log_date,))
    existing = dict(rows)
    return {m: bool(existing.get(m, 0)) for m in MEALS}

def set_meal_check(log_date, meal, done):
    run("""
        INSERT INTO meal_checks (log_date, meal, done) VALUES (%s, %s, %s)
        ON CONFLICT (log_date, meal) DO UPDATE SET done=excluded.done
    """, (log_date, meal, int(done)))

def get_meal_details(log_date):
    cols, rows = fetch(
        "SELECT meal, note, calories, protein_g FROM meal_details WHERE log_date=%s", (log_date,)
    )
    existing = {r[0]: {"note": r[1] or "", "calories": r[2] or 0, "protein_g": r[3] or 0} for r in rows}
    return {m: existing.get(m, {"note": "", "calories": 0, "protein_g": 0}) for m in MEALS}

def set_meal_detail(log_date, meal, note, calories, protein_g):
    run("""
        INSERT INTO meal_details (log_date, meal, note, calories, protein_g) VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (log_date, meal) DO UPDATE SET
            note=excluded.note, calories=excluded.calories, protein_g=excluded.protein_g
    """, (log_date, meal, note, calories, protein_g))

def get_exercise_checks(log_date, exercises):
    cols, rows = fetch("SELECT exercise, done FROM exercise_checks WHERE log_date=%s", (log_date,))
    existing = dict(rows)
    return {e: bool(existing.get(e, 0)) for e in exercises}

def set_exercise_check(log_date, exercise, done):
    run("""
        INSERT INTO exercise_checks (log_date, exercise, done) VALUES (%s, %s, %s)
        ON CONFLICT (log_date, exercise) DO UPDATE SET done=excluded.done
    """, (log_date, exercise, int(done)))

def get_range_df(start, end):
    q = "SELECT * FROM daily_log WHERE log_date BETWEEN %s AND %s ORDER BY log_date"
    return pd.read_sql_query(q, conn, params=(start.isoformat(), end.isoformat()))

def get_meal_completion_for_range(start, end):
    q = """
        SELECT log_date, SUM(done) as done_count FROM meal_checks
        WHERE log_date BETWEEN %s AND %s GROUP BY log_date
    """
    return pd.read_sql_query(q, conn, params=(start.isoformat(), end.isoformat()))

def get_exercise_completion_for_range(start, end):
    q = """
        SELECT log_date, COUNT(*) as total, SUM(done) as done_count FROM exercise_checks
        WHERE log_date BETWEEN %s AND %s GROUP BY log_date
    """
    return pd.read_sql_query(q, conn, params=(start.isoformat(), end.isoformat()))

def get_meal_macro_totals_for_range(start, end):
    q = """
        SELECT log_date, SUM(calories) as calories_total, SUM(protein_g) as meal_protein_total
        FROM meal_details WHERE log_date BETWEEN %s AND %s GROUP BY log_date
    """
    return pd.read_sql_query(q, conn, params=(start.isoformat(), end.isoformat()))

# ---------------------------------------------------------------
# UI
# ---------------------------------------------------------------
st.set_page_config(page_title="Weight Loss Tracker", page_icon="💪", layout="centered")
st.title("💪 Weight Loss Tracker")

page = st.sidebar.radio("View", ["Today", "Weight Log", "Weekly Dashboard"])

if "selected_date" not in st.session_state:
    st.session_state.selected_date = date.today()

if page == "Today":
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if st.button("⬅ Prev day"):
            st.session_state.selected_date -= timedelta(days=1)
    with col3:
        if st.button("Next day ➡"):
            st.session_state.selected_date += timedelta(days=1)
    with col2:
        picked = st.date_input("Date", value=st.session_state.selected_date, label_visibility="collapsed")
        st.session_state.selected_date = picked

    log_date = st.session_state.selected_date
    log_date_str = log_date.isoformat()
    weekday = WEEKDAY_NAMES[log_date.weekday()]
    day_plan = GYM_SPLIT[weekday]

    st.subheader(f"{weekday}, {log_date.strftime('%d %b %Y')}")
    st.caption(f"Gym: **{day_plan['label']}**")

    row = get_daily_row(log_date_str)

    # ---- Meals ----
    st.markdown("### 🍽️ Meals")
    meal_state = get_meal_checks(log_date_str)
    meal_details = get_meal_details(log_date_str)

    day_calories_total = 0
    day_meal_protein_total = 0

    for meal in MEALS:
        details = meal_details[meal]
        header = meal
        if details["note"]:
            header += f" — {details['note'][:30]}{'…' if len(details['note']) > 30 else ''}"

        with st.expander(header, expanded=False):
            checked = st.checkbox("Done", value=meal_state[meal], key=f"meal_{log_date_str}_{meal}")
            if checked != meal_state[meal]:
                set_meal_check(log_date_str, meal, checked)

            note = st.text_input(
                "What did you have?", value=details["note"], key=f"meal_note_{log_date_str}_{meal}"
            )

            mc1, mc2 = st.columns(2)
            with mc1:
                cals = st.number_input(
                    "Calories", min_value=0.0, value=float(details["calories"]),
                    step=10.0, key=f"meal_cal_{log_date_str}_{meal}"
                )
            with mc2:
                prot = st.number_input(
                    "Protein (g)", min_value=0.0, value=float(details["protein_g"]),
                    step=1.0, key=f"meal_prot_{log_date_str}_{meal}"
                )

            if (note, cals, prot) != (details["note"], details["calories"], details["protein_g"]):
                set_meal_detail(log_date_str, meal, note, cals, prot)

            day_calories_total += cals
            day_meal_protein_total += prot

    st.caption(
        f"Meals logged today: {day_calories_total:.0f} kcal · {day_meal_protein_total:.0f}g protein "
        f"(target {TARGETS['calories_min']}-{TARGETS['calories_max']} kcal, "
        f"{TARGETS['protein_min']}-{TARGETS['protein_max']}g protein)"
    )

    # ---- Workout ----
    st.markdown(f"### 🏋️ Workout — {day_plan['label']}")
    ex_state = get_exercise_checks(log_date_str, day_plan["exercises"])
    for ex in day_plan["exercises"]:
        checked = st.checkbox(ex, value=ex_state[ex], key=f"ex_{log_date_str}_{ex}")
        if checked != ex_state[ex]:
            set_exercise_check(log_date_str, ex, checked)

    # ---- Numbers: protein, water, steps ----
    st.markdown("### 📊 Daily Numbers")
    n1, n2, n3 = st.columns(3)
    with n1:
        protein = st.number_input(
            f"Protein (g) — target {TARGETS['protein_min']}-{TARGETS['protein_max']}g",
            min_value=0.0, value=float(row["protein_g"] or 0), step=5.0, key=f"protein_{log_date_str}"
        )
        if st.button("Use meal total", key=f"use_meal_protein_{log_date_str}"):
            protein = day_meal_protein_total
    with n2:
        water = st.number_input(
            f"Water (L) — target {TARGETS['water_min']}-{TARGETS['water_max']}L",
            min_value=0.0, value=float(row["water_l"] or 0), step=0.25, key=f"water_{log_date_str}"
        )
    with n3:
        steps = st.number_input(
            f"Steps — target {TARGETS['steps_min']:,}-{TARGETS['steps_max']:,}",
            min_value=0, value=int(row["steps"] or 0), step=500, key=f"steps_{log_date_str}"
        )

    st.progress(min(protein / TARGETS["protein_min"], 1.0) if TARGETS["protein_min"] else 0,
                text=f"Protein: {protein:.0f}g / {TARGETS['protein_min']}g min")
    st.progress(min(water / TARGETS["water_min"], 1.0) if TARGETS["water_min"] else 0,
                text=f"Water: {water:.2f}L / {TARGETS['water_min']}L min")
    st.progress(min(steps / TARGETS["steps_min"], 1.0) if TARGETS["steps_min"] else 0,
                text=f"Steps: {steps:,} / {TARGETS['steps_min']:,} min")

    # ---- Weight ----
    st.markdown("### ⚖️ Weight")
    weight = st.number_input(
        "Weight (kg) — optional, log on weigh-in days",
        min_value=0.0, value=float(row["weight_kg"]) if row["weight_kg"] else 0.0,
        step=0.1, key=f"weight_{log_date_str}"
    )

    # ---- Notes ----
    notes = st.text_area("Notes (optional)", value=row["notes"] or "", key=f"notes_{log_date_str}")

    if st.button("💾 Save today's log", type="primary"):
        save_daily_row(
            log_date_str, protein, water, steps,
            weight if weight > 0 else None, notes
        )
        st.success("Saved!")

elif page == "Weight Log":
    st.subheader("⚖️ Weight Trend")
    end = date.today()
    start = end - timedelta(days=90)
    df = get_range_df(start, end)
    df = df.dropna(subset=["weight_kg"])

    if df.empty:
        st.info("No weight entries yet. Log your weight on the Today page.")
    else:
        df["log_date"] = pd.to_datetime(df["log_date"])
        df = df.sort_values("log_date")
        df["trend"] = df["weight_kg"].rolling(window=7, min_periods=1).mean()
        chart_df = df.set_index("log_date")[["weight_kg", "trend"]]
        chart_df.columns = ["Weight (kg)", "7-day trend"]
        st.line_chart(chart_df)

        c1, c2, c3 = st.columns(3)
        c1.metric("Latest weight", f"{df['weight_kg'].iloc[-1]:.1f} kg")
        if len(df) > 1:
            change = df["weight_kg"].iloc[-1] - df["weight_kg"].iloc[0]
            c2.metric("Change (period)", f"{change:+.1f} kg")
        c3.metric("Entries logged", len(df))

        st.dataframe(df[["log_date", "weight_kg"]].rename(
            columns={"log_date": "Date", "weight_kg": "Weight (kg)"}
        ).sort_values("Date", ascending=False), hide_index=True, use_container_width=True)

elif page == "Weekly Dashboard":
    st.subheader("📅 Weekly Adherence")
    today = date.today()
    start_of_week = today - timedelta(days=today.weekday())
    end_of_week = start_of_week + timedelta(days=6)
    st.caption(f"Week of {start_of_week.strftime('%d %b')} – {end_of_week.strftime('%d %b %Y')}")

    daily_df = get_range_df(start_of_week, end_of_week)
    meal_df = get_meal_completion_for_range(start_of_week, end_of_week)
    ex_df = get_exercise_completion_for_range(start_of_week, end_of_week)
    macro_df = get_meal_macro_totals_for_range(start_of_week, end_of_week)

    all_days = [(start_of_week + timedelta(days=i)).isoformat() for i in range(7)]
    summary = pd.DataFrame({"log_date": all_days})
    summary = summary.merge(daily_df, on="log_date", how="left")
    summary = summary.merge(meal_df.rename(columns={"done_count": "meals_done"}), on="log_date", how="left")
    summary = summary.merge(ex_df.rename(columns={"done_count": "ex_done", "total": "ex_total"}), on="log_date", how="left")
    summary = summary.merge(macro_df, on="log_date", how="left")
    summary = summary.fillna(0)
    summary["weekday"] = pd.to_datetime(summary["log_date"]).dt.strftime("%a")

    summary["meal_pct"] = (summary["meals_done"] / len(MEALS) * 100).clip(0, 100)
    summary["protein_hit"] = summary["protein_g"] >= TARGETS["protein_min"]
    summary["water_hit"] = summary["water_l"] >= TARGETS["water_min"]
    summary["steps_hit"] = summary["steps"] >= TARGETS["steps_min"]
    summary["calories_hit"] = (summary["calories_total"] >= TARGETS["calories_min"]) & \
        (summary["calories_total"] <= TARGETS["calories_max"])
    summary["ex_pct"] = summary.apply(
        lambda r: (r["ex_done"] / r["ex_total"] * 100) if r["ex_total"] > 0 else 0, axis=1
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Meals hit (avg)", f"{summary['meal_pct'].mean():.0f}%")
    m2.metric("Workouts completed", f"{summary['ex_pct'].mean():.0f}%")
    m3.metric("Protein target days", f"{int(summary['protein_hit'].sum())}/7")
    m4.metric("Steps target days", f"{int(summary['steps_hit'].sum())}/7")

    st.markdown("#### Meal completion by day")
    st.bar_chart(summary.set_index("weekday")["meal_pct"])

    st.markdown("#### Workout completion by day")
    st.bar_chart(summary.set_index("weekday")["ex_pct"])

    st.markdown("#### Protein & Water vs target")
    pw_df = summary.set_index("weekday")[["protein_g", "water_l"]].copy()
    pw_df.columns = ["Protein (g)", "Water (L)"]
    st.line_chart(pw_df)

    st.markdown("#### Calories from logged meals")
    st.bar_chart(summary.set_index("weekday")["calories_total"])

    st.markdown("#### Raw log")
    display_cols = ["log_date", "protein_g", "water_l", "steps", "weight_kg", "meals_done",
                     "ex_done", "ex_total", "calories_total", "meal_protein_total"]
    st.dataframe(summary[display_cols].rename(columns={
        "log_date": "Date", "protein_g": "Protein (g)", "water_l": "Water (L)",
        "steps": "Steps", "weight_kg": "Weight (kg)", "meals_done": "Meals done",
        "ex_done": "Exercises done", "ex_total": "Exercises total",
        "calories_total": "Calories (from meals)", "meal_protein_total": "Protein (from meals)"
    }), hide_index=True, use_container_width=True)