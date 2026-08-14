import re
import requests
import streamlit as st
import streamlit.components.v1 as components
import psycopg2
import pandas as pd
import numpy as np
import plotly.express as px
import base64
from datetime import date, datetime, timedelta

# ---------------------------------------------------------------
# LOCAL TIME
# ---------------------------------------------------------------
# date.today() reads the SERVER clock. Streamlit Community Cloud runs UTC, so
# between midnight and 1am BST the app would file a session under yesterday —
# and silently break the streak. Everything below resolves "today" in the zone
# below instead.
#
# Windows ships no timezone database, so ZoneInfo raises there unless the
# `tzdata` package is installed. The fallback keeps the app running rather than
# crashing; it just reverts to the old server-clock behaviour.
APP_TIMEZONE = "Europe/London"

try:
    from zoneinfo import ZoneInfo
    _APP_TZ = ZoneInfo(APP_TIMEZONE)
except Exception:
    _APP_TZ = None


def local_today():
    """Today's date in APP_TIMEZONE, falling back to the server clock."""
    if _APP_TZ is None:
        return date.today()
    return datetime.now(_APP_TZ).date()


# NOTE: Gym Bro talks to the Gemini REST endpoint directly with `requests` — no
# Google SDK. It uses function calling: the tools declared in GYM_BRO_TOOLS let
# the model request the user's own logged data, which this app then queries and
# hands back. Every tool is read-only; there is no write path and no raw SQL.

# ---------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------
DEFAULT_TARGETS = {
    "calories_min": 2200, "calories_max": 2500,
    "protein_min": 180, "protein_max": 220,
    "water_min": 3.0, "water_max": 4.0,
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
LANGUAGES = ["English", "Afrikaans", "Spanish", "French", "Italian", "Dutch", "Urdu", "Arabic", "German"]

GOALS = ["Lose weight", "Gain muscle", "Maintain", "Body recomposition"]
ACTIVITY_LEVELS = [
    "Sedentary (little exercise)", "Light (1-3 days/week)", "Moderate (3-5 days/week)",
    "Active (6-7 days/week)", "Very active (physical job + training)"
]
ACTIVITY_MULTIPLIERS = {
    "Sedentary (little exercise)": 1.2, "Light (1-3 days/week)": 1.375,
    "Moderate (3-5 days/week)": 1.55, "Active (6-7 days/week)": 1.725,
    "Very active (physical job + training)": 1.9,
}
SEX_OPTIONS = ["Male", "Female", "Prefer not to say"]

# ---------------------------------------------------------------
# EXERCISE INFO (demo images / muscles / form cues)
# ---------------------------------------------------------------
# Images come from https://github.com/yuhonas/free-exercise-db — public domain
# (Unlicense), hotlinked straight from raw.githubusercontent.com, no API key.
# I verified these URLs return HTTP 200.
#
# Honest caveat: these are TWO STATIC PHOTOS per exercise (start position and
# end position) shown side by side, not a looping animated GIF. I looked for a
# free, reliably-hotlinkable source of real demo GIFs covering this exercise
# list and couldn't find one I'd trust not to break or have licensing issues.
# The two-frame version still shows the movement clearly.
#
# To upgrade to real GIFs later: just replace the URLs in any "images" list
# below with your GIF URL(s). The renderer handles any number of URLs and
# st.image() plays animated GIFs natively.

# Canonical muscle regions used by the weekly muscle-volume heatmap. The
# "muscles" strings in EXERCISE_INFO are human-readable; this maps them onto a
# fixed set of regions so volume can be summed per region.
MUSCLE_REGIONS = ["Chest", "Back", "Shoulders", "Biceps", "Triceps",
                  "Core", "Quads", "Hamstrings", "Glutes", "Calves"]

_MUSCLE_ALIASES = {
    "chest": "Chest", "upper chest": "Chest",
    "lats": "Back", "rhomboids": "Back", "upper back": "Back",
    "lower back": "Back", "traps": "Back",
    "shoulders": "Shoulders", "front shoulders": "Shoulders",
    "side shoulders": "Shoulders", "rear shoulders": "Shoulders",
    "rotator cuff": "Shoulders",
    "biceps": "Biceps", "forearms": "Biceps",
    "triceps": "Triceps",
    "core": "Core", "core / abdominals": "Core", "abdominals": "Core",
    "quads": "Quads",
    "hamstrings": "Hamstrings",
    "glutes": "Glutes",
    "calves": "Calves",
}


def muscle_regions_for(exercise_name):
    """Map an exercise to a list of canonical muscle regions (may be empty)."""
    info = EXERCISE_INFO.get(base_exercise_name(exercise_name))
    if not info:
        return []
    out = []
    for m in info.get("muscles", []):
        region = _MUSCLE_ALIASES.get(m.strip().lower())
        if region and region not in out:
            out.append(region)
    return out


# ---------------------------------------------------------------
# EXERCISE LIBRARY (~260 movements, loaded from exercise_library.json)
# ---------------------------------------------------------------
# Keeps gymap.py readable while still covering whatever you swap in. Data is
# from https://github.com/yuhonas/free-exercise-db (public domain / Unlicense).
# The hand-written EXERCISE_INFO entries above always take priority.
@st.cache_data(show_spinner=False)
def load_exercise_library():
    import json, os
    # Look next to this script first, then the working directory. Falls back to
    # an empty library so a missing file degrades gracefully rather than crashing.
    candidates = []
    try:
        candidates.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       "exercise_library.json"))
    except NameError:
        pass
    candidates.append(os.path.join(os.getcwd(), "exercise_library.json"))

    payload = None
    for path in candidates:
        try:
            with open(path, encoding="utf-8") as fh:
                payload = json.load(fh)
                break
        except Exception:
            continue
    if payload is None:
        return {}
    base = payload.get("image_base", "")
    out = {}
    for name, entry in payload.get("exercises", {}).items():
        out[name] = {
            "muscles": entry.get("muscles", []),
            "cues": entry.get("cues", []),
            "images": [base + img for img in entry.get("images", [])],
        }
    return out


def _normalise_exercise_name(name):
    """Lowercase, drop set/rep notation and punctuation, collapse whitespace.

    So 'Pull ups', 'PULL-UPS' and 'Pull Ups 3x10' all resolve to the same thing.
    """
    n = base_exercise_name(name).lower()
    n = re.sub(r"[^a-z0-9 ]+", " ", n)
    return re.sub(r"\s+", " ", n).strip()


# Common gym shorthand -> a name that exists in the library.
EXERCISE_ALIASES = {
    "rdl": "Romanian Deadlift", "sldl": "Stiff-Legged Barbell Deadlift",
    "ohp": "Standing Military Press", "overhead press": "Standing Military Press",
    "bench": "Barbell Bench Press - Medium Grip",
    "bench press": "Barbell Bench Press - Medium Grip",
    "incline bench": "Barbell Incline Bench Press - Medium Grip",
    "squat": "Barbell Squat", "back squat": "Barbell Squat",
    "front squat": "Front Barbell Squat", "deadlift": "Barbell Deadlift",
    "pull up": "Pullups", "pull ups": "Pullups", "pullup": "Pullups",
    "chin up": "Chin-Up", "chin ups": "Chin-Up",
    "dip": "Dips - Triceps Version", "dips": "Dips - Triceps Version",
    "row": "Bent Over Barbell Row", "rows": "Bent Over Barbell Row",
    "barbell row": "Bent Over Barbell Row", "seated row": "Seated Cable Rows",
    "lat pulldown": "Wide-Grip Lat Pulldown", "pulldown": "Wide-Grip Lat Pulldown",
    "curl": "Barbell Curl", "curls": "Barbell Curl", "bicep curl": "Barbell Curl",
    "hammer curl": "Hammer Curls", "tricep pushdown": "Triceps Pushdown",
    "skullcrusher": "Lying Triceps Press", "skull crusher": "Lying Triceps Press",
    "lateral raise": "Side Lateral Raise", "lat raise": "Side Lateral Raise",
    "face pull": "Face Pull", "shrug": "Barbell Shrug", "shrugs": "Barbell Shrug",
    "leg press": "Leg Press", "leg curl": "Lying Leg Curls",
    "leg extension": "Leg Extensions", "lunge": "Dumbbell Lunges",
    "lunges": "Dumbbell Lunges", "hip thrust": "Barbell Hip Thrust",
    "calf raise": "Standing Calf Raises", "calves": "Standing Calf Raises",
    "plank": "Plank", "abs": "Cable Crunch", "core": "Cable Crunch",
    "push up": "Pushups", "push ups": "Pushups", "pushup": "Pushups",
    "split squat": "Split Squats", "bulgarian split squat": "Split Squat with Dumbbells",
    "rear delt fly": "Cable Rear Delt Fly", "reverse fly": "Reverse Flyes",
    "upright row": "Standing Dumbbell Upright Row", "glute bridge": "Barbell Glute Bridge",
    "t bar row": "T-Bar Row with Handle", "tbar row": "T-Bar Row with Handle",
    "farmers walk": "Farmer's Walk", "farmer walk": "Farmer's Walk",
    "leg extension": "Leg Extensions", "leg extensions": "Leg Extensions",
    "crunch": "Crunches", "sit up": "Crunches", "situps": "Crunches",
    "goblet squat": "Goblet Squat", "close grip bench": "Close-Grip Barbell Bench Press",
    "military press": "Standing Military Press", "shoulder press": "Dumbbell Shoulder Press",
    "incline press": "Incline Dumbbell Press", "cable fly": "Cable Crossover",
    "preacher curl": "Preacher Curl", "good morning": "Good Morning",
}


def find_exercise_info(name):
    """Best-effort lookup: hand-written entry, then alias, then library, then
    a fuzzy substring match. Returns None only if nothing plausible is found."""
    # 1. exact hand-written entry (covers your own plan's shorthand)
    info = EXERCISE_INFO.get(base_exercise_name(name))
    if info:
        return info

    library = load_exercise_library()
    if not library:
        return None

    norm = _normalise_exercise_name(name)
    if not norm:
        return None

    # 2. case-insensitive match against hand-written entries
    for key, entry in EXERCISE_INFO.items():
        if _normalise_exercise_name(key) == norm:
            return entry

    # 3. alias table
    alias = EXERCISE_ALIASES.get(norm)
    if alias and alias in library:
        return library[alias]

    # 4. exact (normalised) library match
    norm_library = {_normalise_exercise_name(k): v for k, v in library.items()}
    if norm in norm_library:
        return norm_library[norm]

    # 5. fuzzy: shortest library name containing the query, or vice versa
    candidates = [(k, v) for k, v in norm_library.items() if norm in k or k in norm]
    if candidates:
        return min(candidates, key=lambda kv: len(kv[0]))[1]
    return None


def all_known_exercise_names():
    """Sorted names for the swap picker: your plan's own entries plus the library."""
    names = set(EXERCISE_INFO.keys()) | set(load_exercise_library().keys())
    return sorted(names)


@st.cache_data(ttl=600, show_spinner=False)
def exercise_muscle_filters():
    """Clean muscle groups for the swap filter dropdown.

    Only the library's canonical labels are used. The hand-written EXERCISE_INFO
    entries use looser wording ("Core / Abdominals", "Full body mobility") which
    would otherwise clutter the list with near-duplicates.
    """
    groups = set()
    for entry in load_exercise_library().values():
        groups.update(entry.get("muscles", []))
    return sorted(groups)


@st.cache_data(ttl=600, show_spinner=False)
def search_exercises(query="", muscle=None, limit=400):
    """Filter the exercise catalogue by free-text query and/or muscle group.

    Matching is on the normalised name, so 'db press' finds 'Dumbbell Press'
    and case/punctuation don't matter. Results are ordered so names that START
    with the query come first — typing 'row' should surface 'Rowing' style
    matches before 'Bent Over Barbell Row'.
    """
    library = dict(load_exercise_library())
    for k, v in EXERCISE_INFO.items():
        library.setdefault(k, v)

    q = _normalise_exercise_name(query) if query else ""
    starts, contains = [], []
    for name, entry in library.items():
        if muscle and muscle not in (entry.get("muscles") or []):
            continue
        if q:
            norm = _normalise_exercise_name(name)
            if norm.startswith(q):
                starts.append(name)
            elif q in norm:
                contains.append(name)
        else:
            contains.append(name)
    return (sorted(starts) + sorted(contains))[:limit]


LOG_TYPES = {
    "weight_reps": "Weight & reps",
    "bodyweight": "Reps only (bodyweight)",
    "duration": "Time & distance",
}


def default_log_type(exercise_name):
    """How should this movement be logged? Falls back to weight & reps.

    A 20-minute incline walk has no meaningful 'kg' or 'reps', and a set of
    pull-ups has no barbell load, so forcing every movement into the same three
    boxes produces nonsense data.
    """
    info = EXERCISE_INFO.get(base_exercise_name(exercise_name)) or {}
    return info.get("log_type", "weight_reps")


def estimated_1rm(weight_kg, reps):
    """Epley formula. A rough estimate, useful for trend-spotting, not a max test."""
    if not weight_kg or not reps or reps < 1:
        return 0.0
    return float(weight_kg) * (1 + reps / 30.0)

_FEDB = "https://raw.githubusercontent.com/yuhonas/free-exercise-db/main/exercises/"

EXERCISE_INFO = {
    "Bench Press": {
        "muscles": ["Chest", "Front Shoulders", "Triceps"],
        "cues": ["Retract shoulder blades and keep them pinned", "Feet flat and driving into the floor",
                 "Lower the bar to mid-chest with control", "Press up and slightly back over your face"],
        "images": [_FEDB + "Barbell_Bench_Press_-_Medium_Grip/0.jpg", _FEDB + "Barbell_Bench_Press_-_Medium_Grip/1.jpg"],
    },
    "Incline DB Press": {
        "muscles": ["Upper Chest", "Front Shoulders", "Triceps"],
        "cues": ["Bench at a 30-45° incline", "Keep wrists stacked over elbows",
                 "Lower dumbbells to chest level with control", "Avoid flaring elbows past 45°"],
        "images": [_FEDB + "Incline_Dumbbell_Press/0.jpg", _FEDB + "Incline_Dumbbell_Press/1.jpg"],
    },
    "Shoulder Press": {
        "muscles": ["Shoulders", "Triceps", "Upper Chest"],
        "cues": ["Brace your core, avoid over-arching the lower back", "Press straight overhead, not forward",
                 "Full lockout at the top without shrugging"],
        "images": [_FEDB + "Dumbbell_Shoulder_Press/0.jpg", _FEDB + "Dumbbell_Shoulder_Press/1.jpg"],
    },
    "Lateral Raise": {
        "muscles": ["Side Shoulders"],
        "cues": ["Slight bend in the elbows throughout", "Lead with the elbows, not the hands",
                 "Raise to roughly shoulder height", "Control the descent — don't swing"],
        "images": [_FEDB + "Side_Lateral_Raise/0.jpg", _FEDB + "Side_Lateral_Raise/1.jpg"],
    },
    "Cable Fly": {
        "muscles": ["Chest"],
        "cues": ["Slight forward lean, chest up", "Soft bend in the elbows",
                 "Squeeze at the midline, don't just swing the arms", "Control the stretch back out"],
        "images": [_FEDB + "Cable_Crossover/0.jpg", _FEDB + "Cable_Crossover/1.jpg"],
    },
    "Triceps": {
        "muscles": ["Triceps"],
        "cues": ["Keep elbows tucked and stationary", "Full extension at the bottom",
                 "Control the eccentric (lowering) portion"],
        "images": [_FEDB + "Triceps_Pushdown/0.jpg", _FEDB + "Triceps_Pushdown/1.jpg"],
    },
    "Deadlift": {
        "muscles": ["Hamstrings", "Glutes", "Lower Back", "Lats", "Traps"],
        "cues": ["Bar over mid-foot to start", "Flat back — brace before you pull",
                 "Push the floor away rather than yanking the bar", "Hips and shoulders rise together"],
        "images": [_FEDB + "Barbell_Deadlift/0.jpg", _FEDB + "Barbell_Deadlift/1.jpg"],
    },
    "Lat Pulldown": {
        "muscles": ["Lats", "Biceps", "Rear Shoulders"],
        "cues": ["Slight lean back, chest up", "Pull elbows down and back",
                 "Bring the bar to upper chest", "Control the return, don't let it snap up"],
        "images": [_FEDB + "Wide-Grip_Lat_Pulldown/0.jpg", _FEDB + "Wide-Grip_Lat_Pulldown/1.jpg"],
    },
    "Rows": {
        "muscles": ["Lats", "Rhomboids", "Rear Shoulders", "Biceps"],
        "cues": ["Flat back, hinge at the hips", "Pull elbows back, squeeze shoulder blades",
                 "Avoid using momentum to heave the weight"],
        "images": [_FEDB + "Bent_Over_Barbell_Row/0.jpg", _FEDB + "Bent_Over_Barbell_Row/1.jpg"],
    },
    "Face Pulls": {
        "muscles": ["Rear Shoulders", "Upper Back", "Rotator Cuff"],
        "cues": ["Pull to eye level", "Lead with the elbows high",
                 "Externally rotate at the end range", "Light weight, controlled tempo"],
        "images": [_FEDB + "Face_Pull/0.jpg", _FEDB + "Face_Pull/1.jpg"],
    },
    "Curls": {
        "muscles": ["Biceps"],
        "cues": ["Elbows pinned at your sides", "No swinging — control the weight",
                 "Full range of motion, squeeze at the top"],
        "images": [_FEDB + "Barbell_Curl/0.jpg", _FEDB + "Barbell_Curl/1.jpg"],
    },
    "Squat": {
        "muscles": ["Quads", "Glutes", "Hamstrings", "Core"],
        "cues": ["Feet roughly shoulder width", "Chest up, brace your core",
                 "Break at the hips and knees together", "Knees track over the toes", "Drive through mid-foot to stand"],
        "images": [_FEDB + "Barbell_Squat/0.jpg", _FEDB + "Barbell_Squat/1.jpg"],
    },
    "RDL": {
        "muscles": ["Hamstrings", "Glutes", "Lower Back"],
        "cues": ["Soft knee bend, hinge at the hips", "Keep the bar/dumbbells close to your legs",
                 "Flat back throughout", "Feel the stretch in the hamstrings, then drive hips forward"],
        "images": [_FEDB + "Romanian_Deadlift/0.jpg", _FEDB + "Romanian_Deadlift/1.jpg"],
    },
    "Leg Press": {
        "muscles": ["Quads", "Glutes", "Hamstrings"],
        "cues": ["Feet shoulder width on the platform", "Lower until knees reach ~90°",
                 "Don't let your lower back round off the pad", "Avoid locking knees hard at the top"],
        "images": [_FEDB + "Leg_Press/0.jpg", _FEDB + "Leg_Press/1.jpg"],
    },
    "Lunges": {
        "muscles": ["Quads", "Glutes", "Hamstrings"],
        "cues": ["Step far enough forward for a 90° front knee", "Keep torso upright",
                 "Back knee drops toward the floor without slamming it", "Push through the front heel to return"],
        "images": [_FEDB + "Dumbbell_Lunges/0.jpg", _FEDB + "Dumbbell_Lunges/1.jpg"],
    },
    "Leg Curl": {
        "muscles": ["Hamstrings"],
        "cues": ["Hips pinned to the pad", "Curl through a full range of motion",
                 "Control the weight on the way back"],
        "images": [_FEDB + "Lying_Leg_Curls/0.jpg", _FEDB + "Lying_Leg_Curls/1.jpg"],
    },
    "Calves": {
        "muscles": ["Calves"],
        "cues": ["Full stretch at the bottom", "Rise fully onto the toes",
                 "Pause briefly at the top for a real contraction"],
        "images": [_FEDB + "Standing_Calf_Raises/0.jpg", _FEDB + "Standing_Calf_Raises/1.jpg"],
    },
    "Incline Bench": {
        "muscles": ["Upper Chest", "Front Shoulders", "Triceps"],
        "cues": ["Bench at a 30-45° incline", "Retract shoulder blades",
                 "Lower to upper chest with control", "Press up and slightly back"],
        "images": [_FEDB + "Barbell_Incline_Bench_Press_-_Medium_Grip/0.jpg", _FEDB + "Barbell_Incline_Bench_Press_-_Medium_Grip/1.jpg"],
    },
    "Pull Ups/Pulldown": {
        "log_type": "bodyweight",
        "muscles": ["Lats", "Biceps", "Upper Back"],
        "cues": ["Full hang at the bottom", "Pull elbows down and back, chest to the bar",
                 "Avoid excessive kipping/swinging"],
        "images": [_FEDB + "Pullups/0.jpg", _FEDB + "Pullups/1.jpg"],
    },
    "Pull Ups": {
        "log_type": "bodyweight",
        "muscles": ["Lats", "Biceps", "Upper Back"],
        "cues": ["Full hang at the bottom, shoulders active", "Pull elbows down and back, chin over the bar",
                 "Lower under control rather than dropping"],
        "images": [_FEDB + "Pullups/0.jpg", _FEDB + "Pullups/1.jpg"],
    },
    "Chest Press": {
        "muscles": ["Chest", "Front Shoulders", "Triceps"],
        "cues": ["Seat height so handles line up with mid-chest", "Press forward without shrugging",
                 "Control the return to a full stretch"],
        "images": [_FEDB + "Cable_Chest_Press/0.jpg", _FEDB + "Cable_Chest_Press/1.jpg"],
    },
    "Row": {
        "muscles": ["Lats", "Rhomboids", "Rear Shoulders", "Biceps"],
        "cues": ["Chest against the pad or flat back if standing", "Pull elbows back, squeeze shoulder blades",
                 "Avoid jerking the weight with momentum"],
        "images": [_FEDB + "Seated_Cable_Rows/0.jpg", _FEDB + "Seated_Cable_Rows/1.jpg"],
    },
    "Shoulders": {
        "muscles": ["Shoulders"],
        "cues": ["Controlled tempo, avoid using momentum", "Full range of motion",
                 "Keep the core braced to protect the lower back"],
        "images": [_FEDB + "Dumbbell_Shoulder_Press/0.jpg", _FEDB + "Dumbbell_Shoulder_Press/1.jpg"],
    },
    "Arms": {
        "muscles": ["Biceps", "Triceps"],
        "cues": ["Elbows stay fixed in place", "Control both the lift and the lowering phase",
                 "Full range of motion each rep"],
        "images": [_FEDB + "Barbell_Curl/0.jpg", _FEDB + "Barbell_Curl/1.jpg"],
    },
    "Cardio": {
        "log_type": "duration",
        "muscles": ["Heart/Lungs (conditioning)"],
        "cues": ["Warm up for a few minutes first", "Keep a pace you can sustain for the full duration",
                 "Cool down and stretch after"],
        "images": [],
    },
    "Abs": {
        "log_type": "bodyweight",
        "muscles": ["Core / Abdominals"],
        "cues": ["Control the movement, avoid yanking with the neck", "Exhale on the contraction",
                 "Keep the lower back from over-arching"],
        "images": [_FEDB + "Cable_Crunch/0.jpg", _FEDB + "Cable_Crunch/1.jpg"],
    },
    "Stretch": {
        "log_type": "duration",
        "muscles": ["Full body mobility"],
        "cues": ["Hold each stretch 20-30 seconds", "Breathe, don't bounce",
                 "Stretch to mild tension, not pain"],
        "images": [],
    },
    # --- Walking / rest entries: no demo image needed, just context ---
    "15 min incline walk": {
        "log_type": "duration",
        "muscles": ["Calves", "Glutes", "Heart/Lungs"],
        "cues": ["Treadmill incline around 8-12%", "Comfortable pace you can hold the whole time",
                 "Don't hold the handrails — it kills the effort"],
        "images": [],
    },
    "20 min incline walk": {
        "log_type": "duration",
        "muscles": ["Calves", "Glutes", "Heart/Lungs"],
        "cues": ["Treadmill incline around 8-12%", "Comfortable pace you can hold the whole time",
                 "Don't hold the handrails — it kills the effort"],
        "images": [],
    },
    "30-45 min walk": {
        "log_type": "duration",
        "muscles": ["Heart/Lungs", "Legs"],
        "cues": ["Easy conversational pace", "Outdoors or treadmill both fine",
                 "Great for hitting the daily step target"],
        "images": [],
    },
    "45 min walk or football": {
        "log_type": "duration",
        "muscles": ["Heart/Lungs", "Legs"],
        "cues": ["Keep it genuinely easy — this is recovery, not a session",
                 "If playing football, warm up properly first"],
        "images": [],
    },
    "Rest and 10k steps": {
        "log_type": "duration",
        "muscles": ["Recovery day"],
        "cues": ["No lifting today — let the body repair", "Still aim for your step target",
                 "Prioritise sleep and hydration"],
        "images": [],
    },
}
# NOTE: images come from https://github.com/yuhonas/free-exercise-db (public
# domain / Unlicense). These are two static photos (start position, end
# position) per exercise, NOT an animated GIF — I checked, and I couldn't find
# a source of real looping demo GIFs that reliably hotlinks and covers this
# exercise list without risking broken images or licensing issues. Shown
# side-by-side it still gives a clear "how it looks" reference. If you find a
# GIF host you like better, just swap the URLs in the "images" lists above —
# any number of image URLs in that list will render side by side.


def base_exercise_name(name):
    """Strip trailing set/rep notation like '4x6-8' or '3x10' so the name
    can be matched against EXERCISE_INFO regardless of the day's set/rep scheme."""
    cleaned = re.sub(r"\s*\d+\s*x\s*[\d\-]+\s*$", "", name, flags=re.IGNORECASE).strip()
    return cleaned


# ---------------------------------------------------------------
# TRANSLATIONS (UI chrome only — your own data/notes stay as typed)
# ---------------------------------------------------------------
BASE_EN = {
    "app_title": "⚡ Momentum", "nav_today": "Today", "nav_weight_log": "Weight Log",
    "nav_weekly_dashboard": "Weekly Dashboard", "nav_measurements": "Measurements",
    "nav_photos": "Progress Photos", "nav_settings": "Settings", "nav_profile": "Profile",
    "gym_label": "Gym", "meals_header": "🍽️ Meals", "workout_header": "🏋️ Workout",
    "daily_numbers_header": "📊 Daily Numbers", "weight_section_header": "⚖️ Weight",
    "notes_label": "Notes (optional)", "save_button": "💾 Save today's log", "saved_msg": "Saved!",
    "prev_day": "⬅ Prev day", "next_day": "Next day ➡", "done_label": "Done",
    "what_did_you_have": "What did you have?", "calories_label": "Calories", "protein_label": "Protein (g)",
    "use_meal_total": "Use meal total", "water_label": "Water (L)", "steps_label": "Steps",
    "weight_label": "Weight (kg)", "weight_trend_header": "⚖️ Weight Trend",
    "no_weight_entries": "No weight entries yet. Log your weight on the Today page.",
    "latest_weight": "Latest weight", "change_period": "Change (period)", "entries_logged": "Entries logged",
    "weekly_adherence_header": "📅 Weekly Adherence", "meals_hit_avg": "Meals hit (avg)",
    "workouts_completed": "Workouts completed", "protein_target_days": "Protein target days",
    "steps_target_days": "Steps target days", "measurements_header": "📏 Body Measurements",
    "waist_label": "Waist (cm)", "chest_label": "Chest (cm)", "hips_label": "Hips (cm)",
    "arms_label": "Arms (cm)", "thighs_label": "Thighs (cm)", "save_measurement": "Save measurement",
    "photos_header": "📸 Progress Photos", "upload_photo": "Upload a photo",
    "caption_label": "Caption (optional)", "save_photo": "Save photo",
    "settings_header": "⚙️ Settings", "edit_targets_header": "Edit your targets",
    "save_targets": "Save targets", "language_label": "Language",
    "streak_header": "🔥 Streak & Badges", "current_streak_label": "Current streak",
    "longest_streak_label": "Longest streak", "days_logged_label": "Days logged",
    "badges_label": "Badges earned", "compare_header": "Compare two photos", "delete_label": "Delete",
    "profile_header": "🙋 Your Profile", "goal_label": "Goal", "height_label": "Height (cm)",
    "current_weight_label": "Current Weight (kg)", "age_label": "Age",
    "sex_label": "Biological sex (for calorie calculation accuracy)", "country_label": "Country",
    "activity_label": "Activity level", "recalc_button": "Save & recalculate my targets",
    "targets_updated_msg": "Targets updated based on your profile!",
    "profile_disclaimer": "These are estimates from standard formulas, not medical advice. Adjust freely in Settings.",
    "bottle_0": "Let's get started 💪", "bottle_25": "Nice, warming up! 🔥",
    "bottle_50": "Halfway there, keep going! 💦", "bottle_75": "Almost full — don't stop now! ⚡",
    "bottle_100": "🎉 Bottle full! Workout complete!",
    "extra_exercises_header": "➕ Extra (not in today's plan)",
    "add_extra_placeholder": "e.g. Extra push-ups, evening run...",
    "add_extra_button": "Add",
    "nav_home": "Home", "nav_achievements": "Achievements",
    "momentum_score_label": "⚡ Today's Momentum Score",
    "coach_header": "🤖 Coach's Notes", "coach_tomorrow_focus": "Tomorrow, focus on:",
    "achievements_header": "🥇 Achievements", "achievements_unlocked": "unlocked",
    "lifetime_stats_header": "📊 Lifetime Stats", "personal_records_header": "🏅 Personal Records",
    "prediction_header": "📈 Weight Prediction", "prediction_30": "Projected weight in 30 days",
    "prediction_90": "Projected weight in 90 days", "prediction_insufficient": "Log a few more weight entries (at least 2, spread over time) to see a prediction.",
    "perfect_day_header": "🏆 PERFECT DAY ACHIEVED", "perfect_day_sub": "Every target hit today. Incredible work.",
    "quote_of_day_label": "💬 Quote of the Day", "verse_of_day_label": "📖 Verse of the Day", "next_badge_label": "Next Badge",
    "todays_workout_label": "Today's Workout", "weekly_change_label": "Weekly Change",
    "current_weight_home_label": "Current Weight", "lifetime_workouts_label": "Total Workouts",
    "lifetime_steps_label": "Lifetime Steps", "lifetime_protein_label": "Lifetime Protein",
    "lifetime_water_label": "Lifetime Water", "pr_max_steps_label": "Most Steps in a Day",
    "pr_longest_streak_label": "Longest Streak", "pr_max_protein_label": "Highest Protein Day",
    "advanced_trends_header": "📈 Advanced Trends (last 30 days)",
    "photo_timeline_header": "🕒 Photo Timeline",
    "prev_week": "⬅ Prev week", "next_week": "Next week ➡", "back_to_this_week": "Back to this week",
    "nav_gym_bro": "🤖 Gym Bro", "gym_bro_header": "🤖 Gym Bro",
    "gym_bro_intro": "Ask me anything about training, nutrition, or recovery.",
    "gym_bro_placeholder": "Ask Gym Bro a question...",
    "gym_bro_disclaimer": "Gym Bro gives general fitness/nutrition info, not medical advice. For injuries, pain, or health conditions, see a doctor.",
    "gym_bro_missing_key": "Gym Bro needs a free Gemini API key to work. Add it in your secrets.toml — see the setup notes.",
    "exercise_info_label": "ℹ️ How to do it",
    "muscles_targeted_label": "Muscles targeted",
    "form_cues_label": "Form cues",
    "no_info_label": "No demo added yet for this exercise — fill in EXERCISE_INFO to add one.",
    "swap_label": "🔄 Swap this exercise",
    "swap_placeholder": "…or type any exercise name",
    "swap_help": "Search or filter, then pick — you'll get demo photos and form cues. "
                 "Or type a name that isn't listed and hit Swap anyway.",
    "swap_pick_placeholder": "— choose a replacement —",
    "swap_search_label": "Search exercises",
    "swap_search_placeholder": "e.g. dip, row, curl…",
    "swap_muscle_label": "Filter by muscle", "swap_all_muscles": "All muscles",
    "swap_matches_suffix": "matches — pick one below",
    "swap_narrow_hint": "Too many to list. Type in the search box above to narrow it down.",
    "swap_narrow_help": "Type in the search box above to shorten this list.",
    "swap_no_matches": "Nothing matched. Clear the filters, or just hit Swap to use "
                       "exactly what you typed.",
    "no_demo_for_swap": "No demo found for this name. Try picking from the list above.",
    "swap_confirm": "Swap",
    "revert_swap_label": "↩ Revert to original",
    "swapped_from_label": "Swapped from",
    "gym_bro_clear": "🗑 Clear chat",
    # --- sets / reps / weight logging ---
    "sets_header": "Sets", "set_label": "Set", "reps_label": "Reps",
    "weight_kg_label": "kg", "add_set": "➕ Add set", "remove_set": "➖ Remove last set",
    "last_time_label": "Last time", "best_ever_label": "Best ever",
    "last_peak_label": "Last session peak",
    "new_pb_label": "New personal best —", "no_history_label": "No history yet for this lift",
    "session_volume_label": "Session volume", "e1rm_label": "Est. 1RM",
    "calories_burned_label": "Est. kcal", "session_time_label": "Est. session time",
    "calorie_per_exercise_help": "Rough MET-based estimate over ~{mins} min of work and rest. "
                                 "Expect roughly ±30% accuracy.",
    "day_total_calories_label": "Whole session — est. calories",
    "calorie_detail_header": "🔥 Calorie estimate — how it's worked out",
    "calorie_method_note": "Based on MET values from the 2024 Adult Compendium of Physical "
                           "Activities, using a bodyweight of {bw} kg. Lifting is scored between "
                           "3.5 METs (moderate) and 6.0 METs (vigorous) depending on how heavy "
                           "the average rep was; walks and cardio use their logged time.",
    "calorie_accuracy_warning": "Treat this as a rough guide, not a measurement. Real energy "
                                "cost varies a lot with rest length, effort, fitness and body "
                                "composition — a realistic error range is roughly ±30%. Useful "
                                "for spotting trends between your own sessions; not accurate "
                                "enough to plan a calorie deficit around.",
    "exercise_note_label": "Note for this exercise",
    "log_type_label": "How do you log this?",
    "per_side_label": "Weight is per hand (dumbbells)",
    "per_side_help": "Tick this if you hold the weight in each hand — e.g. 22kg dumbbells "
                     "means 44kg total load. Enter the weight of ONE dumbbell.",
    "weight_each_label": "kg (each)",
    "duration_label": "Minutes", "distance_label": "Distance (km)",
    "total_time_label": "Total time", "total_distance_label": "Distance",
    # --- rest timer ---
    "rest_timer_label": "⏱ Rest timer", "start_timer": "Start",
    # --- tabs ---
    "tab_workout": "🏋️ Workout", "tab_meals": "🍽️ Meals", "tab_numbers": "📊 Numbers",
    # --- plan editor ---
    "nav_plan": "Workout Plan", "plan_header": "🗓️ Edit Your Workout Plan",
    "plan_intro": "Change your split here — no code editing needed. Applies from today onwards.",
    "day_label_label": "Day label", "exercises_label": "Exercises (one per line)",
    "save_plan": "Save plan", "reset_plan": "↺ Reset this day to default",
    # --- rest week ---
    "rest_week_label": "Mark this week as a deload / rest week",
    "rest_week_on": "🌙 Deload week — adherence stats won't count against you.",
    # --- PRs & analytics ---
    "lift_prs_header": "🏋️ Lift Records", "strength_trend_header": "📈 Strength Trend",
    "pick_lift_label": "Pick a lift",
    "sessions_logged_label": "Sessions", "e1rm_now_label": "Current est. 1RM",
    "e1rm_peak_label": "Best ever", "heaviest_set_label": "Heaviest set",
    "logged_on_label": "Logged on",
    "trend_one_session": "Only one session logged for this lift so far — train it again "
                         "and a trend line will appear here.",
    "trend_no_data": "No weighted sets recorded for this lift yet.",
    "no_weighted_sets": "No weighted sets logged yet. Add weight and reps on the Today "
                        "page and your lifts will show up here.", "muscle_volume_header": "💪 Weekly Muscle Volume",
    "muscle_volume_caption": "Total kg lifted per muscle region this week",
    "total_volume_label": "Total lifted this week",
    "volume_overlap_note": "Muscle figures overlap — a bench press counts toward chest, shoulders and triceps, so they add up to more than the total.",
    "no_volume_yet": "Log some sets with weights to see your muscle volume breakdown.",
    # --- export ---
    "export_header": "📤 Export Data", "export_caption": "Download your logs as CSV.",
    "download_daily": "⬇ Daily log", "download_sets": "⬇ Training sets",
    "download_measurements": "⬇ Measurements",
}

BASE_AF = {
    "nav_today": "Vandag", "nav_weight_log": "Gewigslog", "nav_weekly_dashboard": "Weeklikse Oorsig",
    "nav_measurements": "Afmetings", "nav_photos": "Vorderingsfoto's", "nav_settings": "Instellings",
    "nav_profile": "Profiel", "gym_label": "Gimnasium", "meals_header": "🍽️ Maaltye",
    "workout_header": "🏋️ Oefening", "daily_numbers_header": "📊 Daaglikse Syfers",
    "weight_section_header": "⚖️ Gewig", "notes_label": "Notas (opsioneel)",
    "save_button": "💾 Stoor vandag se log", "saved_msg": "Gestoor!", "prev_day": "⬅ Vorige dag",
    "next_day": "Volgende dag ➡", "done_label": "Klaar", "what_did_you_have": "Wat het jy geëet?",
    "calories_label": "Kalorieë", "protein_label": "Proteïen (g)", "use_meal_total": "Gebruik maaltyd totaal",
    "water_label": "Water (L)", "steps_label": "Stappe", "weight_label": "Gewig (kg)",
    "weight_trend_header": "⚖️ Gewigstendens",
    "no_weight_entries": "Nog geen gewig aangeteken nie. Teken jou gewig op die Vandag-bladsy aan.",
    "latest_weight": "Jongste gewig", "change_period": "Verandering (periode)", "entries_logged": "Inskrywings",
    "weekly_adherence_header": "📅 Weeklikse Nakoming", "meals_hit_avg": "Maaltye behaal (gem.)",
    "workouts_completed": "Oefeninge voltooi", "protein_target_days": "Proteïen-teiken dae",
    "steps_target_days": "Stappe-teiken dae", "measurements_header": "📏 Liggaamsafmetings",
    "waist_label": "Middel (cm)", "chest_label": "Bors (cm)", "hips_label": "Heupe (cm)",
    "arms_label": "Arms (cm)", "thighs_label": "Boude (cm)", "save_measurement": "Stoor afmeting",
    "photos_header": "📸 Vorderingsfoto's", "upload_photo": "Laai 'n foto op",
    "caption_label": "Byskrif (opsioneel)", "save_photo": "Stoor foto",
    "settings_header": "⚙️ Instellings", "edit_targets_header": "Wysig jou teikens",
    "save_targets": "Stoor teikens", "language_label": "Taal",
    "streak_header": "🔥 Reeks & Kentekens", "current_streak_label": "Huidige reeks",
    "longest_streak_label": "Langste reeks", "days_logged_label": "Dae aangeteken",
    "badges_label": "Kentekens verdien", "compare_header": "Vergelyk twee foto's", "delete_label": "Verwyder",
    "profile_header": "🙋 Jou Profiel", "goal_label": "Doel", "height_label": "Lengte (cm)",
    "current_weight_label": "Huidige Gewig (kg)", "age_label": "Ouderdom",
    "sex_label": "Biologiese geslag (vir akkurate berekening)", "country_label": "Land",
    "activity_label": "Aktiwiteitsvlak", "recalc_button": "Stoor & herbereken my teikens",
    "targets_updated_msg": "Teikens opgedateer op grond van jou profiel!",
    "profile_disclaimer": "Dit is skattings van standaardformules, nie mediese advies nie. Pas gerus aan by Instellings.",
    "bottle_0": "Kom ons begin 💪", "bottle_25": "Mooi, warm op! 🔥",
    "bottle_50": "Op pad, hou aan! 💦", "bottle_75": "Amper vol — moenie nou ophou nie! ⚡",
    "bottle_100": "🎉 Bottel vol! Oefening voltooi!",
}

BASE_ES = {
    "nav_today": "Hoy", "nav_weight_log": "Registro de Peso", "nav_weekly_dashboard": "Panel Semanal",
    "nav_measurements": "Medidas", "nav_photos": "Fotos de Progreso", "nav_settings": "Configuración",
    "nav_profile": "Perfil", "gym_label": "Gimnasio", "meals_header": "🍽️ Comidas",
    "workout_header": "🏋️ Entrenamiento", "daily_numbers_header": "📊 Datos Diarios",
    "weight_section_header": "⚖️ Peso", "notes_label": "Notas (opcional)",
    "save_button": "💾 Guardar registro de hoy", "saved_msg": "¡Guardado!", "prev_day": "⬅ Día anterior",
    "next_day": "Día siguiente ➡", "done_label": "Hecho", "what_did_you_have": "¿Qué comiste?",
    "calories_label": "Calorías", "protein_label": "Proteína (g)", "use_meal_total": "Usar total de comidas",
    "water_label": "Agua (L)", "steps_label": "Pasos", "weight_label": "Peso (kg)",
    "weight_trend_header": "⚖️ Tendencia de Peso",
    "no_weight_entries": "Aún no hay registros de peso. Registra tu peso en la página de Hoy.",
    "latest_weight": "Peso más reciente", "change_period": "Cambio (período)", "entries_logged": "Registros",
    "weekly_adherence_header": "📅 Cumplimiento Semanal", "meals_hit_avg": "Comidas logradas (prom.)",
    "workouts_completed": "Entrenamientos completados", "protein_target_days": "Días con meta de proteína",
    "steps_target_days": "Días con meta de pasos", "measurements_header": "📏 Medidas Corporales",
    "waist_label": "Cintura (cm)", "chest_label": "Pecho (cm)", "hips_label": "Caderas (cm)",
    "arms_label": "Brazos (cm)", "thighs_label": "Muslos (cm)", "save_measurement": "Guardar medida",
    "photos_header": "📸 Fotos de Progreso", "upload_photo": "Subir una foto",
    "caption_label": "Descripción (opcional)", "save_photo": "Guardar foto",
    "settings_header": "⚙️ Configuración", "edit_targets_header": "Edita tus metas",
    "save_targets": "Guardar metas", "language_label": "Idioma",
    "streak_header": "🔥 Racha e Insignias", "current_streak_label": "Racha actual",
    "longest_streak_label": "Racha más larga", "days_logged_label": "Días registrados",
    "badges_label": "Insignias obtenidas", "compare_header": "Comparar dos fotos", "delete_label": "Eliminar",
    "profile_header": "🙋 Tu Perfil", "goal_label": "Objetivo", "height_label": "Altura (cm)",
    "current_weight_label": "Peso Actual (kg)", "age_label": "Edad",
    "sex_label": "Sexo biológico (para mayor precisión)", "country_label": "País",
    "activity_label": "Nivel de actividad", "recalc_button": "Guardar y recalcular mis metas",
    "targets_updated_msg": "¡Metas actualizadas según tu perfil!",
    "profile_disclaimer": "Estas son estimaciones de fórmulas estándar, no un consejo médico. Ajusta libremente en Configuración.",
    "bottle_0": "Empecemos 💪", "bottle_25": "¡Bien, calentando! 🔥",
    "bottle_50": "¡A mitad de camino, sigue así! 💦", "bottle_75": "¡Casi lleno, no pares ahora! ⚡",
    "bottle_100": "🎉 ¡Botella llena! ¡Entrenamiento completo!",
}

BASE_FR = {
    "nav_today": "Aujourd'hui", "nav_weight_log": "Suivi du Poids", "nav_weekly_dashboard": "Tableau Hebdomadaire",
    "nav_measurements": "Mesures", "nav_photos": "Photos de Progrès", "nav_settings": "Paramètres",
    "nav_profile": "Profil", "gym_label": "Salle de sport", "meals_header": "🍽️ Repas",
    "workout_header": "🏋️ Entraînement", "daily_numbers_header": "📊 Chiffres du Jour",
    "weight_section_header": "⚖️ Poids", "notes_label": "Notes (facultatif)",
    "save_button": "💾 Enregistrer le journal du jour", "saved_msg": "Enregistré !", "prev_day": "⬅ Jour précédent",
    "next_day": "Jour suivant ➡", "done_label": "Fait", "what_did_you_have": "Qu'avez-vous mangé ?",
    "calories_label": "Calories", "protein_label": "Protéines (g)", "use_meal_total": "Utiliser le total des repas",
    "water_label": "Eau (L)", "steps_label": "Pas", "weight_label": "Poids (kg)",
    "weight_trend_header": "⚖️ Tendance du Poids",
    "no_weight_entries": "Aucune entrée de poids pour l'instant. Enregistrez votre poids sur la page Aujourd'hui.",
    "latest_weight": "Dernier poids", "change_period": "Changement (période)", "entries_logged": "Entrées",
    "weekly_adherence_header": "📅 Assiduité Hebdomadaire", "meals_hit_avg": "Repas atteints (moy.)",
    "workouts_completed": "Entraînements terminés", "protein_target_days": "Jours objectif protéines",
    "steps_target_days": "Jours objectif pas", "measurements_header": "📏 Mesures Corporelles",
    "waist_label": "Taille (cm)", "chest_label": "Poitrine (cm)", "hips_label": "Hanches (cm)",
    "arms_label": "Bras (cm)", "thighs_label": "Cuisses (cm)", "save_measurement": "Enregistrer la mesure",
    "photos_header": "📸 Photos de Progrès", "upload_photo": "Téléverser une photo",
    "caption_label": "Légende (facultatif)", "save_photo": "Enregistrer la photo",
    "settings_header": "⚙️ Paramètres", "edit_targets_header": "Modifier vos objectifs",
    "save_targets": "Enregistrer les objectifs", "language_label": "Langue",
    "streak_header": "🔥 Série & Badges", "current_streak_label": "Série actuelle",
    "longest_streak_label": "Plus longue série", "days_logged_label": "Jours enregistrés",
    "badges_label": "Badges obtenus", "compare_header": "Comparer deux photos", "delete_label": "Supprimer",
    "profile_header": "🙋 Votre Profil", "goal_label": "Objectif", "height_label": "Taille (cm)",
    "current_weight_label": "Poids Actuel (kg)", "age_label": "Âge",
    "sex_label": "Sexe biologique (pour plus de précision)", "country_label": "Pays",
    "activity_label": "Niveau d'activité", "recalc_button": "Enregistrer et recalculer mes objectifs",
    "targets_updated_msg": "Objectifs mis à jour selon votre profil !",
    "profile_disclaimer": "Ce sont des estimations issues de formules standards, pas un avis médical. Ajustez librement dans Paramètres.",
    "bottle_0": "Commençons 💪", "bottle_25": "Bien, on chauffe ! 🔥",
    "bottle_50": "À mi-chemin, continuez ! 💦", "bottle_75": "Presque plein, ne lâchez rien ! ⚡",
    "bottle_100": "🎉 Bouteille pleine ! Entraînement terminé !",
}

BASE_IT = {
    "nav_today": "Oggi", "nav_weight_log": "Registro Peso", "nav_weekly_dashboard": "Cruscotto Settimanale",
    "nav_measurements": "Misure", "nav_photos": "Foto di Progresso", "nav_settings": "Impostazioni",
    "nav_profile": "Profilo", "gym_label": "Palestra", "meals_header": "🍽️ Pasti",
    "workout_header": "🏋️ Allenamento", "daily_numbers_header": "📊 Numeri Giornalieri",
    "weight_section_header": "⚖️ Peso", "notes_label": "Note (opzionale)",
    "save_button": "💾 Salva registro di oggi", "saved_msg": "Salvato!", "prev_day": "⬅ Giorno precedente",
    "next_day": "Giorno successivo ➡", "done_label": "Fatto", "what_did_you_have": "Cosa hai mangiato?",
    "calories_label": "Calorie", "protein_label": "Proteine (g)", "use_meal_total": "Usa totale pasti",
    "water_label": "Acqua (L)", "steps_label": "Passi", "weight_label": "Peso (kg)",
    "weight_trend_header": "⚖️ Andamento del Peso",
    "no_weight_entries": "Nessuna voce di peso ancora. Registra il tuo peso nella pagina Oggi.",
    "latest_weight": "Peso più recente", "change_period": "Variazione (periodo)", "entries_logged": "Voci registrate",
    "weekly_adherence_header": "📅 Aderenza Settimanale", "meals_hit_avg": "Pasti raggiunti (media)",
    "workouts_completed": "Allenamenti completati", "protein_target_days": "Giorni obiettivo proteine",
    "steps_target_days": "Giorni obiettivo passi", "measurements_header": "📏 Misure Corporee",
    "waist_label": "Vita (cm)", "chest_label": "Petto (cm)", "hips_label": "Fianchi (cm)",
    "arms_label": "Braccia (cm)", "thighs_label": "Cosce (cm)", "save_measurement": "Salva misura",
    "photos_header": "📸 Foto di Progresso", "upload_photo": "Carica una foto",
    "caption_label": "Didascalia (opzionale)", "save_photo": "Salva foto",
    "settings_header": "⚙️ Impostazioni", "edit_targets_header": "Modifica i tuoi obiettivi",
    "save_targets": "Salva obiettivi", "language_label": "Lingua",
    "streak_header": "🔥 Serie & Distintivi", "current_streak_label": "Serie attuale",
    "longest_streak_label": "Serie più lunga", "days_logged_label": "Giorni registrati",
    "badges_label": "Distintivi ottenuti", "compare_header": "Confronta due foto", "delete_label": "Elimina",
    "profile_header": "🙋 Il Tuo Profilo", "goal_label": "Obiettivo", "height_label": "Altezza (cm)",
    "current_weight_label": "Peso Attuale (kg)", "age_label": "Età",
    "sex_label": "Sesso biologico (per maggiore precisione)", "country_label": "Paese",
    "activity_label": "Livello di attività", "recalc_button": "Salva e ricalcola i miei obiettivi",
    "targets_updated_msg": "Obiettivi aggiornati in base al tuo profilo!",
    "profile_disclaimer": "Queste sono stime da formule standard, non consigli medici. Modifica liberamente in Impostazioni.",
    "bottle_0": "Iniziamo 💪", "bottle_25": "Bene, ti stai scaldando! 🔥",
    "bottle_50": "A metà strada, continua così! 💦", "bottle_75": "Quasi pieno, non fermarti ora! ⚡",
    "bottle_100": "🎉 Bottiglia piena! Allenamento completato!",
}

BASE_NL = {
    "nav_today": "Vandaag", "nav_weight_log": "Gewichtslog", "nav_weekly_dashboard": "Weekoverzicht",
    "nav_measurements": "Metingen", "nav_photos": "Voortgangsfoto's", "nav_settings": "Instellingen",
    "nav_profile": "Profiel", "gym_label": "Sportschool", "meals_header": "🍽️ Maaltijden",
    "workout_header": "🏋️ Training", "daily_numbers_header": "📊 Dagelijkse Cijfers",
    "weight_section_header": "⚖️ Gewicht", "notes_label": "Notities (optioneel)",
    "save_button": "💾 Log van vandaag opslaan", "saved_msg": "Opgeslagen!", "prev_day": "⬅ Vorige dag",
    "next_day": "Volgende dag ➡", "done_label": "Klaar", "what_did_you_have": "Wat heb je gegeten?",
    "calories_label": "Calorieën", "protein_label": "Eiwit (g)", "use_meal_total": "Gebruik maaltijdtotaal",
    "water_label": "Water (L)", "steps_label": "Stappen", "weight_label": "Gewicht (kg)",
    "weight_trend_header": "⚖️ Gewichtstrend",
    "no_weight_entries": "Nog geen gewicht geregistreerd. Registreer je gewicht op de pagina Vandaag.",
    "latest_weight": "Laatste gewicht", "change_period": "Verandering (periode)", "entries_logged": "Registraties",
    "weekly_adherence_header": "📅 Wekelijkse Naleving", "meals_hit_avg": "Maaltijden behaald (gem.)",
    "workouts_completed": "Trainingen voltooid", "protein_target_days": "Eiwitdoel-dagen",
    "steps_target_days": "Stappendoel-dagen", "measurements_header": "📏 Lichaamsmaten",
    "waist_label": "Taille (cm)", "chest_label": "Borst (cm)", "hips_label": "Heupen (cm)",
    "arms_label": "Armen (cm)", "thighs_label": "Dijen (cm)", "save_measurement": "Meting opslaan",
    "photos_header": "📸 Voortgangsfoto's", "upload_photo": "Foto uploaden",
    "caption_label": "Bijschrift (optioneel)", "save_photo": "Foto opslaan",
    "settings_header": "⚙️ Instellingen", "edit_targets_header": "Bewerk je doelen",
    "save_targets": "Doelen opslaan", "language_label": "Taal",
    "streak_header": "🔥 Reeks & Badges", "current_streak_label": "Huidige reeks",
    "longest_streak_label": "Langste reeks", "days_logged_label": "Dagen geregistreerd",
    "badges_label": "Verdiende badges", "compare_header": "Vergelijk twee foto's", "delete_label": "Verwijderen",
    "profile_header": "🙋 Jouw Profiel", "goal_label": "Doel", "height_label": "Lengte (cm)",
    "current_weight_label": "Huidig Gewicht (kg)", "age_label": "Leeftijd",
    "sex_label": "Biologisch geslacht (voor nauwkeurigheid)", "country_label": "Land",
    "activity_label": "Activiteitsniveau", "recalc_button": "Opslaan & doelen herberekenen",
    "targets_updated_msg": "Doelen bijgewerkt op basis van je profiel!",
    "profile_disclaimer": "Dit zijn schattingen op basis van standaardformules, geen medisch advies. Pas gerust aan bij Instellingen.",
    "bottle_0": "Laten we beginnen 💪", "bottle_25": "Mooi, aan het opwarmen! 🔥",
    "bottle_50": "Halverwege, ga zo door! 💦", "bottle_75": "Bijna vol, hou vol! ⚡",
    "bottle_100": "🎉 Fles vol! Training voltooid!",
}

BASE_UR = {
    "nav_today": "آج", "nav_weight_log": "وزن کا ریکارڈ", "nav_weekly_dashboard": "ہفتہ وار ڈیش بورڈ",
    "nav_measurements": "پیمائش", "nav_photos": "پیش رفت کی تصاویر", "nav_settings": "ترتیبات",
    "nav_profile": "پروفائل", "gym_label": "جم", "meals_header": "🍽️ کھانے", "workout_header": "🏋️ ورزش",
    "daily_numbers_header": "📊 روزانہ کے اعداد", "weight_section_header": "⚖️ وزن",
    "notes_label": "نوٹس (اختیاری)", "save_button": "💾 آج کا ریکارڈ محفوظ کریں", "saved_msg": "محفوظ ہو گیا!",
    "prev_day": "⬅ پچھلا دن", "next_day": "اگلا دن ➡", "done_label": "مکمل",
    "what_did_you_have": "آپ نے کیا کھایا؟", "calories_label": "کیلوریز", "protein_label": "پروٹین (g)",
    "use_meal_total": "کھانے کا مجموعہ استعمال کریں", "water_label": "پانی (L)", "steps_label": "قدم",
    "weight_label": "وزن (kg)", "weight_trend_header": "⚖️ وزن کا رجحان",
    "no_weight_entries": "ابھی تک کوئی وزن درج نہیں ہوا۔ آج کے صفحے پر اپنا وزن درج کریں۔",
    "latest_weight": "تازہ ترین وزن", "change_period": "تبدیلی (مدت)", "entries_logged": "درج اندراجات",
    "weekly_adherence_header": "📅 ہفتہ وار پابندی", "meals_hit_avg": "حاصل شدہ کھانے (اوسط)",
    "workouts_completed": "مکمل ورزشیں", "protein_target_days": "پروٹین ہدف کے دن",
    "steps_target_days": "قدم ہدف کے دن", "measurements_header": "📏 جسمانی پیمائش",
    "waist_label": "کمر (cm)", "chest_label": "سینہ (cm)", "hips_label": "کولہے (cm)",
    "arms_label": "بازو (cm)", "thighs_label": "ران (cm)", "save_measurement": "پیمائش محفوظ کریں",
    "photos_header": "📸 پیش رفت کی تصاویر", "upload_photo": "تصویر اپ لوڈ کریں",
    "caption_label": "کیپشن (اختیاری)", "save_photo": "تصویر محفوظ کریں",
    "settings_header": "⚙️ ترتیبات", "edit_targets_header": "اپنے اہداف میں ترمیم کریں",
    "save_targets": "اہداف محفوظ کریں", "language_label": "زبان",
    "streak_header": "🔥 تسلسل اور بیجز", "current_streak_label": "موجودہ تسلسل",
    "longest_streak_label": "طویل ترین تسلسل", "days_logged_label": "درج دن",
    "badges_label": "حاصل شدہ بیجز", "compare_header": "دو تصاویر کا موازنہ کریں", "delete_label": "حذف کریں",
    "profile_header": "🙋 آپ کا پروفائل", "goal_label": "ہدف", "height_label": "قد (cm)",
    "current_weight_label": "موجودہ وزن (kg)", "age_label": "عمر",
    "sex_label": "حیاتیاتی جنس (درست حساب کے لیے)", "country_label": "ملک",
    "activity_label": "سرگرمی کی سطح", "recalc_button": "محفوظ کریں اور اہداف دوبارہ شمار کریں",
    "targets_updated_msg": "آپ کے پروفائل کی بنیاد پر اہداف اپ ڈیٹ ہو گئے!",
    "profile_disclaimer": "یہ معیاری فارمولوں سے تخمینے ہیں، طبی مشورہ نہیں۔ ترتیبات میں آزادانہ طور پر ایڈجسٹ کریں۔",
    "bottle_0": "چلیں شروع کرتے ہیں 💪", "bottle_25": "بہت خوب، گرمائش! 🔥",
    "bottle_50": "آدھا سفر مکمل، جاری رکھیں! 💦", "bottle_75": "تقریباً بھر گیا — اب مت رکیں! ⚡",
    "bottle_100": "🎉 بوتل بھر گئی! ورزش مکمل!",
}

BASE_AR = {
    "nav_today": "اليوم", "nav_weight_log": "سجل الوزن", "nav_weekly_dashboard": "لوحة أسبوعية",
    "nav_measurements": "القياسات", "nav_photos": "صور التقدم", "nav_settings": "الإعدادات",
    "nav_profile": "الملف الشخصي", "gym_label": "النادي الرياضي", "meals_header": "🍽️ الوجبات",
    "workout_header": "🏋️ التمرين", "daily_numbers_header": "📊 الأرقام اليومية", "weight_section_header": "⚖️ الوزن",
    "notes_label": "ملاحظات (اختياري)", "save_button": "💾 حفظ سجل اليوم", "saved_msg": "تم الحفظ!",
    "prev_day": "⬅ اليوم السابق", "next_day": "اليوم التالي ➡", "done_label": "منجز",
    "what_did_you_have": "ماذا أكلت؟", "calories_label": "السعرات الحرارية", "protein_label": "البروتين (g)",
    "use_meal_total": "استخدام مجموع الوجبات", "water_label": "الماء (L)", "steps_label": "الخطوات",
    "weight_label": "الوزن (kg)", "weight_trend_header": "⚖️ اتجاه الوزن",
    "no_weight_entries": "لا توجد إدخالات وزن بعد. سجل وزنك في صفحة اليوم.",
    "latest_weight": "آخر وزن", "change_period": "التغيير (الفترة)", "entries_logged": "الإدخالات المسجلة",
    "weekly_adherence_header": "📅 الالتزام الأسبوعي", "meals_hit_avg": "الوجبات المحققة (متوسط)",
    "workouts_completed": "التمارين المكتملة", "protein_target_days": "أيام هدف البروتين",
    "steps_target_days": "أيام هدف الخطوات", "measurements_header": "📏 قياسات الجسم",
    "waist_label": "الخصر (cm)", "chest_label": "الصدر (cm)", "hips_label": "الوركين (cm)",
    "arms_label": "الذراعين (cm)", "thighs_label": "الفخذين (cm)", "save_measurement": "حفظ القياس",
    "photos_header": "📸 صور التقدم", "upload_photo": "رفع صورة", "caption_label": "تعليق (اختياري)",
    "save_photo": "حفظ الصورة", "settings_header": "⚙️ الإعدادات", "edit_targets_header": "تعديل أهدافك",
    "save_targets": "حفظ الأهداف", "language_label": "اللغة", "streak_header": "🔥 التتابع والشارات",
    "current_streak_label": "التتابع الحالي", "longest_streak_label": "أطول تتابع",
    "days_logged_label": "الأيام المسجلة", "badges_label": "الشارات المكتسبة",
    "compare_header": "قارن بين صورتين", "delete_label": "حذف",
    "profile_header": "🙋 ملفك الشخصي", "goal_label": "الهدف", "height_label": "الطول (cm)",
    "current_weight_label": "الوزن الحالي (kg)", "age_label": "العمر",
    "sex_label": "الجنس البيولوجي (لدقة الحساب)", "country_label": "البلد",
    "activity_label": "مستوى النشاط", "recalc_button": "حفظ وإعادة حساب أهدافي",
    "targets_updated_msg": "تم تحديث الأهداف بناءً على ملفك الشخصي!",
    "profile_disclaimer": "هذه تقديرات من معادلات قياسية، وليست نصيحة طبية. عدّل بحرية في الإعدادات.",
    "bottle_0": "لنبدأ 💪", "bottle_25": "جيد، بدأ الإحماء! 🔥",
    "bottle_50": "في منتصف الطريق، واصل! 💦", "bottle_75": "أوشكت على الامتلاء — لا تتوقف الآن! ⚡",
    "bottle_100": "🎉 الزجاجة ممتلئة! التمرين اكتمل!",
}

BASE_DE = {
    "nav_today": "Heute", "nav_weight_log": "Gewichtsprotokoll", "nav_weekly_dashboard": "Wochenübersicht",
    "nav_measurements": "Maße", "nav_photos": "Fortschrittsfotos", "nav_settings": "Einstellungen",
    "nav_profile": "Profil", "gym_label": "Fitnessstudio", "meals_header": "🍽️ Mahlzeiten",
    "workout_header": "🏋️ Training", "daily_numbers_header": "📊 Tageswerte", "weight_section_header": "⚖️ Gewicht",
    "notes_label": "Notizen (optional)", "save_button": "💾 Heutigen Eintrag speichern", "saved_msg": "Gespeichert!",
    "prev_day": "⬅ Vorheriger Tag", "next_day": "Nächster Tag ➡", "done_label": "Erledigt",
    "what_did_you_have": "Was hast du gegessen?", "calories_label": "Kalorien", "protein_label": "Eiweiß (g)",
    "use_meal_total": "Mahlzeit-Summe verwenden", "water_label": "Wasser (L)", "steps_label": "Schritte",
    "weight_label": "Gewicht (kg)", "weight_trend_header": "⚖️ Gewichtsverlauf",
    "no_weight_entries": "Noch keine Gewichtseinträge. Trage dein Gewicht auf der Heute-Seite ein.",
    "latest_weight": "Letztes Gewicht", "change_period": "Veränderung (Zeitraum)", "entries_logged": "Einträge",
    "weekly_adherence_header": "📅 Wöchentliche Einhaltung", "meals_hit_avg": "Mahlzeiten erreicht (Ø)",
    "workouts_completed": "Trainings abgeschlossen", "protein_target_days": "Eiweißziel-Tage",
    "steps_target_days": "Schrittziel-Tage", "measurements_header": "📏 Körpermaße",
    "waist_label": "Taille (cm)", "chest_label": "Brust (cm)", "hips_label": "Hüfte (cm)",
    "arms_label": "Arme (cm)", "thighs_label": "Oberschenkel (cm)", "save_measurement": "Maß speichern",
    "photos_header": "📸 Fortschrittsfotos", "upload_photo": "Foto hochladen",
    "caption_label": "Bildunterschrift (optional)", "save_photo": "Foto speichern",
    "settings_header": "⚙️ Einstellungen", "edit_targets_header": "Ziele bearbeiten",
    "save_targets": "Ziele speichern", "language_label": "Sprache", "streak_header": "🔥 Serie & Abzeichen",
    "current_streak_label": "Aktuelle Serie", "longest_streak_label": "Längste Serie",
    "days_logged_label": "Erfasste Tage", "badges_label": "Verdiente Abzeichen",
    "compare_header": "Zwei Fotos vergleichen", "delete_label": "Löschen",
    "profile_header": "🙋 Dein Profil", "goal_label": "Ziel", "height_label": "Größe (cm)",
    "current_weight_label": "Aktuelles Gewicht (kg)", "age_label": "Alter",
    "sex_label": "Biologisches Geschlecht (für Genauigkeit)", "country_label": "Land",
    "activity_label": "Aktivitätslevel", "recalc_button": "Speichern & Ziele neu berechnen",
    "targets_updated_msg": "Ziele basierend auf deinem Profil aktualisiert!",
    "profile_disclaimer": "Dies sind Schätzungen aus Standardformeln, kein medizinischer Rat. Passe sie frei in den Einstellungen an.",
    "bottle_0": "Los geht's 💪", "bottle_25": "Gut, du wärmst dich auf! 🔥",
    "bottle_50": "Auf halbem Weg, weiter so! 💦", "bottle_75": "Fast voll — jetzt nicht aufhören! ⚡",
    "bottle_100": "🎉 Flasche voll! Training abgeschlossen!",
}

TRANSLATIONS = {
    "English": BASE_EN, "Afrikaans": BASE_AF, "Spanish": BASE_ES, "French": BASE_FR,
    "Italian": BASE_IT, "Dutch": BASE_NL, "Urdu": BASE_UR, "Arabic": BASE_AR, "German": BASE_DE,
}
# app_title stays the same brand name across all languages
for lang in TRANSLATIONS:
    TRANSLATIONS[lang]["app_title"] = "⚡ Momentum"

def t(key):
    lang = st.session_state.get("language", "English")
    return TRANSLATIONS.get(lang, {}).get(key, TRANSLATIONS["English"].get(key, key))
# ---------------------------------------------------------------
# DATABASE (Postgres via Supabase)
# ---------------------------------------------------------------
@st.cache_resource
def get_conn():
    conn = psycopg2.connect(st.secrets["connections"]["postgres"]["url"])
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS daily_log (
        log_date TEXT PRIMARY KEY, protein_g REAL DEFAULT 0, water_l REAL DEFAULT 0,
        steps INTEGER DEFAULT 0, weight_kg REAL, notes TEXT)""")
    cur.execute("""CREATE TABLE IF NOT EXISTS meal_checks (
        log_date TEXT, meal TEXT, done INTEGER DEFAULT 0, PRIMARY KEY (log_date, meal))""")
    cur.execute("""CREATE TABLE IF NOT EXISTS meal_details (
        log_date TEXT, meal TEXT, note TEXT DEFAULT '', calories REAL DEFAULT 0,
        protein_g REAL DEFAULT 0, PRIMARY KEY (log_date, meal))""")
    cur.execute("""CREATE TABLE IF NOT EXISTS exercise_checks (
        log_date TEXT, exercise TEXT, done INTEGER DEFAULT 0, PRIMARY KEY (log_date, exercise))""")
    cur.execute("""CREATE TABLE IF NOT EXISTS app_settings (key TEXT PRIMARY KEY, value TEXT)""")
    cur.execute("""CREATE TABLE IF NOT EXISTS body_measurements (
        log_date TEXT PRIMARY KEY, waist_cm REAL, chest_cm REAL, hips_cm REAL,
        arms_cm REAL, thighs_cm REAL)""")
    cur.execute("""CREATE TABLE IF NOT EXISTS progress_photos (
        id SERIAL PRIMARY KEY, log_date TEXT, caption TEXT, photo_data BYTEA,
        uploaded_at TIMESTAMP DEFAULT now())""")
    cur.execute("""CREATE TABLE IF NOT EXISTS profile (
        id INTEGER PRIMARY KEY DEFAULT 1, goal TEXT, height_cm REAL, weight_kg REAL,
        age INTEGER, sex TEXT, country TEXT, activity_level TEXT)""")
    # NEW: per-day exercise substitutions (e.g. did Pull Ups instead of Deadlift)
    cur.execute("""CREATE TABLE IF NOT EXISTS exercise_swaps (
        log_date TEXT, original_exercise TEXT, replacement_exercise TEXT,
        PRIMARY KEY (log_date, original_exercise))""")
    # NEW: actual training data — sets, reps, load
    cur.execute("""CREATE TABLE IF NOT EXISTS exercise_sets (
        log_date TEXT, exercise TEXT, set_number INTEGER,
        reps INTEGER DEFAULT 0, weight_kg REAL DEFAULT 0,
        PRIMARY KEY (log_date, exercise, set_number))""")
    # Cardio/walks are time-and-distance, not weight-and-reps. Added as separate
    # columns so existing rows are untouched.
    for _col, _type in (("duration_min", "REAL"), ("distance_km", "REAL")):
        try:
            cur.execute(f"ALTER TABLE exercise_sets ADD COLUMN IF NOT EXISTS {_col} {_type} DEFAULT 0")
        except Exception:
            pass
    # How each movement is logged, and whether the weight entered is per-hand.
    cur.execute("""CREATE TABLE IF NOT EXISTS exercise_prefs (
        exercise TEXT PRIMARY KEY, log_type TEXT, per_side INTEGER DEFAULT 0)""")
    # NEW: free-text note per exercise per day ("left shoulder tight")
    cur.execute("""CREATE TABLE IF NOT EXISTS exercise_notes (
        log_date TEXT, exercise TEXT, note TEXT DEFAULT '',
        PRIMARY KEY (log_date, exercise))""")
    # NEW: the workout split itself, so it's editable in-app instead of in code
    cur.execute("""CREATE TABLE IF NOT EXISTS workout_plan (
        weekday TEXT, position INTEGER, exercise TEXT, day_label TEXT,
        PRIMARY KEY (weekday, position))""")
    # NEW: deload / rest weeks (keyed by the Monday of that week)
    cur.execute("""CREATE TABLE IF NOT EXISTS rest_weeks (week_start TEXT PRIMARY KEY)""")
    # Indexes on the columns every query filters by. Without these, Postgres
    # scans the whole table for each lookup — fine at 50 rows, not at 50,000.
    for _idx, _spec in (
        ("idx_sets_exercise_date", "exercise_sets (exercise, log_date)"),
        ("idx_sets_date", "exercise_sets (log_date)"),
        ("idx_checks_date", "exercise_checks (log_date)"),
        ("idx_meal_checks_date", "meal_checks (log_date)"),
        ("idx_meal_details_date", "meal_details (log_date)"),
    ):
        try:
            cur.execute(f"CREATE INDEX IF NOT EXISTS {_idx} ON {_spec}")
        except Exception:
            pass
    cur.close()
    return conn

def get_live_conn():
    """Returns the cached connection, transparently reconnecting if Supabase closed it (e.g. idle timeout)."""
    c = get_conn()
    if c.closed:
        get_conn.clear()
        c = get_conn()
    return c

conn = get_live_conn()

def run(query, params=()):
    c = get_live_conn()
    try:
        cur = c.cursor()
        cur.execute(query, params)
        cur.close()
    except (psycopg2.InterfaceError, psycopg2.OperationalError):
        get_conn.clear()
        c = get_live_conn()
        cur = c.cursor()
        cur.execute(query, params)
        cur.close()

def fetch(query, params=()):
    c = get_live_conn()
    try:
        cur = c.cursor()
        cur.execute(query, params)
        cols = [d[0] for d in cur.description]
        rows = cur.fetchall()
        cur.close()
        return cols, rows
    except (psycopg2.InterfaceError, psycopg2.OperationalError):
        get_conn.clear()
        c = get_live_conn()
        cur = c.cursor()
        cur.execute(query, params)
        cols = [d[0] for d in cur.description]
        rows = cur.fetchall()
        cur.close()
        return cols, rows

# ---------------------------------------------------------------
# CURRENT USER
# ---------------------------------------------------------------
# Every query below filters on this. It's read from the module global that the
# UI section sets once per script run, rather than being threaded through fifty
# function signatures — Python resolves globals at call time, so by the time any
# of these run, authentication has already resolved.
#
# The fallback matters: a few functions can be reached before auth resolves (and
# in single-user mode CURRENT_USER_ID is never set to anything else), so falling
# back to SINGLE_USER_ID keeps the app working rather than throwing.

def current_user_id():
    """The user_id every query filters on."""
    uid = globals().get("CURRENT_USER_ID")
    if uid is None or uid == "DENIED":
        return SINGLE_USER_ID
    return uid


# ---------------------------------------------------------------
# SETTINGS / TARGETS
# ---------------------------------------------------------------
def get_setting(key, default=None):
    cols, rows = fetch("SELECT value FROM app_settings WHERE user_id=%s AND key=%s",
                       (current_user_id(), key))
    return rows[0][0] if rows else default

def set_setting(key, value):
    run("""INSERT INTO app_settings (user_id, key, value) VALUES (%s, %s, %s)
           ON CONFLICT (user_id, key) DO UPDATE SET value=excluded.value""",
        (current_user_id(), key, str(value)))

def load_targets():
    targets = DEFAULT_TARGETS.copy()
    for key in DEFAULT_TARGETS:
        val = get_setting(f"target_{key}")
        if val is not None:
            targets[key] = float(val)
    return targets

def save_targets(new_targets):
    for key, val in new_targets.items():
        set_setting(f"target_{key}", val)

# ---------------------------------------------------------------
# PROFILE
# ---------------------------------------------------------------
def get_profile():
    cols, rows = fetch("SELECT * FROM profile WHERE user_id=%s", (current_user_id(),))
    if not rows:
        return {"goal": GOALS[2], "height_cm": None, "weight_kg": None, "age": None,
                "sex": SEX_OPTIONS[2], "country": "", "activity_level": ACTIVITY_LEVELS[2]}
    return dict(zip(cols, rows[0]))

def save_profile(goal, height_cm, weight_kg, age, sex, country, activity_level):
    run("""
        INSERT INTO profile (user_id, goal, height_cm, weight_kg, age, sex, country, activity_level)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (user_id) DO UPDATE SET
            goal=excluded.goal, height_cm=excluded.height_cm, weight_kg=excluded.weight_kg,
            age=excluded.age, sex=excluded.sex, country=excluded.country,
            activity_level=excluded.activity_level
    """, (current_user_id(), goal, height_cm, weight_kg, age, sex, country, activity_level))

def compute_targets_from_profile(profile):
    """Standard BMR/TDEE-based estimate. Not medical advice — a sensible starting point."""
    h, w, age = profile["height_cm"], profile["weight_kg"], profile["age"]
    sex, activity, goal = profile["sex"], profile["activity_level"], profile["goal"]
    if not h or not w or not age:
        return None

    if sex == "Male":
        bmr = 10 * w + 6.25 * h - 5 * age + 5
    elif sex == "Female":
        bmr = 10 * w + 6.25 * h - 5 * age - 161
    else:
        bmr = (10 * w + 6.25 * h - 5 * age + 5 + 10 * w + 6.25 * h - 5 * age - 161) / 2

    tdee = bmr * ACTIVITY_MULTIPLIERS.get(activity, 1.4)

    if goal == "Lose weight":
        cal_min, cal_max = tdee - 600, tdee - 300
        prot_min, prot_max = w * 1.8, w * 2.2
    elif goal == "Gain muscle":
        cal_min, cal_max = tdee + 200, tdee + 450
        prot_min, prot_max = w * 1.8, w * 2.2
    elif goal == "Body recomposition":
        cal_min, cal_max = tdee - 200, tdee + 100
        prot_min, prot_max = w * 2.0, w * 2.4
    else:  # Maintain
        cal_min, cal_max = tdee - 100, tdee + 100
        prot_min, prot_max = w * 1.6, w * 2.0

    water_min, water_max = w * 0.035, w * 0.045
    steps_min = 8000 if activity in ("Sedentary (little exercise)", "Light (1-3 days/week)") else 10000
    steps_max = steps_min + 5000

    return {
        "calories_min": round(cal_min), "calories_max": round(cal_max),
        "protein_min": round(prot_min), "protein_max": round(prot_max),
        "water_min": round(water_min, 1), "water_max": round(water_max, 1),
        "steps_min": steps_min, "steps_max": steps_max,
    }

# ---------------------------------------------------------------
# DAILY LOG / MEALS / EXERCISES
# ---------------------------------------------------------------
def get_daily_row(log_date):
    uid = current_user_id()
    cols, rows = fetch("SELECT * FROM daily_log WHERE user_id=%s AND log_date=%s",
                       (uid, log_date))
    if not rows:
        run("INSERT INTO daily_log (user_id, log_date) VALUES (%s, %s) "
            "ON CONFLICT (user_id, log_date) DO NOTHING", (uid, log_date))
        return {"log_date": log_date, "protein_g": 0, "water_l": 0, "steps": 0,
                "weight_kg": None, "notes": ""}
    return dict(zip(cols, rows[0]))

def save_daily_row(log_date, protein_g, water_l, steps, weight_kg, notes):
    run("""INSERT INTO daily_log (user_id, log_date, protein_g, water_l, steps, weight_kg, notes)
           VALUES (%s, %s, %s, %s, %s, %s, %s)
           ON CONFLICT (user_id, log_date) DO UPDATE SET
           protein_g=excluded.protein_g, water_l=excluded.water_l, steps=excluded.steps,
           weight_kg=excluded.weight_kg, notes=excluded.notes""",
        (current_user_id(), log_date, protein_g, water_l, steps, weight_kg, notes))

def get_meal_checks(log_date):
    cols, rows = fetch("SELECT meal, done FROM meal_checks WHERE user_id=%s AND log_date=%s",
                       (current_user_id(), log_date))
    existing = dict(rows)
    return {m: bool(existing.get(m, 0)) for m in MEALS}

def set_meal_check(log_date, meal, done):
    run("""INSERT INTO meal_checks (user_id, log_date, meal, done) VALUES (%s, %s, %s, %s)
           ON CONFLICT (user_id, log_date, meal) DO UPDATE SET done=excluded.done""",
        (current_user_id(), log_date, meal, int(done)))

def get_meal_details(log_date):
    cols, rows = fetch(
        "SELECT meal, note, calories, protein_g FROM meal_details "
        "WHERE user_id=%s AND log_date=%s", (current_user_id(), log_date))
    existing = {r[0]: {"note": r[1] or "", "calories": r[2] or 0, "protein_g": r[3] or 0} for r in rows}
    return {m: existing.get(m, {"note": "", "calories": 0, "protein_g": 0}) for m in MEALS}

def set_meal_detail(log_date, meal, note, calories, protein_g):
    run("""INSERT INTO meal_details (user_id, log_date, meal, note, calories, protein_g)
           VALUES (%s, %s, %s, %s, %s, %s)
           ON CONFLICT (user_id, log_date, meal) DO UPDATE SET
           note=excluded.note, calories=excluded.calories, protein_g=excluded.protein_g""",
        (current_user_id(), log_date, meal, note, calories, protein_g))

def get_exercise_checks(log_date, exercises):
    cols, rows = fetch("SELECT exercise, done FROM exercise_checks "
                       "WHERE user_id=%s AND log_date=%s", (current_user_id(), log_date))
    existing = dict(rows)
    return {e: bool(existing.get(e, 0)) for e in exercises}

def set_exercise_check(log_date, exercise, done):
    run("""INSERT INTO exercise_checks (user_id, log_date, exercise, done)
           VALUES (%s, %s, %s, %s)
           ON CONFLICT (user_id, log_date, exercise) DO UPDATE SET done=excluded.done""",
        (current_user_id(), log_date, exercise, int(done)))

def get_all_logged_exercises(log_date):
    """All exercise_checks rows for a date, including anything added beyond the fixed plan."""
    cols, rows = fetch("SELECT exercise, done FROM exercise_checks "
                       "WHERE user_id=%s AND log_date=%s", (current_user_id(), log_date))
    return dict(rows)

def remove_exercise(log_date, exercise):
    run("DELETE FROM exercise_checks WHERE user_id=%s AND log_date=%s AND exercise=%s",
        (current_user_id(), log_date, exercise))

# ---- exercise swaps (e.g. "did Pull Ups instead of Deadlift today") ----
def get_exercise_swap(log_date, original_exercise):
    """Returns the replacement name for this original exercise on this date, or None."""
    cols, rows = fetch(
        "SELECT replacement_exercise FROM exercise_swaps "
        "WHERE user_id=%s AND log_date=%s AND original_exercise=%s",
        (current_user_id(), log_date, original_exercise))
    return rows[0][0] if rows else None

def set_exercise_swap(log_date, original_exercise, replacement_exercise):
    run("""INSERT INTO exercise_swaps (user_id, log_date, original_exercise, replacement_exercise)
           VALUES (%s, %s, %s, %s)
           ON CONFLICT (user_id, log_date, original_exercise) DO UPDATE SET
           replacement_exercise=excluded.replacement_exercise""",
        (current_user_id(), log_date, original_exercise, replacement_exercise))

def get_all_swaps(log_date):
    """{original_exercise: replacement_exercise} for one date."""
    cols, rows = fetch("""SELECT original_exercise, replacement_exercise
                          FROM exercise_swaps WHERE user_id=%s AND log_date=%s""",
                       (current_user_id(), log_date))
    return dict(rows)


def effective_exercises_for_day(log_date, planned_exercises):
    """Every exercise actually trained on a date, under the name its data is
    stored against.

    Sets are saved under the SWAPPED name, so anything reading set data has to
    resolve swaps first — otherwise a swapped exercise silently contributes
    nothing to volume or calorie totals.
    """
    swaps = get_all_swaps(log_date)
    names = []
    for ex in planned_exercises:
        name = swaps.get(ex, ex)
        if name not in names:
            names.append(name)
    # Extras: anything with a check or logged sets that isn't already covered
    for extra in get_all_logged_exercises(log_date):
        if extra not in names and extra not in planned_exercises:
            names.append(extra)
    cols, rows = fetch("SELECT DISTINCT exercise FROM exercise_sets "
                       "WHERE user_id=%s AND log_date=%s",
                       (current_user_id(), log_date))
    for (name,) in rows:
        if name not in names:
            names.append(name)
    return names


def remove_exercise_swap(log_date, original_exercise):
    run("DELETE FROM exercise_swaps WHERE user_id=%s AND log_date=%s "
        "AND original_exercise=%s", (current_user_id(), log_date, original_exercise))

def get_range_df(start, end):
    q = ("SELECT * FROM daily_log WHERE user_id=%s AND log_date BETWEEN %s AND %s "
         "ORDER BY log_date")
    return pd.read_sql_query(q, get_live_conn(),
                             params=(current_user_id(), start.isoformat(), end.isoformat()))

def get_meal_completion_for_range(start, end):
    q = """SELECT log_date, SUM(done) as done_count FROM meal_checks
           WHERE user_id=%s AND log_date BETWEEN %s AND %s GROUP BY log_date"""
    return pd.read_sql_query(q, get_live_conn(),
                             params=(current_user_id(), start.isoformat(), end.isoformat()))

def get_exercise_completion_for_range(start, end):
    q = """SELECT log_date, COUNT(*) as total, SUM(done) as done_count FROM exercise_checks
           WHERE user_id=%s AND log_date BETWEEN %s AND %s GROUP BY log_date"""
    return pd.read_sql_query(q, get_live_conn(),
                             params=(current_user_id(), start.isoformat(), end.isoformat()))

def get_meal_macro_totals_for_range(start, end):
    q = """SELECT log_date, SUM(calories) as calories_total, SUM(protein_g) as meal_protein_total
           FROM meal_details WHERE user_id=%s AND log_date BETWEEN %s AND %s GROUP BY log_date"""
    return pd.read_sql_query(q, get_live_conn(),
                             params=(current_user_id(), start.isoformat(), end.isoformat()))

# ---------------------------------------------------------------
# BODY MEASUREMENTS
# ---------------------------------------------------------------
def get_measurement_row(log_date):
    cols, rows = fetch("SELECT * FROM body_measurements WHERE user_id=%s AND log_date=%s",
                       (current_user_id(), log_date))
    if not rows:
        return {"waist_cm": None, "chest_cm": None, "hips_cm": None, "arms_cm": None, "thighs_cm": None}
    return dict(zip(cols, rows[0]))

def save_measurement(log_date, waist, chest, hips, arms, thighs):
    run("""INSERT INTO body_measurements
             (user_id, log_date, waist_cm, chest_cm, hips_cm, arms_cm, thighs_cm)
           VALUES (%s, %s, %s, %s, %s, %s, %s)
           ON CONFLICT (user_id, log_date) DO UPDATE SET
           waist_cm=excluded.waist_cm, chest_cm=excluded.chest_cm, hips_cm=excluded.hips_cm,
           arms_cm=excluded.arms_cm, thighs_cm=excluded.thighs_cm""",
        (current_user_id(), log_date, waist, chest, hips, arms, thighs))

def get_all_measurements():
    return pd.read_sql_query(
        "SELECT * FROM body_measurements WHERE user_id=%s ORDER BY log_date",
        get_live_conn(), params=(current_user_id(),))

# ---------------------------------------------------------------
# PROGRESS PHOTOS
# ---------------------------------------------------------------
def save_photo(log_date, caption, photo_bytes):
    run("INSERT INTO progress_photos (user_id, log_date, caption, photo_data) "
        "VALUES (%s, %s, %s, %s)",
        (current_user_id(), log_date, caption, psycopg2.Binary(photo_bytes)))

def get_all_photos():
    cols, rows = fetch("SELECT id, log_date, caption, photo_data FROM progress_photos "
                       "WHERE user_id=%s ORDER BY log_date DESC", (current_user_id(),))
    return [dict(zip(cols, r)) for r in rows]

def delete_photo(photo_id):
    # user_id in the WHERE clause as well as the id: without it, a stale photo id
    # from another session could delete someone else's row.
    run("DELETE FROM progress_photos WHERE user_id=%s AND id=%s",
        (current_user_id(), photo_id))

# ---------------------------------------------------------------
# WORKOUT PLAN (editable in-app, seeded from GYM_SPLIT defaults)
# ---------------------------------------------------------------
def repair_set_numbering():
    """One-off cleanup for data written by an earlier buggy version.

    That version could create a first set numbered 2 (and then keep overwriting
    it), leaving gaps like [2] or [2,3]. This renumbers every affected exercise
    to a clean 1..n, preserving order and values. Safe to run repeatedly.
    """
    if get_setting("set_numbering_repaired") == "1":
        return
    uid = current_user_id()
    cols, rows = fetch("""SELECT log_date, exercise FROM exercise_sets
                          WHERE user_id=%s
                          GROUP BY log_date, exercise
                          HAVING MIN(set_number) > 1
                              OR MAX(set_number) <> COUNT(*)""", (uid,))
    for log_date, exercise in rows:
        _, ordered = fetch("""SELECT set_number FROM exercise_sets
                              WHERE user_id=%s AND log_date=%s AND exercise=%s
                              ORDER BY set_number""", (uid, log_date, exercise))
        for new_num, (old_num,) in enumerate([r for r in ordered], start=1):
            if old_num != new_num:
                run("""UPDATE exercise_sets SET set_number=%s
                       WHERE user_id=%s AND log_date=%s AND exercise=%s AND set_number=%s""",
                    (new_num, uid, log_date, exercise, old_num))
    set_setting("set_numbering_repaired", "1")


def seed_plan_if_empty():
    """First run for THIS user: copy the hardcoded GYM_SPLIT defaults into the DB.

    Per-user, not global — a new account needs its own plan rows, otherwise it
    would land on an empty split.
    """
    uid = current_user_id()
    _, rows = fetch("SELECT COUNT(*) FROM workout_plan WHERE user_id=%s", (uid,))
    if rows and rows[0][0] > 0:
        return
    for weekday, plan in GYM_SPLIT.items():
        for i, ex in enumerate(plan["exercises"]):
            run("""INSERT INTO workout_plan (user_id, weekday, position, exercise, day_label)
                   VALUES (%s, %s, %s, %s, %s)
                   ON CONFLICT (user_id, weekday, position) DO NOTHING""",
                (uid, weekday, i, ex, plan["label"]))


def load_plan():
    """Returns the same shape as GYM_SPLIT, but read from the DB."""
    seed_plan_if_empty()
    cols, rows = fetch(
        "SELECT weekday, position, exercise, day_label FROM workout_plan "
        "WHERE user_id=%s ORDER BY weekday, position", (current_user_id(),))
    plan = {}
    for weekday, position, exercise, day_label in rows:
        entry = plan.setdefault(weekday, {"label": day_label or "", "exercises": []})
        if day_label:
            entry["label"] = day_label
        entry["exercises"].append(exercise)
    # Any weekday missing from the DB falls back to the hardcoded default
    for weekday, default in GYM_SPLIT.items():
        if weekday not in plan or not plan[weekday]["exercises"]:
            plan[weekday] = {"label": default["label"], "exercises": list(default["exercises"])}
    return plan


def save_plan_day(weekday, day_label, exercises):
    uid = current_user_id()
    run("DELETE FROM workout_plan WHERE user_id=%s AND weekday=%s", (uid, weekday))
    for i, ex in enumerate(exercises):
        ex = ex.strip()
        if ex:
            run("""INSERT INTO workout_plan (user_id, weekday, position, exercise, day_label)
                   VALUES (%s, %s, %s, %s, %s)""", (uid, weekday, i, ex, day_label))


def reset_plan_day(weekday):
    default = GYM_SPLIT[weekday]
    save_plan_day(weekday, default["label"], list(default["exercises"]))


# ---------------------------------------------------------------
# SETS / REPS / WEIGHT  (the actual training log)
# ---------------------------------------------------------------
def get_sets(log_date, exercise):
    """Ordered list of sets, including duration/distance for cardio-style entries."""
    cols, rows = fetch("""SELECT set_number, reps, weight_kg,
                                 COALESCE(duration_min,0), COALESCE(distance_km,0)
                          FROM exercise_sets
                          WHERE user_id=%s AND log_date=%s AND exercise=%s
                          ORDER BY set_number""",
                       (current_user_id(), log_date, exercise))
    return [{"set_number": r[0], "reps": r[1] or 0, "weight_kg": r[2] or 0.0,
             "duration_min": r[3] or 0.0, "distance_km": r[4] or 0.0} for r in rows]


def save_set(log_date, exercise, set_number, reps, weight_kg,
             duration_min=0, distance_km=0):
    run("""INSERT INTO exercise_sets
             (user_id, log_date, exercise, set_number, reps, weight_kg, duration_min, distance_km)
           VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
           ON CONFLICT (user_id, log_date, exercise, set_number) DO UPDATE SET
           reps=excluded.reps, weight_kg=excluded.weight_kg,
           duration_min=excluded.duration_min, distance_km=excluded.distance_km""",
        (current_user_id(), log_date, exercise, set_number, int(reps or 0),
         float(weight_kg or 0), float(duration_min or 0), float(distance_km or 0)))


# ---------------------------------------------------------------
# PER-EXERCISE LOGGING PREFERENCES
# ---------------------------------------------------------------
def _all_exercise_prefs():
    """Every preference for the current user in one query, held for this run.

    get_exercise_pref() used to fire a SELECT every time it was called — once
    per set widget, once per row in each volume calculation, once per PR lookup.
    A single Today page could issue a hundred round trips.

    The cache lives in session_state, which is per browser session and therefore
    already per user. The version counter invalidates it the moment a preference
    changes, so the per-hand toggle still takes effect on the next rerun.
    """
    version = st.session_state.get("_prefs_version", 0)
    cached = st.session_state.get("_prefs_cache")
    if cached is not None and cached[0] == version:
        return cached[1]
    cols, rows = fetch("SELECT exercise, log_type, per_side FROM exercise_prefs "
                       "WHERE user_id=%s", (current_user_id(),))
    prefs = {r[0]: {"log_type": r[1], "per_side": bool(r[2])} for r in rows}
    st.session_state["_prefs_cache"] = (version, prefs)
    return prefs


def get_exercise_pref(exercise):
    """How to log this movement, and whether the weight entered is per-hand."""
    entry = _all_exercise_prefs().get(exercise)
    if entry:
        return {"log_type": entry["log_type"] or default_log_type(exercise),
                "per_side": entry["per_side"]}
    return {"log_type": default_log_type(exercise), "per_side": False}


def set_exercise_pref(exercise, log_type, per_side):
    run("""INSERT INTO exercise_prefs (user_id, exercise, log_type, per_side)
           VALUES (%s, %s, %s, %s)
           ON CONFLICT (user_id, exercise) DO UPDATE SET
           log_type=excluded.log_type, per_side=excluded.per_side""",
        (current_user_id(), exercise, log_type, int(bool(per_side))))
    st.session_state["_prefs_version"] = st.session_state.get("_prefs_version", 0) + 1


def effective_load(weight_kg, per_side):
    """Total load actually moved. 22kg dumbbells in each hand = 44kg on the body."""
    return float(weight_kg or 0) * (2 if per_side else 1)


def next_set_number(log_date, exercise):
    """Next free set number, derived from what's actually in the DB.

    Must not be based on how many rows the UI is displaying: when an exercise
    has no saved sets the UI shows an unsaved placeholder, so counting rows
    would skip set 1 and then repeatedly overwrite set 2.
    """
    cols, rows = fetch("""SELECT COALESCE(MAX(set_number), 0) FROM exercise_sets
                          WHERE user_id=%s AND log_date=%s AND exercise=%s""",
                       (current_user_id(), log_date, exercise))
    return (rows[0][0] if rows else 0) + 1


def renumber_sets(log_date, exercise):
    """Close any gaps so sets read 1, 2, 3... after a delete."""
    uid = current_user_id()
    existing = get_sets(log_date, exercise)
    for i, entry in enumerate(existing, start=1):
        if entry["set_number"] != i:
            run("""UPDATE exercise_sets SET set_number=%s
                   WHERE user_id=%s AND log_date=%s AND exercise=%s AND set_number=%s""",
                (i, uid, log_date, exercise, entry["set_number"]))


def delete_last_set(log_date, exercise):
    uid = current_user_id()
    run("""DELETE FROM exercise_sets WHERE user_id=%s AND log_date=%s AND exercise=%s
           AND set_number = (SELECT MAX(set_number) FROM exercise_sets
                             WHERE user_id=%s AND log_date=%s AND exercise=%s)""",
        (uid, log_date, exercise, uid, log_date, exercise))


def get_last_session_peak(exercise, before_date):
    """Heaviest set from the most recent previous session of this exercise."""
    prev_date, prev_sets = get_last_session(exercise, before_date)
    if not prev_sets:
        return None
    per_side = get_exercise_pref(exercise)["per_side"]
    top = None
    for x in prev_sets:
        load = effective_load(x["weight_kg"], per_side)
        reps = int(x["reps"] or 0)
        if reps <= 0 or load <= 0:
            continue
        if top is None or load > top["weight_kg"]:
            top = {"weight_kg": load, "reps": reps,
                   "e1rm": estimated_1rm(load, reps), "log_date": prev_date}
    return top


def get_best_ever(exercise, before_date=None):
    """Heaviest set and best estimated 1RM ever logged for this exercise."""
    uid = current_user_id()
    if before_date:
        cols, rows = fetch("""SELECT reps, weight_kg, log_date FROM exercise_sets
                              WHERE user_id=%s AND exercise=%s
                              AND weight_kg > 0 AND reps > 0
                              AND log_date < %s""", (uid, exercise, before_date))
    else:
        cols, rows = fetch("""SELECT reps, weight_kg, log_date FROM exercise_sets
                              WHERE user_id=%s AND exercise=%s
                              AND weight_kg > 0 AND reps > 0""", (uid, exercise))
    if not rows:
        return None
    per_side = get_exercise_pref(exercise)["per_side"]
    best = None
    top_weight = 0.0
    for reps, weight, log_date in rows:
        load = effective_load(weight, per_side)
        e1rm = estimated_1rm(load, reps)
        top_weight = max(top_weight, load)
        if best is None or e1rm > best["e1rm"]:
            best = {"weight_kg": load, "reps": int(reps), "e1rm": e1rm, "log_date": log_date}
    if best:
        best["top_weight"] = top_weight
    return best


def get_last_session(exercise, before_date):
    """Most recent day BEFORE `before_date` where this exercise had logged sets."""
    cols, rows = fetch("""SELECT MAX(log_date) FROM exercise_sets
                          WHERE user_id=%s AND exercise=%s AND log_date < %s
                          AND (reps > 0 OR weight_kg > 0)""",
                       (current_user_id(), exercise, before_date))
    if not rows or not rows[0][0]:
        return None, []
    prev_date = rows[0][0]
    return prev_date, get_sets(prev_date, exercise)


def get_exercise_note(log_date, exercise):
    cols, rows = fetch("SELECT note FROM exercise_notes "
                       "WHERE user_id=%s AND log_date=%s AND exercise=%s",
                       (current_user_id(), log_date, exercise))
    return rows[0][0] if rows else ""


def set_exercise_note(log_date, exercise, note):
    run("""INSERT INTO exercise_notes (user_id, log_date, exercise, note)
           VALUES (%s, %s, %s, %s)
           ON CONFLICT (user_id, log_date, exercise) DO UPDATE SET note=excluded.note""",
        (current_user_id(), log_date, exercise, note))


def get_all_sets_df():
    return pd.read_sql_query(
        "SELECT log_date, exercise, set_number, reps, weight_kg FROM exercise_sets "
        "WHERE user_id=%s ORDER BY log_date, exercise, set_number",
        get_live_conn(), params=(current_user_id(),))


def get_sets_for_range(start, end):
    return pd.read_sql_query(
        "SELECT log_date, exercise, set_number, reps, weight_kg FROM exercise_sets "
        "WHERE user_id=%s AND log_date BETWEEN %s AND %s ORDER BY log_date",
        get_live_conn(), params=(current_user_id(), start.isoformat(), end.isoformat()))


def get_logged_exercise_names():
    """Distinct exercises that have any set data — for the strength-trend picker."""
    cols, rows = fetch("""SELECT DISTINCT exercise FROM exercise_sets
                          WHERE user_id=%s AND weight_kg > 0 ORDER BY exercise""",
                       (current_user_id(),))
    return [r[0] for r in rows]


# ---------------------------------------------------------------
# REST / DELOAD WEEKS
# ---------------------------------------------------------------
def is_rest_week(week_start):
    cols, rows = fetch("SELECT 1 FROM rest_weeks WHERE user_id=%s AND week_start=%s",
                       (current_user_id(), week_start.isoformat()))
    return bool(rows)


def set_rest_week(week_start, enabled):
    if enabled:
        run("INSERT INTO rest_weeks (user_id, week_start) VALUES (%s, %s) "
            "ON CONFLICT DO NOTHING", (current_user_id(), week_start.isoformat()))
    else:
        run("DELETE FROM rest_weeks WHERE user_id=%s AND week_start=%s",
            (current_user_id(), week_start.isoformat()))
# ---------------------------------------------------------------
# STREAKS & BADGES
# ---------------------------------------------------------------
def get_active_dates_df():
    uid = current_user_id()
    # The user filter has to be inside each subquery as well as on the outer
    # table. Filtering only the outer query would still join in every other
    # user's meal and exercise counts for the same dates.
    cols, rows = fetch("""
        SELECT d.log_date, COALESCE(d.protein_g,0) as protein_g, COALESCE(d.water_l,0) as water_l,
               COALESCE(d.steps,0) as steps, COALESCE(m.done_count,0) as meals_done,
               COALESCE(e.done_count,0) as ex_done
        FROM daily_log d
        LEFT JOIN (SELECT log_date, SUM(done) as done_count FROM meal_checks
                   WHERE user_id=%s GROUP BY log_date) m
            ON d.log_date = m.log_date
        LEFT JOIN (SELECT log_date, SUM(done) as done_count FROM exercise_checks
                   WHERE user_id=%s GROUP BY log_date) e
            ON d.log_date = e.log_date
        WHERE d.user_id=%s
    """, (uid, uid, uid))
    df = pd.DataFrame(rows, columns=cols)
    if df.empty:
        return df
    df["active"] = (df["protein_g"] > 0) | (df["water_l"] > 0) | (df["steps"] > 0) | \
                    (df["meals_done"] > 0) | (df["ex_done"] > 0)
    return df[df["active"]]

def compute_streak_stats():
    df = get_active_dates_df()
    if df.empty:
        return 0, 0, 0
    dates = sorted(pd.to_datetime(df["log_date"]).dt.date.tolist())
    date_set = set(dates)
    longest, current_run = 1, 1
    for i in range(1, len(dates)):
        if (dates[i] - dates[i - 1]).days == 1:
            current_run += 1
        else:
            longest = max(longest, current_run)
            current_run = 1
    longest = max(longest, current_run)
    today = local_today()
    streak, check = 0, today
    while check in date_set:
        streak += 1
        check -= timedelta(days=1)
    if streak == 0:
        check = today - timedelta(days=1)
        while check in date_set:
            streak += 1
            check -= timedelta(days=1)
    return streak, longest, len(dates)

STREAK_BADGES = [(3, "🔥 3-Day Streak"), (7, "🔥 7-Day Streak"), (14, "🔥 14-Day Streak"),
                  (30, "🏆 30-Day Streak"), (100, "💎 100-Day Streak")]
DAYS_BADGES = [(10, "📅 10 Days Logged"), (50, "📅 50 Days Logged"), (100, "📅 100 Days Logged")]

def get_earned_badges(longest_streak, total_days):
    earned = [name for threshold, name in STREAK_BADGES if longest_streak >= threshold]
    earned += [name for threshold, name in DAYS_BADGES if total_days >= threshold]
    return earned

# ---------------------------------------------------------------
# MOMENTUM SCORE & PERFECT DAY
# ---------------------------------------------------------------
def fetch_daily_row_readonly(log_date):
    """Like get_daily_row but never inserts a blank row — safe for scoring past/other dates."""
    cols, rows = fetch("SELECT * FROM daily_log WHERE log_date = %s", (log_date,))
    if not rows:
        return {"protein_g": 0, "water_l": 0, "steps": 0, "weight_kg": None, "notes": ""}
    return dict(zip(cols, rows[0]))

def compute_momentum_score(log_date_str, weekday, targets):
    row = fetch_daily_row_readonly(log_date_str)
    meal_state = get_meal_checks(log_date_str)
    day_plan = PLAN[weekday]
    ex_state = get_exercise_checks(log_date_str, day_plan["exercises"])

    protein = row["protein_g"] or 0
    water = row["water_l"] or 0
    steps = row["steps"] or 0
    meals_done = sum(1 for v in meal_state.values() if v)
    ex_total = len(day_plan["exercises"])
    ex_done = sum(1 for v in ex_state.values() if v)

    protein_score = min(protein / targets["protein_min"], 1.0) * 25 if targets["protein_min"] else 0
    water_score = min(water / targets["water_min"], 1.0) * 20 if targets["water_min"] else 0
    steps_score = min(steps / targets["steps_min"], 1.0) * 20 if targets["steps_min"] else 0
    meal_score = (meals_done / len(MEALS)) * 15 if MEALS else 0
    workout_score = (ex_done / ex_total) * 20 if ex_total else 0

    total = round(protein_score + water_score + steps_score + meal_score + workout_score)
    return max(0, min(100, total))

def get_perfect_day_status(log_date_str, weekday, targets):
    row = fetch_daily_row_readonly(log_date_str)
    meal_state = get_meal_checks(log_date_str)
    day_plan = PLAN[weekday]
    ex_state = get_exercise_checks(log_date_str, day_plan["exercises"])

    protein_hit = (row["protein_g"] or 0) >= targets["protein_min"]
    water_hit = (row["water_l"] or 0) >= targets["water_min"]
    steps_hit = (row["steps"] or 0) >= targets["steps_min"]
    meals_hit = all(meal_state.values()) if meal_state else False
    ex_hit = all(ex_state.values()) if ex_state else False

    return protein_hit and water_hit and steps_hit and meals_hit and ex_hit

def render_momentum_score_badge(score):
    if score >= 80:
        color = "#10b981"
    elif score >= 50:
        color = "#f59e0b"
    else:
        color = "#ef4444"
    html = f"""
    <style>
    @keyframes scoreIn {{ 0% {{ transform: scale(0.8); opacity:0; }} 100% {{ transform: scale(1); opacity:1; }} }}
    .score-box {{ animation: scoreIn 0.4s ease-out; text-align:center; padding:8px 0; }}
    .score-num {{ font-size:2.4rem; font-weight:800; color:{color}; }}
    </style>
    <div class="score-box">
      <div style="font-size:0.9rem; color:#888;">{t('momentum_score_label')}</div>
      <div class="score-num">{score} / 100</div>
    </div>
    """
    components.html(html, height=92)

def render_perfect_day_celebration():
    html = """
    <script>
    try {
      const ctx = new (window.AudioContext || window.webkitAudioContext)();
      const notes = [523.25, 659.25, 783.99, 1046.5];
      notes.forEach((freq, i) => {
        const osc = ctx.createOscillator();
        const gain = ctx.createGain();
        osc.frequency.value = freq;
        osc.type = 'sine';
        gain.gain.setValueAtTime(0.15, ctx.currentTime + i*0.12);
        gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + i*0.12 + 0.3);
        osc.connect(gain); gain.connect(ctx.destination);
        osc.start(ctx.currentTime + i*0.12);
        osc.stop(ctx.currentTime + i*0.12 + 0.3);
      });
    } catch(e) {}
    </script>
    """
    components.html(html, height=0)
    render_confetti()
    st.success(f"🏆 {t('perfect_day_header')} — {t('perfect_day_sub')}")

# ---------------------------------------------------------------
# LIFETIME STATS, PERSONAL RECORDS & ACHIEVEMENTS
# ---------------------------------------------------------------
@st.cache_data(ttl=60, show_spinner=False)
def compute_perfect_days_count(targets, _user_id=None):
    # _user_id is part of the cache KEY, not just the query. Without it in the
    # signature, Streamlit would hand the second user the first user's cached
    # result — the cache is global, not per session.
    uid = _user_id if _user_id is not None else current_user_id()
    cols, rows = fetch("""
        SELECT d.log_date, COALESCE(d.protein_g,0) as protein_g, COALESCE(d.water_l,0) as water_l,
               COALESCE(d.steps,0) as steps, COALESCE(m.done_count,0) as meals_done,
               COALESCE(e.done_count,0) as ex_done
        FROM daily_log d
        LEFT JOIN (SELECT log_date, SUM(done) as done_count FROM meal_checks
                   WHERE user_id=%s GROUP BY log_date) m
            ON d.log_date = m.log_date
        LEFT JOIN (SELECT log_date, SUM(done) as done_count FROM exercise_checks
                   WHERE user_id=%s GROUP BY log_date) e
            ON d.log_date = e.log_date
        WHERE d.user_id=%s
    """, (uid, uid, uid))
    df = pd.DataFrame(rows, columns=cols)
    if df.empty:
        return 0
    count = 0
    for _, r in df.iterrows():
        try:
            wd = WEEKDAY_NAMES[pd.to_datetime(r["log_date"]).weekday()]
        except Exception:
            continue
        expected_ex = len(PLAN[wd]["exercises"])
        if (r["protein_g"] >= targets["protein_min"] and r["water_l"] >= targets["water_min"] and
                r["steps"] >= targets["steps_min"] and r["meals_done"] >= len(MEALS) and
                r["ex_done"] >= expected_ex):
            count += 1
    return count

@st.cache_data(ttl=60, show_spinner=False)
def get_lifetime_stats(targets, _user_id=None):
    # _user_id is in the signature so it forms part of the cache key. A cached
    # value keyed only on `targets` would leak one user's lifetime totals to
    # anyone whose targets happened to match.
    uid = _user_id if _user_id is not None else current_user_id()
    cols, rows = fetch("""
        SELECT COALESCE(SUM(steps),0), COALESCE(SUM(protein_g),0), COALESCE(SUM(water_l),0),
               COALESCE(MAX(steps),0), COALESCE(MAX(protein_g),0),
               COALESCE(SUM(CASE WHEN weight_kg IS NOT NULL AND weight_kg > 0 THEN 1 ELSE 0 END),0)
        FROM daily_log WHERE user_id=%s
    """, (uid,))
    lifetime_steps, lifetime_protein, lifetime_water, max_steps_day, max_protein_day, weight_entries = \
        rows[0] if rows else (0, 0, 0, 0, 0, 0)

    _, wo_rows = fetch("SELECT COUNT(*) FROM exercise_checks WHERE user_id=%s AND done=1",
                       (uid,))
    total_workouts = wo_rows[0][0] if wo_rows else 0

    _, meal_rows = fetch("SELECT COUNT(*) FROM meal_checks WHERE user_id=%s AND done=1",
                         (uid,))
    total_meals = meal_rows[0][0] if meal_rows else 0

    _, photo_rows = fetch("SELECT COUNT(*) FROM progress_photos WHERE user_id=%s", (uid,))
    total_photos = photo_rows[0][0] if photo_rows else 0

    _, bfast_rows = fetch("SELECT COUNT(*) FROM meal_checks "
                          "WHERE user_id=%s AND meal='Breakfast' AND done=1", (uid,))
    first_breakfast = (bfast_rows[0][0] if bfast_rows else 0) > 0

    streak, longest, total_active_days = compute_streak_stats()
    perfect_days = compute_perfect_days_count(targets, _user_id=uid)

    return {
        "lifetime_steps": lifetime_steps, "lifetime_protein": lifetime_protein,
        "lifetime_water": lifetime_water, "max_steps_day": max_steps_day,
        "max_protein_day": max_protein_day, "weight_entries": weight_entries,
        "total_workouts": total_workouts, "total_meals": total_meals,
        "total_photos": total_photos, "first_breakfast": first_breakfast,
        "longest_streak": longest, "total_active_days": total_active_days,
        "perfect_days": perfect_days,
    }

def _threshold_achievements(stat_key, icon, name_template, thresholds, fmt="{n}"):
    out = []
    for n in thresholds:
        out.append({
            "id": f"{stat_key}_{n}", "label": f"{icon} {name_template.format(n=fmt.format(n=n))}",
            "check": (lambda stats, k=stat_key, nn=n: stats.get(k, 0) >= nn)
        })
    return out

ACHIEVEMENT_DEFS = (
    _threshold_achievements("longest_streak", "🔥", "{n} Day Streak", [3, 7, 14, 30, 60, 100, 180, 365])
    + _threshold_achievements("total_active_days", "📅", "{n} Days Logged", [5, 10, 25, 50, 100, 200, 365])
    + _threshold_achievements("total_workouts", "🏋️", "{n} Workouts Completed", [1, 10, 25, 50, 100, 200, 365])
    + _threshold_achievements("lifetime_steps", "🚶", "{n} Lifetime Steps", [10000, 50000, 100000, 250000, 500000, 1000000], fmt="{n:,}")
    + _threshold_achievements("lifetime_water", "💧", "{n}L Lifetime Water", [10, 50, 100, 250, 500])
    + [
        {"id": f"protein_kg_{n}", "label": f"🍗 {n}kg Lifetime Protein",
         "check": (lambda stats, nn=n: stats.get("lifetime_protein", 0) / 1000 >= nn)}
        for n in [1, 5, 10, 25, 50]
    ]
    + _threshold_achievements("total_meals", "🍽️", "{n} Meals Logged", [10, 50, 100, 250, 500])
    + _threshold_achievements("total_photos", "📸", "{n} Photos Uploaded", [1, 5, 10, 25])
    + _threshold_achievements("perfect_days", "⚡", "{n} Perfect Days", [1, 5, 10, 25, 50])
    + [
        {"id": "first_breakfast", "label": "🥚 First Breakfast Logged", "check": lambda s: s.get("first_breakfast", False)},
        {"id": "first_workout", "label": "💪 First Workout", "check": lambda s: s.get("total_workouts", 0) >= 1},
        {"id": "first_photo", "label": "📸 First Progress Photo", "check": lambda s: s.get("total_photos", 0) >= 1},
        {"id": "first_meal", "label": "🍴 First Meal Logged", "check": lambda s: s.get("total_meals", 0) >= 1},
        {"id": "first_perfect_day", "label": "⚡ First Perfect Day", "check": lambda s: s.get("perfect_days", 0) >= 1},
        {"id": "first_weight", "label": "⚖️ First Weight Logged", "check": lambda s: s.get("weight_entries", 0) >= 1},
    ]
)

def get_achievement_status(stats):
    return [(a["id"], a["label"], a["check"](stats)) for a in ACHIEVEMENT_DEFS]

def get_next_streak_badge(current_streak):
    for threshold, name in STREAK_BADGES:
        if current_streak < threshold:
            return name, current_streak, threshold
    return None, current_streak, None

# ---------------------------------------------------------------
# DAILY COACH (rule-based, no external API)
# ---------------------------------------------------------------
def generate_coach_notes(log_date_str, weekday, targets):
    row = fetch_daily_row_readonly(log_date_str)
    meal_state = get_meal_checks(log_date_str)
    day_plan = PLAN[weekday]
    ex_state = get_exercise_checks(log_date_str, day_plan["exercises"])

    protein = row["protein_g"] or 0
    water = row["water_l"] or 0
    steps = row["steps"] or 0
    meals_done = sum(1 for v in meal_state.values() if v)
    ex_done = sum(1 for v in ex_state.values() if v)
    ex_total = len(day_plan["exercises"])

    lines, focus = [], []

    if protein >= targets["protein_min"]:
        lines.append("✅ Protein target hit")
    else:
        lines.append(f"⚠️ Protein low ({protein:.0f}g / {targets['protein_min']:.0f}g)")
        focus.append(f"Get protein up to at least {targets['protein_min']:.0f}g")

    if water >= targets["water_min"]:
        lines.append("✅ Water target hit")
    else:
        lines.append(f"⚠️ Water low ({water:.1f}L / {targets['water_min']}L)")
        focus.append(f"Drink {targets['water_min'] - water:.1f}L more water")

    if steps >= targets["steps_min"]:
        lines.append("✅ Steps target hit")
    else:
        lines.append(f"⚠️ Steps below target ({steps:,} / {targets['steps_min']:,.0f})")
        focus.append(f"Aim for {targets['steps_min']:,.0f} steps")

    if meals_done >= len(MEALS):
        lines.append("✅ All meals logged")
    else:
        lines.append(f"⚠️ Only {meals_done}/{len(MEALS)} meals logged")

    if ex_total and ex_done >= ex_total:
        lines.append("✅ Workout completed")
    elif ex_total:
        lines.append(f"⚠️ Workout incomplete ({ex_done}/{ex_total})")
        focus.append("Finish today's workout")

    if not focus:
        focus.append("Keep doing exactly what you're doing 🔥")

    return lines, focus

# Quote + a thematically matching verse for each day.
# Verses are from the World English Bible (WEB), which is public domain.
DAILY_INSPIRATION = [
    {"quote": "Discipline is choosing between what you want now and what you want most.",
     "verse": "Every man who strives in the games exercises self-control in all things.",
     "ref": "1 Corinthians 9:25 (WEB)"},
    {"quote": "Small steps every day lead to big results over time.",
     "verse": "Indeed, who despises the day of small things?",
     "ref": "Zechariah 4:10 (WEB)"},
    {"quote": "You don't have to be extreme, just consistent.",
     "verse": "Let's not be weary in doing good, for we will reap in due season, if we don't give up.",
     "ref": "Galatians 6:9 (WEB)"},
    {"quote": "The only bad workout is the one that didn't happen.",
     "verse": "In all hard work there is profit, but the talk of the lips leads only to poverty.",
     "ref": "Proverbs 14:23 (WEB)"},
    {"quote": "Progress, not perfection.",
     "verse": "Forgetting the things which are behind, and stretching forward to the things which are before, I press on toward the goal.",
     "ref": "Philippians 3:13-14 (WEB)"},
    {"quote": "Your future self is watching you right now through memories.",
     "verse": "The plans of the diligent surely lead to profit.",
     "ref": "Proverbs 21:5 (WEB)"},
    {"quote": "Motivation gets you started. Habit keeps you going.",
     "verse": "Whatever you do, work heartily, as for the Lord, and not for men.",
     "ref": "Colossians 3:23 (WEB)"},
    {"quote": "Every rep counts, every meal matters, every day adds up.",
     "verse": "He who is faithful in a very little is faithful also in much.",
     "ref": "Luke 16:10 (WEB)"},
    {"quote": "The body achieves what the mind believes.",
     "verse": "I can do all things through Christ, who strengthens me.",
     "ref": "Philippians 4:13 (WEB)"},
    {"quote": "Consistency beats intensity every single time.",
     "verse": "Run like that, that you may win.",
     "ref": "1 Corinthians 9:24 (WEB)"},
    {"quote": "Rest is part of the work, not a break from it.",
     "verse": "Come away into a deserted place, and rest a while.",
     "ref": "Mark 6:31 (WEB)"},
    {"quote": "Show up on the days you don't feel like it.",
     "verse": "Those who wait for Yahweh will renew their strength. They will mount up with wings like eagles.",
     "ref": "Isaiah 40:31 (WEB)"},
    {"quote": "Your body is worth taking care of.",
     "verse": "Don't you know that your body is a temple of the Holy Spirit? Therefore glorify God in your body.",
     "ref": "1 Corinthians 6:19-20 (WEB)"},
    {"quote": "Strength is built under load, not in comfort.",
     "verse": "The testing of your faith produces endurance. Let endurance have its perfect work.",
     "ref": "James 1:3-4 (WEB)"},
]

# Kept for backwards compatibility with anything referencing QUOTES directly.
QUOTES = [item["quote"] for item in DAILY_INSPIRATION]


def get_daily_inspiration():
    """Same pairing every day, rotating through the list — so the quote and the
    verse always share a theme rather than being picked independently."""
    return DAILY_INSPIRATION[local_today().toordinal() % len(DAILY_INSPIRATION)]


def get_daily_quote():
    return get_daily_inspiration()["quote"]


# ---------------------------------------------------------------
# STRENGTH: PRs, estimated 1RM trend, muscle volume
# ---------------------------------------------------------------
# ---------------------------------------------------------------
# CALORIE ESTIMATION
# ---------------------------------------------------------------
# Method: MET-based, per the 2024 Adult Compendium of Physical Activities.
# 1 MET = roughly 1 kcal per kg of bodyweight per hour, so:
#     kcal = MET x bodyweight_kg x hours
#
# Compendium reference values for resistance training:
#   light/moderate effort  ~3.5 METs
#   vigorous effort        ~6.0 METs
#   circuit style, minimal rest ~8.0 METs
#
# Mechanical work (weight x reps x bar path) is deliberately NOT used as the
# primary figure. An 80kg x 8 set is only ~0.75 kcal of external work, and even
# allowing for ~25% muscular efficiency it's a few kcal — nowhere near what you
# actually burn. Most of the cost is cardiovascular and comes from time spent
# training, which is what METs capture. Work IS used to nudge the MET value up
# or down, so a heavy session scores higher than an easy one of equal length.

MET_RESISTANCE_LIGHT = 3.5
MET_RESISTANCE_VIGOROUS = 6.0
MET_WALKING_MODERATE = 3.5
MET_WALKING_INCLINE = 6.0
MET_RUNNING = 9.0

# Assumed seconds per rep and rest between sets, used to turn a set count into
# a session duration when you haven't logged actual time.
SECONDS_PER_REP = 3.0
SECONDS_REST_BETWEEN_SETS = 90.0


def _bodyweight_kg(default=75.0):
    """Latest logged bodyweight, falling back to the profile, then a default."""
    cols, rows = fetch("""SELECT weight_kg FROM daily_log
                          WHERE user_id=%s AND weight_kg IS NOT NULL AND weight_kg > 0
                          ORDER BY log_date DESC LIMIT 1""", (current_user_id(),))
    if rows and rows[0][0]:
        return float(rows[0][0])
    profile = get_profile()
    if profile.get("weight_kg"):
        return float(profile["weight_kg"])
    return default


def estimate_exercise_calories(log_date, exercise, bodyweight_kg=None):
    """Rough kcal for one exercise on one day.

    Returns (kcal, minutes). Time-based entries use their logged duration with a
    walking/running MET; lifting estimates duration from reps and set count, then
    picks a MET between light and vigorous based on average load per rep.
    """
    if bodyweight_kg is None:
        bodyweight_kg = _bodyweight_kg()

    sets = get_sets(log_date, exercise)
    if not sets:
        return 0.0, 0.0

    # ---- time-based (walks, cardio) ----
    logged_minutes = sum(float(x.get("duration_min") or 0) for x in sets)
    if logged_minutes > 0:
        name = exercise.lower()
        if "run" in name or "sprint" in name:
            met = MET_RUNNING
        elif "incline" in name or "hill" in name:
            met = MET_WALKING_INCLINE
        else:
            met = MET_WALKING_MODERATE
        return met * bodyweight_kg * (logged_minutes / 60.0), logged_minutes

    # ---- lifting ----
    per_side = get_exercise_pref(exercise)["per_side"]
    total_reps = sum(int(x["reps"] or 0) for x in sets)
    working_sets = sum(1 for x in sets if (x["reps"] or 0) > 0)
    if total_reps == 0:
        return 0.0, 0.0

    minutes = (total_reps * SECONDS_PER_REP
               + max(working_sets - 1, 0) * SECONDS_REST_BETWEEN_SETS) / 60.0

    # Intensity: average load per rep relative to bodyweight. Ratio >= 1.0
    # (lifting your own bodyweight or more per rep) counts as vigorous.
    volume = sum(effective_load(x["weight_kg"], per_side) * int(x["reps"] or 0) for x in sets)
    avg_load = volume / total_reps if total_reps else 0
    ratio = min(avg_load / bodyweight_kg, 1.0) if bodyweight_kg else 0
    met = MET_RESISTANCE_LIGHT + (MET_RESISTANCE_VIGOROUS - MET_RESISTANCE_LIGHT) * ratio

    return met * bodyweight_kg * (minutes / 60.0), minutes


def estimate_day_calories(log_date, exercises):
    """Total estimated kcal and minutes across a day's exercises."""
    bw = _bodyweight_kg()
    total_kcal = 0.0
    total_min = 0.0
    breakdown = []
    for ex in exercises:
        kcal, mins = estimate_exercise_calories(log_date, ex, bw)
        if kcal > 0:
            breakdown.append({"exercise": ex, "kcal": kcal, "minutes": mins})
            total_kcal += kcal
            total_min += mins
    return {"kcal": total_kcal, "minutes": total_min,
            "breakdown": sorted(breakdown, key=lambda r: r["kcal"], reverse=True),
            "bodyweight_kg": bw}


def get_lift_prs(limit=12):
    """Heaviest single set per exercise, plus the best estimated 1RM.

    Ranked by estimated 1RM so a heavy triple beats a light set of 20.
    """
    cols, rows = fetch("""SELECT exercise, reps, weight_kg, log_date FROM exercise_sets
                          WHERE user_id=%s AND weight_kg > 0 AND reps > 0""",
                       (current_user_id(),))
    best = {}
    per_side_cache = {}
    for exercise, reps, weight, log_date in rows:
        if exercise not in per_side_cache:
            per_side_cache[exercise] = get_exercise_pref(exercise)["per_side"]
        load = effective_load(weight, per_side_cache[exercise])
        e1rm = estimated_1rm(load, reps)
        cur = best.get(exercise)
        if cur is None or e1rm > cur["e1rm"]:
            best[exercise] = {"exercise": exercise, "weight_kg": load,
                              "reps": int(reps), "e1rm": e1rm, "log_date": log_date}
    ranked = sorted(best.values(), key=lambda r: r["e1rm"], reverse=True)
    return ranked[:limit]


def get_strength_trend(exercise):
    """Per-session best estimated 1RM for one exercise, as a tidy DataFrame."""
    cols, rows = fetch("""SELECT log_date, reps, weight_kg FROM exercise_sets
                          WHERE user_id=%s AND exercise=%s
                          AND weight_kg > 0 AND reps > 0
                          ORDER BY log_date""", (current_user_id(), exercise))
    if not rows:
        return pd.DataFrame()
    per_side = get_exercise_pref(exercise)["per_side"]
    per_day = {}
    for log_date, reps, weight in rows:
        load = effective_load(weight, per_side)
        e1rm = estimated_1rm(load, reps)
        if log_date not in per_day or e1rm > per_day[log_date]["e1rm"]:
            per_day[log_date] = {"log_date": log_date, "e1rm": round(e1rm, 1),
                                 "top_weight": load}
    df = pd.DataFrame(sorted(per_day.values(), key=lambda r: r["log_date"]))
    df["log_date"] = pd.to_datetime(df["log_date"])
    return df


def get_muscle_volume_for_range(start, end):
    """Total kg-volume (load x reps) per canonical muscle region.

    Volume from a multi-muscle lift is credited in full to each region it hits,
    so these numbers show relative emphasis rather than a strict kg total.
    Per-hand dumbbell loads are doubled; time-based entries contribute nothing.
    """
    df = get_sets_for_range(start, end)
    totals = {region: 0.0 for region in MUSCLE_REGIONS}
    if df.empty:
        return totals
    per_side_cache = {}
    for _, r in df.iterrows():
        ex = r["exercise"]
        if ex not in per_side_cache:
            per_side_cache[ex] = get_exercise_pref(ex)["per_side"]
        vol = effective_load(r["weight_kg"], per_side_cache[ex]) * int(r["reps"] or 0)
        if vol <= 0:
            continue
        for region in muscle_regions_for(ex):
            totals[region] += vol
    return totals


def get_total_volume_for_range(start, end):
    """True total kg lifted (weight x reps), counting each set exactly once.

    Deliberately NOT the sum of the per-muscle figures: a multi-muscle lift is
    credited in full to every region it hits, so adding those up would roughly
    triple the real number.
    """
    df = get_sets_for_range(start, end)
    if df.empty:
        return 0.0
    per_side_cache = {}
    total = 0.0
    for _, r in df.iterrows():
        ex = r["exercise"]
        if ex not in per_side_cache:
            per_side_cache[ex] = get_exercise_pref(ex)["per_side"]
        total += effective_load(r["weight_kg"], per_side_cache[ex]) * int(r["reps"] or 0)
    return total


def get_session_volume(log_date, exercise):
    per_side = get_exercise_pref(exercise)["per_side"]
    return sum(effective_load(s["weight_kg"], per_side) * int(s["reps"] or 0)
               for s in get_sets(log_date, exercise))


def get_session_duration(log_date, exercise):
    """Total minutes and km logged for a time-based movement."""
    sets = get_sets(log_date, exercise)
    return (sum(float(s["duration_min"] or 0) for s in sets),
            sum(float(s["distance_km"] or 0) for s in sets))


def format_set_summary(sets, per_side=False):
    """Compact one-liner, shaped to how the movement is actually logged.

    '80kg × 8, 80kg × 7'          — barbell
    '22kg ×2 × 8'                 — dumbbells, per hand
    'BW × 10'                     — bodyweight
    '25 min · 3.2 km'             — cardio
    """
    parts = []
    for s in sets:
        dur = float(s.get("duration_min") or 0)
        dist = float(s.get("distance_km") or 0)
        if dur or dist:
            bits = []
            if dur:
                bits.append(f"{dur:g} min")
            if dist:
                bits.append(f"{dist:g} km")
            parts.append(" · ".join(bits))
            continue
        w, r = s["weight_kg"], s["reps"]
        if not r:
            continue
        if w and per_side:
            parts.append(f"{w:g}kg ×2 × {r}")
        elif w:
            parts.append(f"{w:g}kg × {r}")
        else:
            parts.append(f"BW × {r}")
    return ", ".join(parts)
# ---------------------------------------------------------------
# GYM BRO (AI chatbot — free Gemini API)
# ---------------------------------------------------------------
# `gemini-flash-latest` is an alias that always resolves to the current stable
# flash model. Pinning an exact version (e.g. gemini-2.5-flash) means the app
# breaks with a 404 the day Google retires it — which is what happened before.
GEMINI_MODEL = "gemini-flash-latest"

GYM_BRO_SYSTEM_PROMPT = """You are "Gym Bro" — a friendly, knowledgeable fitness buddy inside the Momentum app.

You can look up the user's own logged data using the tools provided. Use them whenever a
question refers to their training, food, weight, steps or measurements — don't guess, and
don't answer from memory of earlier turns if a fresh lookup would be more accurate.

Scope: only answer questions about training, exercise technique, nutrition, recovery, sleep,
hydration, supplements (general info, not dosing prescriptions), and general healthy habits.
If asked about something unrelated, politely redirect back to fitness/health topics.

Rules you always follow:
- You are not a doctor. Never diagnose, never give specific dosing/medical treatment advice.
- If someone describes an injury, sharp pain, chest pain, dizziness, or anything that sounds
  medically concerning, tell them clearly to see a doctor or medical professional rather than
  trying to solve it yourself.
- Do NOT prescribe a specific daily calorie deficit, a goal weight, or a rate of weight loss.
  You can explain general principles and refer to the targets the user has already set in the
  app, but the numbers are theirs to choose.
- The tools are the ONLY source of truth about this user. If a tool returns nothing for a
  date or exercise, say so plainly — never invent numbers, and never estimate a figure that
  looks like it came from their log.
- Calorie burn figures in this app are rough MET-based estimates (roughly ±30%). Treat them
  as trend indicators, not measurements, and say so if you quote one.
- Keep answers practical, encouraging, and concise — a few short paragraphs or a bullet list,
  not an essay.
"""

# How many times Gemini may call tools before we stop and answer with what we have.
# A normal question needs 1 round; a comparison might need 2-3. The cap exists so a
# confused model can't loop forever burning quota.
MAX_TOOL_ROUNDS = 5


def get_gemini_api_key():
    """Reads the Gemini API key from secrets.toml. Supports both
    [gemini] api_key = "..."  and a bare  gemini_api_key = "..."  entry.

    Note: TOML scopes keys to the [section] above them. An `api_key` line sitting
    under [connections.postgres] is NOT found here — it needs its own [gemini] header.
    """
    try:
        if "gemini" in st.secrets and "api_key" in st.secrets["gemini"]:
            return str(st.secrets["gemini"]["api_key"]).strip()
        if "gemini_api_key" in st.secrets:
            return str(st.secrets["gemini_api_key"]).strip()
    except Exception:
        pass
    return None


# ---------------------------------------------------------------
# GYM BRO TOOLS — read-only lookups over the user's own data
# ---------------------------------------------------------------
# Gemini can call these by name. Everything here is SELECT-only: there is no tool
# that writes, updates or deletes, and no tool that takes raw SQL. The model can
# only reach the specific queries defined below, with the arguments declared.

def _json_safe(obj):
    """Make DB/pandas values JSON-serialisable.

    psycopg2 hands back Decimal, pandas hands back numpy scalars and NaT, and
    json.dumps chokes on all of them. Also converts NaN to None so the model sees
    a clear 'no value' rather than a float it might quote back as a number.
    """
    import math
    from decimal import Decimal

    if obj is None:
        return None
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, Decimal):
        return float(obj)
    if isinstance(obj, (date, )):
        return obj.isoformat()
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, (int, str)):
        return obj
    if isinstance(obj, float):
        return None if math.isnan(obj) or math.isinf(obj) else round(obj, 3)
    # numpy / pandas scalars
    if hasattr(obj, "item"):
        try:
            return _json_safe(obj.item())
        except Exception:
            pass
    if pd.isna(obj) if not hasattr(obj, "__len__") else False:
        return None
    return str(obj)


def _resolve_dates(days=None, start_date=None, end_date=None, default_days=14):
    """Turn the model's loose date arguments into a concrete (start, end) pair."""
    today = local_today()
    end = today
    if end_date:
        try:
            end = date.fromisoformat(str(end_date)[:10])
        except ValueError:
            end = today
    if start_date:
        try:
            return date.fromisoformat(str(start_date)[:10]), end
        except ValueError:
            pass
    n = int(days) if days else default_days
    n = max(1, min(n, 730))
    return end - timedelta(days=n - 1), end


def _match_exercise_name(name):
    """Map whatever the model typed onto a name that actually has logged sets.

    Returns (resolved_name, candidates). Falls back to the app's own fuzzy
    exercise lookup so 'bench' finds 'Bench Press'.
    """
    cols, rows = fetch("SELECT DISTINCT exercise FROM exercise_sets WHERE user_id=%s",
                       (current_user_id(),))
    known = [r[0] for r in rows]
    if not name:
        return None, known
    if name in known:
        return name, known
    norm = _normalise_exercise_name(name)
    for k in known:
        if _normalise_exercise_name(k) == norm:
            return k, known
    partial = [k for k in known if norm and norm in _normalise_exercise_name(k)]
    if len(partial) == 1:
        return partial[0], known
    if partial:
        return min(partial, key=len), known
    return None, known


def _gb_daily_numbers(days=None, start_date=None, end_date=None):
    """Steps, protein, water, weight and meal calories per day."""
    start, end = _resolve_dates(days, start_date, end_date, default_days=14)
    daily = get_range_df(start, end)
    macros = get_meal_macro_totals_for_range(start, end)
    if daily.empty:
        return {"range": [start.isoformat(), end.isoformat()], "days": [],
                "note": "No daily log entries in this range."}
    merged = daily.merge(macros, on="log_date", how="left")
    out = []
    for _, r in merged.iterrows():
        out.append({
            "date": r["log_date"],
            "steps": r.get("steps"),
            "protein_g": r.get("protein_g"),
            "water_l": r.get("water_l"),
            "weight_kg": r.get("weight_kg"),
            "meal_calories": r.get("calories_total"),
        })
    return {"range": [start.isoformat(), end.isoformat()], "days": out}


def _gb_workout_on(date_str=None):
    """What was planned and what was actually logged on one date."""
    try:
        d = date.fromisoformat(str(date_str)[:10]) if date_str else local_today()
    except ValueError:
        d = local_today()
    ds = d.isoformat()
    weekday = WEEKDAY_NAMES[d.weekday()]
    day_plan = PLAN.get(weekday, {"label": weekday, "exercises": []})
    swaps = get_all_swaps(ds)
    checks = get_all_logged_exercises(ds)

    entries = []
    for name in effective_exercises_for_day(ds, day_plan["exercises"]):
        sets = get_sets(ds, name)
        pref = get_exercise_pref(name)
        entry = {
            "exercise": name,
            "completed": bool(checks.get(name, 0)),
            "sets": [{"set": s["set_number"], "reps": s["reps"],
                      "weight_kg": s["weight_kg"], "duration_min": s["duration_min"],
                      "distance_km": s["distance_km"]} for s in sets],
            "weight_is_per_hand": pref["per_side"],
        }
        note = get_exercise_note(ds, name)
        if note:
            entry["note"] = note
        if sets:
            entry["volume_kg"] = get_session_volume(ds, name)
        entries.append(entry)

    return {
        "date": ds, "weekday": weekday, "day_label": day_plan["label"],
        "planned_exercises": day_plan["exercises"],
        "swaps": swaps or None,
        "logged": entries,
        "estimated_kcal_note": "Session calorie estimates are MET-based, roughly ±30%.",
    }


def _gb_exercise_history(exercise=None, days=None):
    """Every session of one exercise, with the sets from each."""
    resolved, known = _match_exercise_name(exercise)
    if not resolved:
        return {"error": f"No logged sets found for '{exercise}'.",
                "exercises_with_data": sorted(known)[:60]}
    start, end = _resolve_dates(days, default_days=180)
    cols, rows = fetch("""SELECT log_date, set_number, reps, weight_kg,
                                 COALESCE(duration_min,0), COALESCE(distance_km,0)
                          FROM exercise_sets
                          WHERE user_id=%s AND exercise=%s
                          AND log_date BETWEEN %s AND %s
                          ORDER BY log_date, set_number""",
                       (current_user_id(), resolved, start.isoformat(), end.isoformat()))
    per_side = get_exercise_pref(resolved)["per_side"]
    sessions = {}
    for log_date, n, reps, weight, dur, dist in rows:
        sessions.setdefault(log_date, []).append(
            {"set": n, "reps": reps, "weight_kg": weight,
             "duration_min": dur, "distance_km": dist})
    best = get_best_ever(resolved)
    return {
        "exercise": resolved,
        "matched_from": exercise if resolved != exercise else None,
        "weight_is_per_hand": per_side,
        "sessions": [{"date": d, "sets": s} for d, s in sorted(sessions.items())],
        "best_ever": best,
    }


def _gb_lift_records():
    """Best estimated 1RM per exercise, ranked."""
    prs = get_lift_prs(limit=25)
    if not prs:
        return {"note": "No weighted sets logged yet."}
    return {"records": prs,
            "method": "Estimated 1RM uses the Epley formula — an estimate, not a tested max."}


def _gb_muscle_volume(days=None):
    """Training volume per muscle region."""
    start, end = _resolve_dates(days, default_days=7)
    by_region = get_muscle_volume_for_range(start, end)
    total = get_total_volume_for_range(start, end)
    if total <= 0:
        return {"range": [start.isoformat(), end.isoformat()],
                "note": "No weighted sets logged in this range."}
    return {
        "range": [start.isoformat(), end.isoformat()],
        "total_kg_lifted": total,
        "by_muscle_kg": {k: v for k, v in by_region.items() if v > 0},
        "note": ("Muscle figures overlap — a bench press counts toward chest, shoulders "
                 "and triceps — so they sum to more than the total."),
    }


def _gb_profile_and_targets():
    """The user's profile and their current daily targets."""
    return {"profile": get_profile(), "targets": load_targets(),
            "today": local_today().isoformat()}


def _gb_measurements(days=None):
    """Body measurement history."""
    df = get_all_measurements()
    if df.empty:
        return {"note": "No measurements logged yet."}
    start, end = _resolve_dates(days, default_days=365)
    df = df[(df["log_date"] >= start.isoformat()) & (df["log_date"] <= end.isoformat())]
    if df.empty:
        return {"note": "No measurements logged in this range."}
    return {"measurements": df.to_dict("records")}


def _gb_streaks_and_totals():
    """Streak, days logged, and lifetime totals."""
    streak, longest, total_days = compute_streak_stats()
    stats = get_lifetime_stats(load_targets(), _user_id=current_user_id())
    return {"current_streak_days": streak, "longest_streak_days": longest,
            "days_logged": total_days, "lifetime": stats}


GYM_BRO_TOOL_DISPATCH = {
    "get_daily_numbers": _gb_daily_numbers,
    "get_workout_on_date": _gb_workout_on,
    "get_exercise_history": _gb_exercise_history,
    "get_lift_records": _gb_lift_records,
    "get_muscle_volume": _gb_muscle_volume,
    "get_profile_and_targets": _gb_profile_and_targets,
    "get_measurements": _gb_measurements,
    "get_streaks_and_totals": _gb_streaks_and_totals,
}

_DAYS_PARAM = {"type": "INTEGER",
               "description": "How many days back from today to include."}

GYM_BRO_TOOLS = [{"functionDeclarations": [
    {
        "name": "get_daily_numbers",
        "description": ("Daily steps, protein, water, bodyweight and meal calories over a "
                        "date range. Use for questions about steps, hydration, protein "
                        "intake, weight trend or calories eaten."),
        "parameters": {"type": "OBJECT", "properties": {
            "days": _DAYS_PARAM,
            "start_date": {"type": "STRING", "description": "ISO date YYYY-MM-DD."},
            "end_date": {"type": "STRING", "description": "ISO date YYYY-MM-DD."},
        }},
    },
    {
        "name": "get_workout_on_date",
        "description": ("The planned session and everything actually logged on one date — "
                        "exercises, sets, reps, weights, notes and swaps. Defaults to today."),
        "parameters": {"type": "OBJECT", "properties": {
            "date_str": {"type": "STRING", "description": "ISO date YYYY-MM-DD."},
        }},
    },
    {
        "name": "get_exercise_history",
        "description": ("Every logged session of one exercise, with the sets from each, plus "
                        "the best ever set. Use for 'how has my bench progressed' questions."),
        "parameters": {"type": "OBJECT", "properties": {
            "exercise": {"type": "STRING",
                         "description": "Exercise name; partial names are matched."},
            "days": _DAYS_PARAM,
        }, "required": ["exercise"]},
    },
    {
        "name": "get_lift_records",
        "description": "Personal records across all lifts, ranked by estimated 1RM.",
        "parameters": {"type": "OBJECT", "properties": {}},
    },
    {
        "name": "get_muscle_volume",
        "description": ("Training volume in kg per muscle region over a period. Use for "
                        "'which muscles am I neglecting' questions."),
        "parameters": {"type": "OBJECT", "properties": {"days": _DAYS_PARAM}},
    },
    {
        "name": "get_profile_and_targets",
        "description": ("The user's goal, activity level, height, age, and their current "
                        "protein/water/steps/calorie targets. Also returns today's date."),
        "parameters": {"type": "OBJECT", "properties": {}},
    },
    {
        "name": "get_measurements",
        "description": "Body measurement history — waist, chest, hips, arms, thighs.",
        "parameters": {"type": "OBJECT", "properties": {"days": _DAYS_PARAM}},
    },
    {
        "name": "get_streaks_and_totals",
        "description": "Current streak, longest streak, days logged, and lifetime totals.",
        "parameters": {"type": "OBJECT", "properties": {}},
    },
]}]


def _gemini_call(contents, system_prompt, tools=None, timeout=60):
    """One REST call to Gemini. Retries with the header auth route on 401/403."""
    api_key = get_gemini_api_key()
    url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
           f"{GEMINI_MODEL}:generateContent")
    payload = {
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "contents": contents,
    }
    if tools:
        payload["tools"] = tools
    resp = requests.post(url, params={"key": api_key}, json=payload, timeout=timeout)
    if resp.status_code in (401, 403):
        resp = requests.post(url, headers={"x-goog-api-key": api_key},
                             json=payload, timeout=timeout)
    return resp


def _gemini_error_message(resp):
    """Human-readable explanation for a non-200 response."""
    if resp.status_code in (401, 403):
        return (
            "\u26a0\ufe0f Gemini rejected the key (401/403).\n\n"
            "Usually the key the app reads isn't the one you think it is. Check:\n"
            "- secrets.toml has a `[gemini]` section header directly above `api_key` "
            "(TOML scopes every key to the section above it)\n"
            "- the file is saved to disk, and Streamlit has been fully restarted\n"
            "- no stray quotes, whitespace or line breaks in the value\n\n"
            "Verify the key from a terminal:\n"
            "`curl \"https://generativelanguage.googleapis.com/v1beta/models?key=YOUR_KEY\"`"
            f"\n\nRaw response: {resp.text[:200]}"
        )
    if resp.status_code == 404:
        return (
            f"\u26a0\ufe0f Model not found (404). `{GEMINI_MODEL}` may have been renamed "
            "or retired. List what your key can reach:\n\n"
            "`curl \"https://generativelanguage.googleapis.com/v1beta/models?key=YOUR_KEY\"`"
            "\n\nthen update GEMINI_MODEL near the top of the Gym Bro section."
            f"\n\nDetails: {resp.text[:200]}"
        )
    if resp.status_code == 429:
        return "\u26a0\ufe0f Gemini rate limit hit (429). Give it a minute and try again."
    if resp.status_code == 400:
        return ("\u26a0\ufe0f Gemini rejected the request (400). Often a malformed tool "
                f"declaration or an unavailable model.\n\nDetails: {resp.text[:300]}")
    return f"\u26a0\ufe0f Gemini returned HTTP {resp.status_code}: {resp.text[:300]}"


def ask_gym_bro(user_message, chat_history, profile, targets):
    """Answer a question, letting Gemini query the user's own logged data.

    Flow: send the question with the tool declarations attached. If the model
    replies with functionCall parts, run those functions locally, append the
    results, and ask again. Repeat until it produces text or we hit
    MAX_TOOL_ROUNDS.

    Returns None (not an error string) when no API key is configured, which is
    how the caller knows to show the 'add a key' message instead.
    """
    if not get_gemini_api_key():
        return None

    system_prompt = (
        GYM_BRO_SYSTEM_PROMPT
        + f"\n\nToday's date is {local_today().isoformat()} "
          f"({WEEKDAY_NAMES[local_today().weekday()]}). Resolve relative dates like "
          f"'yesterday' or 'last week' against this."
        + f"\n\nQuick context (use tools for anything more specific): "
          f"goal {profile.get('goal')}, activity {profile.get('activity_level')}, "
          f"protein target {targets.get('protein_min')}-{targets.get('protein_max')}g, "
          f"steps target {targets.get('steps_min')}-{targets.get('steps_max')}."
    )

    contents = [
        {"role": "user" if role == "user" else "model", "parts": [{"text": text}]}
        for role, text in chat_history[-10:]
    ]
    contents.append({"role": "user", "parts": [{"text": user_message}]})

    tools_used = []

    for _ in range(MAX_TOOL_ROUNDS):
        try:
            resp = _gemini_call(contents, system_prompt, tools=GYM_BRO_TOOLS)
        except Exception as e:
            return f"\u26a0\ufe0f Couldn't reach Gemini: {e}"

        if resp.status_code != 200:
            return _gemini_error_message(resp)

        try:
            candidates = resp.json().get("candidates") or []
            if not candidates:
                return ("\u26a0\ufe0f Gemini returned no answer — it may have been blocked "
                        "by a safety filter. Try rephrasing.")
            parts = candidates[0].get("content", {}).get("parts", []) or []
        except Exception:
            return f"\u26a0\ufe0f Couldn't parse Gemini's reply: {resp.text[:300]}"

        calls = [p["functionCall"] for p in parts if "functionCall" in p]

        if not calls:
            text = "".join(p.get("text", "") for p in parts).strip()
            if not text:
                return "\u26a0\ufe0f Gemini returned an empty reply. Try rephrasing."
            return text

        # Model asked for data — run the requested lookups and feed them back.
        contents.append({"role": "model", "parts": parts})
        tool_parts = []
        for call in calls:
            name = call.get("name", "")
            args = call.get("args") or {}
            tools_used.append(name)
            fn = GYM_BRO_TOOL_DISPATCH.get(name)
            if fn is None:
                result = {"error": f"Unknown tool '{name}'."}
            else:
                try:
                    result = fn(**args)
                except TypeError as e:
                    result = {"error": f"Bad arguments for {name}: {e}"}
                except Exception as e:
                    result = {"error": f"{type(e).__name__} while running {name}: {e}"}
            if not isinstance(result, dict):
                result = {"result": result}
            tool_parts.append({"functionResponse": {
                "name": name, "response": _json_safe(result)}})
        contents.append({"role": "user", "parts": tool_parts})

    return ("\u26a0\ufe0f Gym Bro kept looking things up without settling on an answer "
            f"(tried: {', '.join(tools_used[:8])}). Try asking something narrower.")

def render_gym_bro_widget():
    """Single floating chat bubble, available on every page.

    Everything (messages + the input box) lives INSIDE the popover. Previously
    st.chat_input sat outside the popover, which is why a second stray input
    panel appeared on screen next to the bubble.
    """
    st.markdown("""
    <style>
    @keyframes gymBroPulse {
      0%, 100% { box-shadow: 0 4px 16px rgba(59,130,246,0.5); }
      50% { box-shadow: 0 4px 26px rgba(59,130,246,0.9); }
    }
    div.st-key-gym_bro_float {
        position: fixed !important;
        bottom: 28px !important;
        right: 28px !important;
        left: auto !important;
        z-index: 999999 !important;
        width: auto !important;
    }
    div.st-key-gym_bro_float > div > div > button {
        border-radius: 50% !important;
        width: 64px !important;
        height: 64px !important;
        min-width: 64px !important;
        padding: 0 !important;
        font-size: 1.8rem !important;
        background: linear-gradient(135deg, #3b82f6, #8b5cf6) !important;
        color: white !important;
        border: 2px solid rgba(255,255,255,0.3) !important;
        animation: gymBroPulse 2.5s ease-in-out infinite;
    }
    </style>
    """, unsafe_allow_html=True)

    if "gym_bro_messages" not in st.session_state:
        st.session_state.gym_bro_messages = []

    with st.container(key="gym_bro_float"):
        with st.popover("🏋️", use_container_width=False):
            st.markdown(f"**{t('gym_bro_header')}**")
            st.caption(t("gym_bro_intro"))
            st.caption(t("gym_bro_disclaimer"))

            if not get_gemini_api_key():
                st.warning(t("gym_bro_missing_key"))

            chat_box = st.container(height=280)
            with chat_box:
                if not st.session_state.gym_bro_messages:
                    st.caption("No messages yet — ask something below 👇")
                for role, text in st.session_state.gym_bro_messages:
                    with st.chat_message("user" if role == "user" else "assistant"):
                        st.markdown(text)

            # Plain text_input + button instead of st.chat_input, because
            # st.chat_input always renders pinned to the page rather than
            # inside the popover — that was the duplicate box.
            gb1, gb2 = st.columns([4, 1])
            with gb1:
                user_input = st.text_input(
                    t("gym_bro_placeholder"), key="gym_bro_text",
                    label_visibility="collapsed", placeholder=t("gym_bro_placeholder"))
            with gb2:
                send = st.button("➤", key="gym_bro_send", use_container_width=True)

            if st.session_state.gym_bro_messages:
                if st.button(t("gym_bro_clear"), key="gym_bro_clear_btn"):
                    st.session_state.gym_bro_messages = []
                    st.rerun()

            if send and user_input.strip():
                st.session_state.gym_bro_messages.append(("user", user_input.strip()))
                profile = get_profile()
                with st.spinner("Thinking..."):
                    reply = ask_gym_bro(
                        user_input.strip(), st.session_state.gym_bro_messages[:-1], profile, TARGETS)
                if reply is None:
                    reply = t("gym_bro_missing_key")
                st.session_state.gym_bro_messages.append(("assistant", reply))
                st.rerun()
# ---------------------------------------------------------------
# WEIGHT PREDICTION
# ---------------------------------------------------------------
def compute_weight_prediction():
    """Linear projection of bodyweight, with guards against nonsense.

    A naive polyfit over a couple of noisy weeks will happily project a 30kg
    loss. Four guards stop that:
      - needs at least 4 entries spanning 14+ days, so one heavy-then-light
        fortnight can't set the trend
      - dates are rebased to "days since first entry" before fitting. Fitting
        against raw ordinals (~739,000) makes the intercept enormous, and any
        later adjustment to the slope then gets multiplied by that number.
      - the slope is clamped to +/- 0.15 kg/day (about 1kg a week), beyond which
        a straight line isn't describing anything real
      - projection is anchored to the fitted value at the LAST weigh-in and
        extended forward, so a clamped slope can't drag the whole line with it
    """
    df = get_range_df(local_today() - timedelta(days=180), local_today())
    df = df.dropna(subset=["weight_kg"])
    if len(df) < 4:
        return None
    df["log_date"] = pd.to_datetime(df["log_date"])
    df = df.sort_values("log_date")
    span_days = (df["log_date"].iloc[-1] - df["log_date"].iloc[0]).days
    if span_days < 14:
        return None

    ordinals = df["log_date"].map(pd.Timestamp.toordinal).to_numpy(dtype=float)
    x = ordinals - ordinals[0]          # days since the first weigh-in
    y = df["weight_kg"].to_numpy(dtype=float)
    slope, intercept = np.polyfit(x, y, 1)

    residuals = y - (slope * x + intercept)
    band = float(np.std(residuals)) * 1.96 if len(y) > 2 else 0.0
    band = max(band, 0.3)

    MAX_KG_PER_DAY = 0.15
    clamped = bool(abs(slope) > MAX_KG_PER_DAY)
    safe_slope = max(-MAX_KG_PER_DAY, min(MAX_KG_PER_DAY, float(slope)))

    # Anchor on the fitted value at the last weigh-in, then extend. Using the
    # raw intercept with a clamped slope would shift the entire line.
    anchor = float(slope * x[-1] + intercept)

    def project(days_ahead):
        return max(30.0, anchor + safe_slope * days_ahead)

    return {
        "current": float(y[-1]),
        "pred_30": project(30), "pred_90": project(90),
        "band": band, "slope_per_week": safe_slope * 7,
        "clamped": clamped, "n": int(len(y)), "span_days": int(span_days),
    }

# ---------------------------------------------------------------
# BOTTLE ANIMATION
# ---------------------------------------------------------------
def render_confetti():
    html = """
    <canvas id="confetti-canvas" style="width:100%;height:160px;display:block;"></canvas>
    <script>
    const canvas = document.getElementById('confetti-canvas');
    const ctx = canvas.getContext('2d');
    canvas.width = canvas.offsetWidth; canvas.height = 160;
    const colors = ['#3b82f6','#f59e0b','#10b981','#ef4444','#8b5cf6'];
    let particles = [];
    for (let i=0;i<80;i++){
      particles.push({
        x: Math.random()*canvas.width, y: -20 - Math.random()*100,
        vx: (Math.random()-0.5)*3, vy: 2+Math.random()*3,
        size: 4+Math.random()*4, color: colors[Math.floor(Math.random()*colors.length)],
        rot: Math.random()*360, vrot: (Math.random()-0.5)*10
      });
    }
    let frame = 0;
    function draw(){
      ctx.clearRect(0,0,canvas.width,canvas.height);
      particles.forEach(p=>{
        p.x+=p.vx; p.y+=p.vy; p.rot+=p.vrot;
        ctx.save();
        ctx.translate(p.x,p.y); ctx.rotate(p.rot*Math.PI/180);
        ctx.fillStyle=p.color;
        ctx.fillRect(-p.size/2,-p.size/2,p.size,p.size);
        ctx.restore();
      });
      frame++;
      if (frame<90) requestAnimationFrame(draw);
    }
    draw();
    </script>
    """
    components.html(html, height=128)

def render_meal_stamp():
    html = """
    <style>
    @keyframes stampIn {
      0% { transform: scale(2) rotate(-15deg); opacity:0; }
      60% { transform: scale(0.9) rotate(5deg); opacity:1; }
      100% { transform: scale(1) rotate(-8deg); opacity:1; }
    }
    .stamp {
      display:inline-block; font-weight:800; font-size:1rem; color:#10b981;
      border:2px solid #10b981; border-radius:6px; padding:2px 8px;
      animation: stampIn 0.4s ease-out; transform: rotate(-8deg);
    }
    </style>
    <div class="stamp">✅ DONE</div>
    """
    components.html(html, height=36)

def render_animated_number(value, suffix="kg", duration_ms=900):
    html = f"""
    <div style="font-size:2rem; font-weight:700; color:#3b82f6;"
         id="counter">0{suffix}</div>
    <script>
    let start = null; const target = {value}; const duration = {duration_ms};
    function step(ts){{
      if(!start) start = ts;
      const progress = Math.min((ts-start)/duration, 1);
      const current = (target*progress).toFixed(1);
      document.getElementById('counter').innerText = current + "{suffix}";
      if(progress<1) requestAnimationFrame(step);
    }}
    requestAnimationFrame(step);
    </script>
    """
    components.html(html, height=52)

def render_trend_arrow(delta, unit="kg"):
    color = "#ef4444" if delta > 0 else ("#10b981" if delta < 0 else "#6b7280")
    arrow = "▲" if delta > 0 else ("▼" if delta < 0 else "▬")
    html = f"""
    <style>
    @keyframes slideIn {{
      0% {{ transform: translateX(-12px); opacity:0; }}
      100% {{ transform: translateX(0); opacity:1; }}
    }}
    .trend-arrow {{ animation: slideIn 0.4s ease-out; font-size:1.4rem; font-weight:700; color:{color}; }}
    </style>
    <div class="trend-arrow">{arrow} {abs(delta):.1f} {unit}</div>
    """
    components.html(html, height=40)

def render_delta_badge(delta, unit="cm"):
    if delta is None:
        return
    color = "#ef4444" if delta > 0 else ("#10b981" if delta < 0 else "#6b7280")
    arrow = "▲" if delta > 0 else ("▼" if delta < 0 else "▬")
    html = f"""
    <style>
    @keyframes slideInSmall {{ 0% {{transform:translateX(-8px);opacity:0;}} 100%{{transform:translateX(0);opacity:1;}} }}
    .delta-badge {{ animation: slideInSmall 0.35s ease-out; font-size:0.85rem; font-weight:600; color:{color}; }}
    </style>
    <div class="delta-badge">{arrow} {abs(delta):.1f}{unit} since last</div>
    """
    components.html(html, height=26)

def render_animated_bars(labels, values, max_value, value_suffix=""):
    bars = ""
    for i, (lab, val) in enumerate(zip(labels, values)):
        pct_of_max = (val / max_value * 100) if max_value else 0
        bars += f"""
        <div class="bar-col" data-target="{pct_of_max}" style="display:flex;flex-direction:column;align-items:center;flex:1;">
          <div style="font-size:0.7rem;margin-bottom:4px;color:#e6edf3;font-weight:600;">{val:.0f}{value_suffix}</div>
          <div style="width:70%;height:120px;background:rgba(147,163,184,0.18);border-radius:4px;display:flex;align-items:flex-end;overflow:hidden;">
            <div class="bar-fill" style="width:100%;height:0%;background:linear-gradient(180deg,#8b5cf6,#3b82f6);border-radius:4px 4px 0 0;transition:height 0.6s ease-out;"></div>
          </div>
          <div style="font-size:0.7rem;margin-top:4px;color:#9aa5b1;">{lab}</div>
        </div>
        """
    html = f"""
    <div style="display:flex; gap:6px; align-items:flex-end; padding:10px 0;">{bars}</div>
    <script>
    const cols = document.querySelectorAll('.bar-col');
    cols.forEach((col, i) => {{
      const target = col.getAttribute('data-target');
      const fill = col.querySelector('.bar-fill');
      setTimeout(() => {{ fill.style.height = target + '%'; }}, i*80);
    }});
    </script>
    """
    components.html(html, height=200)

def render_compare_slider(bytes_a, bytes_b):
    b64_a = base64.b64encode(bytes_a).decode()
    b64_b = base64.b64encode(bytes_b).decode()
    html = f"""
    <style>
    .compare-wrap {{ position:relative; width:100%; max-width:400px; margin:auto; overflow:hidden; border-radius:8px; }}
    .compare-wrap img {{ display:block; width:100%; }}
    .compare-after {{ position:absolute; top:0; left:0; width:50%; height:100%; overflow:hidden; }}
    .compare-after img {{ max-width:none; }}
    .compare-slider {{ position:absolute; top:0; bottom:0; left:50%; width:3px; background:white;
                        cursor:ew-resize; box-shadow:0 0 4px rgba(0,0,0,0.5); }}
    .compare-handle {{ position:absolute; top:50%; left:50%; transform:translate(-50%,-50%);
                        width:32px; height:32px; background:white; border-radius:50%; cursor:ew-resize;
                        display:flex; align-items:center; justify-content:center; font-size:14px; box-shadow:0 0 4px rgba(0,0,0,0.5); }}
    </style>
    <div class="compare-wrap" id="cwrap">
      <img src="data:image/png;base64,{b64_b}" id="img-before"/>
      <div class="compare-after" id="cafter">
        <img src="data:image/png;base64,{b64_a}" id="img-after"/>
      </div>
      <div class="compare-slider" id="cslider" style="left:50%;">
        <div class="compare-handle">↔</div>
      </div>
    </div>
    <script>
    const wrap = document.getElementById('cwrap');
    const after = document.getElementById('cafter');
    const slider = document.getElementById('cslider');
    const imgAfter = document.getElementById('img-after');
    function setWidth() {{ imgAfter.style.width = wrap.offsetWidth + 'px'; }}
    setWidth();
    window.addEventListener('resize', setWidth);
    let dragging = false;
    function moveTo(x) {{
      const rect = wrap.getBoundingClientRect();
      let pct = ((x - rect.left) / rect.width) * 100;
      pct = Math.max(0, Math.min(100, pct));
      after.style.width = pct + '%';
      slider.style.left = pct + '%';
    }}
    slider.addEventListener('mousedown', ()=>dragging=true);
    window.addEventListener('mouseup', ()=>dragging=false);
    window.addEventListener('mousemove', (e)=>{{ if(dragging) moveTo(e.clientX); }});
    slider.addEventListener('touchstart', ()=>dragging=true);
    window.addEventListener('touchend', ()=>dragging=false);
    window.addEventListener('touchmove', (e)=>{{ if(dragging) moveTo(e.touches[0].clientX); }});
    </script>
    """
    components.html(html, height=420)

def render_timeline_visual(dates):
    dots = ""
    n = len(dates)
    for i, d in enumerate(dates):
        dots += f"""
        <div style="display:flex; flex-direction:column; align-items:center; flex:1; min-width:60px;">
          <div style="width:14px;height:14px;border-radius:50%;
                      background:linear-gradient(135deg,#8b5cf6,#3b82f6);
                      box-shadow:0 0 0 3px rgba(59,130,246,0.22);"></div>
          <div style="font-size:0.65rem; margin-top:4px; text-align:center; color:#9aa5b1;">{d}</div>
        </div>
        """
    line_style = "flex:1; height:2px; background:#93a3b8; margin-top:6px;"
    html = f"""
    <div style="display:flex; align-items:flex-start; padding:10px 0; overflow-x:auto;">
      {dots}
    </div>
    """
    components.html(html, height=62)

def render_bottle(pct, message):
    pct = max(0, min(100, pct))
    glow_style = ""
    if 75 <= pct < 100:
        glow_style = """
        <style>
        @keyframes pulseGlow {
          0%,100% { filter: drop-shadow(0 0 2px #3b82f6); }
          50% { filter: drop-shadow(0 0 12px #3b82f6); }
        }
        .bottle-wrap svg { animation: pulseGlow 1.4s ease-in-out infinite; }
        </style>
        """
    html = f"""
    {glow_style}
    <div class="bottle-wrap" style="display:flex; flex-direction:column; align-items:center; padding:10px 0;">
      <svg width="110" height="220" viewBox="0 0 110 220">
        <defs>
          <clipPath id="bottleClip">
            <path d="M40,10 L70,10 L70,35 C70,35 85,45 85,65 L85,195
                     C85,208 75,215 55,215 C35,215 25,208 25,195
                     L25,65 C25,45 40,35 40,35 Z"/>
          </clipPath>
          <linearGradient id="bottleFill" x1="0" y1="1" x2="0" y2="0">
            <stop offset="0%" stop-color="#3b82f6"/>
            <stop offset="100%" stop-color="#8b5cf6"/>
          </linearGradient>
        </defs>
        <path d="M40,10 L70,10 L70,35 C70,35 85,45 85,65 L85,195
                 C85,208 75,215 55,215 C35,215 25,208 25,195
                 L25,65 C25,45 40,35 40,35 Z"
              fill="rgba(147,163,184,0.14)" stroke="rgba(147,163,184,0.55)" stroke-width="2"/>
        <rect x="0" y="{215 - (205 * pct / 100)}" width="110" height="{205 * pct / 100}"
              fill="url(#bottleFill)" clip-path="url(#bottleClip)"
              style="transition: y 0.6s ease, height 0.6s ease;"/>
        <rect x="42" y="2" width="26" height="12" rx="3" fill="rgba(147,163,184,0.55)"/>
      </svg>
      <div style="margin-top:8px; font-weight:700; font-size:1.05rem; text-align:center;
                  color:#3b82f6;">{pct:.0f}%</div>
      <div style="margin-top:2px; font-size:0.9rem; text-align:center; color:#9aa5b1;">{message}</div>
    </div>
    """
    components.html(html, height=268)
    if pct >= 100:
        render_confetti()

def get_bottle_message(pct):
    if pct >= 100:
        return t("bottle_100")
    elif pct >= 75:
        return t("bottle_75")
    elif pct >= 50:
        return t("bottle_50")
    elif pct >= 25:
        return t("bottle_25")
    else:
        return t("bottle_0")

def render_clock():
    """Ticking date + time in APP_TIMEZONE.

    Runs entirely in the browser. Driving this from Python would mean a rerun
    every second, and a Streamlit rerun re-executes the whole script — every
    query on the page, once a second.

    The zone is passed through from APP_TIMEZONE so this always agrees with
    local_today(). If the clock reads 00:15 Thursday, Thursday is the date your
    workout will be logged against.
    """
    zone = APP_TIMEZONE if _APP_TZ is not None else "UTC"
    html = f"""
    <div style="font-family:-apple-system,'Segoe UI',sans-serif;padding:2px 0 6px 0;">
      <div id="mo-date" style="font-size:0.72rem;opacity:0.6;letter-spacing:0.02em;">
        &nbsp;</div>
      <div id="mo-time" style="font-size:1.35rem;font-weight:700;color:#3b82f6;
                               font-variant-numeric:tabular-nums;line-height:1.2;">
        --:--:--</div>
    </div>
    <script>
    (function() {{
      const zone = "{zone}";
      const dateEl = document.getElementById('mo-date');
      const timeEl = document.getElementById('mo-time');
      function tick() {{
        const now = new Date();
        try {{
          timeEl.innerText = now.toLocaleTimeString('en-GB', {{
            timeZone: zone, hour: '2-digit', minute: '2-digit', second: '2-digit'
          }});
          dateEl.innerText = now.toLocaleDateString('en-GB', {{
            timeZone: zone, weekday: 'short', day: 'numeric', month: 'short'
          }});
        }} catch (e) {{
          // Unknown zone in this browser — fall back to the device clock rather
          // than leaving the dashes on screen.
          timeEl.innerText = now.toLocaleTimeString('en-GB');
          dateEl.innerText = now.toLocaleDateString('en-GB',
            {{ weekday: 'short', day: 'numeric', month: 'short' }});
        }}
      }}
      tick();
      setInterval(tick, 1000);
    }})();
    </script>
    """
    components.html(html, height=58)


def render_exercise_demo(info, display_name):
    """Demo photos + target muscles + form cues for one exercise."""
    if not info:
        st.caption(t("no_info_label"))
        return
    imgs = info.get("images") or []
    if imgs:
        img_cols = st.columns(len(imgs))
        phase_labels = ["Start position", "End position"]
        for i, (c, img_url) in enumerate(zip(img_cols, imgs)):
            with c:
                st.image(img_url,
                         caption=phase_labels[i] if i < len(phase_labels) else f"Step {i+1}",
                         use_container_width=True)
    if info.get("muscles"):
        st.markdown(f"**{t('muscles_targeted_label')}:** " + ", ".join(info["muscles"]))
    if info.get("cues"):
        st.markdown(f"**{t('form_cues_label')}:**")
        for cue in info["cues"]:
            st.markdown(f"- {cue}")


def render_rest_timer(seconds, key_suffix):
    """Self-contained countdown that runs in the browser, so it keeps ticking
    without Streamlit reruns. Beeps at zero using the same WebAudio trick as
    the perfect-day chime."""
    uid = re.sub(r"[^A-Za-z0-9_]", "_", str(key_suffix))
    html = f"""
    <style>
    .rt-wrap-{uid} {{ display:flex; align-items:center; gap:10px; padding:4px 0;
                      font-family: -apple-system, "Segoe UI", sans-serif; }}
    .rt-time-{uid} {{ font-size:1.9rem; font-weight:800; color:#3b82f6;
                      font-variant-numeric: tabular-nums; min-width:96px; }}
    .rt-bar-{uid} {{ flex:1; height:8px; background:rgba(147,163,184,0.25);
                     border-radius:4px; overflow:hidden; }}
    .rt-fill-{uid} {{ height:100%; width:100%; background:linear-gradient(90deg,#3b82f6,#8b5cf6);
                      border-radius:4px; transition:width 1s linear; }}
    .rt-done-{uid} {{ color:#10b981 !important; }}
    </style>
    <div class="rt-wrap-{uid}">
      <div class="rt-time-{uid}" id="rt-time-{uid}">--:--</div>
      <div class="rt-bar-{uid}"><div class="rt-fill-{uid}" id="rt-fill-{uid}"></div></div>
    </div>
    <script>
    (function() {{
      const total = {int(seconds)};
      let left = total;
      const timeEl = document.getElementById('rt-time-{uid}');
      const fillEl = document.getElementById('rt-fill-{uid}');
      function fmt(s) {{
        const m = Math.floor(s / 60), r = s % 60;
        return m + ':' + String(r).padStart(2, '0');
      }}
      function beep() {{
        try {{
          const ctx = new (window.AudioContext || window.webkitAudioContext)();
          [880, 1174.7, 880].forEach((f, i) => {{
            const o = ctx.createOscillator(), g = ctx.createGain();
            o.frequency.value = f; o.type = 'sine';
            g.gain.setValueAtTime(0.2, ctx.currentTime + i * 0.18);
            g.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + i * 0.18 + 0.32);
            o.connect(g); g.connect(ctx.destination);
            o.start(ctx.currentTime + i * 0.18); o.stop(ctx.currentTime + i * 0.18 + 0.32);
          }});
        }} catch (e) {{}}
      }}
      timeEl.innerText = fmt(left);
      const iv = setInterval(function() {{
        left -= 1;
        if (left <= 0) {{
          clearInterval(iv);
          timeEl.innerText = "GO 💪";
          timeEl.classList.add('rt-done-{uid}');
          fillEl.style.width = '0%';
          beep();
          return;
        }}
        timeEl.innerText = fmt(left);
        fillEl.style.width = (left / total * 100) + '%';
      }}, 1000);
    }})();
    </script>
    """
    components.html(html, height=64)


def render_muscle_heatmap(volume_by_region):
    """Body map with a compact colour-swatch legend beside it.

    Layout note: the silhouette and the legend are two native Streamlit columns
    rather than one HTML block. On desktop they sit side by side (the original
    look); on a phone Streamlit stacks columns automatically, so the legend
    drops neatly underneath instead of being clipped by the SVG iframe's fixed
    height — which is what was cutting the numbers off before.
    """
    max_vol = max(volume_by_region.values()) if volume_by_region else 0
    if max_vol <= 0:
        return False

    def shade(region):
        v = volume_by_region.get(region, 0)
        ratio = (v / max_vol) if max_vol else 0
        if ratio <= 0:
            return "rgba(147,163,184,0.18)"
        # pale blue -> hot violet as volume climbs
        r = int(59 + (139 - 59) * ratio)
        g = int(130 + (92 - 130) * ratio)
        b = 246
        alpha = 0.28 + 0.72 * ratio
        return f"rgba({r},{g},{b},{alpha:.2f})"

    map_col, legend_col = st.columns([3, 2])

    with map_col:
        html = f"""
        <div style="display:flex;justify-content:center;">
          <svg viewBox="0 0 290 250" width="100%" style="max-width:330px;height:auto;"
               preserveAspectRatio="xMidYMid meet">
            <!-- ============ FRONT ============ -->
            <text x="62" y="12" font-size="9" fill="#9aa5b1" text-anchor="middle">FRONT</text>
            <circle cx="62" cy="34" r="13" fill="rgba(147,163,184,0.22)"/>
            <rect x="52" y="48" width="20" height="8" rx="3" fill="rgba(147,163,184,0.22)"/>
            <circle cx="40" cy="64" r="11" fill="{shade('Shoulders')}"/>
            <circle cx="84" cy="64" r="11" fill="{shade('Shoulders')}"/>
            <rect x="43" y="58" width="38" height="26" rx="7" fill="{shade('Chest')}"/>
            <rect x="47" y="86" width="30" height="34" rx="6" fill="{shade('Core')}"/>
            <rect x="26" y="74" width="13" height="34" rx="6" fill="{shade('Biceps')}"/>
            <rect x="85" y="74" width="13" height="34" rx="6" fill="{shade('Biceps')}"/>
            <rect x="45" y="124" width="16" height="52" rx="7" fill="{shade('Quads')}"/>
            <rect x="63" y="124" width="16" height="52" rx="7" fill="{shade('Quads')}"/>
            <rect x="47" y="180" width="12" height="42" rx="6" fill="{shade('Calves')}"/>
            <rect x="65" y="180" width="12" height="42" rx="6" fill="{shade('Calves')}"/>

            <!-- ============ BACK ============ -->
            <text x="205" y="12" font-size="9" fill="#9aa5b1" text-anchor="middle">BACK</text>
            <circle cx="205" cy="34" r="13" fill="rgba(147,163,184,0.22)"/>
            <rect x="195" y="48" width="20" height="8" rx="3" fill="rgba(147,163,184,0.22)"/>
            <circle cx="183" cy="64" r="11" fill="{shade('Shoulders')}"/>
            <circle cx="227" cy="64" r="11" fill="{shade('Shoulders')}"/>
            <rect x="186" y="58" width="38" height="46" rx="7" fill="{shade('Back')}"/>
            <rect x="169" y="74" width="13" height="34" rx="6" fill="{shade('Triceps')}"/>
            <rect x="228" y="74" width="13" height="34" rx="6" fill="{shade('Triceps')}"/>
            <rect x="188" y="108" width="34" height="20" rx="8" fill="{shade('Glutes')}"/>
            <rect x="188" y="132" width="16" height="46" rx="7" fill="{shade('Hamstrings')}"/>
            <rect x="206" y="132" width="16" height="46" rx="7" fill="{shade('Hamstrings')}"/>
            <rect x="190" y="182" width="12" height="42" rx="6" fill="{shade('Calves')}"/>
            <rect x="208" y="182" width="12" height="42" rx="6" fill="{shade('Calves')}"/>
          </svg>
        </div>
        """
        components.html(html, height=260)

    with legend_col:
        # Compact legend: swatch, muscle name, volume. Fixed region order so the
        # rows don't jump around between visits.
        rows = ""
        for region in MUSCLE_REGIONS:
            v = volume_by_region.get(region, 0)
            rows += (
                f"<div style='display:flex;align-items:center;gap:6px;"
                f"padding:2px 0;font-size:0.75rem;'>"
                f"<span style='width:11px;height:11px;border-radius:3px;flex:none;"
                f"background:{shade(region)};display:inline-block;"
                f"border:1px solid rgba(147,163,184,0.35);'></span>"
                f"<span style='opacity:0.65;'>{region}</span>"
                f"<span style='margin-left:auto;font-weight:600;color:inherit;'>"
                f"{v:,.0f} kg</span></div>"
            )
        st.markdown(f"<div style='padding-top:14px;'>{rows}</div>",
                    unsafe_allow_html=True)

    st.caption(t("volume_overlap_note"))

    return True



# ---------------------------------------------------------------
# AUTHENTICATION (stage 1 of multi-user)
# ---------------------------------------------------------------
# Sign-in uses Streamlit's built-in OIDC support, so no password ever touches
# this app: Google authenticates the person and hands back a verified email.
#
# Identity is stored as a small integer user_id in the `users` table rather than
# the email itself. Two reasons: the email then lives in exactly one row instead
# of being copied across every workout set you've ever logged, and swapping
# provider later (GitHub, say) means updating one mapping row rather than
# rewriting the whole database.
#
# STAGE 1 SCOPE: this resolves who you are. It does NOT yet filter data by user —
# every query still returns the same rows it always did. Stages 2 and 3 add the
# user_id columns and the WHERE clauses.

SINGLE_USER_ID = 1          # the implicit user all existing data belongs to


def auth_is_configured():
    """True when secrets.toml has an [auth] section for Streamlit to use."""
    try:
        return "auth" in st.secrets and bool(st.secrets["auth"])
    except Exception:
        return False


def _allowed_emails():
    """Optional allowlist. Absent or empty means anyone who signs in is allowed.

    Set it in secrets.toml as:
        [auth]
        allowed_emails = ["you@gmail.com", "mate@gmail.com"]
    """
    try:
        raw = st.secrets["auth"].get("allowed_emails")
    except Exception:
        return None
    if not raw:
        return None
    return {str(e).strip().lower() for e in raw if str(e).strip()}


def ensure_users_table():
    run("""CREATE TABLE IF NOT EXISTS users (
        user_id SERIAL PRIMARY KEY,
        email TEXT UNIQUE NOT NULL,
        display_name TEXT,
        created_at TIMESTAMP DEFAULT now(),
        last_seen TIMESTAMP DEFAULT now())""")


def get_or_create_user(email, display_name=None):
    """Resolve an email to a stable integer user_id, creating the row if new.

    The first account created takes user_id 1, which is the id all pre-existing
    data is assigned to in stage 2. So sign in with your own account first.
    """
    email = (email or "").strip().lower()
    if not email:
        return None
    ensure_users_table()
    cols, rows = fetch("SELECT user_id FROM users WHERE email=%s", (email,))
    if rows:
        uid = rows[0][0]
        run("UPDATE users SET last_seen=now(), display_name=COALESCE(%s, display_name) "
            "WHERE user_id=%s", (display_name, uid))
        return uid
    run("INSERT INTO users (email, display_name) VALUES (%s, %s) "
        "ON CONFLICT (email) DO NOTHING", (email, display_name))
    cols, rows = fetch("SELECT user_id FROM users WHERE email=%s", (email,))
    return rows[0][0] if rows else None


def current_user():
    """Who is this? Returns (user_id, email, display_name).

    Falls back to the single-user identity when auth isn't configured, so the
    app keeps working before Google credentials are in place.
    """
    if not auth_is_configured():
        return SINGLE_USER_ID, None, None

    user = getattr(st, "user", None)
    if user is None or not getattr(user, "is_logged_in", False):
        return None, None, None

    email = (getattr(user, "email", "") or "").strip().lower()
    name = getattr(user, "name", None)

    allowed = _allowed_emails()
    if allowed is not None and email not in allowed:
        return "DENIED", email, name

    cached = st.session_state.get("_auth_user")
    if cached and cached[1] == email:
        return cached
    resolved = (get_or_create_user(email, name), email, name)
    st.session_state["_auth_user"] = resolved
    return resolved


def render_login_gate():
    """Show the sign-in screen and stop the script. Returns nothing."""
    st.markdown("### ⚡ Momentum")
    st.write("Sign in to reach your training log.")
    if hasattr(st, "login"):
        st.button("Sign in with Google", type="primary",
                  on_click=lambda: st.login("google"))
    else:
        st.error(
            "This version of Streamlit has no built-in login. "
            "Run `pip install --upgrade streamlit` (1.42 or newer is required), "
            "or remove the [auth] section from secrets.toml to go back to "
            "single-user mode.")
    st.stop()


def render_account_box():
    """Sidebar footer: who's signed in, and a way out."""
    if not auth_is_configured():
        st.caption("👤 Single-user mode — sign-in not configured yet.")
        return
    user = getattr(st, "user", None)
    label = (getattr(user, "name", None) or getattr(user, "email", None) or "Signed in")
    st.caption(f"👤 {label}")
    if hasattr(st, "logout"):
        st.button("Sign out", key="_logout_btn", use_container_width=True,
                  on_click=st.logout)


# ---------------------------------------------------------------
# UI
# ---------------------------------------------------------------
st.set_page_config(page_title="Momentum", page_icon="⚡", layout="centered")

# --- Who is using the app? ---------------------------------------------------
# Resolved before anything reads the database. In stage 1 this only gates access;
# from stage 3 onward CURRENT_USER_ID is what every query filters on.
CURRENT_USER_ID, CURRENT_USER_EMAIL, CURRENT_USER_NAME = current_user()

if CURRENT_USER_ID == "DENIED":
    st.error("That account isn't on the allowlist for this app.")
    st.caption(f"Signed in as {CURRENT_USER_EMAIL}")
    if hasattr(st, "logout"):
        st.button("Sign out", on_click=st.logout)
    st.stop()

if CURRENT_USER_ID is None:
    render_login_gate()

if "language" not in st.session_state:
    st.session_state.language = get_setting("language", "English")
if "selected_date" not in st.session_state:
    st.session_state.selected_date = local_today()

with st.sidebar:
    render_clock()
    lang_choice = st.selectbox(t("language_label"), LANGUAGES, index=LANGUAGES.index(st.session_state.language))
    if lang_choice != st.session_state.language:
        st.session_state.language = lang_choice
        set_setting("language", lang_choice)
        st.rerun()
    st.caption("💡 Tip: use the ⋮ menu (top right) → Settings → Theme for light/dark mode.")
    render_account_box()
    page = st.radio(
        t("app_title"),
        [t("nav_home"), t("nav_today"), t("nav_weight_log"), t("nav_weekly_dashboard"),
         t("nav_measurements"), t("nav_photos"), t("nav_achievements"), t("nav_plan"),
         t("nav_profile"), t("nav_settings")],
        label_visibility="collapsed"
    )

# PLAN is the editable workout split, loaded from the DB (seeded from GYM_SPLIT
# on first run). Everything downstream reads PLAN, never GYM_SPLIT directly, so
# edits made on the Workout Plan page take effect everywhere.
repair_set_numbering()   # one-off cleanup of sets numbered from 2 by an older build
PLAN = load_plan()
TARGETS = load_targets()
st.title(t("app_title"))
render_gym_bro_widget()

if page == t("nav_home"):
    today = local_today()
    today_str = today.isoformat()
    weekday = WEEKDAY_NAMES[today.weekday()]
    day_plan = PLAN[weekday]

    streak, longest, total_days = compute_streak_stats()
    score = compute_momentum_score(today_str, weekday, TARGETS)

    # --- Weekly muscle volume: the most distinctive thing on this page, so it
    # --- leads rather than sitting below four rows of metric cards.
    st.markdown(f"#### {t('muscle_volume_header')}")
    st.caption(t("muscle_volume_caption"))
    wk_start = today - timedelta(days=today.weekday())
    wk_end = wk_start + timedelta(days=6)
    total_vol = get_total_volume_for_range(wk_start, wk_end)
    if total_vol > 0:
        st.caption(f"**{total_vol:,.0f} kg** {t('total_volume_label').lower()}")
    if not render_muscle_heatmap(get_muscle_volume_for_range(wk_start, wk_end)):
        st.info(t("no_volume_yet"))

    st.markdown("---")

    # --- Today at a glance ---
    latest_weight_df = get_range_df(today - timedelta(days=90), today).dropna(subset=["weight_kg"])
    g1, g2, g3, g4 = st.columns(4)
    g1.metric(t("momentum_score_label").replace("⚡ Today's ", ""), f"{score}")
    g2.metric(t("current_streak_label"), f"{streak} 🔥")
    if not latest_weight_df.empty:
        g3.metric(t("current_weight_home_label"),
                  f"{latest_weight_df['weight_kg'].iloc[-1]:.1f} kg")
        week_ago_df = latest_weight_df[
            pd.to_datetime(latest_weight_df["log_date"])
            <= pd.Timestamp(today - timedelta(days=7))]
        if not week_ago_df.empty:
            change = (latest_weight_df["weight_kg"].iloc[-1]
                      - week_ago_df["weight_kg"].iloc[-1])
            g4.metric(t("weekly_change_label"), f"{change:+.1f} kg")
        else:
            g4.metric(t("weekly_change_label"), "—")
    else:
        g3.metric(t("current_weight_home_label"), "—")
        g4.metric(t("weekly_change_label"), "—")

    st.caption(f"**{t('todays_workout_label')}:** {day_plan['label']}")

    next_name, current, threshold = get_next_streak_badge(longest)
    if next_name:
        st.caption(f"{t('next_badge_label')}: {next_name} ({current}/{threshold})")
    else:
        st.caption(f"{t('next_badge_label')}: 🏅 All streak badges unlocked!")

    st.markdown("---")

    # --- Quote + verse, moved to the bottom: nice to have, not why you opened
    # --- the app on a training day.
    inspiration = get_daily_inspiration()
    st.caption(t("quote_of_day_label"))
    st.markdown(f"*{inspiration['quote']}*")
    st.markdown(
        f"<div style='border-left:3px solid #3b82f6;padding:6px 0 6px 12px;"
        f"margin:8px 0 4px 0;'>"
        f"<div style='font-size:0.92rem;font-style:italic;opacity:0.85;'>"
        f"&ldquo;{inspiration['verse']}&rdquo;</div>"
        f"<div style='font-size:0.76rem;opacity:0.6;margin-top:4px;'>"
        f"{inspiration['ref']}</div></div>",
        unsafe_allow_html=True)

elif page == t("nav_today"):
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if st.button(t("prev_day"), use_container_width=True):
            st.session_state.selected_date -= timedelta(days=1)
    with col3:
        if st.button(t("next_day"), use_container_width=True):
            st.session_state.selected_date += timedelta(days=1)
    with col2:
        picked = st.date_input("Date", value=st.session_state.selected_date, label_visibility="collapsed")
        st.session_state.selected_date = picked

    log_date = st.session_state.selected_date
    log_date_str = log_date.isoformat()
    weekday = WEEKDAY_NAMES[log_date.weekday()]
    day_plan = PLAN[weekday]
    week_start = log_date - timedelta(days=log_date.weekday())

    st.subheader(f"{weekday}, {log_date.strftime('%d %b %Y')}")
    st.caption(f"{t('gym_label')}: **{day_plan['label']}**")

    row = get_daily_row(log_date_str)

    score = compute_momentum_score(log_date_str, weekday, TARGETS)
    render_momentum_score_badge(score)

    if is_rest_week(week_start):
        st.info(t("rest_week_on"))

    if get_perfect_day_status(log_date_str, weekday, TARGETS):
        render_perfect_day_celebration()

    # Workout first — on a training day that's what you actually open the app for.
    tab_workout, tab_meals, tab_numbers = st.tabs(
        [t("tab_workout"), t("tab_meals"), t("tab_numbers")])

    # =========================== WORKOUT ===========================
    with tab_workout:
        ex_state = get_exercise_checks(log_date_str, day_plan["exercises"])

        total_ex = len(day_plan["exercises"])
        done_ex = sum(1 for ex in day_plan["exercises"] if ex_state[ex])
        pct = (done_ex / total_ex * 100) if total_ex else 0

        bottle_col, head_col = st.columns([1, 2])
        with bottle_col:
            render_bottle(pct, get_bottle_message(pct))
        with head_col:
            st.markdown(f"### {day_plan['label']}")
            st.caption(f"{done_ex}/{total_ex} exercises done")
            # Resolve swaps: a swapped exercise's sets live under the NEW name.
            all_today_ex = effective_exercises_for_day(log_date_str, day_plan["exercises"])

            with st.expander(t("rest_timer_label"), expanded=False):
                rt1, rt2, rt3 = st.columns(3)
                for secs, col in ((60, rt1), (90, rt2), (180, rt3)):
                    with col:
                        if st.button(f"{secs//60}:{secs%60:02d}", key=f"rt_{log_date_str}_{secs}",
                                     use_container_width=True):
                            st.session_state["active_timer"] = secs
                if st.session_state.get("active_timer"):
                    render_rest_timer(st.session_state["active_timer"],
                                      f"{log_date_str}_{st.session_state['active_timer']}")

        st.markdown("---")

        for ex in day_plan["exercises"]:
            swap = get_exercise_swap(log_date_str, ex)
            effective_name = swap if swap else ex

            checkbox_label = effective_name if not swap else f"{effective_name}  🔄"
            checked = st.checkbox(checkbox_label, value=ex_state[ex], key=f"ex_{log_date_str}_{ex}")
            # Checkbox state stays keyed on the ORIGINAL name so streaks, momentum
            # score and perfect-day logic are unaffected by swaps.
            if checked != ex_state[ex]:
                set_exercise_check(log_date_str, ex, checked)
                ex_state[ex] = checked

            # "Last time" hint — the single most useful thing mid-workout
            prev_date, prev_sets = get_last_session(effective_name, log_date_str)
            if prev_sets:
                summary = format_set_summary(
                    prev_sets, per_side=get_exercise_pref(effective_name)["per_side"])
                if summary:
                    st.caption(f"↩ {t('last_time_label')} ({prev_date}): {summary}")

            # Peak from the last session, and best ever — the two numbers you're
            # aiming at. Both exclude today so they don't mirror what you just typed.
            last_peak = get_last_session_peak(effective_name, log_date_str)
            pb = get_best_ever(effective_name, before_date=log_date_str)
            peak_bits = []
            if last_peak:
                peak_bits.append(
                    f"⬆ {t('last_peak_label')}: **{last_peak['weight_kg']:g} kg** "
                    f"× {last_peak['reps']}")
            if pb:
                peak_bits.append(f"🏅 {t('best_ever_label')}: **{pb['top_weight']:g} kg**")
            if peak_bits:
                st.markdown(
                    f"<div style='font-size:0.82rem;color:#9aa5b1;margin:-6px 0 6px 0;'>"
                    + "&nbsp;&nbsp;·&nbsp;&nbsp;".join(peak_bits).replace("**", "")
                    + "</div>", unsafe_allow_html=True)

            with st.expander(f"{t('exercise_info_label')} — {effective_name}", expanded=False):
                info = find_exercise_info(effective_name)

                if swap:
                    st.caption(f"{t('swapped_from_label')}: {ex}")
                    if st.button(t("revert_swap_label"), key=f"revert_{log_date_str}_{ex}"):
                        remove_exercise_swap(log_date_str, ex)
                        st.rerun()

                # ---------------- HOW THIS MOVEMENT IS LOGGED ----------------
                pref = get_exercise_pref(effective_name)
                type_keys = list(LOG_TYPES.keys())
                chosen_type = st.selectbox(
                    t("log_type_label"), type_keys,
                    index=type_keys.index(pref["log_type"]) if pref["log_type"] in type_keys else 0,
                    format_func=lambda k: LOG_TYPES[k],
                    key=f"logtype_{log_date_str}_{effective_name}")

                per_side = pref["per_side"]
                if chosen_type == "weight_reps":
                    per_side = st.checkbox(
                        t("per_side_label"), value=pref["per_side"],
                        help=t("per_side_help"),
                        key=f"perside_{log_date_str}_{effective_name}")

                if (chosen_type, per_side) != (pref["log_type"], pref["per_side"]):
                    set_exercise_pref(effective_name, chosen_type, per_side)
                    st.rerun()

                # ---------------- THE SETS THEMSELVES ----------------
                st.markdown(f"**{t('sets_header')}**")
                sets = get_sets(log_date_str, effective_name)
                if not sets:
                    sets = [{"set_number": 1, "reps": 0, "weight_kg": 0.0,
                             "duration_min": 0.0, "distance_km": 0.0}]

                for sset in sets:
                    n = sset["set_number"]
                    kbase = f"{log_date_str}_{effective_name}_{n}"

                    if chosen_type == "duration":
                        dc1, dc2, dc3 = st.columns([1, 2, 2])
                        with dc1:
                            st.markdown(
                                f"<div style='padding-top:32px;color:#9aa5b1;font-size:0.8rem;'>"
                                f"#{n}</div>", unsafe_allow_html=True)
                        with dc2:
                            dur = st.number_input(t("duration_label"), min_value=0.0, step=5.0,
                                                  value=float(sset.get("duration_min") or 0),
                                                  key=f"dur_{kbase}")
                        with dc3:
                            dist = st.number_input(t("distance_label"), min_value=0.0, step=0.1,
                                                   value=float(sset.get("distance_km") or 0),
                                                   key=f"dist_{kbase}")
                        if (dur, dist) != (sset.get("duration_min"), sset.get("distance_km")):
                            save_set(log_date_str, effective_name, n, 0, 0, dur, dist)

                    elif chosen_type == "bodyweight":
                        bc1, bc2 = st.columns([1, 4])
                        with bc1:
                            st.markdown(
                                f"<div style='padding-top:32px;color:#9aa5b1;font-size:0.8rem;'>"
                                f"{t('set_label')} {n}</div>", unsafe_allow_html=True)
                        with bc2:
                            r = st.number_input(t("reps_label"), min_value=0, step=1,
                                                value=int(sset["reps"] or 0), key=f"r_{kbase}")
                        if r != sset["reps"]:
                            save_set(log_date_str, effective_name, n, r, 0)

                    else:  # weight_reps
                        sc1, sc2, sc3 = st.columns([1, 2, 2])
                        with sc1:
                            st.markdown(
                                f"<div style='padding-top:32px;color:#9aa5b1;font-size:0.8rem;'>"
                                f"{t('set_label')} {n}</div>", unsafe_allow_html=True)
                        with sc2:
                            w = st.number_input(
                                t("weight_each_label") if per_side else t("weight_kg_label"),
                                min_value=0.0, step=2.5,
                                value=float(sset["weight_kg"] or 0), key=f"w_{kbase}")
                        with sc3:
                            r = st.number_input(t("reps_label"), min_value=0, step=1,
                                                value=int(sset["reps"] or 0), key=f"r_{kbase}")
                        if (w, r) != (sset["weight_kg"], sset["reps"]):
                            save_set(log_date_str, effective_name, n, r, w)
                        if per_side and w:
                            st.caption(f"= {effective_load(w, True):g} kg total")

                ac1, ac2 = st.columns(2)
                with ac1:
                    if st.button(t("add_set"), key=f"addset_{log_date_str}_{effective_name}",
                                 use_container_width=True):
                        # If the only row on screen is the unsaved placeholder,
                        # persist it first so the new set lands at 2, not 1.
                        stored = get_sets(log_date_str, effective_name)
                        if not stored:
                            first = sets[0]
                            save_set(log_date_str, effective_name, 1,
                                     first.get("reps", 0), first.get("weight_kg", 0),
                                     first.get("duration_min", 0), first.get("distance_km", 0))
                        save_set(log_date_str, effective_name,
                                 next_set_number(log_date_str, effective_name), 0, 0)
                        st.rerun()
                with ac2:
                    if len(sets) > 1 and st.button(
                            t("remove_set"), key=f"delset_{log_date_str}_{effective_name}",
                            use_container_width=True):
                        delete_last_set(log_date_str, effective_name)
                        renumber_sets(log_date_str, effective_name)
                        st.rerun()

                # ---------------- SESSION SUMMARY ----------------
                if chosen_type == "duration":
                    mins, km = get_session_duration(log_date_str, effective_name)
                    if mins or km:
                        ex_kcal, _ = estimate_exercise_calories(log_date_str, effective_name)
                        d1, d2, d3 = st.columns(3)
                        d1.metric(t("total_time_label"), f"{mins:g} min")
                        d2.metric(t("total_distance_label"), f"{km:g} km")
                        d3.metric(t("calories_burned_label"), f"~{ex_kcal:,.0f}")
                else:
                    vol = get_session_volume(log_date_str, effective_name)
                    if vol > 0:
                        today_sets = get_sets(log_date_str, effective_name)
                        best = max((estimated_1rm(effective_load(x["weight_kg"], per_side), x["reps"])
                                    for x in today_sets), default=0)
                        today_top = max((effective_load(x["weight_kg"], per_side)
                                         for x in today_sets), default=0)
                        prior = get_best_ever(effective_name, before_date=log_date_str)

                        ex_kcal, ex_min = estimate_exercise_calories(
                            log_date_str, effective_name)
                        v1, v2, v3 = st.columns(3)
                        v1.metric(t("session_volume_label"), f"{vol:,.0f} kg")
                        if best > 0:
                            delta = f"{best - prior['e1rm']:+.1f} kg" if prior else None
                            v2.metric(t("e1rm_label"), f"{best:.1f} kg", delta=delta)
                        v3.metric(t("calories_burned_label"), f"~{ex_kcal:,.0f}",
                                  help=t("calorie_per_exercise_help").format(
                                      mins=f"{ex_min:.0f}"))

                        if prior and today_top > prior["top_weight"]:
                            st.success(
                                f"🎉 {t('new_pb_label')} {today_top:g} kg "
                                f"(was {prior['top_weight']:g} kg)")

                note = st.text_input(
                    t("exercise_note_label"), value=get_exercise_note(log_date_str, effective_name),
                    key=f"exnote_{log_date_str}_{effective_name}")
                if note != get_exercise_note(log_date_str, effective_name):
                    set_exercise_note(log_date_str, effective_name, note)

                st.markdown("---")
                # ---------------- DEMO / FORM ----------------
                if info:
                    render_exercise_demo(info, effective_name)
                else:
                    st.caption(t("no_demo_for_swap") if swap else t("no_info_label"))

                st.markdown(f"**{t('swap_label')}**")
                st.caption(t("swap_help"))
                # Picking from the known list means the swapped exercise still
                # gets its demo images, muscles and cues. Free text still works,
                # and is matched fuzzily against the library.
                # Filters are stacked, not in columns. Side-by-side columns get
                # squeezed to ~40% width on a phone, which made the muscle
                # dropdown nearly unusable.
                muscle_opts = [t("swap_all_muscles")] + exercise_muscle_filters()
                swap_muscle = st.selectbox(
                    t("swap_muscle_label"), muscle_opts, index=0,
                    key=f"swap_m_{log_date_str}_{ex}")

                swap_query = st.text_input(
                    t("swap_search_label"), value="",
                    placeholder=t("swap_search_placeholder"),
                    key=f"swap_q_{log_date_str}_{ex}")

                muscle_filter = None if swap_muscle == t("swap_all_muscles") else swap_muscle
                matches = search_exercises(swap_query, muscle_filter)

                picked_swap = None
                if matches:
                    st.caption(f"**{len(matches)}** {t('swap_matches_suffix')}")
                    # A radio list for short result sets: no dropdown to open, no
                    # scrolling inside a tiny native picker on mobile.
                    if len(matches) <= 12:
                        picked_swap = st.radio(
                            t("swap_label"), matches, index=0,
                            label_visibility="collapsed",
                            key=f"swap_radio_{log_date_str}_{ex}")
                    else:
                        picked_swap = st.selectbox(
                            t("swap_label"), matches, index=0,
                            label_visibility="collapsed",
                            help=t("swap_narrow_help"),
                            key=f"swap_pick_{log_date_str}_{ex}")
                        st.caption(t("swap_narrow_hint"))
                else:
                    st.caption(t("swap_no_matches"))

                if st.button(t("swap_confirm"), key=f"swap_btn_{log_date_str}_{ex}",
                             type="primary", use_container_width=True):
                    # A typed query that matched nothing is still honoured — you
                    # might be logging something the library has never heard of.
                    chosen_swap = picked_swap or swap_query.strip()
                    if chosen_swap:
                        set_exercise_swap(log_date_str, ex, chosen_swap)
                        st.rerun()

        # ---- Extra exercises (anything beyond today's plan) ----
        st.markdown(f"##### {t('extra_exercises_header')}")
        all_logged = get_all_logged_exercises(log_date_str)
        extra_names = [name for name in all_logged if name not in day_plan["exercises"]]

        for name in extra_names:
            ec1, ec2 = st.columns([5, 1])
            with ec1:
                checked = st.checkbox(name, value=bool(all_logged[name]),
                                      key=f"extra_{log_date_str}_{name}")
                if checked != bool(all_logged[name]):
                    set_exercise_check(log_date_str, name, checked)
            with ec2:
                if st.button(t("delete_label"), key=f"extra_del_{log_date_str}_{name}"):
                    remove_exercise(log_date_str, name)
                    st.rerun()
            extra_info = find_exercise_info(name)
            if extra_info:
                with st.expander(f"{t('exercise_info_label')} — {name}", expanded=False):
                    render_exercise_demo(extra_info, name)

        new_extra_col1, new_extra_col2 = st.columns([4, 1])
        with new_extra_col1:
            new_extra_name = st.text_input(
                t("add_extra_placeholder"), key=f"new_extra_{log_date_str}",
                label_visibility="collapsed", placeholder=t("add_extra_placeholder"))
        with new_extra_col2:
            if st.button(t("add_extra_button"), key=f"add_extra_btn_{log_date_str}",
                         use_container_width=True):
                if new_extra_name.strip():
                    set_exercise_check(log_date_str, new_extra_name.strip(), True)
                    st.rerun()

        # ---- Estimated energy cost of this session ----
        burn = estimate_day_calories(log_date_str, all_today_ex)
        if burn["kcal"] > 0:
            st.markdown("---")
            bt1, bt2 = st.columns(2)
            bt1.metric(t("day_total_calories_label"), f"~{burn['kcal']:,.0f} kcal")
            bt2.metric(t("session_time_label"), f"~{burn['minutes']:.0f} min")
            with st.expander(t("calorie_detail_header"), expanded=False):
                st.caption(t("calorie_method_note").format(bw=f"{burn['bodyweight_kg']:.0f}"))
                rows_df = pd.DataFrame([{
                    "Exercise": b["exercise"],
                    "Minutes": f"{b['minutes']:.0f}",
                    "kcal": f"{b['kcal']:,.0f}",
                } for b in burn["breakdown"]])
                st.dataframe(rows_df, hide_index=True, use_container_width=True)
                st.metric(t("calories_burned_label"), f"~{burn['kcal']:,.0f} kcal")
                st.warning(t("calorie_accuracy_warning"))

    # ============================ MEALS ============================
    with tab_meals:
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
                checked = st.checkbox(t("done_label"), value=meal_state[meal],
                                      key=f"meal_{log_date_str}_{meal}")
                if checked != meal_state[meal]:
                    set_meal_check(log_date_str, meal, checked)
                if checked:
                    render_meal_stamp()
                note = st.text_input(t("what_did_you_have"), value=details["note"],
                                     key=f"meal_note_{log_date_str}_{meal}")
                mc1, mc2 = st.columns(2)
                with mc1:
                    cals = st.number_input(t("calories_label"), min_value=0.0,
                                           value=float(details["calories"]), step=10.0,
                                           key=f"meal_cal_{log_date_str}_{meal}")
                with mc2:
                    prot = st.number_input(t("protein_label"), min_value=0.0,
                                           value=float(details["protein_g"]), step=1.0,
                                           key=f"meal_prot_{log_date_str}_{meal}")
                if (note, cals, prot) != (details["note"], details["calories"], details["protein_g"]):
                    set_meal_detail(log_date_str, meal, note, cals, prot)
                day_calories_total += cals
                day_meal_protein_total += prot

        st.caption(
            f"{day_calories_total:.0f} kcal · {day_meal_protein_total:.0f}g "
            f"(target {TARGETS['calories_min']:.0f}-{TARGETS['calories_max']:.0f} kcal, "
            f"{TARGETS['protein_min']:.0f}-{TARGETS['protein_max']:.0f}g protein)")

    # =========================== NUMBERS ===========================
    with tab_numbers:
        n1, n2, n3 = st.columns(3)
        with n1:
            protein = st.number_input(
                f"{t('protein_label')} — target {TARGETS['protein_min']:.0f}-{TARGETS['protein_max']:.0f}g",
                min_value=0.0, value=float(row["protein_g"] or 0), step=5.0,
                key=f"protein_{log_date_str}")
            if st.button(t("use_meal_total"), key=f"use_meal_protein_{log_date_str}"):
                # Assigning to `protein` here did nothing: the click triggers a
                # rerun, so the save button never fires on this pass and the
                # number_input re-renders from its own widget state. Write into
                # the widget's session key instead, then rerun. Read the total
                # straight from the DB so it doesn't depend on tab render order.
                _md = get_meal_details(log_date_str)
                st.session_state[f"protein_{log_date_str}"] = float(
                    sum(d["protein_g"] for d in _md.values()))
                st.rerun()
        with n2:
            water = st.number_input(
                f"{t('water_label')} — target {TARGETS['water_min']}-{TARGETS['water_max']}L",
                min_value=0.0, value=float(row["water_l"] or 0), step=0.25,
                key=f"water_{log_date_str}")
        with n3:
            steps = st.number_input(
                f"{t('steps_label')} — target {TARGETS['steps_min']:,.0f}-{TARGETS['steps_max']:,.0f}",
                min_value=0, value=int(row["steps"] or 0), step=500, key=f"steps_{log_date_str}")

        st.progress(min(protein / TARGETS["protein_min"], 1.0) if TARGETS["protein_min"] else 0,
                    text=f"{t('protein_label')}: {protein:.0f}g / {TARGETS['protein_min']:.0f}g min")
        st.progress(min(water / TARGETS["water_min"], 1.0) if TARGETS["water_min"] else 0,
                    text=f"{t('water_label')}: {water:.2f}L / {TARGETS['water_min']}L min")
        st.progress(min(steps / TARGETS["steps_min"], 1.0) if TARGETS["steps_min"] else 0,
                    text=f"{t('steps_label')}: {steps:,} / {TARGETS['steps_min']:,.0f} min")

        st.markdown(f"##### {t('weight_section_header')}")
        weight = st.number_input(t("weight_label"), min_value=0.0,
                                 value=float(row["weight_kg"]) if row["weight_kg"] else 0.0,
                                 step=0.1, key=f"weight_{log_date_str}")
        notes = st.text_area(t("notes_label"), value=row["notes"] or "", key=f"notes_{log_date_str}")

        if st.button(t("save_button"), type="primary", use_container_width=True):
            save_daily_row(log_date_str, protein, water, steps,
                           weight if weight > 0 else None, notes)
            st.success(t("saved_msg"))

        st.markdown("---")
        deload = st.checkbox(t("rest_week_label"), value=is_rest_week(week_start),
                             key=f"restweek_{week_start.isoformat()}")
        if deload != is_rest_week(week_start):
            set_rest_week(week_start, deload)
            st.rerun()

        with st.expander(t("streak_header"), expanded=False):
            streak, longest, total_days = compute_streak_stats()
            badges = get_earned_badges(longest, total_days)
            s1, s2, s3 = st.columns(3)
            s1.metric(t("current_streak_label"), f"{streak} 🔥")
            s2.metric(t("longest_streak_label"), f"{longest}")
            s3.metric(t("days_logged_label"), f"{total_days}")
            if badges:
                st.caption(f"{t('badges_label')}: " + " · ".join(badges))

        with st.expander(t("coach_header"), expanded=False):
            coach_lines, coach_focus = generate_coach_notes(log_date_str, weekday, TARGETS)
            for line in coach_lines:
                st.write(line)
            st.markdown(f"**{t('coach_tomorrow_focus')}**")
            for f in coach_focus:
                st.write(f"• {f}")
elif page == t("nav_weight_log"):
    st.subheader(t("weight_trend_header"))
    end = local_today()
    start = end - timedelta(days=90)
    df = get_range_df(start, end)
    df = df.dropna(subset=["weight_kg"])
    if df.empty:
        st.info(t("no_weight_entries"))
    else:
        df["log_date"] = pd.to_datetime(df["log_date"])
        df = df.sort_values("log_date")
        df["trend"] = df["weight_kg"].rolling(window=7, min_periods=1).mean()
        chart_df = df.set_index("log_date")[["weight_kg", "trend"]]
        chart_df.columns = [t("weight_label"), "7-day trend"]
        st.line_chart(chart_df)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.caption(t("latest_weight"))
            render_animated_number(float(df['weight_kg'].iloc[-1]), suffix=" kg")
        if len(df) > 1:
            change = df["weight_kg"].iloc[-1] - df["weight_kg"].iloc[0]
            with c2:
                st.caption(t("change_period"))
                render_trend_arrow(change, unit="kg")
        c3.metric(t("entries_logged"), len(df))
        st.dataframe(df[["log_date", "weight_kg"]].rename(
            columns={"log_date": "Date", "weight_kg": t("weight_label")}
        ).sort_values("Date", ascending=False), hide_index=True, use_container_width=True)

    st.markdown(f"#### {t('prediction_header')}")
    prediction = compute_weight_prediction()
    if prediction:
        band = prediction["band"]
        pr1, pr2 = st.columns(2)
        pr1.metric(t("prediction_30"),
                   f"{prediction['pred_30'] - band:.1f}–{prediction['pred_30'] + band:.1f} kg")
        pr2.metric(t("prediction_90"),
                   f"{prediction['pred_90'] - band:.1f}–{prediction['pred_90'] + band:.1f} kg")
        st.caption(
            f"Straight-line projection from {prediction['n']} weigh-ins over "
            f"{prediction['span_days']} days — currently about "
            f"{prediction['slope_per_week']:+.2f} kg/week. The range reflects how "
            f"scattered your entries are around that line. It assumes nothing "
            f"changes, which it will — treat it as a direction, not a forecast.")
        if prediction["clamped"]:
            st.caption(
                "Your recent trend is steeper than a straight line can sensibly "
                "extend, so the projection has been capped. Short-term swings are "
                "usually water and food weight rather than real change.")
    else:
        st.info(t("prediction_insufficient"))

elif page == t("nav_weekly_dashboard"):
    st.subheader(t("weekly_adherence_header"))

    if "week_offset" not in st.session_state:
        st.session_state.week_offset = 0

    wcol1, wcol2, wcol3 = st.columns([1, 2, 1])
    with wcol1:
        if st.button(t("prev_week"), key="prev_week_btn"):
            st.session_state.week_offset -= 1
    with wcol3:
        if st.button(t("next_week"), key="next_week_btn"):
            st.session_state.week_offset += 1
    with wcol2:
        if st.session_state.week_offset != 0:
            if st.button(t("back_to_this_week"), key="reset_week_btn"):
                st.session_state.week_offset = 0

    today = local_today()
    start_of_week = today - timedelta(days=today.weekday()) + timedelta(weeks=st.session_state.week_offset)
    end_of_week = start_of_week + timedelta(days=6)
    st.caption(f"{start_of_week.strftime('%d %b')} – {end_of_week.strftime('%d %b %Y')}")
    if is_rest_week(start_of_week):
        st.info(t("rest_week_on"))

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
    # Force chronological order (Mon->Sun for this week) instead of the default alphabetical sort
    weekday_order = summary["weekday"].tolist()
    summary["weekday"] = pd.Categorical(summary["weekday"], categories=weekday_order, ordered=True)
    summary["meal_pct"] = (summary["meals_done"] / len(MEALS) * 100).clip(0, 100)
    summary["protein_hit"] = summary["protein_g"] >= TARGETS["protein_min"]
    summary["steps_hit"] = summary["steps"] >= TARGETS["steps_min"]
    summary["ex_pct"] = summary.apply(
        lambda r: (r["ex_done"] / r["ex_total"] * 100) if r["ex_total"] > 0 else 0, axis=1)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric(t("meals_hit_avg"), f"{summary['meal_pct'].mean():.0f}%")
    m2.metric(t("workouts_completed"), f"{summary['ex_pct'].mean():.0f}%")
    m3.metric(t("protein_target_days"), f"{int(summary['protein_hit'].sum())}/7")
    m4.metric(t("steps_target_days"), f"{int(summary['steps_hit'].sum())}/7")

    st.markdown("#### " + t("meals_hit_avg").split(" (")[0])
    render_animated_bars(summary["weekday"].astype(str).tolist(), summary["meal_pct"].tolist(), 100, "%")

    st.markdown("#### " + t("workouts_completed"))
    render_animated_bars(summary["weekday"].astype(str).tolist(), summary["ex_pct"].tolist(), 100, "%")

    pw_df = summary.set_index("weekday")[["protein_g", "water_l"]].copy()
    pw_df.columns = [t("protein_label"), t("water_label")]
    st.line_chart(pw_df)

    cal_max = max(summary["calories_total"].max(), 1)
    render_animated_bars(summary["weekday"].astype(str).tolist(), summary["calories_total"].tolist(), cal_max, "")

    display_cols = ["log_date", "protein_g", "water_l", "steps", "weight_kg", "meals_done",
                     "ex_done", "ex_total", "calories_total", "meal_protein_total"]
    st.dataframe(summary[display_cols].rename(columns={
        "log_date": "Date", "protein_g": t("protein_label"), "water_l": t("water_label"),
        "steps": t("steps_label"), "weight_kg": t("weight_label"), "meals_done": "Meals done",
        "ex_done": "Exercises done", "ex_total": "Exercises total",
        "calories_total": "Calories", "meal_protein_total": "Protein (meals)"
    }), hide_index=True, use_container_width=True)

    with st.expander(t("advanced_trends_header"), expanded=False):
        end30 = local_today()
        start30 = end30 - timedelta(days=30)
        daily30 = get_range_df(start30, end30)
        macro30 = get_meal_macro_totals_for_range(start30, end30)
        ex30 = get_exercise_completion_for_range(start30, end30)

        all_days30 = [(start30 + timedelta(days=i)).isoformat() for i in range(31)]
        s30 = pd.DataFrame({"log_date": all_days30})
        s30 = s30.merge(daily30, on="log_date", how="left")
        s30 = s30.merge(macro30, on="log_date", how="left")
        s30 = s30.merge(ex30.rename(columns={"done_count": "ex_done", "total": "ex_total"}), on="log_date", how="left")
        s30 = s30.fillna(0)
        s30["log_date"] = pd.to_datetime(s30["log_date"])
        s30 = s30.sort_values("log_date")
        s30["calories_roll7"] = s30["calories_total"].rolling(7, min_periods=1).mean()
        s30["protein_roll7"] = s30["protein_g"].rolling(7, min_periods=1).mean()
        s30["steps_roll7"] = s30["steps"].rolling(7, min_periods=1).mean()
        s30["ex_pct"] = s30.apply(lambda r: (r["ex_done"] / r["ex_total"] * 100) if r["ex_total"] > 0 else 0, axis=1)
        s30["compliance_roll7"] = s30["ex_pct"].rolling(7, min_periods=1).mean()

        weight30 = get_range_df(start30, end30).dropna(subset=["weight_kg"])
        if not weight30.empty:
            weight30["log_date"] = pd.to_datetime(weight30["log_date"])
            weight30 = weight30.sort_values("log_date")
            weight30["weight_roll7"] = weight30["weight_kg"].rolling(7, min_periods=1).mean()
            fig_w = px.line(weight30, x="log_date", y="weight_roll7", title="7-Day Rolling Avg — Weight (kg)")
            st.plotly_chart(fig_w, use_container_width=True)

        fig_cal = px.line(s30, x="log_date", y="calories_roll7", title="7-Day Rolling Avg — Calories")
        st.plotly_chart(fig_cal, use_container_width=True)

        fig_prot = px.line(s30, x="log_date", y="protein_roll7", title="7-Day Rolling Avg — Protein (g)")
        st.plotly_chart(fig_prot, use_container_width=True)

        fig_steps = px.line(s30, x="log_date", y="steps_roll7", title="7-Day Rolling Avg — Steps")
        st.plotly_chart(fig_steps, use_container_width=True)

        fig_comp = px.line(s30, x="log_date", y="compliance_roll7", title="7-Day Rolling Avg — Workout Compliance %")
        st.plotly_chart(fig_comp, use_container_width=True)

elif page == t("nav_measurements"):
    st.subheader(t("measurements_header"))
    log_date = st.session_state.selected_date
    log_date_str = log_date.isoformat()
    m_row = get_measurement_row(log_date_str)
    mc1, mc2 = st.columns(2)
    all_m_prior = get_all_measurements()
    all_m_prior = all_m_prior[all_m_prior["log_date"] != log_date_str]
    prev_row = None
    if not all_m_prior.empty:
        prev_row = all_m_prior.sort_values("log_date").iloc[-1]

    def delta_for(field):
        if prev_row is None or pd.isna(prev_row[field]):
            return None
        cur_val = m_row[field]
        if not cur_val:
            return None
        return float(cur_val) - float(prev_row[field])

    with mc1:
        waist = st.number_input(t("waist_label"), min_value=0.0, value=float(m_row["waist_cm"] or 0), step=0.5)
        render_delta_badge(delta_for("waist_cm"))
        chest = st.number_input(t("chest_label"), min_value=0.0, value=float(m_row["chest_cm"] or 0), step=0.5)
        render_delta_badge(delta_for("chest_cm"))
        hips = st.number_input(t("hips_label"), min_value=0.0, value=float(m_row["hips_cm"] or 0), step=0.5)
        render_delta_badge(delta_for("hips_cm"))
    with mc2:
        arms = st.number_input(t("arms_label"), min_value=0.0, value=float(m_row["arms_cm"] or 0), step=0.5)
        render_delta_badge(delta_for("arms_cm"))
        thighs = st.number_input(t("thighs_label"), min_value=0.0, value=float(m_row["thighs_cm"] or 0), step=0.5)
        render_delta_badge(delta_for("thighs_cm"))
    if st.button(t("save_measurement"), type="primary"):
        save_measurement(log_date_str, waist or None, chest or None, hips or None, arms or None, thighs or None)
        st.success(t("saved_msg"))
    all_m = get_all_measurements()
    if not all_m.empty:
        all_m["log_date"] = pd.to_datetime(all_m["log_date"])
        all_m = all_m.sort_values("log_date").set_index("log_date")
        chart_df = all_m[["waist_cm", "chest_cm", "hips_cm", "arms_cm", "thighs_cm"]].dropna(how="all")
        if not chart_df.empty:
            st.line_chart(chart_df)

elif page == t("nav_photos"):
    st.subheader(t("photos_header"))
    log_date = st.session_state.selected_date
    log_date_str = log_date.isoformat()
    uploaded = st.file_uploader(t("upload_photo"), type=["png", "jpg", "jpeg"])
    caption = st.text_input(t("caption_label"))
    if uploaded is not None and st.button(t("save_photo"), type="primary"):
        save_photo(log_date_str, caption, uploaded.getvalue())
        st.success(t("saved_msg"))
        st.rerun()
    photos = get_all_photos()
    if photos:
        st.markdown(f"#### {t('photo_timeline_header')}")
        sorted_photos = sorted(photos, key=lambda p: p["log_date"])
        timeline_dates = [p["log_date"] for p in sorted_photos]
        render_timeline_visual(timeline_dates)

        timeline_labels = {f"{p['log_date']} — {p['caption'] or 'no caption'}": p for p in sorted_photos}
        selected_label = st.selectbox("View details for:", list(timeline_labels.keys()), index=len(timeline_labels) - 1)
        selected_photo = timeline_labels[selected_label]

        detail_row = fetch_daily_row_readonly(selected_photo["log_date"])
        detail_measurements = get_measurement_row(selected_photo["log_date"])
        td1, td2, td3 = st.columns(3)
        td1.metric(t("weight_label"), f"{detail_row['weight_kg']:.1f} kg" if detail_row.get("weight_kg") else "—")
        td2.metric(t("waist_label"), f"{detail_measurements['waist_cm']:.1f}" if detail_measurements.get("waist_cm") else "—")
        td3.caption(f"💬 {selected_photo['caption'] or '—'}")
        st.image(bytes(selected_photo["photo_data"]), width=250)

        st.markdown(f"#### {t('compare_header')}")
        options = {f"{p['log_date']} — {p['caption'] or 'no caption'} (#{p['id']})": p for p in photos}
        choice_labels = list(options.keys())
        c1, c2 = st.columns(2)
        with c1:
            pick1 = st.selectbox("A", choice_labels, key="photo_a")
        with c2:
            pick2 = st.selectbox("B", choice_labels, index=min(1, len(choice_labels) - 1), key="photo_b")
        cc1, cc2 = st.columns(2)
        with cc1:
            st.caption(pick1)
        with cc2:
            st.caption(pick2)
        p1, p2 = options[pick1], options[pick2]
        render_compare_slider(bytes(p1["photo_data"]), bytes(p2["photo_data"]))
        st.markdown("---")
        for p in photos:
            cols = st.columns([1, 3, 1])
            with cols[0]:
                st.image(bytes(p["photo_data"]), width=100)
            with cols[1]:
                st.caption(f"{p['log_date']} — {p['caption'] or ''}")
            with cols[2]:
                if st.button(t("delete_label"), key=f"del_photo_{p['id']}"):
                    delete_photo(p["id"])
                    st.rerun()

elif page == t("nav_achievements"):
    st.subheader(t("achievements_header"))
    stats = get_lifetime_stats(TARGETS, _user_id=current_user_id())
    achievements = get_achievement_status(stats)
    unlocked = [a for a in achievements if a[2]]
    st.caption(f"{len(unlocked)}/{len(achievements)} {t('achievements_unlocked')}")
    st.progress(len(unlocked) / len(achievements) if achievements else 0)

    cols = st.columns(2)
    for i, (aid, label, done) in enumerate(achievements):
        with cols[i % 2]:
            if done:
                st.markdown(f"✅ {label}")
            else:
                st.markdown(f"<span style='color:#999;'>🔒 {label}</span>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(f"#### {t('lifetime_stats_header')}")
    l1, l2, l3, l4 = st.columns(4)
    l1.metric(t("lifetime_workouts_label"), f"{stats['total_workouts']:,}")
    l2.metric(t("lifetime_steps_label"), f"{stats['lifetime_steps']:,.0f}")
    l3.metric(t("lifetime_protein_label"), f"{stats['lifetime_protein']:,.0f}g")
    l4.metric(t("lifetime_water_label"), f"{stats['lifetime_water']:,.1f}L")

    st.markdown(f"#### {t('personal_records_header')}")
    p1, p2, p3 = st.columns(3)
    p1.metric(t("pr_max_steps_label"), f"{stats['max_steps_day']:,.0f}")
    p2.metric(t("pr_longest_streak_label"), f"{stats['longest_streak']} days")
    p3.metric(t("pr_max_protein_label"), f"{stats['max_protein_day']:,.0f}g")

    # ---- Lift records, driven by the sets/reps/weight log ----
    st.markdown(f"#### {t('lift_prs_header')}")
    prs = get_lift_prs()
    if prs:
        pr_df = pd.DataFrame([{
            "Exercise": r["exercise"],
            "Best set": f"{r['weight_kg']:g}kg × {r['reps']}",
            "Est. 1RM": f"{r['e1rm']:.1f}kg",
            "Date": r["log_date"],
        } for r in prs])
        st.dataframe(pr_df, hide_index=True, use_container_width=True)
    else:
        st.info("Log some sets with weight on the Today page to start building lift records.")

    st.markdown(f"#### {t('strength_trend_header')}")
    lifts = get_logged_exercise_names()
    if not lifts:
        st.info(t("no_weighted_sets"))
    else:
        chosen = st.selectbox(t("pick_lift_label"), lifts, key="strength_pick")
        trend = get_strength_trend(chosen)

        if len(trend) >= 2:
            fig_s = px.line(trend, x="log_date", y="e1rm", markers=True,
                            title=f"Estimated 1RM — {chosen}")
            fig_s.update_layout(yaxis_title="Est. 1RM (kg)", xaxis_title="")
            st.plotly_chart(fig_s, use_container_width=True)

            first, last = trend["e1rm"].iloc[0], trend["e1rm"].iloc[-1]
            peak = trend["e1rm"].max()
            m1, m2, m3 = st.columns(3)
            m1.metric(t("sessions_logged_label"), len(trend))
            m2.metric(t("e1rm_now_label"), f"{last:.1f} kg", delta=f"{last - first:+.1f} kg")
            m3.metric(t("e1rm_peak_label"), f"{peak:.1f} kg")

            show = trend.copy()
            show["log_date"] = show["log_date"].dt.strftime("%d %b %Y")
            st.dataframe(
                show.rename(columns={"log_date": "Date", "e1rm": "Est. 1RM (kg)",
                                     "top_weight": "Heaviest set (kg)"}),
                hide_index=True, use_container_width=True)

        elif len(trend) == 1:
            # One session is still worth showing — previously this branch printed
            # a small caption and looked like the picker had done nothing.
            only = trend.iloc[0]
            st.info(t("trend_one_session"))
            o1, o2 = st.columns(2)
            o1.metric(t("e1rm_label"), f"{only['e1rm']:.1f} kg")
            o2.metric(t("heaviest_set_label"), f"{only['top_weight']:g} kg")
            st.caption(f"{t('logged_on_label')} {only['log_date'].strftime('%d %b %Y')}")
        else:
            st.info(t("trend_no_data"))

elif page == t("nav_plan"):
    st.subheader(t("plan_header"))
    st.caption(t("plan_intro"))

    for weekday in WEEKDAY_NAMES:
        current_day = PLAN[weekday]
        with st.expander(f"{weekday} — {current_day['label']}", expanded=False):
            label = st.text_input(t("day_label_label"), value=current_day["label"],
                                  key=f"plan_label_{weekday}")
            ex_text = st.text_area(
                t("exercises_label"), value="\n".join(current_day["exercises"]),
                height=200, key=f"plan_ex_{weekday}",
                help="One exercise per line. Set/rep notation like '4x6-8' is optional "
                     "and is stripped when matching the demo images.")
            pc1, pc2 = st.columns(2)
            with pc1:
                if st.button(t("save_plan"), key=f"plan_save_{weekday}",
                             type="primary", use_container_width=True):
                    exercises = [line.strip() for line in ex_text.split("\n") if line.strip()]
                    if exercises:
                        save_plan_day(weekday, label.strip() or weekday, exercises)
                        st.success(t("saved_msg"))
                        st.rerun()
                    else:
                        st.warning("Add at least one exercise before saving.")
            with pc2:
                if st.button(t("reset_plan"), key=f"plan_reset_{weekday}",
                             use_container_width=True):
                    reset_plan_day(weekday)
                    st.rerun()

elif page == t("nav_profile"):
    st.subheader(t("profile_header"))
    st.caption(t("profile_disclaimer"))
    profile = get_profile()

    p1, p2 = st.columns(2)
    with p1:
        goal = st.selectbox(t("goal_label"), GOALS, index=GOALS.index(profile["goal"]) if profile["goal"] in GOALS else 2)
        height = st.number_input(t("height_label"), min_value=0.0, value=float(profile["height_cm"] or 0), step=0.5)
        weight = st.number_input(t("current_weight_label"), min_value=0.0, value=float(profile["weight_kg"] or 0), step=0.1)
        age = st.number_input(t("age_label"), min_value=0, value=int(profile["age"] or 0), step=1)
    with p2:
        sex = st.selectbox(t("sex_label"), SEX_OPTIONS, index=SEX_OPTIONS.index(profile["sex"]) if profile["sex"] in SEX_OPTIONS else 2)
        country = st.text_input(t("country_label"), value=profile["country"] or "")
        activity = st.selectbox(t("activity_label"), ACTIVITY_LEVELS,
                                 index=ACTIVITY_LEVELS.index(profile["activity_level"]) if profile["activity_level"] in ACTIVITY_LEVELS else 2)

    if st.button(t("recalc_button"), type="primary"):
        save_profile(goal, height or None, weight or None, age or None, sex, country, activity)
        new_targets = compute_targets_from_profile({
            "height_cm": height, "weight_kg": weight, "age": age,
            "sex": sex, "activity_level": activity, "goal": goal
        })
        if new_targets:
            save_targets(new_targets)
            st.success(t("targets_updated_msg"))
        else:
            st.success(t("saved_msg"))
        st.rerun()

elif page == t("nav_settings"):
    st.subheader(t("settings_header"))
    st.markdown(f"#### {t('edit_targets_header')}")
    current = load_targets()
    t1, t2 = st.columns(2)
    with t1:
        cal_min = st.number_input("Calories min", value=float(current["calories_min"]), step=50.0)
        cal_max = st.number_input("Calories max", value=float(current["calories_max"]), step=50.0)
        prot_min = st.number_input("Protein min (g)", value=float(current["protein_min"]), step=5.0)
        prot_max = st.number_input("Protein max (g)", value=float(current["protein_max"]), step=5.0)
    with t2:
        water_min = st.number_input("Water min (L)", value=float(current["water_min"]), step=0.25)
        water_max = st.number_input("Water max (L)", value=float(current["water_max"]), step=0.25)
        steps_min = st.number_input("Steps min", value=float(current["steps_min"]), step=500.0)
        steps_max = st.number_input("Steps max", value=float(current["steps_max"]), step=500.0)
    if st.button(t("save_targets"), type="primary"):
        save_targets({
            "calories_min": cal_min, "calories_max": cal_max,
            "protein_min": prot_min, "protein_max": prot_max,
            "water_min": water_min, "water_max": water_max,
            "steps_min": steps_min, "steps_max": steps_max,
        })
        st.success(t("saved_msg"))
        st.rerun()

    st.markdown("---")
    st.markdown(f"#### {t('export_header')}")
    st.caption(t("export_caption"))
    e1, e2, e3 = st.columns(3)
    with e1:
        daily_all = get_range_df(date(2000, 1, 1), local_today())
        st.download_button(t("download_daily"), daily_all.to_csv(index=False).encode("utf-8"),
                           file_name="momentum_daily_log.csv", mime="text/csv",
                           use_container_width=True)
    with e2:
        sets_all = get_all_sets_df()
        st.download_button(t("download_sets"), sets_all.to_csv(index=False).encode("utf-8"),
                           file_name="momentum_training_sets.csv", mime="text/csv",
                           use_container_width=True)
    with e3:
        meas_all = get_all_measurements()
        st.download_button(t("download_measurements"), meas_all.to_csv(index=False).encode("utf-8"),
                           file_name="momentum_measurements.csv", mime="text/csv",
                           use_container_width=True)
