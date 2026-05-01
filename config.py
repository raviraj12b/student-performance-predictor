import os

# ── Paths ──────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'model.pkl')
DATA_PATH  = os.path.join(BASE_DIR, 'data', 'raw', 'StudentsPerformance.csv')
LOG_DIR    = os.path.join(BASE_DIR, 'logs')

# ── Model settings ─────────────────────────────────────
TEST_SIZE    = 0.2
RANDOM_STATE = 42

# ── Grade thresholds ───────────────────────────────────
GRADE_A = 80
GRADE_B = 60
GRADE_C = 40

# ── Valid input values (used for validation) ───────────
VALID_INPUTS = {
    'gender': [
        'male', 'female'
    ],
    'race/ethnicity': [
        'group A', 'group B', 'group C', 'group D', 'group E'
    ],
    'parental_level_of_education': [
        'some high school', 'high school', 'some college',
        "associate's degree", "bachelor's degree", "master's degree"
    ],
    'lunch': [
        'standard', 'free/reduced'
    ],
    'test_preparation_course': [
        'none', 'completed'
    ]
}