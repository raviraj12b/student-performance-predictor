from flask import Flask, render_template, request
import pickle
import pandas as pd
from config import MODEL_PATH, VALID_INPUTS, GRADE_A, GRADE_B, GRADE_C
from logger import get_logger

app = Flask(__name__)
log = get_logger(__name__)

# ── Load model ─────────────────────────────────────────
try:
    with open(MODEL_PATH, 'rb') as f:
        pipeline = pickle.load(f)
    log.info(f"Model loaded successfully from {MODEL_PATH}")
except FileNotFoundError:
    log.error(f"model.pkl not found at {MODEL_PATH}. Run the notebook first.")
    pipeline = None


# ── Input validation ───────────────────────────────────
def validate_inputs(form):
    errors = []
    checks = {
        'gender':             ('gender',                      form.get('gender',             '')),
        'ethnicity':          ('race/ethnicity',              form.get('ethnicity',          '')),
        'parental_education': ('parental_level_of_education', form.get('parental_education', '')),
        'lunch':              ('lunch',                       form.get('lunch',              '')),
        'test_preparation':   ('test_preparation_course',     form.get('test_preparation',   '')),
    }
    for field, (config_key, value) in checks.items():
        if not value:
            errors.append(f"'{field}' is required.")
        elif value not in VALID_INPUTS[config_key]:
            errors.append(f"Invalid value for '{field}': {value}")
    return errors


# ── Grade helper ───────────────────────────────────────
def get_grade(score):
    if   score >= GRADE_A: return 'A'
    elif score >= GRADE_B: return 'B'
    elif score >= GRADE_C: return 'C'
    else:                  return 'D'


# ── Routes ─────────────────────────────────────────────
@app.route('/')
def index():
    log.info("Home page accessed")
    return render_template('index.html', prediction=None, error=None)


@app.route('/predict', methods=['POST'])
def predict():
    log.info(f"Prediction requested → {dict(request.form)}")

    errors = validate_inputs(request.form)
    if errors:
        log.warning(f"Validation failed: {errors}")
        return render_template('index.html', prediction=None, error=errors[0])

    if pipeline is None:
        return render_template('index.html', prediction=None,
                               error="Model not loaded. Please train the model first.")

    try:
        input_data = pd.DataFrame({
            'gender':                       [request.form['gender']],
            'race/ethnicity':               [request.form['ethnicity']],
            'parental_level_of_education':  [request.form['parental_education']],
            'lunch':                        [request.form['lunch']],
            'test_preparation_course':      [request.form['test_preparation']]
        })

        predicted_score = round(float(pipeline.predict(input_data)[0]), 2)
        grade           = get_grade(predicted_score)

        log.info(f"Prediction result → score={predicted_score}, grade={grade}")

        return render_template('index.html',
            prediction         = predicted_score,
            grade              = grade,
            error              = None,
            gender             = request.form['gender'],
            ethnicity          = request.form['ethnicity'],
            parental_education = request.form['parental_education'],
            lunch              = request.form['lunch'],
            test_preparation   = request.form['test_preparation']
        )

    except Exception as e:
        log.error(f"Prediction failed: {e}", exc_info=True)
        return render_template('index.html', prediction=None, error=str(e))


if __name__ == '__main__':
    log.info("Starting Flask app...")
    app.run(debug=True)