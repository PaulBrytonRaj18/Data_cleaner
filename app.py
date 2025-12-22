import os
from flask import Flask, render_template, request, redirect, url_for, flash, send_file
from werkzeug.utils import secure_filename
from data_engine import DataEngine
import numpy as np

app = Flask(__name__)
app.secret_key = 'super_secret_key_ml_app'

# --- PATH FIX: Create uploads directly in the current working directory ---
BASE_DIR = os.getcwd()
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize Engine
engine = DataEngine(app.config['UPLOAD_FOLDER'])


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files: return redirect(request.url)
        file = request.files['file']
        if file.filename == '' or not file.filename.endswith('.csv'):
            flash('Please upload a valid CSV file.', 'warning')
            return redirect(request.url)

        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        success, msg = engine.load_data(filename)
        if success:
            flash('Dataset loaded successfully!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash(f'Error: {msg}', 'danger')

    return render_template('base.html')


@app.route('/dashboard')
def dashboard():
    if engine.df is None: return redirect(url_for('index'))
    return render_template('dashboard.html', summary=engine.get_summary(),
                           preview=engine.df.head().to_html(classes='table table-striped', index=False))


@app.route('/cleaning', methods=['GET', 'POST'])
def cleaning():
    if engine.df is None: return redirect(url_for('index'))
    if request.method == 'POST':
        action = request.form.get('action')
        target = request.form.get('target')
        success, msg = engine.clean_data(action, target_col=target)
        flash(msg, 'success' if success else 'danger')
        return redirect(url_for('cleaning'))

    return render_template('cleaning.html', columns=engine.df.columns.tolist())


@app.route('/transform', methods=['GET', 'POST'])
def transform():
    if engine.df is None: return redirect(url_for('index'))

    unique_values = []
    selected_col_for_map = None

    if request.method == 'POST':
        action = request.form.get('action')

        # --- 1. RENAME ---
        if action == 'rename':
            success, msg = engine.rename_column(request.form.get('old_name'), request.form.get('new_name'))
            flash(msg, 'success' if success else 'danger')

        # --- 2. NEW: ENCODING ---
        elif action == 'apply_encoding':
            col = request.form.get('target_col')
            method = request.form.get('method')
            success, msg = engine.apply_encoding(col, method)
            flash(msg, 'success' if success else 'danger')

        # --- 3. MAPPING (FETCH) ---
        elif action == 'fetch_values':
            selected_col_for_map = request.form.get('target_col')
            unique_values = engine.get_column_unique_values(selected_col_for_map)

        # --- 4. MAPPING (APPLY) ---
        elif action == 'apply_mapping':
            target_col = request.form.get('target_col')
            mapping_dict = {}
            for key, value in request.form.items():
                if key.startswith('map_origin_') and value.strip() != '':
                    mapping_dict[key.replace('map_origin_', '')] = value
            success, msg = engine.map_column_values(target_col, mapping_dict)
            flash(msg, 'success' if success else 'danger')
            return redirect(url_for('transform'))

    return render_template('transform.html', columns=engine.df.columns.tolist(),
                           selected_col_for_map=selected_col_for_map,
                           unique_values=unique_values,
                           applied_mappings=engine.applied_mappings)


@app.route('/visualize', methods=['GET', 'POST'])
def visualize():
    if engine.df is None: return redirect(url_for('index'))
    plot_content, is_interactive = None, False

    if request.method == 'POST':
        content, msg, is_html = engine.visualize(
            request.form.get('plot_type'), request.form.get('x_col'),
            request.form.get('y_col'), request.form.get('z_col'),
            request.form.get('color_col'), request.form.get('theme')
        )
        if content:
            plot_content, is_interactive = content, is_html
        else:
            flash(msg, 'danger')

    return render_template('visualize.html', columns=engine.df.columns.tolist(),
                           numeric_columns=engine.df.select_dtypes(include=np.number).columns.tolist(),
                           plot_content=plot_content, is_interactive=is_interactive)


@app.route('/download')
def download():
    if engine.df is None: return redirect(url_for('index'))
    new_name = f"clean_{engine.current_file}"
    if engine.save_data(new_name):
        return send_file(os.path.join(app.config['UPLOAD_FOLDER'], new_name), as_attachment=True)
    return redirect(url_for('dashboard'))


if __name__ == '__main__':
    app.run(debug=True)