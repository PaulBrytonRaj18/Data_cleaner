import os
from flask import Flask, render_template, request, redirect, url_for, flash, send_file
from werkzeug.utils import secure_filename
from data_engine import DataEngine

app = Flask(__name__)
app.secret_key = 'enterprise_ml_secret_key_change_in_prod'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max limit

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize Data Engine
engine = DataEngine(app.config['UPLOAD_FOLDER'])


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        if file and file.filename.endswith('.csv'):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            success, msg = engine.load_data(filename)
            if success:
                flash('File uploaded and analyzed successfully!', 'success')
                return redirect(url_for('dashboard'))
            else:
                flash(f'Error loading file: {msg}', 'danger')
        else:
            flash('Only CSV files are allowed.', 'warning')

    return render_template('base.html', page='home')


@app.route('/dashboard')
def dashboard():
    if engine.df is None:
        flash('Please upload a dataset first.', 'info')
        return redirect(url_for('index'))

    summary = engine.get_summary()
    # Get first 5 rows for preview
    preview = engine.df.head().to_html(classes='table table-striped table-hover', index=False)

    return render_template('dashboard.html', summary=summary, preview=preview)


@app.route('/cleaning', methods=['GET', 'POST'])
def cleaning():
    if engine.df is None:
        return redirect(url_for('index'))

    if request.method == 'POST':
        action = request.form.get('action')
        target = request.form.get('target')

        success = False
        msg = ""

        if action == 'drop_rows':
            success, msg = engine.clean_data('drop_na_rows')
        elif action == 'drop_cols':
            success, msg = engine.clean_data('drop_na_cols')
        elif action == 'drop_specific':
            success, msg = engine.clean_data('drop_col', target_col=target)
        elif action == 'fill_mean':
            success, msg = engine.clean_data('fill_mean', target_col=target)
        elif action == 'fill_zero':
            success, msg = engine.clean_data('fill_zero', target_col=target)
        elif action == 'fill_mode':
            success, msg = engine.clean_data('fill_mode', target_col=target)

        if success:
            flash(msg, 'success')
        else:
            flash(msg, 'danger')

        return redirect(url_for('cleaning'))

    columns = engine.df.columns.tolist()
    numeric_columns = engine.df.select_dtypes(include=['float64', 'int64']).columns.tolist()

    return render_template('cleaning.html', columns=columns, numeric_columns=numeric_columns)


# --- NEW ROUTE FOR TRANSFORMATION (MAPPING & RENAMING) ---
@app.route('/transform', methods=['GET', 'POST'])
def transform():
    if engine.df is None:
        return redirect(url_for('index'))

    columns = engine.df.columns.tolist()
    unique_values = []
    selected_col_for_map = None

    if request.method == 'POST':
        action = request.form.get('action')

        if action == 'rename':
            old_name = request.form.get('old_name')
            new_name = request.form.get('new_name')
            success, msg = engine.rename_column(old_name, new_name)
            if success:
                flash(msg, 'success')
                return redirect(url_for('transform'))
            else:
                flash(msg, 'danger')

        elif action == 'fetch_values':
            # User selected a column to map, show its values
            selected_col_for_map = request.form.get('target_col')
            unique_values = engine.get_column_unique_values(selected_col_for_map)
            if not unique_values:
                flash(f"Could not fetch unique values for {selected_col_for_map} (or too many unique values).",
                      'warning')

        elif action == 'apply_mapping':
            # User submitted the mapping form
            target_col = request.form.get('target_col')

            # Construct mapping dictionary from form data
            mapping_dict = {}
            # We iterate through keys starting with 'map_'
            for key, value in request.form.items():
                if key.startswith('map_origin_'):
                    origin_val = key.replace('map_origin_', '')
                    # If value is empty, we skip or map to 0? Let's assume user input is required.
                    if value.strip() != '':
                        mapping_dict[origin_val] = value

            success, msg = engine.map_column_values(target_col, mapping_dict)
            if success:
                flash(msg, 'success')
                return redirect(url_for('transform'))  # Reset state
            else:
                flash(msg, 'danger')

    return render_template('transform.html',
                           columns=columns,
                           selected_col_for_map=selected_col_for_map,
                           unique_values=unique_values)


@app.route('/visualize', methods=['GET', 'POST'])
def visualize():
    if engine.df is None:
        return redirect(url_for('index'))

    plot_content = None
    is_interactive = False

    columns = engine.df.columns.tolist()
    numeric_columns = engine.df.select_dtypes(include=['float64', 'int64']).columns.tolist()

    if request.method == 'POST':
        plot_type = request.form.get('plot_type')
        x_col = request.form.get('x_col')
        y_col = request.form.get('y_col')
        z_col = request.form.get('z_col')
        color_col = request.form.get('color_col')
        theme = request.form.get('theme')

        if y_col == "None": y_col = None
        if z_col == "None": z_col = None
        if color_col == "None": color_col = None

        content, msg, is_html = engine.visualize(plot_type, x_col, y_col, z_col, color_col, theme=theme)

        if content:
            plot_content = content
            is_interactive = is_html
        else:
            flash(msg, 'danger')

    return render_template('visualize.html',
                           columns=columns,
                           numeric_columns=numeric_columns,
                           plot_content=plot_content,
                           is_interactive=is_interactive)


@app.route('/download')
def download():
    if engine.df is None:
        return redirect(url_for('index'))

    filename = "modified_dataset.csv"
    engine.save_data(filename)
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    return send_file(path, as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True, port=5000)