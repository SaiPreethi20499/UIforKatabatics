from flask import Flask, render_template, request, redirect, url_for, send_file, jsonify, Response
from werkzeug.utils import secure_filename
import os
import tempfile
import time
import pandas as pd
from meg import preprocess_data, build_kge_matrix, train_meg_model, generate_synthetic_data, save_synthetic_data

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = tempfile.mkdtemp()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/services')
def services():
    return render_template('services.html')

@app.route('/Contact')
def Contact():
    return render_template('Contact.html')

@app.route('/glanblr')
def glanblr():
    return render_template('models/glanblr.html')

@app.route('/ctgan')
def ctgan():
    return render_template('models/CTGAN.html')

@app.route('/meg')
def meg():
    return render_template('models/meg.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part', 'status': 'error'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file', 'status': 'error'}), 400

    if not file.filename.lower().endswith('.csv'):
        return jsonify({'error': 'Invalid file type. Only CSV files are allowed', 'status': 'error'}), 400

    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Read CSV with error handling
        try:
            df = pd.read_csv(filepath)
        except Exception as e:
            return jsonify({'error': f'Error reading CSV file: {str(e)}', 'status': 'error'}), 400
        
        if len(df) < 10:
            return jsonify({'error': 'Dataset too small (min 10 rows required)', 'status': 'error'}), 400
        
        rows_to_process = request.form.get('num_rows', type=int, default=None)
        if rows_to_process:
            df = df.head(rows_to_process)

        def generate_events():
            try:
                start_time = time.time()
                
                # Step 1: Preprocess data
                yield "data: Starting data preprocessing...\n\n"
                X_scaled, y, cat_cols, df_full, scaler, encoders = preprocess_data(df.copy())
                target_col = df.columns[-1]
                yield f"data: Preprocessing completed in {time.time() - start_time:.1f} seconds\n\n"
                
                # Step 2: Build KGE Matrix
                yield "data: Building KGE matrix...\n\n"
                kge_start = time.time()
                kge_matrix = build_kge_matrix(df_full, cat_cols)
                yield f"data: KGE Matrix built in {time.time() - kge_start:.1f} seconds\n\n"
                
                # Step 3: Train MEG Model
                yield "data: Training MEG model...\n\n"
                train_start = time.time()
                meg_model = train_meg_model(
                    X_scaled, 
                    kge_matrix, 
                    input_dim=X_scaled.shape[1], 
                    kge_dim=kge_matrix.shape[1],
                    epochs=15
                )
                yield f"data: Model trained in {time.time() - train_start:.1f} seconds\n\n"
                
                # Step 4: Generate Synthetic Data
                yield "data: Generating synthetic data...\n\n"
                gen_start = time.time()
                num_samples = request.form.get('num_samples', default=len(df), type=int)
                synthetic_data = generate_synthetic_data(meg_model, X_scaled, kge_matrix, num_samples)
                yield f"data: Data generated in {time.time() - gen_start:.1f} seconds\n\n"
                
                # Step 5: Save Results
                yield "data: Saving results...\n\n"
                save_start = time.time()
                output_path = save_synthetic_data(synthetic_data, df_full, scaler, encoders, target_col)
                yield f"data: Results saved in {time.time() - save_start:.1f} seconds\n\n"
                
                total_time = time.time() - start_time
                yield f"data: Generation completed in {total_time:.1f} seconds\n\n"
                yield f"data: DONE|{output_path}\n\n"
                
            except Exception as e:
                yield f"data: ERROR|Error during processing: {str(e)}\n\n"

        return Response(generate_events(), mimetype="text/event-stream")

    except Exception as e:
        return jsonify({'error': f'Error processing file: {str(e)}', 'status': 'error'}), 500

@app.route('/download')
def download():
    try:
        filepath = request.args.get('path')
        if not filepath or not os.path.exists(filepath):
            return jsonify({"error": "File not found"}), 404
            
        return send_file(
            filepath,
            as_attachment=True,
            download_name="synthetic_data.csv",
            mimetype="text/csv"
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)