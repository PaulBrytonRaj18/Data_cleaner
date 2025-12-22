# 📊 EnterpriseML Dashboard: CSV Analysis and Visualization Tool

The EnterpriseML Dashboard is a robust, modular, Python-based web application built using the Flask framework. It provides a comprehensive solution for rapid data analysis, defect cleaning, and interactive visualization of CSV datasets. Designed for data scientists, analysts, and engineering teams, the tool allows users to quickly ingest raw data, identify structural problems, and derive initial insights without requiring complex local environment setup or lengthy scripting.

## ✨ Core Features and Functionality

The application is engineered around three core phases of the data lifecycle: **analysis**, **cleaning**, and **visualization**.

### Data Analysis & Profiling

Instantly initiates a deep scan of the uploaded CSV file. It calculates and presents comprehensive dataset statistics, including total records, number of features, memory consumption, and a per-column breakdown.

- **Missing Values**: Total count and percentage of nulls (`NaN`) for every column.
- **Unique Counts**: Cardinality of each feature to distinguish between continuous, discrete, and categorical data types.
- **Data Types**: Identification of feature types (e.g., `float64`, `int64`, `object`).

### Defect Detection and Reporting

The integrated dashboard provides immediate visibility into data quality issues.

- **Missing Value Flagging**: Columns containing any missing values are prominently highlighted (e.g., color-coded in red) on the dashboard table, allowing analysts to prioritize their cleaning efforts.

### Data Cleaning and Transformation

Offers robust, one-click options to handle defects and prepare the data for modeling:

- **Missing Data Removal**:
  - **Drop rows** where any column contains a null value, resulting in a cleaner but smaller dataset
  - **Drop columns** that contain any missing data, useful when features are too sparse
- **Imputation Strategies**:
  - **Numeric Imputation**: Missing numeric values filled using **Mean (Average)** or **Zero (0)**
  - **Categorical Imputation**: Missing values in object/string columns filled using **Mode** (Most Frequent Value)

### Interactive Visualization

The visualization workbench allows users to explore relationships between features:

- **Distribution Plots**: Histograms and Box Plots for univariate analysis
- **Relational Plots**: Scatter plots and Line charts for bivariate analysis  
- **Structural Plots**: Correlation **Heatmaps** for numeric feature relationships (legend positioned outside plot area)

### Rotatable 3D Plotting (Interactive)

For datasets with three or more numerical features, the application switches to Plotly to generate fully interactive, web-compatible 3D scatter plots. This allows users to click, drag, and rotate the plot in real-time within the browser, providing a much deeper understanding of spatial data clustering and relationships than static images can offer.

### Data Export

After performing cleaning and imputation steps, users have the option to download the modified and corrected dataset as a new CSV file, ensuring persistence and readiness for the next stage of the data pipeline.

To ensure maximum compatibility and avoid environment issues (such as the binary incompatibility errors common with bleeding-edge Python versions), it is highly recommended to use a stable Python environment.

Ensure your environment meets the following requirements before proceeding with installation:

- **Python 3.10 or 3.11**  
  These versions guarantee the availability of pre-compiled binary wheels for the scientific stack, eliminating the need for large C++ build tools.

- **pip**  
  Python package installer


## 🛠️ Installation & Setup

**Prerequisites**:
- Python **3.10** or **3.11** (recommended for binary wheel compatibility)
- `pip` (Python package installer)

Ensure your file structure matches this hierarchy before proceeding.

### 1. Clone the Repository

```bash
git clone https://github.com/rmn2178/Data_cleaner.git
cd EnterpriseML
```

### 2. Set Up a Virtual Environment (Recommended)

**Linux/macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note**: If `requirements.txt` is missing, install core libraries:
> ```bash
> pip install flask pandas numpy matplotlib seaborn plotly openpyxl
> ```

## 🚀 How to Run

### Start the Flask Application

```bash
python app.py
```
*or*

```bash
python main.py
```
*(Replace with actual main Flask app filename - check your project files)*

The application will start on `http://localhost:5000` (or the port shown in terminal).

### Usage Workflow

1. **Upload CSV**: Drag & drop or select your CSV file on the dashboard
2. **Analyze**: View auto-generated statistics and data quality report
3. **Clean**: Apply missing value treatments (drop/fill strategies)
4. **Visualize**: Generate plots (2D/3D interactive)
5. **Export**: Download cleaned CSV

## 📂 Project Structure

```
Data_cleaner/
│
├── app.py                  # Main Flask application controller
├── data_engine.py          # Logic for data processing (Pandas, Scikit-learn)
├── requirements.txt        # List of Python dependencies
│
├── templates/              # HTML files for the frontend
│   ├── base.html           # Main layout template (navbar, footer)
│   ├── dashboard.html      # Overview page (stats, preview)
│   ├── cleaning.html       # Data cleaning interface
│   ├── transform.html      # Feature engineering (encoding/mapping) interface
│   └── visualize.html      # Chart plotting interface
│
└── uploads/                # Directory where user CSV files are stored

```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/new-feature`)
3. Commit changes (`git commit -m 'Add new feature'`)
4. Push to branch (`git push origin feature/new-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

*⭐ Star this repository if it helps you! Made with ❤️ for data enthusiasts by [rmn2178](https://github.com/rmn2178)*
