import pandas as pd
import numpy as np
import matplotlib

# Use Agg backend to prevent GUI errors on servers
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import os
import plotly.express as px
import plotly.io as pio

# Import Encoders
from sklearn.preprocessing import LabelEncoder


class DataEngine:
    def __init__(self, upload_folder):
        # We rely strictly on the passed upload_folder path
        self.upload_folder = upload_folder
        self.current_file = None
        self.df = None
        self.applied_mappings = {}

    def _get_filepath(self, filename):
        return os.path.join(self.upload_folder, filename)

    def load_data(self, filename):
        try:
            filepath = self._get_filepath(filename)
            self.df = pd.read_csv(filepath)
            self.current_file = filename
            self.applied_mappings = {}
            return True, "Data loaded successfully."
        except Exception as e:
            return False, str(e)

    def save_data(self, filename=None):
        if self.df is None: return False, "No data to save."
        if filename:
            save_path = self._get_filepath(filename)
        else:
            save_path = self._get_filepath(self.current_file)
        try:
            self.df.to_csv(save_path, index=False)
            return True, f"File saved to {save_path}"
        except Exception as e:
            return False, f"Error saving: {str(e)}"

    def get_summary(self):
        if self.df is None: return None
        rows, cols = self.df.shape
        memory_usage = self.df.memory_usage(deep=True).sum() / 1024 ** 2

        column_stats = []
        for col in self.df.columns:
            dtype = str(self.df[col].dtype)
            missing = self.df[col].isnull().sum()
            unique = self.df[col].nunique()
            stat = {
                'name': col, 'dtype': dtype, 'missing': int(missing),
                'missing_pct': round((missing / rows) * 100, 2), 'unique': unique,
                'sample': str(self.df[col].iloc[0]) if rows > 0 else 'N/A'
            }
            column_stats.append(stat)

        return {
            'rows': rows, 'cols': cols, 'memory_mb': round(memory_usage, 2),
            'columns': column_stats, 'numerical_summary': self.df.describe().to_dict()
        }

    # --- CLEANING METHODS (FIXED TO MATCH HTML ACTIONS) ---
    def clean_data(self, strategy, target_col=None):
        if self.df is None: return False, "No data loaded."
        try:
            # 1. DROP ROWS (Matches HTML 'drop_rows')
            if strategy == 'drop_rows':
                prev_rows = len(self.df)
                self.df.dropna(inplace=True)
                return True, f"Dropped {prev_rows - len(self.df)} rows with missing values."

            # 2. DROP COLUMNS (Matches HTML 'drop_cols')
            elif strategy == 'drop_cols':
                prev_cols = len(self.df.columns)
                self.df.dropna(axis=1, inplace=True)
                return True, f"Dropped {prev_cols - len(self.df.columns)} columns with missing values."

            # 3. DROP SPECIFIC COLUMN (Matches HTML 'drop_specific')
            elif strategy == 'drop_specific':
                if target_col and target_col in self.df.columns:
                    self.df.drop(columns=[target_col], inplace=True)
                    return True, f"Dropped column '{target_col}'."
                else:
                    return False, f"Column '{target_col}' not found."

            # 4. IMPUTATION STRATEGIES
            elif strategy == 'fill_mean':
                if target_col == 'ALL_NUMERIC':
                    cols = self.df.select_dtypes(include=np.number).columns
                    self.df[cols] = self.df[cols].fillna(self.df[cols].mean())
                elif target_col in self.df.columns:
                    self.df[target_col].fillna(self.df[target_col].mean(), inplace=True)
                return True, "Filled missing values with Mean."

            elif strategy == 'fill_zero':
                if target_col == 'ALL_NUMERIC':
                    cols = self.df.select_dtypes(include=np.number).columns
                    self.df[cols] = self.df[cols].fillna(0)
                elif target_col in self.df.columns:
                    self.df[target_col].fillna(0, inplace=True)
                return True, "Filled missing values with Zero."

            elif strategy == 'fill_mode':
                if target_col in self.df.columns:
                    self.df[target_col].fillna(self.df[target_col].mode()[0], inplace=True)
                return True, "Filled missing values with Mode."

            return False, f"Unknown strategy: {strategy}"

        except Exception as e:
            return False, f"Error: {str(e)}"

    def rename_column(self, old_name, new_name):
        if self.df is None or old_name not in self.df.columns: return False, "Invalid column."
        self.df.rename(columns={old_name: new_name}, inplace=True)
        return True, f"Renamed {old_name} to {new_name}"

    # --- ADVANCED ENCODING LOGIC ---
    def apply_encoding(self, col, method):
        if self.df is None or col not in self.df.columns:
            return False, "Column not found."

        try:
            if method == 'label_encode':
                le = LabelEncoder()
                # Ensure data is string type for encoding
                temp_col = self.df[col].astype(str)
                self.df[col] = le.fit_transform(temp_col)
                return True, f"Applied Label Encoding to '{col}'."

            elif method == 'one_hot_encode':
                # Use pandas get_dummies
                dummies = pd.get_dummies(self.df[col], prefix=col, drop_first=False).astype(int)
                self.df = pd.concat([self.df, dummies], axis=1)
                self.df.drop(columns=[col], inplace=True)
                return True, f"Applied One-Hot Encoding to '{col}'. Created {dummies.shape[1]} new columns."

            elif method == 'ordinal_encode':
                # Default to factorization (sorted)
                codes, uniques = pd.factorize(self.df[col], sort=True)
                self.df[col] = codes
                return True, f"Applied Ordinal Encoding (sorted) to '{col}'."

            return False, "Unknown encoding method."
        except Exception as e:
            return False, f"Encoding Error: {str(e)}"

    # --- MAPPING LOGIC ---
    def get_column_unique_values(self, col, limit=50):
        if self.df is None or col not in self.df.columns: return []
        uniques = self.df[col].dropna().unique()
        if len(uniques) > limit: return []
        return sorted([str(x) for x in uniques])

    def map_column_values(self, col, mapping_dict):
        if self.df is None or col not in self.df.columns: return False, "Invalid Column"
        try:
            # Convert mapping values
            self.df[col] = self.df[col].astype(str).map(mapping_dict).fillna(self.df[col])
            # Try converting back to numeric
            self.df[col] = pd.to_numeric(self.df[col], errors='ignore')

            if col in self.applied_mappings:
                self.applied_mappings[col].update(mapping_dict)
            else:
                self.applied_mappings[col] = mapping_dict
            return True, f"Mapped values for {col}."
        except Exception as e:
            return False, str(e)

    # --- VISUALIZATION LOGIC ---
    def visualize(self, plot_type, x_col, y_col=None, z_col=None, color_col=None, theme='viridis'):
        if self.df is None: return None, "No data.", False
        try:
            # 3D Plotly
            if plot_type == '3d_scatter':
                if not x_col or not y_col or not z_col:
                    return None, "X, Y, and Z axes are required for 3D plots.", False

                fig = px.scatter_3d(self.df, x=x_col, y=y_col, z=z_col, color=color_col, template='plotly_white',
                                    color_continuous_scale=theme)
                return pio.to_html(fig, full_html=False), "Success", True

            # Matplotlib/Seaborn
            plt.figure(figsize=(10, 6))
            sns.set_theme(style="whitegrid")

            if plot_type == 'histogram':
                sns.histplot(data=self.df, x=x_col, kde=True, hue=color_col, palette=theme if color_col else None)
            elif plot_type == 'scatter':
                sns.scatterplot(data=self.df, x=x_col, y=y_col, hue=color_col, palette=theme if color_col else None)
            elif plot_type == 'line':
                sns.lineplot(data=self.df, x=x_col, y=y_col, hue=color_col, palette=theme if color_col else None)
            elif plot_type == 'bar':
                sns.barplot(data=self.df.head(20), x=x_col, y=y_col, hue=color_col,
                            palette=theme if color_col else None)
            elif plot_type == 'box':
                sns.boxplot(data=self.df, x=x_col, y=y_col, hue=color_col, palette=theme if color_col else None)
            elif plot_type == 'heatmap':
                num = self.df.select_dtypes(include=[np.number])
                sns.heatmap(num.corr(), annot=True, cmap=theme)

            plt.tight_layout()
            img = io.BytesIO()
            plt.savefig(img, format='png')
            img.seek(0)
            plt.close()
            return base64.b64encode(img.getvalue()).decode(), "Success", False
        except Exception as e:
            plt.close()
            return None, str(e), False