import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('Agg')  # Use non-interactive backend for server
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import os
import plotly.express as px
import plotly.io as pio


class DataEngine:
    def __init__(self, upload_folder='upload'):
        self.upload_folder = upload_folder
        self.current_file = None
        self.df = None

    def load_data(self, filename):
        """Loads CSV data into a Pandas DataFrame."""
        try:
            filepath = os.path.join(self.upload_folder, filename)
            self.df = pd.read_csv(filepath)
            self.current_file = filename
            return True, "Data loaded successfully."
        except Exception as e:
            return False, str(e)

    def save_data(self, filename=None):
        """Saves the current DataFrame to CSV."""
        if self.df is None:
            return False, "No data to save."

        if filename:
            save_path = os.path.join(self.upload_folder, filename)
        else:
            save_path = os.path.join(self.upload_folder, self.current_file)

        try:
            self.df.to_csv(save_path, index=False)
            return True, f"File saved to {save_path}"
        except Exception as e:
            return False, str(e)

    def get_summary(self):
        """Returns comprehensive statistics about the dataset."""
        if self.df is None:
            return None

        rows, cols = self.df.shape
        memory_usage = self.df.memory_usage(deep=True).sum() / 1024 ** 2

        column_stats = []
        for col in self.df.columns:
            dtype = str(self.df[col].dtype)
            missing = self.df[col].isnull().sum()
            missing_pct = (missing / rows) * 100
            unique = self.df[col].nunique()

            stat = {
                'name': col,
                'dtype': dtype,
                'missing': int(missing),
                'missing_pct': round(missing_pct, 2),
                'unique': unique,
                'sample': str(self.df[col].iloc[0]) if rows > 0 else 'N/A'
            }
            column_stats.append(stat)

        desc = self.df.describe().to_dict()

        return {
            'rows': rows,
            'cols': cols,
            'memory_mb': round(memory_usage, 2),
            'columns': column_stats,
            'numerical_summary': desc
        }

    def clean_data(self, strategy, target_col=None, fill_value=0):
        if self.df is None:
            return False, "No data loaded."

        try:
            if strategy == 'drop_na_rows':
                prev = len(self.df)
                self.df.dropna(inplace=True)
                return True, f"Removed {prev - len(self.df)} rows."
            elif strategy == 'drop_na_cols':
                prev = len(self.df.columns)
                self.df.dropna(axis=1, inplace=True)
                return True, f"Removed {prev - len(self.df.columns)} columns."
            elif strategy == 'drop_col':
                if target_col in self.df.columns:
                    self.df.drop(columns=[target_col], inplace=True)
                    return True, f"Column '{target_col}' removed."
            elif strategy == 'fill_mean':
                if target_col == 'ALL_NUMERIC':
                    cols = self.df.select_dtypes(include=np.number).columns
                    self.df[cols] = self.df[cols].fillna(self.df[cols].mean())
                    return True, "Filled numeric missing values with mean."
                elif target_col in self.df.columns:
                    self.df[target_col].fillna(self.df[target_col].mean(), inplace=True)
                    return True, f"Filled '{target_col}' with mean."
            elif strategy == 'fill_zero':
                if target_col == 'ALL_NUMERIC':
                    cols = self.df.select_dtypes(include=np.number).columns
                    self.df[cols] = self.df[cols].fillna(0)
                    return True, "Filled numeric missing values with 0."
                elif target_col in self.df.columns:
                    self.df[target_col].fillna(0, inplace=True)
                    return True, f"Filled '{target_col}' with 0."
            elif strategy == 'fill_mode':
                if target_col in self.df.columns:
                    self.df[target_col].fillna(self.df[target_col].mode()[0], inplace=True)
                    return True, f"Filled '{target_col}' with mode."

            return False, "Action failed or unknown strategy."
        except Exception as e:
            return False, f"Error: {str(e)}"

    # --- NEW METHODS FOR IMPROVEMENTS ---

    def rename_column(self, old_name, new_name):
        """Renames a specific column."""
        if self.df is None: return False, "No data."
        if old_name not in self.df.columns: return False, "Column not found."

        try:
            self.df.rename(columns={old_name: new_name}, inplace=True)
            return True, f"Renamed '{old_name}' to '{new_name}'"
        except Exception as e:
            return False, str(e)

    def get_column_unique_values(self, col, limit=50):
        """Returns unique values for mapping previews."""
        if self.df is None or col not in self.df.columns:
            return []

        # Get unique values, sort them, and convert to string for display
        uniques = self.df[col].dropna().unique()
        # Limit to avoid crashing browser with too many options
        if len(uniques) > limit:
            return []  # Indicate too many values

        return sorted([str(x) for x in uniques])

    def map_column_values(self, col, mapping_dict):
        """Maps values in a column to new values (usually numeric)."""
        if self.df is None or col not in self.df.columns:
            return False, "Invalid column."

        try:
            # Create a backup for safety (optional, but good practice in memory)
            # Apply mapping. Using .map() matches keys to values.
            # Values not in dict become NaN, so we fillna with original or handle carefully.
            # Here we assume user maps all relevant values.

            # Convert column to string temporarily to match mapping keys coming from HTML forms
            temp_series = self.df[col].astype(str)
            mapped_series = temp_series.map(mapping_dict)

            # If mapping creates NaNs (values not in map), decide strategy.
            # For now, we only update rows that were mapped.
            self.df[col] = mapped_series.fillna(self.df[col])

            # Attempt to convert to numeric if possible after mapping
            self.df[col] = pd.to_numeric(self.df[col], errors='ignore')

            return True, f"Mapped values in '{col}' successfully."
        except Exception as e:
            return False, f"Mapping error: {str(e)}"

    # ------------------------------------

    def visualize(self, plot_type, x_col, y_col=None, z_col=None, color_col=None,
                  title="Analysis", width=10, height=6, theme='viridis'):
        """
        Generates plots.
        Returns tuple: (content_string, message, is_interactive_boolean)
        """
        if self.df is None:
            return None, "No data loaded.", False

        # --- OPTION 1: INTERACTIVE 3D PLOT (PLOTLY) ---
        if plot_type == '3d_scatter':
            try:
                if not x_col or not y_col or not z_col:
                    return None, "X, Y, and Z columns required for 3D Scatter", False

                fig = px.scatter_3d(
                    self.df, x=x_col, y=y_col, z=z_col,
                    color=color_col if color_col else None,
                    title=f"3D Analysis: {x_col}, {y_col}, {z_col}",
                    template="plotly_white",
                    color_continuous_scale=theme
                )

                # Make it responsive
                fig.update_layout(autosize=True, margin=dict(l=0, r=0, b=0, t=40))

                plot_html = pio.to_html(fig, full_html=False, include_plotlyjs='cdn')
                return plot_html, "3D Plot Generated (Interactive)", True

            except Exception as e:
                return None, f"3D Plot Error: {str(e)}", False

        # --- OPTION 2: STATIC 2D PLOTS (MATPLOTLIB/SEABORN) ---
        plt.figure(figsize=(width, height))
        sns.set_theme(style="whitegrid")

        try:
            ax = None
            if plot_type == 'histogram':
                if not x_col: return None, "X Column required", False
                ax = sns.histplot(data=self.df, x=x_col, kde=True, hue=color_col if color_col else None,
                                  palette=theme if color_col else None)
                plt.title(f"Distribution of {x_col}")

            elif plot_type == 'scatter':
                if not x_col or not y_col: return None, "X and Y required", False
                ax = sns.scatterplot(data=self.df, x=x_col, y=y_col, hue=color_col if color_col else None,
                                     palette=theme if color_col else None)
                plt.title(f"{x_col} vs {y_col}")

            elif plot_type == 'line':
                if not x_col or not y_col: return None, "X and Y required", False
                ax = sns.lineplot(data=self.df, x=x_col, y=y_col, hue=color_col if color_col else None,
                                  palette=theme if color_col else None)
                plt.title(f"Trend of {y_col} over {x_col}")

            elif plot_type == 'bar':
                if not x_col or not y_col: return None, "X and Y required", False
                top_data = self.df.head(20)
                ax = sns.barplot(data=top_data, x=x_col, y=y_col, hue=color_col if color_col else None,
                                 palette=theme if color_col else None)
                plt.xticks(rotation=45)
                plt.title(f"{y_col} by {x_col} (First 20 rows)")

            elif plot_type == 'box':
                if not x_col: return None, "X Column required", False
                ax = sns.boxplot(data=self.df, x=x_col, y=y_col if y_col else None,
                                 hue=color_col if color_col else None, palette=theme if color_col else None)
                plt.title(f"Box Plot of {x_col}")

            elif plot_type == 'heatmap':
                numeric_df = self.df.select_dtypes(include=[np.number])
                if numeric_df.empty: return None, "No numeric data", False
                plt.figure(figsize=(12, 10))
                sns.heatmap(numeric_df.corr(), annot=True, cmap=theme, fmt=".2f")
                plt.title("Correlation Matrix")

            if color_col and plot_type != 'heatmap':
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

            plt.tight_layout()

            img = io.BytesIO()
            plt.savefig(img, format='png', bbox_inches='tight')
            img.seek(0)
            plt.close()
            return base64.b64encode(img.getvalue()).decode(), "Plot Generated", False

        except Exception as e:
            plt.close()
            return None, f"Visualization Error: {str(e)}", False