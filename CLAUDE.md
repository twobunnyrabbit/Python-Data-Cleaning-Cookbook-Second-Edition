# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is the code repository for **Python Data Cleaning Cookbook - Second Edition** by Michael Walker, published by Packt. The repository contains recipe-style examples demonstrating data cleaning, wrangling, and preparation techniques using pandas, NumPy, Matplotlib, scikit-learn, and OpenAI.

## Project Structure

The repository is organized into 12 numbered chapters, each corresponding to a chapter in the book:

1. **ImportingTabularData** - CSV, Excel, SQL, SPSS, and R data imports with pandas
2. **ImportingJSONHTMLData** - JSON, web scraping, Spark data, data versioning
3. **TakingMeasureOfData** - Initial data exploration, descriptive statistics, column/row selection
4. **OutliersMultivariate** - Outlier detection in subsets of data
5. **Visualization** - Histograms, boxplots, violin plots, scatter plots, line plots, heat maps
6. **SeriesOperations** - pandas Series operations for data cleaning
7. **IdentifyingandFixingMissingValues** - Missing value identification, cleaning, imputation (regression, KNN, RF, AI)
8. **PreprocessingFeatures** - Train/test splits, encoding, scaling, binning, feature transforms
9. **Aggregating** - Data aggregation techniques
10. **CombiningDataFrames** - Merging and combining DataFrames
11. **TidyingReshaping** - Data tidying and reshaping operations
12. **FunctionsClasses** - Automation with user-defined functions, classes, and scikit-learn pipelines

### Chapter 12 Structure

Chapter 12 contains reusable helper modules in the `helperfunctions/` directory:
- `respondent.py` - Class for respondent data processing
- `basicdescriptives.py` - Basic descriptive statistics functions
- `collectionitem.py` - Collection item processing
- `combineagg.py` - Combining and aggregation utilities
- `outliers.py` - Outlier detection functions
- `preprocfunc.py` - Preprocessing functions
- `runchecks.py` - Data validation checks

Scripts in Chapter 12 demonstrate importing these helpers using:
```python
import sys
sys.path.append(os.getcwd() + "/helperfunctions")
```

## Development Environment

**Python Version**: 3.12+

**Package Management**: Uses `uv` (fast Python package installer and resolver)

**Key Dependencies** (from `pyproject.toml`):
- marimo[lsp] (>=0.16.5) - Interactive notebook environment with language server support
- watchdog (>=6.0.0) - File system event monitoring

**Full dependencies** are managed in `uv.lock`

## Common Commands

### Environment Setup
```bash
# Install dependencies using uv
uv sync

# Add a new package
uv add package-name

# Add package with extras
uv add "package-name[extra]"

# Activate virtual environment
source .venv/bin/activate  # Unix/macOS
```

### Running Scripts

**Traditional Python Scripts:**
Each Python script in the numbered chapter directories is standalone and can be run directly:
```bash
python "1. ImportingTabularData/1. importing_csv.py"
python "12. FunctionsClasses/5. class_cleaning.py"
```

Scripts typically:
- Load data from a `data/` subdirectory within their chapter folder
- Configure pandas display options at the top
- Print output to console (not write to files)

**Marimo Notebooks:**
The repository now includes interactive marimo notebooks (e.g., `ch_01.py`) for exploratory data cleaning:

```bash
# Run a marimo notebook in edit mode (opens in browser)
marimo edit ch_01.py

# Run a marimo notebook as an app (read-only)
marimo run ch_01.py

# Convert traditional script to marimo notebook
marimo convert script.py -o notebook.py
```

### Interactive Development

This repository is designed for interactive exploration. Two approaches are used:

1. **Traditional scripts** - Numbered files in chapter directories contain inline examples that print results
2. **Marimo notebooks** - Interactive, reactive notebooks (`.py` files using marimo's cell structure) for live data exploration

## Code Patterns and Conventions

### Standard pandas Configuration

Most scripts start with these display settings:
```python
import pandas as pd
pd.set_option('display.width', 150)
pd.set_option('display.max_columns', 15)
pd.set_option('display.max_rows', 100)
pd.options.display.float_format = '{:,.2f}'.format
```

### Data File Locations

- Each chapter has its own `data/` subdirectory
- Data files include: CSV, Excel, JSON, SPSS, SQL databases
- Relative paths used: `pd.read_csv("data/filename.csv")`

### Scikit-learn Pipelines

Chapter 8 and 12 demonstrate sklearn pipeline patterns:
```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression

pipeline = make_pipeline(
    StandardScaler(),
    SimpleImputer(strategy="mean"),
    LinearRegression()
)
```

### Marimo Notebook Structure

Marimo notebooks are Python files with a special cell-based structure:

```python
import marimo

__generated_with = "0.16.5"
app = marimo.App(width="medium")

@app.cell
def _():
    import marimo as mo
    import pandas as pd
    return (pd,)  # Return variables to make them available to other cells

@app.cell
def _(pd):
    df = pd.read_csv('data/file.csv')
    return (df,)  # Must return tuple of variables

if __name__ == "__main__":
    app.run()
```

Key marimo patterns:
- Each `@app.cell` function is a reactive cell
- Variables must be explicitly returned as tuples: `return (var1, var2,)`
- Cell function parameters indicate dependencies on other cells
- Cells automatically re-execute when dependencies change

### pandas Best Practices

**Prefer assignment over inplace:**
```python
# Recommended
df = df.drop(columns=['col1', 'col2'])
df = df.rename(columns={'old': 'new'})

# Avoid (no performance benefit, prevents chaining)
df.drop(columns=['col1'], inplace=True)
```

**Common data cleaning patterns:**
```python
# Combine date columns
df['date'] = pd.to_datetime(df[['year', 'month']].assign(day=1))

# Drop original columns
df = df.drop(columns=['year', 'month'])

# Rename columns
df = df.rename(columns={'old_name': 'new_name'})

# Handle missing values
df = df.dropna(subset=['important_column'])
```

### Class-based Data Processing

Chapter 12 demonstrates object-oriented approaches to data cleaning:
- Classes encapsulate data validation and transformation logic
- Class attributes track instance counts
- Methods perform specific data transformations (e.g., calculating averages, ages)

## Working with This Repository

### Understanding Recipe Scripts

Each numbered Python file is a self-contained recipe demonstrating specific techniques. They are:
- **Educational examples**, not production code
- **Sequential** - meant to be read and executed line by line
- **Console-oriented** - use `pprint` and `print()` for output

### Modifying or Extending Examples

When adapting recipes:
1. Check the corresponding chapter's `data/` directory for required datasets
2. Scripts may depend on specific data file structures - check data loading code
3. Some Chapter 12 scripts import from `helperfunctions/` - ensure paths resolve correctly
4. Display settings may need adjustment for your terminal width

### Data Dependencies

Scripts assume data files exist in their chapter's `data/` directory. Missing data will cause import errors. The repository includes sample datasets for all examples.

## OpenAI Integration

Some scripts (e.g., `3. TakingMeasureOfData/6. open_ai.py`, `7. IdentifyingandFixingMissingValues/6. impute_missings_ai.py`) use OpenAI APIs for data analysis tasks. These may require:
- OpenAI API keys (not included in repository)
- Additional configuration for PandasAI or similar libraries
