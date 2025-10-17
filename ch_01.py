import marimo

__generated_with = "0.16.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import os

    from dotenv import load_dotenv
    return load_dotenv, mo, pd


@app.cell
def _(load_dotenv):
    load_dotenv()
    # os.getenv('OLLAMA_API_KEY')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Import CSV file""")
    return


@app.cell
def _(pd):
    df = pd.read_csv('1. ImportingTabularData/data/landtempssample.csv')
    # Create year_month column from year and month columns
    df['year_month'] = pd.to_datetime(df[['year', 'month']].assign(day=1))
    df = df.drop(columns=['year', 'month'])
    df.rename(columns = {'year_month' : 'measured_date'}, inplace=True)
    return (df,)


@app.cell
def _(df):
    df
    return


@app.cell
def _(df):
    df.isna().sum()
    return


@app.cell
def _(df):
    # remove missing temps
    df.dropna(subset=['temp'], inplace=True)
    return


@app.cell
def _(df, pd):
    df_num_obs_country = pd.DataFrame(
        {
            'n': df['country'].value_counts(),
            'prop': df['country'].value_counts(normalize=True).values
        }
    )
    df_num_obs_country
    return (df_num_obs_country,)


@app.cell
def _(df, df_num_obs_country):
    # List the top five countries with the most observations
    top_five_countries = df_num_obs_country[:5].index.to_list()
    # top_five_countries
    is_in_top_five = df['country'].isin(top_five_countries)
    # is_in_top_five[:10]
    df_top_five = df[is_in_top_five]
    df_top_five.shape
    return (df_top_five,)


@app.cell
def _(df_top_five):
    df_gb_country_station = df_top_five.groupby(['country'])['station'].value_counts()
    df_gb_country_station
    return (df_gb_country_station,)


@app.cell
def _(df_gb_country_station):
    df_gb_country_station.index
    return


@app.cell
def _(df_gb_country_station):
    # Get top 5 stations for each country
    top_5_stations_per_country = df_gb_country_station.groupby(level=0).head(5)
    top_5_stations_per_country
    return (top_5_stations_per_country,)


@app.cell
def _(top_5_stations_per_country):
    top_5_stations_per_country.reset_index()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Import Excel file""")
    return


@app.cell(hide_code=True)
def _(pd):
    df_xl = pd.read_excel(
        '1. ImportingTabularData/data/GDPpercapita22b.xlsx', 
        sheet_name="OECD.Stat export", 
        skiprows=4, 
        skipfooter=1,
        usecols="A,C:W"

    )
    return (df_xl,)


@app.cell
def _(df_xl):
    df_xl
    return


@app.cell
def _(df_xl):
    df_xl.columns
    return


@app.cell
def _(df_xl):
    # rename column Year to metro
    df_xl.rename(columns={'Year': 'metro'}, inplace=True)
    return


@app.cell
def _(df_xl):
    starts_with_space = df_xl.metro.str.startswith(' ')
    starts_with_space
    return


@app.cell
def _(df_xl):
    df_xl['metro'] = df_xl['metro'].str.strip()
    return


@app.cell
def _(df_xl, pd):
    # convert all columsn to numeric values
    for col in df_xl.columns[1:]:
        df_xl[col] = pd.to_numeric(df_xl[col], errors='coerce')
        df_xl.rename(columns={col:'pcGDP' + col}, inplace=True)
    return


@app.cell
def _(df_xl):
    # drop na columns only if all rows have na
    df_xl.dropna(subset=df_xl.columns[1:], how='all', inplace=True)
    return


@app.cell
def _(df_xl):
    df_xl.shape
    return


@app.cell
def _(df_xl):
    # set index using metro column
    df_xl.set_index('metro', inplace=True)
    return


@app.cell
def _(df_xl):
    original_index = df_xl.index
    return (original_index,)


@app.cell
def _(original_index):
    original_index.to_list()
    return


@app.cell
def _(df):
    df
    return


@app.cell
def _(df):
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Create a scatter plot with temperature as color
    plt.figure(figsize=(12, 8))

    # Scatter plot with color representing temperature
    scatter = plt.scatter(df['longitude'], df['latitude'], 
                        c=df['temp'], 
                        cmap='coolwarm', 
                        alpha=0.7,
                        s=50)

    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Temperature')

    # Add labels and title
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Temperature Distribution by Geographic Location')

    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.gca()  # This displays the plot in marimo
    return


@app.cell
def _(mo):
    mo.md(r"""## Import with SQL""")
    return


@app.cell
def _(pd):
    df_student_math = pd.read_csv('./1. ImportingTabularData/data/student-mat.csv', sep=";")
    return (df_student_math,)


@app.cell
def _(df_student_math):
    df_student_math
    return


@app.cell
def _(pd):
    df_student_por = pd.read_csv('./1. ImportingTabularData/data/student-por.csv', sep=";")
    return


@app.cell
def _(df_student_math):
    is_missing = (df_student_math.isna().sum() > 0).values
    is_missing
    # no missing values in df_student_math
    return


@app.cell
def _(df_student_math):
    cols_original = df_student_math.columns.to_list()
    cols_original
    return (cols_original,)


@app.cell
def _(cols_original):
    # order columns
    last_three_cols = cols_original[-3:]
    time_cols = [x for x in cols_original if x.endswith('time')]
    remaining_cols = [x for x in cols_original if x not in (last_three_cols + time_cols)]
    new_ordercols = last_three_cols + time_cols + remaining_cols
    return (new_ordercols,)


@app.cell
def _(df_student_math, new_ordercols):
    df_student_math_2 = df_student_math[new_ordercols]
    return (df_student_math_2,)


@app.cell
def _(df_student_math_2):
    df_student_math_2
    return


@app.cell
def _(df_student_math_2):
    df_student_math_2['school']
    return


@app.cell
def _(df_student_math, mo):
    # Create a dropdown widget to select a column
    column_selector = mo.ui.dropdown(
        options=df_student_math.columns.tolist(),
        value=df_student_math.columns[0],  # Default to first column
        label="Select Column"
    )

    # Display the dropdown
    column_selector
    return (column_selector,)


@app.cell
def _(column_data, mo, pd, selected_column):
    # len(column_selector.value)
    # df_student_math_2[selected_column].count()
    # pd.api.types.is_numeric_dtype(column_data)

    def get_categorical_stats(col_name):
        summary_stats_1 = pd.DataFrame({
            'Count': [column_data.count()],
            'Unique': [column_data.nunique()],
            'Top': [column_data.mode().iloc[0] if not column_data.mode().empty else None],
            'Freq': [column_data.value_counts().iloc[0] if not column_data.empty else None]
        })
        summary_md_1 = mo.md(f"""
        ## Summary for Column: `{selected_column}`
    
        ### Data Type: `{column_data.dtype}`
    
        ### Statistics:
        {summary_stats_1.to_string()}
    
        ### Value Counts:
        """)
    
        value_counts_1 = pd.DataFrame(column_data.value_counts().head(10))
    
        # Return both markdown and dataframe
        mo.vstack([summary_md_1, value_counts_1])

    return


@app.cell
def _(column_selector, df_student_math, mo, pd):
    # Get the selected column and display summary statistics
    selected_column = column_selector.value
    column_data = df_student_math[selected_column]

    # Determine column type and display appropriate summary
    if pd.api.types.is_numeric_dtype(column_data):
        # Numeric summary
        print("Numeric dtype")
        summary_stats = pd.DataFrame({
            'Count': [column_data.count()],
            'Mean': [column_data.mean()],
            'Std': [column_data.std()],
            'Min': [column_data.min()],
            '25%': [column_data.quantile(0.25)],
            '50%': [column_data.median()],
            '75%': [column_data.quantile(0.75)],
            'Max': [column_data.max()]
        })

        # Create the output
        summary_md = mo.md(f"""
        ## Summary for Column: `{selected_column}`

        ### Data Type: `{column_data.dtype}`

        ### Statistics:
        {summary_stats.to_string()}

        ### Sample Values:
        """)

        sample_values = pd.DataFrame(column_data.head(10).reset_index(drop=True))

        # Return both markdown and dataframe
        mo.vstack([summary_md, sample_values])
    else:
        # Categorical summary
        summary_stats = pd.DataFrame({
            'Count': [column_data.count()],
            'Unique': [column_data.nunique()],
            'Top': [column_data.mode().iloc[0] if not column_data.mode().empty else None],
            'Freq': [column_data.value_counts().iloc[0] if not column_data.empty else None]
        })

        # Create the output
        summary_md = mo.md(f"""
        ## Summary for Column: `{selected_column}`

        ### Data Type: `{column_data.dtype}`

        ### Statistics:
        {summary_stats.to_string()}

        ### Value Counts:
        """)

        value_counts = pd.DataFrame(column_data.value_counts().head(10))

        # Return both markdown and dataframe
        mo.vstack([summary_md, value_counts])
    return column_data, selected_column


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
