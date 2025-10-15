import marimo

__generated_with = "0.16.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    return mo, pd


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
def _(mo):
    mo.md(r"""## Importing SQL""")
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
    return (plt,)


@app.cell
def _():
    return


@app.cell
def _(df):
    def _():
        import marimo as mo
        import matplotlib.pyplot as plt
        import pandas as pd
        from datetime import datetime

        # Create slider for date range
        date_slider = mo.ui.range_slider(
            start=df['measured_date'].min().timestamp(),
            stop=df['measured_date'].max().timestamp(),
            value=(df['measured_date'].min().timestamp(), df['measured_date'].max().timestamp()),
            label="Date Range",
            step=86400  # 1 day in seconds
        )

        # Interactive scatter plot function
        def create_interactive_plot(start_date, end_date):
            # Convert timestamp back to datetime
            start_date = datetime.fromtimestamp(start_date)
            end_date = datetime.fromtimestamp(end_date)

            # Filter data by date range
            filtered_df = df[(df['measured_date'] >= start_date) & (df['measured_date'] <= end_date)]

            # Create plot
            plt.figure(figsize=(12, 8))

            # Scatter plot with color representing temperature
            scatter = plt.scatter(filtered_df['longitude'], filtered_df['latitude'], 
                                c=filtered_df['temp'], 
                                cmap='coolwarm', 
                                alpha=0.7,
                                s=50,
                                vmin=df['temp'].min(),
                                vmax=df['temp'].max())

            # Add colorbar
            cbar = plt.colorbar(scatter)
            cbar.set_label('Temperature')

            # Add labels and title
            plt.xlabel('Longitude')
            plt.ylabel('Latitude')
            plt.title(f'Temperature Distribution by Geographic Location\n({start_date.strftime("%Y-%m-%d")} to {end_date.strftime("%Y-%m-%d")})')

            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            return plt.gca()

        # Create interactive plot that updates when slider changes
        interactive_plot = mo.ui.interactive(
            create_interactive_plot,
            start_date=(df['measured_date'].min().timestamp(), df['measured_date'].min().timestamp()),
            end_date=(df['measured_date'].max().timestamp(), df['measured_date'].max().timestamp())
        )

        # Display UI elements and plot
        return mo.vstack([date_slider, interactive_plot])


    _()
    return


@app.cell
def _(df, plt):
    from datetime import datetime
    # Interactive scatter plot function
    def create_interactive_plot(start_date, end_date):
        # Convert timestamp back to datetime
        # start_date = datetime.fromtimestamp(start_date)
        # end_date = datetime.fromtimestamp(end_date)

        # Filter data by date range
        filtered_df = df[(df['measured_date'] >= start_date) & (df['measured_date'] <= end_date)]

        # Create plot
        plt.figure(figsize=(12, 8))

        # Scatter plot with color representing temperature
        scatter = plt.scatter(filtered_df['longitude'], filtered_df['latitude'], 
                            c=filtered_df['temp'], 
                            cmap='coolwarm', 
                            alpha=0.7,
                            s=50,
                            vmin=df['temp'].min(),
                            vmax=df['temp'].max())

        # Add colorbar
        cbar = plt.colorbar(scatter)
        cbar.set_label('Temperature')

        # Add labels and title
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.title(f'Temperature Distribution by Geographic Location\n({start_date.strftime("%Y-%m-%d")} to {end_date.strftime("%Y-%m-%d")})')

        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        return plt.gca()
    return (create_interactive_plot,)


@app.cell
def _(create_interactive_plot, df):
    create_interactive_plot(df['measured_date'][1], df['measured_date'][1])
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
    return (df_student_por,)


@app.cell
def _(df_student_por):
    df_student_por
    return


if __name__ == "__main__":
    app.run()
