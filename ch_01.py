import marimo

__generated_with = "0.16.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    return (pd,)


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
def _(df):
    df['country'].value_counts().pipe()
    return


if __name__ == "__main__":
    app.run()
