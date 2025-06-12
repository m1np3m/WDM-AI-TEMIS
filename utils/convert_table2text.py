import pandas as pd
from io import StringIO
from markdown import markdown

def convert_markdown_to_df(markdown_text: str) -> pd.DataFrame:
    try:
        html_table = markdown(markdown_text, extensions=["markdown.extensions.tables"])
        dfs = pd.read_html(StringIO(f"<table>{html_table}</table>"))
        if dfs:
            return dfs[0]
        else:
            print(f"No tables found in {markdown_text}")
            # Return empty DataFrame if no tables found
            return pd.DataFrame()
    except Exception as e:
        # Return empty DataFrame if conversion fails
        return pd.DataFrame()


def convert_table2text(table: str) -> str:
    """
    Convert a table to text help LLM to understand the table.
    """
    df = convert_markdown_to_df(table.strip())
    
    # Handle empty DataFrame
    if df.empty:
        return ""
    
    headers = df.columns.tolist()
    rows = df.values.tolist()
    
    df_string = ""

    for row in rows:
        row_parts = []
        for header, cell in zip(headers, row):
            # Handle NaN values
            cell_value = str(cell) if pd.notna(cell) else ""
            row_parts.append(f"{header}: {cell_value}")
        df_string += ", ".join(row_parts) + "\n"
    
    return df_string.strip()  # Remove trailing newline


if __name__ == "__main__":
    table = """
    | Type        | Competition                           | Titles   | Seasons                                                                                                                                                                              |\n|:------------|:--------------------------------------|:---------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|\n| Domestic    | First Division/Premier League[nb 4]   | 20s      | 1907\u201308, 1910\u201311, 1951\u201352, 1955\u201356, 1956\u201357, 1964\u201365, 1966\u201367, 1992\u201393, 1993\u201394, 1995\u201396, 1996\u201397, 1998\u201399, 1999\u20132000, 2000\u201301, 2002\u201303, 2006\u201307, 2007\u201308, 2008\u201309, 2010\u201311, 2012\u201313 |\n| Domestic    | Second Division[nb 4]                 | 2        | 1935\u201336, 1974\u201375                                                                                                                                                                     |\n| Domestic    | FA Cup                                | 13       | 1908\u201309, 1947\u201348, 1962\u201363, 1976\u201377, 1982\u201383, 1984\u201385, 1989\u201390, 1993\u201394, 1995\u201396, 1998\u201399, 2003\u201304, 2015\u201316, 2023\u201324                                                                  |\n| Domestic    | Football League Cup/EFL Cup           | 6        | 1991\u201392, 2005\u201306, 2008\u201309, 2009\u201310, 2016\u201317, 2022\u201323                                                                                                                                 |\n| Domestic    | FA Charity Shield/FA Community Shield | 21       | 1908, 1911, 1952, 1956, 1957, 1965, 1967, 1977, 1983, 1990, 1993, 1994, 1996, 1997, 2003, 2007, 2008, 2010, 2011, 2013, 2016 (* shared)                                              |\n| Continental | European Cup/UEFA Champions League    | 3        | 1967\u201368, 1998\u201399, 2007\u201308                                                                                                                                                            |\n| Continental | European Cup Winners' Cup             | 1        | 1990\u201391                                                                                                                                                                              |\n| Continental | UEFA Europa League                    | 1        | 2016\u201317                                                                                                                                                                              |\n| Continental | UEFA Super Cup                        | 1        | 1991                                                                                                                                                                                 |\n| Worldwide   | FIFA Club World Cup                   | 1        | 2008                                                                                                                                                                                 |\n| Worldwide   | Intercontinental Cup                  | 1        | 1999                                                                                                                                                                                 |
    """
    
    print(convert_table2text(table))