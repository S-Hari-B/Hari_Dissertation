import pandas as pd
from dotenv import load_dotenv
load_dotenv()

from unified_core import build_panel, rank_companies, ticker_to_name

def ask_tickers() -> list[str]:
    """Prompt the user to select tickers from the predefined universe."""
    universe = sorted(ticker_to_name.items()) # list of (ticker, name)

    print("\nAvailable companies:")
    for tk, nm in universe:
        print(f"  {tk:<5}  {nm}") # e.g. "TSLA   Tesla"

    selected = input(
        "\nType the tickers you want (space-separated) → "
    ).upper().split()

    valid = [t for t in selected if t in ticker_to_name]
    if not valid:
        print("No valid tickers entered. Exiting.")
    return valid

def main():
    tickers = ask_tickers()
    if not tickers:
        return

    # Can expose these as inputs later
    lookback_days = 7
    top_n_headlines = 5

    print("\nBuilding signal panel …")
    panel = build_panel(tickers, lookback=lookback_days, top_n=top_n_headlines)
    if panel.empty:
        print("No complete data for the chosen tickers.")
        return

    print("Calling GPT for final ranking …")
    ranking = rank_companies(panel, model="gpt-4o")

    pd.set_option("display.max_colwidth", None)
    df = pd.DataFrame(ranking).sort_values("rank")
    print("\nFinal ranking\n")
    print(df.to_markdown(index=False))

if __name__ == "__main__":
    main()
