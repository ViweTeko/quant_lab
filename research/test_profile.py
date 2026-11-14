import polars as pl
import numpy as np

def build_auction_profile(df: pl.DataFrame, tick_size: float = 0.00005) -> pl.DataFrame:
    """
    df schema: ['timestamp', 'price', 'volume']
    Calculates Price-binned Volume Profile and cumulative Value Area (70%).
    """
    # 1. Bin prices to exact 6N contract increments
    profile = (
        df.lazy()
        .with_columns(
            (pl.col("price") / tick_size).round(0).cast(pl.Int64) * tick_size
        )
        .group_by("price")
        .agg(pl.col("volume").sum().alias("volume"))
        .sort("price")
        .collect()
    )

    total_volume = profile["volume"].sum()
    target_va_volume = total_volume * 0.70

    # 2. Identify POC (Highest Volume Node)
    poc_row = profile.sort("volume", descending=True).head(1)
    poc_price = poc_row["price"][0]

    # 3. Mark Value Area around the POC
    # Start at POC and expand outward until 70% of total volume is captured
    sorted_by_vol = profile.sort("volume", descending=True).with_columns(
        pl.col("volume").cum_sum().alias("cum_vol")
    )
    va_df = sorted_by_vol.filter(pl.col("cum_vol") <= target_va_volume)

    vah = va_df["price"].max()
    val = va_df["price"].min()

    print(f"=== Session Auction Summary ===")
    print(f"Total Session Volume: {total_volume:,} contracts")
    print(f"Point of Control (POC): {poc_price:.5f}")
    print(f"Value Area High (VAH):   {vah:.5f}")
    print(f"Value Area Low (VAL):    {val:.5f}")
    return profile

if __name__ == "__main__":
    # Generate deterministic mock auction data around 0.60000
    np.random.seed(42)
    sample_size = 50000
    simulated_prices = np.random.normal(loc=0.60000, scale=0.0020, size=sample_size)
    simulated_volumes = np.random.randint(1, 25, size=sample_size)

    df_sample = pl.DataFrame({
        "timestamp": pl.datetime_range(
            start=pl.datetime(2026, 11, 2, 0, 0),
            end=pl.datetime(2026, 11, 2, 23, 59),
            interval="1s",
            eager=True).head(sample_size),
            "price": simulated_prices,
            "volume": simulated_volumes
    })

    profile = build_auction_profile(df_sample)
