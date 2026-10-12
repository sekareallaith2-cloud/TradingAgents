from enum import StrEnum


class AnalystType(StrEnum):
    MARKET = "market"
    # Wire value stays "social" for saved-config and string-keyed-caller
    # back-compat; the user-facing label is "Sentiment Analyst".
    SOCIAL = "social"
    NEWS = "news"
    FUNDAMENTALS = "fundamentals"
    # Precious metals only: real yields, the dollar and inflation expectations.
    MACRO = "macro"


class AssetType(StrEnum):
    STOCK = "stock"
    CRYPTO = "crypto"
    METAL = "metal"
