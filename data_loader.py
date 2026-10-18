"""
Data loading and preprocessing module.
Handles CSV ingestion, column detection, cleaning, aggregation,
and merging of the three input datasets.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
import logging

from utils import logger, find_best_column

# Tunisian public holidays
TUNISIAN_HOLIDAYS = {
    "2022-01-01": "New Year",
    "2022-01-14": "Revolution Day",
    "2022-03-20": "Independence Day",
    "2022-04-21": "Eid al-Fitr",
    "2022-04-22": "Eid al-Fitr",
    "2022-07-08": "Eid al-Adha",
    "2022-07-09": "Eid al-Adha",
    "2022-07-25": "Republic Day",
    "2022-08-13": "Women's Day",
    "2022-10-25": "Evacuation Day",
    "2023-01-01": "New Year",
    "2023-01-14": "Revolution Day",
    "2023-03-20": "Independence Day",
    "2023-04-10": "Eid al-Fitr",
    "2023-04-11": "Eid al-Fitr",
    "2023-06-28": "Eid al-Adha",
    "2023-06-29": "Eid al-Adha",
    "2023-07-25": "Republic Day",
    "2023-08-13": "Women's Day",
    "2023-10-25": "Evacuation Day",
    "2024-01-01": "New Year",
    "2024-01-14": "Revolution Day",
    "2024-03-20": "Independence Day",
    "2024-04-10": "Eid al-Fitr",
    "2024-04-11": "Eid al-Fitr",
    "2024-06-16": "Eid al-Adha",
    "2024-06-17": "Eid al-Adha",
    "2024-07-25": "Republic Day",
    "2024-08-13": "Women's Day",
    "2024-10-25": "Evacuation Day",
    "2025-01-01": "New Year",
    "2025-01-14": "Revolution Day",
    "2025-03-20": "Independence Day",
    "2025-03-30": "Eid al-Fitr",
    "2025-03-31": "Eid al-Fitr",
    "2025-06-06": "Eid al-Adha",
    "2025-06-07": "Eid al-Adha",
    "2025-07-25": "Republic Day",
    "2025-08-13": "Women's Day",
    "2025-10-25": "Evacuation Day",
}


class DataLoader:
    """Handles loading, cleaning, and merging of sales data."""

    def __init__(self):
        self.df_commands: Optional[pd.DataFrame] = None
        self.df_stores: Optional[pd.DataFrame] = None
        self.df_products: Optional[pd.DataFrame] = None
        self.df_daily: Optional[pd.DataFrame] = None
        self.df_merged: Optional[pd.DataFrame] = None
        self.df_product_daily: Optional[pd.DataFrame] = None
        self._col_map: Dict[str, str] = {}

    # ------------------------------------------------------------------
    # Column detection helpers
    # ------------------------------------------------------------------
    def _get(self, key: str) -> Optional[str]:
        """Safely get a detected column name."""
        return self._col_map.get(key)

    def _require(self, key: str) -> str:
        """Get a detected column name or raise a clear error."""
        val = self._col_map.get(key)
        if val is None:
            raise ValueError(
                f"Could not detect required column for '{key}'. "
                f"Available columns: {list(self.df_commands.columns) if self.df_commands is not None else 'N/A'}"
            )
        return val

    # ------------------------------------------------------------------
    # Loaders
    # ------------------------------------------------------------------
    def load_commands(self, file_path_or_buf) -> pd.DataFrame:
        """Load and clean the command/order CSV."""
        logger.info("Loading command data...")
        df = pd.read_csv(file_path_or_buf, low_memory=False)

        # Strip BOM if present
        df.columns = df.columns.str.replace("^\ufeff", "", regex=True)
        df.columns = df.columns.str.strip()

        # Detect columns
        self._col_map["date"] = find_best_column(
            df, ["dateCreation", "date", "date_creation", "Date", "createdAt", "created_at"]
        )
        self._col_map["store"] = find_best_column(
            df, ["idPointVente", "idpointvente", "id_point_vente", "pointVenteId", "store_id"]
        )
        self._col_map["amount"] = find_best_column(
            df, ["montant", "amount", "total", "prix", "revenue", "totalAmount"]
        )
        self._col_map["cancelled"] = find_best_column(
            df, ["isAnnule", "isannule", "is_annule", "cancelled", "isCancelled", "annule"]
        )
        self._col_map["product"] = find_best_column(
            df, ["idProduit", "idproduit", "id_produit", "productId", "product_id"]
        )
        self._col_map["quantity"] = find_best_column(
            df, ["quantiteVendue", "quantite", "quantity", "qty", "qte"]
        )

        # ── Parse dates and create standardized "date" column ──
        raw_date_col = self._get("date")
        if raw_date_col is None:
            raise ValueError(
                "No date column found in command file. "
                f"Columns found: {list(df.columns)}"
            )

        # Try multiple date formats explicitly, then fall back to infer
        parsed = pd.to_datetime(df[raw_date_col], errors="coerce")
        null_count = parsed.isna().sum()
        if null_count > len(parsed) * 0.5:
            # Try common formats explicitly
            for fmt in ["%m/%d/%y %H:%M", "%m/%d/%y", "%Y-%m-%d %H:%M:%S",
                         "%Y-%m-%d %H:%M", "%Y-%m-%d", "%d/%m/%Y %H:%M",
                         "%d/%m/%Y", "%m/%d/%Y %H:%M", "%m/%d/%Y"]:
                parsed2 = pd.to_datetime(df[raw_date_col], format=fmt, errors="coerce")
                if parsed2.notna().sum() > parsed.notna().sum():
                    parsed = parsed2
                    logger.info(f"Date format detected: {fmt}")
                    break

        df["date"] = parsed  # ← STANDARDIZED COLUMN CREATED HERE
        df = df.dropna(subset=["date"])
        logger.info(f"Parsed dates, {len(df)} rows with valid dates")

        # ── Filter cancelled orders ──
        cancelled_col = self._get("cancelled")
        if cancelled_col:
            df[cancelled_col] = pd.to_numeric(df[cancelled_col], errors="coerce").fillna(0)
            before = len(df)
            df = df[df[cancelled_col] == 0]
            logger.info(f"Filtered out {before - len(df)} cancelled orders")

        # ── Ensure amount is numeric and non-negative ──
        amount_col = self._get("amount")
        if amount_col:
            df[amount_col] = pd.to_numeric(df[amount_col], errors="coerce").fillna(0)
            df = df[df[amount_col] >= 0]

        self.df_commands = df
        logger.info(f"Loaded {len(df)} valid commands")
        return df

    def load_stores(self, file_path_or_buf) -> pd.DataFrame:
        """Load and clean the point of sale CSV."""
        logger.info("Loading store data...")
        df = pd.read_csv(file_path_or_buf, low_memory=False)
        df.columns = df.columns.str.replace("^\ufeff", "", regex=True)
        df.columns = df.columns.str.strip()

        self._col_map["store_id"] = find_best_column(
            df, ["_id", "id", "idPointVente", "pointVenteId"]
        )
        self._col_map["store_name"] = find_best_column(
            df, ["designation", "name", "nom", "storeName", "libelle"]
        )
        self._col_map["coord_x"] = find_best_column(
            df, ["coordX", "coord_x", "longitude", "lng", "lon"]
        )
        self._col_map["coord_y"] = find_best_column(
            df, ["coordY", "coord_y", "latitude", "lat"]
        )
        self._col_map["address"] = find_best_column(
            df, ["adresse", "address", "addresse", "location"]
        )

        # Numeric coordinates
        for key in ["coord_x", "coord_y"]:
            col = self._get(key)
            if col:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        self.df_stores = df
        logger.info(f"Loaded {len(df)} stores")
        return df

    def load_products(self, file_path_or_buf) -> pd.DataFrame:
        """Load product or product-point-of-sale CSV."""
        logger.info("Loading product data...")
        df = pd.read_csv(file_path_or_buf, low_memory=False)
        df.columns = df.columns.str.replace("^\ufeff", "", regex=True)
        df.columns = df.columns.str.strip()

        self._col_map["product_id"] = find_best_column(
            df, ["idProduit", "idproduit", "id_produit", "productId", "product_id", "_id"]
        )
        self._col_map["product_name"] = find_best_column(
            df, ["designation", "name", "nom", "productName", "libelle"]
        )
        self._col_map["category"] = find_best_column(
            df, ["category", "categorie", "catégorie", "type", "categoryName", "categ"]
        )
        self._col_map["price"] = find_best_column(
            df, ["prix", "price", "prixttc", "unitPrice", "prixVente"]
        )
        self._col_map["product_store"] = find_best_column(
            df, ["idPointVente", "idpointvente", "pointVenteId", "store_id"]
        )

        # Generate name if missing
        if not self._get("product_name") and self._get("product_id"):
            df["designation"] = "Product_" + df[self._get("product_id")].astype(str).str[:8]
            self._col_map["product_name"] = "designation"

        if not self._get("category"):
            df["category"] = "Uncategorized"
            self._col_map["category"] = "category"

        price_col = self._get("price")
        if price_col:
            df[price_col] = pd.to_numeric(df[price_col], errors="coerce").fillna(0)

        self.df_products = df
        logger.info(f"Loaded {len(df)} product records")
        return df

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------
    def aggregate_daily_sales(self) -> pd.DataFrame:
        """Aggregate commands to daily revenue per store."""
        if self.df_commands is None:
            raise ValueError("No command data loaded")

        if "date" not in self.df_commands.columns:
            raise ValueError("Standardized 'date' column missing from commands")

        amount_col = self._require("amount")
        store_col = self._get("store")

        # Normalize to date-only (strip time)
        df = self.df_commands.copy()
        df["date_day"] = df["date"].dt.normalize()

        # ── Per-store daily aggregation ──
        group_cols = ["date_day"]
        if store_col and store_col in df.columns:
            group_cols.append(store_col)

        df_daily = (
            df.groupby(group_cols, as_index=False)
            .agg(
                daily_revenue=(amount_col, "sum"),
                order_count=(amount_col, "count"),
                avg_order_value=(amount_col, "mean"),
            )
        )
        df_daily = df_daily.rename(columns={"date_day": "date"})

        # ── Total daily revenue across all stores ──
        df_total = (
            df.groupby("date_day", as_index=False)
            .agg(total_revenue=(amount_col, "sum"))
        )
        df_total = df_total.rename(columns={"date_day": "date"})

        if store_col and store_col in df_daily.columns:
            df_daily = df_daily.merge(df_total, on="date", how="left")
        else:
            df_daily["total_revenue"] = df_daily["daily_revenue"]
            if store_col:
                df_daily["store_id"] = df_daily[store_col]
            else:
                df_daily["store_id"] = "all"

        # Drop the helper column from commands if we added it
        if "date_day" in self.df_commands.columns:
            self.df_commands = self.df_commands.drop(columns=["date_day"])

        self.df_daily = df_daily
        logger.info(f"Aggregated to {len(df_daily)} daily records")
        return df_daily

    def estimate_product_sales(self) -> pd.DataFrame:
        """
        Estimate product-level daily sales.
        If idProduit/quantiteVendue exist in commands, use directly.
        Otherwise, distribute revenue proportionally by product price.
        """
        if self.df_commands is None or self.df_products is None:
            return pd.DataFrame()

        if "date" not in self.df_commands.columns:
            return pd.DataFrame()

        amount_col = self._get("amount") or "montant"
        prod_col = self._get("product")
        qty_col = self._get("quantity")
        price_col = self._get("price")
        prod_name_col = self._get("product_name")
        cat_col = self._get("category")
        prod_id_col = self._get("product_id")

        # ── Case 1: Direct product data in commands ──
        if prod_col and qty_col and prod_col in self.df_commands.columns and qty_col in self.df_commands.columns:
            logger.info("Using direct product quantities from commands")

            self.df_commands[qty_col] = pd.to_numeric(self.df_commands[qty_col], errors="coerce").fillna(0)

            df_prod = (
                self.df_commands
                .groupby([pd.Grouper(key="date", freq="D"), prod_col], as_index=False)
                .agg(
                    quantity_sold=(qty_col, "sum"),
                    revenue=(amount_col, "sum"),
                )
            )

            # Map product names
            if self.df_products is not None and prod_id_col and prod_name_col:
                name_map = (
                    self.df_products
                    .drop_duplicates(subset=[prod_id_col])
                    .set_index(prod_id_col)[prod_name_col]
                    .to_dict()
                )
                df_prod["product_name"] = df_prod[prod_col].map(name_map).fillna("Unknown")
            else:
                df_prod["product_name"] = "Product_" + df_prod[prod_col].astype(str).str[:8]

            # Map categories
            if self.df_products is not None and prod_id_col and cat_col:
                cat_map = (
                    self.df_products
                    .drop_duplicates(subset=[prod_id_col])
                    .set_index(prod_id_col)[cat_col]
                    .to_dict()
                )
                df_prod["category"] = df_prod[prod_col].map(cat_map).fillna("Uncategorized")
            else:
                df_prod["category"] = "Uncategorized"

            # Compute unit price
            df_prod["unit_price"] = np.where(
                df_prod["quantity_sold"] > 0,
                df_prod["revenue"] / df_prod["quantity_sold"],
                0,
            )

            self.df_product_daily = df_prod
            logger.info(f"Direct product sales: {len(df_prod)} records")
            return df_prod

        # ── Case 2: Estimate from revenue and product prices ──
        logger.info("Estimating product sales from revenue distribution")

        if not price_col or price_col not in self.df_products.columns:
            logger.warning("No price column found, cannot estimate product sales")
            return pd.DataFrame()

        products = self.df_products[self.df_products[price_col] > 0].copy()
        if len(products) == 0:
            logger.warning("No products with positive price")
            return pd.DataFrame()

        # Weight by inverse price
        products["weight"] = 1.0 / products[price_col]
        products["weight"] = products["weight"] / products["weight"].sum()

        # Daily revenue
        df = self.df_commands.copy()
        df["date_day"] = df["date"].dt.normalize()
        daily_rev = (
            df.groupby("date_day", as_index=False)
            .agg(revenue=(amount_col, "sum"))
            .rename(columns={"date_day": "date"})
        )

        # Distribute revenue
        records = []
        for _, row in daily_rev.iterrows():
            for _, prod in products.iterrows():
                p_id = prod[prod_id_col] if prod_id_col else "unknown"
                p_name = prod[prod_name_col] if prod_name_col else "Unknown"
                p_cat = prod[cat_col] if cat_col else "Uncategorized"
                p_price = prod[price_col]

                est_qty = max(1, round((row["revenue"] * prod["weight"]) / p_price))
                est_rev = est_qty * p_price

                records.append({
                    "date": row["date"],
                    "product_id": p_id,
                    "product_name": p_name,
                    "category": p_cat,
                    "quantity_sold": est_qty,
                    "revenue": round(est_rev, 2),
                    "unit_price": p_price,
                })

        if not records:
            return pd.DataFrame()

        df_prod = pd.DataFrame(records)
        self.df_product_daily = df_prod
        logger.info(f"Estimated {len(df_prod)} product-daily records")
        return df_prod

    def merge_datasets(self) -> pd.DataFrame:
        """Merge daily sales with store information."""
        if self.df_daily is None:
            raise ValueError("No daily data. Call aggregate_daily_sales first.")

        if "date" not in self.df_daily.columns:
            raise ValueError("'date' column missing from daily data")

        df = self.df_daily.copy()

        if self.df_stores is not None and self._get("store"):
            store_col = self._get("store")           # e.g. "idPointVente"
            store_id_col = self._get("store_id") or "_id"  # e.g. "_id"

            if store_col not in df.columns:
                logger.warning(f"Store column '{store_col}' not in daily data, skipping merge")
                self.df_merged = df
                return df

            if store_id_col not in self.df_stores.columns:
                logger.warning(f"Store ID column '{store_id_col}' not in stores data, skipping merge")
                self.df_merged = df
                return df

            stores_clean = self.df_stores.copy()
            rename_map = {}
            name_col = self._get("store_name")
            if name_col and name_col in stores_clean.columns:
                rename_map[name_col] = "store_name"
            cx = self._get("coord_x")
            if cx and cx in stores_clean.columns:
                rename_map[cx] = "coord_x"
            cy = self._get("coord_y")
            if cy and cy in stores_clean.columns:
                rename_map[cy] = "coord_y"
            addr = self._get("address")
            if addr and addr in stores_clean.columns:
                rename_map[addr] = "address"

            keep_cols = [store_id_col] + [c for c in rename_map.values() if c in stores_clean.columns]
            stores_clean = stores_clean[keep_cols].rename(columns=rename_map)

            # Remove duplicate store IDs
            stores_clean = stores_clean.drop_duplicates(subset=[store_id_col])

            df = df.merge(stores_clean, left_on=store_col, right_on=store_id_col, how="left")
            df = df.drop(columns=[store_id_col], errors="ignore")

            # Fill missing store info
            if "store_name" in df.columns:
                df["store_name"] = df["store_name"].fillna("All Stores")
            if "coord_x" in df.columns:
                df["coord_x"] = df["coord_x"].fillna(df["coord_x"].mean() if df["coord_x"].notna().any() else 0)
            if "coord_y" in df.columns:
                df["coord_y"] = df["coord_y"].fillna(df["coord_y"].mean() if df["coord_y"].notna().any() else 0)

        self.df_merged = df
        logger.info(f"Merged dataset shape: {df.shape}")
        return df

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def get_date_range(self) -> Tuple[str, str]:
        """Return min/max dates."""
        if self.df_commands is None or "date" not in self.df_commands.columns:
            return "", ""
        return (
            self.df_commands["date"].min().strftime("%Y-%m-%d"),
            self.df_commands["date"].max().strftime("%Y-%m-%d"),
        )

    def get_store_coordinates(self) -> pd.DataFrame:
        """Return store coordinates for map visualization."""
        if self.df_stores is None:
            return pd.DataFrame()

        sid = self._get("store_id") or "_id"
        sname = self._get("store_name") or sid
        cy = self._get("coord_y")
        cx = self._get("coord_x")

        data = {"store_id": self.df_stores[sid], "store_name": self.df_stores[sname]}
        if cy and cy in self.df_stores.columns:
            data["lat"] = self.df_stores[cy]
        else:
            data["lat"] = pd.Series(dtype=float)
        if cx and cx in self.df_stores.columns:
            data["lon"] = self.df_stores[cx]
        else:
            data["lon"] = pd.Series(dtype=float)

        return pd.DataFrame(data)

    def get_summary_stats(self) -> Dict:
        """Return summary statistics."""
        stats: Dict = {
            "total_commands": 0,
            "total_stores": 0,
            "total_products": 0,
            "total_revenue": 0.0,
            "avg_daily_revenue": 0.0,
            "date_range": ("", ""),
            "missing_values": 0,
        }

        if self.df_commands is not None:
            stats["total_commands"] = len(self.df_commands)
            stats["missing_values"] = int(self.df_commands.isnull().sum().sum())

            amount_col = self._get("amount")
            if amount_col and amount_col in self.df_commands.columns:
                stats["total_revenue"] = float(self.df_commands[amount_col].sum())

        if self.df_stores is not None:
            stats["total_stores"] = len(self.df_stores)

        if self.df_products is not None:
            stats["total_products"] = len(self.df_products)

        stats["date_range"] = self.get_date_range()

        if self.df_daily is not None and "total_revenue" in self.df_daily.columns:
            stats["avg_daily_revenue"] = float(self.df_daily["total_revenue"].mean())
            stats["daily_records"] = len(self.df_daily)

        return stats