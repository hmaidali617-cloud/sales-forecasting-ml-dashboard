"""
data_loader.py
==============
Robust CSV/JSON loader with fuzzy column detection, cleaning, merging, aggregation.
NOW SUPPORTS: BPrice CommandeCommandeDetails.json format
"""
from __future__ import annotations

import io
import json
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd

from utils import get_logger

logger = get_logger("data-loader")


# ---------------------------------------------------------------- #
# Column aliases (fuzzy matching)                                  #
# ---------------------------------------------------------------- #
COLUMN_ALIASES = {
    # Orders
    "order_id": ["_id", "id", "idCommande", "orderId", "commande_id", "idCommandeString"],
    "date": ["dateCreation", "date_creation", "createdAt", "dateCmd", "date_commande", "date"],
    "store_id": ["idPointVente", "pointVenteId", "store_id", "store", "pv_id"],
    "amount": ["montant", "amount", "total", "revenue", "ca", "sales"],
    "quantity": ["quantiteVendue", "quantite", "qty", "quantity", "qte"],
    "product_id": ["idProduit", "produitId", "product_id", "productId"],
    "is_cancelled": ["isAnnule", "cancelled", "isCancelled"],
    # Stores
    "store_internal_id": ["_id", "id"],
    "store_name": ["designation", "name", "label", "nom"],
    "coord_x": ["coordX", "latitude", "lng", "lon", "x"],
    "coord_y": ["coordY", "longitude", "lat", "y"],
    "address": ["adresse", "address", "addr"],
    # Products
    "product_internal_id": ["_id", "id"],
    "product_name": ["designation", "name", "label", "nom"],
    "category": ["category", "categorie", "cat", "type"],
    "price": ["prix", "price"],
    "stock": ["stockReel", "stock", "quantite"],
}


def _normalize_col(c: str) -> str:
    return str(c).strip().lower().replace(" ", "").replace("_", "").replace("-", "")


def find_column(df_cols, candidates) -> Optional[str]:
    """Find first column matching any candidate (case/symbol insensitive)."""
    norm_cols = {_normalize_col(c): c for c in df_cols}
    for cand in candidates:
        nc = _normalize_col(cand)
        if nc in norm_cols:
            return norm_cols[nc]
    return None


# ---------------------------------------------------------------- #
# MongoDB Date Parser                                              #
# ---------------------------------------------------------------- #
def parse_mongo_date(date_val) -> Optional[pd.Timestamp]:
    """Parse MongoDB date format to pandas Timestamp."""
    if date_val is None:
        return None
    if isinstance(date_val, pd.Timestamp):
        return date_val
    if isinstance(date_val, str):
        try:
            return pd.to_datetime(date_val)
        except:
            return None
    if isinstance(date_val, dict) and "$date" in date_val:
        try:
            return pd.to_datetime(date_val["$date"])
        except:
            return None
    return None


# ---------------------------------------------------------------- #
# BPrice JSON Parser                                               #
# ---------------------------------------------------------------- #
# ---------------------------------------------------------------- #
# BPrice JSON Parser                                               #
# ---------------------------------------------------------------- #
# ---------------------------------------------------------------- #
# BPrice JSON Parser (DEBUG VERSION)                               #
# ---------------------------------------------------------------- #
def parse_bprice_json(raw_data: list) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Parse BPrice CommandeCommandeDetails.json format.
    Returns: (orders_df, products_df)
    
    EXPANDED FORMAT: Each line item becomes a separate row so that
    product forecasting gets accurate per-product data.
    """
    orders_list = []
    details_list = []
    
    skipped_reasons = {"no_id": 0, "no_date": 0, "cancelled": 0, "no_details": 0, "no_amount": 0, "success": 0}
    
    for record in raw_data:
        # Parse order ID - handle multiple formats
        order_id = record.get('idCommandeString')
        if not order_id:
            _id = record.get('_id')
            if isinstance(_id, dict):
                order_id = _id.get('$oid', str(_id))
            elif _id:
                order_id = str(_id)
            else:
                order_id = None
        
        if not order_id:
            skipped_reasons["no_id"] += 1
            continue
        
        # Parse date - handle MongoDB date format and plain strings
        order_date = parse_mongo_date(record.get('dateCreation'))
        if order_date is None:
            # Try nested commande object
            nested_cmd = record.get('commande', {})
            if isinstance(nested_cmd, dict):
                order_date = parse_mongo_date(nested_cmd.get('dateCreation'))
            
            if order_date is None:
                skipped_reasons["no_date"] += 1
                continue
        
        # Get store ID
        store_id = str(record.get('idPointVente', 'UNKNOWN'))
        
        # Check if cancelled - handle int, bool, or string
        is_cancelled = record.get('isAnnule', 0)
        if isinstance(is_cancelled, bool):
            is_cancelled = 1 if is_cancelled else 0
        else:
            try:
                is_cancelled = int(is_cancelled)
            except (ValueError, TypeError):
                is_cancelled = 0
        
        if is_cancelled == 1:
            skipped_reasons["cancelled"] += 1
            continue
        
        # Get montant
        try:
            total_montant = float(record.get('montant', 0))
        except (ValueError, TypeError):
            total_montant = 0
        
        if total_montant <= 0:
            skipped_reasons["no_amount"] += 1
            continue
        
        # Get product details - try top level first, then nested
        commande_details = record.get('commandeDetails', [])
        if not commande_details:
            # Try nested inside 'commande' object
            nested_cmd = record.get('commande', {})
            if isinstance(nested_cmd, dict):
                commande_details = nested_cmd.get('commandeDetails', [])
        
        if not commande_details:
            skipped_reasons["no_details"] += 1
            continue
        
        # ═══════════════════════════════════════════════════════════════════
        # FIXED: Filter out cancelled line items - keep where isAnnule == 0
        # ═══════════════════════════════════════════════════════════════════
        valid_details = []
        for d in commande_details:
            if not isinstance(d, dict):
                continue
            
            # Check if line item is cancelled
            item_cancelled = d.get('isAnnule', 0)
            if isinstance(item_cancelled, bool):
                item_cancelled = 1 if item_cancelled else 0
            else:
                try:
                    item_cancelled = int(item_cancelled)
                except (ValueError, TypeError):
                    item_cancelled = 0
            
            # FIXED: Keep items where isAnnule == 0 (NOT cancelled)
            if item_cancelled != 0:
                continue
            
            # Check if product ID exists
            product_id = d.get('idProduit')
            if not product_id or str(product_id).strip() == '':
                continue
            
            valid_details.append(d)
        
        # Skip orders with no valid items
        if not valid_details:
            skipped_reasons["no_details"] += 1
            continue
        
        # ═══════════════════════════════════════════════════════════════════
        # Get product name - the 'designation' field is at TOP LEVEL
        # ═══════════════════════════════════════════════════════════════════
        def get_product_name(detail):
            if not isinstance(detail, dict):
                return None
            # FIRST: Check top-level designation (this is where it is in BPrice JSON!)
            name = detail.get('designation')
            if name and str(name).strip():
                return str(name).strip()
            # SECOND: Try nested commandeDetails (rare, but just in case)
            nested = detail.get('commandeDetails')
            if isinstance(nested, dict):
                name = nested.get('designation')
                if name and str(name).strip():
                    return str(name).strip()
            # FALLBACK: Generate from product ID
            product_id = detail.get('idProduit', '')
            return f'Product {str(product_id)[-6:]}'
        
        # ═══════════════════════════════════════════════════════════════════
        # EXPAND: Create one row per line item
        # ═══════════════════════════════════════════════════════════════════
        for detail in valid_details:
            product_id = str(detail.get('idProduit', ''))
            product_name = get_product_name(detail)  # Gets "Capucin" etc.
            
            try:
                qty = int(detail.get('quantite', 1))
                if qty <= 0:
                    qty = 1
            except (ValueError, TypeError):
                qty = 1
            
            try:
                prix = float(detail.get('prix', 0))
                if prix < 0:
                    prix = 0
            except (ValueError, TypeError):
                prix = 0
            
            line_amount = prix * qty
            
            orders_list.append({
                'order_id': str(order_id),
                'date': order_date,
                'store_id': store_id,
                'amount': line_amount,
                'quantity': qty,
                'product_id': product_id,
                'product_name': product_name,  # ← THIS is now "Capucin" not "Product 95e24"
            })
            
            details_list.append({
                'date': order_date,
                'store_id': store_id,
                'product_id': product_id,
                'product_name': product_name,
                'quantity': qty,
                'amount': line_amount,
                'unit_price': prix,
            })
        
        skipped_reasons["success"] += 1
    
    # Log skip reasons for debugging
    logger.info(f"JSON parsing results: {skipped_reasons}")
    
    orders_df = pd.DataFrame(orders_list)
    details_df = pd.DataFrame(details_list) if details_list else pd.DataFrame()
    
    logger.info(f"Parsed BPrice JSON: {len(orders_df)} line items from {len(raw_data)} records")
    
    return orders_df, details_df

def create_products_from_details(details_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create products DataFrame from order line item details.
    Auto-categorizes products based on name.
    """
    if details_df.empty:
        return pd.DataFrame()
    
    # Aggregate by product
    products = details_df.groupby(['product_id', 'product_name']).agg({
        'quantity': 'sum',
        'amount': 'sum',
        'unit_price': 'first'
    }).reset_index()
    
    products = products.rename(columns={
        'quantity': 'total_sold',
        'amount': 'total_revenue',
        'unit_price': 'price'
    })
    
    # Auto-categorize
    def detect_category(name):
        name_lower = str(name).lower()
        
        # Café
        if any(x in name_lower for x in ['café', 'coffee', 'express', 'capucin', 
                                           'américain', 'crème', 'noisette', 'allongé',
                                           'turc', 'macchiato', 'latte', 'cappuccino']):
            return 'Café'
        # Thé
        elif any(x in name_lower for x in ['thé', 'tea', 'tisan', 'infus', 'menthe',
                                             'vert', 'noir', 'jasmin']):
            return 'Thé'
        # Pâtisserie
        elif any(x in name_lower for x in ['chocolat', 'choco', 'moelleux', 'gâteau', 
                                             'cake', 'brownie', 'cookie', 'muffin',
                                             'tarte', 'éclair', 'croissant']):
            return 'Pâtisserie'
        # Boisson froide
        elif any(x in name_lower for x in ['jus', 'juice', 'smoothie', 'frappé', 
                                             'milkshake', 'glacé', 'slush', 'ice']):
            return 'Boisson Froide'
        # Eau
        elif any(x in name_lower for x in ['eau', 'water']):
            return 'Eau'
        # Boulangerie
        elif any(x in name_lower for x in ['sandwich', 'pain', 'baguette', 'wrap',
                                             'foccacia', 'brioch']):
            return 'Boulangerie'
        # Repas
        elif any(x in name_lower for x in ['salade', 'salad', 'plat', 'menu', 
                                             'burger', 'pizza', 'pasta', 'omelette',
                                             'quiche']):
            return 'Repas'
        else:
            return 'Autres'
    
    products['category'] = products['product_name'].apply(detect_category)
    
    # Final columns - KEEP product_name!
    products = products[['product_id', 'product_name', 'category', 'price']]
    
    logger.info(f"Created products table: {len(products)} products, {products['category'].nunique()} categories")
    
    return products


# ---------------------------------------------------------------- #
# Loaders                                                          #
# ---------------------------------------------------------------- #
def load_csv_from_upload(uploaded_file) -> pd.DataFrame:
    """Load CSV from Streamlit UploadedFile."""
    if uploaded_file is None:
        raise ValueError("No file uploaded")
    try:
        uploaded_file.seek(0)
    except Exception:
        pass
    content = uploaded_file.read()
    if content.startswith(b"\xef\xbb\xbf"):
        content = content[3:]
    for sep in [",", ";", "\t", "|"]:
        try:
            df = pd.read_csv(io.BytesIO(content), sep=sep, encoding="utf-8", low_memory=False)
            if df.shape[1] > 1:
                return df
        except Exception:
            continue
    return pd.read_csv(io.BytesIO(content), low_memory=False)


def load_json_from_upload(uploaded_file) -> list:
    """Load JSON from Streamlit UploadedFile."""
    if uploaded_file is None:
        raise ValueError("No file uploaded")
    try:
        uploaded_file.seek(0)
    except Exception:
        pass
    content = uploaded_file.read()
    
    try:
        data = json.loads(content.decode('utf-8'))
    except json.JSONDecodeError:
        # Try with BOM removal
        if content.startswith(b"\xef\xbb\xbf"):
            content = content[3:]
        data = json.loads(content.decode('utf-8'))
    
    if not isinstance(data, list):
        raise ValueError("JSON must be an array of objects")
    
    logger.info(f"Loaded JSON with {len(data)} records")
    return data


def load_orders_from_upload(uploaded_file) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Load orders from either CSV or JSON upload.
    Returns: (orders_df, products_df_or_None)
    
    If JSON (BPrice format), also extracts products from commandeDetails.
    If CSV, returns (orders_df, None) - products must be uploaded separately.
    """
    if uploaded_file is None:
        raise ValueError("No file uploaded")
    
    filename = uploaded_file.name.lower()
    
    if filename.endswith('.json'):
        # BPrice JSON format
        raw_data = load_json_from_upload(uploaded_file)
        orders_df, details_df = parse_bprice_json(raw_data)
        
        # Create products from details
        if not details_df.empty:
            products_df = create_products_from_details(details_df)
            return orders_df, products_df
        else:
            return orders_df, None
    else:
        # CSV format
        raw_df = load_csv_from_upload(uploaded_file)
        return raw_df, None


def clean_orders(df: pd.DataFrame, from_json: bool = False) -> pd.DataFrame:
    """Clean orders dataframe (from CSV or JSON)."""
    if df is None or df.empty:
        raise ValueError("Orders dataframe is empty")

    if from_json:
        # JSON data is already cleaned by parse_bprice_json
        # Just ensure correct types and KEEP product_name!
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
        df["date"] = df["date"].dt.normalize()
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0).clip(lower=0)
        df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce").fillna(1.0).clip(lower=1)
        
        if "store_id" not in df.columns:
            df["store_id"] = "UNKNOWN"
        
        # Return with product_id AND product_name if available
        cols = ["date", "store_id", "amount", "quantity"]
        if "product_id" in df.columns:
            cols.append("product_id")
        if "product_name" in df.columns:
            cols.append("product_name")  # KEEP THIS!
        
        return df[cols]

    # Original CSV cleaning logic
    col_date = find_column(df.columns, COLUMN_ALIASES["date"])
    col_store = find_column(df.columns, COLUMN_ALIASES["store_id"])
    col_amount = find_column(df.columns, COLUMN_ALIASES["amount"])
    col_cancel = find_column(df.columns, COLUMN_ALIASES["is_cancelled"])

    if not col_date:
        raise ValueError("Date column not found in orders file")
    if not col_amount:
        raise ValueError("Amount column not found in orders file")

    df = df.copy()
    rename_map = {col_date: "date"}
    if col_store:
        rename_map[col_store] = "store_id"
    if col_amount:
        rename_map[col_amount] = "amount"
    col_qty = find_column(df.columns, COLUMN_ALIASES["quantity"])
    col_prod = find_column(df.columns, COLUMN_ALIASES["product_id"])
    col_prod_name = find_column(df.columns, COLUMN_ALIASES["product_name"])
    if col_qty:
        rename_map[col_qty] = "quantity"
    if col_prod:
        rename_map[col_prod] = "product_id"
    if col_prod_name:
        rename_map[col_prod_name] = "product_name"  # ADD THIS!
    df = df.rename(columns=rename_map)

    # Parse date
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=False)
    df = df.dropna(subset=["date"])
    df["date"] = df["date"].dt.tz_localize(None) if df["date"].dt.tz is not None else df["date"]
    df["date"] = df["date"].dt.normalize()

    # Convert amount
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)
    df["amount"] = df["amount"].clip(lower=0)

    # Quantity (default 1)
    if "quantity" not in df.columns:
        df["quantity"] = 1.0
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce").fillna(1.0).clip(lower=0)

    # Filter cancelled
    if col_cancel and col_cancel in df.columns:
        cancel_series = pd.to_numeric(df[col_cancel], errors="coerce").fillna(0)
        df = df[cancel_series == 0]

    # Store id fallback
    if "store_id" not in df.columns:
        df["store_id"] = "UNKNOWN"

    # Return with product_name if available
    cols = ["date", "store_id", "amount", "quantity"]
    if "product_id" in df.columns:
        cols.append("product_id")
    if "product_name" in df.columns:
        cols.append("product_name")

    return df[cols]


def clean_products(df: pd.DataFrame, from_json: bool = False) -> pd.DataFrame:
    """Clean products dataframe (from JSON extraction or CSV/PPV upload)."""
    if df is None or df.empty:
        raise ValueError("Products dataframe is empty")

    if from_json:
        # Already cleaned by create_products_from_details
        df = df.copy()
        df["product_id"] = df["product_id"].astype(str)
        if "product_name" not in df.columns:
            df["product_name"] = "Product " + df["product_id"].str[-6:]
        if "category" not in df.columns:
            df["category"] = "General"
        return df[["product_id", "product_name", "category"]].drop_duplicates(subset=["product_id"])

    # Original CSV/PPV cleaning logic
    is_ppv = bool(find_column(df.columns, COLUMN_ALIASES["store_id"]) and find_column(df.columns, COLUMN_ALIASES["price"]))

    if is_ppv:
        col_id = find_column(df.columns, ["idProduit", "product_id", "productId"])
        col_store = find_column(df.columns, COLUMN_ALIASES["store_id"])
        col_price = find_column(df.columns, COLUMN_ALIASES["price"])
        col_stock = find_column(df.columns, COLUMN_ALIASES["stock"])
        df = df.copy()
        rename = {col_id: "product_id"}
        if col_store:
            rename[col_store] = "store_id"
        if col_price:
            rename[col_price] = "price"
        if col_stock:
            rename[col_stock] = "stock"
        df = df.rename(columns=rename)
        for c in ("price", "stock"):
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df["product_id"] = df["product_id"].astype(str)
        df["product_name"] = "Product " + df["product_id"].astype(str).str[-6:]
        df["category"] = "General"
        return df[["product_id", "product_name", "category", "price", "stock", "store_id"]]
    else:
        col_id = find_column(df.columns, ["_id", "id"])
        col_name = find_column(df.columns, COLUMN_ALIASES["product_name"])
        col_cat = find_column(df.columns, COLUMN_ALIASES["category"])
        df = df.copy()
        rename = {col_id: "product_id"}
        if col_name:
            rename[col_name] = "product_name"
        if col_cat:
            rename[col_cat] = "category"
        df = df.rename(columns=rename)
        if "product_name" not in df.columns:
            df["product_name"] = df["product_id"].astype(str)
        if "category" not in df.columns:
            df["category"] = "General"
        df["product_id"] = df["product_id"].astype(str)
        return df[["product_id", "product_name", "category"]].drop_duplicates(subset=["product_id"])


def clean_stores(df: pd.DataFrame) -> pd.DataFrame:
    """Clean stores dataframe (unchanged - uses bprice_prod.pointVente.csv)."""
    if df is None or df.empty:
        raise ValueError("Stores dataframe is empty")

    col_id = find_column(df.columns, COLUMN_ALIASES["store_internal_id"]) or find_column(df.columns, COLUMN_ALIASES["store_id"])
    col_name = find_column(df.columns, COLUMN_ALIASES["store_name"])
    col_x = find_column(df.columns, COLUMN_ALIASES["coord_x"])
    col_y = find_column(df.columns, COLUMN_ALIASES["coord_y"])
    col_addr = find_column(df.columns, COLUMN_ALIASES["address"])

    if not col_id:
        raise ValueError("Store id column not found")

    df = df.copy()
    rename = {col_id: "store_id"}
    if col_name:
        rename[col_name] = "store_name"
    if col_x:
        rename[col_x] = "coord_x"
    if col_y:
        rename[col_y] = "coord_y"
    if col_addr:
        rename[col_addr] = "address"
    df = df.rename(columns=rename)

    if "coord_x" in df.columns:
        df["coord_x"] = pd.to_numeric(df["coord_x"], errors="coerce")
    if "coord_y" in df.columns:
        df["coord_y"] = pd.to_numeric(df["coord_y"], errors="coerce")
    if "store_name" not in df.columns:
        df["store_name"] = df["store_id"].astype(str)
    if "address" not in df.columns:
        df["address"] = ""
    
    # Filter out invalid coordinates (like 55555)
    if "coord_x" in df.columns and "coord_y" in df.columns:
        valid_coord = (
            df["coord_x"].between(-90, 90) & 
            df["coord_y"].between(-180, 180)
        )
        invalid_count = (~valid_coord).sum()
        if invalid_count > 0:
            logger.warning(f"Filtering {invalid_count} stores with invalid coordinates")
            df.loc[~valid_coord, ["coord_x", "coord_y"]] = np.nan
    
    df["store_id"] = df["store_id"].astype(str)
    return df[["store_id", "store_name", "coord_x", "coord_y", "address"]].drop_duplicates(subset=["store_id"])


# ---------------------------------------------------------------- #
# Aggregation                                                      #
# ---------------------------------------------------------------- #
def aggregate_daily_revenue(orders: pd.DataFrame) -> pd.DataFrame:
    """Aggregate to daily revenue per store + global."""
    df = orders.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    daily = (
        df.groupby(["date", "store_id"], as_index=False)
        .agg(revenue=("amount", "sum"), orders=("amount", "size"), quantity=("quantity", "sum"))
    )
    return daily.sort_values(["store_id", "date"]).reset_index(drop=True)


def build_global_daily_series(daily_store: pd.DataFrame) -> pd.DataFrame:
    """Build complete daily series for all stores combined."""
    if daily_store.empty:
        return pd.DataFrame(columns=["date", "revenue", "orders", "quantity"])
    s = (
        daily_store.groupby("date", as_index=False)
        .agg(revenue=("revenue", "sum"), orders=("orders", "sum"), quantity=("quantity", "sum"))
        .sort_values("date")
        .reset_index(drop=True)
    )
    s = s.set_index("date").asfreq("D").reset_index()
    s[["revenue", "orders", "quantity"]] = s[["revenue", "orders", "quantity"]].fillna(0)
    return s


def merge_with_stores(daily_store: pd.DataFrame, stores: pd.DataFrame) -> pd.DataFrame:
    return daily_store.merge(stores, on="store_id", how="left")