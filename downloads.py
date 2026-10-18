"""
downloads.py
============
Helpers to export Plotly figures and DataFrames as PNG images.
"""
from __future__ import annotations

import io
from typing import Optional

import pandas as pd
import streamlit as st


def fig_to_png_bytes(fig, width: int = 1400, height: int = 600, scale: int = 2) -> bytes:
    """Convert a Plotly figure to PNG bytes (requires kaleido)."""
    return fig.to_image(format="png", width=width, height=height, scale=scale)


def df_to_png_bytes(
    df: pd.DataFrame,
    title: str = "",
    figsize=(12, None),
    dpi: int = 150,
) -> bytes:
    """Render a DataFrame as a styled PNG image using matplotlib."""
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    n_rows = len(df)
    n_cols = len(df.columns)

    # Auto figure height
    if figsize[1] is None:
        height_in = max(2.0, 0.45 * n_rows + 1.5)
    else:
        height_in = figsize[1]
    width_in = figsize[0]

    fig, ax = plt.subplots(figsize=(width_in, height_in))
    ax.axis("off")

    if title:
        ax.set_title(
            title, fontsize=14, color="white", pad=14, weight="bold"
        )

    # Format cell values
    cell_text = []
    for _, row in df.iterrows():
        formatted = []
        for v in row:
            if isinstance(v, float):
                formatted.append(f"{v:,.2f}")
            elif pd.isna(v):
                formatted.append("")
            else:
                formatted.append(str(v))
        cell_text.append(formatted)

    table = ax.table(
        cellText=cell_text,
        colLabels=[str(c) for c in df.columns],
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.6)

    # Style cells (dark theme)
    for (r, c), cell in table.get_celld().items():
        if r == 0:  # header
            cell.set_facecolor("#636EFA")
            cell.set_text_props(color="white", weight="bold")
        else:
            if r % 2 == 0:
                cell.set_facecolor("#1a1f29")
            else:
                cell.set_facecolor("#252b38")
            cell.set_text_props(color="white")
        cell.set_edgecolor("#2a2f3a")

    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")

    buf = io.BytesIO()
    fig.savefig(
        buf,
        format="png",
        dpi=dpi,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def show_chart(
    fig,
    filename: str = "chart.png",
    key: Optional[str] = None,
    width: str = "stretch",
    height: int = 600,
):
    """Display a Plotly chart + a download-as-PNG button."""
    st.plotly_chart(fig, width=width, height=height)
    try:
        img = fig_to_png_bytes(fig, width=1400, height=height, scale=2)
        st.download_button(
            label=f"📥 Download PNG ({filename})",
            data=img,
            file_name=filename,
            mime="image/png",
            key=key or f"dl_{filename}",
        )
    except Exception as e:
        st.warning(f"⚠️ PNG export unavailable for {filename}: {e}")


def show_table(
    df: pd.DataFrame,
    filename: str = "table.png",
    key: Optional[str] = None,
    title: str = "",
    width: str = "stretch",
    download_png: bool = True,
    download_csv: bool = False,
):
    """Display a DataFrame + optional PNG / CSV download buttons."""
    # Display interactively
    st.dataframe(df, width=width, hide_index=True)

    cols = st.columns([1, 1, 4]) if (download_png and download_csv) else st.columns([1, 4])
    if download_png:
        with cols[0]:
            try:
                img = df_to_png_bytes(df, title=title)
                st.download_button(
                    label=f"📥 PNG",
                    data=img,
                    file_name=filename,
                    mime="image/png",
                    key=key or f"dl_{filename}",
                )
            except Exception as e:
                st.warning(f"⚠️ Table PNG export failed: {e}")
    if download_csv:
        with cols[1]:
            st.download_button(
                label="📄 CSV",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name=filename.replace(".png", ".csv"),
                mime="text/csv",
                key=(key or f"dl_{filename}") + "_csv",
            )