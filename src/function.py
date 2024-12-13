from vnstock3 import Vnstock
import streamlit as st

vn_stock = Vnstock()

@st.cache_data
def get_list_industries():
    industries = vn_stock.stock().listing.symbols_by_industries()["icb_name2"].unique().tolist()
    industries.insert(0, "Tất cả")
    return industries

@st.cache_data
def get_list_symbols_by_industry(industry):
    df = vn_stock.stock().listing.symbols_by_industries()
    if industry == "Tất cả":
        filtered_df = df
    else:
        filtered_df = df[df["icb_name2"] == industry]
    return (filtered_df["symbol"] + " - " + filtered_df["organ_name"]).tolist()