from option_strat_grapher import underlying
import OptionAnalyzer as OA
import numpy as np
import streamlit as st 

st.title("Option Portfolio Analyzer")

@st.cache(allow_output_mutation = True)
def persistdata():
    return dict(), OA.portfolio([]), list(), list()

pos_dict, port, name_list, current_name_list = persistdata()

def get_pos_list(name_list: list) -> list:
    return [pos_dict[name] for name in name_list]

portfolio_selection = st.multiselect(
    "Portfolio Positions",
    name_list,
    current_name_list
)
current_name_list = portfolio_selection
sigma_box = st.number_input("Volatility", min_value = 0.0)

with st.sidebar:
    port.pos = get_pos_list(current_name_list)
    port_anal = OA.PortfolioAnalyzer(port)
    st.info("Welcome to Option Portfolio Analyzer!")
    c_p_u = st.selectbox(
        "Call/Put/Underlying",
        (
            "Call",
            "Put",
            "Underlying"
        )
    )
    s = st.number_input("Spot Price")
    l_or_s = st.selectbox(
        "Long (1) or Short (-1)",
        (1, -1)
    )
    unit = st.number_input("Unit", value = 1, min_value = 1)
    name = st.text_input("Position Name")
    if c_p_u != "Underlying":
        k = st.number_input("Strike")
        premium = st.number_input("Premium (Optional)")
        expiry = st.number_input("Expiry (Year)")
        r = st.number_input("Risk-free Rate")
        delta = st.number_input("Dividend Yield")
    
    add_new_pos = st.button("Add New Position")
    if add_new_pos:
        if c_p_u != "Underlying":
            c_or_p = 1 if c_p_u == "Call" else 0
            pos = OA.option(c_or_p, k, premium, l_or_s, unit, name, expiry, s, r, delta)
        else:
            pos = OA.underlying(s, l_or_s, unit, name)
        pos_dict[pos.name] = pos
        port.pos.append(pos)  
        name_list.append(pos.name)
        current_name_list.append(pos.name)

if portfolio_selection:
    port.pos = [pos_dict[name] for name in portfolio_selection]
    port_anal.bs_greek_plot([sigma_box] * len(portfolio_selection), True)





