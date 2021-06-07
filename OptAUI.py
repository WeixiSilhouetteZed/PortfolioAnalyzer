import OptionAnalyzer as OA
import streamlit as st 
from PIL import Image

from streamlit import caching

img = Image.open("greek-helmet.png")

st.set_page_config(
    page_title = "Option Portfolio Analyzer",
    page_icon = img,
    layout = 'wide'
)

st.title("Option Portfolio Analyzer")

st.write("By Bill Zhuo")

st.markdown("Input call/put/underlying from the sidebar on the left. \
    All positions need to be named uniquely. Select positions to \
        consist portfolios and check the portfolio risk visualizations.")

@st.cache(allow_output_mutation = True)
def persistdata():
    return dict(), OA.portfolio([]), list()

pos_dict, port, name_list = persistdata()

portfolio_selection = None


def get_pos_list(name_list: list) -> list:
    return [pos_dict[name] for name in name_list]

with st.sidebar:
    st.info("Welcome to Option Portfolio Analyzer!")
    c_p_u = st.selectbox(
        "Call/Put/Underlying",
        (
            "Call",
            "Put",
            "Underlying"
        )
    )
    s = st.number_input("Spot Price", value = 100.)
    l_or_s = st.selectbox(
        "Long (1) or Short (-1)",
        (1, -1)
    )
    unit = st.number_input("Unit", value = 1, min_value = 1)
    name = st.text_input("Position Name")
    if name in name_list:
        st.warning("Already used name!")
    else:
        if c_p_u != "Underlying":
            k = st.number_input("Strike", value = 100.)
            premium = st.number_input("Premium (Optional)")
            expiry = st.number_input("Expiry (Year)", value = 1.)
            r = st.number_input("Risk-free Rate")
            delta = st.number_input("Dividend Yield")
        
    add_new_pos = st.button("Add New Position")
    cache_button = st.button("Cache Clear")
if add_new_pos:
    if c_p_u != "Underlying":
        c_or_p = 1 if c_p_u == "Call" else 0
        pos = OA.option(c_or_p, k, premium, l_or_s, unit, name, expiry, s, r, delta)
    else:
        pos = OA.underlying(s, l_or_s, unit, name)
        
    pos_dict[pos.name] = pos
    port.pos.append(pos)  
    name_list.append(pos.name)

portfolio_selection = st.multiselect(
        "Portfolio Positions",
        name_list,
        []
    )
sigma_box = st.number_input("Volatility", min_value = 1e-5, value = 0.1)
if portfolio_selection:
    port.pos = [pos_dict[name] for name in portfolio_selection]
    port_anal = OA.PortfolioAnalyzer(port)
    with st.spinner('Wait for it...'):
        port_anal.bs_greek_plot([sigma_box] * len(portfolio_selection), True)
    st.success('Done!')

if cache_button:
    caching.clear_cache()



