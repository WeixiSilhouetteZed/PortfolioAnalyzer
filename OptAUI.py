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

formula_expander = st.beta_expander("BS Greek Formulae", expanded=True)

with formula_expander:
    st.markdown("All formulae are based on BS model for options on futures. The notations are self-explanatory. \
        The source is *Option Volatility and Pricing* by Sheldon Natenberg.")
    for_col_1 = st.beta_columns((1, 1, 2, 2, 1))
    st.markdown("---")
    for_col_2 = st.beta_columns((1, 1, 2, 2, 1))
    st.markdown("---")
    for_col_3 = st.beta_columns((1, 1, 2, 2, 1))
    st.markdown("---")
    for_col_4 = st.beta_columns((1, 1, 2, 2, 1))
    st.markdown("---")
    for_col_5 = st.beta_columns((1, 1, 2, 2, 1))
    st.markdown("---")
    for_col_6 = st.beta_columns((1, 1, 2, 2, 1))
    st.markdown("---")
    for_col_7 = st.beta_columns((1, 1, 2, 2, 1))
    st.markdown("---")
    for_col_8 = st.beta_columns((1, 1, 2, 2, 1))
    st.markdown("---")
    for_col_9 = st.beta_columns((1, 1, 2, 2, 1))

    with for_col_1[0]:
        st.markdown("**Greeks**")
    with for_col_2[0]:
        st.markdown("Delta($\Delta$)")
    with for_col_3[0]:
        st.markdown("Gamma($\Gamma$)")
    with for_col_4[0]:
        st.markdown(r"Vega($\nu$)")
    with for_col_5[0]:
        st.markdown(r"Theta($\Theta$)")
    with for_col_6[0]:
        st.markdown(r"Rho($\Rho$)")
    with for_col_7[0]:
        st.markdown(r"Vanna")
    with for_col_8[0]:
        st.markdown(r"Charm")
    with for_col_9[0]:
        st.markdown(r"Vomma")

    with for_col_1[1]:
        st.markdown("**Derivative Form**")
    with for_col_2[1]:
        st.markdown(r"$\frac{\partial V}{\partial S}$")
    with for_col_3[1]:
        st.markdown(r"$\frac{\partial^2V}{\partial S^2}$")
    with for_col_4[1]:
        st.markdown(r"$\frac{\partial V}{\partial \sigma}$")
    with for_col_5[1]:
        st.markdown(r"$\frac{\partial V}{\partial t}$")
    with for_col_6[1]:
        st.markdown(r"$\frac{\partial V}{\partial r}$")
    with for_col_7[1]:
        st.markdown(r"$\frac{\partial^2 V}{\partial S\partial\sigma}$")
    with for_col_8[1]:
        st.markdown(r"$\frac{\partial^2 V}{\partial S\partial t}$")
    with for_col_9[1]:
        st.markdown(r"$\frac{\partial^2 V}{\partial \sigma^2}$")

    with for_col_1[2]:
        st.markdown("**Call**")
    with for_col_2[2]:
        st.markdown("$e^{-rt}\Phi(d_1)$")
    with for_col_3[2]:
        st.markdown(r"$\frac{e^{-rt}\phi(d_1)}{S\sigma\sqrt{t}}$")
    with for_col_4[2]:
        st.markdown(r"$Se^{-rt}\phi(d_1)\sqrt{t}$")
    with for_col_5[2]:
        st.markdown(r"$\frac{-Se^{-rt}\phi(d_1)\sigma}{2\sqrt{t}}-rse^{-rt}\Phi(d_1)-rKe^{-rt}\Phi(d_2)$")
    with for_col_6[2]:
        st.markdown(r"$-tSe^{-rt}\Phi(d_1)$")
    with for_col_7[2]:
        st.markdown(r"$-e^{-rt}d_2\frac{\phi(d_1)}{\sigma}$")
    with for_col_8[2]:
        st.markdown(r"$-e^{-rt}\left[\phi(d_1)\left(-\frac{d_2}{2t}\right)-r\Phi(d_1)\right]$")
    with for_col_9[2]:
        st.markdown(r"\nu\left(\frac{d_1d_2}{\sigma}\right)")

    with for_col_1[3]:
        st.markdown("**Put**")
    with for_col_2[3]:
        st.markdown("$e^{-rt}[\Phi(d_1)-1]$")
    with for_col_3[3]:
        st.markdown("Same as call")
    with for_col_4[3]:
        st.markdown("Same as call")
    with for_col_5[3]:
        st.markdown(r"$\frac{-Se^{-rt}\phi(d_1)\sigma}{2\sqrt{t}}-rse^{-rt}\Phi(d_1)+rKe^{-rt}\Phi(-d_2)$")
    with for_col_6[3]:
        st.markdown(r"$tSe^{-rt}\Phi(-d_1)$")
    with for_col_7[3]:
        st.markdown(r"Same as call")
    with for_col_8[3]:
        st.markdown(r"$-e^{-rt}\left[\phi(d_1)\left(-\frac{d_2}{2t}\right)+r\Phi(d_1)\right]$")
    with for_col_9[3]:
        st.markdown(r"Same as call")

    with for_col_1[4]:
        st.markdown("Maximized")
    with for_col_2[4]:
        st.markdown(r"Deeply ITM")
    with for_col_3[4]:
        st.markdown(r"ATM/close to expiration/low volatlity")
    with for_col_4[4]:
        st.markdown(r"ATM/long term")
    with for_col_5[4]:
        st.markdown(r"ATM/close to expiration/low volatility")
    with for_col_6[4]:
        st.markdown(r"Deeply ITM/long term")
    with for_col_7[4]:
        st.markdown(r"15-20, 80-85 delta/low volatility")
    with for_col_8[4]:
        st.markdown(r"15-20, 80-85 delta/close to expiration")
    with for_col_9[4]:
        st.markdown(r"10, 90 delta/long term/low volatility")


@st.cache(allow_output_mutation = True)
def persistdata():
    return dict(), OA.portfolio([]), list(), [""]

pos_dict, port, name_list, note_pad = persistdata()

portfolio_selection = None

note_pad_update = None

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

note_pad_update = st.text_input("Notepad", note_pad[0] if note_pad is not None else "")
if note_pad is not None:
    note_pad.pop()
    note_pad.append(note_pad_update)



table_col, _ = st.beta_columns((5,1))

graph_vs_s_col, graph_vs_t_col = st.beta_columns(2)

if portfolio_selection:
    port.pos = [pos_dict[name] for name in portfolio_selection]
    port_anal = OA.PortfolioAnalyzer(port)
    with st.spinner('Wait for it...'):
        with table_col:
            st.subheader("Position Table")
            position_table = port_anal.position_table()
            st.table(position_table)
        with graph_vs_s_col:
            port_anal.bs_greek_S_plot([sigma_box] * len(portfolio_selection), True)
        with graph_vs_t_col:
            port_anal.bs_greek_t_plot([sigma_box] * len(portfolio_selection), s, True)
    st.success('Done!')

if cache_button:
    caching.clear_cache()



