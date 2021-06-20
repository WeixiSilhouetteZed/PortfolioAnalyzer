import numpy as np 
import pandas as pd 
import scipy.stats as stats 
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from traitlets.traitlets import Instance 

ANNUAL = 256
class option:
    def __init__(
        self, 
        c_or_p: bool, 
        strike: float, 
        price: float,
        l_or_s: bool,
        unit: int,
        name: str,
        expiry: float,
        s: float,
        r: float = 0,
        delta: float = 0) -> None:
        self.cp = "C" if c_or_p else "P"
        self.strike = strike
        self.price = price
        self.sign = l_or_s
        self.unit = unit
        self.name = name
        self.expiry = expiry
        self.r = r
        self.s = s
        self.delta = delta
    
    def get_pf_func(self):
        if self.cp == "C":
            return lambda x: self.unit * self.sign * max(x - self.strike, 0)
        else:
            return lambda x: self.unit * self.sign * max(self.strike - x, 0)

    def get_pnl_func(self):
        if self.cp == "C":
            return lambda x: self.unit * self.sign * max(x - self.strike, 0) - self.sign * self.price * self.unit
        else:
            return lambda x: self.unit * self.sign * max(self.strike - x, 0) - self.sign * self.price * self.unit

class underlying:
    def __init__(
        self,
        price: float,
        l_or_s: bool,
        unit: int,
        name: str
        ) -> None:
        self.price = price
        self.sign = l_or_s
        self.unit = unit
        self.name = name
    def get_pf_func(self):
        return lambda x: self.unit * self.sign * (x - self.price)
    def get_pnl_func(self):
        return lambda x: self.unit * self.sign * (x - self.price)

class portfolio:
    def __init__(
        self,
        list_of_pos: list,
        name: str = "Overall"):
        self.pos = list_of_pos
        self.name = name
    def get_pf_func(self):
        return lambda x: sum((ind_pos.get_pf_func()(x) for ind_pos in self.pos))
    def get_pnl_func(self):
        return lambda x: sum((ind_pos.get_pnl_func()(x) for ind_pos in self.pos))
    def get_pf_graph(self, ind_include: bool = False):
        max_strike = max([ind_pos.price if isinstance(ind_pos, underlying) else ind_pos.strike for ind_pos in self.pos])
        x_domain = np.linspace(0, 1.5 * max_strike, 1000)
        fig = go.Figure()
        if ind_include:
            for ind_pos in self.pos:
                fig.add_trace(go.Scatter(x = x_domain, y = np.array([ind_pos.get_pf_func()(x) for x in x_domain]), mode='lines',
                name=ind_pos.name),
                )
        fig.add_trace(go.Scatter(x = x_domain, y = np.array([self.get_pf_func()(x) for x in x_domain]), mode='lines',
                name=self.name))
        fig.show()
    def get_pnl_graph(self, ind_include: bool = False):
        max_strike = max([ind_pos.price if isinstance(ind_pos, underlying) else ind_pos.strike for ind_pos in self.pos])
        x_domain = np.linspace(0, 1.5 * max_strike, 1000)
        fig = go.Figure()
        if ind_include:
            for ind_pos in self.pos:
                fig.add_trace(go.Scatter(x = x_domain, y = np.array([ind_pos.get_pnl_func()(x) for x in x_domain]), mode='lines',
                name=ind_pos.name),
                )
        fig.add_trace(go.Scatter(x = x_domain, y = np.array([self.get_pnl_func()(x) for x in x_domain]), mode='lines',
                name=self.name))
        fig.show()

class OptionAnalyzer:
    def __init__(self, option: option) -> None:
        self.option = option
        self.d_1 = None
        self.d_2 = None
        self.sign = self.option.sign
        self.unit = self.option.unit
        self.mult = self.option.sign * self.option.unit
    
    def update_d(self, sigma: float) -> None:
        self.d_1 = (np.log(self.option.s / self.option.strike) + (self.option.r - self.option.delta + 0.5 * sigma ** 2) * self.option.expiry) / (sigma * np.sqrt(self.option.expiry))
        self.d_2 = self.d_1 - sigma * np.sqrt(self.option.expiry)

    def d_func_S_gen(self, sigma: float) -> None:
        d_1_func = lambda s: (np.log(s / self.option.strike) + (self.option.r - self.option.delta + 0.5 * sigma ** 2) * self.option.expiry) / (sigma * np.sqrt(self.option.expiry))
        d_2_func = lambda s: d_1_func(s) - sigma * np.sqrt(self.option.expiry)
        return d_1_func, d_2_func

    def d_func_t_gen(self, sigma: float, new_s:float) -> None:
        d_1_func = lambda t: (np.log(new_s / self.option.strike) + (self.option.r - self.option.delta + 0.5 * sigma ** 2) * t) / (sigma * np.sqrt(t))
        d_2_func = lambda t: d_1_func(new_s) - sigma * np.sqrt(t)
        return d_1_func, d_2_func

    def bs_premium(self, sigma: float) -> float:
        self.update_d(sigma)
        S, D, T, K, r, d_1, d_2 = self.option.s, self.option.delta, self.option.expiry, self.option.strike, self.option.r, self.d_1, self.d_2
        return self.mult * (S * np.exp(- D * T) * stats.norm.cdf(d_1) - K * np.exp(- r * T) * stats.norm.cdf(d_2)) if self.option.cp == "C" else self.mult * (- S * np.exp(- D * T) * stats.norm.cdf(-d_1) + K * np.exp(- r * T) * stats.norm.cdf(-d_2))
    
    def bs_premium_S_gen(self, sigma: float) -> None:
        D, T, K, r = self.option.delta, self.option.expiry, self.option.strike, self.option.r
        d_1_func, d_2_func = self.d_func_S_gen(sigma)
        bs_premium_call_func = lambda s: self.mult * (s * np.exp(- D * T) * stats.norm.cdf(d_1_func(s)) - K * np.exp(- r * T) * stats.norm.cdf(d_2_func(s)))
        bs_premium_put_func = lambda s: self.mult * (- s * np.exp(- D * T) * stats.norm.cdf(-d_1_func(s)) + K * np.exp(- r * T) * stats.norm.cdf(-d_2_func(s)))
        return bs_premium_call_func if self.option.cp == "C" else bs_premium_put_func

    def bs_premium_t_gen(self, sigma: float, new_s: float) -> None:
        D, _, K, r = self.option.delta, self.option.expiry, self.option.strike, self.option.r
        d_1_func, d_2_func = self.d_func_t_gen(sigma, new_s)
        bs_premium_call_func = lambda t: self.mult * (new_s * np.exp(- D * t) * stats.norm.cdf(d_1_func(t)) - K * np.exp(- r * t) * stats.norm.cdf(d_2_func(t)))
        bs_premium_put_func = lambda t: self.mult * (- new_s * np.exp(- D * t) * stats.norm.cdf(-d_1_func(t)) + K * np.exp(- r * t) * stats.norm.cdf(-d_2_func(t)))
        return bs_premium_call_func if self.option.cp == "C" else bs_premium_put_func

    def bs_delta(self, sigma: float) -> float:
        self.update_d(sigma)
        return self.mult * np.exp(- self.option.delta * self.option.expiry) * stats.norm.cdf(self.d_1) if self.option.cp == "C" else -self.mult * np.exp(- self.option.delta * self.option.expiry) * stats.norm.cdf(-self.d_1)

    def bs_delta_S_gen(self, sigma: float) -> None:
        d_1_func, _ = self.d_func_S_gen(sigma)
        bs_delta_func = lambda s: self.mult * np.exp(- self.option.delta * self.option.expiry) * stats.norm.cdf(d_1_func(s)) if self.option.cp == "C" else -self.mult * np.exp(- self.option.delta * self.option.expiry) * stats.norm.cdf(-d_1_func(s))
        return bs_delta_func

    def bs_delta_t_gen(self, sigma: float, new_s: float) -> None:
        d_1_func, _ = self.d_func_t_gen(sigma, new_s)
        bs_delta_func = lambda t: self.mult * np.exp(- self.option.delta * t) * stats.norm.cdf(d_1_func(t)) if self.option.cp == "C" else -self.mult * np.exp(- self.option.delta * t) * stats.norm.cdf(-d_1_func(t))
        return bs_delta_func

    def bs_gamma(self, sigma: float) -> float:
        self.update_d(sigma)
        return self.mult * np.exp(- self.option.delta * self.option.expiry) * stats.norm.pdf(self.d_1) / (self.option.s * sigma * np.sqrt(self.option.expiry))

    def bs_gamma_S_gen(self, sigma: float) -> None:
        d_1_func, _ = self.d_func_S_gen(sigma)
        bs_gamma_func = lambda s: self.mult * np.exp(- self.option.delta * self.option.expiry) * stats.norm.pdf(d_1_func(s)) / (s * sigma * np.sqrt(self.option.expiry))
        return bs_gamma_func

    def bs_gamma_t_gen(self, sigma: float, new_s: float) -> None:
        d_1_func, _ = self.d_func_t_gen(sigma, new_s)
        bs_gamma_func = lambda t: self.mult * np.exp(- self.option.delta * t) * stats.norm.pdf(d_1_func(t)) / (self.option.s * sigma * np.sqrt(t))
        return bs_gamma_func

    def bs_vega(self, sigma: float) -> float:
        self.update_d(sigma)
        return self.mult * self.option.s * np.exp(- self.option.delta * self.option.expiry) * stats.norm.pdf(self.d_1) * np.sqrt(self.option.expiry)

    def bs_vega_S_gen(self, sigma: float) -> None:
        d_1_func, _ = self.d_func_S_gen(sigma)
        bs_vega_func = lambda s: self.mult * s * np.exp(- self.option.delta * self.option.expiry) * stats.norm.pdf(d_1_func(s)) * np.sqrt(self.option.expiry)
        return bs_vega_func

    def bs_vega_t_gen(self, sigma: float, new_s: float) -> None:
        d_1_func, _ = self.d_func_t_gen(sigma, new_s)
        bs_vega_func = lambda t: self.mult * new_s * np.exp(- self.option.delta * t) * stats.norm.pdf(d_1_func(t)) * np.sqrt(t)
        return bs_vega_func

    def bs_theta(self, sigma: float) -> float:
        self.update_d(sigma)
        S, D, T, K, r, d_1, d_2 = self.option.s, self.option.delta, self.option.expiry, self.option.strike, self.option.r, self.d_1, self.d_2 
        call_theta = D * S * np.exp(- D * T) * stats.norm.cdf(d_1) - r * K * np.exp(- r * T) * stats.norm.cdf(d_2) - K * np.exp(- r * T) * stats.norm.pdf(d_2) * sigma / (2 * np.sqrt(T))
        put_theta = call_theta + (r * K * np.exp(- r * T) - D * S * np.exp(- D * T)) / ANNUAL
        print(call_theta)
        return self.mult * call_theta if self.option.cp == "C" else self.mult * put_theta

    def bs_theta_S_gen(self, sigma: float) -> None:
        d_1_func, d_2_func = self.d_func_S_gen(sigma)
        D, T, K, r = self.option.delta, self.option.expiry, self.option.strike, self.option.r
        call_theta_func = lambda s: self.mult * (D * s * np.exp(- D * T) * stats.norm.cdf(d_1_func(s)) - r * K * np.exp(- r * T) * stats.norm.cdf(d_2_func(s)) - K * np.exp(- r * T) * stats.norm.pdf(d_2_func(s)) * sigma / (2 * np.sqrt(T)))
        put_theta_func = lambda s: (call_theta_func(s) + (r * K * np.exp(- r * T) - D * s * np.exp(- D * T)) / ANNUAL)
        return call_theta_func if self.option.cp == "C" else put_theta_func

    def bs_rho(self, sigma: float) -> float:
        self.update_d(sigma)
        _, _, T, K, r, _, d_2 = self.option.s, self.option.delta, self.option.expiry, self.option.strike, self.option.r, self.d_1, self.d_2 
        common = T * K * np.exp(-r * T)
        return self.mult * common * stats.norm.cdf(d_2) if self.option.cp == "C" else - self.mult * common * stats.norm.cdf(-d_2)

    def bs_rho_S_gen(self, sigma: float) -> None:
        _, d_2_func = self.d_func_S_gen(sigma)
        D, T, K, r = self.option.delta, self.option.expiry, self.option.strike, self.option.r
        call_rho_func = lambda s: self.mult * T * K * np.exp(-r * T) * stats.norm.cdf(d_2_func(s))
        put_rho_func = lambda s: self.mult * (call_rho_func(s) - T * K * stats.norm.cdf(-d_2_func(s)))
        return call_rho_func if self.option.cp == "C" else put_rho_func

    def bs_vanna(self, sigma: float) -> float:
        self.update_d(sigma)
        _, D, T, _, _, d_1, _ = self.option.s, self.option.delta, self.option.expiry, self.option.strike, self.option.r, self.d_1, self.d_2 
        return self.mult * T * np.exp(- D * T) * (stats.norm.pdf(d_1) - (d_1 * stats.norm.pdf(d_1) / (sigma * np.sqrt(T))))

    def bs_vanna_S_gen(self, sigma: float) -> None:
        d_1_func, _ = self.d_func_S_gen(sigma)
        _, D, T, _, _  = self.option.s, self.option.delta, self.option.expiry, self.option.strike, self.option.r
        return lambda s: self.mult * T * np.exp(- D * T) * (stats.norm.pdf(d_1_func(s)) - (d_1_func(s) * stats.norm.pdf(d_1_func(s)) / (sigma * np.sqrt(T))))

    def bs_charm(self, sigma: float) -> float:
        self.update_d(sigma)
        S, D, T, K, r, d_1, _ = self.option.s, self.option.delta, self.option.expiry, self.option.strike, self.option.r, self.d_1, self.d_2 
        call_charm = D * np.exp(- D * T) * stats.norm.cdf(d_1) - stats.norm.pdf(d_1) * (- np.log(S / K) / (2 * sigma * (T ** (3/2))) + (r - D + 0.5 * sigma ** 2) / (2 * sigma * np.sqrt(T)))
        put_charm = - D * np.exp(- D * T) + call_charm
        return self.mult * call_charm if self.option.cp == "C" else self.mult * put_charm
    
    def bs_charm_S_gen(self, sigma: float) -> None:
        d_1_func, _ = self.d_func_S_gen(sigma)
        _, D, T, K, r  = self.option.s, self.option.delta, self.option.expiry, self.option.strike, self.option.r
        call_charm_func = lambda s: self.mult * (D * np.exp(- D * T) * stats.norm.cdf(d_1_func(s)) - stats.norm.pdf(d_1_func(s)) * (- np.log(s / K) / (2 * sigma * (T ** (3/2))) + (r - D + 0.5 * sigma ** 2) / (2 * sigma * np.sqrt(T))))
        put_charm_func = lambda s: - self.mult * (D * np.exp(- D * T) + call_charm_func(s))
        return call_charm_func if self.option.cp == "C" else put_charm_func

    def bs_vomma(self, sigma: float) -> None:
        self.update_d(sigma)
        S, D, T, _, _, d_1, _ = self.option.s, self.option.delta, self.option.expiry, self.option.strike, self.option.r, self.d_1, self.d_2 
        return self.mult * T * S * np.exp(- D * T) * (d_1 * d_1 * stats.norm.pdf(d_1) / (sigma * np.sqrt(T)) - d_1 * stats.norm.pdf(d_1))

    def bs_vomma_S_gen(self, sigma: float) -> None:
        d_1_func, _ = self.d_func_S_gen(sigma)
        _, D, T, _, _  = self.option.s, self.option.delta, self.option.expiry, self.option.strike, self.option.r
        return lambda s: self.mult * T * s * np.exp(- D * T) * (d_1_func(s) * d_1_func(s) * stats.norm.pdf(d_1_func(s)) / (sigma * np.sqrt(T)) - d_1_func(s) * stats.norm.pdf(d_1_func(s)))

    def bs_greek_plot(self, sigma: float, inter: bool = False) -> None:
        premium_func = self.bs_premium_S_gen(sigma)
        delta_func = self.bs_delta_S_gen(sigma)
        gamma_func = self.bs_gamma_S_gen(sigma)
        vega_func = self.bs_vega_S_gen(sigma)
        theta_func = self.bs_theta_S_gen(sigma)
        rho_func = self.bs_rho_S_gen(sigma)
        vanna_func = self.bs_vanna_S_gen(sigma)
        vomma_func = self.bs_vomma_S_gen(sigma)
        x_domain = np.linspace(self.option.strike * 0.3, self.option.strike * 1.7, 1000)
        func_list = [premium_func, delta_func, gamma_func, vega_func, theta_func, rho_func, vanna_func, vomma_func]
        plot_name = "Greeks Plot (Black-Scholes)"
        local_plotter = FuncPlotter(
            domain = x_domain,
            func_list = func_list,
            plot_name = plot_name
        )
        if inter:
            fig = local_plotter.plot(inter)
            st.plotly_chart(fig)
        else:
            local_plotter.plot()

    def implied_vol(self, s: float = None, sigma: float = 0.01, maxiter = 100, tol = 1.0e-5):
        if s is None:
            s = self.option.price
        for _ in range(maxiter):
            price = self.bs_premium(sigma)
            vega = self.bs_vega(sigma)
            diff = s - price
            if abs(diff) < tol:
                return sigma
            sigma = sigma + diff / vega 
        return sigma 

class FuncPlotter:
    def __init__(
        self, 
        domain: np.array,
        func_list: list,
        plot_name: str,
        name_list: list = ["Premium", "Delta", "Gamma", "Vega", "Theta", "Rho", "Vanna", "Charm"]):
        self.domain = domain
        self.func_list = func_list
        self.plot_row = len(self.func_list)
        self.name = plot_name
        self.func_name = name_list

    def plot(self, inter = False):
        fig = make_subplots(
            rows = self.plot_row, 
            cols = 1,
            subplot_titles = self.func_name)
        for index, func in enumerate(self.func_list):
            y = np.array([func(x) for x in self.domain])
            fig.append_trace(go.Scatter(
                x = self.domain,
                y = y,
                mode = 'lines', 
                name = self.func_name[index]
            ), row = index + 1, col = 1)
        fig.update_layout(height = 1200, width = 1000, title_text = self.name)
        if inter:
            return fig 
        fig.show()
class PortfolioAnalyzer:
    def __init__(self, portfolio: portfolio) -> None:
        self.portfolio = portfolio.pos
        self.portfolio_size = len(self.portfolio)

    def value(self, sigma_list: list) -> float:
        return sum((ins.sign * ins.unit * ins.price if isinstance(ins, underlying) else \
            OptionAnalyzer(ins).bs_premium(sigma) for sigma, ins in zip(sigma_list, self.portfolio)))
        
    def value_S_gen(self, sigma_list: list) -> None:
        return lambda s: sum((ins.sign * ins.unit * s if isinstance(ins, underlying) else \
            OptionAnalyzer(ins).bs_premium_S_gen(sigma)(s) for sigma, ins in zip(sigma_list, self.portfolio)))

    def bs_delta(self, sigma_list: list) -> float:
        return sum((ins.sign * ins.unit if isinstance(ins, underlying) else \
            OptionAnalyzer(ins).bs_delta(sigma) for sigma, ins in zip(sigma_list, self.portfolio)))

    def bs_delta_S_gen(self, sigma_list: list) -> None:
        return lambda s: sum((ins.sign * ins.unit if isinstance(ins, underlying) else \
            OptionAnalyzer(ins).bs_delta_S_gen(sigma)(s) for sigma, ins in zip(sigma_list, self.portfolio)))

    def bs_gamma(self, sigma_list: list) -> float:
        return sum((OptionAnalyzer(ins).bs_gamma(sigma) for sigma, ins in zip(sigma_list, self.portfolio) if isinstance(ins, option)))

    def bs_gamma_S_gen(self, sigma_list: list) -> None:
        return lambda s: sum((OptionAnalyzer(ins).bs_gamma_S_gen(sigma)(s) for sigma, ins in zip(sigma_list, self.portfolio) if isinstance(ins, option)))

    def bs_vega(self, sigma_list: list) -> float:
        return sum((OptionAnalyzer(ins).bs_vega(sigma) for sigma, ins in zip(sigma_list, self.portfolio) if isinstance(ins, option)))

    def bs_vega_S_gen(self, sigma_list: list) -> None:
        return lambda s: sum((OptionAnalyzer(ins).bs_vega_S_gen(sigma)(s) for sigma, ins in zip(sigma_list, self.portfolio) if isinstance(ins, option)))

    def bs_theta(self, sigma_list: list) -> float:
        return sum((OptionAnalyzer(ins).bs_theta(sigma) for sigma, ins in zip(sigma_list, self.portfolio) if isinstance(ins, option)))

    def bs_theta_S_gen(self, sigma_list: list) -> None:
        return lambda s: sum((OptionAnalyzer(ins).bs_theta_S_gen(sigma)(s) for sigma, ins in zip(sigma_list, self.portfolio) if isinstance(ins, option)))

    def bs_rho(self, sigma_list: list) -> float:
        return sum((OptionAnalyzer(ins).bs_rho(sigma) for sigma, ins in zip(sigma_list, self.portfolio) if isinstance(ins, option)))

    def bs_rho_S_gen(self, sigma_list: list) -> None:
        return lambda s: sum((OptionAnalyzer(ins).bs_rho_S_gen(sigma)(s) for sigma, ins in zip(sigma_list, self.portfolio) if isinstance(ins, option)))
    
    def bs_vanna(self, sigma_list: list) -> float:
        return sum((OptionAnalyzer(ins).bs_vanna(sigma) for sigma, ins in zip(sigma_list, self.portfolio) if isinstance(ins, option)))
    
    def bs_vanna_S_gen(self, sigma_list: list) -> None:
        return lambda s: sum((OptionAnalyzer(ins).bs_vanna_S_gen(sigma)(s) for sigma, ins in zip(sigma_list, self.portfolio) if isinstance(ins, option)))

    def bs_charm(self, sigma_list: list) -> float:
        return sum((OptionAnalyzer(ins).bs_charm(sigma) for sigma, ins in zip(sigma_list, self.portfolio) if isinstance(ins, option)))
    
    def bs_charm_S_gen(self, sigma_list: list) -> None:
        return lambda s: sum((OptionAnalyzer(ins).bs_charm_S_gen(sigma)(s) for sigma, ins in zip(sigma_list, self.portfolio) if isinstance(ins, option)))

    def bs_vomma(self, sigma_list: list) -> float:
        return sum((OptionAnalyzer(ins).bs_vomma(sigma) for sigma, ins in zip(sigma_list, self.portfolio) if isinstance(ins, option)))
    
    def bs_vomma_S_gen(self, sigma_list: list) -> None:
        return lambda s: sum((OptionAnalyzer(ins).bs_vomma_S_gen(sigma)(s) for sigma, ins in zip(sigma_list, self.portfolio) if isinstance(ins, option)))

    def bs_greek_plot(self, sigma_list: list, inter = False) -> None:
        value_func = self.value_S_gen(sigma_list)
        delta_func = self.bs_delta_S_gen(sigma_list)
        gamma_func = self.bs_gamma_S_gen(sigma_list)
        vega_func = self.bs_vega_S_gen(sigma_list)
        theta_func = self.bs_theta_S_gen(sigma_list)
        rho_func = self.bs_rho_S_gen(sigma_list)
        vanna_func = self.bs_vanna_S_gen(sigma_list)
        charm_func = self.bs_charm_S_gen(sigma_list)
        vomma_func = self.bs_vomma_S_gen(sigma_list)
        max_strike = max([ind_pos.price if isinstance(ind_pos, underlying) else ind_pos.strike for ind_pos in self.portfolio])
        x_domain = np.linspace(max_strike * 0.3, max_strike * 1.7, 1000)
        func_list = [value_func, delta_func, gamma_func, vega_func, theta_func, rho_func, vanna_func, charm_func, vomma_func]
        plot_name = "Greeks Plot (Black-Scholes)"
        local_plotter = FuncPlotter(
            domain = x_domain,
            func_list = func_list,
            plot_name = plot_name,
            name_list = ["Value", "Delta", "Gamma", "Vega", "Theta", "Rho", "Vanna", "Charm", "Vomma"]
        )
        if inter:
            fig = local_plotter.plot(inter)
            st.plotly_chart(fig)
        else:
            local_plotter.plot()
    
    def position_table(self):
        pd_list = []
        for pos in self.portfolio:
            temp_dict = dict(
                name = [pos.name],
                l_or_s = ["Long" if pos.sign == 1 else "Short"],
                unit = [pos.unit],
                s = [pos.price if isinstance(pos, underlying) else pos.s],
                c_or_p = [pos.cp if isinstance(pos, option) else "None"],
                strike = [pos.strike if isinstance(pos, option) else "None"],
                premium = [pos.price if isinstance(pos, option) else "None"],
                expiry = [pos.expiry if isinstance(pos, option) else "None"],
                r = [pos.r if isinstance(pos, option) else "None"],
                dividend = [pos.delta if isinstance(pos, option) else "None"]
            )
            temp_df = pd.DataFrame.from_dict(temp_dict)
            pd_list.append(temp_df)
        result_df = pd.concat(pd_list, axis = 0)
        result_df.columns = [
            "Name",
            "Long or Short",
            "Unit",
            "Underlying Price",
            "Call or Put",
            "Strike",
            "Premium",
            "Expiry",
            "Interest Rate",
            "Dividend Yield Rate"
        ]
        return result_df

if __name__ == "__main__":
    c1 = option(1, 95, 1, 1, 1, "C1", 1, 100)
    c1_a = OptionAnalyzer(c1)
    c2 = option(1, 100, 1, -1, 2, "C1", 1, 100)
    c2_a = OptionAnalyzer(c2)
    c3 = option(1, 105, 1, 1, 1, "C1", 1, 100)
    c3_a = OptionAnalyzer(c3)
    print(c1_a.bs_theta(0.1))
    print(c2_a.bs_theta(0.1))
    print(c3_a.bs_theta(0.1))