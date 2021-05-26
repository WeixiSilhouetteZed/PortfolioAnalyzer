import pandas as pd
import numpy as np
import plotly.graph_objects as go

class option:
    def __init__(
        self, 
        c_or_p: bool, 
        strike: float, 
        price: float,
        l_or_s: bool,
        unit: int,
        name: str) -> None:
        self.cp = "C" if c_or_p else "P"
        self.strike = strike
        self.price = price
        self.sign = l_or_s
        self.unit = unit
        self.name = name
    
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

# if __name__ == "__main__":
#     test_option = option(1, 100, 90, 1, 1)
#     test_stock = underlying(80, 1, 2)
#     test_portfolio = portfolio([test_option, test_stock])
#     test_portfolio.get_pf_graph(True)


