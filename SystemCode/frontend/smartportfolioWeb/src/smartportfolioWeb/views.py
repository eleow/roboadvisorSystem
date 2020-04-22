from django.views import generic
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.utils.decorators import method_decorator
from django.shortcuts import redirect, render
from django.urls import reverse
# from django.shortcuts import get_object_or_404, redirect
# from profiles import models as p_models
# from profiles import forms as p_forms
# from django_tables2 import SingleTableView
# from profiles.models import Profile
# from profiles.tables import PortfolioTable
import numbers
import pandas as pd
import os
from datetime import datetime
from .portfolio import calculate_portfolio, calculate_current_val

PORTFOLIO_SELECTION_PATH = "portfolio_details.xlsx"
portfolio_selection = pd.read_excel(os.path.join(os.path.dirname(__file__), PORTFOLIO_SELECTION_PATH), index_col=0)
col_repl = {c: c.replace(" ", "_") for c in portfolio_selection.columns if " " in c}  # rename column if there is a space
portfolio_selection = portfolio_selection.rename(columns=col_repl)


class HomePage(generic.TemplateView):
    template_name = "home.html"


class AboutPage(generic.TemplateView):
    template_name = "about.html"


def portfolio_reset(request, mode=None):
    # shortcut to reset all profile information for the logged in user
    p = request.user.profile
    p.avail_cash = 0
    p.asset_transfers = 0
    p.gross_asset_value = 0
    p.portfolio = {}

    if mode is None:
        messages.warning(request, "All portfolio values have been reset to zero")
    else:
        # Quick reset to some default starting values for everything
        p.avail_cash = 100000
        p.asset_transfers = 100000

        crb = "crb_all_weather_crb"
        p.portfolio[crb] = {"total_invested": 0, "transactions": []}
        p.portfolio[crb]["transactions"].append({
            "type": "system",
            "date": datetime.now(),
            "stocks": [
                {"ticker": 'VTI', "price/share": 140, "shares": 20, "commision": 1},
                {"ticker": 'TLT', "price/share": 170, "shares": 20, "commision": 1}
            ]
        })
        p.portfolio[crb]["transactions"].append({
            "type": "system",
            "date": datetime.now(),
            "stocks": [
                {"ticker": 'VTI', "price/share": 150, "shares": 12, "commision": 1},
                {"ticker": 'DBC', "price/share": 170, "shares": 11, "commision": 1}
            ]
        })
        p.portfolio[crb]["total_invested"] = (140 * 20) + (170 * 20) + (150 * 12) + (170 * 11) + 4
        p.avail_cash -= p.portfolio[crb]["total_invested"]

    p.save()

    # Redirect to portfolio page
    return redirect(reverse("portfolio"))


def portfolio_sell(request, pid, amt):
    return portfolio_transact(request, pid, -amt)  # switch to negative amount


def portfolio_buy(request, pid, amt):
    return portfolio_transact(request, pid, amt)


def portfolio_transact(request, _id, amt):
    p = request.user.profile

    # find_name = portfolio_selection[portfolio_selection.index == "mpt_spdr_max_sharpe"]['name']
    portfolio_data = portfolio_selection[portfolio_selection.index == _id]

    if (portfolio_data.empty):
        messages.error(request, f"Portfolio with {_id} is not available")
    else:
        p_name = portfolio_data['name'][0]
        t = portfolio_data['type'][0]
        s = portfolio_data['stocks'][0]
        c = portfolio_data['criteria'][0]
        m = portfolio_data['model'][0]

        if (amt > p.avail_cash):
            messages.error(request, f"Investment of ${amt:,.2f} in {p_name.upper()} is not possible as you only have ${p.avail_cash:,.2f}!")

        else:
            # Create portfolio data if new, otherwise, we should retrieve existing transactions
            # p.portfolio = {}
            if p.portfolio is None: p.portfolio = {}
            if p.portfolio.get(_id, None) is None: p.portfolio[_id] = {"total_invested": 0, "transactions": []}

            # Add data, passing in existing transactions as this will be used for rebalancing
            stocks, invested = calculate_portfolio(amt, p.portfolio[_id]["transactions"], t, s, c, m)

            diff = amt - invested

            if invested > 0:
                messages.success(request, f"Bought ${invested:,.2f} in {p_name.upper()} successfully! (${diff:,.2f} returned to avail cash)")
            else:
                messages.success(request, f"Sold ${-invested:,.2f} in {p_name.upper()} successfully!")

            transaction = {
                "type": "user",
                "date": datetime.now(),
                "stocks": stocks
            }

            p.portfolio[_id]["transactions"].append(transaction)
            p.portfolio[_id]["total_invested"] += invested

            p.avail_cash -= invested  # substract amt used for investment
            p.save()

    # Redirect to portfolioEdit
    return redirect(reverse("portfolio_edit"))


def portfolio_details(request, pid=""):
    print("Portfolio details - ", pid)
    return render(request, 'portfolio_details.html', {})


@method_decorator(login_required(login_url='/login'), name='dispatch')
class PortfolioEditPage(generic.TemplateView):
    template_name = "portfolio_edit.html"
    # table_class = PortfolioTable

    def dispatch(self, request, *args, **kwargs):
        p = self.request.user.profile
        kwargs["cash"] = p.avail_cash

        # portfolios will be displayed using jquery datatables in template
        kwargs["avail_portfolios"] = portfolio_selection.T.to_dict()

        # get current portfolios
        current_portfolios = dict.fromkeys(p.portfolio.keys())
        all_portfolios = portfolio_selection.to_dict("index")

        gross_asset_value = 0
        for k in current_portfolios:
            current_portfolios[k] = all_portfolios[k]
            current_portfolios[k]["total_invested"] = p.portfolio[k]["total_invested"]
            current_portfolios[k]["current_value"] = calculate_current_val(p.portfolio[k]["transactions"])
            current_portfolios[k]["earnings"] = current_portfolios[k]["current_value"] - current_portfolios[k]["total_invested"]

            gross_asset_value += current_portfolios[k]["current_value"]

            current_portfolios[k]["class"] = {}
            for attr, v in current_portfolios[k].items():
                if isinstance(v, numbers.Number):
                    current_portfolios[k]["class"][attr] = "good" if v > 0 else "bad"

        p.gross_asset_value = gross_asset_value  # update gross asset value
        p.save()

        kwargs["current_portfolios"] = current_portfolios
        return super(PortfolioEditPage, self).dispatch(request, *args, **kwargs)


@method_decorator(login_required(login_url='/login'), name='dispatch')
class PortfolioPage(generic.TemplateView):
    template_name = "portfolio.html"

    def dispatch(self, request, *args, **kwargs):
        user = self.request.user
        asset = user.profile

        kwargs["account"] = asset.gross_asset_value + asset.avail_cash
        kwargs["earnings"] = kwargs["account"] - asset.asset_transfers

        kwargs["account_title"] = "Sum of gross asset value and available cash"
        kwargs["asset_title"] = "Total transfers into account"
        kwargs["earnings_title"] = "Difference between account and asset transfers"
        kwargs["gross_asset_title"] = "How much your assets are worth now"
        kwargs["cash_title"] = "Available cash that can be used for investment"

        return super(PortfolioPage, self).dispatch(request, *args, **kwargs)
