from django.views import generic
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.utils.decorators import method_decorator
from django.shortcuts import redirect
from django.urls import reverse
# from django.shortcuts import get_object_or_404, redirect
# from profiles import models as p_models
# from profiles import forms as p_forms
# from django_tables2 import SingleTableView
# from profiles.models import Profile
# from profiles.tables import PortfolioTable
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


def portfolio_reset(request):
    # shortcut to reset all profile information for the logged in user
    p = request.user.profile

    p.avail_cash = 0
    p.asset_transfers = 0
    p.gross_asset_value = 0
    p.portfolio = {}
    p.save()

    messages.warning(request, "All portfolio values have been reset to zero")

    # Redirect to portfolio page
    return redirect(reverse("portfolio"))


def portfolio_sell(request, id, amt):
    # p = request.user.profile

    find_name = portfolio_selection[portfolio_selection.index == "mpt_spdr_max_sharpe"]['name']
    if (len(find_name) == 0):
        messages.error(request, f"Portfolio with {id} is not available")
    else:
        p_name = find_name[0]

        # TODO
        messages.warning(request, f"This feature has not been implemented yet! {p_name.upper()} cannot be sold!")
        # messages.success(request, f"${amt:,.2f} in {p_name.upper()} has been sold successfully!")

    # Redirect to portfolioEdit
    return redirect(reverse("portfolio_edit"))


def portfolio_buy(request, id, amt):
    p = request.user.profile

    find_name = portfolio_selection[portfolio_selection.index == "mpt_spdr_max_sharpe"]['name']
    if (len(find_name) == 0):
        messages.error(request, f"Portfolio with {id} is not available")
    else:
        p_name = find_name[0]

        if (amt > p.avail_cash):
            messages.error(request, f"Investment of ${amt:,.2f} in {p_name.upper()} is not possible as you only have ${p.avail_cash:,.2f}!")
        else:
            # Add data
            stocks, invested = calculate_portfolio(id, amt)
            leftover_cash = amt - invested
            messages.success(request, f"Investment of ${invested:,.2f} in {p_name.upper()} successful! (${leftover_cash} returned to avail cash)")

            transaction = {
                "type": "user",
                "date": datetime.now(),
                "stocks": stocks
            }

            # p.portfolio = {}
            if p.portfolio is None: p.portfolio = {}
            if p.portfolio.get(id, None) is None: p.portfolio[id] = {"total_invested": 0, "transactions": []}
            p.portfolio[id]["transactions"].append(transaction)
            p.portfolio[id]["total_invested"] += invested

            p.avail_cash -= invested  # substract amt used for investment
            p.save()

    # Redirect to portfolioEdit
    return redirect(reverse("portfolio_edit"))


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
