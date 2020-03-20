from django.views import generic
from django.contrib.auth.decorators import login_required
from django.utils.decorators import method_decorator
# from django.shortcuts import get_object_or_404, redirect
# from profiles import models as p_models
# from profiles import forms as p_forms

class HomePage(generic.TemplateView):
    template_name = "home.html"


class AboutPage(generic.TemplateView):
    template_name = "about.html"


class PortfolioPage(generic.TemplateView):
    template_name = "portfolio.html"

    @method_decorator(login_required(login_url=''))
    def dispatch(self, request, *args, **kwargs):
        user = self.request.user
        asset = user.profile
        asset.avail_cash += 100
        asset.asset_transfers = 10000
        asset.save()

        kwargs["account"] = asset.gross_asset_value + asset.avail_cash
        kwargs["earnings"] = kwargs["account"] - asset.asset_transfers

        return super(PortfolioPage, self).dispatch(request, *args, **kwargs)
