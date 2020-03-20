from django.views import generic
from django.contrib.auth.decorators import login_required
from django.utils.decorators import method_decorator


class HomePage(generic.TemplateView):
    template_name = "home.html"


class AboutPage(generic.TemplateView):
    template_name = "about.html"


class PortfolioPage(generic.TemplateView):
    template_name = "portfolio.html"

    @method_decorator(login_required(login_url=''))
    def dispatch(self, *args, **kwargs):
        return super(PortfolioPage, self).dispatch(*args, **kwargs)
