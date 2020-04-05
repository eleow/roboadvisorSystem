from django.contrib import admin
from django.conf import settings
from django.conf.urls.static import static
from django.urls import include, path
import profiles.urls
import accounts.urls
from . import views

# Personalized admin site settings like title and header
admin.site.site_title = "Smartportfolioweb Site Admin"
admin.site.site_header = "Smartportfolioweb Administration"

urlpatterns = [
    path("", views.HomePage.as_view(), name="home"),
    path("about/", views.AboutPage.as_view(), name="about"),
    path("users/", include(profiles.urls)),
    path("admin/", admin.site.urls),
    path("", include(accounts.urls)),
    path("portfolio/", views.PortfolioPage.as_view(), name="portfolio"),
    path("portfolio_edit/", views.PortfolioEditPage.as_view(), name="portfolio_edit"),
    path("portfolio_edit/buy/<str:id>/<int:amt>/", views.portfolio_buy, name="portfolio_buy"),
    path("portfolio_edit/sell/<str:id>/<int:amt>/", views.portfolio_sell, name="portfolio_sell"),
    path("portfolio/reset", views.portfolio_reset, name="portfolio_reset")
]

# User-uploaded files like profile pics need to be served in development
urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

# Include django debug toolbar if DEBUG is on
if settings.DEBUG:
    import debug_toolbar

    urlpatterns += [path("__debug__/", include(debug_toolbar.urls))]
