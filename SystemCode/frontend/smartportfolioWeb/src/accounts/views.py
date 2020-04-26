from __future__ import unicode_literals
from django.urls import reverse_lazy
from django.views import generic
from django.contrib.auth import get_user_model
from django.contrib import auth
from django.contrib import messages
from authtools import views as authviews
from braces import views as bracesviews
from django.conf import settings
from . import forms
from django.contrib.auth.decorators import login_required
from django.utils.decorators import method_decorator

User = get_user_model()


@method_decorator(login_required(login_url='/login'), name='dispatch')
class CashTransferView(generic.FormView):
    template_name = "accounts/cash_transfer.html"
    success_url = reverse_lazy("portfolio_edit")  # Redirect to Portfolio page on successful transfer (invoked via super().form_valid(form))
    form_class = forms.CashTransferForm

    def form_valid(self, form):
        user = self.request.user
        asset = user.profile

        asset_transfers = form.cleaned_data["asset_transfers"]
        bWithdraw = form.cleaned_data["withdraw"]
        sTransfer = 'Withdrawal' if bWithdraw else 'Deposit'

        if bWithdraw:
            # check if we have that much cash to withdraw
            if (asset.avail_cash < asset_transfers):
                form._errors["asset_transfers"] = [f"You can only withdraw up to {asset.avail_cash}"]
                return super().form_invalid(form)

            asset_transfers *= -1

        asset.asset_transfers += asset_transfers
        asset.avail_cash += asset_transfers
        asset.save()
        messages.success(self.request, f"Cash {sTransfer} of ${abs(asset_transfers):,.2f} successful!")

        return super().form_valid(form)

    def form_invalid(self, form):
        print('!!! Form invalid')

        # user = self.request.user
        # asset = user.profile

        # asset.asset_transfers = 0
        # asset.avail_cash = 0
        # # asset.gross_asset_value = 0
        # asset.save()

        return super().form_invalid(form)

    # https://stackoverflow.com/questions/19687375/django-formview-does-not-have-form-context
    def get_context_data(self, **kwargs):
        user = self.request.user
        asset = user.profile

        context = super(CashTransferView, self).get_context_data(**kwargs)
        context['cash'] = asset.avail_cash
        return context


# if user already logged in, AnonymousRequiredMixin will automatically redirect to settings.LOGIN_REDIRECT_URL
class LoginView(bracesviews.AnonymousRequiredMixin, authviews.LoginView):
    template_name = "accounts/login.html"
    success_url = reverse_lazy("portfolio_edit")  # Redirect to Portfolio page on successful login (invoked via super().form_valid(form))
    form_class = forms.LoginForm

    def form_valid(self, form):
        redirect = super().form_valid(form)
        remember_me = form.cleaned_data.get("remember_me")
        if remember_me is True:
            ONE_MONTH = 30 * 24 * 60 * 60
            expiry = getattr(settings, "KEEP_LOGGED_DURATION", ONE_MONTH)
            self.request.session.set_expiry(expiry)
        return redirect


class LogoutView(authviews.LogoutView):
    url = reverse_lazy("home")


class SignUpView(
    bracesviews.AnonymousRequiredMixin,
    bracesviews.FormValidMessageMixin,
    generic.CreateView,
):
    form_class = forms.SignupForm
    model = User
    template_name = "accounts/signup.html"
    success_url = reverse_lazy("home")  # Redirect to Home on successful login (invoked via super().form_valid(form))
    form_valid_message = "You're signed up!"

    def form_valid(self, form):
        r = super().form_valid(form)
        username = form.cleaned_data["email"]
        password = form.cleaned_data["password1"]
        user = auth.authenticate(email=username, password=password)
        auth.login(self.request, user)
        return r


class PasswordChangeView(authviews.PasswordChangeView):
    form_class = forms.PasswordChangeForm
    template_name = "accounts/password-change.html"
    success_url = reverse_lazy("accounts:logout")

    def form_valid(self, form):
        form.save()
        messages.success(
            self.request,
            "Your password was changed, "
            "hence you have been logged out. Please relogin",
        )

        return super().form_valid(form)


class PasswordResetView(authviews.PasswordResetView):
    form_class = forms.PasswordResetForm
    template_name = "accounts/password-reset.html"
    success_url = reverse_lazy("accounts:password-reset-done")
    subject_template_name = "accounts/emails/password-reset-subject.txt"
    email_template_name = "accounts/emails/password-reset-email.html"


class PasswordResetDoneView(authviews.PasswordResetDoneView):
    template_name = "accounts/password-reset-done.html"


class PasswordResetConfirmView(authviews.PasswordResetConfirmAndLoginView):
    template_name = "accounts/password-reset-confirm.html"
    form_class = forms.SetPasswordForm
