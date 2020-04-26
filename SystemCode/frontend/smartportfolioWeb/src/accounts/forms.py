from __future__ import unicode_literals
from django.contrib.auth.forms import AuthenticationForm
from django import forms
from crispy_forms.helper import FormHelper
from crispy_forms.layout import Layout, Submit, HTML, Field
from crispy_forms.utils import TEMPLATE_PACK
from authtools import forms as authtoolsforms
from django.contrib.auth import forms as authforms
from django.urls import reverse
from crispy_forms.bootstrap import PrependedText
from django.contrib.auth import get_user_model
from profiles import models as p_models

User = get_user_model()


class LoginForm(AuthenticationForm):
    remember_me = forms.BooleanField(required=False, initial=False)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.helper = FormHelper()
        self.fields["username"].widget.input_type = "email"  # ugly hack

        # self.helper.layout = Layout(
        #     Field("username", placeholder="Enter Email", autofocus=""),
        #     Field("password", placeholder="Enter Password"),
        #     HTML(
        #         '<a href="{}">Forgot Password?</a>'.format(
        #             reverse("accounts:password-reset")
        #         )
        #     ),
        #     Field("remember_me"),
        #     Submit("sign_in", "Log in", css_class="btn btn-lg btn-primary btn-block"),
        # )

        # https://mtik00.com/2015/08/django-and-cripsy-form-login-with-icons/
        self.fields["username"].label = ""
        self.fields["password"].label = ""

        self.helper.layout = Layout(
            PrependedText('username', '<i class="fa fa-envelope-o"></i>', placeholder="Enter Email Address"),
            PrependedText('password', '<i class="fa fa-key"></i>', placeholder="Enter Password"),
            # HTML('<a href="{}">Forgot Password?</a>'.format(
            #     reverse("accounts:password-reset"))),
            Field('remember_me'),
            Submit('sign_in', 'Log in',
                   css_class="btn btn-lg btn-primary btn-block"),
        )


class SignupForm(authtoolsforms.UserCreationForm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.helper = FormHelper()
        self.fields["email"].widget.input_type = "email"  # ugly hack

        self.helper.layout = Layout(
            Field("email", placeholder="Enter Email", autofocus=""),
            Field("name", placeholder="Enter Full Name"),
            Field("password1", placeholder="Enter Password"),
            Field("password2", placeholder="Re-enter Password"),
            Submit("sign_up", "Sign up", css_class="btn-warning"),
        )


class PasswordChangeForm(authforms.PasswordChangeForm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.helper = FormHelper()

        self.helper.layout = Layout(
            Field("old_password", placeholder="Enter old password", autofocus=""),
            Field("new_password1", placeholder="Enter new password"),
            Field("new_password2", placeholder="Enter new password (again)"),
            Submit("pass_change", "Change Password", css_class="btn-warning"),
        )


class PasswordResetForm(authtoolsforms.FriendlyPasswordResetForm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.helper = FormHelper()

        self.helper.layout = Layout(
            Field("email", placeholder="Enter email", autofocus=""),
            Submit("pass_reset", "Reset Password", css_class="btn-warning"),
        )


class SetPasswordForm(authforms.SetPasswordForm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.helper = FormHelper()

        self.helper.layout = Layout(
            Field("new_password1", placeholder="Enter new password", autofocus=""),
            Field("new_password2", placeholder="Enter new password (again)"),
            Submit("pass_change", "Change Password", css_class="btn-warning"),
        )


class CashTransferForm(forms.ModelForm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # add temporary boolean field (boolean fields must have required=False, otherwise Django thinks it is invalid)
        # see https://docs.djangoproject.com/en/dev/ref/forms/fields/#booleanfield
        self.fields['withdraw'] = forms.BooleanField(required=False, widget=forms.CheckboxInput(
            attrs={'class': 'toggle', 'id': 'toggle'}))
        self.helper = FormHelper()
        # self.helper.layout = Layout(Field("name"))

        self.fields["asset_transfers"].label = ""
        self.helper.layout = Layout(
            ToggleSwitch('withdraw', extra_context={'label1': 'Withdraw', 'label2': 'Deposit'}),
            PrependedText('asset_transfers', '<i class="fa fa-money"></i>', placeholder="Enter amount to transfer"),
            Submit("portfolio", "Confirm",
                   css_class="btn btn-lg btn-primary btn-block"),
        )

    def clean_asset_transfers(self):
        data = self.cleaned_data.get("asset_transfers")
        if (data <= 0):
            raise forms.ValidationError("Enter a value more than 0")
        return data

    class Meta:
        model = p_models.Profile
        fields = ["asset_transfers"]
        # fields = '__all__'


# https://codepen.io/twickstrom/pen/ECfot
# https://simpleisbetterthancomplex.com/tutorial/2018/11/28/advanced-form-rendering-with-django-crispy-forms.html
# https://stackoverflow.com/questions/23223443/how-can-extra-context-be-passed-to-django-crispy-forms-field-templates
# https://speckyboy.com/toggle-switch-css/
class ToggleSwitch(Field):
    template = 'accounts/css_toggle_switch.html'
    extra_context = {}

    def __init__(self, *args, **kwargs):
        self.extra_context = kwargs.pop('extra_context', self.extra_context)
        super(ToggleSwitch, self).__init__(*args, **kwargs)

    # def render(self, form, form_style, context, extra_context=None, **kwargs):
    def render(self, form, form_style, context, template_pack=TEMPLATE_PACK, extra_context=None, **kwargs):
        if self.extra_context:
            extra_context = extra_context.update(self.extra_context) if extra_context else self.extra_context
        return super(ToggleSwitch, self).render(form, form_style, context, template_pack, extra_context, **kwargs)
