from __future__ import unicode_literals
from django.utils.encoding import python_2_unicode_compatible
import uuid
from django.db import models
from django.conf import settings
from picklefield.fields import PickledObjectField
from random import randint


class BaseProfile(models.Model):
    user = models.OneToOneField(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, primary_key=True
    )
    slug = models.UUIDField(default=uuid.uuid4, blank=True, editable=False)

    pic_number = randint(1, 9)
    pic_default = "145896-user-avatar-set/male/man (%i).png" % pic_number

    picture = models.ImageField(
        # "Profile picture", upload_to="profile_pics/%Y-%m-%d/", null=True, default="default_profile.png"
        "Profile picture", upload_to="profile_pics/%Y-%m-%d/", null=True, default=pic_default
    )
    bio = models.CharField("Short Bio", max_length=200, blank=True, null=True)
    email_verified = models.BooleanField("Email verified", default=False)

    # Add more user profile fields here. Make sure they are nullable
    # or with default values

    # fields to store portfolio details
    asset_transfers = models.DecimalField(max_digits=20, decimal_places=2, default=0)
    gross_asset_value = models.DecimalField(max_digits=20, decimal_places=2, default=0)
    avail_cash = models.DecimalField(max_digits=20, decimal_places=2, default=0)


    # https://pypi.org/project/django-picklefield/
    portfolio = PickledObjectField(null=True, blank=True)    # lazy method. Just store entire portfolio details as a pickle

    class Meta:
        abstract = True


@python_2_unicode_compatible
class Profile(BaseProfile):
    def __str__(self):
        return "{}'s profile".format(self.user)
