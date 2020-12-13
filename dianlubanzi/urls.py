from django.urls import path

from dianlubanzi import views

urlpatterns = [
    path('dlbz/', views.dianlubanzi, name='dianlubanzi'),
]