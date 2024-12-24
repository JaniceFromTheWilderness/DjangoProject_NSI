from django.urls import path
from . import views
from django.urls import include
urlpatterns = [
    path('', views.sentiment, name='新聞情緒指標'),
    path('key', views.sentiment_key, name='新聞情緒指標key'),
    path('key_heatmap', views.key_heatmap, name='新聞key_heatmap'),
]