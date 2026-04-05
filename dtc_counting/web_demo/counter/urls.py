from django.urls import path

from .views import dashboard, manual_index, auto_index, run_status, run_result, history_index

urlpatterns = [
    path("", dashboard, name="dashboard"),
    path("manual/", manual_index, name="manual"),
    path("auto/", auto_index, name="auto"),
    path("history/", history_index, name="history"),
    path("status/<str:run_id>/", run_status, name="run_status"),
    path("result/<str:run_id>/", run_result, name="run_result"),
]
