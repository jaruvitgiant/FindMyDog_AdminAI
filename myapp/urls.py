# urls.py
from django.urls import path
from django.contrib.auth.views import LogoutView
from . import views
from . import PMai_views

urlpatterns = [
    path("", views.home, name="home"),
    path("home/", views.home, name="home"),
    path("register/", views.register, name="register"),
    path('logout/', LogoutView.as_view(), name='logout'),
    path("login/", views.login, name="login"),
    path('dog/register/', views.register_dog_page, name='register_dog_page'),
    path('dogs/', views.dog_list, name='dog_list'),
    path('dogsall/', views.dog_all_list, name='dog_all_list'),
    path('dogs/<int:dog_id>/', views.dog_detail, name='dog_detail'),
    path('notifications/', views.notification_list_view, name='notification_list'),
    path('notifications/create/', views.create_notification_view, name='create_notification'),
    path('notifications/<int:notification_id>/detail_hx/', views.notification_detail_hx_view, name='notification_detail_hx'),
    path('notifications/<int:notification_id>/edit/', views.edit_notification_view, name='edit_notification'),
    path('notifications/<int:notification_id>/delete/', views.delete_notification_view, name='delete_notification'),
    path('dogs/<int:dog_id>/delete/', views.delete_dog_page, name='delete_dog_page'),
    
    # Admin Training Routes
    path('admin_page/', views.admin_page, name='admin_page'),
    path('page_training/', PMai_views.page_training, name='page_training'),
    path('SetautoTraining/', PMai_views.set_auto_training, name='set_auto_training'),

    #training
    path('start-training/', PMai_views.start_training, name='start_training'),
    path('train_model/', PMai_views.train_resnet18, name='train_model'),


]
