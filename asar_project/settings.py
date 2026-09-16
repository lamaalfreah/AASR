from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "dashboard",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "asar_project.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "asar_project.wsgi.application"

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR / "db.sqlite3",
    }
}

AUTH_PASSWORD_VALIDATORS = []

LANGUAGE_CODE = "ar"
TIME_ZONE = "Asia/Riyadh"
USE_I18N = True
USE_TZ = True

STATIC_URL = "static/"
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# Production environment configuration. Models/threshold are intentionally not configurable.
import os
DEBUG = os.environ.get('DJANGO_DEBUG', '1') == '1'
SECRET_KEY = os.environ.get('DJANGO_SECRET_KEY', 'asar-local-development-only')
if not DEBUG and SECRET_KEY == 'asar-local-development-only':
    raise RuntimeError('DJANGO_SECRET_KEY is required when DJANGO_DEBUG=0')
ALLOWED_HOSTS = os.environ.get('DJANGO_ALLOWED_HOSTS', 'localhost,127.0.0.1').split(',')
STATIC_ROOT = BASE_DIR / 'staticfiles'
DATA_UPLOAD_MAX_MEMORY_SIZE = 256 * 1024
AASR_MAX_QUESTION_CHARS = 2000
AASR_CATALOG_PATH = Path(os.environ.get('AASR_CATALOG_PATH', BASE_DIR/'dashboard/data/demo_locations.json'))
AASR_ALLOW_CUSTOM_CATALOG = os.environ.get('AASR_ALLOW_CUSTOM_CATALOG','0') == '1'
AASR_MODAL_APP = os.environ.get('AASR_MODAL_APP','')
AASR_LONG_TIMEOUT = int(os.environ.get('AASR_LONG_TIMEOUT','240'))
AASR_CPU_THREADS = int(os.environ.get('AASR_CPU_THREADS','2'))
AASR_LOG_REQUESTS = os.environ.get('AASR_LOG_REQUESTS','0') == '1'
SESSION_COOKIE_SECURE = not DEBUG
CSRF_COOKIE_SECURE = not DEBUG
SECURE_SSL_REDIRECT = not DEBUG
SECURE_HSTS_SECONDS = 31536000 if not DEBUG else 0
LOGGING = {'version':1,'disable_existing_loggers':False,
           'handlers':{'console':{'class':'logging.StreamHandler'}},
           'loggers':{'dashboard':{'handlers':['console'],'level':'INFO','propagate':False}}}
AASR_REQUIRE_LOGIN = os.environ.get('AASR_REQUIRE_LOGIN', '0' if DEBUG else '1') == '1'

AASR_REGISTRY_PATH = Path(os.environ.get('AASR_REGISTRY_PATH', BASE_DIR/'dashboard/data/runtime_registry.json'))
AASR_USE_RUNTIME_REGISTRY = not bool(os.environ.get('AASR_CATALOG_PATH'))
