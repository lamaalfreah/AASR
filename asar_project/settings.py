import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
BASE_DIR = Path(__file__).resolve().parent.parent

SECRET_KEY = os.environ.get("DJANGO_SECRET_KEY", "asar-dev-key-change-me")
DEBUG = os.environ.get("DJANGO_DEBUG", "true").lower() in {"1", "true", "yes", "on"}
ALLOWED_HOSTS = [
    host.strip()
    for host in os.environ.get("DJANGO_ALLOWED_HOSTS", "127.0.0.1,localhost").split(",")
    if host.strip()
]

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
ASAR_LLM_MODEL = os.environ.get("ASAR_LLM_MODEL", "gpt-5.4-mini")
ASAR_LLM_TIMEOUT_SECONDS = float(os.environ.get("ASAR_LLM_TIMEOUT_SECONDS", "45"))
ASAR_MAX_QUESTION_CHARS = int(os.environ.get("ASAR_MAX_QUESTION_CHARS", "6000"))

# Geographic decision-support settings. Public OSM services are appropriate for
# low-volume prototype use only; every endpoint is configurable for a managed or
# self-hosted replacement. Set ASAR_GEO_USER_AGENT to an identifying value that
# includes a project name and contact URL/email before enabling geographic calls.
ASAR_GEO_USER_AGENT = os.environ.get("ASAR_GEO_USER_AGENT", "")
ASAR_NOMINATIM_URL = os.environ.get(
    "ASAR_NOMINATIM_URL", "https://nominatim.openstreetmap.org"
).rstrip("/")
ASAR_OVERPASS_URL = os.environ.get(
    "ASAR_OVERPASS_URL", "https://overpass-api.de/api/interpreter"
)
ASAR_GEO_TIMEOUT_SECONDS = float(os.environ.get("ASAR_GEO_TIMEOUT_SECONDS", "20"))
ASAR_OVERPASS_QUERY_TIMEOUT_SECONDS = int(
    os.environ.get("ASAR_OVERPASS_QUERY_TIMEOUT_SECONDS", "25")
)
ASAR_GEO_CACHE_SECONDS = int(os.environ.get("ASAR_GEO_CACHE_SECONDS", "86400"))
ASAR_GEO_DEFAULT_COUNTRY_CODE = os.environ.get("ASAR_GEO_DEFAULT_COUNTRY_CODE", "sa")
ASAR_GEO_MAX_RADIUS_M = int(os.environ.get("ASAR_GEO_MAX_RADIUS_M", "15000"))
ASAR_GEO_MAX_RESULTS = int(os.environ.get("ASAR_GEO_MAX_RESULTS", "80"))

# These 50/30/20 candidate weights are transparent product design choices. They
# are not learned, scientifically optimized, or a substitute for planning data.
ASAR_CANDIDATE_GAP_WEIGHT = float(os.environ.get("ASAR_CANDIDATE_GAP_WEIGHT", "0.50"))
ASAR_CANDIDATE_ANCHOR_WEIGHT = float(os.environ.get("ASAR_CANDIDATE_ANCHOR_WEIGHT", "0.30"))
ASAR_CANDIDATE_SUPPORT_WEIGHT = float(os.environ.get("ASAR_CANDIDATE_SUPPORT_WEIGHT", "0.20"))

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

CACHES = {
    "default": {"BACKEND": "django.core.cache.backends.locmem.LocMemCache"},
    "geospatial": {
        "BACKEND": "django.core.cache.backends.filebased.FileBasedCache",
        "LOCATION": os.environ.get("ASAR_GEO_CACHE_DIR", "/tmp/asar_geo_cache"),
        "TIMEOUT": ASAR_GEO_CACHE_SECONDS,
        "OPTIONS": {"MAX_ENTRIES": 1000},
    },
}
ASAR_OSM_PBF_PATH = os.getenv(
    "ASAR_OSM_PBF_PATH",
    str(BASE_DIR / "data" / "osm" / "gcc-states-latest.osm.pbf"),
)
ASAR_GOOGLE_MAPS_API_KEY = os.getenv(
    "GOOGLE_MAPS_API_KEY",
    ""
)