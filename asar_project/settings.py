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
# Frozen research runtime; environment variables configure serving, never the threshold.
ASAR_GOOGLE_MAPS_API_KEY = os.environ.get('ASAR_GOOGLE_MAPS_API_KEY', ASAR_GOOGLE_MAPS_API_KEY)
AASR_SHORT_DIR = BASE_DIR / 'experiments/E5_lightweight_adaptive/step2_neural_short'
AASR_ROUTER_PATH = BASE_DIR / 'experiments/E5_lightweight_adaptive/router_v2/router_v2.joblib'
AASR_MODAL_APP = 'aasr-step2b-qwen3-4b-long'
AASR_MODAL_REVISION = '1cfa9a7208912126459214e8b04321603b3df60c'
AASR_LONG_TIMEOUT = float(os.environ.get('AASR_LONG_TIMEOUT','240'))
AASR_DEFAULT_ANCHOR_NAME = os.environ.get('AASR_DEFAULT_ANCHOR_NAME','وسط الرياض')
AASR_DEFAULT_LAT = float(os.environ.get('AASR_DEFAULT_LAT','24.7136'))
AASR_DEFAULT_LON = float(os.environ.get('AASR_DEFAULT_LON','46.6753'))
# Local demo coverage; expand deliberately and rebuild the spatial cache as needed.
AASR_OSM_BBOX = [float(v) for v in os.environ.get('AASR_OSM_BBOX','46.3,24.3,47.1,25.1').split(',')]
AASR_SEARCH_KM = float(os.environ.get('AASR_SEARCH_KM','15'))
AASR_PARSER_CANDIDATES = 80
DATA_UPLOAD_MAX_MEMORY_SIZE = 256 * 1024

AASR_DEFAULT_AREA_NAME = os.environ.get('AASR_DEFAULT_AREA_NAME','الرياض')
