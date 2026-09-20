from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
import hashlib
import math
import re

from django.conf import settings

from language.long_parser import LongParser
from language.modal_checkpoint import ModalCheckpointProvider
from language.neural_short import (
    NeuralShortParser,
    PLURALS,
    ShortPrediction,
    encode,
    extract_slots,
    normalize_words,
)
from language.router_v2 import RouterV2

from spatial import execute
from spatial.geometry import distance
from spatial.identity import resolve
from spatial.schema import (
    CATEGORIES,
    DIRECTIONS,
    EntityReference,
    Location,
    OPERATIONS,
    ParsedContext,
    QueryError,
)

from .geo.client import OSMClient


# =========================================================
# Runtime configuration
# =========================================================

MODAL_APP_NAME = "aasr-step2b-qwen3-4b-long"
MODAL_CLASS_NAME = "QwenLong"

QWEN_REVISION = (
    "1cfa9a7208912126459214e8b04321603b3df60c"
)


AR_TO_OSM = {
    "مستشفى": "hospital",
    "عيادة": "clinic",
    "صيدلية": "pharmacy",
    "مدرسة": "school",
    "جامعة": "university",
    "كلية": "college",
    "مطعم": "restaurant",
    "مقهى": "cafe",
    "بنك": "bank",
    "محطة وقود": "fuel",
    "مطعم وجبات سريعة": "fast_food",
    "صراف آلي": "atm",
    "روضة أطفال": "kindergarten",
    "مكان عبادة": "place_of_worship",
}


SLOT_ERRORS = {
    "count_origin_not_explicit_unique": ["origin"],
    "origin_not_explicit": ["origin"],
    "wrong_named_target_count": ["targets"],
    "category_slot_count": ["categories"],
    "two_category_slot_count": ["categories"],
    "missing_or_conflicting_radius": ["radius_km"],
    "direction_slot_count": ["direction"],
}


# =========================================================
# Runtime errors
# =========================================================

class AdaptiveRuntimeError(RuntimeError):
    pass


class AdaptiveConfigurationError(AdaptiveRuntimeError):
    pass


class AdaptiveInputError(AdaptiveRuntimeError):
    pass


class AdaptiveLongUnavailableError(AdaptiveRuntimeError):
    """
    LONG was selected, but remote inference / strict JSON failed.

    If SHORT already produced a valid query, runtime may
    safely fall back to it.
    """

    pass


# =========================================================
# Frozen models
# =========================================================

@lru_cache(maxsize=1)
def get_short_parser() -> NeuralShortParser:
    artifact_dir = (
        Path(settings.BASE_DIR)
        / "experiments"
        / "E5_lightweight_adaptive"
        / "step2_neural_short"
    )

    return NeuralShortParser(
        artifact_dir
    )


@lru_cache(maxsize=1)
def get_router() -> RouterV2:
    artifact_path = (
        Path(settings.BASE_DIR)
        / "experiments"
        / "E5_lightweight_adaptive"
        / "router_v2"
        / "router_v2.joblib"
    )

    return RouterV2(
        artifact_path
    )


# =========================================================
# 1. Frozen BiGRU
# =========================================================

def classify_short(question: str):
    """
    BiGRU always runs first.

    No hand-written task selection happens here.
    """

    parser = get_short_parser()
    torch = parser.torch

    ids = encode(
        question,
        parser.vocabulary,
    )

    with torch.inference_mode():
        probs = parser.model(
            torch.tensor(
                [ids],
                dtype=torch.long,
            ),
            torch.tensor(
                [len(ids)]
            ),
        ).softmax(-1)[0]

    probabilities = [
        float(value)
        for value in probs.tolist()
    ]

    index = int(
        probs.argmax()
    )

    operation = OPERATIONS[index]

    confidence = float(
        probs[index]
    )

    return (
        operation,
        confidence,
        probabilities,
    )


# =========================================================
# 2. Natural-language place resolution
#
# IMPORTANT:
# This does NOT decide the spatial operation.
# BiGRU already handles operation classification.
#
# This layer only extracts visible named places.
# =========================================================

CATEGORY_FORMS = set(CATEGORIES)

for category, aliases in PLURALS.items():
    CATEGORY_FORMS.add(category)

    for alias in aliases:
        CATEGORY_FORMS.add(alias)


GENERIC_CATEGORY_NORMS = set()

for value in CATEGORY_FORMS:
    norm = normalize_words(value)

    GENERIC_CATEGORY_NORMS.add(norm)

    GENERIC_CATEGORY_NORMS.add(
        normalize_words(
            "ال" + value
        )
    )


DIRECTION_NORMS = {
    normalize_words(value)
    for value in DIRECTIONS
}


PLACE_HEADS = {
    "جامعة",
    "مستشفى",
    "عيادة",
    "صيدلية",
    "كلية",
    "مدرسة",
    "روضة",
    "مطار",
    "حي",
    "شارع",
    "طريق",
    "مدينة",
    "مركز",
    "مول",
    "مجمع",
    "برج",
    "حديقة",
    "محطة",
    "سوق",
    "مستوصف",
    "بنك",
    "مقهى",
    "مطعم",
    "مسجد",
}


ENTITY_STOP_WORDS = {
    "من",
    "الى",
    "إلى",
    "حول",
    "قرب",
    "عند",
    "بجوار",
    "داخل",
    "ضمن",
    "على",
    "بعد",
    "كم",
    "كيلومتر",
    "كيلومترات",
    "او",
    "أو",
    "ام",
    "أم",
    "ولا",
    "ثم",
    "بعدها",
    "وبعدها",
    "بعدين",
    "وش",
    "ايش",
    "إيش",
    "أيش",
    "ما",
    "هل",
    "كم",
    "ايهما",
    "أيهما",
}


LEADING_NOISE_PATTERNS = (
    r"^(?:لو\s+سمحت|من\s+فضلك)\s+",
    r"^(?:ابي|أبي|ابغى|أبغى|اريد|أريد|احتاج|أحتاج)\s+",
    r"^(?:اعرف|أعرف|اعطني|أعطني|قولي|قل\s+لي)\s+",
    r"^(?:اذا|إذا)\s+",
    r"^(?:ما|ماذا|هل|أين|اين|وين|وش|ايش|إيش|أيش)\s+",
    r"^(?:هو|هي)\s+",
    r"^(?:حدد|حددي|دور|ابحث|أبحث|استخدم)\s+",
    r"^(?:اولا|أولا|أولاً|اولاً|اول|أول)\s+",
    r"^(?:ثم|بعدها|وبعدها|بعدين|وبعدين)\s+",
    r"^(?:في\s+أي\s+اتجاه|في\s+اي\s+اتجاه|ما\s+اتجاه|اتجاه)\s+",
    r"^(?:يقع|تقع)\s+",
    r"^(?:أي\s+الموقعين|اي\s+الموقعين|أيهما|ايهما)\s+",
    r"^(?:الأقرب|الاقرب|أقرب|اقرب)\s+",
    r"^(?:كم\s+عدد|كم)\s+",
    r"^(?:هل\s+يوجد|هل\s+فيه|يوجد|فيه)\s+",
)


REFERENCE_PATTERN = re.compile(
    r"(?:"
    r"بالنسبة\s+(?:إلى|الى)"
    r"|بالنسبة\s+ل"
    r"|انطلاق(?:ا|ًا)\s+من"
    r"|ابتداء\s+من"
    r"|بالقرب\s+من"
    r"|قريب(?:ة)?\s+من"
    r"|من\s+عند"
    r"|بجوار"
    r"|حول"
    r"|قرب"
    r"|عند"
    r"|(?:إلى|الى)"
    r"|(?<!\S)من"
    r")"
    r"\s+"
    r"(?!"
    r"هذا\s+(?:الموقع|المكان)"
    r"|الموقع\s+نفسه"
    r"|المكان\s+نفسه"
    r"|المكان\s+المرجعي"
    r")"
    r"(?P<place>«[^»]+»|[^،,؟?!:؛]+)",
    re.S,
)


ATTACHED_L_PATTERN = re.compile(
    r"(?:^|\s)"
    r"ل(?:ـ)?"
    r"(?P<place>"
    r"(?:ال)?"
    r"(?:جامعة|مستشفى|عيادة|صيدلية|كلية|مدرسة|"
    r"مطار|حي|شارع|طريق|مدينة|مركز|مول|مجمع|"
    r"برج|حديقة|محطة|سوق|مستوصف|بنك|مقهى|مطعم)"
    r"\s+"
    r"[^،,؟?!:؛]+"
    r")",
    re.S,
)


CHUNK_SPLITTER = re.compile(
    r"[،,:؛;؟?!]+"
    r"|"
    r"\s+(?:"
    r"أو|او|أم|ام|ولا|"
    r"ثم|وبعدها|بعدها|وبعدين|بعدين|"
    r"بالنسبة\s+(?:إلى|الى|ل)|"
    r"انطلاق(?:ا|ًا)\s+من|"
    r"ابتداء\s+من|"
    r"بالقرب\s+من|"
    r"قريب(?:ة)?\s+من|"
    r"بجوار|حول|قرب|عند|"
    r"إلى|الى|من"
    r")\s+",
    re.S,
)


def _strip_article(value: str) -> str:
    value = value.strip()

    if value.startswith("ال"):
        return value[2:]

    return value


def _clean_place(value: str) -> str:
    value = " ".join(
        str(value).strip().split()
    )

    value = value.strip(
        "«»\"' ،,؛;؟?!.:"
    )

    # لجامعة الملك سعود -> جامعة الملك سعود
    value = re.sub(
        r"^ل(?:ـ)?(?=(?:ال)?(?:"
        r"جامعة|مستشفى|عيادة|صيدلية|كلية|مدرسة|"
        r"مطار|حي|شارع|طريق|مدينة|مركز|مول|مجمع|"
        r"برج|حديقة|محطة|سوق|مستوصف|بنك|مقهى|مطعم"
        r")\b)",
        "",
        value,
    )

    # Stop before radius / constraint / sequence phrases.
    value = re.split(
        r"\s+(?:"
        r"داخل|ضمن|على\s+بعد|في\s+نطاق|بنطاق|بمسافة|"
        r"ثم|وبعدها|بعدها|وبعدين|بعدين"
        r")\s+",
        value,
        maxsplit=1,
    )[0]

    # Natural-language question continuing after entity name.
    value = re.split(
        r"\s+(?:"
        r"وش|ايش|إيش|أيش|هل|كم|أيهما|ايهما"
        r")\s+",
        value,
        maxsplit=1,
    )[0]

    changed = True

    while changed:
        changed = False

        for pattern in LEADING_NOISE_PATTERNS:
            cleaned = re.sub(
                pattern,
                "",
                value,
                count=1,
            ).strip()

            if cleaned != value:
                value = cleaned
                changed = True

    value = re.sub(
        r"\s+(?:تقريبًا|تقريبا|لو\s+سمحت|من\s+فضلك)$",
        "",
        value,
    ).strip()

    return value.strip(
        "«»\"' ،,؛;؟?!.:"
    )


def _is_generic_category(value: str) -> bool:
    norm = normalize_words(
        value
    )

    if norm in GENERIC_CATEGORY_NORMS:
        return True

    no_article = _strip_article(
        norm
    )

    return (
        no_article
        in {
            _strip_article(x)
            for x in GENERIC_CATEGORY_NORMS
        }
    )


def _is_non_place(value: str) -> bool:
    value = _clean_place(
        value
    )

    if not value:
        return True

    if len(value) > 140:
        return True

    norm = normalize_words(
        value
    )

    if _is_generic_category(
        value
    ):
        return True

    if norm in DIRECTION_NORMS:
        return True

    if re.fullmatch(
        r"[\d.\s]+(?:كم|كيلومتر|كيلومترات)?",
        norm,
    ):
        return True

    non_places = (
        "هذا الموقع",
        "هذا المكان",
        "الموقع نفسه",
        "المكان نفسه",
        "المكان المرجعي",
        "نقطة مرجعية",
        "نقطه مرجعيه",
        "اسم المعلم النهائي",
        "المعلم النهائي",
    )

    if any(
        normalize_words(item)
        in norm
        for item in non_places
    ):
        return True

    if not re.search(
        r"[A-Za-z\u0600-\u06FF]",
        value,
    ):
        return True

    return False


def _add_unique(
    items: list[str],
    value: str,
):
    value = _clean_place(
        value
    )

    if (
        value
        and not _is_non_place(value)
        and value not in items
    ):
        items.append(
            value
        )


def _head_based_names(
    question: str,
) -> list[str]:
    """
    Find natural named-place phrases such as:

    جامعة الملك سعود
    مستشفى دلة
    مستشفى المملكة
    صيدلية النهدي

    without deciding any spatial operation.
    """

    tokens = re.findall(
        r"«[^»]+»|[\w\-]+",
        question,
        flags=re.UNICODE,
    )

    results = []

    normalized_heads = {
        normalize_words(x)
        for x in PLACE_HEADS
    }

    normalized_stops = {
        normalize_words(x)
        for x in ENTITY_STOP_WORDS
    }

    for index, token in enumerate(tokens):
        raw = token.strip(
            "«»"
        )

        norm = normalize_words(
            raw
        )

        head = _strip_article(
            norm
        )

        if head not in normalized_heads:
            continue

        collected = [
            raw
        ]

        for next_token in tokens[
            index + 1:
            index + 6
        ]:
            next_raw = next_token.strip(
                "«»"
            )

            next_norm = normalize_words(
                next_raw
            )

            if next_norm in normalized_stops:
                break

            if re.fullmatch(
                r"\d+(?:\.\d+)?",
                next_raw,
            ):
                break

            collected.append(
                next_raw
            )

        candidate = " ".join(
            collected
        )

        _add_unique(
            results,
            candidate,
        )

    return results


def _named_chunks(
    question: str,
) -> list[str]:
    """
    Collect likely named entities from natural language.

    This function never decides the operation.
    """

    names = []

    # Explicit quoted entities.
    for quoted in re.findall(
        r"«([^»]+)»",
        question,
        flags=re.S,
    ):
        _add_unique(
            names,
            quoted,
        )

    # Natural institutional / location names.
    for candidate in _head_based_names(
        question
    ):
        _add_unique(
            names,
            candidate,
        )

    # General fallback chunks:
    # الرياض
    # الدرعية
    # مركز الملك عبدالله المالي
    for chunk in CHUNK_SPLITTER.split(
        question
    ):
        candidate = _clean_place(
            chunk
        )

        if len(
            candidate.split()
        ) <= 8:
            _add_unique(
                names,
                candidate,
            )

    return names


def _anchor_candidates(
    question: str,
) -> list[str]:
    """
    Return likely spatial reference places in confidence order.

    Explicit reference wording is preferred.
    """

    positioned = []

    for match in REFERENCE_PATTERN.finditer(
        question
    ):
        candidate = _clean_place(
            match.group(
                "place"
            )
        )

        if not _is_non_place(
            candidate
        ):
            positioned.append(
                (
                    match.start(),
                    candidate,
                )
            )

    for match in ATTACHED_L_PATTERN.finditer(
        question
    ):
        candidate = _clean_place(
            match.group(
                "place"
            )
        )

        if not _is_non_place(
            candidate
        ):
            positioned.append(
                (
                    match.start(),
                    candidate,
                )
            )

    positioned.sort(
        key=lambda item:
            item[0],
        reverse=True,
    )

    results = []

    for _, candidate in positioned:
        _add_unique(
            results,
            candidate,
        )

    for candidate in _named_chunks(
        question
    ):
        _add_unique(
            results,
            candidate,
        )

    return results


# =========================================================
# 3. Geocoding
#
# Priority:
# 1. Nominatim
# 2. Google Maps fallback
#
# Google is optional. If there is no key, current behavior
# continues normally with Nominatim.
# =========================================================

def geocode(
    client: OSMClient,
    name: str,
) -> Location:
    """
    Resolve a visible place name.

    Priority:
    1. Nominatim / OpenStreetMap
    2. Google Maps fallback, if configured

    Existing OSM/Nominatim behavior remains unchanged.
    """

    queries = (
        name,
        f"{name}, الرياض",
        f"{name}, المملكة العربية السعودية",
    )

    # -----------------------------------------------------
    # 1. Nominatim first
    # -----------------------------------------------------

    for query in dict.fromkeys(
        queries
    ):
        try:
            rows = client.geocode_place(
                query,
                settings.ASAR_GEO_DEFAULT_COUNTRY_CODE,
            )

        except Exception:
            rows = []

        if not rows:
            continue

        row = rows[0]

        return Location(
            row["source_ref"],
            name,
            float(
                row["lat"]
            ),
            float(
                row["lng"]
            ),
            None,
        )

    # -----------------------------------------------------
    # 2. Google Maps fallback
    # -----------------------------------------------------

    google_api_key = getattr(
        settings,
        "ASAR_GOOGLE_MAPS_API_KEY",
        "",
    )

    if google_api_key:
        for query in dict.fromkeys(
            queries
        ):
            try:
                rows = (
                    client
                    .google_geocode_place(
                        query
                    )
                )

            except Exception:
                rows = []

            if not rows:
                continue

            row = rows[0]

            return Location(
                row["source_ref"],
                name,
                float(
                    row["lat"]
                ),
                float(
                    row["lng"]
                ),
                None,
            )

    # -----------------------------------------------------
    # 3. Nothing found
    # -----------------------------------------------------

    raise AdaptiveInputError(
        f"تعذر العثور على الموقع: {name}"
    )


def _try_geocode(
    client: OSMClient,
    name: str,
) -> Location | None:
    try:
        return geocode(
            client,
            name,
        )

    except AdaptiveInputError:
        return None


def build_seed_context(
    question: str,
    operation: str | None = None,
) -> ParsedContext:
    """
    Build minimal grounding context from natural Arabic.

    `operation` is accepted for compatibility only.
    Place extraction does NOT determine the spatial task.
    """

    anchor_names = _anchor_candidates(
        question
    )

    all_names = _named_chunks(
        question
    )

    if not anchor_names:
        anchor_names = list(
            all_names
        )

    if not anchor_names:
        raise AdaptiveInputError(
            "تعذر تحديد اسم المكان المرجعي من السؤال."
        )

    client = OSMClient()

    try:
        anchor = None

        # Try highest-confidence origin candidates.
        for name in anchor_names[:6]:
            anchor = _try_geocode(
                client,
                name,
            )

            if anchor is not None:
                break

        if anchor is None:
            raise AdaptiveInputError(
                "تعذر تحديد المكان المرجعي بوضوح."
            )

        candidates = []

        used_refs = {
            anchor.identity
        }

        # Ground other visible names too.
        #
        # This is important because LONG may correct an initially
        # wrong BiGRU operation and still need those entities.
        for name in all_names:
            if name == anchor.name:
                continue

            location = _try_geocode(
                client,
                name,
            )

            if location is None:
                continue

            if location.identity in used_refs:
                continue

            used_refs.add(
                location.identity
            )

            candidates.append(
                location
            )

            if len(candidates) >= 4:
                break

    finally:
        client.close()

    return ParsedContext(
        anchor,
        tuple(
            candidates
        ),
        (),
    )


# =========================================================
# 4. Protect proper names before Short slot extraction
# =========================================================

def prepare_question_for_slots(
    question: str,
    context: ParsedContext,
) -> str:
    """
    Example:

    جامعة الملك سعود

    becomes internally:

    «جامعة الملك سعود»

    so "جامعة" inside the proper name is not mistakenly
    treated as a requested POI category.
    """

    names = []

    if context.anchor:
        names.append(
            context.anchor.name
        )

    names.extend(
        point.name
        for point in context.candidates
    )

    names = sorted(
        {
            name
            for name in names
            if name
        },
        key=len,
        reverse=True,
    )

    prepared = question

    for name in names:
        quoted_name = (
            f"«{name}»"
        )

        if quoted_name in prepared:
            continue

        prepared = prepared.replace(
            name,
            quoted_name,
        )

    return prepared


# =========================================================
# 5. Short Structured Query
# =========================================================

def build_short_prediction(
    question: str,
    context: ParsedContext,
    operation: str,
    confidence: float,
) -> ShortPrediction:
    slot_question = (
        prepare_question_for_slots(
            question,
            context,
        )
    )

    try:
        query = extract_slots(
            operation,
            slot_question,
            context,
        )

        return ShortPrediction(
            operation,
            confidence,
            query,
            "success",
        )

    except QueryError as exc:
        return ShortPrediction(
            operation,
            confidence,
            None,
            exc.status,
            exc.reason,
        )


# =========================================================
# 6. Router V2 features
# =========================================================

def build_router_features(
    question,
    context,
    prediction,
    probabilities,
):
    query = prediction.query

    missing = (
        []
        if query
        else SLOT_ERRORS.get(
            prediction.reason
        )
    )

    if query:
        refs = (
            query.origin,
        ) + query.targets

    else:
        visible_names = {
            point.name
            for point in context.candidates
        }

        if context.anchor:
            visible_names.add(
                context.anchor.name
            )

        refs = tuple(
            EntityReference(
                name,
                "unspecified",
            )
            for name in visible_names
            if (
                name
                and name in question
            )
        )

    ambiguity_count = sum(
        resolve(
            reference,
            context,
        ).status
        == "AMBIGUOUS"
        for reference in refs
    )

    ranked = sorted(
        probabilities,
        reverse=True,
    )

    constraint_count = None

    if query:
        constraint_count = (
            len(
                query.targets
            )
            + len(
                query.categories
            )
            + int(
                query.radius_km
                is not None
            )
            + int(
                query.direction
                is not None
            )
            + int(
                query.first_hop_category
                is not None
            )
            + int(
                query.second_hop_category
                is not None
            )
        )

    return {
        "short_confidence":
            float(
                prediction.confidence
            ),

        "short_entropy":
            -sum(
                probability
                * math.log(
                    probability
                )
                for probability
                in probabilities
                if probability > 0
            ),

        "short_top1_top2_margin":
            (
                ranked[0]
                - ranked[1]
            ),

        "query_valid":
            query is not None,

        "missing_slots":
            missing,

        "entity_ambiguity_count":
            ambiguity_count,

        "question_length_chars":
            len(
                question
            ),

        "question_length_words":
            len(
                question.split()
            ),

        "predicted_operation":
            prediction.operation,

        "constraint_count":
            constraint_count,

        "operation_probabilities":
            dict(
                zip(
                    OPERATIONS,
                    probabilities,
                )
            ),
    }


# =========================================================
# 7. LONG — frozen Qwen3-4B on Modal
# =========================================================

def run_long(
    question: str,
    context: ParsedContext,
):
    try:
        import modal

    except ImportError as exc:
        raise AdaptiveConfigurationError(
            "Modal Python package is not installed."
        ) from exc

    try:
        ModalQwen = modal.Cls.from_name(
            MODAL_APP_NAME,
            MODAL_CLASS_NAME,
        )

        remote_model = ModalQwen(
            revision=QWEN_REVISION
        )

        provider = ModalCheckpointProvider(
            remote_model.generate.remote,

            (
                Path(
                    settings.BASE_DIR
                )
                / ".runtime"
                / "modal_qwen_calls"
            ),

            {
                "revision":
                    QWEN_REVISION
            },
        )

        case_id = hashlib.sha256(
            question.encode(
                "utf-8"
            )
        ).hexdigest()[:24]

        provider.begin_case(
            case_id
        )

        outcome = LongParser(
            provider
        ).parse(
            question,
            context,
        )

    except Exception as exc:
        raise AdaptiveLongUnavailableError(
            "Modal Qwen3-4B LONG path failed: "
            f"{exc}"
        ) from exc

    # Infrastructure/schema failure.
    #
    # If SHORT already produced a valid query, analyze()
    # can safely fall back to it.
    if outcome.status == "invalid_output":
        raise AdaptiveLongUnavailableError(
            outcome.message
            or
            "Qwen returned invalid structured output."
        )

    # Genuine language abstention should not be silently
    # converted into a different operation.
    if (
        outcome.status
        != "success"
        or outcome.query
        is None
    ):
        raise AdaptiveInputError(
            outcome.message
            or
            "السؤال يحتاج توضيحًا أو يطلب عملية خارج نطاق النظام."
        )

    return outcome.query


# =========================================================
# 8. OSM category helpers
# =========================================================

def query_categories(
    query,
) -> list[str]:
    categories = list(
        query.categories
    )

    if query.first_hop_category:
        categories.append(
            query.first_hop_category
        )

    if query.second_hop_category:
        categories.append(
            query.second_hop_category
        )

    return list(
        dict.fromkeys(
            categories
        )
    )


def fetch_pois(
    client,
    origin,
    categories,
    radius_m,
):
    osm_categories = [
        AR_TO_OSM[
            category
        ]
        for category in categories
        if category in AR_TO_OSM
    ]

    if (
        len(osm_categories)
        != len(categories)
    ):
        unsupported = [
            category
            for category in categories
            if category not in AR_TO_OSM
        ]

        raise AdaptiveInputError(
            "فئة مكانية غير مدعومة: "
            + ", ".join(
                unsupported
            )
        )

    response = (
        client
        .search_nearby_pois(
            origin.latitude,
            origin.longitude,
            osm_categories,
            radius_m,
        )
    )

    reverse = {
        AR_TO_OSM[
            arabic
        ]:
            arabic
        for arabic in categories
    }

    locations = []

    for item in response.get(
        "items",
        [],
    ):
        category = reverse.get(
            item.get(
                "category"
            )
        )

        if not category:
            continue

        locations.append(
            Location(
                item[
                    "source_ref"
                ],
                item[
                    "name"
                ],
                float(
                    item[
                        "lat"
                    ]
                ),
                float(
                    item[
                        "lng"
                    ]
                ),
                category,
            )
        )

    return (
        locations,
        response.get(
            "data_timestamp"
        ),
    )


# =========================================================
# 9. Progressive nearest search
# =========================================================

def fetch_nearest_pois(
    client,
    origin,
    categories,
):
    """
    Search progressively:

    3 km -> 7 km -> configured maximum.

    For two-category comparison, do not stop until every
    required category has at least one candidate.
    """

    max_radius = int(
        settings
        .ASAR_GEO_MAX_RADIUS_M
    )

    radii = [
        3000,
        7000,
        max_radius,
    ]

    radii = [
        radius
        for radius
        in dict.fromkeys(
            radii
        )
        if (
            100
            <= radius
            <= max_radius
        )
    ]

    required = set(
        categories
    )

    last_locations = []
    last_timestamp = None

    for radius_m in radii:
        (
            locations,
            timestamp,
        ) = fetch_pois(
            client,
            origin,
            categories,
            radius_m,
        )

        last_locations = (
            locations
        )

        last_timestamp = (
            timestamp
            or last_timestamp
        )

        found = {
            point.category
            for point in locations
        }

        if required.issubset(
            found
        ):
            return (
                locations,
                last_timestamp,
            )

    return (
        last_locations,
        last_timestamp,
    )


# =========================================================
# 10. Live geographic context
# =========================================================

def enrich_context(
    query,
    seed,
):
    origin_resolution = resolve(
        query.origin,
        seed,
    )

    if (
        origin_resolution.status
        != "UNIQUE"
    ):
        return (
            seed,
            None,
        )

    origin = (
        origin_resolution
        .matches[0]
    )

    radius_km = (
        query.radius_km
        if query.radius_km
        is not None
        else (
            settings
            .ASAR_GEO_MAX_RADIUS_M
            / 1000
        )
    )

    radius_m = max(
        100,
        min(
            int(
                radius_km
                * 1000
            ),
            settings
            .ASAR_GEO_MAX_RADIUS_M,
        ),
    )

    candidates = list(
        seed.candidates
    )

    identities = {
        point.identity
        for point in candidates
    }

    timestamp = None

    client = OSMClient()

    try:
        # -------------------------------------------------
        # TWO-HOP
        # -------------------------------------------------

        if (
            query.operation
            == "two_hop_nearest"
        ):
            (
                first_candidates,
                timestamp,
            ) = fetch_nearest_pois(
                client,
                origin,
                [
                    query
                    .first_hop_category
                ],
            )

            for point in first_candidates:
                if (
                    point.identity
                    not in identities
                ):
                    candidates.append(
                        point
                    )

                    identities.add(
                        point.identity
                    )

            if first_candidates:
                first_nearest = min(
                    first_candidates,
                    key=lambda point:
                        distance(
                            origin.coordinates,
                            point.coordinates,
                        ),
                )

                (
                    second_candidates,
                    second_timestamp,
                ) = fetch_nearest_pois(
                    client,
                    first_nearest,
                    [
                        query
                        .second_hop_category
                    ],
                )

                timestamp = (
                    second_timestamp
                    or timestamp
                )

                for point in second_candidates:
                    if (
                        point.identity
                        not in identities
                    ):
                        candidates.append(
                            point
                        )

                        identities.add(
                            point.identity
                        )

        # -------------------------------------------------
        # OTHER OPERATIONS
        # -------------------------------------------------

        else:
            categories = (
                query_categories(
                    query
                )
            )

            if categories:
                if (
                    query.operation
                    in {
                        "nearest_category",
                        "nearest_of_two_categories",
                    }
                    and query.radius_km
                    is None
                ):
                    (
                        points,
                        timestamp,
                    ) = fetch_nearest_pois(
                        client,
                        origin,
                        categories,
                    )

                else:
                    (
                        points,
                        timestamp,
                    ) = fetch_pois(
                        client,
                        origin,
                        categories,
                        radius_m,
                    )

                for point in points:
                    if (
                        point.identity
                        not in identities
                    ):
                        candidates.append(
                            point
                        )

                        identities.add(
                            point.identity
                        )

    finally:
        client.close()

    return (
        ParsedContext(
            seed.anchor,
            tuple(
                candidates
            ),
            (),
        ),
        timestamp,
    )


# =========================================================
# 11. Frontend helpers
# =========================================================

def point_to_dict(
    location,
    role,
    origin,
    step=None,
):
    parts = (
        location
        .identity
        .split(":")
    )

    is_osm = (
        location
        .identity
        .startswith(
            "osm:"
        )
    )

    is_google = (
        location
        .identity
        .startswith(
            "google:"
        )
    )

    if is_osm:
        source = "openstreetmap"

    elif is_google:
        source = "google"

    else:
        source = "nominatim"

    return {
        "name":
            location.name,

        "lat":
            location.latitude,

        "lng":
            location.longitude,

        "role":
            role,

        "category":
            location.category,

        "distance_km":
            (
                None
                if (
                    location.identity
                    == origin.identity
                )
                else round(
                    distance(
                        origin.coordinates,
                        location.coordinates,
                    ),
                    3,
                )
            ),

        "step":
            step,

        "score":
            None,

        "source_ref":
            location.identity,

        "source":
            source,

        "osm_type":
            (
                parts[1]
                if (
                    is_osm
                    and len(parts) >= 3
                    and parts[1]
                    in {
                        "node",
                        "way",
                        "relation",
                    }
                )
                else None
            ),

        "osm_id":
            (
                parts[2]
                if (
                    is_osm
                    and len(parts) >= 3
                )
                else None
            ),
    }


def nearest_location(
    context,
    origin,
    category,
):
    candidates = [
        point
        for point in context.candidates
        if (
            point.category
            == category
        )
    ]

    if not candidates:
        return None

    return min(
        candidates,
        key=lambda point:
            distance(
                origin.coordinates,
                point.coordinates,
            ),
    )


# =========================================================
# 12. Response builder
# =========================================================

def build_response(
    question,
    route,
    router_selected_route,
    fallback_used,
    long_probability,
    query,
    execution,
    context,
    timestamp,
):
    origin_resolution = resolve(
        query.origin,
        context,
    )

    origin = (
        origin_resolution
        .matches[0]
        if (
            origin_resolution.status
            == "UNIQUE"
        )
        else context.anchor
    )

    common_evidence = [
        {
            "label":
                "Executed route",

            "value":
                route,
        },
        {
            "label":
                "Router selection",

            "value":
                router_selected_route,
        },
        {
            "label":
                "Long probability",

            "value":
                round(
                    long_probability,
                    4,
                ),
        },
    ]

    if fallback_used:
        common_evidence.append(
            {
                "label":
                    "Fallback",

                "value":
                    "LONG unavailable → valid SHORT query",
            }
        )

    # -----------------------------------------------------
    # No qualifying spatial result
    # -----------------------------------------------------

    if (
        execution.status
        != "success"
        or execution.answer
        is None
        or origin
        is None
    ):
        return {
            "status":
                "needs_clarification",

            "task_type":
                query.operation,

            "reasoning_depth":
                (
                    "deep"
                    if route == "LONG"
                    else
                    "concise"
                ),

            "data_mode":
                "osm_assisted",

            "answer": {
                "value":
                    None,

                "text":
                    (
                        "لم أجد نتيجة مكانية تحقق "
                        "الشروط المطلوبة ضمن البيانات المتاحة."
                    ),
            },

            "reasoning":
                (
                    execution.reason
                    or
                    "تعذر التنفيذ المكاني."
                ),

            "visualization":
                "none",

            "anchor":
                (
                    point_to_dict(
                        origin,
                        "anchor",
                        origin,
                    )
                    if origin
                    else None
                ),

            "locations":
                [],

            "metrics": {
                "distance_km":
                    None,

                "radius_km":
                    query.radius_km,

                "nearest_distance_km":
                    None,

                "count":
                    None,

                "direction":
                    None,

                "suitability":
                    None,
            },

            "evidence":
                common_evidence,

            "comparison":
                [],

            "constraints":
                [],

            "sources":
                [],

            "limitations": [
                (
                    "تعتمد النتيجة على اكتمال "
                    "البيانات الجغرافية المتاحة."
                )
            ],

            "route":
                route,

            "router_selected_route":
                router_selected_route,

            "fallback_used":
                fallback_used,

            "router_long_probability":
                round(
                    long_probability,
                    6,
                ),

            "structured_query":
                asdict(
                    query
                ),
        }

    selected_ids = {
        item.get(
            "identity"
        )
        for item
        in execution.trace.get(
            "selected_candidates",
            [],
        )
        if isinstance(
            item,
            dict,
        )
    }

    first_hop_ids = {
        item.get(
            "identity"
        )
        for item
        in execution.trace.get(
            "first_hop_result",
            [],
        )
        if isinstance(
            item,
            dict,
        )
    }

    target_names = {
        target.name
        for target in query.targets
    }

    finalist_ids = set()

    if (
        query.operation
        == "nearest_of_two_categories"
    ):
        for category in query.categories:
            finalist = nearest_location(
                context,
                origin,
                category,
            )

            if finalist:
                finalist_ids.add(
                    finalist.identity
                )

    locations = []

    for location in (
        origin,
    ) + context.candidates:
        if (
            location.identity
            == origin.identity
        ):
            role = "anchor"

        elif (
            location.identity
            in first_hop_ids
        ):
            role = "intermediate"

        elif (
            query.operation
            == "count_within_radius"
        ):
            role = "facility"

        elif (
            location.identity
            in selected_ids
        ):
            role = "answer"

        elif (
            query.operation
            == "nearest_of_two_categories"
            and location.identity
            in finalist_ids
        ):
            role = "alternative"

        elif (
            location.name
            in target_names
        ):
            role = "alternative"

        else:
            role = "facility"

        step = None

        if role == "intermediate":
            step = 1

        elif (
            query.operation
            == "two_hop_nearest"
            and role == "answer"
        ):
            step = 2

        locations.append(
            point_to_dict(
                location,
                role,
                origin,
                step,
            )
        )

    facilities = [
        point
        for point in locations
        if (
            point["role"]
            == "facility"
        )
    ]

    if (
        len(facilities) > 30
        and query.operation
        != "count_within_radius"
    ):
        locations = [
            point
            for point in locations
            if (
                point["role"]
                != "facility"
            )
        ] + facilities[:30]

    answer_point = next(
        (
            point
            for point in locations
            if (
                point["role"]
                == "answer"
            )
        ),
        None,
    )

    distance_km = (
        answer_point[
            "distance_km"
        ]
        if answer_point
        else None
    )

    # -----------------------------------------------------
    # Two-hop total distance
    # -----------------------------------------------------

    if (
        query.operation
        == "two_hop_nearest"
    ):
        intermediate = next(
            (
                point
                for point in locations
                if (
                    point["role"]
                    == "intermediate"
                )
            ),
            None,
        )

        if (
            intermediate
            and answer_point
        ):
            first_distance = distance(
                origin.coordinates,
                (
                    intermediate[
                        "lat"
                    ],
                    intermediate[
                        "lng"
                    ],
                ),
            )

            second_distance = distance(
                (
                    intermediate[
                        "lat"
                    ],
                    intermediate[
                        "lng"
                    ],
                ),
                (
                    answer_point[
                        "lat"
                    ],
                    answer_point[
                        "lng"
                    ],
                ),
            )

            distance_km = round(
                first_distance
                + second_distance,
                3,
            )

    nearest_distance = min(
        (
            point[
                "distance_km"
            ]
            for point
            in locations
            if (
                point["role"]
                != "anchor"
                and point[
                    "distance_km"
                ]
                is not None
            )
        ),
        default=None,
    )

    comparison = []

    if (
        query.operation
        == "closer_of_two"
    ):
        for target in query.targets:
            point = next(
                (
                    item
                    for item in locations
                    if (
                        item["name"]
                        == target.name
                    )
                ),
                None,
            )

            if point:
                comparison.append(
                    {
                        "label":
                            target.name,

                        "value":
                            point[
                                "distance_km"
                            ],

                        "score":
                            None,
                    }
                )

    elif (
        query.operation
        == "nearest_of_two_categories"
    ):
        for category in query.categories:
            candidate = nearest_location(
                context,
                origin,
                category,
            )

            if candidate:
                candidate_distance = round(
                    distance(
                        origin.coordinates,
                        candidate.coordinates,
                    ),
                    3,
                )

                comparison.append(
                    {
                        "label":
                            category,

                        "value":
                            (
                                f"{candidate.name}"
                                f" — "
                                f"{candidate_distance} كم"
                            ),

                        "score":
                            None,
                    }
                )

    constraints = []

    if (
        query.operation
        == "within_radius_yes_no"
    ):
        constraints.append(
            {
                "label":
                    "يوجد عنصر مطابق داخل النطاق",

                "passed":
                    bool(
                        execution
                        .answer
                        .value
                    ),
            }
        )

    metrics = {
        "distance_km":
            distance_km,

        "radius_km":
            query.radius_km,

        "nearest_distance_km":
            nearest_distance,

        "count":
            (
                int(
                    execution
                    .answer
                    .value
                )
                if (
                    execution
                    .answer
                    .kind
                    == "integer"
                )
                else None
            ),

        "direction":
            (
                str(
                    execution
                    .answer
                    .value
                )
                if (
                    execution
                    .answer
                    .kind
                    == "direction"
                )
                else None
            ),

        "suitability":
            None,
    }

    visualization = {
        "nearest_category":
            "nearest",

        "cardinal_direction":
            "direction",

        "within_radius_yes_no":
            "radius_yes_no",

        "closer_of_two":
            "comparison",

        "count_within_radius":
            "count",

        "nearest_of_two_categories":
            "category_comparison",

        "two_hop_nearest":
            "two_hop",

        "spatial_multi_constraint":
            "multi_constraint",
    }[
        query.operation
    ]

    sources = [
        {
            "provider":
                "OpenStreetMap / geographic services",

            "attribution":
                "© OpenStreetMap contributors",

            "url":
                "https://www.openstreetmap.org/copyright",

            "retrieved_at":
                datetime.now(
                    timezone.utc
                ).isoformat(),

            "data_timestamp":
                timestamp,
        }
    ]

    if (
        origin.identity
        .startswith(
            "google:"
        )
    ):
        sources.append(
            {
                "provider":
                    "Google Maps",

                "attribution":
                    "Google Maps geocoding fallback",

                "url":
                    "https://maps.google.com",

                "retrieved_at":
                    datetime.now(
                        timezone.utc
                    ).isoformat(),

                "data_timestamp":
                    None,
            }
        )

    return {
        "status":
            "answered",

        "task_type":
            query.operation,

        "reasoning_depth":
            (
                "deep"
                if route == "LONG"
                else
                "concise"
            ),

        "data_mode":
            "osm_assisted",

        "answer": {
            "value":
                execution
                .answer
                .value,

            "text":
                execution
                .answer
                .text,
        },

        "reasoning":
            (
                "تم فهم السؤال وتحويله إلى Structured Query، "
                "ثم تنفيذ الحسابات عبر Geo Engine."
            ),

        "visualization":
            visualization,

        "anchor":
            point_to_dict(
                origin,
                "anchor",
                origin,
            ),

        "locations":
            locations,

        "metrics":
            metrics,

        "evidence":
            common_evidence,

        "comparison":
            comparison,

        "constraints":
            constraints,

        "sources":
            sources,

        "limitations": [
            (
                "تعتمد النتيجة على اكتمال "
                "وتحديث البيانات الجغرافية المتاحة."
            )
        ],

        "route":
            route,

        "router_selected_route":
            router_selected_route,

        "fallback_used":
            fallback_used,

        "router_long_probability":
            round(
                long_probability,
                6,
            ),

        "structured_query":
            asdict(
                query
            ),
    }


# =========================================================
# 13. Main Adaptive Runtime
# =========================================================

def analyze(
    question: str,
) -> dict:
    question = " ".join(
        question.strip().split()
    )

    if not question:
        raise AdaptiveInputError(
            "السؤال فارغ."
        )

    # -----------------------------------------------------
    # STEP 1
    # BiGRU understands natural-language intent.
    # -----------------------------------------------------

    (
        predicted_operation,
        short_confidence,
        probabilities,
    ) = classify_short(
        question
    )

    # -----------------------------------------------------
    # STEP 2
    # Natural-language entity grounding.
    # -----------------------------------------------------

    seed = build_seed_context(
        question,
        predicted_operation,
    )

    # -----------------------------------------------------
    # STEP 3
    # Candidate SHORT query.
    # -----------------------------------------------------

    short_prediction = (
        build_short_prediction(
            question,
            seed,
            predicted_operation,
            short_confidence,
        )
    )

    # -----------------------------------------------------
    # STEP 4
    # Router V2 features.
    # -----------------------------------------------------

    router_features = (
        build_router_features(
            question,
            seed,
            short_prediction,
            probabilities,
        )
    )

    # -----------------------------------------------------
    # STEP 5
    # Router V2 selects SHORT / LONG.
    # -----------------------------------------------------

    router = get_router()

    long_probability = float(
        router.long_probabilities(
            [
                router_features
            ]
        )[0]
    )

    router_selected_route = (
        "LONG"
        if (
            long_probability
            >= router.threshold
        )
        else
        "SHORT"
    )

    route = (
        router_selected_route
    )

    fallback_used = False

    # -----------------------------------------------------
    # STEP 6
    # Execute selected language strategy.
    # -----------------------------------------------------

    if route == "SHORT":
        query = (
            short_prediction.query
        )

        if query is None:
            raise AdaptiveInputError(
                short_prediction.reason
                or
                "تعذر تكوين Structured Query صالح."
            )

    else:
        try:
            query = run_long(
                question,
                seed,
            )

        except AdaptiveLongUnavailableError:
            # LONG infrastructure/schema failure:
            #
            # fall back only when SHORT independently
            # produced a fully valid query.
            if (
                short_prediction.query
                is None
            ):
                raise

            query = (
                short_prediction.query
            )

            route = "SHORT"
            fallback_used = True

    # -----------------------------------------------------
    # STEP 7
    # Geographic data.
    #
    # POIs:
    # Local Pyrosm/PBF first via OSMClient.
    #
    # Named-place geocoding:
    # Nominatim first, Google fallback.
    # -----------------------------------------------------

    (
        context,
        timestamp,
    ) = enrich_context(
        query,
        seed,
    )

    # -----------------------------------------------------
    # STEP 8
    # Deterministic Geo Engine.
    # -----------------------------------------------------

    execution = execute(
        query,
        context,
    )

    # -----------------------------------------------------
    # STEP 9
    # Stable Django/frontend response.
    # -----------------------------------------------------

    return build_response(
        question,
        route,
        router_selected_route,
        fallback_used,
        long_probability,
        query,
        execution,
        context,
        timestamp,
    )