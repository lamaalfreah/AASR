"""
Verifier agent.

Cross-checks the spatial result independently from the
language model.

Direction verification follows the same four cardinal
sectors used by the deterministic Geo Engine:

شمال / شرق / جنوب / غرب

Supported geographic provenance:
- Nominatim
- OpenStreetMap
- Google Maps
- Derived geographic points
"""

import math


EARTH_RADIUS_KM = 6371.0088

DISTANCE_TOLERANCE_KM = 0.15

DIRECTION_LABELS = [
    "شمال",
    "شرق",
    "جنوب",
    "غرب",
]


# =========================================================
# Geometry helpers
# =========================================================

def haversine_km(
    lat1,
    lon1,
    lat2,
    lon2,
):
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)

    d_phi = math.radians(
        lat2 - lat1
    )

    d_lambda = math.radians(
        lon2 - lon1
    )

    a = (
        math.sin(
            d_phi / 2
        ) ** 2
        + math.cos(phi1)
        * math.cos(phi2)
        * math.sin(
            d_lambda / 2
        ) ** 2
    )

    return (
        2
        * EARTH_RADIUS_KM
        * math.asin(
            math.sqrt(a)
        )
    )


def bearing_degrees(
    lat1,
    lon1,
    lat2,
    lon2,
):
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)

    d_lambda = math.radians(
        lon2 - lon1
    )

    x = (
        math.sin(d_lambda)
        * math.cos(phi2)
    )

    y = (
        math.cos(phi1)
        * math.sin(phi2)
        - math.sin(phi1)
        * math.cos(phi2)
        * math.cos(d_lambda)
    )

    return (
        math.degrees(
            math.atan2(
                x,
                y,
            )
        )
        + 360
    ) % 360


def bearing_cardinal(
    lat1,
    lon1,
    lat2,
    lon2,
):
    """
    Match the four-sector direction system used by:

    spatial.geometry.direction(..., sectors=4)

    شمال:
        315° -> 45°

    شرق:
        45° -> 135°

    جنوب:
        135° -> 225°

    غرب:
        225° -> 315°
    """

    bearing = bearing_degrees(
        lat1,
        lon1,
        lat2,
        lon2,
    )

    index = int(
        (
            bearing + 45
        )
        // 90
    ) % 4

    return DIRECTION_LABELS[index]


# =========================================================
# Point helpers
# =========================================================

def _find(
    locations,
    role,
):
    return next(
        (
            point
            for point in locations
            if point.get("role") == role
        ),
        None,
    )


def _valid_point(
    point,
):
    if not isinstance(
        point,
        dict,
    ):
        return False

    lat = point.get("lat")
    lng = point.get("lng")

    return (
        isinstance(
            lat,
            (int, float),
        )
        and not isinstance(
            lat,
            bool,
        )
        and math.isfinite(lat)
        and -90 <= lat <= 90

        and isinstance(
            lng,
            (int, float),
        )
        and not isinstance(
            lng,
            bool,
        )
        and math.isfinite(lng)
        and -180 <= lng <= 180
    )


# =========================================================
# Main verifier
# =========================================================

def verify_result(
    result: dict,
) -> dict:
    """
    Add a verification block to the runtime result.

    Verification failures never crash the API response.
    """

    checks = []

    # -----------------------------------------------------
    # Only answered results require verification
    # -----------------------------------------------------

    if result.get("status") != "answered":
        result["verification"] = {
            "passed": None,
            "checks": checks,
        }

        return result

    anchor = result.get("anchor")

    locations = result.get(
        "locations",
        [],
    )

    answer_point = _find(
        locations,
        "answer",
    )

    task_type = result.get(
        "task_type"
    )

    try:

        # =================================================
        # 1. Standard distance + direction checks
        # =================================================

        if (
            task_type != "two_hop_nearest"
            and _valid_point(anchor)
            and _valid_point(answer_point)
        ):
            real_km = haversine_km(
                anchor["lat"],
                anchor["lng"],
                answer_point["lat"],
                answer_point["lng"],
            )

            claimed_km = (
                result
                .get(
                    "metrics",
                    {},
                )
                .get(
                    "distance_km"
                )
            )

            if claimed_km is None:
                claimed_km = (
                    answer_point
                    .get(
                        "distance_km"
                    )
                )

            if claimed_km is not None:
                passed = (
                    abs(
                        real_km
                        - claimed_km
                    )
                    <= DISTANCE_TOLERANCE_KM
                )

                checks.append(
                    {
                        "check": "distance_km",
                        "claimed": claimed_km,
                        "computed": round(
                            real_km,
                            3,
                        ),
                        "passed": passed,
                    }
                )

            # ---------------------------------------------
            # Direction verification
            # ---------------------------------------------

            claimed_direction = (
                result
                .get(
                    "metrics",
                    {},
                )
                .get(
                    "direction"
                )
            )

            if claimed_direction:
                real_direction = (
                    bearing_cardinal(
                        anchor["lat"],
                        anchor["lng"],
                        answer_point["lat"],
                        answer_point["lng"],
                    )
                )

                checks.append(
                    {
                        "check": "direction",
                        "claimed": claimed_direction,
                        "computed": real_direction,
                        "passed": (
                            claimed_direction
                            == real_direction
                        ),
                    }
                )

        # =================================================
        # 2. Within-radius yes/no
        # =================================================

        if (
            task_type
            == "within_radius_yes_no"
            and _valid_point(anchor)
        ):
            radius = (
                result
                .get(
                    "metrics",
                    {},
                )
                .get(
                    "radius_km"
                )
            )

            claimed = (
                result
                .get(
                    "answer",
                    {},
                )
                .get(
                    "value"
                )
            )

            if radius is not None:

                qualifying_points = [
                    point
                    for point in locations
                    if (
                        point.get("role")
                        in {
                            "answer",
                            "facility",
                        }
                        and _valid_point(point)
                    )
                ]

                computed = any(
                    haversine_km(
                        anchor["lat"],
                        anchor["lng"],
                        point["lat"],
                        point["lng"],
                    )
                    <= radius

                    for point
                    in qualifying_points
                )

                checks.append(
                    {
                        "check":
                            "within_radius",

                        "claimed":
                            claimed,

                        "computed":
                            computed,

                        "passed":
                            claimed is computed,
                    }
                )

        # =================================================
        # 3. Closer-of-two / two-category comparison
        # =================================================

        if (
            task_type
            in {
                "closer_of_two",
                "nearest_of_two_categories",
            }
            and _valid_point(anchor)
        ):
            compared = [
                point
                for point in locations
                if (
                    point.get("role")
                    in {
                        "answer",
                        "alternative",
                    }
                    and _valid_point(point)
                )
            ]

            if (
                len(compared) == 2
                and _valid_point(
                    answer_point
                )
            ):
                computed_winner = min(
                    compared,

                    key=lambda point:
                        haversine_km(
                            anchor["lat"],
                            anchor["lng"],
                            point["lat"],
                            point["lng"],
                        ),
                )

                checks.append(
                    {
                        "check":
                            "comparison_winner",

                        "claimed":
                            answer_point.get(
                                "name"
                            ),

                        "computed":
                            computed_winner.get(
                                "name"
                            ),

                        "passed":
                            (
                                answer_point.get(
                                    "source_ref"
                                )
                                ==
                                computed_winner.get(
                                    "source_ref"
                                )
                            ),
                    }
                )

        # =================================================
        # 4. Two-hop nearest
        # =================================================

        if (
            task_type
            == "two_hop_nearest"
            and _valid_point(anchor)
        ):
            intermediate = _find(
                locations,
                "intermediate",
            )

            if (
                _valid_point(
                    intermediate
                )
                and _valid_point(
                    answer_point
                )
            ):
                hop_one = haversine_km(
                    anchor["lat"],
                    anchor["lng"],
                    intermediate["lat"],
                    intermediate["lng"],
                )

                hop_two = haversine_km(
                    intermediate["lat"],
                    intermediate["lng"],
                    answer_point["lat"],
                    answer_point["lng"],
                )

                claimed_total = (
                    result
                    .get(
                        "metrics",
                        {},
                    )
                    .get(
                        "distance_km"
                    )
                )

                checks.append(
                    {
                        "check":
                            "two_hop_distance",

                        "claimed":
                            claimed_total,

                        "computed":
                            round(
                                hop_one
                                + hop_two,
                                3,
                            ),

                        "passed":
                            (
                                claimed_total
                                is not None

                                and abs(
                                    claimed_total
                                    - (
                                        hop_one
                                        + hop_two
                                    )
                                )
                                <= (
                                    DISTANCE_TOLERANCE_KM
                                    * 2
                                )
                            ),
                    }
                )

        # =================================================
        # 5. Count within radius
        # =================================================

        if (
            task_type
            == "count_within_radius"
        ):
            radius = (
                result
                .get(
                    "metrics",
                    {},
                )
                .get(
                    "radius_km"
                )
            )

            claimed_count = (
                result
                .get(
                    "metrics",
                    {},
                )
                .get(
                    "count"
                )
            )

            if (
                _valid_point(anchor)
                and radius is not None
                and claimed_count is not None
            ):
                real_count = sum(
                    1
                    for point in locations
                    if (
                        point.get("role")
                        == "facility"

                        and _valid_point(point)

                        and haversine_km(
                            anchor["lat"],
                            anchor["lng"],
                            point["lat"],
                            point["lng"],
                        )
                        <= radius
                    )
                )

                checks.append(
                    {
                        "check":
                            "count_within_radius",

                        "claimed":
                            claimed_count,

                        "computed":
                            real_count,

                        "passed":
                            claimed_count
                            == real_count,
                    }
                )

        # =================================================
        # 6. Geographic provenance
        # =================================================

        if (
            result.get("data_mode")
            == "osm_assisted"
        ):
            grounded_points = [
                anchor,
                *locations,
            ]

            provenance_ok = all(
                (
                    not isinstance(
                        point,
                        dict,
                    )
                )
                or
                (
                    point.get("source")
                    in {
                        "nominatim",
                        "openstreetmap",
                        "google",
                        "derived",
                    }

                    and isinstance(
                        point.get(
                            "source_ref"
                        ),
                        str,
                    )

                    and bool(
                        point["source_ref"]
                    )

                    and _valid_point(
                        point
                    )
                )

                for point
                in grounded_points
            )

            checks.append(
                {
                    "check":
                        "geographic_provenance",

                    "passed":
                        provenance_ok,
                }
            )

        # =================================================
        # 7. Optional service coverage verification
        # =================================================

        if (
            result.get("visualization")
            == "service_coverage"
        ):
            claimed_count = (
                result
                .get(
                    "metrics",
                    {},
                )
                .get(
                    "count"
                )
            )

            facility_count = sum(
                point.get("role")
                == "facility"

                for point
                in locations

                if isinstance(
                    point,
                    dict,
                )
            )

            checks.append(
                {
                    "check":
                        "facility_count",

                    "claimed":
                        claimed_count,

                    "computed":
                        facility_count,

                    "passed":
                        claimed_count
                        == facility_count,
                }
            )

            distance_checks = []

            if _valid_point(anchor):
                for point in locations:

                    if not isinstance(
                        point,
                        dict,
                    ):
                        continue

                    claimed = point.get(
                        "distance_km"
                    )

                    if (
                        claimed is None
                        or not _valid_point(
                            point
                        )
                    ):
                        continue

                    computed = haversine_km(
                        anchor["lat"],
                        anchor["lng"],
                        point["lat"],
                        point["lng"],
                    )

                    distance_checks.append(
                        abs(
                            computed
                            - claimed
                        )
                        <= DISTANCE_TOLERANCE_KM
                    )

            checks.append(
                {
                    "check":
                        "grounded_point_distances",

                    "passed":
                        (
                            all(
                                distance_checks
                            )
                            if distance_checks
                            else True
                        ),
                }
            )

    except (
        KeyError,
        OverflowError,
        TypeError,
        ValueError,
    ) as exc:

        checks.append(
            {
                "check":
                    "error",

                "passed":
                    False,

                "detail":
                    str(exc),
            }
        )

    # =====================================================
    # Final verification result
    # =====================================================

    all_passed = (
        all(
            check["passed"]
            for check in checks
        )
        if checks
        else None
    )

    result["verification"] = {
        "passed":
            all_passed,

        "checks":
            checks,
    }

    return result