let map;
let markersLayer;
let shapeLayer;

document.addEventListener("DOMContentLoaded", () => {
    initMap();

    document.querySelectorAll(".tag").forEach((tag) => {
        tag.addEventListener("click", () => {
            document.querySelectorAll(".tag").forEach((item) => item.classList.remove("active"));
            tag.classList.add("active");
        });
    });

    document.getElementById("analysis-form").addEventListener("submit", handleAnalyze);
});

function initMap() {
    map = L.map("map", { zoomControl: true, scrollWheelZoom: true })
        .setView([24.7136, 46.6753], 12);

    L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
        maxZoom: 19,
        attribution: "&copy; OpenStreetMap"
    }).addTo(map);

    markersLayer = L.layerGroup().addTo(map);
    shapeLayer = L.layerGroup().addTo(map);
}

async function handleAnalyze(event) {
    event.preventDefault();

    const button = document.getElementById("analyze-button");
    const buttonText = button.querySelector(".button-text");
    const question = document.getElementById("question-input").value.trim();

    if (!question) {
        showToast("اكتبي سؤالًا مكانيًا أولاً.");
        return;
    }

    button.disabled = true;
    buttonText.textContent = "جاري التحليل...";

    try {
        const response = await fetch("/api/analyze/", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                "X-CSRFToken": getCookie("csrftoken")
            },
            body: JSON.stringify({ question })
        });

        const data = await response.json();
        if (!response.ok) throw new Error(data.error || "تعذر تنفيذ التحليل.");

        renderSpatialResponse(data);
        showToast("تم تحديث العرض حسب نوع السؤال.");
    } catch (error) {
        showToast(error.message);
    } finally {
        button.disabled = false;
        buttonText.textContent = "تحليل واستدلال";
    }
}

function renderSpatialResponse(data) {
    const meta = taskMeta(data.task_type);

    document.getElementById("result-task-label").textContent = meta.label;
    document.getElementById("answer-symbol").textContent = meta.symbol;
    document.getElementById("answer-title").textContent =
        data.answer?.text || String(data.answer?.value ?? "—");
    document.getElementById("answer-subtitle").textContent =
        `نوع المهمة: ${meta.label}`;
    document.getElementById("reasoning-text").textContent =
        data.reasoning || "لا يوجد شرح إضافي.";

    renderComparison(data.comparison || []);
    renderConstraints(data.constraints || []);
    renderEvidence(data.evidence || []);
    renderMetrics(data, meta);
    renderMap(data);
}

function renderComparison(items) {
    const card = document.getElementById("comparison-card");
    const container = document.getElementById("comparison-list");

    if (!items.length) {
        card.classList.add("hidden");
        container.innerHTML = "";
        return;
    }

    card.classList.remove("hidden");
    container.innerHTML = items.map((item, index) => `
        <div class="rank-item">
            <div class="rank-number">${index + 1}</div>
            <div class="rank-main">
                <div class="rank-title">
                    <span>${escapeHtml(item.label)}</span>
                    <span>${escapeHtml(item.value)}</span>
                </div>
                <div class="rank-bar">
                    <div class="rank-fill" style="width:${Math.round((item.score ?? 0.5) * 100)}%"></div>
                </div>
            </div>
            <div class="rank-score">${Math.round((item.score ?? 0.5) * 100)}%</div>
        </div>
    `).join("");
}

function renderConstraints(items) {
    const card = document.getElementById("constraint-card");
    const list = document.getElementById("constraint-list");

    if (!items.length) {
        card.classList.add("hidden");
        list.innerHTML = "";
        return;
    }

    card.classList.remove("hidden");
    list.innerHTML = items.map((item) =>
        `<li>${escapeHtml(item.label)}${item.passed ? "" : " — غير متحقق"}</li>`
    ).join("");
}

function renderEvidence(items) {
    const card = document.getElementById("evidence-card");
    const container = document.getElementById("evidence-list");

    if (!items.length) {
        card.classList.add("hidden");
        container.innerHTML = "";
        return;
    }

    card.classList.remove("hidden");
    container.innerHTML = items.map((item) => `
        <div class="evidence-item">
            <div class="evidence-copy">
                <div class="evidence-icon">◎</div>
                <div class="evidence-label">${escapeHtml(item.label)}</div>
            </div>
            <strong>${escapeHtml(item.value)}</strong>
        </div>
    `).join("");
}

function renderMetrics(data, meta) {
    setMetric(1, "نوع المهمة", meta.label, "Task Type");

    const m = data.metrics || {};
    if (m.distance_km !== undefined)
        setMetric(2, "المسافة", `${m.distance_km} كم`, "أقرب مسافة محسوبة");
    else if (m.radius_km !== undefined)
        setMetric(2, "نصف القطر", `${m.radius_km} كم`, "النطاق المستخدم");
    else if (m.direction !== undefined)
        setMetric(2, "الاتجاه", m.direction, "الاتجاه النسبي");
    else if (m.count !== undefined)
        setMetric(2, "العدد", m.count, "عناصر داخل النطاق");
    else if (m.suitability !== undefined)
        setMetric(2, "درجة الملاءمة", Number(m.suitability).toFixed(2), "تقييم متعدد القيود");
    else
        setMetric(2, "المؤشر المكاني", "—", "يعتمد على نوع السؤال");

    setMetric(3, "الإجابة", String(data.answer?.value ?? "—"), "FINAL output");
}

function setMetric(i, label, value, note) {
    document.getElementById(`metric-${i}-label`).textContent = label;
    document.getElementById(`metric-${i}-value`).textContent = value;
    document.getElementById(`metric-${i}-note`).textContent = note;
}

function renderMap(data) {
    markersLayer.clearLayers();
    shapeLayer.clearLayers();

    const locations = data.locations || [];
    const anchor = data.anchor;

    locations.forEach((loc) => {
        if (loc.lat === undefined || loc.lng === undefined) return;

        const cls =
            loc.role === "answer" ? "marker-best" :
            loc.role === "intermediate" ? "marker-alt" :
            loc.role === "anchor" ? "marker-muted" : "marker-alt";

        const label =
            loc.role === "anchor" ? "●" :
            loc.step ? String(loc.step) :
            (loc.name || "•").charAt(0);

        const icon = L.divIcon({
            className: "",
            html: `<div class="marker-badge ${cls}"><span>${escapeHtml(label)}</span></div>`,
            iconSize: [44, 44],
            iconAnchor: [22, 22]
        });

        L.marker([loc.lat, loc.lng], { icon })
            .bindPopup(`<div dir="rtl"><strong>${escapeHtml(loc.name || "موقع")}</strong>${
                loc.distance_km !== undefined ? `<br>المسافة: ${loc.distance_km} كم` : ""
            }</div>`)
            .addTo(markersLayer);
    });

    if (anchor && data.metrics?.radius_km) {
        L.circle([anchor.lat, anchor.lng], {
            radius: Number(data.metrics.radius_km) * 1000,
            weight: 2,
            fillOpacity: 0.06
        }).addTo(shapeLayer);
    }

    if (["nearest", "direction"].includes(data.visualization) && locations.length >= 2) {
        drawPath(locations.slice(0, 2));
    }

    if (data.visualization === "two_hop" && locations.length >= 3) {
        drawPath(locations.slice(0, 3));
    }

    const valid = locations.filter((x) => x.lat !== undefined && x.lng !== undefined);
    if (valid.length) {
        map.fitBounds(L.latLngBounds(valid.map((x) => [x.lat, x.lng])).pad(0.22));
    }
}

function drawPath(points) {
    const latlngs = points.map((p) => [p.lat, p.lng]);
    L.polyline(latlngs, { weight: 3, opacity: 0.75, dashArray: "7,7" }).addTo(shapeLayer);
}

function taskMeta(taskType) {
    const items = {
        nearest_category: { label: "أقرب عنصر من فئة", symbol: "⌖" },
        cardinal_direction: { label: "اتجاه مكاني", symbol: "↗" },
        within_radius_yes_no: { label: "داخل نطاق", symbol: "◎" },
        closer_of_two: { label: "مقارنة مسافتين", symbol: "⇄" },
        count_within_radius: { label: "عد داخل نطاق", symbol: "#" },
        nearest_of_two_categories: { label: "مقارنة بين فئتين", symbol: "≋" },
        two_hop_nearest: { label: "استدلال على خطوتين", symbol: "②" },
        spatial_multi_constraint: { label: "قيود مكانية متعددة", symbol: "✓" }
    };
    return items[taskType] || { label: "استدلال مكاني", symbol: "✦" };
}

function getCookie(name) {
    const cookies = document.cookie ? document.cookie.split(";") : [];
    for (const cookie of cookies) {
        const trimmed = cookie.trim();
        if (trimmed.startsWith(`${name}=`))
            return decodeURIComponent(trimmed.slice(name.length + 1));
    }
    return "";
}

function showToast(message) {
    const toast = document.getElementById("toast");
    toast.textContent = message;
    toast.classList.add("show");
    clearTimeout(showToast.timeout);
    showToast.timeout = setTimeout(() => toast.classList.remove("show"), 2800);
}

function escapeHtml(value) {
    return String(value ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
}