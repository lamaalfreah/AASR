let map;
let markersLayer;
let shapeLayer;
let processingStateTimer;
let mascotTransitionTimer;

const MASCOT_STATES = {
    welcome: {
        file: "01_welcome.png",
        caption: "مرحبًا، أنا جاهز لسؤالك المكاني.",
        alt: "شخصية ASAR ترحب بك",
        status: "النظام جاهز للتحليل",
        statusClass: ""
    },
    waiting: {
        file: "02_waiting_for_question.png",
        caption: "بانتظار سؤالك… اكتب المكان والقرار الذي تريد تحليله.",
        alt: "شخصية ASAR تنتظر السؤال",
        status: "بانتظار السؤال",
        statusClass: ""
    },
    searching: {
        file: "03_searching_answer_laptop.png",
        caption: "أحلل السؤال وأحدد عمق الاستدلال المناسب.",
        alt: "شخصية ASAR تحلل السؤال باستخدام الحاسوب",
        status: "جاري تحليل السؤال",
        statusClass: "is-loading"
    },
    found: {
        file: "04_answer_found_idea.png",
        caption: "وجدت إجابة مكانية قابلة للعرض.",
        alt: "شخصية ASAR توصلت إلى الإجابة",
        status: "اكتمل التحليل",
        statusClass: "is-success"
    },
    map: {
        file: "05_spatial_guidance_map.png",
        caption: "أعرض العلاقات والنتائج على الخريطة.",
        alt: "شخصية ASAR تقدم إرشادًا مكانيًا",
        status: "عرض مكاني",
        statusClass: ""
    },
    "geo-search": {
        file: "06_searching_magnifier.png",
        caption: "أراجع المعطيات المكانية المتاحة قبل صياغة الإجابة.",
        alt: "شخصية ASAR تبحث في المعطيات المكانية",
        status: "جاري التحقق من المعطيات",
        statusClass: "is-loading"
    },
    success: {
        file: "07_success.png",
        caption: "اكتمل التحليل والتحقق بنجاح.",
        alt: "شخصية ASAR تحتفل بنجاح التحليل",
        status: "النتيجة جاهزة",
        statusClass: "is-success"
    },
    error: {
        file: "08_error.png",
        caption: "لم أتمكن من إكمال التحليل. راجع السؤال أو حاول مجددًا.",
        alt: "شخصية ASAR تشير إلى تعذر التحليل",
        status: "تعذر إكمال التحليل",
        statusClass: "is-error"
    }
};

document.addEventListener("DOMContentLoaded", () => {
    initMap();
    preloadMascots();

    document.querySelectorAll(".tag").forEach((tag) => {
        tag.addEventListener("click", () => {
            document.querySelectorAll(".tag").forEach((item) => item.classList.remove("active"));
            tag.classList.add("active");
        });
    });

    const form = document.getElementById("analysis-form");
    const questionInput = document.getElementById("question-input");
    form.addEventListener("submit", handleAnalyze);
    questionInput.addEventListener("input", () => {
        if (document.getElementById("analyze-button").disabled) return;
        if (!questionInput.value.trim()) setMascotState("waiting");
        else if (document.getElementById("mascot-stage").dataset.state === "waiting") {
            setMascotState("welcome");
        }
    });
});

function preloadMascots() {
    const base = document.body.dataset.mascotBase || "";
    Object.values(MASCOT_STATES).forEach((state) => {
        const image = new Image();
        image.src = `${base}${state.file}`;
    });
}

function setMascotState(name) {
    const state = MASCOT_STATES[name];
    const stage = document.getElementById("mascot-stage");
    const image = document.getElementById("state-mascot");
    const caption = document.getElementById("mascot-caption");
    const statusElement = document.getElementById("system-status");
    if (!state || !stage || !image || !caption || !statusElement) return;

    clearTimeout(mascotTransitionTimer);
    stage.classList.add("is-changing");
    stage.dataset.state = name;
    statusElement.classList.remove("is-loading", "is-success", "is-error");
    if (state.statusClass) statusElement.classList.add(state.statusClass);
    document.getElementById("system-status-text").textContent = state.status;

    mascotTransitionTimer = setTimeout(() => {
        const base = document.body.dataset.mascotBase || "";
        image.src = `${base}${state.file}`;
        image.alt = state.alt;
        caption.textContent = state.caption;
        stage.classList.remove("is-changing");
    }, 110);
}

function initMap() {
    if(!window.L){document.getElementById("map").textContent="الخريطة غير متاحة؛ تبقى الإجابة النصية متاحة.";return;}
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
        setMascotState("waiting");
        showToast("اكتبي سؤالًا مكانيًا أولاً.");
        return;
    }

    button.disabled = true;
    buttonText.textContent = "جاري التحليل...";
    document.getElementById("analysis-form").setAttribute("aria-busy", "true");
    setMascotState("searching");
    clearTimeout(processingStateTimer);
    processingStateTimer = setTimeout(() => {
        if (button.disabled) setMascotState("geo-search");
    }, 1600);

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
        if (!response.ok) {
            const message = typeof data.error === "object" ? data.error?.message : data.error;
            throw new Error(message || "تعذر تنفيذ التحليل.");
        }

        renderSpatialResponse(data);
        setMascotState(
            data.status === "insufficient_information"
                ? "error"
                : data.status === "needs_clarification"
                    ? "waiting"
                    : data.verification?.passed === true
                        ? "success"
                        : "found"
        );
        showToast("تم تحديث العرض حسب نوع السؤال.");
    } catch (error) {
        renderSpatialResponse({status:'insufficient_information',answer:{value:null,text:error.message},locations:[],metrics:{},limitations:[]});
        setMascotState("error");
        showToast(error.message);
    } finally {
        clearTimeout(processingStateTimer);
        button.disabled = false;
        buttonText.textContent = "تحليل واستدلال";
        document.getElementById("analysis-form").setAttribute("aria-busy", "false");
    }
}

function renderSpatialResponse(data) {
    document.getElementById('runtime-route').textContent = data.route ? `${data.route} • ${Math.round(data.latency?.total_ms || 0)} ms` : '';
    document.getElementById('runtime-trace').textContent = JSON.stringify({question:data.interpreted_question,query:data.structured_query,trace:data.trace,latency:data.latency},null,2);
    const meta = taskMeta(data.task_type);
    const insufficient =
    data.status === "insufficient_information" ||
    data.status === "needs_clarification";
    const depthBadge = document.getElementById("reasoning-depth");

    document.getElementById("result-task-label").textContent = meta.label;
    document.getElementById("answer-symbol").textContent = insufficient ? "؟" : meta.symbol;
    document.getElementById("answer-title").textContent =
        data.answer?.text || String(data.answer?.value ?? "—");
    document.getElementById("answer-subtitle").textContent =
        insufficient ? "المعلومات المكانية غير مكتملة" : `نوع المهمة: ${meta.label}`;
    document.getElementById("reasoning-text").textContent =
        data.reasoning || "لا يوجد شرح إضافي.";

    depthBadge.classList.remove("hidden", "concise", "deep");
    depthBadge.classList.add(data.reasoning_depth === "deep" ? "deep" : "concise");
    depthBadge.textContent = data.reasoning_depth === "deep" ? "استدلال متعمق" : "استدلال موجز";

    renderComparison(data.comparison || []);
    renderConstraints(data.constraints || []);
    renderEvidence(data.evidence || []);
    renderSources(data.sources || []);
    renderLimitations(data.limitations || []);
    renderMetrics(data, meta);
    renderMap(data);
}

function renderSources(items) {
    const card = document.getElementById("sources-card");
    const container = document.getElementById("sources-list");
    if (!items.length) {
        card.classList.add("hidden");
        container.innerHTML = "";
        return;
    }
    card.classList.remove("hidden");
    container.innerHTML = items.map((item) => {
        const href = safeHttpUrl(item.url);
        const provider = escapeHtml(item.provider || "مصدر جغرافي");
        const title = href ? `<a href="${escapeHtml(href)}" target="_blank" rel="noopener noreferrer">${provider}</a>` : provider;
        const timestamp = item.data_timestamp || item.retrieved_at || "";
        return `<div class="source-item"><strong>${title}</strong><span>${escapeHtml(item.attribution || "")}</span><small>${escapeHtml(timestamp)}</small></div>`;
    }).join("");
}

function renderLimitations(items) {
    const card = document.getElementById("limitations-card");
    const list = document.getElementById("limitations-list");
    if (!items.length) {
        card.classList.add("hidden");
        list.innerHTML = "";
        return;
    }
    card.classList.remove("hidden");
    list.innerHTML = items.map((item) => `<li>${escapeHtml(item)}</li>`).join("");
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
    container.innerHTML = items.map((item, index) => {
        const score = normalizedScore(item.score);
        const scorePercent = score === null ? null : Math.round(score * 100);
        return `
            <div class="rank-item">
                <div class="rank-number">${index + 1}</div>
                <div class="rank-main">
                    <div class="rank-title">
                        <span>${escapeHtml(item.label)}</span>
                        <span>${escapeHtml(item.value)}</span>
                    </div>
                    <div class="rank-bar">
                        <div class="rank-fill" style="width:${scorePercent ?? 0}%"></div>
                    </div>
                </div>
                <div class="rank-score">${scorePercent === null ? "—" : `${scorePercent}%`}</div>
            </div>
        `;
    }).join("");
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
    list.innerHTML = items.map((item) => {
        const state = item.passed === true ? "" : item.passed === false ? " — غير متحقق" : " — غير محدد";
        return `<li>${escapeHtml(item.label)}${state}</li>`;
    }).join("");
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
    if (m.distance_km != null)
        setMetric(2, "المسافة", `${m.distance_km} كم`, "أقرب مسافة محسوبة");
    else if (m.radius_km != null)
        setMetric(2, "نصف القطر", `${m.radius_km} كم`, "النطاق المستخدم");
    else if (m.direction != null)
        setMetric(2, "الاتجاه", m.direction, "الاتجاه النسبي");
    else if (m.count != null)
        setMetric(2, "العدد", m.count, "عناصر داخل النطاق");
    else if (m.suitability != null)
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
    if(!map)return;
    markersLayer.clearLayers();
    shapeLayer.clearLayers();

    const locations = data.locations || [];
    const anchor = data.anchor;
    const osmMode = data.data_mode === "osm_assisted";
    document.getElementById("legend-osm").classList.toggle("hidden", !osmMode);
    document.getElementById("legend-best").lastChild.textContent = " النتيجة";
    document.getElementById("legend-alt").lastChild.textContent = " بديل";

    locations.forEach((loc) => {
        if (!hasCoordinates(loc)) return;

        const cls =
            loc.role === "answer" ? "marker-best" :
            loc.source === "openstreetmap" ? "marker-osm" :
            loc.source === "derived" && loc.role !== "answer" ? "marker-derived" :
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

        const osmHref = osmObjectUrl(loc);
        const sourceLine = loc.source === "openstreetmap"
            ? `<br><small>${osmHref ? `<a href="${osmHref}" target="_blank" rel="noopener noreferrer">OpenStreetMap</a>` : "OpenStreetMap"}</small>`
            : loc.source === "derived" ? "<br><small>حساب حتمي — نقطة تغطية تحليلية</small>" : "";
        L.marker([loc.lat, loc.lng], { icon })
            .bindPopup(`<div dir="rtl"><strong>${escapeHtml(loc.name || "موقع")}</strong>${
                loc.distance_km != null ? `<br>المسافة: ${escapeHtml(loc.distance_km)} كم` : ""
            }${sourceLine}</div>`)
            .addTo(markersLayer);
    });

    if (hasCoordinates(anchor) && data.metrics?.radius_km != null) {
        L.circle([anchor.lat, anchor.lng], {
            radius: Number(data.metrics.radius_km) * 1000,
            weight: 2,
            fillOpacity: 0.06
        }).addTo(shapeLayer);
    }

    const mappedLocations = locations.filter(hasCoordinates);

    if (["nearest", "direction"].includes(data.visualization) && mappedLocations.length >= 2) {
        drawPath(mappedLocations.slice(0, 2));
    }

    if (data.visualization === "two_hop" && mappedLocations.length >= 3) {
        drawPath(mappedLocations.slice(0, 3));
    }

    if (mappedLocations.length) {
        map.fitBounds(L.latLngBounds(mappedLocations.map((x) => [x.lat, x.lng])).pad(0.22));
    } else {
        map.setView([24.7136, 46.6753], 12);
    }
}

function osmObjectUrl(point) {
    if (point?.source !== "openstreetmap") return null;
    if (!["node", "way", "relation"].includes(point.osm_type)) return null;
    if (!/^\d+$/.test(String(point.osm_id || ""))) return null;
    return `https://www.openstreetmap.org/${point.osm_type}/${point.osm_id}`;
}

function safeHttpUrl(value) {
    try {
        const url = new URL(String(value || ""));
        return ["http:", "https:"].includes(url.protocol) ? url.href : null;
    } catch (_error) {
        return null;
    }
}

function hasCoordinates(point) {
    return point
        && point.lat !== null
        && point.lng !== null
        && Number.isFinite(Number(point.lat))
        && Number.isFinite(Number(point.lng));
}

function normalizedScore(value) {
    if (value === null || value === undefined || !Number.isFinite(Number(value))) return null;
    return Math.min(1, Math.max(0, Number(value)));
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
        spatial_multi_constraint: { label: "قيود مكانية متعددة", symbol: "✓" },
        general_spatial: { label: "استدلال مكاني عام", symbol: "✦" }
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
