/**
 * MedicalVision AI Suite — Shared Frontend JavaScript
 *
 * Handles: theme toggle, image upload, prediction, XAI display, report download.
 */

/* ── Theme ───────────────────────────────────────────────── */
function initTheme() {
    const saved = localStorage.getItem('medvision-theme') || 'light';
    document.documentElement.setAttribute('data-theme', saved);
    const btn = document.getElementById('themeToggle');
    if (btn) {
        btn.textContent = saved === 'dark' ? '☀️' : '🌙';
        btn.addEventListener('click', () => {
            const current = document.documentElement.getAttribute('data-theme');
            const next = current === 'dark' ? 'light' : 'dark';
            document.documentElement.setAttribute('data-theme', next);
            localStorage.setItem('medvision-theme', next);
            btn.textContent = next === 'dark' ? '☀️' : '🌙';
        });
    }
}

/* ── Upload Zone (Drag & Drop) ───────────────────────────── */
function initUploadZone() {
    const zone = document.getElementById('uploadZone');
    const input = document.getElementById('fileInput');
    const preview = document.getElementById('previewContainer');

    if (!zone || !input) return;

    // Click to select
    zone.addEventListener('click', () => input.click());

    // File selected via dialog
    input.addEventListener('change', (e) => {
        if (e.target.files.length > 0) handleFile(e.target.files[0]);
    });

    // Drag events
    ['dragenter', 'dragover'].forEach(evt => {
        zone.addEventListener(evt, (e) => {
            e.preventDefault();
            zone.classList.add('drag-over');
        });
    });
    ['dragleave', 'drop'].forEach(evt => {
        zone.addEventListener(evt, (e) => {
            e.preventDefault();
            zone.classList.remove('drag-over');
        });
    });

    zone.addEventListener('drop', (e) => {
        if (e.dataTransfer.files.length > 0) handleFile(e.dataTransfer.files[0]);
    });
}

let uploadedFile = null;

function handleFile(file) {
    // Validate: must be image
    const validTypes = ['image/jpeg', 'image/png', 'image/bmp', 'image/tiff', 'image/webp'];
    if (!validTypes.includes(file.type)) {
        showError('Please upload a medical image (JPEG, PNG, BMP, TIFF, or WebP).');
        return;
    }

    uploadedFile = file;

    // Show preview
    const preview = document.getElementById('previewContainer');
    const reader = new FileReader();
    reader.onload = (e) => {
        preview.innerHTML = `<img src="${e.target.result}" alt="Uploaded scan preview">`;
    };
    reader.readAsDataURL(file);

    // Hide old results
    document.getElementById('resultsSection')?.classList.add('hidden');
    document.getElementById('errorMessage')?.classList.add('hidden');

    // Enable predict button
    const btn = document.getElementById('predictBtn');
    if (btn) btn.disabled = false;
}

/* ── Prediction ──────────────────────────────────────────── */
async function predictImage() {
    if (!uploadedFile) return;

    const formData = new FormData();
    formData.append('file', uploadedFile);

    const predictBtn = document.getElementById('predictBtn');
    const resultsSection = document.getElementById('resultsSection');
    const loading = document.getElementById('loadingOverlay');
    const errorMsg = document.getElementById('errorMessage');

    predictBtn.disabled = true;
    loading?.classList.add('active');
    errorMsg?.classList.add('hidden');
    resultsSection?.classList.add('hidden');

    try {
        // Get project ID from the data attribute on the body or a meta tag
        const projectId = document.body.dataset.projectId || '';
        if (!projectId) throw new Error('Project ID not found.');

        // Step 1: Predict
        const predResp = await fetch(`/${projectId}/predict`, {
            method: 'POST',
            body: formData,
        });
        if (!predResp.ok) {
            const err = await predResp.json();
            throw new Error(err.detail || 'Prediction failed');
        }
        const predData = await predResp.json();
        renderPrediction(predData);

        // Step 2: Explain (XAI)
        const xaiResp = await fetch(`/${projectId}/explain`, {
            method: 'POST',
            body: formData,
        });
        if (xaiResp.ok) {
            const xaiData = await xaiResp.json();
            renderExplanations(xaiData.explanations);
        } else {
            document.getElementById('explainerGrid').innerHTML =
                '<div class="alert alert-error">XAI analysis failed. The explainers may not be available for this model.</div>';
        }

        resultsSection?.classList.remove('hidden');
        resultsSection?.scrollIntoView({ behavior: 'smooth', block: 'start' });

    } catch (err) {
        showError(err.message);
    } finally {
        loading?.classList.remove('active');
        predictBtn.disabled = false;
    }
}

/* ── Render Prediction Results ───────────────────────────── */
function renderPrediction(data) {
    const pred = data.predictions || data;

    // Predicted class
    const nameEl = document.getElementById('predictedDisease');
    if (nameEl) nameEl.textContent = pred.predicted_class || 'Unknown';

    // Confidence gauge
    const confidence = pred.confidence || 0;
    const gaugeEl = document.getElementById('confidenceGauge');
    if (gaugeEl) {
        const circumference = 2 * Math.PI * 50;
        const offset = circumference * (1 - confidence);
        gaugeEl.innerHTML = `
            <svg width="120" height="120" viewBox="0 0 120 120">
                <circle cx="60" cy="60" r="50" fill="none" stroke="var(--bg-input)"
                        stroke-width="10"/>
                <circle cx="60" cy="60" r="50" fill="none" stroke="var(--accent)"
                        stroke-width="10" stroke-dasharray="${circumference}"
                        stroke-dashoffset="${offset}" stroke-linecap="round"
                        style="transition: stroke-dashoffset 1s ease"/>
            </svg>
            <div class="percentage">${(confidence * 100).toFixed(1)}%</div>
        `;
    }

    // Probability bars
    const barContainer = document.getElementById('probabilityBars');
    if (barContainer && pred.probabilities) {
        const probs = pred.probabilities;
        const entries = Object.entries(probs).sort((a, b) => b[1] - a[1]);
        const colors = ['#3182ce', '#38a169', '#d69e2e', '#e53e3e', '#805ad5',
                        '#dd6b20', '#319795', '#d53f8c', '#2b6cb0', '#48bb78'];

        barContainer.innerHTML = entries.map(([label, value], i) => `
            <div class="prob-bar">
                <span class="label">${label}</span>
                <div class="track">
                    <div class="fill" style="width: ${(value * 100).toFixed(1)}%;
                         background: ${colors[i % colors.length]}"></div>
                </div>
                <span class="value">${(value * 100).toFixed(1)}%</span>
            </div>
        `).join('');
    }
}

/* ── Render XAI Explanations ──────────────────────────────── */
function renderExplanations(explainers) {
    const grid = document.getElementById('explainerGrid');
    if (!grid) return;

    if (!explainers || Object.keys(explainers).length === 0) {
        grid.innerHTML = '<p style="color: var(--text-muted);">No explanations available.</p>';
        return;
    }

    const entries = Object.values(explainers);
    grid.innerHTML = entries.map((exp, i) => {
        if (exp.error) {
            return `
                <div class="explainer-card">
                    <div class="card-header"><h4>${exp.label || 'Explainer'}</h4></div>
                    <div class="card-body">
                        <div class="alert alert-error">${exp.error}</div>
                    </div>
                </div>`;
        }

        const overlaySrc = exp.overlay_base64
            ? `data:image/png;base64,${exp.overlay_base64}` : '';
        const heatmapSrc = exp.heatmap_base64
            ? `data:image/png;base64,${exp.heatmap_base64}` : '';

        return `
            <div class="explainer-card" style="animation: fadeIn 0.4s ease ${i * 0.1}s both;">
                <div class="card-header">
                    <h4>${exp.label || 'Unknown'}</h4>
                    <span style="font-size:0.75rem; color:var(--text-muted); cursor:pointer;"
                          onclick="toggleHeatmap(this, '${heatmapSrc}', '${overlaySrc}')"
                          title="Toggle heatmap/overlay">🔄</span>
                </div>
                <div class="card-body">
                    <img src="${overlaySrc}" alt="${exp.label}" class="explainer-image"
                         style="cursor:pointer;" onclick="expandImage(this)">
                    <p>${exp.description || ''}</p>
                </div>
            </div>`;
    }).join('');
}

/* ── Toggle between heatmap and overlay view ─────────────── */
function toggleHeatmap(btn, heatmapSrc, overlaySrc) {
    const img = btn.closest('.explainer-card').querySelector('.explainer-image');
    if (!img) return;
    const isOverlay = img.src.includes(overlaySrc);
    img.src = isOverlay && heatmapSrc ? `data:image/png;base64,${heatmapSrc}` : `data:image/png;base64,${overlaySrc}`;
}

/* ── Expand image (lightbox) ──────────────────────────────── */
function expandImage(img) {
    const overlay = document.createElement('div');
    overlay.style.cssText = `
        position:fixed;inset:0;background:rgba(0,0,0,0.8);z-index:99999;
        display:flex;align-items:center;justify-content:center;cursor:pointer;
        animation:fadeIn 0.2s ease;
    `;
    const fullImg = document.createElement('img');
    fullImg.src = img.src;
    fullImg.style.cssText = 'max-width:90%;max-height:90%;border-radius:8px;box-shadow:0 8px 32px rgba(0,0,0,0.5);';
    overlay.appendChild(fullImg);
    overlay.addEventListener('click', () => overlay.remove());
    document.body.appendChild(overlay);
}

/* ── Download Report ──────────────────────────────────────── */
async function downloadReport() {
    const projectId = document.body.dataset.projectId || '';
    if (!projectId) {
        showError('Project ID not available for report download.');
        return;
    }

    const loading = document.getElementById('loadingOverlay');
    loading?.classList.add('active');

    try {
        const resp = await fetch(`/${projectId}/report`);
        if (!resp.ok) throw new Error('Report generation failed');

        const blob = await resp.blob();
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `${projectId}_report_${new Date().toISOString().slice(0,10)}.pdf`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    } catch (err) {
        showError(err.message);
    } finally {
        loading?.classList.remove('active');
    }
}

/* ── Error Display ────────────────────────────────────────── */
function showError(msg) {
    const el = document.getElementById('errorMessage');
    if (!el) {
        alert(msg);
        return;
    }
    el.textContent = msg;
    el.classList.remove('hidden');
    el.scrollIntoView({ behavior: 'smooth', block: 'center' });
}

/* ── Utility: fade-in animation ───────────────────────────── */
const styleSheet = document.createElement('style');
styleSheet.textContent = `
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .hidden { display: none !important; }
`;
document.head.appendChild(styleSheet);

/* ── Init on DOM ready ────────────────────────────────────── */
document.addEventListener('DOMContentLoaded', () => {
    initTheme();
    initUploadZone();

    // Global predict button
    const predictBtn = document.getElementById('predictBtn');
    if (predictBtn) predictBtn.addEventListener('click', predictImage);

    // Global download report button
    const reportBtn = document.getElementById('reportBtn');
    if (reportBtn) reportBtn.addEventListener('click', downloadReport);
});
