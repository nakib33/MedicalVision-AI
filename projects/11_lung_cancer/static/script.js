/**
 * Lung Cancer CT Scan Classification - Project-Specific Script
 *
 * Enhances the shared project_ui.html with:
 *   - Lung-cancer-branded prediction display
 *   - Per-class colour-coded probability bars
 *   - Animated UI transitions
 */

document.addEventListener('DOMContentLoaded', () => {
    // ── Class colour lookup ──────────────────────────────────────
    const CLASS_COLORS = {
        'Benign': '#38a169',
        'Malignant': '#e53e3e',
        'Adenocarcinoma': '#6b46c1',
        'Large Cell Carcinoma': '#dd6b20',
        'Normal': '#3182ce',
        'Squamous Cell Carcinoma': '#319795',
    };

    // ── Brand the header ─────────────────────────────────────────
    const h1 = document.querySelector('h1');
    if (h1) {
        h1.style.borderLeft = '4px solid #6b46c1';
        h1.style.paddingLeft = '1rem';
    }

    // ── Colour code the predicted disease name ───────────────────
    const diseaseName = document.getElementById('predictedDisease');
    if (diseaseName) {
        const observer = new MutationObserver(() => {
            const text = diseaseName.textContent.trim();
            if (text && text !== '—') {
                const colour = CLASS_COLORS[text] || '#6b46c1';
                diseaseName.style.color = colour;
            }
        });
        observer.observe(diseaseName, { childList: true, characterData: true, subtree: true });
    }

    // ── Colour coded probability bars ────────────────────────────
    const probBars = document.getElementById('probabilityBars');
    if (probBars) {
        const barObserver = new MutationObserver(() => {
            const items = probBars.querySelectorAll('.prob-bar-item');
            items.forEach(item => {
                const label = item.querySelector('.prob-bar-label');
                if (label) {
                    const cls = label.textContent.trim();
                    const colour = CLASS_COLORS[cls];
                    if (colour) {
                        label.style.color = colour;
                        label.style.fontWeight = '600';
                    }
                }
            });
        });
        barObserver.observe(probBars, { childList: true, subtree: true });
    }

    // ── Apply accent colour to the confidence gauge ──────────────
    const gaugeCircle = document.querySelector('.confidence-gauge circle:last-child');
    if (gaugeCircle) {
        gaugeCircle.setAttribute('stroke', '#6b46c1');
    }
});
