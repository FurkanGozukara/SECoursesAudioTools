// Pure helpers shared by the SE LTX-2.5 A2V frontend extension (no ComfyUI imports, unit-testable with node).
// Mirrors resolution_from_aspect() / stage1_size() in ltx25_a2v_nodes.py - keep both in sync.

export const PIXEL_BUDGET = 1920 * 1080;
export const TARGET_MULTIPLE = 64;
export const SPATIAL_MULTIPLE = 32;

function compareKeys(a, b) {
    for (let i = 0; i < a.length; i++) {
        if (a[i] < b[i]) return -1;
        if (a[i] > b[i]) return 1;
    }
    return 0;
}

export function resolutionFromAspect(width, height, pixelBudget = PIXEL_BUDGET, multiple = TARGET_MULTIPLE) {
    width = Number(width);
    height = Number(height);
    if (!(width > 0) || !(height > 0)) {
        return [1920, 1080];
    }
    const aspect = width / height;
    const standards = [
        [16 / 9, [1920, 1080]],
        [9 / 16, [1080, 1920]],
    ];
    for (const [standardAspect, size] of standards) {
        if (Math.abs(aspect / standardAspect - 1) < 0.02) {
            return size;
        }
    }

    const idealW = Math.sqrt(pixelBudget * aspect);
    const widths = [...new Set([Math.floor(idealW / multiple) * multiple, Math.ceil(idealW / multiple) * multiple])].sort((a, b) => a - b);
    let best = null;
    for (const w of widths) {
        if (w < multiple) continue;
        const idealH = w / aspect;
        const heights = [...new Set([Math.floor(idealH / multiple) * multiple, Math.ceil(idealH / multiple) * multiple])].sort((a, b) => a - b);
        for (const h of heights) {
            if (h < multiple) continue;
            const aspectError = Math.round(Math.abs(Math.log((w / h) / aspect)) * 1e6) / 1e6;
            const pixelError = Math.abs(w * h - pixelBudget);
            const key = [aspectError, pixelError, w];
            if (best === null || compareKeys(key, best.key) < 0) {
                best = { key, w, h };
            }
        }
    }
    return best ? [best.w, best.h] : [1920, 1080];
}

export function stage1Size(targetWidth, targetHeight) {
    const gw = Math.ceil(targetWidth / 2 / SPATIAL_MULTIPLE) * SPATIAL_MULTIPLE;
    const gh = Math.ceil(targetHeight / 2 / SPATIAL_MULTIPLE) * SPATIAL_MULTIPLE;
    return [Math.max(SPATIAL_MULTIPLE * 2, gw), Math.max(SPATIAL_MULTIPLE * 2, gh)];
}

export function parseFileValue(value) {
    let text = String(value ?? "").trim();
    let type = "input";
    const annotated = text.match(/^(.*?)\s*\[(input|output|temp)\]$/);
    if (annotated) {
        text = annotated[1].trim();
        type = annotated[2];
    }
    text = text.replace(/[\\/]+/g, "/").replace(/^\//, "").replace(/\/$/, "");
    const slash = text.lastIndexOf("/");
    return {
        filename: slash === -1 ? text : text.slice(slash + 1),
        subfolder: slash === -1 ? "" : text.slice(0, slash),
        type,
    };
}
