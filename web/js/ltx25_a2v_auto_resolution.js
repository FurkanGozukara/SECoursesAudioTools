import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";
import { parseFileValue, resolutionFromAspect } from "./ltx25_a2v_resolution.js";

// SE LTX-2.5 audio-to-video helpers:
//  * when an image is uploaded / picked on "SELTX25LoadImageOptional", derive the
//    target video size from its aspect ratio at the LTX-2.5 1080p pixel budget
//    (mirrors resolution_from_aspect() in ltx25_a2v_nodes.py);
//  * show the backend "info" text (ui.text) on the SE LTX-2.5 nodes after a run.

const IMAGE_NODE = "SELTX25LoadImageOptional";
const INFO_NODES = new Set(["SELTX25LoadImageOptional", "SELTX25AudioPrepare"]);
const NONE_OPTION = "none";
const LOG = "[SE LTX-2.5 A2V]";

function isImageNode(node) {
    return node?.comfyClass === IMAGE_NODE || node?.type === IMAGE_NODE;
}

function viewUrl(value) {
    const parsed = parseFileValue(value);
    if (!parsed.filename) {
        return null;
    }
    const params = new URLSearchParams({
        filename: parsed.filename,
        type: parsed.type,
        subfolder: parsed.subfolder,
        rand: String(Date.now()),
    });
    return api.apiURL("/view?" + params.toString());
}

function findWidget(node, name) {
    return node.widgets?.find((w) => w.name === name);
}

function setWidgetValue(node, name, value) {
    const widget = findWidget(node, name);
    if (!widget) {
        return false;
    }
    widget.value = value;
    try {
        widget.callback?.(value, app.canvas, node);
    } catch (error) {
        console.warn(LOG, "widget callback failed for", name, error);
    }
    return true;
}

function applyAutoResolution(node, value) {
    if (!value || value === NONE_OPTION) {
        return;
    }
    const autoWidget = findWidget(node, "auto_resolution_from_image");
    if (autoWidget && autoWidget.value === false) {
        return;
    }
    const url = viewUrl(value);
    if (!url) {
        return;
    }
    const image = new Image();
    image.onload = () => {
        const [targetW, targetH] = resolutionFromAspect(image.naturalWidth, image.naturalHeight);
        const changedW = setWidgetValue(node, "target_width", targetW);
        const changedH = setWidgetValue(node, "target_height", targetH);
        if (changedW || changedH) {
            node.setDirtyCanvas?.(true, true);
            app.graph?.setDirtyCanvas(true, true);
            console.log(LOG, `image ${image.naturalWidth}x${image.naturalHeight} -> target video ${targetW}x${targetH}`);
        }
    };
    image.onerror = () => {
        console.warn(LOG, "could not load image to detect its aspect ratio:", url);
    };
    image.src = url;
}

// Wrap the combo callback.  The core image-upload widget assigns its own callback while the
// node is built, so this may run more than once; it only re-wraps when the callback changed.
function wrapImageCallback(node, imageWidget) {
    if (imageWidget.callback === node.__seLtx25Wrapper) {
        return;
    }
    const originalCallback = imageWidget.callback;
    const wrapper = function (value, ...rest) {
        const result = originalCallback?.call(this, value, ...rest);
        try {
            node.__seLtx25HandleChange?.(value);
        } catch (error) {
            console.warn(LOG, error);
        }
        return result;
    };
    imageWidget.callback = wrapper;
    node.__seLtx25Wrapper = wrapper;
}

function hookImageNode(node) {
    const imageWidget = findWidget(node, "image");
    if (!imageWidget) {
        return;
    }
    if (!node.__seLtx25HandleChange) {
        node.__seLtx25LastImage = imageWidget.value;
        node.__seLtx25HandleChange = (value) => {
            const next = value ?? imageWidget.value;
            if (next === node.__seLtx25LastImage) {
                return;
            }
            node.__seLtx25LastImage = next;
            applyAutoResolution(node, next);
        };
        const originalWidgetChanged = node.onWidgetChanged;
        node.onWidgetChanged = function (name, value, oldValue, widget) {
            const result = originalWidgetChanged?.apply(this, arguments);
            if (name === "image") {
                try {
                    node.__seLtx25HandleChange?.(value);
                } catch (error) {
                    console.warn(LOG, error);
                }
            }
            return result;
        };
    }
    wrapImageCallback(node, imageWidget);
}

function scheduleHooks(node) {
    try {
        hookImageNode(node);
    } catch (error) {
        console.warn(LOG, "could not hook image node", error);
    }
    // Re-check after the node finished building (works in hidden tabs too, unlike requestAnimationFrame).
    for (const delay of [0, 250, 1000]) {
        setTimeout(() => {
            try {
                hookImageNode(node);
            } catch (error) {
                console.warn(LOG, "could not hook image node", error);
            }
        }, delay);
    }
}

function ensureInfoWidget(node) {
    let widget = findWidget(node, "se_info");
    if (widget) {
        return widget;
    }
    const element = document.createElement("textarea");
    element.className = "comfy-multiline-input";
    element.readOnly = true;
    element.spellcheck = false;
    element.style.width = "100%";
    element.style.height = "100%";
    element.style.resize = "none";
    element.style.opacity = "0.85";
    element.style.fontSize = "11px";
    element.style.lineHeight = "1.3";
    element.title = "Result of the last run";
    for (const eventName of ["pointerdown", "mousedown", "wheel", "dblclick", "contextmenu"]) {
        element.addEventListener(eventName, (event) => event.stopPropagation(), true);
    }
    widget = node.addDOMWidget("se_info", "customtext", element, {
        serialize: false,
        hideOnZoom: true,
        getValue() {
            return element.value;
        },
        setValue(value) {
            element.value = value ?? "";
        },
    });
    widget.serialize = false;
    widget.computeSize = function (width) {
        return [width, 70];
    };
    widget.computeLayoutSize = () => ({ minHeight: 70, minWidth: 200 });
    node.setSize?.([Math.max(node.size[0], 320), node.computeSize()[1]]);
    return widget;
}

function showInfo(node, message) {
    const lines = message?.text;
    if (!Array.isArray(lines) || lines.length === 0) {
        return;
    }
    const text = lines.map((line) => (Array.isArray(line) ? line.join("") : String(line))).join("\n");
    const widget = ensureInfoWidget(node);
    widget.value = text;
    if (widget.element) {
        widget.element.value = text;
    }
    node.setDirtyCanvas?.(true, true);
}

app.registerExtension({
    name: "SECoursesAudioTools.LTX25AudioToVideoHelpers",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (!INFO_NODES.has(nodeData.name)) {
            return;
        }
        const originalOnExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            const result = originalOnExecuted?.apply(this, arguments);
            try {
                showInfo(this, message);
            } catch (error) {
                console.warn(LOG, "could not show info", error);
            }
            return result;
        };
    },

    async nodeCreated(node) {
        if (!isImageNode(node)) {
            return;
        }
        scheduleHooks(node);
    },

    async loadedGraphNode(node) {
        if (!isImageNode(node)) {
            return;
        }
        // Values restored from the workflow are not "changes": sync the last seen image first.
        const imageWidget = findWidget(node, "image");
        if (imageWidget) {
            node.__seLtx25LastImage = imageWidget.value;
        }
        scheduleHooks(node);
        setTimeout(() => {
            const widget = findWidget(node, "image");
            if (widget) {
                node.__seLtx25LastImage = widget.value;
            }
        }, 0);
    },
});
