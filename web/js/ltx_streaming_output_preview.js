import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

const TARGET_TITLE = "STREAMING SAVE OUTPUT - VHS meta-batch MP4";

function chainCallback(object, property, callback) {
    const original = object?.[property];
    object[property] = function () {
        const result = original?.apply(this, arguments);
        return callback.apply(this, arguments) ?? result;
    };
}

function isTargetOutputNode(node) {
    return node?.type === "VHS_VideoCombine" && (
        node?.properties?.ltx_streaming_native_controls_preview === true ||
        node?.title === TARGET_TITLE
    );
}

function stopCanvasHandling(element) {
    for (const eventName of ["pointerdown", "mousedown", "mouseup", "click", "dblclick", "wheel", "contextmenu"]) {
        element.addEventListener(eventName, (event) => {
            event.stopPropagation();
        }, true);
    }
}

function getVhsPreviewWidget(node) {
    return node.widgets?.find((widget) => widget.name === "videopreview");
}

function getPreviewParams(node) {
    return getVhsPreviewWidget(node)?.value?.params;
}

function makeViewUrl(params) {
    if (!params?.filename || params?.format?.split("/")?.[0] !== "video") {
        return null;
    }

    const viewParams = { ...params, timestamp: Date.now() };
    delete viewParams.fullpath;
    return api.apiURL("/view?" + new URLSearchParams(viewParams));
}

function fitNativePreview(node, widget) {
    const aspectRatio = widget.aspectRatio || 16 / 9;
    const previewHeight = Math.max(180, (node.size[0] - 20) / aspectRatio) + 10;
    widget.computedHeight = previewHeight;
    app.graph?.setDirtyCanvas(true, false);
}

function hideVhsInlinePreview(node) {
    const vhsWidget = getVhsPreviewWidget(node);
    if (!vhsWidget) {
        return;
    }

    if (!vhsWidget.ltxNativePreviewHidden) {
        vhsWidget.ltxNativePreviewHidden = true;
        vhsWidget.computeSize = function (width) {
            return [width, -4];
        };
    }

    if (vhsWidget.parentEl) {
        vhsWidget.parentEl.hidden = true;
    }
}

function updateNativePreview(node) {
    if (!isTargetOutputNode(node)) {
        return;
    }

    const widget = node.ltxNativeOutputPreview;
    const url = makeViewUrl(getPreviewParams(node));
    if (!widget || !url || widget.videoEl.src === url) {
        return;
    }

    widget.videoEl.src = url;
    widget.videoEl.hidden = false;
    fitNativePreview(node, widget);
}

function ensureNativePreview(node) {
    if (!isTargetOutputNode(node)) {
        return;
    }

    if (!node.ltxNativeOutputPreview) {
        const container = document.createElement("div");
        container.style.width = "100%";
        container.style.padding = "0";
        container.style.margin = "0";
        stopCanvasHandling(container);

        const video = document.createElement("video");
        video.controls = true;
        video.loop = false;
        video.muted = false;
        video.preload = "metadata";
        video.style.width = "100%";
        video.style.background = "#000";
        video.style.borderRadius = "4px";
        video.style.display = "block";
        video.hidden = true;
        stopCanvasHandling(video);

        container.appendChild(video);

        const widget = node.addDOMWidget("native_output_preview", "preview", container, {
            serialize: false,
            hideOnZoom: false,
            getValue() {
                return null;
            },
            setValue() {},
        });
        widget.videoEl = video;
        widget.aspectRatio = 16 / 9;
        widget.computeSize = function (width) {
            const aspectRatio = this.aspectRatio || 16 / 9;
            return [width, Math.max(180, (node.size[0] - 20) / aspectRatio) + 10];
        };
        widget.serialize = false;

        video.addEventListener("loadedmetadata", () => {
            if (video.videoWidth && video.videoHeight) {
                widget.aspectRatio = video.videoWidth / video.videoHeight;
                fitNativePreview(node, widget);
            }
        });

        node.ltxNativeOutputPreview = widget;
    }

    const vhsWidget = getVhsPreviewWidget(node);
    if (vhsWidget?.updateSource && !vhsWidget.ltxNativeUpdateChained) {
        vhsWidget.ltxNativeUpdateChained = true;
        chainCallback(vhsWidget, "updateSource", function () {
            requestAnimationFrame(() => {
                hideVhsInlinePreview(node);
                updateNativePreview(node);
            });
        });
    }

    hideVhsInlinePreview(node);
    updateNativePreview(node);
}

if (!globalThis.__ltxStreamingOutputNativePreviewRegistered) {
    globalThis.__ltxStreamingOutputNativePreviewRegistered = true;

    app.registerExtension({
        name: "SECoursesAudioTools.LTXStreamingOutputNativePreview",

        async beforeRegisterNodeDef(nodeType, nodeData) {
            if (nodeData.name !== "VHS_VideoCombine") {
                return;
            }

            chainCallback(nodeType.prototype, "onNodeCreated", function () {
                requestAnimationFrame(() => ensureNativePreview(this));
            });

            chainCallback(nodeType.prototype, "onExecuted", function () {
                requestAnimationFrame(() => {
                    ensureNativePreview(this);
                    updateNativePreview(this);
                });
            });
        },

        async nodeCreated(node) {
            requestAnimationFrame(() => ensureNativePreview(node));
        },

        async loadedGraphNode(node) {
            requestAnimationFrame(() => ensureNativePreview(node));
        },
    });
}
