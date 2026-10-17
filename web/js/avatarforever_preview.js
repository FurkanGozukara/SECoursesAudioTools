import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

const PANEL = Symbol("avatarForeverPreview");

function findNode(executionId) {
    const ids = String(executionId).split(":");
    let graph = app.graph;
    for (const id of ids.slice(0, -1)) graph = graph?.getNodeById(id)?.subgraph;
    return graph?.getNodeById(ids.at(-1));
}

function installPanel(node) {
    if (node.comfyClass !== "SEAvatarForeverPreview" || node[PANEL]) return;

    const root = document.createElement("div");
    root.style.cssText = "display:flex;flex-direction:column;gap:8px;width:100%;height:100%;padding:8px;box-sizing:border-box;color:#e4eee9;font:14px sans-serif;";
    const status = document.createElement("div");
    status.textContent = "Live preview while sampling · Final video when complete";
    const stage = document.createElement("div");
    stage.style.cssText = "position:relative;flex:1;min-height:0;background:#101b17;border-radius:6px;overflow:hidden;";
    const idle = document.createElement("div");
    idle.textContent = "Run the workflow to see the preview here.";
    idle.style.cssText = "position:absolute;inset:0;display:flex;align-items:center;justify-content:center;padding:24px;text-align:center;color:#b3c5bc;";
    const image = document.createElement("img");
    image.alt = "AvatarForever live sampling preview";
    const video = document.createElement("video");
    video.controls = true;
    video.preload = "metadata";
    for (const element of [image, video]) {
        element.style.cssText = "position:absolute;inset:0;width:100%;height:100%;object-fit:contain;";
        element.hidden = true;
    }
    image.style.pointerEvents = "none";
    for (const event of ["pointerdown", "mousedown", "mouseup", "click", "dblclick"]) {
        video.addEventListener(event, (e) => e.stopPropagation());
    }
    stage.append(idle, image, video);
    root.append(status, stage);
    const widget = node.addDOMWidget("avatar_preview", "preview", root, {
        serialize: false, hideOnZoom: false, getMinHeight: () => 300,
        surfaces: { canvas: "shown", vueNode: "shown", panel: "never" },
        getValue: () => undefined, setValue: () => {},
    });
    widget.serialize = false;
    node.setSize([Math.max(node.size[0], 420), Math.max(node.size[1], 500)]);
    // This panel owns both media elements; the native still cache otherwise masks the MP4.
    node.hideOutputImages = true;
    node.onDrawBackground = function () {};

    function clearPreview() {
        image.removeAttribute("src");
        image.hidden = true;
    }
    node[PANEL] = {
        preview(url) {
            image.src = url;
            image.hidden = false;
            video.pause();
            video.hidden = true;
            idle.style.display = "none";
            status.textContent = "Live preview · Sampling in progress";
        },
        output(output) {
            const file = output?.images?.[0];
            if (!file?.filename) return;
            clearPreview();
            const url = api.apiURL("/view?" + new URLSearchParams({
                filename: file.filename, subfolder: file.subfolder ?? "", type: file.type ?? "output",
            }));
            if (video.getAttribute("src") !== url) video.src = url;
            video.hidden = false;
            idle.style.display = "none";
            status.textContent = "Final video · Play to watch with audio";
        },
    };
    const onExecuted = node.onExecuted;
    node.onExecuted = function (output) {
        onExecuted?.apply(this, arguments);
        this[PANEL].output(output);
    };
    const onRemoved = node.onRemoved;
    node.onRemoved = function () {
        clearPreview();
        video.pause();
        video.removeAttribute("src");
        video.load();
        delete this[PANEL];
        return onRemoved?.apply(this, arguments);
    };
}

api.addEventListener("se_avatarforever_preview", ({ detail }) => {
    for (const id of detail.node_ids ?? []) findNode(id)?.[PANEL]?.preview(detail.image);
});

app.registerExtension({
    name: "SECoursesAudioTools.AvatarForeverPreview",
    nodeCreated: installPanel,
    onNodeOutputsUpdated(outputs) {
        for (const [id, output] of Object.entries(outputs ?? {})) findNode(id)?.[PANEL]?.output(output);
    },
});
