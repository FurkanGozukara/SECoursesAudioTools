import { app } from "../../../scripts/app.js";

// Reuse the installed gallery; this opt-in profile only hides video controls.
function configure(node) {
    if (!node.properties?.qwen21_images_only || !node.__refGallery) return;
    const ui = node.__refGallery;
    if (!ui.qwen21ProfileInstalled) {
        ui.qwen21ProfileInstalled = true;
        const render = ui.render.bind(ui);
        ui.render = () => {
            render();
            ui.counter.textContent = `${ui.state.images.length} images (10 total incl. init)`;
            ui.soundtrackHint.style.setProperty("display", "none", "important");
        };
    }
    ui.fileInput.accept = "image/*";
    ui.addButton.textContent = "+ Add reference images";
    ui.addButton.title = "Add, paste, remove or reorder images. Up to 10 total, including the optional init canvas.";
    ui.soundtrackHint.style.setProperty("display", "none", "important");
    ui.trimToggle.style.display = "none";
    ui.continuationRow.style.display = "none";
    ui.mergeToggle.style.setProperty("display", "none", "important");
    for (const widget of node.widgets ?? []) {
        if (["video_fps", "max_seconds", "match_batch_init_media"].includes(widget.name)) {
            widget.hidden = true;
            widget.computeSize = () => [0, -4];
        }
    }
    ui.refreshLayout();
    ui.render();
}

app.registerExtension({
    name: "SECourses.QwenImage21Gallery",
    loadedGraphNode: configure,
    nodeCreated(node) {
        if (node.comfyClass !== "SEQwenImage21Canvas") return;
        const image = node.widgets.find(w => w.name === "image");
        const original = image.callback;
        const clearPreview = () => {
            delete app.nodeOutputs?.[String(node.id)];
            delete app.nodePreviewImages?.[String(node.id)];
            node.imgs = null;
            node.images = null;
            node.imageIndex = null;
            node.setDirtyCanvas(true, true);
        };
        image.callback = function (value) {
            if (value === "(none - disabled)") return clearPreview();
            return original?.apply(this, arguments);
        };
        const paint = node.addWidget("button", "Paint mask (native editor)", null, () => {
            if (image.value === "(none - disabled)") {
                app.extensionManager.toast.add({severity:"info", summary:"Upload an init image first", life:3000});
                return;
            }
            app.canvas.deselectAllNodes();
            app.canvas.selectNode(node);
            app.extensionManager.command.execute("Comfy.MaskEditor.OpenMaskEditor");
        });
        paint.serialize = false;
        const clear = node.addWidget("button", "Clear init image + mask", null, () => {
            image.value = "(none - disabled)";
            clearPreview();
        });
        clear.serialize = false;
    },
});
