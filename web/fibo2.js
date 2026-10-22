import { app } from "../../../scripts/app.js";

const NO_IMAGE = "(none - disabled)";

// Fibo 2.0 presets opt in with the gallery property fibo2_images_only. The gallery then takes images only, and its
// prompt box sends the text as typed: the frontend's dynamic-prompt pass would pick {a|b} with Math.random and could
// change a JSON caption, while SEFibo2GalleryEncode resolves choices with the shared seed and passes JSON captions through.
function configureGallery(node) {
    if (!node.properties?.fibo2_images_only || !node.__refGallery) return;
    const ui = node.__refGallery;
    if (!ui.fibo2ProfileInstalled) {
        ui.fibo2ProfileInstalled = true;
        const render = ui.render.bind(ui);
        ui.render = () => {
            render();
            const count = ui.state.images.length;
            ui.counter.textContent = `${count} image${count === 1 ? "" : "s"} to edit (up to 4 recommended)`;
            ui.soundtrackHint.style.setProperty("display", "none", "important");
        };
    }
    ui.fileInput.accept = "image/*";
    ui.addButton.textContent = "+ Add images to edit";
    ui.addButton.title = "Add, paste, remove or reorder images. One image: edit it. Several: refer to them as @image1, @image2 in the instruction. Empty gallery = text to image.";
    ui.root.classList.add("secourses-flux3-gallery");
    ui.minContentHeight = () => 360;
    ui.cardsWrap.style.maxHeight = "128px";
    ui.root.style.overflowY = "auto";
    ui.empty.textContent = "Optional images to edit: add, drop or paste. Leave empty for text to image.";
    ui.promptWidget.serializeValue = () => ui.promptWidget.value;
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
    name: "SECourses.Fibo2",
    loadedGraphNode: configureGallery,
    nodeCreated(node) {
        if (node.comfyClass !== "SEFibo2EditCanvas") return;
        const image = node.widgets.find(w => w.name === "image");
        const upload = node.widgets.find(w => w.name === "upload");
        if (upload) upload.serialize = false;
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
            if (value === NO_IMAGE) return clearPreview();
            return original?.apply(this, arguments);
        };
        const paint = node.addWidget("button", "Paint mask (native editor)", null, () => {
            if (image.value === NO_IMAGE) {
                app.extensionManager.toast.add({severity: "info", summary: "Upload the image to edit first", life: 3000});
                return;
            }
            app.canvas.deselectAllNodes();
            app.canvas.selectNode(node);
            app.extensionManager.command.execute("Comfy.MaskEditor.OpenMaskEditor");
        });
        paint.serialize = false;
        paint.options.serialize = false;
        const clear = node.addWidget("button", "Clear image + mask", null, () => {
            image.value = NO_IMAGE;
            clearPreview();
        });
        clear.serialize = false;
        clear.options.serialize = false;
    },
});
