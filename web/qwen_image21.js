import { app } from "../../../scripts/app.js";

function addMaskEditorHelp(dialog, node) {
    const thickness = dialog.querySelector('[data-testid="brush-thickness-input"]');
    const blend = dialog.querySelector(".maskEditor_sidePanelDropdown");
    if (!thickness || !blend) return;
    const brush = thickness.parentElement.parentElement.parentElement;
    const layers = blend.parentElement.parentElement;
    const numbers = brush.querySelectorAll('input[type="number"]');
    if (numbers.length !== 4) return;
    const sidebar = brush.closest(".overflow-y-auto");
    if (!sidebar) return;
    sidebar.style.width = "min(320px, 42vw)";
    sidebar.style.flexShrink = "0";
    for (const slider of sidebar.querySelectorAll('input[type="range"]')) slider.style.width = "100%";

    const defaults = node.properties?.mask_editor_brush;
    const applyBrushDefaults = () => {
        ["thickness", "opacity", "hardness"].forEach((key, i) => {
            if (!Number.isFinite(defaults?.[key])) return;
            numbers[i].value = defaults[key];
            numbers[i].dispatchEvent(new Event("input", {bubbles: true}));
        });
    };
    if (defaults && !dialog.dataset.seQwenBrushDefaults) {
        dialog.dataset.seQwenBrushDefaults = "true";
        applyBrushDefaults();
    }

    function hint(container, text, labelControls = true) {
        const help = document.createElement("p");
        help.className = "se-qwen-mask-help";
        help.textContent = text;
        help.style.cssText = "margin:0;font-size:12px;line-height:1.4;color:var(--descrip-text, #b8bec8);overflow-wrap:anywhere;";
        container.append(help);
        for (const input of labelControls ? container.querySelectorAll("input, select") : []) {
            input.title = text;
        }
    }

    if (!brush.dataset.seQwenMaskHelp) {
        brush.dataset.seQwenMaskHelp = "true";
        const heading = brush.querySelector("h3");
        const intro = document.createElement("p");
        intro.className = "se-qwen-mask-help";
        intro.textContent = "Qwen inpaint: brush settings affect new strokes. Save the mask, then use Denoise strength below Paint mask to control how much the model redraws.";
        intro.style.cssText = "margin:0;padding:8px;border:1px solid var(--border-color, #555);border-radius:6px;font-size:12px;line-height:1.4;";
        heading.after(intro);
        const reset = brush.querySelector("button");
        reset.title = "Reset brush settings for future strokes. Existing painted pixels stay unchanged.";
        if (defaults) {
            reset.title += ` Preset: thickness ${defaults.thickness}, opacity ${defaults.opacity}, hardness ${defaults.hardness}.`;
            reset.addEventListener("click", () => queueMicrotask(applyBrushDefaults));
        }
        const shape = brush.querySelector(".maskEditor_sidePanelBrushShapeCircle")?.parentElement.parentElement;
        if (shape) hint(shape, "Changes the saved stroke shape: round or square.");
        const color = brush.querySelector('input[type="color"]');
        if (color) hint(color.parentElement, "Paint layer only: adds color to the source image. Mask strokes use coverage, not this color.");
        [
            "Changes the saved stroke size. Larger covers more of the image.",
            "Mask: 1 = full edit coverage; lower values blend with the original. Paint: controls color opacity. Repeated strokes build coverage.",
            "Stroke edges: 1 = sharp; lower = softer and slightly wider. Soft mask edges blend the edit into the original.",
            "Brush dab spacing, not generation steps. Lower gives smoother coverage; higher can leave gaps. 5 is a useful starting point.",
        ].forEach((text, i) => hint(numbers[i].parentElement.parentElement, text));
    }
    if (!layers.dataset.seQwenMaskHelp) {
        layers.dataset.seQwenMaskHelp = "true";
        const previewOpacity = layers.querySelector('input[type="range"]');
        if (previewOpacity) hint(previewOpacity.parentElement, "Preview only: fades the mask overlay. Does not change the saved mask or output.");
        hint(blend.parentElement, "Preview only: Black / White / Negative changes how the overlay looks. Does not change which pixels are edited.");
        blend.parentElement.style.flexWrap = "wrap";
        blend.parentElement.style.height = "auto";
        hint(layers, "Layer checkboxes only show or hide previews. Mask tools change the edit area; the paint tool changes source-image pixels. Save applies your edits.", false);
    }
}

function watchMaskEditor() {
    const annotate = () => {
        const node = Object.values(app.canvas?.selected_nodes ?? {}).find(n => n.comfyClass === "SEQwenImage21Canvas");
        if (!node) return;
        for (const dialog of document.querySelectorAll('[role="dialog"]')) addMaskEditorHelp(dialog, node);
    };
    new MutationObserver(annotate).observe(document.body, {childList: true, subtree: true});
    annotate();
}

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
    setup: watchMaskEditor,
    loadedGraphNode: configure,
    nodeCreated(node) {
        if (node.comfyClass === "SEQwenImage21Prepare") {
            node.widgets.find(w => w.name === "reference_resolution").label = "Reference / edit size (0 = original)";
            node.widgets.find(w => w.name === "transparent").label = "Request transparent background";
        }
        if (node.comfyClass !== "SEQwenImage21Canvas") return;
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
        paint.options.serialize = false;
        const denoise = node.widgets.find(w => w.name === "denoise");
        if (denoise) {
            denoise.label = "Denoise strength (img2img / mask)";
            // Move only the new serialized widget past the upload/paint buttons.
            // Serialized values stay in schema order: image, denoise.
            node.widgets.splice(node.widgets.indexOf(denoise), 1);
            node.widgets.push(denoise);
            const onConfigure = node.onConfigure;
            node.onConfigure = function (data) {
                const result = onConfigure?.apply(this, arguments);
                // Old saved canvases had an upload-button placeholder here.
                if (typeof denoise.value !== "number") denoise.value = 0.85;
                return result;
            };
        }
        const clear = node.addWidget("button", "Clear init image + mask", null, () => {
            image.value = "(none - disabled)";
            clearPreview();
        });
        clear.serialize = false;
        clear.options.serialize = false;
    },
});
