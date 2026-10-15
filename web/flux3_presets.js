import { app } from "../../../scripts/app.js";

let nativeNoteStyle;

function configureLoraStack(node) {
    if (!node.properties?.flux3_lora_stack || node.__flux3LoraStack) return;
    node.__flux3LoraStack = true;
    const update = () => {
        const count = node.widgets.find(w => w.name === "num_loras").value;
        const simple = node.widgets.find(w => w.name === "mode").value === "simple";
        for (const w of node.widgets) {
            const match = /^lora_(\d+)_(name|strength|model_strength|clip_strength)$/.exec(w.name);
            if (!match) continue;
            const show = Number(match[1]) <= count && (match[2] === "name" || (simple ? match[2] === "strength" : match[2] !== "strength"));
            // Easy Use restores the widget type, but Nodes 2.0 also needs its hidden flag reset.
            w.hidden = !show;
        }
        node.setSize([node.size[0], Math.max(260, node.computeSize()[1])]);
        node.setDirtyCanvas(true, true);
    };
    for (const name of ["num_loras", "mode"]) {
        const widget = node.widgets.find(w => w.name === name);
        const callback = widget.callback;
        widget.callback = function () {
            const result = callback?.apply(this, arguments);
            // Vue commits the new value after invoking the callback.
            requestAnimationFrame(update);
            return result;
        };
    }
    update();
    requestAnimationFrame(update);
}

function configureLayoutSync(node) {
    if (!node.properties?.flux3_layout_node || node.__flux3LayoutSync) return;
    node.__flux3LayoutSync = true;
    const sync = () => {
        const box = node.graph?.getNodeById(node.properties.flux3_layout_node);
        if (!box) return;
        const source = Object.fromEntries(node.widgets.map(w => [w.name, w]));
        const target = Object.fromEntries(box.widgets.map(w => [w.name, w]));
        const width = source.width.value, height = source.height.value;
        const oldWidth = target.width.value, oldHeight = target.height.value;
        if (width === oldWidth && height === oldHeight) return;
        target.editor_state.value = target.editor_state.value.map(b => ({
            ...b, x: Math.round(b.x * width / oldWidth), y: Math.round(b.y * height / oldHeight),
            width: Math.round(b.width * width / oldWidth), height: Math.round(b.height * height / oldHeight),
        }));
        target.width.value = width;
        target.height.value = height;
        box.setDirtyCanvas(true, true);
    };
    for (const widget of node.widgets) {
        const callback = widget.callback;
        widget.callback = function () {
            const result = callback?.apply(this, arguments);
            sync();
            return result;
        };
    }
    sync();
}

function configure(node) {
    configureLayoutSync(node);
    configureLoraStack(node);
    if (node.properties?.flux3_note) node.widgets?.[0]?.element?.classList.add("secourses-flux3-note");
    if (!node.properties?.flux3_compact_sampler || node.__flux3Controls) return;
    node.__flux3Controls = true;
    const advancedNames = new Set([
        "start_at_step", "end_at_step", "sigma_max", "sigma_min", "rho",
        "add_noise", "return_with_leftover_noise", "tile_sample", "tile_size",
    ]);
    // The seed is connected to the visible shared Seed node in these presets.
    const oldSeedControl = node.widgets.find(w => w.name === "control_after_generate");
    if (oldSeedControl) {
        oldSeedControl.hidden = true;
        oldSeedControl.computeSize = () => [0, -4];
    }
    const advanced = node.widgets.filter(w => advancedNames.has(w.name));
    const sizes = advanced.map(w => w.computeSize);
    let expanded = false;
    const update = () => {
        advanced.forEach((widget, i) => {
            widget.hidden = !expanded;
            widget.computeSize = expanded ? sizes[i] : () => [0, -4];
        });
        button.name = expanded ? "Hide advanced sampler settings" : "Show advanced sampler settings";
        node.setSize([node.size[0], node.computeSize()[1]]);
        node.setDirtyCanvas(true, true);
    };
    const button = node.addWidget("button", "Show advanced sampler settings", null, () => {
        expanded = !expanded;
        update();
    });
    button.serialize = false;
    update();
}

app.registerExtension({
    name: "SECourses.Flux3PresetControls",
    setup() {
        const style = document.createElement("style");
        style.textContent = ".secourses-flux3-note .tiptap, .secourses-flux3-note .tiptap p, .secourses-flux3-note .tiptap li, .secourses-flux3-note .tiptap code {font-size:16px !important;line-height:1.45 !important;} .secourses-flux3-gallery .secourses-refgal-prompt {font-size:17px !important;}";
        document.head.appendChild(style);
        nativeNoteStyle = document.createElement("style");
        document.head.appendChild(nativeNoteStyle);
    },
    loadedGraphNode: configure,
    afterConfigureGraph() {
        // Nodes 2.0 renders MarkdownNote itself rather than reusing its DOM widget.
        const selectors = app.graph._nodes.filter(n => n.properties?.flux3_note)
            .flatMap(n => {
                const base = `[data-node-id="${CSS.escape(String(n.id))}"] .comfy-markdown-content`;
                return [base, `${base} *`];
            });
        const boxes = app.graph._nodes.filter(n => n.properties?.flux3_layout_node)
            .map(n => `[data-node-id="${CSS.escape(String(n.properties.flux3_layout_node))}"] textarea`);
        nativeNoteStyle.textContent = (selectors.length
            ? `${selectors.join(",")} {font-size:16px !important;line-height:1.45 !important;}`
            : "") + (boxes.length
            ? `${boxes.join(",")} {font-size:16px !important;line-height:1.4 !important;}`
            : "");
    },
});
