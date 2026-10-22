import { app } from "../../../scripts/app.js";

const NO_IMAGE = "(none - disabled)";

app.registerExtension({
    name: "SECourses.Fibo2",
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
