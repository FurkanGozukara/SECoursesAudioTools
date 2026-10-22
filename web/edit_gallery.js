import { app } from "../../../scripts/app.js";

// Image-editing presets opt in with a gallery property. The gallery then takes images only, and its prompt box sends
// the text as typed: the frontend's dynamic-prompt pass would pick {a|b} with Math.random and could change a JSON
// caption, while the presets' prompt nodes resolve choices with the shared seed.
const PROFILES = {
    // Fibo 2.0: SEFibo2GalleryEncode
    fibo2_images_only: {
        hint: "Add, paste, remove or reorder images. One image: edit it. Several: refer to them as @image1, @image2 in the instruction. Empty gallery = text to image.",
        empty: "Optional images to edit: add, drop or paste. Leave empty for text to image.",
    },
    // MOSS-Image Edit: SEMossImageEditReferences
    moss_edit_images: {
        hint: "Add, paste, remove or reorder images. One image: edit it. Several: refer to them as image 1, image 2 in the instruction. The result has the first image's size.",
        empty: "Images to edit: add, drop or paste. The first image sets the result size.",
    },
};

function configureGallery(node) {
    const profile = PROFILES[Object.keys(PROFILES).find(key => node.properties?.[key])];
    if (!profile || !node.__refGallery) return;
    const ui = node.__refGallery;
    if (!ui.editProfileInstalled) {
        ui.editProfileInstalled = true;
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
    ui.addButton.title = profile.hint;
    ui.root.classList.add("secourses-flux3-gallery");
    ui.minContentHeight = () => 360;
    ui.cardsWrap.style.maxHeight = "128px";
    ui.root.style.overflowY = "auto";
    ui.empty.textContent = profile.empty;
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
    name: "SECourses.EditGallery",
    loadedGraphNode: configureGallery,
});
