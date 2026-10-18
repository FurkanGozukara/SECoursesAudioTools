import { app } from "../../../scripts/app.js";

// MOSS-Image presets opt in with the gallery property moss_text_only. The prompt box then sends its text
// as typed, like the FLUX.3 profile: the frontend's dynamic-prompt pass would pick {a|b} with Math.random,
// while SEMossImageGalleryPrompt resolves choices with the shared seed and keeps JSON layouts intact.
function configure(node) {
    if (!node.properties?.moss_text_only || !node.__refGallery) return;
    const ui = node.__refGallery;
    ui.promptWidget.serializeValue = () => ui.promptWidget.value;
}

app.registerExtension({
    name: "SECourses.MossImageGallery",
    loadedGraphNode: configure,
});
