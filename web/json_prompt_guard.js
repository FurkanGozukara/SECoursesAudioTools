import { app } from "../../../scripts/app.js";

// ComfyUI's dynamic prompts expand {a|b} and strip // comments in every prompt box when a workflow is queued.
// That destroys JSON prompts: MOSS-Image layouts, structured captions and FLUX.3 trailing tables.
// Keep a complete JSON object/array exactly as typed; for "caption + trailing JSON", expand only the caption.

function parsesAsJson(text) {
    try {
        const value = JSON.parse(text);
        return value !== null && typeof value === "object";
    } catch {
        return false;
    }
}

function trailingJsonStart(text) {
    const trimmed = text.trimEnd();
    if (!trimmed.endsWith("}") && !trimmed.endsWith("]")) return -1;
    for (let i = 0; i < trimmed.length; i++) {
        const c = trimmed[i];
        if ((c === "{" || c === "[") && parsesAsJson(trimmed.slice(i))) return i;
    }
    return -1;
}

function guardWidget(widget) {
    if (!widget?.dynamicPrompts || widget.__secoursesJsonGuard || typeof widget.serializeValue !== "function") return;
    widget.__secoursesJsonGuard = true;
    const original = widget.serializeValue;
    // Frontends and other node packs chain serializeValue as async functions; always await the original.
    widget.serializeValue = async function (workflowNode, index) {
        const value = widget.value;
        if (typeof value !== "string") return await original.apply(this, arguments);
        const start = trailingJsonStart(value);
        if (start < 0) return await original.apply(this, arguments);
        let result = value;
        if (start > 0) {
            // Expand only the caption with the frontend's own dynamic prompt implementation.
            widget.value = value.slice(0, start);
            try {
                const caption = await original.apply(this, arguments);
                result = (typeof caption === "string" ? caption : value.slice(0, start)) + value.slice(start);
            } finally {
                widget.value = value;
            }
        }
        if (workflowNode?.widgets_values) workflowNode.widgets_values[index] = result;
        return result;
    };
}

function guardNode(node) {
    for (const widget of node?.widgets ?? []) guardWidget(widget);
}

app.registerExtension({
    name: "SECourses.JsonPromptGuard",
    nodeCreated: guardNode,
    loadedGraphNode: guardNode,
});
