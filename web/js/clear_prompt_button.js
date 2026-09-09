// PromptRelay — adds a one-click "Clear" button to prompt/text nodes, so a pre-filled
// prompt (e.g. the default baked into the Z-Image Turbo workflow) can be wiped in one
// click instead of holding Backspace.
const { app } = window.comfyAPI.app;

// Node types that carry an editable prompt/text field.
const TEXT_NODE_TYPES = ["PrimitiveStringMultiline", "CLIPTextEncode"];
const BTN_LABEL = "🗑 Clear"; // 🗑 Clear

function clearTextWidgets(node) {
  let cleared = false;
  for (const w of node.widgets || []) {
    if (w.name === BTN_LABEL) continue; // never clear the button itself
    // Multiline text widgets are "customtext" (textarea in w.inputEl); plain "text".
    const isText =
      w.type === "customtext" ||
      w.type === "text" ||
      ["value", "text", "string"].includes(w.name);
    if (isText && typeof w.value === "string") {
      w.value = "";
      if (w.inputEl) w.inputEl.value = "";
      if (typeof w.callback === "function") {
        try { w.callback("", app.canvas, node); } catch (e) {}
      }
      cleared = true;
    }
  }
  if (cleared) node.setDirtyCanvas(true, true);
}

app.registerExtension({
  name: "PromptRelay.ClearPromptButton",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (!TEXT_NODE_TYPES.includes(nodeData.name)) return;
    const orig = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = orig ? orig.apply(this, arguments) : undefined;
      if (!(this.widgets || []).some((w) => w.name === BTN_LABEL)) {
        this.addWidget("button", BTN_LABEL, null, () => clearTextWidgets(this));
      }
      return r;
    };
  },
});
