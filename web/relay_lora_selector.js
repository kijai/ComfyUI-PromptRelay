import { app } from "../../scripts/app.js";
const WH = 28; // LiteGraph.NODE_WIDGET_HEIGHT
const MARGIN = 10;
const INNER = 3;

// Track the last adjusted canvas mouse event (mirrors rgthree's approach).
let _lastCanvasMouseEvent = null;
const _origAdjustMouseEvent = LGraphCanvas.prototype.adjustMouseEvent;
LGraphCanvas.prototype.adjustMouseEvent = function (e) {
  _origAdjustMouseEvent.apply(this, arguments);
  _lastCanvasMouseEvent = e;
};

// ---------------------------------------------------------------------------
// Drawing helpers
// ---------------------------------------------------------------------------

function fitString(ctx, str, maxW) {
  if (ctx.measureText(str).width <= maxW) return str;
  const ell = "\u2026";
  let i = str.length;
  while (i > 0 && ctx.measureText(str.substring(0, i) + ell).width > maxW) i--;
  return i > 0 ? str.substring(0, i) + ell : ell;
}

function lowQ() {
  return (app.canvas.ds?.scale || 1) <= 0.5;
}

function drawToggle(ctx, x, y, h, value) {
  const lq = lowQ();
  const tw = h * 1.5;
  if (!lq) {
    ctx.beginPath();
    ctx.roundRect(x + 4, y + 4, tw - 8, h - 8, [h * 0.5]);
    ctx.globalAlpha = app.canvas.editor_alpha * 0.25;
    ctx.fillStyle = "rgba(255,255,255,0.45)";
    ctx.fill();
    ctx.globalAlpha = app.canvas.editor_alpha;
  }
  ctx.fillStyle = value ? "#89B" : "#888";
  const cx = lq || !value ? x + h * 0.5 : x + h;
  if (!lq) {
    ctx.beginPath();
    ctx.arc(cx, y + h * 0.5, h * 0.36, 0, Math.PI * 2);
    ctx.fill();
  }
  return tw;
}

function drawStrength(ctx, x, y, h, value) {
  // Right-aligned [< val >]
  const aW = 9, gap = 3, nW = 32;
  const totalW = aW * 2 + gap * 2 + nW;
  const left = x - totalW;
  const midY = y + h * 0.5;
  let px = left;

  ctx.save();
  ctx.fillStyle = LiteGraph.WIDGET_TEXT_COLOR;
  // < arrow
  ctx.fill(new Path2D(`M${px} ${midY} l${aW} 5 l0 -10z`));
  px += aW + gap;
  // value
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(value.toFixed(2), px + nW * 0.5, midY);
  px += nW + gap;
  // > arrow
  ctx.fill(new Path2D(`M${px} ${midY - 5} l${aW} 5 l-${aW} 5 v-10z`));
  ctx.restore();

  return { left, totalW };
}

function drawRowBg(ctx, x, y, w, h) {
  ctx.fillStyle = LiteGraph.WIDGET_BGCOLOR;
  ctx.beginPath();
  ctx.roundRect(x, y, w, h, lowQ() ? 0 : h * 0.5);
  ctx.fill();
  if (!lowQ()) {
    ctx.strokeStyle = LiteGraph.WIDGET_OUTLINE_COLOR;
    ctx.stroke();
  }
}

function drawAddBtn(ctx, x, y, w, h, over) {
  ctx.fillStyle = over ? "#444" : LiteGraph.WIDGET_BGCOLOR;
  ctx.beginPath();
  ctx.roundRect(x, y, w, h, 4);
  ctx.fill();
  ctx.strokeStyle = LiteGraph.WIDGET_OUTLINE_COLOR;
  ctx.stroke();
  if (!lowQ()) {
    ctx.textBaseline = "middle";
    ctx.textAlign = "center";
    ctx.fillStyle = LiteGraph.WIDGET_TEXT_COLOR;
    ctx.font = "13px sans-serif";
    ctx.fillText("+ Add LoRA", x + w * 0.5, y + h * 0.5);
  }
}

// ---------------------------------------------------------------------------
// Lora list (fetched once from the API)
// ---------------------------------------------------------------------------

let _loras = null;

async function fetchLoras() {
  if (_loras) return _loras;
  try {
    const r = await fetch("/object_info/LoraLoader");
    const d = await r.json();
    _loras = d?.LoraLoader?.input?.required?.lora_name?.[0] ?? [];
    return _loras;
  } catch (e) {
    console.error("[RelayLoraSelector] Failed to fetch lora list", e);
    return [];
  }
}

function pickLora(event, cb) {
  fetchLoras().then((list) => {
    new LiteGraph.ContextMenu(list, {
      event,
      title: "Choose a LoRA",
      scale: Math.max(1, app.canvas.ds?.scale || 1),
      className: "dark",
      callback: cb,
    });
  });
}

// ---------------------------------------------------------------------------
// Widgets
// ---------------------------------------------------------------------------

/**
 * One compact row: [toggle] [lora-name …] [strength arrows]
 * Serialized as { on, lora, strength }.
 */
class LoraEntryWidget {
  constructor(name, value) {
    this.name = name;
    this.type = "custom";
    this.options = {};
    this.value = value || { on: true, lora: null, strength: 1.0 };
    this.last_y = 0;
    this._mdPos = null;
    this._dragging = false;
    this._dragStartVal = 0;
    this._b = {}; // hit-area bounds
  }

  draw(ctx, node, w, posY, h) {
    this.last_y = posY;
    if (lowQ()) return;

    ctx.save();
    drawRowBg(ctx, MARGIN, posY, w - MARGIN * 2, h);

    let px = MARGIN + INNER;

    // Toggle
    px += drawToggle(ctx, px, posY, h, this.value.on) + INNER;
    this._b.toggle = { x: MARGIN + INNER, y: posY, w: h * 1.5, h };

    if (!this.value.on) ctx.globalAlpha = app.canvas.editor_alpha * 0.4;

    // Strength (right-aligned)
    const s = drawStrength(ctx, w - MARGIN - INNER, posY, h, this.value.strength);
    this._b.strFull = { x: s.left, y: posY, w: s.totalW, h };
    this._b.strDec = { x: s.left, y: posY, w: 9, h };
    this._b.strInc = { x: s.left + s.totalW - 9, y: posY, w: 9, h };
    this._b.strVal = { x: s.left + 12, y: posY, w: 32, h };

    // Lora name
    const loraEnd = s.left - INNER * 2;
    this._b.lora = { x: px, y: posY, w: loraEnd - px, h };
    ctx.fillStyle = LiteGraph.WIDGET_TEXT_COLOR;
    ctx.textAlign = "left";
    ctx.textBaseline = "middle";
    ctx.fillText(fitString(ctx, this.value.lora || "None", this._b.lora.w), px, posY + h * 0.5);

    ctx.restore();
  }

  mouse(event, pos, node) {
    const hit = (b) => b && pos[0] >= b.x && pos[0] < b.x + b.w && pos[1] >= b.y && pos[1] < b.y + b.h;

    if (event.type === "pointerdown") {
      this._mdPos = [...pos];
      this._dragging = false;

      if (hit(this._b.toggle)) {
        this.value.on = !this.value.on;
        node.setDirtyCanvas(true);
        return true;
      }
      if (hit(this._b.lora)) {
        pickLora(event, (v) => {
          if (v) { this.value.lora = v; node.setDirtyCanvas(true); }
        });
        return true;
      }
      if (hit(this._b.strDec)) {
        this.value.strength = Math.round((this.value.strength - 0.05) * 100) / 100;
        node.setDirtyCanvas(true);
        return true;
      }
      if (hit(this._b.strInc)) {
        this.value.strength = Math.round((this.value.strength + 0.05) * 100) / 100;
        node.setDirtyCanvas(true);
        return true;
      }
      if (hit(this._b.strFull)) {
        this._dragStartVal = this.value.strength;
        return true;
      }
      return false;
    }

    if (event.type === "pointermove" && this._mdPos) {
      if (hit(this._b.strFull)) {
        this._dragging = true;
        if (event.deltaX) {
          this.value.strength = Math.round((this._dragStartVal + event.deltaX * 0.01) * 100) / 100;
          node.setDirtyCanvas(true);
        }
      }
      return true;
    }

    if (event.type === "pointerup") {
      const wasDown = !!this._mdPos;
      this._mdPos = null;
      if (wasDown && !this._dragging && hit(this._b.strVal)) {
        app.canvas.prompt("Strength", this.value.strength, (v) => {
          this.value.strength = Math.round(Number(v) * 100) / 100;
          node.setDirtyCanvas(true);
        }, event);
      }
      this._dragging = false;
      return wasDown;
    }
    return false;
  }

  serializeValue() {
    return { ...this.value };
  }
}

/**
 * Header with "Toggle All" toggle (not serialized).
 * Drawn only when there is at least one lora widget.
 */
class HeaderWidget {
  constructor() {
    this.name = "header";
    this.type = "custom";
    this.value = {};
    this.options = { serialize: false };
    this.last_y = 0;
    this._b = {};
  }

  draw(ctx, node, w, posY, h) {
    this.last_y = posY;
    const hasLoras = node.widgets.some((x) => x.name?.startsWith("lora_"));
    if (!hasLoras) return;

    ctx.save();
    const midY = posY + h * 0.5;
    let px = MARGIN + INNER;

    // Toggle All
    const allOn = allLorasState(node);
    this._b.toggle = { x: px, y: posY, w: h * 1.5, h };
    px += drawToggle(ctx, px, posY, h, allOn) + INNER;

    // Label
    ctx.globalAlpha = app.canvas.editor_alpha * 0.55;
    ctx.fillStyle = LiteGraph.WIDGET_TEXT_COLOR;
    ctx.textAlign = "left";
    ctx.textBaseline = "middle";
    ctx.fillText("Toggle All", px, midY);

    // Strength label (right-aligned)
    ctx.textAlign = "center";
    ctx.fillText("Strength", w - MARGIN - INNER - 28, midY);

    ctx.restore();
  }

  mouse(event, pos, node) {
    if (event.type === "pointerdown") {
      const b = this._b.toggle;
      if (b && pos[0] >= b.x && pos[0] < b.x + b.w &&
          pos[1] >= b.y && pos[1] < b.y + b.h) {
        toggleAllLoras(node);
        return true;
      }
    }
    return false;
  }
}

/** "+ Add LoRA" button (not serialized). */
class AddButtonWidget {
  constructor(onClick) {
    this.name = "add_lora_btn";
    this.type = "custom";
    this.value = "";
    this.options = { serialize: false };
    this.last_y = 0;
    this._over = false;
    this._cb = onClick;
    this._lastEvent = null;
  }

  draw(ctx, node, w, posY, h) {
    this.last_y = posY;
    if (lowQ()) return;
    drawAddBtn(ctx, 15, posY, w - 30, h, this._over);
  }

  mouse(event, pos, node) {
    if (event.type === "pointerdown") {
      if (pos[0] >= 15 && pos[0] <= node.size[0] - 15 &&
          pos[1] >= this.last_y && pos[1] < this.last_y + WH) {
        this._over = true;
        this._lastEvent = event;
        return true;
      }
    }
    if (event.type === "pointerup" && this._over) {
      this._over = false;
      this._cb?.(this._lastEvent);
      this._lastEvent = null;
      return true;
    }
    return false;
  }
}

/** Thin horizontal divider (not serialized). */
class DividerWidget {
  constructor(top = 2) {
    this.name = "divider";
    this.type = "custom";
    this.value = {};
    this.options = { serialize: false };
    this._top = top;
  }
  draw(ctx, node, w, posY) {
    ctx.strokeStyle = LiteGraph.WIDGET_OUTLINE_COLOR;
    ctx.beginPath();
    ctx.moveTo(15, posY + this._top);
    ctx.lineTo(w - 15, posY + this._top);
    ctx.stroke();
  }
  computeSize() {
    return [0, this._top * 2 + 1];
  }
}

// ---------------------------------------------------------------------------
// Node helpers
// ---------------------------------------------------------------------------

let _counter = 0;

function allLorasState(node) {
  let allOn = true;
  let allOff = true;
  let any = false;
  for (const w of node.widgets) {
    if (w.name?.startsWith("lora_")) {
      any = true;
      if (w.value?.on !== true) allOn = false;
      if (w.value?.on !== false) allOff = false;
      if (!allOn && !allOff) return null;
    }
  }
  return any ? allOn : null;
}

function toggleAllLoras(node) {
  const currentState = allLorasState(node);
  const to = !currentState ? true : false;
  for (const w of node.widgets) {
    if (w.name?.startsWith("lora_") && w.value) w.value.on = to;
  }
}

function addLora(node, event) {
  fetchLoras().then((list) => {
    new LiteGraph.ContextMenu(list, {
      event,
      title: "Choose a LoRA",
      scale: Math.max(1, app.canvas.ds?.scale || 1),
      className: "dark",
      callback: (v) => {
        if (!v) return;
        _counter++;
        const w = new LoraEntryWidget("lora_" + _counter, { on: true, lora: v, strength: 1.0 });
        const btnIdx = node.widgets.findIndex((x) => x.name === "add_lora_btn");
        node.widgets.splice(btnIdx >= 0 ? btnIdx : node.widgets.length, 0, w);
        resizeNode(node);
      },
    });
  });
}

function resizeNode(node) {
  const c = node.computeSize();
  node.size[0] = Math.max(node.size[0], c[0]);
  node.size[1] = c[1];
  node.setDirtyCanvas(true);
}

function initNode(node) {
  node.serialize_widgets = true;
  node.widgets = [
    new DividerWidget(2),
    new HeaderWidget(),
    new AddButtonWidget((event) => addLora(node, event)),
  ];
  resizeNode(node);
}

function loadNode(node, values) {
  node.widgets = [new DividerWidget(2), new HeaderWidget()];
  for (const v of values || []) {
    if (v && typeof v === "object" && "lora" in v) {
      _counter++;
      node.widgets.push(new LoraEntryWidget("lora_" + _counter, v));
    }
  }
  node.widgets.push(new AddButtonWidget((event) => addLora(node, event)));
  resizeNode(node);
}

// ---------------------------------------------------------------------------
// Right-click context menu on lora rows
// ---------------------------------------------------------------------------

function patchMenu(node) {
  const origGetSlotInPosition = node.getSlotInPosition?.bind(node);
  const origGetSlotMenuOptions = node.getSlotMenuOptions?.bind(node);

  // Mirror rgthree's getSlotInPosition: return a synthetic slot with
  // output.type so ComfyUI's processContextMenu doesn't crash.
  node.getSlotInPosition = function (canvasX, canvasY) {
    const slot = origGetSlotInPosition?.(canvasX, canvasY);
    if (!slot) {
      for (const w of node.widgets) {
        if (w.name?.startsWith("lora_") && w.last_y &&
            canvasY > node.pos[1] + w.last_y &&
            canvasY < node.pos[1] + w.last_y + WH) {
          return { widget: w, output: { type: "LORA_WIDGET" } };
        }
      }
    }
    return slot;
  };

  node.getSlotMenuOptions = function (slot) {
    if (slot?.widget?.name?.startsWith("lora_")) {
      const w = slot.widget;
      const i = node.widgets.indexOf(w);
      const prev = node.widgets[i - 1];
      const next = node.widgets[i + 1];
      new LiteGraph.ContextMenu([
        { content: w.value.on ? "Turn Off" : "Turn On", callback: () => { w.value.on = !w.value.on; } },
        { content: "Move Up",   disabled: !prev?.name?.startsWith("lora_"), callback: () => { node.widgets.splice(i, 1); node.widgets.splice(i - 1, 0, w); } },
        { content: "Move Down", disabled: !next?.name?.startsWith("lora_"), callback: () => { node.widgets.splice(i, 1); node.widgets.splice(i + 1, 0, w); } },
        { content: "Remove",    callback: () => { node.widgets.splice(i, 1); resizeNode(node); } },
      ], { event: _lastCanvasMouseEvent });
      return undefined;
    }
    return origGetSlotMenuOptions?.(slot);
  };
}

// ---------------------------------------------------------------------------
// Extension registration
// ---------------------------------------------------------------------------

app.registerExtension({
  name: "ComfyUI-PromptRelay.RelayLoraSelector",

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "RelayLoraSelector") return;

    const origCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = origCreated?.call(this);
      initNode(this);
      patchMenu(this);
      return r;
    };

    const origCfg = nodeType.prototype.configure;
    nodeType.prototype.configure = function (info) {
      origCfg?.call(this, info);
      loadNode(this, info.widgets_values);
    };
  },
});
