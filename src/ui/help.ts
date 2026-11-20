/**
 * Small "?" help buttons next to settings whose meaning is not obvious.
 *
 * Each `button.help[data-tip]` gets a hidden description (read by screen readers through
 * aria-describedby) and shows the same text in one shared popover: on hover with a mouse,
 * on keyboard focus, and on tap or click, which pins it until the next tap, Escape or a
 * click elsewhere. The popover is fixed to the viewport so the scrolling panel cannot clip it.
 */

let pop: HTMLDivElement | null = null;
let current: HTMLButtonElement | null = null;
let pinned = false;
let seq = 0;

function popover(): HTMLDivElement {
  if (pop) return pop;
  pop = document.createElement("div");
  pop.className = "help-pop";
  // The text is already the button's description, so the visual copy stays out of the tree.
  pop.setAttribute("aria-hidden", "true");
  pop.hidden = true;
  document.body.append(pop);
  return pop;
}

function place(btn: HTMLButtonElement, el: HTMLDivElement) {
  const r = btn.getBoundingClientRect();
  const vw = document.documentElement.clientWidth;
  const vh = document.documentElement.clientHeight;
  const margin = 12;
  const w = el.offsetWidth;
  const h = el.offsetHeight;
  const left = Math.max(margin, Math.min(r.left - 10, vw - w - margin));
  const below = r.bottom + 8;
  const top = below + h > vh - margin && r.top - 8 - h > margin ? r.top - 8 - h : below;
  el.style.left = `${Math.round(left)}px`;
  el.style.top = `${Math.round(top)}px`;
}

function show(btn: HTMLButtonElement) {
  const el = popover();
  current = btn;
  el.textContent = btn.dataset.tip ?? "";
  el.hidden = false;
  place(btn, el);
}

function hide() {
  if (pop) pop.hidden = true;
  current = null;
  pinned = false;
}

/** Wire every not-yet-initialised help button under `root`. Safe to call after re-renders. */
export function initHelp(root: ParentNode = document) {
  root.querySelectorAll<HTMLButtonElement>("button.help[data-tip]:not([data-ready])").forEach((btn) => {
    btn.dataset.ready = "";
    btn.type = "button";
    const desc = document.createElement("span");
    desc.id = `help-${++seq}`;
    desc.hidden = true;
    desc.textContent = btn.dataset.tip!;
    btn.after(desc);
    btn.setAttribute("aria-describedby", desc.id);

    btn.addEventListener("pointerenter", (e) => {
      if (e.pointerType === "mouse" && !pinned) show(btn);
    });
    btn.addEventListener("pointerleave", (e) => {
      if (e.pointerType === "mouse" && !pinned && current === btn && document.activeElement !== btn) hide();
    });
    btn.addEventListener("focus", () => {
      if (!pinned || current !== btn) {
        pinned = false;
        show(btn);
      }
    });
    btn.addEventListener("blur", () => {
      if (current === btn) hide();
    });
    btn.addEventListener("click", (e) => {
      // A help button can sit inside a summary; it must not open or close the section.
      e.preventDefault();
      e.stopPropagation();
      if (pinned && current === btn) hide();
      else {
        show(btn);
        pinned = true;
      }
    });
  });
}

// One set of document listeners closes the popover from anywhere.
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && current) {
    hide();
    // Close only the popover, not the settings panel that also listens for Escape.
    e.stopPropagation();
  }
});
document.addEventListener("pointerdown", (e) => {
  if (current && !current.contains(e.target as Node)) hide();
});
// The popover follows its button when the panel or page scrolls (focusing a button can scroll
// it into view), and closes once the button scrolls out of sight.
function follow() {
  if (!current || !pop) return;
  const r = current.getBoundingClientRect();
  const box = (current.closest(".panel-body") ?? document.documentElement).getBoundingClientRect();
  const vh = document.documentElement.clientHeight;
  if (r.bottom < Math.max(box.top, 0) || r.top > Math.min(box.bottom, vh) || !current.isConnected) hide();
  else place(current, pop);
}
document.addEventListener("scroll", follow, true);
window.addEventListener("resize", follow);
