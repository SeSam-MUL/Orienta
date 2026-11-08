// Which calibration pattern the PC Refinement page is showing, and which late
// answers may still be drawn on it.
//
// Every picture, CI and set of Kikuchi lines on the page comes from a request
// that takes a while (Hough indexing). The user can pick another pattern, or
// change the PC again, before it returns. An answer that arrives after that
// describes a pattern (or a PC) that is no longer on screen, and drawing it puts
// the lines of one pattern on the image of another.
//
//   select(idx)      the selection changed: every answer asked before is void;
//                    returns the token of the new selection
//   isCurrent(token) an answer asked under `token` may still be shown
//   isSelected(idx)  an answer that belongs to calibration pattern `idx` may be
//                    shown (nothing is shown for a pattern that is not selected)
//   selected()       the selected pattern, or null
export function createSelectionGuard() {
  let token = 0;
  let selected = null;
  return {
    select(idx) {
      token += 1;
      selected = idx == null ? null : idx;
      return token;
    },
    isCurrent(t) {
      return t === token;
    },
    isSelected(idx) {
      return idx != null && selected === idx;
    },
    selected() {
      return selected;
    },
  };
}

// Answers to "re-index the selected pattern at this PC / tilt" come back in the
// order the server finishes them, not the order they were asked. Only the newest
// request may draw; `next()` starts one, `isLatest(seq)` asks whether it still is.
export function createLatestRequest() {
  let seq = 0;
  return {
    next() {
      seq += 1;
      return seq;
    },
    isLatest(s) {
      return s === seq;
    },
  };
}
