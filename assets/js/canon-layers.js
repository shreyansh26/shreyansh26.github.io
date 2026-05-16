(function () {
  function byId(id) {
    return document.getElementById(id);
  }

  function setText(id, value) {
    var node = byId(id);
    if (node) node.textContent = value;
  }

  function fmtFixed(value) {
    return value.toFixed(2);
  }

  function fmtInt(value) {
    return value.toLocaleString();
  }

  function updateCanonMixer() {
    var values = [0, 1, 2, 3].map(function (i) {
      var node = byId("canon-x-" + i);
      return node ? Number(node.value) : 0;
    });
    var weights = [0, 1, 2, 3].map(function (i) {
      var node = byId("canon-w-" + i);
      return node ? Number(node.value) : 0;
    });
    var mixed = values.reduce(function (acc, v, i) {
      return acc + v * weights[i];
    }, 0);
    var residual = byId("canon-residual-toggle");
    var output = mixed + (residual && residual.checked ? values[3] : 0);

    values.forEach(function (v, i) {
      setText("canon-x-label-" + i, v.toFixed(2));
    });
    weights.forEach(function (w, i) {
      setText("canon-w-label-" + i, w.toFixed(2));
    });
    var terms = weights.map(function (w, i) {
      return fmtFixed(w) + "*" + fmtFixed(values[i]);
    });
    var residualTerm = residual && residual.checked ? "residual(" + fmtFixed(values[3]) + ")" : "residual(0.00)";

    setText("canon-mix-formula", terms.join(" + ") + " = " + mixed.toFixed(3));
    setText("canon-output-formula", mixed.toFixed(3) + " + " + residualTerm + " = " + output.toFixed(3));
    setText("canon-mixed-out", mixed.toFixed(3));
    setText("canon-output-out", output.toFixed(3));
  }

  function updateCanonCost() {
    var dNode = byId("canon-d-slider");
    var kNode = byId("canon-k-slider");
    var d = dNode ? Number(dNode.value) : 4096;
    var k = kNode ? Number(kNode.value) : 4;
    var depthwise = d * k;
    var full = d * d * k;
    setText("canon-d-label", String(d));
    setText("canon-k-label", String(k));
    setText("canon-depthwise-formula", fmtInt(d) + "*" + fmtInt(k) + " = " + fmtInt(depthwise));
    setText("canon-full-formula", fmtInt(d) + "^2*" + fmtInt(k) + " = " + fmtInt(full));
    setText("canon-ratio-formula", fmtInt(full) + "/" + fmtInt(depthwise) + " = " + fmtInt(full / depthwise) + "x");
    setText("canon-depthwise-params", fmtInt(depthwise));
    setText("canon-full-params", fmtInt(full));
    setText("canon-param-ratio", fmtInt(full / depthwise) + "x");
  }

  function init() {
    var controls = document.querySelectorAll("[data-canon-control]");
    controls.forEach(function (control) {
      control.addEventListener("input", function () {
        updateCanonMixer();
        updateCanonCost();
      });
      control.addEventListener("change", function () {
        updateCanonMixer();
        updateCanonCost();
      });
    });
    updateCanonMixer();
    updateCanonCost();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
