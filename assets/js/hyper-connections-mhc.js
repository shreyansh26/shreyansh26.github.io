(function () {
  function setText(root, selector, value) {
    var element = root.querySelector(selector);
    if (element) element.textContent = value;
  }

  function formatGain(value) {
    return value.toLocaleString(undefined, {
      maximumFractionDigits: value > 100 ? 0 : 2,
    });
  }

  function updateGainLab(root) {
    var gainSlider = root.querySelector("#mhc-gain-slider");
    var depthSlider = root.querySelector("#mhc-depth-slider");
    if (!gainSlider || !depthSlider) return;

    var gain = Number(gainSlider.value);
    var depth = Number(depthSlider.value);
    var composite = Math.pow(gain, depth);

    setText(root, "#mhc-gain-label", gain.toFixed(3));
    setText(root, "#mhc-depth-label", String(depth));
    setText(root, "#mhc-gain-out", formatGain(composite) + "x");
    setText(root, "#mhc-log-out", Math.log10(composite).toFixed(2));
  }

  function sinkhorn(matrix, iterations) {
    var m = matrix.map(function (row) {
      return row.slice();
    });

    for (var t = 0; t < iterations; t += 1) {
      for (var col = 0; col < 3; col += 1) {
        var colSum = m[0][col] + m[1][col] + m[2][col];
        for (var row = 0; row < 3; row += 1) m[row][col] /= colSum;
      }

      for (var r = 0; r < 3; r += 1) {
        var rowSum = m[r][0] + m[r][1] + m[r][2];
        for (var c = 0; c < 3; c += 1) m[r][c] /= rowSum;
      }
    }

    return m;
  }

  function updateSinkhornLab(root) {
    var slider = root.querySelector("#mhc-sink-slider");
    var grid = root.querySelector("#mhc-sink-grid");
    if (!slider || !grid) return;

    var iterations = Number(slider.value);
    var base = [
      [2.6, 0.4, 1.2],
      [0.7, 3.1, 0.8],
      [1.4, 0.9, 2.2],
    ];
    var m = sinkhorn(base, iterations);
    var values = [];

    m.forEach(function (row) {
      row.forEach(function (value) {
        values.push(value);
      });
    });

    setText(root, "#mhc-sink-label", String(iterations));
    grid.innerHTML = values
      .map(function (value) {
        return '<div class="mhc-sinkhorn-cell">' + value.toFixed(3) + "</div>";
      })
      .join("");
  }

  function initHyperConnectionsMhc() {
    var root = document.querySelector(".hyper-mhc-post");
    if (!root) return;

    ["#mhc-gain-slider", "#mhc-depth-slider"].forEach(function (selector) {
      var slider = root.querySelector(selector);
      if (!slider) return;
      slider.addEventListener("input", function () {
        updateGainLab(root);
      });
      slider.addEventListener("change", function () {
        updateGainLab(root);
      });
    });

    var sinkSlider = root.querySelector("#mhc-sink-slider");
    if (sinkSlider) {
      sinkSlider.addEventListener("input", function () {
        updateSinkhornLab(root);
      });
      sinkSlider.addEventListener("change", function () {
        updateSinkhornLab(root);
      });
    }

    updateGainLab(root);
    updateSinkhornLab(root);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initHyperConnectionsMhc);
  } else {
    initHyperConnectionsMhc();
  }

  window.addEventListener("load", initHyperConnectionsMhc);
})();
