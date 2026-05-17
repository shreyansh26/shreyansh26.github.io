(function () {
  function initEngramLayers() {
    var root = document.querySelector(".engram-post");
    if (!root) return;

    var videoSourceUrl = "https://www.youtube.com/watch?v=87Q8nf1XHKA";

    if ("IntersectionObserver" in window) {
      var revealObserver = new IntersectionObserver(
        function (entries) {
          entries.forEach(function (entry) {
            if (entry.isIntersecting) entry.target.classList.add("is-visible");
          });
        },
        { threshold: 0.12 }
      );
      root.querySelectorAll(".reveal").forEach(function (element) {
        revealObserver.observe(element);
      });
    } else {
      root.querySelectorAll(".reveal").forEach(function (element) {
        element.classList.add("is-visible");
      });
    }

    root.querySelectorAll(".asset-figure figcaption").forEach(function (caption) {
      if (caption.querySelector(".source-credit")) return;
      var source = document.createElement("span");
      source.className = "source-credit";
      source.innerHTML = 'Source: <a href="' + videoSourceUrl + '">Engram video by Jia-Bin Huang</a>.';
      caption.appendChild(source);
    });

    var lightbox = root.querySelector("#engram-lightbox");
    if (lightbox) {
      var lightboxImg = lightbox.querySelector("img");
      var lightboxClose = lightbox.querySelector("#engram-lightbox-close");
      var closeLightbox = function () {
        lightbox.classList.remove("is-open");
        if (lightboxImg) lightboxImg.removeAttribute("src");
      };

      root.querySelectorAll("[data-lightbox]").forEach(function (img) {
        img.addEventListener("click", function () {
          if (!lightboxImg) return;
          lightboxImg.src = img.currentSrc || img.src;
          lightboxImg.alt = img.alt || "";
          lightbox.classList.add("is-open");
        });
      });

      if (lightboxClose) {
        lightboxClose.addEventListener("click", closeLightbox);
      }

      lightbox.addEventListener("click", function (event) {
        if (event.target === lightbox) closeLightbox();
      });

      window.addEventListener("keydown", function (event) {
        if (event.key === "Escape") closeLightbox();
      });
    }

    var tokenColors = ["#5a3c57", "#8f562c", "#1c6f75", "#527c44", "#315b85"];
    var hashTokens = root.querySelector("#hashTokens");
    var slotList = root.querySelector("#slotList");
    var buttons = root.querySelectorAll("#phraseButtons button");

    function simpleHash(str, seed) {
      var h = 2166136261 ^ seed;
      for (var i = 0; i < str.length; i += 1) {
        h ^= str.charCodeAt(i);
        h = Math.imul(h, 16777619);
        h ^= h >>> 13;
      }
      return Math.abs(h >>> 0);
    }

    function renderHashLab(phrase) {
      if (!hashTokens || !slotList) return;

      var tokens = phrase.split(" ");
      hashTokens.innerHTML = "";
      tokens.forEach(function (token, index) {
        var chip = document.createElement("span");
        chip.className = "token";
        chip.textContent = token;
        chip.style.background = tokenColors[index % tokenColors.length];
        hashTokens.appendChild(chip);
      });

      slotList.innerHTML = "";
      for (var head = 1; head <= 8; head += 1) {
        var slot = simpleHash(phrase, head * 10007) % 9973;
        var width = 12 + (slot % 88);
        var row = document.createElement("div");
        row.className = "slot-line";
        row.innerHTML = "<span>head " + head + '</span><span class="slot-bar"><i style="--w:' + width + '%"></i></span><span>' + slot + "</span>";
        slotList.appendChild(row);
      }
    }

    buttons.forEach(function (button) {
      button.addEventListener("click", function () {
        buttons.forEach(function (item) {
          item.classList.remove("is-active");
        });
        button.classList.add("is-active");
        renderHashLab(button.dataset.phrase);
      });
    });
    renderHashLab("Harry Potter");

    var rhoRange = root.querySelector("#rhoRange");
    var rhoValue = root.querySelector("#rhoValue");
    var moeBar = root.querySelector("#moeBar");
    var engramBar = root.querySelector("#engramBar");
    var moeShare = root.querySelector("#moeShare");
    var engramShare = root.querySelector("#engramShare");
    var lossValue = root.querySelector("#lossValue");

    function updateAllocation() {
      if (!rhoRange || !rhoValue || !moeBar || !engramBar || !moeShare || !engramShare || !lossValue) {
        return;
      }

      var rho = Number(rhoRange.value);
      var engram = 100 - rho;
      var toyLoss = 1.7109 + Math.pow((rho - 80) / 100, 2) * 0.35;
      rhoValue.textContent = (rho / 100).toFixed(2);
      moeBar.style.width = rho + "%";
      engramBar.style.width = engram + "%";
      moeShare.textContent = rho + "%";
      engramShare.textContent = engram + "%";
      lossValue.textContent = toyLoss.toFixed(3);
      moeBar.textContent = rho > 14 ? "MoE" : "";
      engramBar.textContent = engram > 14 ? "Engram" : "";
    }

    if (rhoRange) {
      rhoRange.addEventListener("input", updateAllocation);
      rhoRange.addEventListener("change", updateAllocation);
    }
    updateAllocation();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initEngramLayers);
  } else {
    initEngramLayers();
  }

  window.addEventListener("load", initEngramLayers);
})();
