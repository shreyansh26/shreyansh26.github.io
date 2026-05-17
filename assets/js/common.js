$(document).ready(function () {
  // add toggle functionality to abstract, award and bibtex buttons
  $("a.abstract").click(function () {
    $(this).parent().parent().find(".abstract.hidden").toggleClass("open");
    $(this).parent().parent().find(".award.hidden.open").toggleClass("open");
    $(this).parent().parent().find(".bibtex.hidden.open").toggleClass("open");
  });
  $("a.award").click(function () {
    $(this).parent().parent().find(".abstract.hidden.open").toggleClass("open");
    $(this).parent().parent().find(".award.hidden").toggleClass("open");
    $(this).parent().parent().find(".bibtex.hidden.open").toggleClass("open");
  });
  $("a.bibtex").click(function () {
    $(this).parent().parent().find(".abstract.hidden.open").toggleClass("open");
    $(this).parent().parent().find(".award.hidden.open").toggleClass("open");
    $(this).parent().parent().find(".bibtex.hidden").toggleClass("open");
  });
  $("a").removeClass("waves-effect waves-light");

  // bootstrap-toc
  if ($("#toc-sidebar").length) {
    // remove related publications years from the TOC
    $(".publications h2").each(function () {
      $(this).attr("data-toc-skip", "");
    });

    var navSelector = "#toc-sidebar";
    var $myNav = $(navSelector);

    // Save original functions
    var originalGetHeadings = Toc.helpers.getHeadings;
    var originalPopulateNav = Toc.helpers.populateNav;

    // Override the getHeadings function
    Toc.helpers.getHeadings = function (el, topLevel) {
      return this.findOrFilter(el, "h2, h3, h4").filter(":not([data-toc-skip])");
    };

    // Override the populateNav function so blog section headings are the top level.
    Toc.helpers.populateNav = function (nav, topLevel, headings) {
      var self = this;
      var lastH2Item = null;
      var lastH3Item = null;

      headings.each(function (i, heading) {
        var level = self.getNavLevel(heading);
        var navItem = self.generateNavItem(heading);

        if (level === 2) {
          nav.append(navItem);
          lastH2Item = navItem;
          lastH3Item = null;
        } else if (level === 3) {
          if (lastH2Item) {
            var h2ChildNav = lastH2Item.find("> ul.nav");
            if (h2ChildNav.length === 0) {
              h2ChildNav = self.createChildNavList(lastH2Item);
            }
            h2ChildNav.append(navItem);
            lastH3Item = navItem;
          } else {
            nav.append(navItem);
            lastH3Item = navItem;
          }
        } else if (level === 4) {
          if (lastH3Item) {
            var h3ChildNav = lastH3Item.find("> ul.nav");
            if (h3ChildNav.length === 0) {
              h3ChildNav = self.createChildNavList(lastH3Item);
            }
            h3ChildNav.append(navItem);
          } else if (lastH2Item) {
            var h2FallbackChildNav = lastH2Item.find("> ul.nav");
            if (h2FallbackChildNav.length === 0) {
              h2FallbackChildNav = self.createChildNavList(lastH2Item);
            }
            h2FallbackChildNav.append(navItem);
          } else {
            nav.append(navItem);
          }
        }
      });
    };

    Toc.init($myNav);
    initCollapsibleToc(navSelector);

    $("body").scrollspy({
      target: navSelector,
    });
  }

  // add css to jupyter notebooks
  const cssLink = document.createElement("link");
  cssLink.href = "../css/jupyter.css";
  cssLink.rel = "stylesheet";
  cssLink.type = "text/css";

  let jupyterTheme = determineComputedTheme();

  $(".jupyter-notebook-iframe-container iframe").each(function () {
    $(this).contents().find("head").append(cssLink);

    if (jupyterTheme == "dark") {
      $(this).bind("load", function () {
        $(this).contents().find("body").attr({
          "data-jp-theme-light": "false",
          "data-jp-theme-name": "JupyterLab Dark",
        });
      });
    }
  });

  // trigger popovers
  $('[data-toggle="popover"]').popover({
    trigger: "hover",
  });
});

function initCollapsibleToc(navSelector) {
  var toc = document.querySelector(navSelector);
  if (!toc) return;

  var rootList = toc.querySelector(":scope > ul.nav");
  if (!rootList) return;

  var topItems = Array.prototype.slice.call(rootList.children).filter(function (item) {
    return item.matches("li");
  });
  if (!topItems.length) return;

  var sectionLinks = Array.prototype.slice
    .call(toc.querySelectorAll('a[href^="#"]'))
    .map(function (link) {
      var rawId = link.getAttribute("href").slice(1);
      var id = rawId;
      try {
        id = decodeURIComponent(rawId);
      } catch (error) {
        id = rawId;
      }
      return {
        link: link,
        heading: document.getElementById(id),
        topItem: getTopLevelItem(link, rootList),
      };
    })
    .filter(function (entry) {
      return entry.heading && entry.topItem;
    });

  if (!sectionLinks.length) return;

  var activeTopItem = null;
  var reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  toc.classList.add("toc-auto-collapse");

  topItems.forEach(function (item) {
    var childNav = item.querySelector(":scope > ul.nav");
    if (!childNav) return;

    item.classList.add("toc-has-children");
    var link = item.querySelector(":scope > a");
    if (link) link.setAttribute("aria-expanded", "false");
  });

  function setChildNavHeight(item, expand, animate) {
    var childNav = item.querySelector(":scope > ul.nav");
    if (!childNav) return;

    item.classList.toggle("toc-expanded", expand);
    item.classList.toggle("toc-collapsed", !expand);

    var link = item.querySelector(":scope > a");
    if (link) link.setAttribute("aria-expanded", expand ? "true" : "false");

    if (!animate || reduceMotion) {
      childNav.style.height = expand ? "auto" : "0px";
      return;
    }

    var currentHeight = childNav.getBoundingClientRect().height;
    var targetHeight = expand ? childNav.scrollHeight : 0;

    if (Math.abs(currentHeight - targetHeight) < 1) {
      childNav.style.height = expand ? "auto" : "0px";
      return;
    }

    childNav.style.height = currentHeight + "px";
    childNav.offsetHeight;
    childNav.style.height = targetHeight + "px";

    var onTransitionEnd = function (event) {
      if (event.propertyName !== "height") return;
      childNav.removeEventListener("transitionend", onTransitionEnd);
      if (item.classList.contains("toc-expanded")) {
        childNav.style.height = "auto";
      }
    };

    childNav.addEventListener("transitionend", onTransitionEnd);
  }

  function setActiveTopItem(nextTopItem, animate) {
    if (!nextTopItem || activeTopItem === nextTopItem) return;
    activeTopItem = nextTopItem;

    topItems.forEach(function (item) {
      setChildNavHeight(item, item === nextTopItem, animate);
    });
  }

  function getActiveTopItemFromScroll() {
    var activeEntry = sectionLinks[0];

    sectionLinks.forEach(function (entry) {
      if (entry.heading.getBoundingClientRect().top <= 80) {
        activeEntry = entry;
      }
    });

    return activeEntry.topItem;
  }

  var ticking = false;
  function requestScrollUpdate() {
    if (ticking) return;

    ticking = true;
    window.requestAnimationFrame(function () {
      setActiveTopItem(getActiveTopItemFromScroll(), true);
      ticking = false;
    });
  }

  toc.addEventListener("click", function (event) {
    var link = event.target && event.target.closest ? event.target.closest('a[href^="#"]') : null;
    if (!link || !toc.contains(link)) return;

    var topItem = getTopLevelItem(link, rootList);
    if (topItem) setActiveTopItem(topItem, true);
  });

  window.addEventListener("scroll", requestScrollUpdate, { passive: true });
  window.addEventListener("resize", function () {
    setActiveTopItem(getActiveTopItemFromScroll(), false);
  });

  setActiveTopItem(getActiveTopItemFromScroll(), false);
}

function getTopLevelItem(element, rootList) {
  var item = element.closest("li");

  while (item && item.parentElement !== rootList) {
    item = item.parentElement ? item.parentElement.closest("li") : null;
  }

  return item && item.parentElement === rootList ? item : null;
}
