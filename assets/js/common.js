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
    Toc.helpers.getHeadings = function(el, topLevel) {
      return this.findOrFilter(el, "h1, h2, h3").filter(":not([data-toc-skip])");
    };
    
    // Override the populateNav function to handle proper nesting
    Toc.helpers.populateNav = function(nav, topLevel, headings) {
      var self = this;
      var lastH1Item = null;
      var lastH2Item = null;
      
      headings.each(function(i, heading) {
        var level = self.getNavLevel(heading);
        var navItem = self.generateNavItem(heading);
        
        // Different nesting based on heading level
        if (level === 1) {
          // h1 goes in the main nav
          nav.append(navItem);
          lastH1Item = navItem;
          lastH2Item = null;
        } else if (level === 2) {
          // h2 goes under its parent h1
          if (lastH1Item) {
            var h1ChildNav = lastH1Item.find('> ul.nav');
            if (h1ChildNav.length === 0) {
              h1ChildNav = self.createChildNavList(lastH1Item);
            }
            h1ChildNav.append(navItem);
            lastH2Item = navItem;
          } else {
            // No parent h1, append to main nav
            nav.append(navItem);
            lastH2Item = navItem;
          }
        } else if (level === 3) {
          // h3 goes under its parent h2
          if (lastH2Item) {
            var h2ChildNav = lastH2Item.find('> ul.nav');
            if (h2ChildNav.length === 0) {
              h2ChildNav = self.createChildNavList(lastH2Item);
            }
            h2ChildNav.append(navItem);
          } else {
            // No parent h2, append to main nav or h1 depending on context
            if (lastH1Item) {
              var h1ChildNav = lastH1Item.find('> ul.nav');
              if (h1ChildNav.length === 0) {
                h1ChildNav = self.createChildNavList(lastH1Item);
              }
              h1ChildNav.append(navItem);
            } else {
              nav.append(navItem);
            }
          }
        }
      });
    };
    
    Toc.init($myNav);
    
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
