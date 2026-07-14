(() => {
  const { categories, articles } = window.RADIOTOE_CONTENT;
  const app = document.querySelector("#app");
  const sidebar = document.querySelector("#sidebar");
  const grid = document.querySelector(".page-grid");
  const scrim = document.querySelector("#scrim");
  let filter = "all";

  const categoryOf = (article) => categories.find((item) => item.id === article.category);
  const currentSlug = () => new URLSearchParams(location.search).get("article");

  function go(slug) {
    history.pushState({}, "", slug ? `?article=${slug}` : location.pathname);
    render();
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  function renderSidebar() {
    const active = currentSlug();
    sidebar.innerHTML = `
      <div class="sidebar-mobile-head"><span>文章目录</span><button id="sidebar-close">×</button></div>
      <p class="eyebrow">Knowledge index</p>
      <h2>大模型原理</h2>
      <p class="sidebar-intro">从矩阵运算到对齐训练，沿着模型的内部路径逐层拆解。</p>
      <nav class="category-nav">
        ${categories.map((category, index) => {
          const items = articles.filter((article) => article.category === category.id);
          return `<section class="category open">
            <button class="category-title"><span><b>0${index + 1}</b>${category.name}</span><span>−</span></button>
            <div class="category-items">${items.map((article) => `
              <button data-article="${article.slug}" class="${active === article.slug ? "active" : ""}">
                <span>${article.title}</span>${article.status === "planned" ? "<em>计划中</em>" : ""}
              </button>`).join("")}
            </div>
          </section>`;
        }).join("")}
      </nav>`;
    sidebar.querySelectorAll("[data-article]").forEach((button) => button.addEventListener("click", () => {
      go(button.dataset.article);
      closeMobileSidebar();
    }));
    sidebar.querySelectorAll(".category-title").forEach((button) => button.addEventListener("click", () => {
      const section = button.closest(".category");
      section.classList.toggle("open");
      button.lastElementChild.textContent = section.classList.contains("open") ? "−" : "+";
    }));
    sidebar.querySelector("#sidebar-close").addEventListener("click", closeMobileSidebar);
  }

  function articleCards(items) {
    return items.map((article, index) => {
      const category = categoryOf(article);
      return `<article class="article-card"><button data-article="${article.slug}">
        <span class="article-number">${String(index + 1).padStart(2, "0")}</span>
        <span class="article-card-body">
          <span class="article-meta"><i></i>${category.name}<time>${article.date}</time></span>
          <strong>${article.title}</strong>
          <span class="summary">${article.summary}</span>
          <span class="tags">${article.tags.map((tag) => `#${tag}`).join("　")}</span>
        </span><span class="arrow">↗</span>
      </button></article>`;
    }).join("");
  }

  function homePage() {
    const visible = filter === "all" ? articles : articles.filter((article) => article.category === filter);
    app.innerHTML = `
      <section class="hero">
        <div class="hero-kicker">FIELD NOTES <i></i></div>
        <h1>拆开模型，<br><em>看见智能的结构。</em></h1>
        <p>radiotoe 是一份关于大语言模型底层原理的个人研究笔记。记录理解，保留推导，也标注仍未想清楚的问题。</p>
        <div class="hero-foot"><span>从底层开始</span><span>持续更新中</span></div>
      </section>
      <section class="article-section">
        <div class="section-heading"><div><p class="eyebrow">All notes</p><h2>文章与计划</h2></div>
          <div class="filters"><button data-filter="all">全部</button>${categories.map((item) => `<button data-filter="${item.id}">${item.name}</button>`).join("")}</div>
        </div>
        <div class="article-list">${articleCards(visible)}</div>
      </section>
      <section class="about" id="about"><p class="eyebrow">About radiotoe</p><div><h2>不是答案库，<br>是一张持续展开的地图。</h2><p>这里会整理语言模型的基础结构、训练方法、推理机制与对齐技术。文章尽量从问题本身出发，保留公式、直觉和工程视角之间的联系。</p></div></section>`;
    app.querySelectorAll("[data-filter]").forEach((button) => {
      if (button.dataset.filter === filter) button.classList.add("active");
      button.addEventListener("click", () => { filter = button.dataset.filter; homePage(); });
    });
    app.querySelectorAll("[data-article]").forEach((button) => button.addEventListener("click", () => go(button.dataset.article)));
  }

  function articlePage(article) {
    const index = articles.findIndex((item) => item.slug === article.slug);
    const previous = articles[index - 1];
    const next = articles[index + 1];
    const category = categoryOf(article);
    const planned = `
      <div class="draft-notice"><small>NOTE 00</small><div><h2>这篇笔记正在整理中</h2><p>补充正文后，首页、分类与左侧目录会自动同步。</p></div></div>
      <div class="article-body"><p class="lead">这篇文章将从一个具体问题开始，逐步连接公式、直觉与工程实现。</p><h2>计划包含</h2><ul>${article.outline.map((item) => `<li>${item}</li>`).join("")}</ul></div>`;
    app.innerHTML = `
      <article class="article-page">
        <button class="back" data-home>← 返回全部文章</button>
        <header class="article-header"><div class="article-meta"><i></i>${category.name}<time>${article.date}</time></div><h1>${article.title}</h1><p>${article.summary}</p></header>
        <div class="article-body longform">${article.body || ""}</div>
        ${article.body ? "" : planned}
        <footer>radiotoe / ${category.name}</footer>
        <nav class="post-navigation">
          ${previous ? `<button data-article="${previous.slug}"><small>← 上一篇</small><span>${previous.title}</span></button>` : "<span></span>"}
          ${next ? `<button data-article="${next.slug}" class="next"><small>下一篇 →</small><span>${next.title}</span></button>` : "<span></span>"}
        </nav>
      </article>`;
    app.querySelector("[data-home]").addEventListener("click", () => go(null));
    app.querySelectorAll("[data-article]").forEach((button) => button.addEventListener("click", () => go(button.dataset.article)));
  }

  function render() {
    renderSidebar();
    const article = articles.find((item) => item.slug === currentSlug());
    article ? articlePage(article) : homePage();
  }

  function closeMobileSidebar() {
    sidebar.classList.remove("mobile-open");
    scrim.classList.remove("visible");
  }

  document.querySelectorAll("[data-home]").forEach((button) => button.addEventListener("click", () => go(null)));
  document.querySelector("#sidebar-toggle").addEventListener("click", (event) => {
    grid.classList.toggle("sidebar-collapsed");
    const collapsed = grid.classList.contains("sidebar-collapsed");
    localStorage.setItem("radiotoe-sidebar", collapsed ? "collapsed" : "expanded");
    event.currentTarget.setAttribute("aria-label", collapsed ? "展开文章目录" : "收起文章目录");
  });
  document.querySelector("#mobile-menu").addEventListener("click", () => {
    sidebar.classList.add("mobile-open");
    scrim.classList.add("visible");
  });
  scrim.addEventListener("click", closeMobileSidebar);

  const savedTheme = localStorage.getItem("radiotoe-theme") || "ivory";
  document.documentElement.dataset.theme = savedTheme;
  document.querySelectorAll(".theme-switcher button").forEach((button) => {
    if (button.dataset.theme === savedTheme) button.classList.add("active");
    button.addEventListener("click", () => {
      document.documentElement.dataset.theme = button.dataset.theme;
      localStorage.setItem("radiotoe-theme", button.dataset.theme);
      document.querySelectorAll(".theme-switcher button").forEach((item) => item.classList.toggle("active", item === button));
    });
  });
  if (localStorage.getItem("radiotoe-sidebar") === "collapsed") grid.classList.add("sidebar-collapsed");

  const dialog = document.querySelector("#search-dialog");
  const searchInput = document.querySelector("#search-input");
  const searchResults = document.querySelector("#search-results");
  function runSearch() {
    const query = searchInput.value.trim().toLowerCase();
    const results = articles.filter((article) => !query || `${article.title} ${article.summary} ${article.tags.join(" ")}`.toLowerCase().includes(query));
    searchResults.innerHTML = `<p>${query ? `找到 ${results.length} 篇` : "所有文章"}</p>${results.map((article) => `<button data-article="${article.slug}"><span>${article.title}</span><small>${categoryOf(article).name}</small></button>`).join("")}`;
    searchResults.querySelectorAll("[data-article]").forEach((button) => button.addEventListener("click", () => { go(button.dataset.article); dialog.hidden = true; }));
  }
  document.querySelector("#search-open").addEventListener("click", () => { dialog.hidden = false; runSearch(); searchInput.focus(); });
  dialog.addEventListener("click", (event) => { if (event.target === dialog) dialog.hidden = true; });
  searchInput.addEventListener("input", runSearch);
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") dialog.hidden = true;
    if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") { event.preventDefault(); dialog.hidden = false; runSearch(); searchInput.focus(); }
  });
  addEventListener("popstate", render);
  render();
})();
