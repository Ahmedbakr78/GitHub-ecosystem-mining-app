/* GitHub Repository Mining Dashboard */
(async function () {
  'use strict';

  const BASE = '../data/processed/';
  const PAGE_SIZE = 60;
  const COLORS = [
    '#0f766e', '#2563eb', '#b45309', '#7c3aed', '#15803d', '#be123c',
    '#0891b2', '#9333ea', '#ca8a04', '#4f46e5', '#047857', '#c2410c',
    '#0369a1', '#a21caf', '#65a30d', '#9f1239'
  ];

  const $ = (selector) => document.querySelector(selector);
  const fmt = new Intl.NumberFormat('en-US');
  const compact = new Intl.NumberFormat('en-US', { notation: 'compact', maximumFractionDigits: 1 });

  let stats;
  let repos;
  let nodes;
  let edges;
  let rules;
  let similarities;
  let currentPage = 1;
  let currentRows = [];
  let reposByName = new Map();

  try {
    [stats, repos, nodes, edges, rules, similarities] = await Promise.all([
      fetch(BASE + 'stats.json').then(okJson),
      fetch(BASE + 'repos.json').then(okJson),
      fetch(BASE + 'graph_nodes.json').then(okJson),
      fetch(BASE + 'graph_edges.json').then(okJson),
      fetch(BASE + 'association_rules.json').then(okJson),
      fetch(BASE + 'similarities.json').then(okJson),
    ]);
  } catch (error) {
    document.body.innerHTML = `<main class="load-error">
      <h1>Processed data was not found</h1>
      <p>Run <code>python3 scripts/collect_data.py --target 1000 --readme-limit 50</code> and <code>python3 scripts/run_pipeline.py</code>, then refresh the dashboard.</p>
      <pre>${escapeHtml(error.message)}</pre>
    </main>`;
    return;
  }

  repos = repos.map((repo) => ({
    ...repo,
    _search: [
      repo.name,
      repo.language,
      repo.bert_category,
      repo.popularity,
      ...(repo.topics || []),
      ...(repo.tech_stack || []),
      repo.description || '',
    ].join(' ').toLowerCase(),
  }));
  reposByName = new Map(repos.map((repo) => [repo.name, repo]));

  Chart.defaults.color = '#667085';
  Chart.defaults.borderColor = 'rgba(17, 24, 39, 0.08)';
  Chart.defaults.font.family = 'Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';

  renderMeta();
  renderKpis();
  renderInsights();
  renderRubricProof();
  renderFilters();
  renderDatasetCharts();
  renderTopRepos();
  renderLinkAnalysis();
  renderAssociationRules();
  renderSemantic();
  renderRecommendations();
  renderExplorer();
  bindNavigation();

  function okJson(response) {
    if (!response.ok) {
      throw new Error(`${response.url} returned ${response.status}`);
    }
    return response.json();
  }

  function escapeHtml(value) {
    return String(value ?? '').replace(/[&<>"']/g, (char) => ({
      '&': '&amp;',
      '<': '&lt;',
      '>': '&gt;',
      '"': '&quot;',
      "'": '&#039;',
    }[char]));
  }

  function pct(value) {
    return `${Math.round((value || 0) * 100)}%`;
  }

  function entries(obj) {
    return Object.entries(obj || {});
  }

  function renderMeta() {
    const generated = stats.generated_at ? new Date(stats.generated_at).toLocaleString() : 'Not available';
    const sources = entries(stats.data_source_distribution)
      .map(([source, count]) => `${escapeHtml(source)}: ${fmt.format(count)}`)
      .join('<br>');
    $('#pipeline-meta').innerHTML = `
      <div><span>Generated</span><strong>${escapeHtml(generated)}</strong></div>
      <div><span>Runtime</span><strong>${escapeHtml(stats.runtime_seconds || 0)}s</strong></div>
      <div><span>Source</span><strong>${sources || 'Unknown'}</strong></div>
    `;
  }

  function renderKpis() {
    const items = [
      ['Repositories', stats.total_repos, 'Collected and preprocessed'],
      ['Technologies', stats.total_technologies, 'Unique topics and languages'],
      ['Graph Edges', stats.total_edges, 'Weighted co-occurrences'],
      ['Rules', stats.total_rules, 'Apriori associations'],
      ['Communities', stats.total_communities, 'Label propagation groups'],
      ['Avg. BERT Confidence', pct(stats.average_bert_confidence), 'Text classification score'],
    ];
    $('#kpi-grid').innerHTML = items.map(([label, value, detail]) => `
      <article class="kpi">
        <span>${escapeHtml(label)}</span>
        <strong>${typeof value === 'number' ? fmt.format(value) : escapeHtml(value)}</strong>
        <small>${escapeHtml(detail)}</small>
      </article>
    `).join('');
  }

  function renderInsights() {
    const insights = stats.insights || [];
    $('#insight-strip').innerHTML = insights.slice(0, 3).map((item) => `
      <div class="insight-item">${escapeHtml(item)}</div>
    `).join('');
  }

  function renderRubricProof() {
    const items = [
      ['Collection + GUI', `${fmt.format(stats.total_repos)} repos, API collector, large dashboard`],
      ['Preprocessing', 'Cleaned README/description text, normalized topics, engineered tech stacks'],
      ['Association Mining', `${fmt.format(stats.total_rules)} Apriori rules with support, confidence, lift`],
      ['Link Analysis', `${fmt.format(stats.total_technologies)} nodes, PageRank, HITS, ${fmt.format(stats.total_communities)} communities`],
      ['Visualization', 'Charts, D3 network, PCA clusters, repository explorer'],
      ['BERT/Semantic', `${escapeHtml(stats.embedding_method || 'Semantic embeddings')} with recommendations`],
    ];
    $('#rubric-proof').innerHTML = items.map(([title, body]) => `
      <article>
        <strong>${escapeHtml(title)}</strong>
        <span>${body}</span>
      </article>
    `).join('');
  }

  function renderFilters() {
    fillSelect($('#category-filter'), ['All Categories', ...Object.keys(stats.category_distribution || {})]);
    fillSelect($('#language-filter'), ['All Languages', ...Object.keys(stats.language_distribution || {})]);
    fillSelect($('#popularity-filter'), ['All Popularity', 'mega', 'high', 'mid', 'rising']);

    $('#category-filter').addEventListener('change', resetAndRenderExplorer);
    $('#language-filter').addEventListener('change', resetAndRenderExplorer);
    $('#popularity-filter').addEventListener('change', resetAndRenderExplorer);
    $('#global-search').addEventListener('input', debounce(resetAndRenderExplorer, 120));
  }

  function fillSelect(select, values) {
    select.innerHTML = values.map((value) => `<option value="${escapeHtml(value)}">${escapeHtml(value)}</option>`).join('');
  }

  function debounce(fn, delay) {
    let handle;
    return (...args) => {
      clearTimeout(handle);
      handle = setTimeout(() => fn(...args), delay);
    };
  }

  function renderDatasetCharts() {
    barChart('lang-chart', entries(stats.language_distribution).slice(0, 14), 'Repositories', true);
    barChart('category-chart', entries(stats.category_distribution), 'Repositories', true);
    doughnutChart('era-chart', entries(stats.era_distribution), 'Repositories');
    trendChart();
  }

  function barChart(id, data, label, horizontal = false) {
    new Chart($(`#${id}`), {
      type: 'bar',
      data: {
        labels: data.map(([name]) => name),
        datasets: [{
          label,
          data: data.map(([, value]) => value),
          backgroundColor: data.map((_, i) => COLORS[i % COLORS.length]),
          borderRadius: 5,
          borderSkipped: false,
        }],
      },
      options: {
        indexAxis: horizontal ? 'y' : 'x',
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: horizontal
          ? { x: { beginAtZero: true }, y: { grid: { display: false } } }
          : { x: { grid: { display: false } }, y: { beginAtZero: true } },
      },
    });
  }

  function doughnutChart(id, data) {
    new Chart($(`#${id}`), {
      type: 'doughnut',
      data: {
        labels: data.map(([name]) => name),
        datasets: [{
          data: data.map(([, value]) => value),
          backgroundColor: data.map((_, i) => COLORS[i % COLORS.length]),
          borderWidth: 0,
        }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        cutout: '62%',
        plugins: {
          legend: {
            position: 'bottom',
            labels: { boxWidth: 10, boxHeight: 10, padding: 10 },
          },
        },
      },
    });
  }

  function trendChart() {
    const trends = stats.trends || {};
    const years = [...new Set(Object.values(trends).flatMap((item) => Object.keys(item)))].sort();
    const datasets = Object.entries(trends).slice(0, 9).map(([tech, yearCounts], i) => ({
      label: tech,
      data: years.map((year) => yearCounts[year] || 0),
      borderColor: COLORS[i % COLORS.length],
      backgroundColor: COLORS[i % COLORS.length] + '22',
      borderWidth: 2,
      tension: 0.25,
      pointRadius: 2,
    }));
    new Chart($('#trend-chart'), {
      type: 'line',
      data: { labels: years, datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: 'index', intersect: false },
        plugins: { legend: { position: 'bottom', labels: { boxWidth: 10, boxHeight: 10 } } },
        scales: {
          y: { beginAtZero: true },
          x: { grid: { display: false } },
        },
      },
    });
  }

  function renderTopRepos() {
    const top = (stats.top_repositories || repos.slice(0, 20)).slice(0, 12);
    $('#top-repos').innerHTML = top.map((repo, index) => `
      <a class="list-row" href="${escapeHtml(repo.url || repo.html_url || '#')}" target="_blank" rel="noreferrer">
        <span class="rank">${index + 1}</span>
        <span class="list-main">
          <strong>${escapeHtml(repo.name)}</strong>
          <small>${escapeHtml(repo.language)} | ${escapeHtml(repo.category || repo.bert_category || '')}</small>
        </span>
        <span class="metric">${compact.format(repo.stars || 0)}</span>
      </a>
    `).join('');
  }

  function renderLinkAnalysis() {
    const topPr = nodes.slice(0, 18).map((node) => [node.id, node.pagerank]);
    barChart('pagerank-chart', topPr, 'PageRank', true);

    const hitsTop = nodes.slice(0, 30).map((node, i) => ({
      x: node.authority,
      y: node.hub,
      r: Math.max(4, Math.min(18, Math.sqrt(node.count) * 1.4)),
      label: node.id,
      backgroundColor: COLORS[i % COLORS.length] + 'aa',
      borderColor: COLORS[i % COLORS.length],
    }));
    new Chart($('#hits-chart'), {
      type: 'bubble',
      data: {
        datasets: [{
          label: 'Technology',
          data: hitsTop,
          backgroundColor: hitsTop.map((point) => point.backgroundColor),
          borderColor: hitsTop.map((point) => point.borderColor),
          borderWidth: 1,
        }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: (ctx) => `${hitsTop[ctx.dataIndex].label}: authority ${ctx.raw.x.toFixed(4)}, hub ${ctx.raw.y.toFixed(4)}`,
            },
          },
        },
        scales: {
          x: { title: { display: true, text: 'Authority' } },
          y: { title: { display: true, text: 'Hub' } },
        },
      },
    });

    $('#tech-table tbody').innerHTML = nodes.slice(0, 45).map((node, index) => `
      <tr>
        <td>${index + 1}</td>
        <td><span class="strong">${escapeHtml(node.id)}</span><br><small>${escapeHtml(node.tribe || '')}</small></td>
        <td>${node.pagerank.toFixed(5)}</td>
        <td>${node.authority.toFixed(5)}</td>
        <td>${node.hub.toFixed(5)}</td>
        <td>${fmt.format(node.count)}</td>
      </tr>
    `).join('');

    buildNetwork();
  }

  function renderAssociationRules() {
    new Chart($('#rules-bubble'), {
      type: 'bubble',
      data: {
        datasets: [{
          label: 'Rules',
          data: rules.slice(0, 120).map((rule) => ({
            x: rule.support,
            y: rule.confidence,
            r: Math.max(4, Math.min(24, rule.lift * 2.2)),
          })),
          backgroundColor: rules.slice(0, 120).map((_, i) => COLORS[i % COLORS.length] + '99'),
          borderColor: rules.slice(0, 120).map((_, i) => COLORS[i % COLORS.length]),
          borderWidth: 1,
        }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: (ctx) => {
                const rule = rules[ctx.dataIndex];
                return `${rule.antecedent.join(' + ')} -> ${rule.consequent.join(' + ')} | lift ${rule.lift}`;
              },
            },
          },
        },
        scales: {
          x: { title: { display: true, text: 'Support' } },
          y: { title: { display: true, text: 'Confidence' }, min: 0 },
        },
      },
    });

    $('#rules-table tbody').innerHTML = rules.slice(0, 60).map((rule) => `
      <tr>
        <td>${tags(rule.antecedent, 'blue')}</td>
        <td>${tags(rule.consequent, 'amber')}</td>
        <td>${rule.support}<br><small>${fmt.format(rule.support_count || 0)} repos</small></td>
        <td>${rule.confidence}</td>
        <td><span class="score-pill">${rule.lift}</span></td>
      </tr>
    `).join('');
  }

  function renderSemantic() {
    const buckets = new Array(10).fill(0);
    repos.forEach((repo) => {
      const index = Math.max(0, Math.min(9, Math.floor((repo.bert_confidence || 0) * 10)));
      buckets[index]++;
    });
    new Chart($('#confidence-chart'), {
      type: 'bar',
      data: {
        labels: buckets.map((_, i) => `${i * 10}-${i * 10 + 9}%`),
        datasets: [{
          label: 'Repositories',
          data: buckets,
          backgroundColor: COLORS[0],
          borderRadius: 5,
        }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: { y: { beginAtZero: true }, x: { grid: { display: false } } },
      },
    });

    barChart('cluster-chart', entries(stats.cluster_distribution).sort((a, b) => Number(a[0]) - Number(b[0])), 'Repositories');

    const frequent = stats.frequent_itemsets || {};
    $('#method-summary').innerHTML = `
      <div class="method-line"><span>Embedding</span><strong>${escapeHtml(stats.embedding_method || 'Unknown')}</strong></div>
      <div class="method-line"><span>Categories</span><strong>${fmt.format(stats.total_categories || 0)}</strong></div>
      <div class="method-line"><span>Average confidence</span><strong>${pct(stats.average_bert_confidence)}</strong></div>
      <div class="method-line"><span>Min support count</span><strong>${fmt.format(frequent.min_support_count || 0)}</strong></div>
      <div class="method-note">Install sentence-transformers and rerun the pipeline to compute true transformer embeddings when a local model or internet access is available.</div>
    `;

    buildPcaScatter();
  }

  function renderRecommendations() {
    const list = $('#repo-list');
    list.innerHTML = repos.slice(0, 8000).map((repo) => `<option value="${escapeHtml(repo.name)}"></option>`).join('');

    const defaultRepo = repos[0] ? repos[0].name : '';
    $('#repo-picker').value = defaultRepo;
    drawRecommendations(defaultRepo);
    $('#repo-picker').addEventListener('input', (event) => drawRecommendations(event.target.value));
  }

  function drawRecommendations(name) {
    const host = $('#rec-results');
    const repo = reposByName.get(name);
    if (!repo) {
      host.innerHTML = '<div class="empty-state">Select a repository from the dataset.</div>';
      return;
    }

    const sims = similarities[name] || [];
    const selected = `
      <article class="selected-repo">
        <small>Selected repository</small>
        <h3>${escapeHtml(repo.name)}</h3>
        <p>${escapeHtml(repo.description || 'No description available.')}</p>
        <div>${tags([repo.language, repo.bert_category, repo.popularity], 'green')}</div>
      </article>
    `;
    const cards = sims.map(([similarName, score]) => {
      const similar = reposByName.get(similarName) || {};
      return `
        <article class="rec-card">
          <div>
            <a href="${escapeHtml(similar.html_url || '#')}" target="_blank" rel="noreferrer">${escapeHtml(similarName)}</a>
            <p>${escapeHtml(similar.description || '')}</p>
          </div>
          <div class="rec-foot">
            <span>${pct(score)}</span>
            <div class="bar"><i style="width:${Math.round(score * 100)}%"></i></div>
          </div>
        </article>
      `;
    }).join('');
    host.innerHTML = selected + cards;
  }

  function renderExplorer() {
    $('#prev-page').addEventListener('click', () => {
      currentPage = Math.max(1, currentPage - 1);
      renderRepoTable();
    });
    $('#next-page').addEventListener('click', () => {
      const totalPages = Math.max(1, Math.ceil(currentRows.length / PAGE_SIZE));
      currentPage = Math.min(totalPages, currentPage + 1);
      renderRepoTable();
    });
    resetAndRenderExplorer();
  }

  function resetAndRenderExplorer() {
    currentPage = 1;
    currentRows = filteredRepos();
    renderRepoTable();
  }

  function filteredRepos() {
    const category = $('#category-filter').value;
    const language = $('#language-filter').value;
    const popularity = $('#popularity-filter').value;
    const query = $('#global-search').value.trim().toLowerCase();

    return repos.filter((repo) => {
      if (category !== 'All Categories' && repo.bert_category !== category) return false;
      if (language !== 'All Languages' && repo.language !== language) return false;
      if (popularity !== 'All Popularity' && repo.popularity !== popularity) return false;
      if (query && !repo._search.includes(query)) return false;
      return true;
    });
  }

  function renderRepoTable() {
    const totalPages = Math.max(1, Math.ceil(currentRows.length / PAGE_SIZE));
    currentPage = Math.max(1, Math.min(currentPage, totalPages));
    const start = (currentPage - 1) * PAGE_SIZE;
    const pageRows = currentRows.slice(start, start + PAGE_SIZE);

    $('#repo-table tbody').innerHTML = pageRows.map((repo) => `
      <tr>
        <td>
          <a class="repo-link" href="${escapeHtml(repo.html_url || '#')}" target="_blank" rel="noreferrer">${escapeHtml(repo.name)}</a>
          <small>${escapeHtml(repo.description || '')}</small>
        </td>
        <td>${compact.format(repo.stars || 0)}</td>
        <td>${compact.format(repo.forks || 0)}</td>
        <td>${tags([repo.language], 'green')}</td>
        <td>${tags([repo.bert_category], 'blue')}</td>
        <td>${pct(repo.bert_confidence)}</td>
        <td>${tags((repo.tech_stack || []).slice(0, 5), 'muted')}</td>
      </tr>
    `).join('');

    $('#result-count').textContent = `${fmt.format(currentRows.length)} repositories`;
    $('#page-label').textContent = `${currentPage} / ${totalPages}`;
    $('#prev-page').disabled = currentPage <= 1;
    $('#next-page').disabled = currentPage >= totalPages;
  }

  function tags(values, tone = 'muted') {
    return (values || []).filter(Boolean).map((value) => (
      `<span class="tag ${tone}">${escapeHtml(value)}</span>`
    )).join('');
  }

  function buildNetwork() {
    const container = $('#network-graph');
    container.innerHTML = '';
    const width = Math.max(720, container.clientWidth);
    const height = 520;
    const graphNodes = nodes.slice(0, 115).map((node) => ({ ...node }));
    const ids = new Set(graphNodes.map((node) => node.id));
    const graphEdges = edges
      .filter((edge) => ids.has(edge.source) && ids.has(edge.target))
      .slice(0, 320)
      .map((edge) => ({ ...edge }));

    const svg = d3.select(container).append('svg')
      .attr('viewBox', `0 0 ${width} ${height}`)
      .attr('role', 'img');
    const group = svg.append('g');
    svg.call(d3.zoom().scaleExtent([0.35, 4]).on('zoom', (event) => group.attr('transform', event.transform)));

    const communityScale = d3.scaleOrdinal()
      .domain([...new Set(graphNodes.map((node) => node.community))])
      .range(COLORS);

    const simulation = d3.forceSimulation(graphNodes)
      .force('link', d3.forceLink(graphEdges).id((node) => node.id).distance(78).strength(0.35))
      .force('charge', d3.forceManyBody().strength(-170))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force('collision', d3.forceCollide().radius((node) => 7 + Math.sqrt(node.count)));

    const link = group.selectAll('line')
      .data(graphEdges)
      .join('line')
      .attr('stroke', '#cbd5e1')
      .attr('stroke-opacity', 0.35)
      .attr('stroke-width', (edge) => Math.max(1, Math.min(6, Math.sqrt(edge.weight))));

    const node = group.selectAll('circle')
      .data(graphNodes)
      .join('circle')
      .attr('r', (item) => Math.max(5, Math.min(24, Math.sqrt(item.count) * 1.2)))
      .attr('fill', (item) => communityScale(item.community))
      .attr('stroke', '#ffffff')
      .attr('stroke-width', 1)
      .style('cursor', 'grab')
      .call(d3.drag()
        .on('start', (event, item) => {
          if (!event.active) simulation.alphaTarget(0.3).restart();
          item.fx = item.x;
          item.fy = item.y;
        })
        .on('drag', (event, item) => {
          item.fx = event.x;
          item.fy = event.y;
        })
        .on('end', (event, item) => {
          if (!event.active) simulation.alphaTarget(0);
          item.fx = null;
          item.fy = null;
        }));

    node.append('title').text((item) => `${item.id}\nPageRank: ${item.pagerank}\nRepos: ${item.count}\n${item.tribe}`);

    const label = group.selectAll('text')
      .data(graphNodes.filter((item) => item.pagerank >= nodes[35]?.pagerank || item.count > 300))
      .join('text')
      .text((item) => item.id)
      .attr('font-size', 10)
      .attr('font-weight', 700)
      .attr('fill', '#1f2937')
      .attr('text-anchor', 'middle')
      .attr('pointer-events', 'none');

    simulation.on('tick', () => {
      link
        .attr('x1', (item) => item.source.x)
        .attr('y1', (item) => item.source.y)
        .attr('x2', (item) => item.target.x)
        .attr('y2', (item) => item.target.y);
      node
        .attr('cx', (item) => item.x)
        .attr('cy', (item) => item.y);
      label
        .attr('x', (item) => item.x)
        .attr('y', (item) => item.y - Math.max(10, Math.sqrt(item.count)));
    });
  }

  function buildPcaScatter() {
    const container = $('#pca-scatter');
    container.innerHTML = '';
    const width = Math.max(760, container.clientWidth);
    const height = 520;
    const maxPoints = 4200;
    const step = Math.max(1, Math.ceil(repos.length / maxPoints));
    const points = repos.filter((_, index) => index % step === 0);

    const svg = d3.select(container).append('svg')
      .attr('viewBox', `0 0 ${width} ${height}`)
      .attr('role', 'img');
    const group = svg.append('g');
    svg.call(d3.zoom().scaleExtent([0.6, 12]).on('zoom', (event) => group.attr('transform', event.transform)));

    const xExtent = d3.extent(points, (item) => item.pca_x);
    const yExtent = d3.extent(points, (item) => item.pca_y);
    const xPad = Math.max(0.01, (xExtent[1] - xExtent[0]) * 0.08);
    const yPad = Math.max(0.01, (yExtent[1] - yExtent[0]) * 0.08);
    const x = d3.scaleLinear().domain([xExtent[0] - xPad, xExtent[1] + xPad]).range([44, width - 28]);
    const y = d3.scaleLinear().domain([yExtent[0] - yPad, yExtent[1] + yPad]).range([height - 36, 24]);
    const color = d3.scaleOrdinal().domain([...new Set(points.map((item) => item.cluster_id))]).range(COLORS);

    group.append('g').attr('transform', `translate(0,${height - 36})`).call(d3.axisBottom(x).ticks(8));
    group.append('g').attr('transform', 'translate(44,0)').call(d3.axisLeft(y).ticks(6));

    group.selectAll('circle')
      .data(points)
      .join('circle')
      .attr('cx', (item) => x(item.pca_x))
      .attr('cy', (item) => y(item.pca_y))
      .attr('r', 3.2)
      .attr('fill', (item) => color(item.cluster_id))
      .attr('opacity', 0.62)
      .append('title')
      .text((item) => `${item.name}\n${item.bert_category}\nCluster ${item.cluster_id}\nStars ${fmt.format(item.stars)}`);
  }

  function bindNavigation() {
    const links = document.querySelectorAll('.nav-links a');
    const observer = new IntersectionObserver((sections) => {
      sections.forEach((section) => {
        if (section.isIntersecting) {
          links.forEach((link) => link.classList.toggle('active', link.getAttribute('href') === `#${section.target.id}`));
        }
      });
    }, { rootMargin: '-35% 0px -55% 0px' });
    document.querySelectorAll('main section[id]').forEach((section) => observer.observe(section));
  }
})();
