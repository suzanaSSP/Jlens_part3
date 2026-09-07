"""
Build a self-contained HTML ground-truth token visualizer.

Reads all per-category ground truth JSONs from data/ground_truths/,
decodes token IDs via the Qwen tokenizer (Qwen/Qwen2.5-32B-Instruct by
default), and writes a single fully self-contained HTML file to
visualizations/gt_token_viz.html.

Features:
  - Word-cloud, bar-chart, and table views
  - Side-by-side category comparison with shared-token highlighting
  - Top-N token count selector
  - Search / filter tokens

Usage (from project root):
    python -m scripts.build_gt_token_viz
    python -m scripts.build_gt_token_viz --top 100 --out visualizations/gt_token_viz.html
    python -m scripts.build_gt_token_viz --model Qwen/Qwen2.5-32B-Instruct
"""

import argparse
import glob
import json
import os

CATEGORY_COLORS = {
    "Religion & Theology":                  "#2a78d6",
    "Law, Politics, Government & History":  "#eb6834",
    "Technology & Definitions":             "#1baf7a",
    "Applied Ethics & Moral Dilemmas":      "#eda100",
    "Marriage & Romantic Partnerships":     "#e87ba4",
    "Human Nature & Philosophy":            "#3db85e",
    "Science, Physics & Math":              "#a47be8",
    "Depression, Addiction & Feeling Lost": "#e34948",
    "Family Duty & Obligations":            "#c9be27",
    "Grief, Loss & Death":                  "#2ec4c4",
}

# Files in data/ground_truths/ that are NOT per-category distributions
SKIP_FILES = {"ground_truth_pool.json", "ground_truths.json"}


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_categories(gt_dir):
    cats = []
    for path in sorted(glob.glob(os.path.join(gt_dir, "*.json"))):
        if os.path.basename(path) in SKIP_FILES:
            continue
        with open(path) as f:
            data = json.load(f)
        if "distribution" not in data:
            continue
        cats.append(data)
    return cats


def decode_category(cat_data, tokenizer, top_n):
    rows = []
    for tok_id_str, prob in cat_data["distribution"].items():
        if tok_id_str == "__residual__":
            continue
        token_text = tokenizer.decode([int(tok_id_str)])
        rows.append({"token": token_text, "probability": prob, "token_id": int(tok_id_str)})
    rows.sort(key=lambda x: x["probability"], reverse=True)
    for i, r in enumerate(rows):
        r["rank"] = i + 1
    return rows[:top_n]


def build_data(categories, tokenizer, top_n):
    result = []
    for cat in categories:
        name = cat["category"]
        tokens = decode_category(cat, tokenizer, top_n)
        total_mass = sum(t["probability"] for t in tokens)
        result.append({
            "name": name,
            "color": CATEGORY_COLORS.get(name, "#888888"),
            "layer": cat.get("layer", 62),
            "segment": cat.get("segment", "thought"),
            "n_questions": cat.get("n_questions", "?"),
            "total_tokens_in_dist": len([k for k in cat["distribution"] if k != "__residual__"]),
            "top_n_mass": round(total_mass, 4),
            "tokens": tokens,
        })
    result.sort(key=lambda x: x["name"])
    return result


# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------

HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>J-Lens Ground Truth Token Explorer</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}

:root{
  --bg:#0b0d12;
  --surface:#111420;
  --surface-2:#181c2a;
  --surface-3:#20263a;
  --border:rgba(255,255,255,0.07);
  --border-2:rgba(255,255,255,0.13);
  --text:#dde2f0;
  --text-2:#8890a8;
  --text-3:#4e5570;
  --radius:8px;
  --trans:180ms ease;
}

html,body{height:100%;overflow:hidden}
body{background:var(--bg);color:var(--text);font-family:'Inter',sans-serif;font-size:13px}

/* ---- Layout ---- */
.app{display:flex;flex-direction:column;height:100vh}

.top-bar{
  display:flex;align-items:center;gap:16px;
  padding:0 24px;height:52px;flex-shrink:0;
  background:var(--surface);border-bottom:1px solid var(--border);
}
.logo{font-weight:700;font-size:14px;letter-spacing:-.3px}
.logo span{opacity:.45;font-weight:400;margin:0 6px}
.subtitle{color:var(--text-3);font-size:11.5px}
.top-spacer{flex:1}

.layout{display:flex;flex:1;overflow:hidden}

/* ---- Sidebar ---- */
.sidebar{
  width:252px;flex-shrink:0;
  background:var(--surface);border-right:1px solid var(--border);
  display:flex;flex-direction:column;overflow:hidden;
}
.sidebar-inner{flex:1;overflow-y:auto;padding:14px 0 4px}
.sidebar-label{
  padding:0 16px 8px;
  font-size:10px;font-weight:600;letter-spacing:.8px;
  text-transform:uppercase;color:var(--text-3);
}
.cat-list{list-style:none}
.cat-item{
  display:flex;align-items:flex-start;gap:10px;
  padding:8px 16px;cursor:pointer;
  transition:background var(--trans);
  border-left:2px solid transparent;
}
.cat-item:hover{background:rgba(255,255,255,0.04)}
.cat-item.active{
  background:rgba(255,255,255,0.07);
  border-left-color:var(--cat-color,#888);
}
.cat-dot{
  width:8px;height:8px;border-radius:50%;flex-shrink:0;margin-top:3px;
}
.cat-name{font-size:12px;line-height:1.4;color:var(--text-2)}
.cat-item.active .cat-name{color:var(--text)}

.sidebar-footer{
  padding:12px 16px;border-top:1px solid var(--border);flex-shrink:0;
}
.compare-btn{
  width:100%;padding:8px 12px;
  border:1px solid var(--border-2);border-radius:6px;
  background:transparent;color:var(--text-2);
  font-size:12px;font-family:'Inter',sans-serif;
  cursor:pointer;transition:all var(--trans);text-align:left;
  display:flex;align-items:center;gap:8px;
}
.compare-btn:hover{border-color:rgba(255,255,255,0.22);color:var(--text)}
.compare-btn.active{
  background:rgba(255,255,255,0.07);
  border-color:rgba(255,255,255,0.2);color:var(--text);
}

/* ---- Main content ---- */
.content{flex:1;display:flex;overflow:hidden}

/* ---- Panel ---- */
.panel{flex:1;display:flex;flex-direction:column;overflow:hidden}

.panel-header{
  padding:18px 28px 14px;
  border-bottom:1px solid var(--border);flex-shrink:0;
}
.panel-title{font-size:17px;font-weight:600;margin-bottom:8px;letter-spacing:-.2px}
.panel-meta{display:flex;gap:6px;flex-wrap:wrap;align-items:center}
.meta-badge{
  padding:2px 9px;border-radius:999px;
  border:1px solid var(--border-2);
  font-size:10.5px;color:var(--text-2);
  white-space:nowrap;
}
.meta-badge.coverage{color:var(--text)}

/* ---- Controls ---- */
.controls{
  display:flex;align-items:center;gap:16px;flex-wrap:wrap;
  padding:10px 28px;
  border-bottom:1px solid var(--border);flex-shrink:0;
}
.view-tabs{
  display:flex;border:1px solid var(--border-2);
  border-radius:6px;overflow:hidden;
}
.view-tab{
  padding:5px 14px;background:transparent;border:none;
  color:var(--text-2);font-size:12px;font-family:'Inter',sans-serif;
  cursor:pointer;transition:all var(--trans);white-space:nowrap;
}
.view-tab:hover:not(.active){background:rgba(255,255,255,0.05)}
.view-tab.active{background:rgba(255,255,255,0.1);color:var(--text)}

.ctrl-group{display:flex;align-items:center;gap:7px;font-size:12px;color:var(--text-2)}
.ctrl-select{
  background:var(--surface-2);border:1px solid var(--border-2);
  color:var(--text);font-size:12px;font-family:'Inter',sans-serif;
  padding:4px 8px;border-radius:5px;
}
.search-input{
  background:var(--surface-2);border:1px solid var(--border-2);
  color:var(--text);font-size:12px;font-family:'Inter',sans-serif;
  padding:4px 10px;border-radius:5px;width:160px;outline:none;
  transition:border-color var(--trans);
}
.search-input:focus{border-color:rgba(255,255,255,0.3)}
.search-input::placeholder{color:var(--text-3)}

/* ---- Viz area ---- */
.viz-area{flex:1;overflow-y:auto}

/* ---- Cloud ---- */
.cloud{
  display:flex;flex-wrap:wrap;align-items:center;
  gap:8px;padding:28px 32px;
}
.cloud-token{
  display:inline-block;border-radius:999px;
  padding:.2em .75em;line-height:1.4;
  cursor:default;border:1px solid transparent;
  transition:transform var(--trans),opacity var(--trans),box-shadow var(--trans);
  font-family:'Inter',sans-serif;
  user-select:none;
}
.cloud-token:hover{transform:scale(1.08);box-shadow:0 3px 14px rgba(0,0,0,.4)}
.cloud-token.dimmed{opacity:.18!important}
.cloud-token.shared{outline:2px solid rgba(255,255,255,.35);outline-offset:1px}

/* ---- Bars ---- */
.bars{padding:14px 28px;display:flex;flex-direction:column;gap:4px}
.bar-row{display:flex;align-items:center;gap:10px}
.bar-rank{width:26px;text-align:right;font-size:10px;color:var(--text-3);flex-shrink:0}
.bar-token{
  width:175px;flex-shrink:0;font-family:'JetBrains Mono',monospace;
  font-size:11.5px;color:var(--text);white-space:pre;
  overflow:hidden;text-overflow:ellipsis;
}
.bar-token.dimmed{opacity:.2}
.bar-track{
  flex:1;height:16px;background:rgba(255,255,255,0.05);
  border-radius:3px;overflow:hidden;position:relative;min-width:60px;
}
.bar-fill{height:100%;border-radius:3px;transition:width 350ms ease}
.bar-prob{
  width:110px;text-align:right;font-size:10.5px;
  color:var(--text-2);font-family:'JetBrains Mono',monospace;flex-shrink:0;
}

/* ---- Table ---- */
.table-wrap{padding:14px 28px}
.gt-table{width:100%;border-collapse:collapse;font-size:12.5px}
.gt-table th{
  padding:7px 12px;text-align:left;font-weight:500;
  color:var(--text-3);border-bottom:1px solid var(--border-2);
  font-size:10px;text-transform:uppercase;letter-spacing:.6px;
}
.gt-table td{padding:5px 12px;border-bottom:1px solid rgba(255,255,255,0.04)}
.gt-table tr:hover td{background:rgba(255,255,255,0.03)}
.gt-table tr.dimmed td{opacity:.2}
.token-pill{
  display:inline-block;padding:1px 9px;border-radius:999px;
  font-family:'JetBrains Mono',monospace;font-size:11px;
}
.mini-bar-wrap{width:120px}
.mini-bar{height:5px;border-radius:3px}

/* ---- Compare ---- */
.compare-layout{display:flex;flex:1;overflow:hidden}
.compare-panel{
  flex:1;display:flex;flex-direction:column;overflow:hidden;
  border-right:1px solid var(--border);
}
.compare-panel:last-child{border-right:none}
.compare-selector{
  display:flex;align-items:center;gap:8px;
  padding:10px 18px;border-bottom:1px solid var(--border);flex-shrink:0;
}
.compare-select{
  flex:1;background:var(--surface-2);border:1px solid var(--border-2);
  color:var(--text);font-size:12px;font-family:'Inter',sans-serif;
  padding:5px 9px;border-radius:5px;
}
.shared-legend{
  display:flex;align-items:center;gap:7px;
  padding:6px 18px;font-size:11px;color:var(--text-3);
  border-bottom:1px solid var(--border);flex-shrink:0;
}
.shared-swatch{
  width:14px;height:14px;border-radius:50%;
  outline:2px solid rgba(255,255,255,.35);outline-offset:1px;
  background:rgba(255,255,255,.1);flex-shrink:0;
}

/* ---- Tooltip ---- */
#tooltip{
  position:fixed;background:#1a1e2e;border:1px solid var(--border-2);
  border-radius:6px;padding:6px 11px;font-size:11px;
  pointer-events:none;z-index:9999;display:none;
  line-height:1.6;color:var(--text);
}
</style>
</head>
<body>
<div class="app">
  <header class="top-bar">
    <div class="logo">J-Lens<span>/</span>Ground Truth Token Explorer</div>
    <div class="subtitle">Category vocabulary · Layer 62 · Thought segment · stopwords excluded</div>
  </header>
  <div class="layout">
    <aside class="sidebar">
      <div class="sidebar-inner">
        <div class="sidebar-label">Categories</div>
        <ul class="cat-list" id="cat-list"></ul>
      </div>
      <div class="sidebar-footer">
        <button class="compare-btn" id="compare-btn">
          <span>⇌</span> Compare two categories
        </button>
      </div>
    </aside>
    <main class="content" id="content"></main>
  </div>
</div>
<div id="tooltip"></div>

<script>
const DATA = __DATA__;
const CAT_MAP = Object.fromEntries(DATA.map(c => [c.name, c]));

const state = {
  selA: DATA[0] ? DATA[0].name : null,
  selB: DATA[1] ? DATA[1].name : null,
  compare: false,
  view: 'cloud',
  topN: 50,
  search: '',
};

// ---- Utilities ----

function esc(s) {
  return String(s)
    .replace(/&/g,'&amp;').replace(/</g,'&lt;')
    .replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

function fmtToken(t) {
  if (t === ' ') return '\u00b7';
  if (t === '\\n') return '\\u21b5';
  if (t.startsWith(' ')) return '\u00b7' + t.slice(1);
  return t;
}

function hexRgb(hex) {
  const r = parseInt(hex.slice(1,3),16);
  const g = parseInt(hex.slice(3,5),16);
  const b = parseInt(hex.slice(5,7),16);
  return r+','+g+','+b;
}

function logScale(val, minV, maxV) {
  if (minV <= 0 || maxV <= 0 || minV === maxV) return 0.5;
  return (Math.log(val) - Math.log(minV)) / (Math.log(maxV) - Math.log(minV));
}

function matchesSearch(tokenText) {
  if (!state.search) return true;
  return tokenText.toLowerCase().includes(state.search.toLowerCase());
}

// ---- Cloud ----

function renderCloud(tokens, color, container, sharedSet) {
  const sliced = tokens.slice(0, state.topN);
  if (!sliced.length) { container.innerHTML = '<div style="padding:40px;color:var(--text-3)">No tokens.</div>'; return; }
  const maxP = sliced[0].probability;
  const minP = sliced[sliced.length - 1].probability;
  const rgb = hexRgb(color);

  const cloud = document.createElement('div');
  cloud.className = 'cloud';

  sliced.forEach(({token, probability, rank}) => {
    const t = logScale(probability, minP, maxP);
    const size = (10 + t * 44).toFixed(1);
    const weight = t > 0.65 ? 600 : t > 0.35 ? 500 : 400;
    const opBg = (0.1 + t * 0.2).toFixed(2);
    const opBorder = (0.15 + t * 0.35).toFixed(2);
    const opText = (0.5 + t * 0.5).toFixed(2);
    const isShared = sharedSet && sharedSet.has(token);
    const matches = matchesSearch(token);

    const el = document.createElement('span');
    el.className = 'cloud-token' +
      (isShared ? ' shared' : '') +
      (!matches && state.search ? ' dimmed' : '');
    el.textContent = fmtToken(token);
    el.style.fontSize = size + 'px';
    el.style.fontWeight = weight;
    el.style.background = isShared
      ? 'rgba(255,255,255,0.12)'
      : 'rgba('+rgb+','+opBg+')';
    el.style.color = isShared
      ? '#fff'
      : 'rgba('+rgb+','+opText+')';
    el.style.borderColor = isShared
      ? 'rgba(255,255,255,0.3)'
      : 'rgba('+rgb+','+opBorder+')';
    el.dataset.tip = '#'+rank+'  '+probability.toExponential(4)+(isShared?' · shared':'');
    cloud.appendChild(el);
  });

  container.innerHTML = '';
  container.appendChild(cloud);
}

// ---- Bars ----

function renderBars(tokens, color, container, sharedSet) {
  const sliced = tokens.slice(0, state.topN);
  if (!sliced.length) { container.innerHTML = '<div style="padding:40px;color:var(--text-3)">No tokens.</div>'; return; }
  const maxP = sliced[0].probability;
  const rgb = hexRgb(color);

  const wrap = document.createElement('div');
  wrap.className = 'bars';

  sliced.forEach(({token, probability, rank}) => {
    const pct = (probability / maxP * 100).toFixed(2);
    const isShared = sharedSet && sharedSet.has(token);
    const matches = matchesSearch(token);
    const barColor = isShared ? 'rgba(255,255,255,0.45)' : 'rgba('+rgb+',0.75)';

    const row = document.createElement('div');
    row.className = 'bar-row';
    row.innerHTML =
      '<span class="bar-rank">'+rank+'</span>'+
      '<span class="bar-token'+((!matches&&state.search)?' dimmed':'')+'">'+esc(fmtToken(token))+'</span>'+
      '<div class="bar-track"><div class="bar-fill" style="width:'+pct+'%;background:'+barColor+'"></div></div>'+
      '<span class="bar-prob">'+probability.toExponential(3)+'</span>';
    wrap.appendChild(row);
  });

  container.innerHTML = '';
  container.appendChild(wrap);
}

// ---- Table ----

function renderTable(tokens, color, container, sharedSet) {
  const sliced = tokens.slice(0, state.topN);
  if (!sliced.length) { container.innerHTML = '<div style="padding:40px;color:var(--text-3)">No tokens.</div>'; return; }
  const maxP = sliced[0].probability;
  const rgb = hexRgb(color);

  const wrap = document.createElement('div');
  wrap.className = 'table-wrap';

  const table = document.createElement('table');
  table.className = 'gt-table';
  table.innerHTML =
    '<thead><tr>' +
    '<th>#</th><th>Token</th><th>Probability</th><th style="width:140px">Rel. weight</th>' +
    '</tr></thead>';

  const tbody = document.createElement('tbody');
  sliced.forEach(({token, probability, rank}) => {
    const pct = (probability / maxP * 100).toFixed(2);
    const isShared = sharedSet && sharedSet.has(token);
    const matches = matchesSearch(token);
    const bgColor = isShared ? 'rgba(255,255,255,0.14)' : 'rgba('+rgb+',0.18)';
    const textColor = isShared ? '#fff' : 'rgba('+rgb+',0.9)';
    const barColor = isShared ? 'rgba(255,255,255,0.5)' : 'rgba('+rgb+',0.65)';

    const tr = document.createElement('tr');
    if (!matches && state.search) tr.className = 'dimmed';
    tr.innerHTML =
      '<td style="color:var(--text-3)">'+rank+'</td>'+
      '<td><span class="token-pill" style="background:'+bgColor+';color:'+textColor+'">'+esc(fmtToken(token))+'</span>'+
        (isShared ? ' <span style="font-size:10px;color:var(--text-3)">shared</span>' : '')+'</td>'+
      '<td style="font-family:\'JetBrains Mono\',monospace;font-size:11px;color:var(--text-2)">'+probability.toExponential(4)+'</td>'+
      '<td><div class="mini-bar-wrap"><div class="mini-bar" style="width:'+pct+'%;background:'+barColor+'"></div></div></td>';
    tbody.appendChild(tr);
  });
  table.appendChild(tbody);
  wrap.appendChild(table);
  container.innerHTML = '';
  container.appendChild(wrap);
}

// ---- Dispatch ----

function renderViz(tokens, color, container, sharedSet) {
  if (state.view === 'cloud') renderCloud(tokens, color, container, sharedSet);
  else if (state.view === 'bars') renderBars(tokens, color, container, sharedSet);
  else renderTable(tokens, color, container, sharedSet);
}

function getTokenSet(catName) {
  const cat = CAT_MAP[catName];
  if (!cat) return new Set();
  return new Set(cat.tokens.slice(0, state.topN).map(t => t.token));
}

// ---- Controls HTML ----

function ctrlsHtml(showSearch) {
  const topNOpts = [10,20,30,50,75,100]
    .map(n => '<option value="'+n+'"'+(state.topN===n?' selected':'')+'>'+n+'</option>')
    .join('');
  return (
    '<div class="controls">'+
    '<div class="view-tabs">'+
    '<button class="view-tab'+(state.view==='cloud'?' active':'')+'" data-view="cloud">&#9729; Cloud</button>'+
    '<button class="view-tab'+(state.view==='bars'?' active':'')+'" data-view="bars">&#9646; Bars</button>'+
    '<button class="view-tab'+(state.view==='table'?' active':'')+'" data-view="table">&#8801; Table</button>'+
    '</div>'+
    '<div class="ctrl-group">Show top <select class="ctrl-select topn-sel">'+topNOpts+'</select> tokens</div>'+
    (showSearch ? '<div class="ctrl-group"><input class="search-input" placeholder="Filter tokens\u2026" value="'+esc(state.search)+'"></div>' : '')+
    '</div>'
  );
}

function bindControls(root) {
  root.querySelectorAll('.view-tab').forEach(btn => {
    btn.addEventListener('click', () => { state.view = btn.dataset.view; render(); });
  });
  root.querySelectorAll('.topn-sel').forEach(sel => {
    sel.addEventListener('change', e => { state.topN = parseInt(e.target.value); render(); });
  });
  const searchEl = root.querySelector('.search-input');
  if (searchEl) {
    searchEl.addEventListener('input', e => { state.search = e.target.value; render(); });
    // Keep focus after re-render
    searchEl.addEventListener('focus', () => { searchEl._focused = true; });
  }
}

// ---- Single view ----

function renderSingle(content) {
  const cat = CAT_MAP[state.selA];
  if (!cat) { content.innerHTML = '<div style="padding:40px;color:var(--text-3)">Select a category.</div>'; return; }

  const coveragePct = (cat.top_n_mass * 100).toFixed(1);
  const topTokens = cat.tokens.slice(0, state.topN);
  const shownMass = (topTokens.reduce((s,t) => s+t.probability, 0) * 100).toFixed(1);

  content.innerHTML =
    '<div class="panel" id="single-panel">'+
    '<div class="panel-header">'+
    '<div class="panel-title" style="color:'+cat.color+'">'+esc(cat.name)+'</div>'+
    '<div class="panel-meta">'+
    '<span class="meta-badge">Layer '+cat.layer+'</span>'+
    '<span class="meta-badge">Segment: '+cat.segment+'</span>'+
    '<span class="meta-badge">'+cat.n_questions+' questions</span>'+
    '<span class="meta-badge">'+cat.total_tokens_in_dist+' unique tokens</span>'+
    '<span class="meta-badge coverage" style="border-color:'+cat.color+'40;color:'+cat.color+'">'+
      'Showing top '+state.topN+' &bull; covers '+shownMass+'% of mass</span>'+
    '</div>'+
    '</div>'+
    ctrlsHtml(true)+
    '<div class="viz-area" id="viz-area"></div>'+
    '</div>';

  renderViz(cat.tokens, cat.color, document.getElementById('viz-area'), null);
  bindControls(document.getElementById('single-panel'));
}

// ---- Compare view ----

function renderCompare(content) {
  const catA = CAT_MAP[state.selA];
  const catB = CAT_MAP[state.selB];

  const opts = (sel) => DATA.map(c =>
    '<option value="'+esc(c.name)+'"'+(c.name===sel?' selected':'')+'>'+esc(c.name)+'</option>'
  ).join('');

  const topNOpts = [10,20,30,50,75,100]
    .map(n => '<option value="'+n+'"'+(state.topN===n?' selected':'')+'>'+n+'</option>')
    .join('');

  content.innerHTML =
    '<div class="compare-layout">'+

    // Left panel
    '<div class="compare-panel" id="cpanel-a">'+
    '<div class="compare-selector">'+
    '<div class="cat-dot" style="background:'+(catA?catA.color:'#888')+'"></div>'+
    '<select class="compare-select" id="sel-a">'+opts(state.selA)+'</select>'+
    '</div>'+
    '<div class="shared-legend"><div class="shared-swatch"></div>Outlined = also in right panel\'s top '+state.topN+'</div>'+
    '<div class="controls">'+
    '<div class="view-tabs">'+
    '<button class="view-tab'+(state.view==='cloud'?' active':'')+'" data-view="cloud">&#9729; Cloud</button>'+
    '<button class="view-tab'+(state.view==='bars'?' active':'')+'" data-view="bars">&#9646; Bars</button>'+
    '<button class="view-tab'+(state.view==='table'?' active':'')+'" data-view="table">&#8801; Table</button>'+
    '</div>'+
    '<div class="ctrl-group">Top <select class="ctrl-select topn-sel">'+topNOpts+'</select></div>'+
    '<div class="ctrl-group"><input class="search-input" placeholder="Filter\u2026" value="'+esc(state.search)+'"></div>'+
    '</div>'+
    '<div class="viz-area" id="viz-a"></div>'+
    '</div>'+

    // Right panel
    '<div class="compare-panel" id="cpanel-b">'+
    '<div class="compare-selector">'+
    '<div class="cat-dot" style="background:'+(catB?catB.color:'#888')+'"></div>'+
    '<select class="compare-select" id="sel-b">'+opts(state.selB)+'</select>'+
    '</div>'+
    '<div class="shared-legend"><div class="shared-swatch"></div>Outlined = also in left panel\'s top '+state.topN+'</div>'+
    '<div class="controls" style="justify-content:flex-end">'+
    '<span style="font-size:11px;color:var(--text-3)">View &amp; top-N synced \u2190</span>'+
    '</div>'+
    '<div class="viz-area" id="viz-b"></div>'+
    '</div>'+
    '</div>';

  const tokSetA = catA ? getTokenSet(state.selA) : new Set();
  const tokSetB = catB ? getTokenSet(state.selB) : new Set();

  if (catA) renderViz(catA.tokens, catA.color, document.getElementById('viz-a'), tokSetB);
  if (catB) renderViz(catB.tokens, catB.color, document.getElementById('viz-b'), tokSetA);

  // Bind
  document.querySelectorAll('.view-tab').forEach(btn => {
    btn.addEventListener('click', () => { state.view = btn.dataset.view; render(); });
  });
  document.querySelectorAll('.topn-sel').forEach(sel => {
    sel.addEventListener('change', e => { state.topN = parseInt(e.target.value); render(); });
  });
  document.querySelectorAll('.search-input').forEach(inp => {
    inp.addEventListener('input', e => { state.search = e.target.value; render(); });
  });
  document.getElementById('sel-a').addEventListener('change', e => {
    state.selA = e.target.value;
    document.querySelector('.cat-dot', '#cpanel-a').style.background = CAT_MAP[state.selA]?.color || '#888';
    render();
  });
  document.getElementById('sel-b').addEventListener('change', e => {
    state.selB = e.target.value;
    render();
  });
}

// ---- Top-level render ----

function render() {
  // Sidebar active state
  document.querySelectorAll('.cat-item').forEach(el => {
    const isActive = state.compare
      ? (el.dataset.name === state.selA || el.dataset.name === state.selB)
      : el.dataset.name === state.selA;
    el.classList.toggle('active', isActive);
  });

  const content = document.getElementById('content');
  if (state.compare) renderCompare(content);
  else renderSingle(content);

  // Restore search focus if needed
  const searchEl = document.querySelector('.search-input');
  if (searchEl && searchEl._focused) {
    searchEl.focus();
    const len = searchEl.value.length;
    searchEl.setSelectionRange(len, len);
  }
}

// ---- Tooltip ----

document.addEventListener('mouseover', e => {
  const el = e.target.closest('[data-tip]');
  const tip = document.getElementById('tooltip');
  if (el) {
    tip.textContent = el.dataset.tip;
    tip.style.display = 'block';
  } else {
    tip.style.display = 'none';
  }
});
document.addEventListener('mousemove', e => {
  const tip = document.getElementById('tooltip');
  if (tip.style.display === 'block') {
    tip.style.left = (e.clientX + 14) + 'px';
    tip.style.top  = (e.clientY + 14) + 'px';
  }
});
document.addEventListener('mouseout', e => {
  if (!e.target.closest('[data-tip]')) {
    document.getElementById('tooltip').style.display = 'none';
  }
});

// ---- Init ----

function init() {
  const catList = document.getElementById('cat-list');
  DATA.forEach(cat => {
    const li = document.createElement('li');
    li.className = 'cat-item';
    li.dataset.name = cat.name;
    li.style.setProperty('--cat-color', cat.color);
    li.innerHTML =
      '<div class="cat-dot" style="background:'+cat.color+'"></div>'+
      '<span class="cat-name">'+esc(cat.name)+'</span>';
    li.addEventListener('click', () => {
      if (state.compare) {
        state.selA = cat.name;
      } else {
        state.selA = cat.name;
      }
      state.compare = false;
      document.getElementById('compare-btn').classList.remove('active');
      render();
    });
    catList.appendChild(li);
  });

  document.getElementById('compare-btn').addEventListener('click', function() {
    state.compare = !state.compare;
    this.classList.toggle('active', state.compare);
    render();
  });

  render();
}

// Script is placed at end of <body>, so DOM is already ready — call directly.
init();
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Build the ground-truth token HTML visualizer.")
    p.add_argument("--gt-dir", default="data/ground_truths",
                   help="Directory containing per-category ground truth JSONs")
    p.add_argument("--out", default="visualizations/gt_token_viz.html",
                   help="Output HTML path")
    p.add_argument("--top", type=int, default=100,
                   help="Max tokens to include per category (default: 100)")
    p.add_argument("--model", default="Qwen/Qwen2.5-32B-Instruct",
                   help="HuggingFace model id to load the tokenizer from")
    args = p.parse_args()

    from transformers import AutoTokenizer
    print(f"Loading tokenizer from {args.model!r} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    print(f"Reading ground truth files from {args.gt_dir!r} ...")
    cats = load_categories(args.gt_dir)
    if not cats:
        print("No category files found. Make sure data/ground_truths/*.json exist.")
        return
    print(f"Found {len(cats)} category files.")

    data = build_data(cats, tokenizer, args.top)

    # Safely embed JSON in a <script> tag by escaping </
    data_json = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    html = HTML_TEMPLATE.replace("__DATA__", data_json)

    out_path = args.out
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Wrote {out_path}  ({len(data)} categories, up to {args.top} tokens each)")
    print(f"Open in browser: file://{os.path.abspath(out_path)}")


if __name__ == "__main__":
    main()
