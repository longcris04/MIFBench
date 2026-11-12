'use strict';

/* MIF review page.  Everything is derived from two server payloads:
 *   /api/index        models, pairs, metrics, which metrics are lower-is-better
 *   /api/pair/<pair>@<fold>  (fold 1-4, or "all": every image from the fold it is
 *                     a test image of) per model, per epoch: image count + version, mean and
 *                     per-image scores (arrays aligned with index.metrics),
 *                     training-loss terms
 * The view state lives in the URL hash, so a link reopens the same view. */

// ---------------------------------------------------------------- DOM helpers

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
const SVG_NS = 'http://www.w3.org/2000/svg';

function setAttrs(node, attrs) {
  for (const [key, value] of Object.entries(attrs || {})) {
    if (value == null || value === false) continue;
    if (key === 'class') node.setAttribute('class', value);
    else if (key === 'text') node.textContent = value;
    else if (key === 'style') {
      for (const [prop, v] of Object.entries(value)) {
        if (prop.startsWith('--')) node.style.setProperty(prop, v);
        else node.style[prop] = v;
      }
    } else if (key.startsWith('on')) node.addEventListener(key.slice(2), value);
    else if (!(node instanceof SVGElement) && key in node) node[key] = value;
    else node.setAttribute(key, value === true ? '' : value);
  }
}

function append(node, children) {
  for (const child of children.flat(Infinity)) {
    if (child == null || child === false) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
}

function el(tag, attrs, ...children) {
  const node = document.createElement(tag);
  setAttrs(node, attrs);
  append(node, children);
  return node;
}

/** replaceChildren() that accepts nested arrays, like el(). */
function fill(node, ...children) {
  node.replaceChildren();
  append(node, children);
}

function sv(tag, attrs, ...children) {
  const node = document.createElementNS(SVG_NS, tag);
  setAttrs(node, attrs);
  append(node, children);
  return node;
}

const enc = encodeURIComponent;
const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
const range = (a, b) => Array.from({ length: Math.max(0, b - a + 1) }, (_, i) => a + i);
const store = {
  get(key) { try { return localStorage.getItem(key); } catch { return null; } },
  set(key, value) { try { localStorage.setItem(key, value); } catch { /* storage unavailable */ } },
};
// Metrics the Scores tab shows in a browser that has no tick choice of its own yet (the "Default" button).
const DEFAULT_METRICS = ['Qabf', 'FMI', 'Xydeas', 'Qp', 'Q', 'Qw', 'Qe', 'SSIM', 'CB'];
// localStorage key of the metric ticks; "-v2" since DEFAULT_METRICS exists, so the choices browsers stored
// before (every page view stored one, often "show all") do not hide the default.
const MHIDE_KEY = 'mif-mhide-v2';
const defaultHiddenMetrics = metrics => new Set(metrics.filter(m => !DEFAULT_METRICS.includes(m)));

// ---------------------------------------------------------------- state

const S = {
  index: null,
  data: {},          // pair -> payload
  pair: null,
  fold: 'all',       // '1'..'4', or 'all' (out-of-fold: every image, from its own test fold)
  epoch: null,       // global epoch
  metric: 'Qabf',
  sort: 'metric',    // rows of the Scores table: 'metric' (the selected one) | 'rank' (rank sum over the ticked metrics)
  mode: 'color',     // gray | color
  tab: 'images',     // images | epochs | scores
  size: 176,
  hidden: new Set(), // models switched off
  hiddenMetrics: new Set(), // metrics left out of the Scores tab (never the selected one)
  override: {},      // model -> epoch shown instead of the global one
  image: null,       // image of the Epochs tab
  lb: null,          // lightbox: {image, model, s, x, y}
  lbSize: 300,
};

function readHash() {
  const p = new URLSearchParams(location.hash.slice(1));
  const idx = S.index;
  if (idx.pairs.some(x => x.name === p.get('pair'))) S.pair = p.get('pair');
  if ((idx.folds || []).includes(p.get('fold'))) S.fold = p.get('fold');
  if (Number(p.get('epoch'))) S.epoch = Number(p.get('epoch'));
  if (idx.metrics.includes(p.get('metric'))) S.metric = p.get('metric');
  if (['metric', 'rank'].includes(p.get('sort'))) S.sort = p.get('sort');
  if (['gray', 'color'].includes(p.get('mode'))) S.mode = p.get('mode');
  if (['images', 'epochs', 'scores'].includes(p.get('tab'))) S.tab = p.get('tab');
  if (Number(p.get('size'))) S.size = clamp(Number(p.get('size')), 96, 448);
  if (p.get('hide')) S.hidden = new Set(p.get('hide').split(',').filter(m => idx.models.includes(m)));
  // The metric ticks are also a per-browser preference: a link without them uses the last choice.
  // A browser with no choice yet starts with DEFAULT_METRICS; an empty choice means "show all".
  const mhide = p.has('mhide') ? p.get('mhide') : store.get(MHIDE_KEY);
  S.hiddenMetrics = mhide === null ? defaultHiddenMetrics(idx.metrics)
    : new Set(mhide.split(',').filter(m => idx.metrics.includes(m)));
  for (const part of (p.get('ep') || '').split(',')) {
    const [model, epoch] = part.split(':');
    if (idx.models.includes(model) && Number(epoch)) S.override[model] = Number(epoch);
  }
  if (p.get('img')) S.image = p.get('img');
}

function writeHash() {
  const p = new URLSearchParams();
  p.set('pair', S.pair);
  p.set('fold', S.fold);
  p.set('tab', S.tab);
  p.set('epoch', S.epoch);
  p.set('metric', S.metric);
  if (S.sort !== 'metric') p.set('sort', S.sort);
  p.set('mode', S.mode);
  p.set('size', S.size);
  if (S.hidden.size) p.set('hide', [...S.hidden].join(','));
  const mhide = S.index.metrics.filter(m => S.hiddenMetrics.has(m)).join(',');
  if (mhide) p.set('mhide', mhide);
  store.set(MHIDE_KEY, mhide);
  const ov = Object.entries(S.override).map(([m, e]) => `${m}:${e}`);
  if (ov.length) p.set('ep', ov.join(','));
  if (S.image) p.set('img', S.image);
  const hash = '#' + p.toString();
  if (location.hash !== hash) history.replaceState(null, '', hash);
}

// ---------------------------------------------------------------- data access

const viewKey = () => `${S.pair}@${S.fold}`;
const D = () => S.data[viewKey()];
const metricIndex = m => S.index.metrics.indexOf(m);
const lowerBetter = m => S.index.lower_is_better.includes(m);
const shownMetrics = () => S.index.metrics.filter(m => !S.hiddenMetrics.has(m));
// A column is shown only where its folder has runs for the pair / fold in view (a benchmark model
// without runs -- e.g. MMIF-INet while its configuration is being chosen -- is left out too).
const inPair = m => epochsOf(m).length > 0;
const pairModels = () => S.index.models.filter(inPair);
const visibleModels = () => pairModels().filter(m => !S.hidden.has(m));
const effEpoch = model => S.override[model] ?? S.epoch;
// The run's settings when they differ between a model's run folders (e.g. the loss of an experiment).
const settingsOf = model => ((D() || {}).settings || {})[model] || null;
const withSettings = (model, text) => (settingsOf(model) ? `${model} · ${settingsOf(model)}${text ? ' · ' + text : ''}` : text || model);
// Training batch size from the runs' config.json ("a/b" when the folds of the view differ).
const batchOf = model => ((D() || {}).batch_sizes || {})[model] ?? null;

function epochsOf(model) {
  const runs = D().models[model] || {};
  return Object.keys(runs).map(Number).sort((a, b) => a - b);
}
function entry(model, epoch) {
  return (D().models[model] || {})[String(epoch)] || null;
}
function hasImages(model, epoch) {
  const e = entry(model, epoch);
  return !!(e && e.images > 0);
}
function meanOf(model, epoch, metric) {
  const e = entry(model, epoch);
  return e && e.mean ? e.mean[metricIndex(metric)] : null;
}
function scoresOf(model, epoch, image) {
  const e = entry(model, epoch);
  return (e && e.per_image && e.per_image[image]) || null;
}
function perImage(model, epoch, image, metric) {
  const s = scoresOf(model, epoch, image);
  return s ? s[metricIndex(metric)] : null;
}
function maxEpoch() {
  let max = 1;
  for (const model of S.index.models) for (const e of epochsOf(model)) max = Math.max(max, e);
  return max;
}
function imgUrl(model, epoch, image) {
  const e = entry(model, epoch);
  return `/img/${enc(viewKey())}/${enc(model)}/${epoch}/${S.mode}/${enc(image)}?v=${e.v}`;
}
const srcUrl = (modality, image) => `/src/${enc(viewKey())}/${enc(modality)}/${enc(image)}?v=1`;

function isBetter(metric, a, b) { return lowerBetter(metric) ? a < b : a > b; }
function bestIndex(metric, values) {
  let best = -1;
  values.forEach((v, i) => {
    if (v == null) return;
    if (best < 0 || isBetter(metric, v, values[best])) best = i;
  });
  return best;
}
// Ranks of `values` under `lower` (lower is better?): 1 = best, ties share their mean rank, null stays null.
function ranksOf(values, lower) {
  const order = values.map((v, i) => i).filter(i => values[i] != null)
    .sort((a, b) => (lower ? values[a] - values[b] : values[b] - values[a]));
  const ranks = values.map(() => null);
  for (let s = 0; s < order.length;) {
    let e = s;
    while (e + 1 < order.length && values[order[e + 1]] === values[order[s]]) e++;
    for (let t = s; t <= e; t++) ranks[order[t]] = (s + e) / 2 + 1;
    s = e + 1;
  }
  return ranks;
}
// Shading class of each value: 'best' for the best value of the column, 'second' for the next one (ties shaded alike).
function podium(values, lower) {
  const distinct = [...new Set(values.filter(v => v != null))].sort((a, b) => (lower ? a - b : b - a));
  return values.map(v => (v == null ? '' : v === distinct[0] ? 'best' : v === distinct[1] ? 'second' : ''));
}
function bestEpoch(model, metric) {
  const epochs = epochsOf(model).filter(e => hasImages(model, e));
  const values = epochs.map(e => meanOf(model, e, metric));
  const i = bestIndex(metric, values);
  return i < 0 ? null : epochs[i];
}

function fmt(v) {
  if (v == null || !Number.isFinite(v)) return '—';
  const a = Math.abs(v);
  if (a >= 1000) return Math.round(v).toLocaleString('en-US');
  if (a >= 100) return v.toFixed(1);
  if (a >= 10) return v.toFixed(2);
  if (a > 0 && a < 0.001) return v.toExponential(2);
  return v.toFixed(4);
}
const arrow = m => (lowerBetter(m) ? '↓' : '↑');
const dirText = m => (lowerBetter(m) ? 'lower is better' : 'higher is better');

function modelStyle(model) {
  const i = S.index.models.indexOf(model);
  return { color: `var(--s${(i % 8) + 1})`, dashed: i >= 8 };
}
function keyLine(style, width = 18) {
  return sv('svg', { class: 'key', width, height: 10, viewBox: `0 0 ${width} 10`, 'aria-hidden': 'true' },
    sv('line', {
      x1: 1, y1: 5, x2: width - 1, y2: 5, 'stroke-width': 2.5, 'stroke-linecap': 'round',
      'stroke-dasharray': style.dashed ? '4 3' : null, style: { stroke: style.color },
    }));
}

// ---------------------------------------------------------------- loading

async function fetchJSON(url) {
  const response = await fetch(url, { cache: 'no-store' });
  if (!response.ok) throw new Error(`${url}: HTTP ${response.status}`);
  return response.json();
}

function setStatus(text) { $('#status').textContent = text; }

async function loadPair(pair, force = false) {
  if (S.data[pair] && !force) return;
  document.body.classList.add('loading');
  setStatus(`Loading ${pair}…`);
  try {
    S.data[pair] = await fetchJSON(`/api/pair/${enc(pair)}`);
    setStatus(`${pair} read at ${new Date().toLocaleTimeString()}`);
  } catch (err) {
    setStatus(`Could not load ${pair}: ${err.message}`);
  } finally {
    document.body.classList.remove('loading');
  }
}

// ---------------------------------------------------------------- render loop

let frame = 0;
function scheduleRender() {
  if (!frame) frame = requestAnimationFrame(() => { frame = 0; render(); });
}

function render() {
  if (!D()) return;
  S.epoch = clamp(S.epoch ?? maxEpoch(), 1, maxEpoch());
  if (!S.image || !D().images.includes(S.image)) S.image = D().images[0];
  S.hiddenMetrics.delete(S.metric); // selecting a metric anywhere ticks it again
  writeHash();
  renderFilters();
  renderChips();
  for (const tab of ['images', 'epochs', 'scores']) $(`#tab-${tab}`).hidden = tab !== S.tab;
  hideTooltip();
  if (S.tab === 'images') renderImages();
  else if (S.tab === 'epochs') renderEpochs();
  else renderScores();
  if (S.lb) renderLightbox();
}

// ---------------------------------------------------------------- filters

function buildFilters() {
  for (const pair of S.index.pairs) {
    $('#pairSeg').append(el('button', {
      'data-pair': pair.name,
      onclick: async () => { S.pair = pair.name; await loadPair(viewKey()); render(); },
    }, pair.name));
  }
  for (const fold of S.index.folds || []) {
    $('#foldSeg').append(el('button', {
      'data-fold': fold,
      title: fold === 'all' ? 'Every image of the pair, each fused by the model of the fold it is a test image of'
                            : `Test split of fold ${fold}`,
      onclick: async () => { S.fold = fold; await loadPair(viewKey()); render(); },
    }, fold === 'all' ? 'All (OOF)' : fold));
  }
  for (const mode of ['gray', 'color']) {
    $('#modeSeg').append(el('button', { 'data-mode': mode, onclick: () => setMode(mode) },
      mode === 'gray' ? 'Gray' : 'Color'));
  }
  for (const m of S.index.metrics) {
    $('#metric').append(el('option', { value: m }, `${m}  ${arrow(m)}`));
  }
  $('#metric').addEventListener('change', ev => { S.metric = ev.target.value; render(); });
  $('#epoch').addEventListener('input', ev => { S.epoch = Number(ev.target.value); scheduleRender(); });
  $('#epPrev').addEventListener('click', () => stepEpoch(-1));
  $('#epNext').addEventListener('click', () => stepEpoch(1));
  $('#size').addEventListener('input', ev => { S.size = Number(ev.target.value); applySize(); writeHash(); });
  $('#bestEp').addEventListener('click', () => {
    for (const m of S.index.models) {
      const e = bestEpoch(m, S.metric);
      if (e != null) S.override[m] = e; else delete S.override[m];
    }
    setStatus(`Each model at its best mean ${S.metric}`);
    render();
  });
  $('#clearEp').addEventListener('click', () => { S.override = {}; render(); });
  $('#reload').addEventListener('click', async () => { await loadPair(viewKey(), true); render(); });
  $('#theme').addEventListener('click', () => {
    const order = ['auto', 'light', 'dark'];
    applyTheme(order[(order.indexOf(currentTheme()) + 1) % order.length]);
  });
  for (const button of $$('.tabs button')) {
    button.addEventListener('click', () => { S.tab = button.dataset.tab; render(); });
  }
}

function renderFilters() {
  for (const b of $$('#pairSeg button')) b.setAttribute('aria-pressed', String(b.dataset.pair === S.pair));
  for (const b of $$('#foldSeg button')) b.setAttribute('aria-pressed', String(b.dataset.fold === S.fold));
  for (const b of $$('#modeSeg button')) b.setAttribute('aria-pressed', String(b.dataset.mode === S.mode));
  for (const b of $$('.tabs button')) b.setAttribute('aria-selected', String(b.dataset.tab === S.tab));
  const slider = $('#epoch');
  slider.max = maxEpoch();
  slider.value = S.epoch;
  const overridden = Object.keys(S.override).length;
  $('#epochOut').textContent = `${S.epoch} / ${maxEpoch()}` + (overridden ? `  (+${overridden} per-model)` : '');
  $('#metric').value = S.metric;
  $('#size').value = S.size;
  applySize();
}

function renderChips() {
  fill($('#modelChips'),
    el('span', { class: 'label' }, 'Models'),
    pairModels().map(m => el('button', {
      class: 'chip', 'aria-pressed': String(!S.hidden.has(m)),
      title: withSettings(m, 'Click to show/hide · Alt+click to show only this model'),
      onclick: ev => {
        if (ev.altKey) S.hidden = new Set(S.index.models.filter(x => x !== m));
        else if (S.hidden.has(m)) S.hidden.delete(m);
        else S.hidden.add(m);
        render();
      },
      onmouseenter: () => highlight(m), onmouseleave: () => highlight(null),
    }, keyLine(modelStyle(m)), m)),
    el('button', { class: 'ghost', onclick: () => { S.hidden.clear(); render(); } }, 'All'),
    el('span', { class: 'hint' }, 'Keys: [ ] epoch · g / c gray / color · click an image to compare and zoom'),
  );
}

function applySize() { document.documentElement.style.setProperty('--cell', `${S.size}px`); }

function setMode(mode) { S.mode = mode; render(); }

function stepEpoch(delta) {
  S.epoch = clamp(S.epoch + delta, 1, maxEpoch());
  render();
}

function currentTheme() { return document.documentElement.dataset.theme || 'auto'; }
function applyTheme(theme) {
  if (theme === 'auto') delete document.documentElement.dataset.theme;
  else document.documentElement.dataset.theme = theme;
  $('#theme').textContent = `Theme: ${theme}`;
  store.set('mif-theme', theme);
}

// ---------------------------------------------------------------- shared pieces

function keepScroll(root, build) {
  const old = root.querySelector('.gridwrap');
  const pos = old ? [old.scrollTop, old.scrollLeft] : [0, 0];
  build();
  const wrap = root.querySelector('.gridwrap');
  if (wrap) [wrap.scrollTop, wrap.scrollLeft] = pos;
}

function imgCell(url, value, best, title, onClick, current = false) {
  const caption = value === undefined ? ' ' : (best ? `★ ${fmt(value)}` : fmt(value));
  return el('button', { class: 'cell' + (current ? ' current' : ''), title, onclick: onClick },
    el('img', { src: url, loading: 'lazy', decoding: 'async', alt: title || '' }),
    el('span', { class: 'val' + (best ? ' best' : '') }, caption));
}

function missingCell(text) {
  return el('div', { class: 'cell missing' }, el('div', { class: 'ph' }, text), el('span', { class: 'val' }, ' '));
}

function scoreTitle(model, epoch, image, value) {
  const e = entry(model, epoch);
  let text = `${model} · epoch ${epoch} · ${image}\n${S.metric} = ${fmt(value)}`;
  if (value == null) text += e && !e.per_image ? ' (epoch not scored)' : '';
  else text += `  (${e.score_file})`;
  if (e && e.stale) text += '\nImages were re-generated after scoring; scores belong to an earlier sample.';
  return text;
}

function flags(model, epoch) {
  const e = entry(model, epoch);
  const out = [];
  if (!e || !e.images) return out;
  const n = D().images.length;
  if (e.images < n) out.push(el('div', { class: 'flag' }, el('span', { class: 'icon' }, '⚠ '), `${e.images}/${n} images`));
  if (e.stale) {
    out.push(el('div', {
      class: 'flag', title: 'This model samples random noise. The images were re-generated after the epoch was scored, so the scores describe an earlier sample.',
    }, el('span', { class: 'icon' }, '⚠ '), 'scores: earlier sample'));
  }
  return out;
}

// ---------------------------------------------------------------- Images tab

function renderImages() {
  const root = $('#tab-images');
  keepScroll(root, () => {
    const d = D();
    const models = visibleModels();
    const metric = S.metric;
    const means = models.map(m => meanOf(m, effEpoch(m), metric));
    const bestMean = bestIndex(metric, means);

    const grid = el('div', {
      class: 'grid', style: { gridTemplateColumns: `auto repeat(${d.sources.length + models.length}, var(--cell))` },
    });
    grid.append(el('div', { class: 'hcell corner' }, 'Image'));
    for (const src of d.sources) {
      grid.append(el('div', { class: 'hcell' }, el('div', { class: 'hname' }, src), el('div', { class: 'hsub' }, 'source')));
    }
    models.forEach((m, i) => grid.append(modelHeader(m, means[i], i === bestMean)));

    for (const image of d.images) {
      grid.append(el('div', { class: 'rowname' }, image));
      for (const src of d.sources) {
        grid.append(imgCell(srcUrl(src, image), undefined, false, `${src} · ${image}`, () => openLightbox(image)));
      }
      const values = models.map(m => perImage(m, effEpoch(m), image, metric));
      const best = bestIndex(metric, values);
      models.forEach((m, i) => {
        const e = effEpoch(m);
        grid.append(hasImages(m, e)
          ? imgCell(imgUrl(m, e, image), values[i], i === best, scoreTitle(m, e, image, values[i]), () => openLightbox(image))
          : missingCell(`no image at epoch ${e}`));
      });
    }

    root.replaceChildren(
      el('p', { class: 'note' },
        `${d.images.length} test images. Under each fused image: its ${metric} (${dirText(metric)}); ★ marks the best model in that row. `,
        'Change one model\'s epoch in its column header.'),
      el('div', { class: 'gridwrap' }, grid));
  });
}

function modelHeader(model, mean, isBest) {
  const e = effEpoch(model);
  const select = el('select', {
    class: 'epsel', 'aria-label': `Epoch for ${model}`,
    onchange: ev => {
      if (ev.target.value === '') delete S.override[model];
      else S.override[model] = Number(ev.target.value);
      render();
    },
  }, el('option', { value: '' }, `global (epoch ${S.epoch})`),
  epochsOf(model).filter(x => hasImages(model, x)).map(x => el('option', {
    value: String(x), selected: S.override[model] === x,
  }, `epoch ${x}${x === bestEpoch(model, S.metric) ? ' ★' : ''}`)));
  return el('div', { class: 'hcell' },
    el('div', { class: 'hname', title: withSettings(model) }, keyLine(modelStyle(model)), model),
    settingsOf(model) && el('div', { class: 'hsub' }, settingsOf(model)),
    select,
    el('div', { class: 'hsub' + (isBest ? ' best' : '') },
      `mean ${fmt(mean)}${isBest ? ' ★' : ''}`),
    flags(model, e));
}

// ---------------------------------------------------------------- Epochs tab

function renderEpochs() {
  const root = $('#tab-epochs');
  keepScroll(root, () => {
    const d = D();
    const metric = S.metric;
    const epochs = range(1, maxEpoch());
    const models = visibleModels();
    const i = d.images.indexOf(S.image);

    const toolbar = el('div', { class: 'toolbar' },
      el('span', { class: 'label' }, 'Image'),
      el('button', { onclick: () => stepImage(-1), 'aria-label': 'Previous image' }, '‹'),
      el('select', {
        'aria-label': 'Image', onchange: ev => { S.image = ev.target.value; render(); },
      }, d.images.map(name => el('option', { value: name, selected: name === S.image }, name))),
      el('button', { onclick: () => stepImage(1), 'aria-label': 'Next image' }, '›'),
      el('span', { class: 'muted' }, `${i + 1} / ${d.images.length}`),
      d.sources.map(src => el('button', {
        class: 'cell', style: { width: '64px' }, title: `${src} · ${S.image}`, onclick: () => openLightbox(S.image),
      }, el('img', { src: srcUrl(src, S.image), alt: src, style: { width: '64px', height: '64px' } }))),
      el('span', { class: 'note', style: { margin: 0 } },
        `Rows: models. Columns: epochs (click one to make it global). Under each image: its ${metric} (${dirText(metric)}); ★ = that model's best epoch for this image.`));

    const grid = el('div', { class: 'grid', style: { gridTemplateColumns: `auto repeat(${epochs.length}, var(--cell))` } });
    grid.append(el('div', { class: 'hcell corner' }, 'Model'));
    for (const e of epochs) {
      grid.append(el('button', {
        class: 'hcell ep' + (e === S.epoch ? ' current' : ''), title: 'Make this the global epoch',
        onclick: () => { S.epoch = e; render(); },
      }, `epoch ${e}`));
    }
    for (const m of models) {
      const values = epochs.map(e => perImage(m, e, S.image, metric));
      const best = bestIndex(metric, values);
      grid.append(el('div', { class: 'rowname' }, keyLine(modelStyle(m)), m));
      epochs.forEach((e, k) => grid.append(hasImages(m, e)
        ? imgCell(imgUrl(m, e, S.image), values[k], k === best, scoreTitle(m, e, S.image, values[k]),
          () => openLightbox(S.image, m), e === S.epoch)
        : missingCell(`no image at epoch ${e}`)));
    }
    root.replaceChildren(toolbar, el('div', { class: 'gridwrap' }, grid));
  });
}

function stepImage(delta) {
  const images = D().images;
  const current = S.lb ? S.lb.image : S.image;
  const next = images[(images.indexOf(current) + delta + images.length) % images.length];
  S.image = next;
  if (S.lb) S.lb.image = next;
  render();
}

// ---------------------------------------------------------------- Scores tab

function renderScores() {
  const root = $('#tab-scores');
  const y = window.scrollY;
  const metric = S.metric;
  const models = visibleModels();
  const xmax = maxEpoch();

  const series = models.map(m => ({
    model: m, label: m, ...modelStyle(m),
    points: epochsOf(m).map(e => [e, meanOf(m, e, metric)]),
  }));

  const main = el('section', { class: 'card' },
    el('h2', {}, `${metric} by epoch`),
    el('p', { class: 'sub' },
      `Mean over the test images, ${dirText(metric)}. Vertical line = global epoch ${S.epoch}; click the chart to move it. `,
      'Gaps are joined: a model scored every second epoch still reads as one line.'),
    legend(pairModels()),
    lineChart({
      series, xmax, height: 320, metric, current: S.epoch,
      onPick: e => { S.epoch = e; render(); },
      ariaLabel: `${metric} by epoch for each model`,
    }),
    el('details', {}, el('summary', {}, 'Table view'), el('div', { class: 'tablewrap' }, epochTable(series))));

  const old = root.querySelector('.meanwrap');
  const left = old ? old.scrollLeft : 0;
  root.replaceChildren(metricPicker(), meanTableCard(), main, multiplesCard(models, xmax), lossCard(models, xmax));
  root.querySelector('.meanwrap').scrollLeft = left;
  window.scrollTo(0, y);
}

function setMetricShown(metric, alone) {
  const metrics = S.index.metrics;
  if (alone) {
    S.hiddenMetrics = new Set(metrics.filter(x => x !== metric));
    S.metric = metric;
  } else if (S.hiddenMetrics.has(metric)) {
    S.hiddenMetrics.delete(metric);
  } else {
    if (shownMetrics().length === 1) return; // keep at least one
    S.hiddenMetrics.add(metric);
    if (metric === S.metric) {
      // The selected metric is always shown, so the selection moves on to the next ticked one.
      const i = metrics.indexOf(metric);
      S.metric = [...metrics.slice(i + 1), ...metrics.slice(0, i)].find(x => !S.hiddenMetrics.has(x));
    }
  }
  render();
  // render() rebuilt the ticks; keep keyboard focus on the one just toggled
  const box = $$('#tab-scores input[data-metric]').find(n => n.dataset.metric === metric);
  if (box) box.focus();
}

function metricPicker() {
  const metrics = S.index.metrics;
  const shown = shownMetrics();
  const only = shown.length === 1;
  return el('section', { class: 'card metricpick' },
    el('span', { class: 'label' }, `Metrics (${shown.length}/${metrics.length})`),
    metrics.map(m => {
      const on = !S.hiddenMetrics.has(m);
      return el('label', {
        class: ['mtick', on ? '' : 'off', m === S.metric ? 'sel' : ''].join(' ').trim(),
        title: `${m}: ${dirText(m)}` + (m === S.metric ? ' · selected metric' : '') + ' · Alt+click to show only this metric',
        // Handle the click here (not the checkbox's change) to see Alt; preventDefault stops the
        // browser's own toggle, since render() redraws the box from S anyway.
        onclick: ev => { ev.preventDefault(); setMetricShown(m, ev.altKey); },
      },
      el('input', {
        type: 'checkbox', checked: on, disabled: on && only, 'aria-label': `Show ${m}`, 'data-metric': m,
      }),
      m, el('span', { class: 'dir' }, arrow(m)));
    }),
    el('button', {
      class: 'ghost', disabled: !S.hiddenMetrics.size,
      onclick: () => { S.hiddenMetrics.clear(); render(); },
    }, 'All'),
    el('button', {
      class: 'ghost', title: `Show ${DEFAULT_METRICS.join(', ')}`,
      onclick: () => { S.hiddenMetrics = defaultHiddenMetrics(metrics); render(); },
    }, 'Default'),
    el('button', {
      class: 'ghost', disabled: only, title: 'Untick every metric but the selected one',
      onclick: () => { S.hiddenMetrics = new Set(metrics.filter(x => x !== S.metric)); render(); },
    }, `Only ${S.metric}`),
    el('span', { class: 'hint' },
      'Columns of the table and charts below. Unticking the selected metric selects the next ticked one.'));
}

function setCompared(models, on) {
  for (const m of models) {
    if (on) S.hidden.delete(m); else S.hidden.add(m);
  }
  render();
}

function meanTableCard() {
  const metrics = shownMetrics();
  const metric = S.metric;
  // Every model of the pair has a row, so an unticked one can be ticked again. The tick is the
  // Models chips' switch: an unticked model also leaves the charts and the other tabs.
  const rows = pairModels().map(m => ({
    model: m, on: !S.hidden.has(m), batch: batchOf(m), epoch: effEpoch(m), best: bestEpoch(m, metric),
  }));
  for (const row of rows) row.values = metrics.map(x => meanOf(row.model, row.epoch, x));
  // Rank sum over the ticked metrics, among the ticked models: 1 = best in a metric, ties share their
  // mean rank, and a model with no score in a metric ranks last there. The lowest sum is the strongest.
  const on = rows.filter(r => r.on);
  for (const row of rows) row.rankSum = null;
  metrics.forEach((x, k) => {
    const ranks = ranksOf(on.map(r => r.values[k]), lowerBetter(x));
    on.forEach((row, i) => { row.rankSum = (row.rankSum ?? 0) + (ranks[i] ?? on.length); });
  });
  const j = metrics.indexOf(metric); // the selected metric is never unticked
  const byMetric = (a, b) => {
    const va = a.values[j], vb = b.values[j];
    if (va == null) return vb == null ? 0 : 1;
    if (vb == null) return -1;
    return lowerBetter(metric) ? va - vb : vb - va;
  };
  rows.sort((a, b) => {
    if (a.on !== b.on) return a.on ? -1 : 1; // ticked models first, ranked among themselves
    if (S.sort === 'rank' && a.on && a.rankSum !== b.rankSum) return a.rankSum - b.rankSum;
    return byMetric(a, b); // also breaks rank-sum ties
  });
  const shade = metrics.map((x, k) => podium(rows.map(r => (r.on ? r.values[k] : null)), lowerBetter(x)));
  const rankShade = podium(rows.map(r => r.rankSum), true);
  const fmtRank = v => (Number.isInteger(v) ? String(v) : v.toFixed(1));
  const ticked = on.length;
  const firstOff = rows.findIndex(r => !r.on);

  const all = el('input', {
    type: 'checkbox', checked: ticked === rows.length, 'aria-label': 'Compare every model',
    title: 'Tick or untick every model',
    onchange: ev => setCompared(rows.map(r => r.model), ev.target.checked),
  });
  all.indeterminate = ticked > 0 && ticked < rows.length;

  const table = el('table', { class: 'scores' },
    el('thead', {}, el('tr', {},
      el('th', {}, el('label', { class: 'pick' }, all, `Model (${ticked}/${rows.length})`)),
      el('th', { title: 'Training batch size of this model\'s runs (config.json); a/b when the folds differ' }, 'Batch size'),
      el('th', { title: 'Epoch shown for this model' }, 'Epoch'),
      el('th', { title: `Epoch with the best mean ${metric}` }, `Best ep. (${metric})`),
      el('th', {
        class: S.sort === 'rank' ? 'sel' : null,
        title: `Sum of each ticked model's ranks over the ${metrics.length} ticked metrics (1 = best in a metric; `
          + 'lower is better) · click to sort by it',
        onclick: () => { S.sort = 'rank'; render(); },
      }, 'Σ rank', el('span', { class: 'dir' }, '↓')),
      metrics.map(x => el('th', {
        class: x === metric && S.sort === 'metric' ? 'sel' : null, title: `${x}: ${dirText(x)} · click to sort and select`,
        onclick: () => { S.metric = x; S.sort = 'metric'; render(); },
      }, x, el('span', { class: 'dir' }, arrow(x)))))),
    el('tbody', {}, rows.map((row, i) => el('tr', {
      class: [row.on ? '' : 'off', i === firstOff && i > 0 ? 'first-off' : ''].join(' ').trim() || null,
    },
      el('td', { title: withSettings(row.model) }, el('label', { class: 'pick' },
        el('input', {
          type: 'checkbox', checked: row.on, 'aria-label': `Compare ${row.model}`,
          onchange: ev => setCompared([row.model], ev.target.checked),
        }),
        keyLine(modelStyle(row.model)), row.model,
        settingsOf(row.model) && el('span', { class: 'dir' }, ` · ${settingsOf(row.model)}`))),
      el('td', {}, row.batch ?? '—'),
      el('td', {}, row.epoch, S.override[row.model] != null ? ' *' : ''),
      el('td', {}, row.best ?? '—', row.best != null ? ` (${fmt(meanOf(row.model, row.best, metric))})` : ''),
      el('td', {
        class: [rankShade[i], S.sort === 'rank' ? 'sel' : ''].join(' ').trim() || null,
        title: row.rankSum == null ? null : `mean rank ${(row.rankSum / metrics.length).toFixed(2)} over ${metrics.length} metrics`,
      }, row.rankSum == null ? '—' : fmtRank(row.rankSum)),
      metrics.map((x, k) => el('td', {
        class: [shade[k][i], x === metric ? 'sel' : ''].join(' ').trim() || null,
      }, fmt(row.values[k])))))));

  return el('section', { class: 'card' },
    el('h2', {}, 'Mean scores'),
    el('p', { class: 'sub' },
      `Each model at global epoch ${S.epoch}, or at its own epoch where marked *. Blue = best, red = second best in the column among the ticked models. `,
      S.sort === 'rank'
        ? `Sorted by Σ rank: each model's ranks over the ${metrics.length} ticked metrics added up (1 = best in a metric, ties share their mean rank, no score = last), lowest first; ties by ${metric}. `
        : 'Sorted by the selected metric; click a column header to select another. ',
      '— = not scored at that epoch. ',
      'Untick a model to leave it out: it moves to the bottom and out of the shading, the ranks, the charts and the other tabs (same switch as the Models chips).'),
    el('div', { class: 'sortpick' },
      el('span', {}, 'Sort rows by'),
      el('div', { class: 'seg' },
        el('button', { 'aria-pressed': String(S.sort === 'metric'), onclick: () => { S.sort = 'metric'; render(); } },
          `Selected metric (${metric})`),
        el('button', {
          'aria-pressed': String(S.sort === 'rank'), title: 'Strongest first: lowest sum of ranks over the ticked metrics',
          onclick: () => { S.sort = 'rank'; render(); },
        }, `Σ rank over ${metrics.length} ticked metrics`))),
    el('div', { class: 'tablewrap meanwrap' }, table));
}

function epochTable(series) {
  const xmax = maxEpoch();
  return el('table', { class: 'scores' },
    el('thead', {}, el('tr', {}, el('th', {}, 'Epoch'), series.map(s => el('th', {}, s.label)))),
    el('tbody', {}, range(1, xmax).map(e => el('tr', {},
      el('td', {}, e),
      series.map(s => el('td', {}, fmt((s.points.find(p => p[0] === e) || [])[1])))))));
}

function multiplesCard(models, xmax) {
  const grid = el('div', { class: 'multiples' });
  for (const metric of shownMetrics()) {
    const series = models.map(m => ({
      model: m, label: m, ...modelStyle(m),
      points: epochsOf(m).map(e => [e, meanOf(m, e, metric)]),
    }));
    grid.append(el('div', { class: 'mini' + (metric === S.metric ? ' sel' : '') },
      el('h3', {}, el('button', {
        title: 'Select this metric', onclick: () => { S.metric = metric; render(); window.scrollTo(0, 0); },
      }, metric), el('span', { class: 'dir' }, `${arrow(metric)} ${dirText(metric)}`)),
      lineChart({
        series, xmax, height: 150, small: true, metric, current: S.epoch,
        onPick: e => { S.epoch = e; render(); }, ariaLabel: `${metric} by epoch`,
      })));
  }
  return el('section', { class: 'card' },
    el('h2', {}, S.hiddenMetrics.size ? 'Ticked metrics by epoch' : 'All metrics by epoch'),
    el('p', { class: 'sub' }, 'Same legend as above. Click a title to select that metric.'),
    grid);
}

function lossCard(models, xmax) {
  const grid = el('div', { class: 'multiples' });
  for (const m of models) {
    const style = modelStyle(m);
    const terms = epochsOf(m).map(e => (entry(m, e) || {}).train || {});
    const name = terms.some(t => 'loss' in t) ? 'loss' : Object.keys(terms.find(t => Object.keys(t).length) || {})[0];
    if (!name) continue;
    const series = [{ model: m, label: name, ...style, points: epochsOf(m).map(e => [e, ((entry(m, e) || {}).train || {})[name] ?? null]) }];
    const rowsAt = e => {
      const train = (entry(m, e) || {}).train || {};
      return Object.keys(train).sort((a, b) => (a === name ? -1 : b === name ? 1 : a.localeCompare(b)))
        .map(k => ({ key: k === name ? style : null, value: fmt(train[k]), name: k }));
    };
    grid.append(el('div', { class: 'mini' },
      el('h3', {}, keyLine(style), m),
      lineChart({ series, xmax, height: 150, small: true, current: S.epoch, rowsAt, ariaLabel: `${m} training ${name} by epoch` })));
  }
  return el('section', { class: 'card' },
    el('h2', {}, 'Training loss by epoch'),
    el('p', { class: 'sub' }, 'Epoch average of the total loss; each model on its own scale (the losses are different functions). Hover for every loss term.'),
    grid);
}

function legend(models) {
  return el('div', { class: 'legend' }, models.map(m => el('button', {
    'aria-pressed': String(!S.hidden.has(m)), title: withSettings(m, 'Show/hide'),
    onclick: () => { if (S.hidden.has(m)) S.hidden.delete(m); else S.hidden.add(m); render(); },
    onmouseenter: () => highlight(m), onmouseleave: () => highlight(null),
    onfocus: () => highlight(m), onblur: () => highlight(null),
  }, keyLine(modelStyle(m)), m)));
}

function highlight(model) {
  for (const node of $$('.chart [data-model]')) {
    node.classList.toggle('dim', model != null && node.dataset.model !== model);
  }
}

// ---------------------------------------------------------------- line chart

function niceTicks(lo, hi, count) {
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) return [0, 1];
  if (lo === hi) {
    const pad = Math.abs(lo) * 0.05 || 1;
    lo -= pad; hi += pad;
  }
  const raw = (hi - lo) / Math.max(1, count);
  const mag = 10 ** Math.floor(Math.log10(raw));
  const norm = raw / mag;
  const step = (norm < 1.5 ? 1 : norm < 3 ? 2 : norm < 7 ? 5 : 10) * mag;
  const ticks = [];
  for (let v = Math.floor(lo / step) * step; v <= hi + step * 0.5; v += step) {
    ticks.push(Number(v.toPrecision(12)));
    if (v >= hi) break;
  }
  return ticks;
}

function tickFormatter(ticks) {
  const step = ticks.length > 1 ? Math.abs(ticks[1] - ticks[0]) : 1;
  const decimals = clamp(-Math.floor(Math.log10(step)), 0, 6);
  return v => (Math.abs(v) >= 1000 ? Math.round(v).toLocaleString('en-US') : v.toFixed(decimals));
}

function lineChart({ series, xmax, height, small = false, metric = null, current = null, onPick = null, rowsAt = null, ariaLabel = '' }) {
  const box = el('div', { class: 'chart', tabindex: 0, role: 'img', 'aria-label': ariaLabel, style: { height: `${height}px` } });
  for (const s of series) s.map = new Map(s.points.filter(p => p[1] != null));
  let geo = null;
  let hoverE = null;
  let lastWidth = 0;

  function dot(x, y, s, cls) {
    return sv('circle', {
      class: `dot ${cls}`, cx: x, cy: y, r: 4, 'data-model': s.model,
      style: s.dashed ? { fill: 'var(--surface)', stroke: s.color } : { fill: s.color, stroke: 'var(--surface)' },
    });
  }

  function draw(width) {
    const m = small ? { l: 46, r: 10, t: 8, b: 22 } : { l: 58, r: 18, t: 12, b: 28 };
    const iw = width - m.l - m.r;
    const ih = height - m.t - m.b;
    const values = series.flatMap(s => [...s.map.values()]);
    const root = sv('svg', { width, height, viewBox: `0 0 ${width} ${height}` });
    if (!values.length) {
      root.append(sv('text', { x: width / 2, y: height / 2, 'text-anchor': 'middle', text: 'no scores' }));
      box.replaceChildren(root);
      geo = null;
      return;
    }
    const ticks = niceTicks(Math.min(...values), Math.max(...values), small ? 3 : 5);
    const y0 = ticks[0];
    const y1 = ticks[ticks.length - 1];
    const X = e => m.l + (xmax <= 1 ? iw / 2 : ((e - 1) / (xmax - 1)) * iw);
    const Y = v => m.t + ih - ((v - y0) / (y1 - y0 || 1)) * ih;
    const tickText = tickFormatter(ticks);

    for (const t of ticks) {
      root.append(sv('line', { class: 'gridline', x1: m.l, x2: m.l + iw, y1: Y(t), y2: Y(t) }));
      root.append(sv('text', { x: m.l - 6, y: Y(t) + 4, 'text-anchor': 'end', text: tickText(t) }));
    }
    root.append(sv('line', { class: 'axis', x1: m.l, x2: m.l + iw, y1: m.t + ih, y2: m.t + ih }));
    const every = small ? 5 : (iw / Math.max(1, xmax - 1) < 28 ? 2 : 1);
    for (const e of range(1, xmax)) {
      if (e === 1 || e % every === 0) {
        root.append(sv('text', { x: X(e), y: height - (small ? 6 : 8), 'text-anchor': 'middle', text: e }));
      }
    }
    if (current != null) root.append(sv('line', { class: 'now', x1: X(current), x2: X(current), y1: m.t, y2: m.t + ih }));

    for (const s of series) {
      const pts = [...s.map.entries()].sort((a, b) => a[0] - b[0]);
      if (!pts.length) continue;
      const d = pts.map(([e, v], i) => `${i ? 'L' : 'M'}${X(e).toFixed(1)},${Y(v).toFixed(1)}`).join('');
      root.append(sv('path', {
        class: 'series', d, 'data-model': s.model, 'stroke-dasharray': s.dashed ? '6 4' : null,
        style: { stroke: s.color },
      }));
      if (pts.length === 1) root.append(dot(X(pts[0][0]), Y(pts[0][1]), s, 'single'));
    }
    if (current != null) {
      for (const s of series) if (s.map.has(current)) root.append(dot(X(current), Y(s.map.get(current)), s, 'nowdot'));
    }

    const hover = sv('g', { class: 'hover' });
    const hit = sv('rect', {
      x: m.l - 10, y: m.t, width: iw + 20, height: ih, fill: 'transparent',
      style: { cursor: onPick ? 'pointer' : 'crosshair' },
    });
    root.append(hover, hit);
    geo = { X, Y, m, ih, hover };
    hit.addEventListener('pointermove', ev => {
      const r = root.getBoundingClientRect();
      const e = clamp(Math.round(1 + ((ev.clientX - r.left - m.l) / iw) * (xmax - 1)), 1, xmax);
      showHover(e, ev.clientX, ev.clientY);
    });
    hit.addEventListener('pointerleave', clearHover);
    if (onPick) hit.addEventListener('click', () => { if (hoverE != null) onPick(hoverE); });
    box.replaceChildren(root);
  }

  function showHover(e, cx, cy) {
    if (!geo) return;
    hoverE = e;
    const { X, Y, m, ih, hover } = geo;
    hover.replaceChildren(sv('line', { class: 'cross', x1: X(e), x2: X(e), y1: m.t, y2: m.t + ih }));
    const found = [];
    for (const s of series) {
      if (!s.map.has(e)) continue;
      const v = s.map.get(e);
      hover.append(dot(X(e), Y(v), s, 'hoverdot'));
      found.push({ s, v });
    }
    if (metric) found.sort((a, b) => (lowerBetter(metric) ? a.v - b.v : b.v - a.v));
    const rows = rowsAt ? rowsAt(e) : found.map(({ s, v }) => ({ key: s, value: fmt(v), name: s.label }));
    showTooltip(`Epoch ${e}${metric ? ` · ${metric} ${arrow(metric)}` : ''}`, rows.length ? rows : [{ value: '—', name: 'not scored' }], cx, cy);
  }

  function clearHover() {
    hoverE = null;
    if (geo) geo.hover.replaceChildren();
    hideTooltip();
  }

  box.addEventListener('keydown', ev => {
    if (!geo) return;
    if (ev.key === 'ArrowLeft' || ev.key === 'ArrowRight') {
      ev.preventDefault();
      const e = clamp((hoverE ?? current ?? 1) + (ev.key === 'ArrowLeft' ? -1 : 1), 1, xmax);
      const r = box.getBoundingClientRect();
      showHover(e, r.left + geo.X(e), r.top + geo.m.t + 10);
    } else if (ev.key === 'Enter' && onPick && hoverE != null) {
      onPick(hoverE);
    }
  });
  box.addEventListener('blur', clearHover);
  new ResizeObserver(entries => {
    const width = Math.floor(entries[0].contentRect.width);
    if (width > 60 && width !== lastWidth) { lastWidth = width; draw(width); }
  }).observe(box);
  return box;
}

function showTooltip(title, rows, cx, cy) {
  const tip = $('#tooltip');
  fill(tip,
    el('div', { class: 'tt-title' }, title),
    rows.map(r => el('div', { class: 'tt-row' },
      r.key ? keyLine(r.key, 16) : el('span'),
      el('span', { class: 'tt-v' }, r.value),
      el('span', { class: 'tt-n' }, r.name))));
  tip.hidden = false;
  const pad = 14;
  const w = tip.offsetWidth;
  const h = tip.offsetHeight;
  let x = cx + pad;
  let y = cy + pad;
  if (x + w > innerWidth - 8) x = cx - pad - w;
  if (y + h > innerHeight - 8) y = Math.max(8, innerHeight - 8 - h);
  tip.style.left = `${x}px`;
  tip.style.top = `${y}px`;
}

function hideTooltip() { $('#tooltip').hidden = true; }

// ---------------------------------------------------------------- lightbox

function openLightbox(image, model = null) {
  S.lb = { image, model, s: 1, x: 0, y: 0 };
  renderLightbox();
}

function closeLightbox() {
  S.lb = null;
  const box = $('#lightbox');
  box.hidden = true;
  box.replaceChildren();
  document.body.classList.remove('noscroll');
  render();
}

function lightboxPanels() {
  const d = D();
  const { image, model } = S.lb;
  const panels = d.sources.map(src => ({ label: src, sub: 'source', url: srcUrl(src, image) }));
  if (model) {
    for (const e of epochsOf(model)) {
      if (!hasImages(model, e)) continue;
      panels.push({
        label: `epoch ${e}`, sub: e === S.epoch ? 'global' : '', style: modelStyle(model),
        url: imgUrl(model, e, image), scores: scoresOf(model, e, image), rowLabel: `epoch ${e}`,
      });
    }
  } else {
    for (const m of visibleModels()) {
      const e = effEpoch(m);
      panels.push({
        label: m, sub: `epoch ${e}`, style: modelStyle(m),
        url: hasImages(m, e) ? imgUrl(m, e, image) : null, scores: scoresOf(m, e, image),
        rowLabel: `${m} (epoch ${e})`,
      });
    }
  }
  return panels;
}

function renderLightbox() {
  const box = $('#lightbox');
  const lb = S.lb;
  const d = D();
  if (!lb || !d) return;
  const oldBody = box.querySelector('.lb-body');
  const scroll = oldBody ? oldBody.scrollTop : 0;
  const panels = lightboxPanels();
  const metric = S.metric;
  const values = panels.map(p => (p.scores ? p.scores[metricIndex(metric)] : null));
  const best = bestIndex(metric, values);
  const i = d.images.indexOf(lb.image);

  const head = el('div', { class: 'lb-head' },
    el('strong', {}, `Image ${lb.image}`),
    el('span', { class: 'muted' }, `${i + 1} / ${d.images.length}`),
    el('button', { onclick: () => stepImage(-1), title: 'Previous image (←)' }, '‹ Prev'),
    el('button', { onclick: () => stepImage(1), title: 'Next image (→)' }, 'Next ›'),
    el('select', {
      'aria-label': 'Compare', onchange: ev => { lb.model = ev.target.value || null; renderLightbox(); },
    }, el('option', { value: '' }, 'All models, each at its epoch'),
    pairModels().map(m => el('option', { value: m, selected: lb.model === m }, `${m} across epochs`))),
    el('span', { class: 'muted' }, `Caption: ${metric} ${arrow(metric)}, ★ best`),
    el('div', { class: 'spacer' }),
    el('span', { id: 'lbZoom' }),
    el('button', { onclick: resetZoom, title: 'Reset zoom (0 or double-click)' }, 'Reset zoom'),
    el('button', { onclick: () => { S.lbSize = clamp(S.lbSize - 60, 160, 900); renderLightbox(); }, 'aria-label': 'Smaller panels' }, '−'),
    el('button', { onclick: () => { S.lbSize = clamp(S.lbSize + 60, 160, 900); renderLightbox(); }, 'aria-label': 'Larger panels' }, '+'),
    el('button', { onclick: closeLightbox, title: 'Close (Esc)' }, 'Close ✕'));

  const grid = el('div', { class: 'lb-grid', style: { '--lbcell': `${S.lbSize}px` } },
    panels.map((p, k) => panelEl(p, values[k], k === best)));
  const body = el('div', { class: 'lb-body' },
    el('p', { class: 'note' }, 'Scroll to zoom, drag to pan — every panel follows. Keys: ← → image · [ ] global epoch · g / c gray / color · 0 reset · Esc close.'),
    grid, lightboxTable(panels));
  box.replaceChildren(head, body);
  box.hidden = false;
  document.body.classList.add('noscroll');
  body.scrollTop = scroll;
  applyZoom();
}

function panelEl(p, value, best) {
  const vp = el('div', { class: 'vp' },
    p.url ? el('img', { src: p.url, alt: `${p.label} ${p.sub || ''}`, draggable: false }) : el('div', { class: 'ph' }, 'no image'));
  attachZoom(vp);
  return el('figure', { class: 'panel' },
    el('figcaption', {},
      p.style ? keyLine(p.style) : null,
      el('span', { class: 'pl' }, p.label),
      p.sub ? el('span', { class: 'muted' }, p.sub) : null,
      el('span', { class: 'pv' + (best ? ' best' : '') }, p.style ? (best ? `★ ${fmt(value)}` : fmt(value)) : '')),
    vp);
}

function lightboxTable(panels) {
  const rows = panels.filter(p => p.scores);
  const title = S.lb.model ? `Per-image scores · ${S.lb.model} · image ${S.lb.image}` : `Per-image scores · image ${S.lb.image}`;
  if (!rows.length) {
    return el('section', { class: 'card' }, el('h2', {}, title), el('p', { class: 'sub' }, 'No per-image scores for these panels.'));
  }
  const metrics = S.index.metrics;
  const best = metrics.map((m, j) => bestIndex(m, rows.map(p => p.scores[j])));
  return el('section', { class: 'card' },
    el('h2', {}, title),
    el('p', { class: 'sub' }, 'Shaded = best in column. Click a column header to select that metric everywhere.'),
    el('div', { class: 'tablewrap' }, el('table', { class: 'scores' },
      el('thead', {}, el('tr', {},
        el('th', {}, S.lb.model ? 'Epoch' : 'Model'),
        metrics.map(m => el('th', {
          class: m === S.metric ? 'sel' : null, title: `${m}: ${dirText(m)}`,
          onclick: () => { S.metric = m; render(); },
        }, m, el('span', { class: 'dir' }, arrow(m)))))),
      el('tbody', {}, rows.map((p, i) => el('tr', {},
        el('td', {}, keyLine(p.style), p.rowLabel),
        metrics.map((m, j) => el('td', {
          class: [best[j] === i ? 'best' : '', m === S.metric ? 'sel' : ''].join(' ').trim() || null,
        }, fmt(p.scores[j])))))))));
}

function attachZoom(vp) {
  vp.addEventListener('wheel', ev => {
    ev.preventDefault();
    const lb = S.lb;
    const r = vp.getBoundingClientRect();
    const px = (ev.clientX - r.left) / r.width;
    const py = (ev.clientY - r.top) / r.height;
    const s = clamp(lb.s * Math.exp(-ev.deltaY * 0.0015), 1, 40);
    lb.x = px - ((px - lb.x) * s) / lb.s;
    lb.y = py - ((py - lb.y) * s) / lb.s;
    lb.s = s;
    applyZoom();
  }, { passive: false });
  let drag = null;
  vp.addEventListener('pointerdown', ev => {
    drag = { x: ev.clientX, y: ev.clientY, w: vp.clientWidth, h: vp.clientHeight };
    vp.setPointerCapture(ev.pointerId);
    vp.classList.add('dragging');
  });
  vp.addEventListener('pointermove', ev => {
    if (!drag) return;
    S.lb.x += (ev.clientX - drag.x) / drag.w;
    S.lb.y += (ev.clientY - drag.y) / drag.h;
    drag.x = ev.clientX;
    drag.y = ev.clientY;
    applyZoom();
  });
  const end = () => { drag = null; vp.classList.remove('dragging'); };
  vp.addEventListener('pointerup', end);
  vp.addEventListener('pointercancel', end);
  vp.addEventListener('dblclick', resetZoom);
}

function applyZoom() {
  const lb = S.lb;
  if (!lb) return;
  lb.x = clamp(lb.x, 1 - lb.s, 0);
  lb.y = clamp(lb.y, 1 - lb.s, 0);
  // translate() percentages are of the image's own box, so the offsets hold at any panel size
  const transform = `translate(${lb.x * 100}%, ${lb.y * 100}%) scale(${lb.s})`;
  for (const img of $$('#lightbox .vp img')) img.style.transform = transform;
  $('#lightbox').classList.toggle('pixelated', lb.s >= 3);
  const label = $('#lbZoom');
  if (label) label.textContent = `${lb.s.toFixed(1)}×`;
}

function resetZoom() {
  Object.assign(S.lb, { s: 1, x: 0, y: 0 });
  applyZoom();
}

// ---------------------------------------------------------------- keys + start

function onKey(ev) {
  if (ev.altKey || ev.ctrlKey || ev.metaKey) return;
  const typing = ev.target.closest && ev.target.closest('input, select, textarea');
  if (S.lb) {
    if (ev.key === 'Escape') closeLightbox();
    else if (typing) return;
    else if (ev.key === 'ArrowLeft') stepImage(-1);
    else if (ev.key === 'ArrowRight') stepImage(1);
    else if (ev.key === '0') resetZoom();
    else if (ev.key === '[') stepEpoch(-1);
    else if (ev.key === ']') stepEpoch(1);
    else if (ev.key === 'g') setMode('gray');
    else if (ev.key === 'c') setMode('color');
    else return;
    ev.preventDefault();
    return;
  }
  if (typing) return;
  if (ev.key === '[') stepEpoch(-1);
  else if (ev.key === ']') stepEpoch(1);
  else if (ev.key === 'g') setMode('gray');
  else if (ev.key === 'c') setMode('color');
}

async function init() {
  applyTheme(store.get('mif-theme') || 'auto');
  S.index = await fetchJSON('/api/index');
  S.pair = S.index.pairs[0].name;
  readHash();
  buildFilters();
  await loadPair(viewKey());
  render();
  window.addEventListener('keydown', onKey);
  const top = $('#top');
  new ResizeObserver(() => document.documentElement.style.setProperty('--hdr', `${top.offsetHeight}px`)).observe(top);
}

init().catch(err => {
  document.body.classList.remove('loading');
  setStatus(`Failed to start: ${err.message}`);
});
