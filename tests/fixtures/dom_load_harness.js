// Load-time smoke harness for web/static/js/dashboard.js.
//
// WHY THIS EXISTS
// dashboard.js is 12k+ lines whose only automated check was `node --check`,
// which is a *syntax* check. A ReferenceError is a runtime failure, so the
// whole file passed while being completely non-functional.
//
// The concrete incident: the `applyAnalysisRec` fix moved one function to
// module scope and left a `resultsEl.addEventListener(...)` statement at
// closure level, but `resultsEl` was declared with `const` *inside* the nested
// `analyzeAllModels`. That threw `ReferenceError: resultsEl is not defined`
// from inside the DOMContentLoaded handler, which ABORTS the rest of the
// handler. The page then rendered: no models in the switcher ("Loading
// models..." forever), "Server Monitor Offline", no tabs wired - while the
// server, the proxy and every API answered 200 in milliseconds. A null-deref
// at load time presents as "the app is broken", not as an error.
//
// WHAT IT DOES
// Evaluates the real shipped file in a `vm` context whose `document` returns
// auto-generated stubs for any id, then fires DOMContentLoaded and load. Any
// thrown error is reported with the dashboard.js frames.
//
// WHAT IT DELIBERATELY DOES NOT DO
// * `querySelectorAll` returns a real one-element array, not a stub, so
//   `.forEach` really runs and the handler bindings inside it really execute -
//   an unbound name is precisely the failure being hunted.
// * `fetch` never resolves. The load-time wiring is synchronous; parking the
//   async tails keeps this hermetic, offline and fast. It is a *load* test,
//   not a render test.
// * Stubs mean a MISSING element id is invisible here. That is the other half
//   of the bug class and is covered separately, by cross-checking the ids this
//   harness reports against the ids index.html actually declares.
//
// Usage: node dom_load_harness.js /path/to/dashboard.js
// Exits 0 with a JSON summary when the page wires up cleanly, 1 otherwise.

'use strict';

const fs = require('fs');
const vm = require('vm');

const SRC = process.argv[2];
if (!SRC) {
  process.stderr.write('usage: node dom_load_harness.js <dashboard.js>\n');
  process.exit(2);
}

const looked_up_ids = new Set();
const problems = [];

/**
 * A value that is simultaneously callable, iterable and string-ish, so
 * `el.x.y(z)`, `for (const o of el.x)`, `el.x.length` and `` `${el.x}` `` all
 * work without the harness having to know what the code will ask for.
 */
function universal(name) {
  const fn = function () {
    return universal(name + '()');
  };
  const cache = {};
  return new Proxy(fn, {
    get(target, prop) {
      if (prop === Symbol.toPrimitive) return () => '';
      if (prop === 'toString') return () => '[stub ' + name + ']';
      if (prop === Symbol.iterator) return function* () {};
      if (prop === 'then') return undefined; // never look thenable
      if (prop === 'length') return 0;
      if (prop === 'value' || prop === 'textContent' || prop === 'innerHTML' || prop === 'id') return '';
      if (prop === 'checked' || prop === 'disabled' || prop === 'selected') return false;
      if (prop === 'files') return [];
      if (typeof prop === 'symbol') return undefined;
      if (!(prop in cache)) cache[prop] = universal(name + '.' + String(prop));
      return cache[prop];
    },
    set(target, prop, value) {
      cache[prop] = value;
      return true;
    },
    has() {
      return true;
    },
  });
}

let onDomReady = null;
let onWindowLoad = null;

const document = {
  readyState: 'loading',
  title: '',
  cookie: '',
  body: universal('document.body'),
  head: universal('document.head'),
  documentElement: universal('document.documentElement'),
  createElement: (tag) => universal('createElement(' + tag + ')'),
  createTextNode: (t) => universal('textNode'),
  createDocumentFragment: () => universal('fragment'),
  addEventListener: (event, fn) => {
    if (event === 'DOMContentLoaded') onDomReady = fn;
  },
  removeEventListener: () => {},
  dispatchEvent: () => true,
  getElementById: (id) => {
    looked_up_ids.add(id);
    return universal('#' + id);
  },
  querySelector: (sel) => universal('querySelector(' + sel + ')'),
  querySelectorAll: (sel) => [universal('qsa(' + sel + ')')],
  getElementsByClassName: (sel) => [universal('gbc(' + sel + ')')],
  getElementsByTagName: (sel) => [universal('gbt(' + sel + ')')],
  elementFromPoint: () => universal('elementFromPoint'),
  fonts: { ready: Promise.resolve(), load: () => Promise.resolve([]) },
};

const store = new Map();
const storage = {
  getItem: (k) => (store.has(k) ? store.get(k) : null),
  setItem: (k, v) => store.set(k, String(v)),
  removeItem: (k) => store.delete(k),
  clear: () => store.clear(),
  key: () => null,
  get length() {
    return store.size;
  },
};

class NullishEvent {
  constructor(type, init) {
    this.type = type;
    this.detail = (init || {}).detail;
  }
}

const sandbox = {
  console: { log() {}, warn() {}, error() {}, info() {}, debug() {} },
  fetch: () => new Promise(() => {}),
  setTimeout: () => 0,
  clearTimeout: () => {},
  setInterval: () => 0,
  clearInterval: () => {},
  requestAnimationFrame: () => 0,
  cancelAnimationFrame: () => {},
  queueMicrotask: (fn) => fn(),
  localStorage: storage,
  sessionStorage: storage,
  alert() {},
  confirm: () => true,
  prompt: () => 'stub',
  atob: (s) => Buffer.from(String(s), 'base64').toString('binary'),
  btoa: (s) => Buffer.from(String(s), 'binary').toString('base64'),
  navigator: { userAgent: 'node', mediaDevices: {}, onLine: true, clipboard: {} },
  location: {
    href: 'http://localhost:5000/',
    origin: 'http://localhost:5000',
    protocol: 'http:',
    host: 'localhost:5000',
    hostname: 'localhost',
    pathname: '/',
    search: '',
    hash: '',
  },
  history: { pushState() {}, replaceState() {} },
  matchMedia: () => ({ matches: false, addEventListener() {}, addListener() {} }),
  getComputedStyle: () => ({ getPropertyValue: () => '' }),
  requestIdleCallback: () => 0,
  IntersectionObserver: class { observe() {} unobserve() {} disconnect() {} },
  ResizeObserver: class { observe() {} unobserve() {} disconnect() {} },
  MutationObserver: class { observe() {} disconnect() {} },
  PerformanceObserver: class { observe() {} disconnect() {} },
  CustomEvent: NullishEvent,
  Event: NullishEvent,
  DOMParser: class { parseFromString() { return universal('parsedDoc'); } },
  XMLHttpRequest: class { open() {} send() {} setRequestHeader() {} addEventListener() {} },
  Image: class { constructor() { this.width = 0; this.height = 0; } addEventListener() {} set src(v) {} },
  Audio: class {
    constructor() { this.duration = 0; this.currentTime = 0; }
    play() { return Promise.resolve(); }
    pause() {}
    addEventListener() {}
  },
  Option: class { constructor(a, b) { this.value = a; this.text = b; } },
  Node: { ELEMENT_NODE: 1, TEXT_NODE: 3 },
  Element: class {},
  HTMLElement: class {},
  io: universal('io'),
  Chart: universal('Chart'),
  URL,
  URLSearchParams,
  Blob: class {},
  FormData: class { append() {} entries() { return []; } },
  Headers: class {},
  Intl,
  TextDecoder,
  TextEncoder,
  devicePixelRatio: 1,
  innerWidth: 1600,
  innerHeight: 900,
  document,
  addEventListener: (event, fn) => {
    if (event === 'load') onWindowLoad = fn;
  },
  removeEventListener: () => {},
  postMessage: () => {},
  __proto__: null,
};
sandbox.window = sandbox;
sandbox.self = sandbox;
sandbox.globalThis = sandbox;
sandbox.top = sandbox;
sandbox.parent = sandbox;

let stage = 'evaluate module scope';
try {
  vm.createContext(sandbox);
  vm.runInContext(fs.readFileSync(SRC, 'utf8'), sandbox, { filename: 'dashboard.js' });

  stage = 'DOMContentLoaded handler';
  if (typeof onDomReady !== 'function') {
    problems.push('no DOMContentLoaded handler was registered');
  } else {
    onDomReady();
  }

  stage = 'window load handler';
  if (typeof onWindowLoad === 'function') onWindowLoad();
} catch (err) {
  problems.push(stage + ': ' + (err && err.message ? err.message : String(err)));
  const frames = String((err && err.stack) || '')
    .split('\n')
    .filter((line) => line.includes('dashboard.js'))
    .slice(0, 3);
  for (const frame of frames) problems.push('    dashboard.js' + frame.split('dashboard.js')[1]);
}

process.stdout.write(
  JSON.stringify({ problems, looked_up_ids: [...looked_up_ids].sort() }, null, 2) + '\n',
);
process.exit(problems.length ? 1 : 0);
