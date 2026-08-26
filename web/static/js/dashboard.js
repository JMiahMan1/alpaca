document.addEventListener('DOMContentLoaded', () => {
    // Escape HTML helper for safe innerHTML interpolation
    function escapeHtml(s) {
        return String(s == null ? '' : s)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#39;');
    }

    // Online model prefix tester
    function isOnlineModelName(model) {
        return /^(openrouter|huggingface|hf|cloudflare|opencode_zen|groq|gemini|openai|custom):/i.test(String(model || '').trim());
    }

    // Socket initialization
    const socket = io();

    // Navigation Tabs Elements
    const tabBtnMonitor = document.getElementById('tab-btn-monitor');
    const tabBtnGeneral = document.getElementById('tab-btn-general');
    const tabBtnShared = document.getElementById('tab-btn-shared');
    const tabBtnTests = document.getElementById('tab-btn-tests');
    const tabBtnProfiles = document.getElementById('tab-btn-profiles');
    const tabBtnRequests = document.getElementById('tab-btn-requests');
    const tabBtnDocs = document.getElementById('tab-btn-docs');
    const tabBtnSd = document.getElementById('tab-btn-sd');
    const tabBtnAudio = document.getElementById('tab-btn-audio');
    const viewMonitor = document.getElementById('view-monitor');
    const viewGeneral = document.getElementById('view-general');
    const viewShared = document.getElementById('view-shared');
    const viewTests = document.getElementById('view-tests');
    const viewProfiles = document.getElementById('view-profiles');
    const viewRequests = document.getElementById('view-requests');
    const viewDocs = document.getElementById('view-docs');
    const viewImageStudio = document.getElementById('view-image-studio');
    const viewAudio = document.getElementById('view-audio');


    // Controls Elements
    const connectionStatusBadge = document.getElementById('connection-status-badge');
    const runnerStatusBadge = document.getElementById('runner-status-badge');
    const modelCheckboxes = document.getElementById('model-checkboxes');
    const modelSwitcherSelect = document.getElementById('model-switcher-select');
    const btnSwitchModel = document.getElementById('btn-switch-model');
    const btnUnloadCurrent = document.getElementById('btn-unload-current');
    const modelSwitcherStatus = document.getElementById('model-switcher-status');
    const modeProxyBtn = document.getElementById('mode-proxy-btn');
    const modeDirectBtn = document.getElementById('mode-direct-btn');
    const btnRun = document.getElementById('btn-run');
    const btnRunShared = document.getElementById('btn-run-shared');
    const btnRunMultistep = document.getElementById('btn-run-multistep');
    const btnRunOutdated = document.getElementById('btn-run-outdated');
    const headerLogoutBtn = document.getElementById('header-logout-btn');

    if (headerLogoutBtn) {
        fetch('/api/auth/status')
            .then(res => res.json())
            .then(data => {
                if (data.authenticated && !data.is_local) {
                    headerLogoutBtn.style.display = 'inline-flex';
                }
            })
            .catch(() => {});
    }
    const btnCancel = document.getElementById('btn-cancel');
     const selectAllBtn = document.getElementById('btn-select-all');
    const deselectAllBtn = document.getElementById('btn-deselect-all');
    const testCheckboxes = document.getElementById('test-checkboxes');
    const groupCheckboxes = document.getElementById('group-checkboxes');
    const selectAllTestsBtn = document.getElementById('btn-select-all-tests');
    const deselectAllTestsBtn = document.getElementById('btn-deselect-all-tests');
    
    // Progress UI Elements
    const progressCard = document.getElementById('progress-card');
    const progressText = document.getElementById('progress-text');
    const progressPercent = document.getElementById('progress-percent');
    const progressBarFill = document.getElementById('progress-bar-fill');
    const statusModel = document.getElementById('status-model');
    const statusTest = document.getElementById('status-test');
    const statusCategory = document.getElementById('status-category');

    // Tab 1: System Monitor Elements
    const proxyStatusCard = document.getElementById('proxy-status-card');
    const proxyConnectionTitle = document.getElementById('proxy-connection-title');
    const proxyConnectionSubtitle = document.getElementById('proxy-connection-subtitle');
    const proxyUptimeBadge = document.getElementById('proxy-uptime-badge');
    const cpuPercent = document.getElementById('cpu-percent');
    const cpuBar = document.getElementById('cpu-bar');
    const ramUsageText = document.getElementById('ram-usage-text');
    const ramBar = document.getElementById('ram-bar');
    const vramUsageText = document.getElementById('vram-usage-text');
    const vramBar = document.getElementById('vram-bar');
    const loadedModelName = document.getElementById('loaded-model-name');
    const loadedModelRequests = document.getElementById('loaded-model-requests');
    const loadedModelContext = document.getElementById('loaded-model-context');
    const loadedModelTtl = document.getElementById('loaded-model-ttl');
    const proxyTotalRequests = document.getElementById('proxy-total-requests');
    const proxyAvgLatency = document.getElementById('proxy-avg-latency');
    const proxyPromptTokens = document.getElementById('proxy-prompt-tokens');
    const proxyGenTokens = document.getElementById('proxy-gen-tokens');
    const monitorClients = document.getElementById('monitor-clients');
    const monitorEndpoints = document.getElementById('monitor-endpoints');
    const slotsGridContainer = document.getElementById('slots-grid-container');

    // Tab 2: General Metrics Overview Cards Elements
    const metricTps = document.getElementById('metric-tps');
    const metricTtft = document.getElementById('metric-ttft');
    const metricSuccess = document.getElementById('metric-success');
    const metricCount = document.getElementById('metric-count');
    const consoleTerminal = document.getElementById('console-terminal');
    const historyList = document.getElementById('history-list');
    const modelTabs = document.getElementById('model-tabs');
    const detailedResultsBody = document.getElementById('detailed-results-body');

    // Tab 3: SharedLLM Metrics Cards Elements
    const sharedMetricFastpath = document.getElementById('shared-metric-fastpath');
    const sharedMetricLibrarian = document.getElementById('shared-metric-librarian');
    const sharedMetricRaven = document.getElementById('shared-metric-raven');
    const sharedMetricCount = document.getElementById('shared-metric-count');
    const sharedConsoleTerminal = document.getElementById('shared-console-terminal');
    const sharedModelTabs = document.getElementById('shared-model-tabs');
    const sharedDetailedResultsBody = document.getElementById('shared-detailed-results-body');

    // Modal Elements
    const modalOverlay = document.getElementById('modal-overlay');
    const modalClose = document.getElementById('modal-close');
    const modalPrompt = document.getElementById('modal-prompt');
    const modalResponse = document.getElementById('modal-response');

    // Chart Variables (General)
    let tpsChart = null;
    let ttftChart = null;
    let categoryChart = null;

    // Chart Variables (SharedLLM)
    let sharedLatencyChart = null;
    let sharedAstChart = null;
    
    // Memory Creep and Optimizations
    let memoryCreepChart = null;
    let currentRecommendations = null;

    // State variables
    let activeTab = 'monitor'; // 'monitor', 'general', 'shared'
    let benchmarkMode = 'proxy'; // 'proxy' or 'direct'
    let availableModels = [];
    let currentResults = []; // Currently loaded run results
    // Per-run trend data collected from run snapshots (not per-model aggregates):
    // entries {label, model, score} where label is the run's timestamp.
    const TREND_DATA = { general: [], shared: [] };
    const TREND_CHARTS = { general: null, shared: null };
    const RADAR_CHARTS = {};
    // Chart.js palette for per-model trend lines (cycled).
    const TREND_COLORS = ['#8b5cf6', '#06b6d4', '#10b981', '#f59e0b', '#ef4444',
        '#3b82f6', '#ec4899', '#84cc16', '#f97316', '#14b8a6'];
    let currentSharedResults = [];
    let monitorIntervalId = null;

    // Model comparison filter state (per view type: 'general' | 'shared')
    let filterAllModels = { general: [], shared: [] };              // every model id seen in each view
    let filterSelection = { general: new Set(), shared: new Set() }; // selected model ids (empty until first results)
    let filterInitialized = { general: false, shared: false };
    let filterCheckboxInputs = { general: {}, shared: {} };          // model id -> <input> element

    // Utility function to truncate long model names
    function truncateModelName(name) {
        if (!name) return '';
        // Router ids use "--" as separator (e.g. family--quant); humanize to the
        // public "family:quant" form so the model family stays visible, and drop
        // the redundant Ollama ":latest" default tag (version is in the name).
        let cleanName = name.replace(/\.gguf$/i, '');
        cleanName = cleanName.replace(/--/g, ':');
        cleanName = cleanName.replace(/:latest$/, '');

        if (cleanName.length <= 25) return cleanName;

        return cleanName.substring(0, 12) + '...' + cleanName.substring(cleanName.length - 10);
    }

    // Tab Navigation Logic
    function switchTab(tabName) {
        activeTab = tabName;
        
        // Update URL hash without causing page reload
        if (window.location.hash !== `#${tabName}`) {
            history.replaceState(null, null, `#${tabName}`);
        }
        
        // Update tab buttons
        tabBtnMonitor.classList.remove('active');
        tabBtnGeneral.classList.remove('active');
        tabBtnShared.classList.remove('active');
        tabBtnTests.classList.remove('active');
        tabBtnProfiles.classList.remove('active');
        tabBtnRequests.classList.remove('active');
        tabBtnDocs.classList.remove('active');
        tabBtnSd.classList.remove('active');
        tabBtnAudio.classList.remove('active');

        // Hide views
        viewMonitor.classList.add('d-none');
        viewGeneral.classList.add('d-none');
        viewShared.classList.add('d-none');
        viewTests.classList.add('d-none');
        viewProfiles.classList.add('d-none');
        viewRequests.classList.add('d-none');
        viewDocs.classList.add('d-none');
        viewImageStudio.classList.add('d-none');
        viewAudio.classList.add('d-none');
        
        // Stop both polls to start clean
        stopMonitorPolling();
        stopRequestsPolling();
        
        if (tabName === 'monitor') {
            tabBtnMonitor.classList.add('active');
            viewMonitor.classList.remove('d-none');
            // Resume/start fast monitor polling
            startMonitorPolling();
        } else if (tabName === 'general') {
            tabBtnGeneral.classList.add('active');
            viewGeneral.classList.remove('d-none');
        } else if (tabName === 'shared') {
            tabBtnShared.classList.add('active');
            viewShared.classList.remove('d-none');
            loadRoutingMatrix();
        } else if (tabName === 'tests') {
            tabBtnTests.classList.add('active');
            viewTests.classList.remove('d-none');
            loadTestBrowser();
        } else if (tabName === 'profiles') {
            tabBtnProfiles.classList.add('active');
            viewProfiles.classList.remove('d-none');
            loadModelProfiles();
            startMonitorPolling();
        } else if (tabName === 'requests') {
            tabBtnRequests.classList.add('active');
            viewRequests.classList.remove('d-none');
            startRequestsPolling();
        } else if (tabName === 'docs') {
            tabBtnDocs.classList.add('active');
            viewDocs.classList.remove('d-none');
            setupDocsMenuHandlers();
        } else if (tabName === 'sd') {
            tabBtnSd.classList.add('active');
            viewImageStudio.classList.remove('d-none');
            loadSdModels();
        } else if (tabName === 'audio') {
            tabBtnAudio.classList.add('active');
            viewAudio.classList.remove('d-none');
            initAudioStudio();
        }
        document.dispatchEvent(new CustomEvent('tabChanged', { detail: tabName }));
    }

    tabBtnMonitor.addEventListener('click', () => switchTab('monitor'));
    tabBtnGeneral.addEventListener('click', () => switchTab('general'));
    tabBtnShared.addEventListener('click', () => switchTab('shared'));
    tabBtnTests.addEventListener('click', () => switchTab('tests'));
    tabBtnProfiles.addEventListener('click', () => switchTab('profiles'));
    tabBtnRequests.addEventListener('click', () => switchTab('requests'));
    tabBtnDocs.addEventListener('click', () => switchTab('docs'));
    tabBtnSd.addEventListener('click', () => switchTab('sd'));
    tabBtnAudio.addEventListener('click', () => switchTab('audio'));

    // Image Studio (Stable Diffusion)
    async function loadSdModels() {
        const sel = document.getElementById('sd-model-select');
        if (!sel) return;
        // Signatures that indicate a true image-generation model
        const SD_PATTERNS = [
            /\.safetensors$/i,
            /\bflux\b/i,
            /\bsdxl\b/i,
            /\bsd[123x]\b/i,
            /\bstable.diffusion\b/i,
            /\bsd-/i,
            /-sd\b/i,
            /\bimage.gen\b/i,
        ];
        const isImageModel = name =>
            SD_PATTERNS.some(re => re.test(name)) ||
            // family tag returned by the API
            false;

        try {
            const res = await fetch('/api/sd/models');
            const data = await res.json();
            const allModels = (data.data || []);
            // Filter: prefer family tag from API; fall back to name heuristic
            const imageModels = allModels.filter(m => {
                const family = (m.family || '').toLowerCase();
                if (family === 'stable-diffusion' || family === 'flux' || family === 'sdxl') return true;
                return isImageModel(m.name || '');
            });
            if (imageModels.length === 0) {
                sel.innerHTML = '<option value="">No image-generation models found</option>';
                return;
            }
            sel.innerHTML = imageModels.map(m => `<option value="${m.name}">${m.name}</option>`).join('');

            // Attach change listener to update UI options dynamically for Qwen vs Diffusion models
            sel.addEventListener('change', () => updateSDUIForModel(sel.value));
            if (sel.value) updateSDUIForModel(sel.value);

            // Populate Vision & Synthesis model selectors in Image-to-Prompt Assistant
            const visionSel = document.getElementById('sd-promptgen-vision-model');
            const synthSel = document.getElementById('sd-promptgen-synth-model');

            // Humanize router alias names (e.g. "qwen2.5-vl--3b" → "qwen2.5-vl:3b")
            const humanizeModelName = (id) => id.replace(/--/g, ':');

            if (visionSel) {
                // Vision: Strictly VL multimodal vision models (e.g. qwen2.5-vl:7b)
                try {
                    const res = await fetch('/api/models/vision');
                    const data = await res.json();
                    const models = data.models || [];
                    visionSel.innerHTML = models.length
                        ? models.map(m => `<option value="${m}">${humanizeModelName(m)}</option>`).join('')
                        : '<option value="" disabled>No vision models available (load a VL model)</option>';
                } catch(err) {
                    console.warn('Could not populate vision model selector:', err);
                }
            }

            if (synthSel) {
                // Synthesis: Ollama text-only models (VL excluded - optimised for image understanding)
                try {
                    const res = await fetch('/api/models');
                    const data = await res.json();
                    const models = (data.models || []).filter(m => !m.toLowerCase().includes('vl'));
                    synthSel.innerHTML = models.length
                        ? models.map(m => `<option value="${m}">${humanizeModelName(m)}</option>`).join('')
                        : '<option value="" disabled>No models available</option>';
                } catch(err) {
                    console.warn('Could not populate synthesis model selector:', err);
                }
            }
        } catch (e) {
            sel.innerHTML = '<option value="">Error loading models</option>';
            logToTerminal('Failed to load SD models: ' + e.message, 'error');
        }
    }

    function updateSDUIForModel(modelName) {
        if (!modelName) return;
        const nameLower = modelName.toLowerCase();
        const isQwen = nameLower.includes('qwen');
        const isFlux = nameLower.includes('flux');

        const badge = document.getElementById('sd-model-type-badge');
        const editNeg = document.getElementById('sd-edit-negative');
        const genNeg = document.getElementById('sd-gen-negative');
        const targetModelSel = document.getElementById('sd-promptgen-target-model');

        if (isQwen) {
            if (badge) {
                badge.textContent = 'Qwen Image Edit (Instruction VLM)';
                badge.style.background = '#7e22ce';
                badge.style.color = 'white';
                badge.style.border = '1px solid #a855f7';
            }
            if (editNeg) {
                editNeg.disabled = true;
                editNeg.style.opacity = '0.4';
                editNeg.title = 'Negative prompts are not used by Qwen Image Edit instruction models';
            }
            if (genNeg) {
                genNeg.disabled = true;
                genNeg.style.opacity = '0.4';
                genNeg.title = 'Negative prompts are not used by Qwen Image Edit instruction models';
            }
            if (targetModelSel) targetModelSel.value = 'qwen-image-edit';
        } else if (isFlux) {
            if (badge) {
                badge.textContent = 'Flux (Natural Description)';
                badge.style.background = '#0284c7';
                badge.style.color = 'white';
                badge.style.border = '1px solid #38bdf8';
            }
            if (editNeg) { editNeg.disabled = false; editNeg.style.opacity = '1.0'; editNeg.title = ''; }
            if (genNeg) { genNeg.disabled = false; genNeg.style.opacity = '1.0'; genNeg.title = ''; }
            if (targetModelSel) targetModelSel.value = 'flux';
        } else {
            if (badge) {
                badge.textContent = 'Diffusion Model (CFG + Negative)';
                badge.style.background = 'rgba(56, 189, 248, 0.15)';
                badge.style.color = '#38bdf8';
                badge.style.border = '1px solid rgba(56, 189, 248, 0.3)';
            }
            if (editNeg) { editNeg.disabled = false; editNeg.style.opacity = '1.0'; editNeg.title = ''; }
            if (genNeg) { genNeg.disabled = false; genNeg.style.opacity = '1.0'; genNeg.title = ''; }
            if (targetModelSel) targetModelSel.value = 'stable-diffusion';
        }
    }


    const sdLoadBtn = document.getElementById('sd-load-btn');
    const sdUnloadBtn = document.getElementById('sd-unload-btn');
    const sdStatus = document.getElementById('sd-status');
    const sdEditBtn = document.getElementById('sd-edit-btn');
    const sdGenBtn = document.getElementById('sd-gen-btn');
    const sdEditStatus = document.getElementById('sd-edit-status');
    const sdGenStatus = document.getElementById('sd-gen-status');
    const sdResults = document.getElementById('sd-results');
    const sdClearResults = document.getElementById('sd-clear-results');

    if (sdClearResults) {
        sdClearResults.addEventListener('click', () => {
            if (sdResults) sdResults.innerHTML = '';
            if (sdEditStatus) sdEditStatus.textContent = 'Results cleared.';
        });
    }

    if (sdLoadBtn) {
        sdLoadBtn.addEventListener('click', async () => {
            const model = document.getElementById('sd-model-select').value;
            if (!model) { sdStatus.textContent = 'Select a model first.'; return; }
            sdStatus.textContent = `Loading ${model} into Stable Diffusion...`;
            sdLoadBtn.disabled = true;
            try {
                const res = await fetch('/api/sd/load', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ model })
                });
                const data = await res.json();
                if (res.ok) {
                    sdStatus.textContent = `✅ Loaded: ${model}`;
                } else {
                    sdStatus.textContent = `❌ ${data.error || 'Failed to load model'}`;
                }
            } catch (e) {
                sdStatus.textContent = `❌ ${e.message}`;
            } finally {
                sdLoadBtn.disabled = false;
            }
        });
    }

    if (sdUnloadBtn) {
        sdUnloadBtn.addEventListener('click', async () => {
            sdStatus.textContent = 'Unloading SD model...';
            try {
                const res = await fetch('/api/sd/unload', { method: 'POST' });
                const data = await res.json();
                sdStatus.textContent = data.status === 'success' ? '⏏️ Unloaded.' : `❌ ${data.error || 'Failed'}`;
            } catch (e) {
                sdStatus.textContent = `❌ ${e.message}`;
            }
        });
    }

    // Helper to render SD result cards with Download & Send to Canvas
    function renderSDResultCard(item, container, filePrefix = 'result') {
        if (item.b64_json) {
            const card = document.createElement('div');
            card.style.display = 'inline-block';
            card.style.margin = '6px';
            card.style.textAlign = 'center';
            card.style.background = '#0f172a';
            card.style.padding = '8px';
            card.style.borderRadius = '8px';
            card.style.border = '1px solid var(--border-color)';

            const img = document.createElement('img');
            img.src = 'data:image/png;base64,' + item.b64_json;
            img.style.maxWidth = '320px';
            img.style.borderRadius = '6px';
            img.style.border = '1px solid var(--border-color)';
            card.appendChild(img);

            const btnBox = document.createElement('div');
            btnBox.style.display = 'flex';
            btnBox.style.gap = '0.5rem';
            btnBox.style.justifyContent = 'center';
            btnBox.style.marginTop = '6px';

            const dl = document.createElement('a');
            dl.textContent = '⬇ Download PNG';
            dl.href = '#';
            dl.title = 'Download this image';
            dl.style.fontSize = '0.75rem';
            dl.style.color = 'var(--color-primary)';
            dl.style.textDecoration = 'none';
            dl.style.cursor = 'pointer';
            dl.addEventListener('click', (ev) => {
                ev.preventDefault();
                const bin = atob(item.b64_json);
                const bytes = new Uint8Array(bin.length);
                for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
                const blob = new Blob([bytes], { type: 'image/png' });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `${filePrefix}_${Date.now()}.png`;
                document.body.appendChild(a);
                a.click();
                a.remove();
                URL.revokeObjectURL(url);
            });
            btnBox.appendChild(dl);

            const canvasBtn = document.createElement('a');
            canvasBtn.textContent = '✏️ Text Canvas';
            canvasBtn.href = '#';
            canvasBtn.title = 'Send image to interactive text canvas studio';
            canvasBtn.style.fontSize = '0.75rem';
            canvasBtn.style.color = '#38bdf8';
            canvasBtn.style.textDecoration = 'none';
            canvasBtn.style.cursor = 'pointer';
            canvasBtn.addEventListener('click', (ev) => {
                ev.preventDefault();
                loadB64IntoCanvas(item.b64_json);
            });
            btnBox.appendChild(canvasBtn);

            card.appendChild(btnBox);
            container.appendChild(card);
        } else if (item.url) {
            const a = document.createElement('a');
            a.href = item.url;
            a.textContent = item.url;
            a.target = '_blank';
            container.appendChild(a);
        }
    }

    // Image Studio Mode Tabs & Enhanced Presets / Canvas Logic
    const modeTabs = {
        gen: document.getElementById('sd-mode-tab-gen'),
        flyer: document.getElementById('sd-mode-tab-flyer'),
        photo: document.getElementById('sd-mode-tab-photo'),
        canvas: document.getElementById('sd-mode-tab-canvas'),
        ocr: document.getElementById('sd-mode-tab-ocr'),
        promptgen: document.getElementById('sd-mode-tab-promptgen')
    };

    const modePanels = {
        gen: document.getElementById('sd-panel-gen'),
        flyer: document.getElementById('sd-panel-flyer'),
        photo: document.getElementById('sd-panel-photo'),
        canvas: document.getElementById('sd-panel-canvas'),
        ocr: document.getElementById('sd-panel-ocr'),
        promptgen: document.getElementById('sd-panel-promptgen')
    };

    function switchSDMode(modeKey) {
        Object.keys(modeTabs).forEach(k => {
            if (modeTabs[k]) {
                if (k === modeKey) {
                    modeTabs[k].classList.add('active');
                    modeTabs[k].style.background = 'var(--color-primary)';
                    modeTabs[k].style.color = 'white';
                } else {
                    modeTabs[k].classList.remove('active');
                    modeTabs[k].style.background = 'transparent';
                    modeTabs[k].style.color = 'var(--text-muted)';
                }
            }
            if (modePanels[k]) {
                if (k === modeKey) {
                    modePanels[k].classList.remove('d-none');
                } else {
                    modePanels[k].classList.add('d-none');
                }
            }
        });
    }

    if (modeTabs.gen) modeTabs.gen.addEventListener('click', () => switchSDMode('gen'));
    if (modeTabs.flyer) modeTabs.flyer.addEventListener('click', () => switchSDMode('flyer'));
    if (modeTabs.photo) modeTabs.photo.addEventListener('click', () => switchSDMode('photo'));
    if (modeTabs.canvas) modeTabs.canvas.addEventListener('click', () => switchSDMode('canvas'));
    if (modeTabs.ocr) modeTabs.ocr.addEventListener('click', () => switchSDMode('ocr'));
    if (modeTabs.promptgen) modeTabs.promptgen.addEventListener('click', () => switchSDMode('promptgen'));

    // --- OCR Document Extractor Handlers ---
    const ocrDropzone = document.getElementById('sd-ocr-dropzone');
    const ocrFileInput = document.getElementById('sd-ocr-file');
    const ocrEmptyState = document.getElementById('sd-ocr-dropzone-empty');
    const ocrPreviewState = document.getElementById('sd-ocr-dropzone-preview');
    const ocrPreviewImg = document.getElementById('sd-ocr-preview-img');
    const ocrPreviewName = document.getElementById('sd-ocr-preview-name');
    const ocrPreviewInfo = document.getElementById('sd-ocr-preview-info');
    const ocrRemoveBtn = document.getElementById('sd-ocr-remove-btn');
    const ocrRunBtn = document.getElementById('sd-ocr-run-btn');
    const ocrStatus = document.getElementById('sd-ocr-status');
    const ocrResultsContainer = document.getElementById('sd-ocr-results-container');
    const ocrResHeadline = document.getElementById('sd-ocr-res-headline');
    const ocrResSubtext = document.getElementById('sd-ocr-res-subtext');
    const ocrResBadge = document.getElementById('sd-ocr-res-badge');
    const ocrResFull = document.getElementById('sd-ocr-res-full');
    const ocrTransferBtn = document.getElementById('sd-ocr-transfer-btn');

    let currentOcrFile = null;

    if (ocrDropzone && ocrFileInput) {
        ocrDropzone.addEventListener('click', (e) => {
            if (e.target !== ocrRemoveBtn) ocrFileInput.click();
        });

        ocrFileInput.addEventListener('change', (e) => {
            if (e.target.files && e.target.files[0]) {
                handleOcrFileSelected(e.target.files[0]);
            }
        });

        ocrDropzone.addEventListener('dragover', (e) => {
            e.preventDefault();
            ocrDropzone.style.borderColor = '#a855f7';
        });

        ocrDropzone.addEventListener('dragleave', () => {
            ocrDropzone.style.borderColor = 'rgba(168, 85, 247, 0.4)';
        });

        ocrDropzone.addEventListener('drop', (e) => {
            e.preventDefault();
            ocrDropzone.style.borderColor = 'rgba(168, 85, 247, 0.4)';
            if (e.dataTransfer.files && e.dataTransfer.files[0]) {
                handleOcrFileSelected(e.dataTransfer.files[0]);
            }
        });
    }

    function handleOcrFileSelected(file) {
        currentOcrFile = file;
        if (ocrPreviewName) ocrPreviewName.textContent = file.name;
        if (ocrPreviewInfo) ocrPreviewInfo.textContent = `${(file.size / 1024).toFixed(1)} KB • Ready for OCR`;

        if (file.type.startsWith('image/')) {
            const reader = new FileReader();
            reader.onload = (e) => {
                if (ocrPreviewImg) {
                    ocrPreviewImg.src = e.target.result;
                    ocrPreviewImg.classList.remove('d-none');
                }
            };
            reader.readAsDataURL(file);
        } else {
            if (ocrPreviewImg) ocrPreviewImg.classList.add('d-none');
        }

        if (ocrEmptyState) ocrEmptyState.classList.add('d-none');
        if (ocrPreviewState) ocrPreviewState.classList.remove('d-none');
    }

    if (ocrRemoveBtn) {
        ocrRemoveBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            currentOcrFile = null;
            if (ocrFileInput) ocrFileInput.value = '';
            if (ocrEmptyState) ocrEmptyState.classList.remove('d-none');
            if (ocrPreviewState) ocrPreviewState.classList.add('d-none');
            if (ocrResultsContainer) ocrResultsContainer.classList.add('d-none');
        });
    }

    if (ocrRunBtn) {
        ocrRunBtn.addEventListener('click', async () => {
            if (!currentOcrFile) {
                showToast('Please upload an image or PDF file first.', 'info');
                return;
            }

            ocrRunBtn.disabled = true;
            if (ocrStatus) ocrStatus.textContent = '⏳ Extracting text & document structure via Qwen2.5-VL...';

            const formData = new FormData();
            formData.append('file', currentOcrFile);

            try {
                const resp = await fetch('/api/vision/ocr', {
                    method: 'POST',
                    body: formData
                });
                const data = await resp.json();

                if (resp.ok && data.status === 'success') {
                    const res = data.ocr_result || {};
                    if (ocrResHeadline) ocrResHeadline.value = res.headline || '';
                    if (ocrResSubtext) ocrResSubtext.value = res.subtext || '';
                    if (ocrResBadge) ocrResBadge.value = res.badge || '';
                    if (ocrResFull) ocrResFull.value = res.full_text || data.raw_response || '';

                    if (ocrResultsContainer) ocrResultsContainer.classList.remove('d-none');
                    if (ocrStatus) ocrStatus.textContent = '✅ Text & Layout extracted successfully!';
                } else {
                    if (ocrStatus) ocrStatus.textContent = `❌ OCR failed: ${data.error || 'Unknown error'}`;
                }
            } catch (err) {
                if (ocrStatus) ocrStatus.textContent = `❌ OCR request error: ${err.message}`;
            } finally {
                ocrRunBtn.disabled = false;
            }
        });
    }

    if (ocrTransferBtn) {
        ocrTransferBtn.addEventListener('click', () => {
            const headline = ocrResHeadline ? ocrResHeadline.value : '';
            const subtext = ocrResSubtext ? ocrResSubtext.value : '';
            const badge = ocrResBadge ? ocrResBadge.value : '';

            const flyerHeadline = document.getElementById('sd-flyer-headline');
            const flyerSubtext = document.getElementById('sd-flyer-subtext');
            const flyerBadge = document.getElementById('sd-flyer-badge');

            if (flyerHeadline && headline) flyerHeadline.value = headline;
            if (flyerSubtext && subtext) flyerSubtext.value = subtext;
            if (flyerBadge && badge) flyerBadge.value = badge;

            switchSDMode('flyer');
            if (typeof updateFlyerPromptPreview === 'function') updateFlyerPromptPreview();
            showToast('✅ Extracted text transferred to Flyer Creator!', 'success');
        });
    }

    // Image-to-Prompt Assistant Logic
    let currentPromptgenFile = null;
    let synthesizedMasterPrompt = '';
    let synthesizedSuggestedStrength = 0.55;
    let synthesizedSuggestedNegative = '';

    const promptgenDropzone = document.getElementById('sd-promptgen-dropzone');
    const promptgenFileInput = document.getElementById('sd-promptgen-file-input');
    const promptgenEmptyState = document.getElementById('sd-promptgen-empty-state');
    const promptgenPreviewState = document.getElementById('sd-promptgen-preview-state');
    const promptgenPreviewImg = document.getElementById('sd-promptgen-preview-img');
    const promptgenPreviewName = document.getElementById('sd-promptgen-preview-name');

    const promptgenAnalyzeBtn = document.getElementById('sd-promptgen-analyze-btn');
    const promptgenDescTextarea = document.getElementById('sd-promptgen-desc');
    const promptgenChangesTextarea = document.getElementById('sd-promptgen-changes');
    const promptgenPresetSelect = document.getElementById('sd-promptgen-preset');
    const promptgenSynthBtn = document.getElementById('sd-promptgen-synth-btn');
    const promptgenStatus = document.getElementById('sd-promptgen-status');
    const promptgenResultPrompt = document.getElementById('sd-promptgen-result-prompt');
    const promptgenSendPhotoBtn = document.getElementById('sd-promptgen-send-photo-btn');

    if (promptgenDropzone && promptgenFileInput) {
        promptgenDropzone.addEventListener('click', () => promptgenFileInput.click());

        promptgenFileInput.addEventListener('change', (e) => {
            if (e.target.files && e.target.files[0]) {
                handlePromptgenFileSelected(e.target.files[0]);
            }
        });

        promptgenDropzone.addEventListener('dragover', (e) => {
            e.preventDefault();
            promptgenDropzone.style.borderColor = '#a855f7';
        });

        promptgenDropzone.addEventListener('dragleave', () => {
            promptgenDropzone.style.borderColor = 'rgba(168, 85, 247, 0.4)';
        });

        promptgenDropzone.addEventListener('drop', (e) => {
            e.preventDefault();
            promptgenDropzone.style.borderColor = 'rgba(168, 85, 247, 0.4)';
            if (e.dataTransfer.files && e.dataTransfer.files[0]) {
                handlePromptgenFileSelected(e.dataTransfer.files[0]);
            }
        });
    }

    function handlePromptgenFileSelected(file) {
        currentPromptgenFile = file;
        if (promptgenPreviewName) promptgenPreviewName.textContent = file.name;

        if (file.type.startsWith('image/')) {
            const reader = new FileReader();
            reader.onload = (e) => {
                if (promptgenPreviewImg) {
                    promptgenPreviewImg.src = e.target.result;
                }
            };
            reader.readAsDataURL(file);
        }

        if (promptgenEmptyState) promptgenEmptyState.classList.add('d-none');
        if (promptgenPreviewState) promptgenPreviewState.classList.remove('d-none');
    }

    if (promptgenAnalyzeBtn) {
        promptgenAnalyzeBtn.addEventListener('click', async () => {
            if (!currentPromptgenFile) {
                showToast('Please upload an image first.', 'info');
                return;
            }

            if (promptgenStatus) promptgenStatus.textContent = '⏳ Analyzing scene with Vision AI...';
            promptgenAnalyzeBtn.disabled = true;

            try {
                const formData = new FormData();
                formData.append('image', currentPromptgenFile);

                const visionModel = document.getElementById('sd-promptgen-vision-model')?.value;
                if (!visionModel) {
                    if (promptgenStatus) promptgenStatus.textContent = '❌ Please select a Vision AI model first.';
                    return;
                }
                formData.append('model', visionModel);

                const res = await fetch('/api/vision/describe', {
                    method: 'POST',
                    body: formData
                });

                const data = await res.json();
                if (res.ok && data.image_description) {
                    if (promptgenDescTextarea) promptgenDescTextarea.value = data.image_description;
                    if (promptgenStatus) promptgenStatus.textContent = '✅ Scene analysis complete!';
                } else {
                    if (promptgenStatus) promptgenStatus.textContent = `❌ Analysis error: ${data.error || 'Failed'}`;
                }
            } catch (err) {
                if (promptgenStatus) promptgenStatus.textContent = `❌ Analysis error: ${err.message}`;
            } finally {
                promptgenAnalyzeBtn.disabled = false;
            }
        });
    }

    if (promptgenSynthBtn) {
        promptgenSynthBtn.addEventListener('click', async () => {
            const baseDesc = promptgenDescTextarea ? promptgenDescTextarea.value.trim() : '';
            const changes = promptgenChangesTextarea ? promptgenChangesTextarea.value.trim() : '';
            const preset = promptgenPresetSelect ? promptgenPresetSelect.value : 'Photorealistic Retouch';
            const synthModel = document.getElementById('sd-promptgen-synth-model')?.value;

            if (!baseDesc || !changes) {
                showToast('Please provide both the base image description and desired modifications.', 'info');
                return;
            }

            if (promptgenStatus) promptgenStatus.textContent = '✨ Synthesizing master prompt...';
            promptgenSynthBtn.disabled = true;

            try {
                const targetModel = document.getElementById('sd-promptgen-target-model')?.value || 'qwen-image-edit';
                const preserveFace = document.getElementById('sd-promptgen-preserve-face')?.checked ?? true;

                const payload = {
                    base_description: baseDesc,
                    desired_changes: changes,
                    style_preset: preset,
                    target_image_model: targetModel,
                    preserve_face: preserveFace
                };
                if (!synthModel) {
                    if (promptgenStatus) promptgenStatus.textContent = '❌ Please select a Synthesis model first.';
                    return;
                }
                payload.model = synthModel;

                const res = await fetch('/api/vision/synthesize_edit_prompt', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });

                const data = await res.json();
                if (res.ok && data.master_prompt) {
                    synthesizedMasterPrompt = data.master_prompt;
                    synthesizedSuggestedStrength = data.suggested_strength ?? 0.55;
                    synthesizedSuggestedNegative = data.suggested_negative ?? '';
                    if (promptgenResultPrompt) promptgenResultPrompt.textContent = data.master_prompt;
                    if (promptgenStatus) promptgenStatus.textContent = `✅ Master prompt ready! (strength: ${synthesizedSuggestedStrength})`;
                } else {
                    if (promptgenStatus) promptgenStatus.textContent = `❌ Error: ${data.error || 'Failed'}`;
                }
            } catch (err) {
                if (promptgenStatus) promptgenStatus.textContent = `❌ Error: ${err.message}`;
            } finally {
                promptgenSynthBtn.disabled = false;
            }
        });
    }

    if (promptgenSendPhotoBtn) {
        promptgenSendPhotoBtn.addEventListener('click', () => {
            const masterPrompt = promptgenResultPrompt ? promptgenResultPrompt.textContent.trim() : '';
            if (!masterPrompt || masterPrompt.includes('Synthesized master prompt will appear here')) {
                showToast('Please synthesize a master prompt first.', 'info');
                return;
            }

            // 1. Transfer master edit prompt to Photo Editor prompt input
            const photoPromptElem = document.getElementById('sd-edit-prompt') || document.getElementById('sd-photo-edit-prompt');
            if (photoPromptElem) photoPromptElem.value = masterPrompt;

            // 2. Auto-select "Custom Edit Prompt" preset so it won't be overwritten by defaults
            const photoPresetSel = document.getElementById('sd-photo-preset-select');
            if (photoPresetSel) {
                photoPresetSel.value = 'custom';
            }

            // 2b. Apply suggested strength from synthesis response
            const photoStrengthInput = document.getElementById('sd-edit-strength');
            const photoStrengthVal = document.getElementById('sd-strength-val');
            if (photoStrengthInput && synthesizedSuggestedStrength) {
                photoStrengthInput.value = synthesizedSuggestedStrength;
                if (photoStrengthVal) photoStrengthVal.textContent = synthesizedSuggestedStrength;
                // Highlight the closest strength preset button
                const presetBtns = Array.from(document.querySelectorAll('.sd-strength-preset'));
                let closest = null, closestDist = Infinity;
                presetBtns.forEach(btn => {
                    const dist = Math.abs(parseFloat(btn.dataset.strength) - parseFloat(synthesizedSuggestedStrength));
                    if (dist < closestDist) { closestDist = dist; closest = btn; }
                });
                presetBtns.forEach(btn => {
                    if (btn === closest) {
                        btn.classList.add('active');
                        btn.style.background = '#38bdf8';
                        btn.style.borderColor = '#38bdf8';
                    } else {
                        btn.classList.remove('active');
                        btn.style.background = '#1e293b';
                        btn.style.borderColor = 'var(--border-color)';
                    }
                });
            }

            // 2c. Apply suggested negative prompt from synthesis response
            const photoNegativeInput = document.getElementById('sd-edit-negative');
            if (photoNegativeInput && synthesizedSuggestedNegative) {
                photoNegativeInput.value = synthesizedSuggestedNegative;
            }

            // 3. Transfer uploaded source image file and update preview in Photo Editor Studio
            if (currentPromptgenFile) {
                const photoInput = document.getElementById('sd-edit-image');
                if (photoInput) {
                    try {
                        const dt = new DataTransfer();
                        dt.items.add(currentPromptgenFile);
                        photoInput.files = dt.files;
                    } catch (e) {
                        console.warn('Could not set DataTransfer on photoInput:', e);
                    }
                }
                if (typeof showPhotoPreview === 'function') {
                    showPhotoPreview(currentPromptgenFile);
                }
            }

            // 4. Switch active panel to Photo Editor Studio
            switchSDMode('photo');
        });
    }

    // Flyer Creator Synthesizer Logic
    const flyerPresetSel = document.getElementById('sd-flyer-preset-select');
    const flyerAspectSel = document.getElementById('sd-flyer-aspect-select');
    const flyerHeadline = document.getElementById('sd-flyer-headline');
    const flyerSubtext = document.getElementById('sd-flyer-subtext');
    const flyerBadge = document.getElementById('sd-flyer-badge');
    const flyerVisuals = document.getElementById('sd-flyer-visuals');
    const flyerPromptPreview = document.getElementById('sd-flyer-prompt-preview');
    const flyerGenBtn = document.getElementById('sd-flyer-gen-btn');
    const flyerStatus = document.getElementById('sd-flyer-status');

    const flyerPresets = {
        music_event: {
            name: 'Music & Party Event',
            prompt: 'vibrant music event poster, energetic neon lighting, dynamic background graphic, high contrast layout',
            size: '832x1216',
            visuals: 'neon stage spotlights, crowd silhouette, dark moody atmosphere'
        },
        corporate_business: {
            name: 'Corporate & Business',
            prompt: 'professional corporate business flyer, clean modern typography layout, sleek geometry, dark navy theme',
            size: '832x1216',
            visuals: 'sleek office building, abstract geometric vector shapes, executive background'
        },
        product_sale: {
            name: 'Product Sale & Offer',
            prompt: 'retail promotional flyer, bold sale badge accents, sleek product display pedestal, crisp studio background',
            size: '1024x1024',
            visuals: 'floating promotional podium, golden confetti accents, studio lighting'
        },
        restaurant_menu: {
            name: 'Restaurant & Food Menu',
            prompt: 'gourmet restaurant food poster, delicious culinary styling, elegant menu border, rustic slate background',
            size: '832x1216',
            visuals: 'steaking hot gourmet burger, fresh ingredients, dark wooden tabletop'
        },
        minimalist_modern: {
            name: 'Minimalist Modern',
            prompt: 'minimalist graphic design flyer, high contrast typography space, geometric aesthetic, subtle gradient',
            size: '768x1344',
            visuals: 'abstract clean waves, pastel accent shapes, spacious aesthetic layout'
        }
    };

    function updateFlyerPromptPreview() {
        if (!flyerPromptPreview) return;
        const presetKey = flyerPresetSel ? flyerPresetSel.value : 'music_event';
        const preset = flyerPresets[presetKey] || flyerPresets.music_event;
        const headline = (flyerHeadline ? flyerHeadline.value.trim() : '') || 'EVENT TITLE';
        const subtext = (flyerSubtext ? flyerSubtext.value.trim() : '') || 'DATE & TIME';
        const badge = (flyerBadge ? flyerBadge.value.trim() : '') || 'SPECIAL OFFER';
        const visuals = (flyerVisuals ? flyerVisuals.value.trim() : '') || preset.visuals;

        const synthesized = `flyer graphic design, main title text reading "${headline}", subtext reading "${subtext}", badge tag "${badge}", ${visuals}, ${preset.prompt}, sharp typography, clean professional layout, 8k resolution`;
        flyerPromptPreview.textContent = synthesized;
    }

    if (flyerPresetSel) {
        flyerPresetSel.addEventListener('change', () => {
            const p = flyerPresets[flyerPresetSel.value];
            if (p) {
                if (flyerAspectSel) flyerAspectSel.value = p.size;
                if (flyerVisuals) flyerVisuals.value = p.visuals;
            }
            updateFlyerPromptPreview();
        });
    }

    [flyerHeadline, flyerSubtext, flyerBadge, flyerVisuals].forEach(elem => {
        if (elem) elem.addEventListener('input', updateFlyerPromptPreview);
    });

    updateFlyerPromptPreview();

    if (flyerGenBtn) {
        flyerGenBtn.addEventListener('click', async () => {
            const model = document.getElementById('sd-model-select').value;
            if (!model) { flyerStatus.textContent = '❌ Load an image model first.'; return; }

            const prompt = flyerPromptPreview ? flyerPromptPreview.textContent.trim() : '';
            const size = flyerAspectSel ? flyerAspectSel.value : '832x1216';
            const negative = 'garbled text, distorted letters, bad typography, misspelled text, blurry letters, low contrast, messy composition';

            const payload = {
                model,
                prompt,
                size,
                n: 1,
                negative_prompt: negative,
                steps: 25,
                guidance: 8.0
            };

            const qrEnable = document.getElementById('sd-flyer-qr-enable');
            const qrUrl = document.getElementById('sd-flyer-qr-url');
            const qrPosition = document.getElementById('sd-flyer-qr-position');
            const qrLabel = document.getElementById('sd-flyer-qr-label');

            if (qrEnable && qrEnable.checked && qrUrl && qrUrl.value.trim()) {
                payload.qr_url = qrUrl.value.trim();
                payload.qr_position = qrPosition ? qrPosition.value : 'bottom_right';
                payload.qr_label = qrLabel ? qrLabel.value.trim() : 'SCAN ME';
            }

            flyerStatus.textContent = '🚀 Generating flyer graphic with QR Code embedding...';
            flyerGenBtn.disabled = true;
            try {
                const res = await fetch('/api/sd/generate', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload),
                });
                const data = await res.json();
                if (res.ok && data.data) {
                    sdResults.innerHTML = '';
                    data.data.forEach(item => renderSDResultCard(item, sdResults, 'flyer'));
                    flyerStatus.textContent = '✅ Flyer generated successfully!';
                } else {
                    flyerStatus.textContent = `❌ ${data.error || 'Flyer generation failed'}`;
                }
            } catch (e) {
                flyerStatus.textContent = `❌ ${e.message}`;
            } finally {
                flyerGenBtn.disabled = false;
            }
        });
    }

    // Photo Realism Retouch Controls
    const photoPresetSel = document.getElementById('sd-photo-preset-select');
    const photoStrengthInput = document.getElementById('sd-edit-strength');
    const photoStrengthVal = document.getElementById('sd-strength-val');
    const photoPromptInput = document.getElementById('sd-edit-prompt');
    const photoNegativeInput = document.getElementById('sd-edit-negative');
    const strengthButtons = document.querySelectorAll('.sd-strength-preset');

    const photoPresets = {
        portrait: {
            prompt: '8k RAW photo, portrait photograph of subject, detailed skin texture, natural soft studio lighting, sharp focus, 85mm lens f/1.8',
            negative: 'cgi, 3d render, illustration, smooth plastic skin, oversaturated, distorted features, overprocessed, low quality, noise',
            strength: 0.45
        },
        studio_product: {
            prompt: 'commercial studio product photo, clean directional studio lighting, sharp details, professional color grade, 4k',
            negative: 'blurry, dark, noisy, amateur photo, harsh reflections, low quality, distorted',
            strength: 0.35
        },
        outdoor_retouch: {
            prompt: 'vibrant natural outdoor photo, golden hour sunlight, sharp detail, high dynamic range, photorealistic',
            negative: 'overexposed, muddy, low contrast, heavy grain, cgi, unnatural colors',
            strength: 0.40
        },
        tone_color_grade: {
            prompt: 'cinematic photo color grading, balanced lighting, deep contrast, natural skin tones, professional photography',
            negative: 'flat color, oversaturated, washed out, noisy, artifact',
            strength: 0.30
        },
        restore_polish: {
            prompt: 'clean sharp photograph, noise reduction, crisp focus, enhanced clarity, realistic texture',
            negative: 'blurry, pixelated, artifact, low resolution, noise, distortion',
            strength: 0.25
        }
    };

    if (photoPresetSel) {
        photoPresetSel.addEventListener('change', () => {
            const p = photoPresets[photoPresetSel.value];
            if (p) {
                if (photoPromptInput) photoPromptInput.value = p.prompt;
                if (photoNegativeInput) photoNegativeInput.value = p.negative;
                if (photoStrengthInput) {
                    photoStrengthInput.value = p.strength;
                    if (photoStrengthVal) photoStrengthVal.textContent = p.strength;
                }
                strengthButtons.forEach(btn => {
                    if (parseFloat(btn.dataset.strength) === p.strength) btn.classList.add('active');
                    else btn.classList.remove('active');
                });
            }
        });
    }

    // Quick Inspiration Chips & Dimension Presets
    document.querySelectorAll('.sd-quick-chip').forEach(chip => {
        chip.addEventListener('click', () => {
            const promptInput = document.getElementById('sd-gen-prompt');
            if (promptInput && chip.dataset.prompt) {
                promptInput.value = chip.dataset.prompt;
                promptInput.focus();
            }
        });
    });

    const genSizePreset = document.getElementById('sd-gen-size-preset');
    const genSizeInput = document.getElementById('sd-gen-size');
    if (genSizePreset && genSizeInput) {
        genSizePreset.addEventListener('change', () => {
            genSizeInput.value = genSizePreset.value;
        });
    }

    // Drag & Drop Photo Upload Zone
    const photoDropzone = document.getElementById('sd-photo-dropzone');
    const photoInput = document.getElementById('sd-edit-image');
    const photoEmpty = document.getElementById('sd-photo-dropzone-empty');
    const photoPreview = document.getElementById('sd-photo-dropzone-preview');
    const photoImg = document.getElementById('sd-photo-preview-img');
    const photoName = document.getElementById('sd-photo-preview-name');
    const photoRemoveBtn = document.getElementById('sd-photo-remove-btn');

    if (photoDropzone && photoInput) {
        photoDropzone.addEventListener('click', (e) => {
            if (e.target !== photoRemoveBtn) photoInput.click();
        });
        photoDropzone.addEventListener('dragover', (e) => {
            e.preventDefault();
            photoDropzone.style.borderColor = '#38bdf8';
            photoDropzone.style.background = 'rgba(56, 189, 248, 0.1)';
        });
        photoDropzone.addEventListener('dragleave', () => {
            photoDropzone.style.borderColor = 'rgba(56, 189, 248, 0.4)';
            photoDropzone.style.background = '#090d16';
        });
        photoDropzone.addEventListener('drop', (e) => {
            e.preventDefault();
            photoDropzone.style.borderColor = 'rgba(56, 189, 248, 0.4)';
            photoDropzone.style.background = '#090d16';
            if (e.dataTransfer.files && e.dataTransfer.files[0]) {
                photoInput.files = e.dataTransfer.files;
                showPhotoPreview(e.dataTransfer.files[0]);
            }
        });
        photoInput.addEventListener('change', () => {
            if (photoInput.files && photoInput.files[0]) {
                showPhotoPreview(photoInput.files[0]);
            }
        });
        if (photoRemoveBtn) {
            photoRemoveBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                photoInput.value = '';
                if (photoPreview) photoPreview.classList.add('d-none');
                if (photoEmpty) photoEmpty.classList.remove('d-none');
            });
        }
    }

    function showPhotoPreview(file) {
        if (!file || !photoImg || !photoName) return;
        photoName.textContent = file.name;
        const reader = new FileReader();
        reader.onload = (e) => {
            photoImg.src = e.target.result;
            if (photoEmpty) photoEmpty.classList.add('d-none');
            if (photoPreview) photoPreview.classList.remove('d-none');
        };
        reader.readAsDataURL(file);
    }

    const strengthGuide = document.getElementById('sd-strength-guide');
    const strengthGuides = {
        0.25: 'Subtle Retouch (Enhance skin & remove minor blemishes)',
        0.45: 'Balanced Edit (Modify subject details, outfit, studio lighting)',
        0.65: 'Medium Style Refresh (Change background, color style, theme)',
        0.85: 'Heavy Reimagine (Transform photo composition & style)'
    };

    function updateStrengthLabel(val) {
        if (photoStrengthVal) photoStrengthVal.textContent = val;
        if (strengthGuide) {
            const v = parseFloat(val);
            let text = 'Custom Strength';
            if (v <= 0.3) text = strengthGuides[0.25];
            else if (v <= 0.55) text = strengthGuides[0.45];
            else if (v <= 0.75) text = strengthGuides[0.65];
            else text = strengthGuides[0.85];
            strengthGuide.textContent = text;
        }
    }

    if (photoStrengthInput) {
        photoStrengthInput.addEventListener('input', () => {
            updateStrengthLabel(photoStrengthInput.value);
        });
    }

    strengthButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const val = parseFloat(btn.dataset.strength);
            if (photoStrengthInput) photoStrengthInput.value = val;
            updateStrengthLabel(val);
            strengthButtons.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
        });
    });

    // Interactive Canvas Text & Typography Studio Engine
    const canvasElem = document.getElementById('sd-text-canvas');
    const canvasHint = document.getElementById('sd-canvas-empty-hint');
    const canvasUpload = document.getElementById('sd-canvas-upload');
    const canvasUploadTriggerBtn = document.getElementById('sd-canvas-upload-trigger-btn');
    const canvasText1 = document.getElementById('sd-canvas-text1');
    const canvasFont1 = document.getElementById('sd-canvas-font1');
    const canvasColor1 = document.getElementById('sd-canvas-color1');
    const canvasText2 = document.getElementById('sd-canvas-text2');
    const canvasText3 = document.getElementById('sd-canvas-text3');
    const canvasBadgeBg = document.getElementById('sd-canvas-badge-bg');
    const canvasLayout = document.getElementById('sd-canvas-layout');
    const canvasExportBtn = document.getElementById('sd-canvas-export-btn');

    let currentCanvasImage = null;

    if (canvasUploadTriggerBtn && canvasUpload) {
        canvasUploadTriggerBtn.addEventListener('click', () => canvasUpload.click());
    }

    if (canvasUpload) {
        canvasUpload.addEventListener('change', (e) => {
            if (e.target.files && e.target.files[0]) {
                const reader = new FileReader();
                reader.onload = (ev) => {
                    const img = new Image();
                    img.onload = () => {
                        currentCanvasImage = img;
                        renderCanvasTextOverlay();
                        switchSDMode('canvas');
                    };
                    img.src = ev.target.result;
                };
                reader.readAsDataURL(e.target.files[0]);
            }
        });
    }

    function loadB64IntoCanvas(b64Data) {
        const img = new Image();
        img.onload = () => {
            currentCanvasImage = img;
            renderCanvasTextOverlay();
            switchSDMode('canvas');
        };
        img.src = 'data:image/png;base64,' + b64Data;
    }

    function renderCanvasTextOverlay() {
        if (!canvasElem) return;
        const ctx = canvasElem.getContext('2d');
        if (!currentCanvasImage) {
            if (canvasHint) canvasHint.style.display = 'block';
            canvasElem.style.display = 'none';
            return;
        }

        if (canvasHint) canvasHint.style.display = 'none';
        canvasElem.style.display = 'block';

        canvasElem.width = currentCanvasImage.width || 1024;
        canvasElem.height = currentCanvasImage.height || 1024;

        ctx.clearRect(0, 0, canvasElem.width, canvasElem.height);
        ctx.drawImage(currentCanvasImage, 0, 0, canvasElem.width, canvasElem.height);

        const w = canvasElem.width;
        const h = canvasElem.height;

        const t1 = canvasText1 ? canvasText1.value.trim() : '';
        const t2 = canvasText2 ? canvasText2.value.trim() : '';
        const t3 = canvasText3 ? canvasText3.value.trim() : '';
        const font1 = canvasFont1 ? canvasFont1.value : 'Impact, sans-serif';
        const color1 = canvasColor1 ? canvasColor1.value : '#ffffff';
        const badgeBg = canvasBadgeBg ? canvasBadgeBg.value : '#3b82f6';
        const layout = canvasLayout ? canvasLayout.value : 'top';

        let startY = h * 0.15;
        if (layout === 'center') startY = h * 0.40;
        else if (layout === 'bottom') startY = h * 0.72;

        ctx.textAlign = 'center';
        ctx.shadowColor = 'rgba(0, 0, 0, 0.85)';
        ctx.shadowBlur = 12;
        ctx.shadowOffsetX = 2;
        ctx.shadowOffsetY = 4;

        // Headline
        if (t1) {
            const size1 = Math.round(w * 0.058);
            ctx.font = `900 ${size1}px ${font1}`;
            ctx.fillStyle = color1;
            ctx.fillText(t1, w / 2, startY);
            startY += size1 * 1.15;
        }

        // Subtitle
        if (t2) {
            const size2 = Math.round(w * 0.032);
            ctx.font = `600 ${size2}px system-ui, sans-serif`;
            ctx.fillStyle = '#e2e8f0';
            ctx.fillText(t2, w / 2, startY);
            startY += size2 * 1.4;
        }

        // Badge Tag (Pill)
        if (t3) {
            const size3 = Math.round(w * 0.028);
            ctx.font = `700 ${size3}px system-ui, sans-serif`;
            const textMetrics = ctx.measureText(t3);
            const padX = size3 * 1.2;
            const padY = size3 * 0.5;
            const bw = textMetrics.width + padX * 2;
            const bh = size3 + padY * 2;
            const bx = (w - bw) / 2;
            const by = startY - bh + size3 * 0.2;

            ctx.shadowColor = 'transparent';
            ctx.fillStyle = badgeBg;
            ctx.beginPath();
            if (ctx.roundRect) ctx.roundRect(bx, by, bw, bh, bh / 2);
            else ctx.rect(bx, by, bw, bh);
            ctx.fill();

            ctx.shadowColor = 'rgba(0,0,0,0.5)';
            ctx.shadowBlur = 4;
            ctx.fillStyle = '#ffffff';
            ctx.fillText(t3, w / 2, by + bh / 2 + size3 * 0.35);
        }
    }

    [canvasText1, canvasFont1, canvasColor1, canvasText2, canvasText3, canvasBadgeBg, canvasLayout].forEach(el => {
        if (el) el.addEventListener('input', renderCanvasTextOverlay);
    });

    if (canvasExportBtn) {
        canvasExportBtn.addEventListener('click', () => {
            if (!canvasElem || !currentCanvasImage) return;
            const link = document.createElement('a');
            link.download = 'flyer_graphic_export.png';
            link.href = canvasElem.toDataURL('image/png');
            document.body.appendChild(link);
            link.click();
            link.remove();
        });
    }

    if (sdEditBtn) {
        sdEditBtn.addEventListener('click', async () => {
            const model = document.getElementById('sd-model-select').value;
            const fileInput = document.getElementById('sd-edit-image');
            const prompt = document.getElementById('sd-edit-prompt').value.trim();
            const size = document.getElementById('sd-edit-size').value.trim();
            const n = document.getElementById('sd-edit-n').value;
            const strength = document.getElementById('sd-edit-strength').value;
            const negative = document.getElementById('sd-edit-negative').value.trim();
            if (!model) { sdEditStatus.textContent = 'Load an image model first.'; return; }
            if (!fileInput.files || fileInput.files.length === 0) { sdEditStatus.textContent = 'Choose a source image.'; return; }
            if (!prompt) { sdEditStatus.textContent = 'Enter an edit prompt.'; return; }

            const fullPrompt = `${prompt}<sd_cpp_extra_args>{"strength": ${parseFloat(strength) || 0.45}, "negative_prompt": "${negative.replace(/"/g, '\\"')}"}</sd_cpp_extra_args>`;

            const fd = new FormData();
            fd.append('model', model);
            fd.append('prompt', fullPrompt);
            fd.append('size', size);
            fd.append('n', n);
            fd.append('image', fileInput.files[0]);

            sdEditStatus.textContent = 'Editing image (this can take a while)...';
            sdEditBtn.disabled = true;
            try {
                const res = await fetch('/api/sd/edit', { method: 'POST', body: fd });
                const data = await res.json();
                if (res.ok && data.data) {
                    sdResults.innerHTML = '';
                    data.data.forEach(item => renderSDResultCard(item, sdResults, 'photo_edit'));
                    sdEditStatus.textContent = `✅ Edited ${data.data.length} image(s).`;
                } else {
                    sdEditStatus.textContent = `❌ ${data.error || 'Edit failed'}`;
                }
            } catch (e) {
                sdEditStatus.textContent = `❌ ${e.message}`;
            } finally {
                sdEditBtn.disabled = false;
            }
        });
    }

    if (sdGenBtn) {
        sdGenBtn.addEventListener('click', async () => {
            const model = document.getElementById('sd-model-select').value;
            const prompt = document.getElementById('sd-gen-prompt').value.trim();
            const size = document.getElementById('sd-gen-size').value.trim();
            const n = document.getElementById('sd-gen-n').value;
            const negative = document.getElementById('sd-gen-negative').value.trim();
            const steps = document.getElementById('sd-gen-steps').value;
            const guidance = document.getElementById('sd-gen-guidance').value;
            let seed = document.getElementById('sd-gen-seed').value;
            if (!model) { sdGenStatus.textContent = 'Load an image model first.'; return; }
            if (!prompt) { sdGenStatus.textContent = 'Enter a prompt.'; return; }
            if (seed === '' || seed === '-1') seed = -1;
            else seed = parseInt(seed, 10);

            const payload = {
                model,
                prompt,
                size,
                n: parseInt(n, 10) || 1,
            };
            if (negative) payload.negative_prompt = negative;
            if (steps && parseInt(steps, 10) > 0) payload.steps = parseInt(steps, 10);
            if (guidance && parseFloat(guidance) > 0) payload.guidance = parseFloat(guidance);
            if (seed >= 0) payload.seed = seed;

            sdGenStatus.textContent = 'Generating image (this can take a while)...';
            sdGenBtn.disabled = true;
            try {
                const res = await fetch('/api/sd/generate', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload),
                });
                const data = await res.json();
                if (res.ok && data.data) {
                    sdResults.innerHTML = '';
                    data.data.forEach(item => renderSDResultCard(item, sdResults, 'generation'));
                    sdGenStatus.textContent = `✅ Generated ${data.data.length} image(s).`;
                } else {
                    sdGenStatus.textContent = `❌ ${data.error || 'Generation failed'}`;
                }
            } catch (e) {
                sdGenStatus.textContent = `❌ ${e.message}`;
            } finally {
                sdGenBtn.disabled = false;
            }
        });
    }


    let docsMenuSetup = false;
    function setupDocsMenuHandlers() {
        if (docsMenuSetup) return;
        docsMenuSetup = true;

        document.querySelectorAll('.doc-menu-item').forEach(item => {
            item.addEventListener('click', () => {
                // Deactivate all menu items
                document.querySelectorAll('.doc-menu-item').forEach(mi => {
                    mi.classList.remove('active');
                    mi.style.borderColor = 'var(--border-color)';
                });
                // Activate selected menu item
                item.classList.add('active');
                item.style.borderColor = 'var(--color-primary)';

                // Hide all details panes
                document.querySelectorAll('.doc-detail-pane').forEach(pane => {
                    pane.classList.add('d-none');
                });
                // Show matching details pane
                const targetId = item.getAttribute('data-target');
                const targetPane = document.getElementById(targetId);
                if (targetPane) {
                    targetPane.classList.remove('d-none');
                }
            });
        });
    }

    // Setup Charts
    function initCharts() {
        const ctxTps = document.getElementById('tps-chart').getContext('2d');
        const ctxTtft = document.getElementById('ttft-chart').getContext('2d');
        const ctxCategory = document.getElementById('category-chart').getContext('2d');
        
        const ctxSharedLatency = document.getElementById('shared-latency-chart').getContext('2d');
        const ctxSharedAst = document.getElementById('shared-ast-chart').getContext('2d');
        
        const ctxMemoryCreep = document.getElementById('memory-creep-chart').getContext('2d');

        // Common Chart.js styling overrides
        Chart.defaults.color = '#94a3b8';
        Chart.defaults.borderColor = 'rgba(255, 255, 255, 0.06)';
        Chart.defaults.font.family = "'Inter', sans-serif";

        const perModelTooltip = {
            callbacks: {
                title: function(context) {
                    // For per-model datasets, show the original full model name
                    return context[0].dataset.originalLabel || context[0].dataset.label || '';
                },
                label: function(context) {
                    return ` ${context.parsed.y}`;
                }
            }
        };

        tpsChart = new Chart(ctxTps, {
            type: 'bar',
            data: {
                labels: [''],
                datasets: []
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { 
                    legend: { 
                        display: true,
                        position: 'top',
                        labels: { boxWidth: 10, padding: 8, font: { size: 11 } }
                    },
                    tooltip: perModelTooltip
                },
                scales: {
                    x: { ticks: { display: false }, grid: { display: false } },
                    y: { beginAtZero: true, title: { display: true, text: 'Tokens/s' } }
                }
            }
        });

        ttftChart = new Chart(ctxTtft, {
            type: 'bar',
            data: {
                labels: [''],
                datasets: []
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { 
                    legend: { 
                        display: true,
                        position: 'top',
                        labels: { boxWidth: 10, padding: 8, font: { size: 11 } }
                    },
                    tooltip: perModelTooltip
                },
                scales: {
                    x: { ticks: { display: false }, grid: { display: false } },
                    y: { beginAtZero: true, title: { display: true, text: 'Time (ms)' } }
                }
            }
        });

        categoryChart = new Chart(ctxCategory, {
            type: 'bar',
            data: {
                labels: ['Coding', 'Reasoning', 'Instruction', 'Creative', 'Home Automation'],
                datasets: []
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { position: 'top', labels: { boxWidth: 12, padding: 10 } },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                let label = context.dataset.originalLabel || context.dataset.label || '';
                                if (label) {
                                    label += ': ';
                                }
                                if (context.parsed.y !== null) {
                                    label += context.parsed.y + '%';
                                }
                                return label;
                            }
                        }
                    }
                },
                scales: { y: { beginAtZero: true, max: 100, title: { display: true, text: 'Success Rate (%)' } } }
            }
        });

        // SharedLLM latency comparison chart - uses legend instead of X-axis labels
        // to avoid diagonal/overflowing model name text on the axis
        sharedLatencyChart = new Chart(ctxSharedLatency, {
            type: 'bar',
            data: {
                labels: [],
                datasets: [
                    { label: 'FastPath (Intent)', data: [], backgroundColor: 'rgba(139, 92, 246, 0.7)', borderRadius: 4 },
                    { label: 'Librarian (Tools)', data: [], backgroundColor: 'rgba(6, 182, 212, 0.7)', borderRadius: 4 },
                    { label: 'Raven (Coding)', data: [], backgroundColor: 'rgba(16, 185, 129, 0.7)', borderRadius: 4 },
                    { label: 'Troubleshoot & Patch', data: [], backgroundColor: 'rgba(245, 158, 11, 0.7)', borderRadius: 4 },
                    { label: 'Media & Docs', data: [], backgroundColor: 'rgba(236, 72, 153, 0.7)', borderRadius: 4 },
                    { label: 'Composite Chaining', data: [], backgroundColor: 'rgba(59, 130, 246, 0.7)', borderRadius: 4 }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { 
                    legend: { position: 'top', labels: { boxWidth: 12, padding: 8 } },
                    tooltip: {
                        callbacks: {
                            title: function(context) {
                                const index = context[0].dataIndex;
                                const originalLabels = context[0].chart.data.originalLabels;
                                return (originalLabels && originalLabels[index]) ? originalLabels[index] : context[0].label;
                            },
                            label: function(context) {
                                let label = context.dataset.label || '';
                                if (label) label += ': ';
                                if (context.parsed.y !== null) label += context.parsed.y.toFixed(2) + 's';
                                return label;
                            }
                        }
                    }
                },
                scales: {
                    // Hide X labels - models are in the tooltip title and legend
                    x: {
                        ticks: { display: false },
                        grid: { display: false }
                    },
                    y: { beginAtZero: true, title: { display: true, text: 'Latency (seconds)' } }
                }
            }
        });

        // SharedLLM AST Compliance & Execution Breakdown chart
        sharedAstChart = new Chart(ctxSharedAst, {
            type: 'bar',
            data: {
                labels: ['Syntax / Parsing', 'Schema / Structure', 'Contracts / Logic', 'Overall Pass Rate'],
                datasets: []
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { position: 'top', labels: { boxWidth: 12, padding: 8 } } },
                scales: { y: { beginAtZero: true, max: 100, title: { display: true, text: 'Pass Rate (%)' } } }
            }
        });

        // Memory Creep and OOM monitoring line chart
        memoryCreepChart = new Chart(ctxMemoryCreep, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'System RAM (%)',
                        data: [],
                        borderColor: '#06b6d4',
                        backgroundColor: 'rgba(6, 182, 212, 0.05)',
                        borderWidth: 2,
                        tension: 0.2,
                        fill: true
                    },
                    {
                        label: 'GPU VRAM (%)',
                        data: [],
                        borderColor: '#10b981',
                        backgroundColor: 'rgba(16, 185, 129, 0.05)',
                        borderWidth: 2,
                        tension: 0.2,
                        fill: true
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { position: 'top', labels: { boxWidth: 12 } }
                },
                scales: {
                    x: {
                        grid: { display: false },
                        title: { display: true, text: 'Time' },
                        ticks: { maxRotation: 0, autoSkip: true, maxTicksLimit: 8 }
                    },
                    y: {
                        beginAtZero: true,
                        max: 100,
                        title: { display: true, text: 'Memory Usage (%)' }
                    }
                }
            }
        });
    }

    // Modal Control - supports thinking models with <think>...</think> blocks
    const modalThinking = document.getElementById('modal-thinking');
    const modalThinkingSection = document.getElementById('modal-thinking-section');

    function openModal(prompt, rawResponse, thinking) {
        modalPrompt.textContent = prompt || 'No prompt recorded.';

        // Parse out <think>...</think> block if present, or use separate thinking field
        // (when --reasoning-format deepseek routes thoughts to message.thinking).
        let thinkingContent = '';
        let actualResponse = '';
        let thinkingBlockFound = false;

        if (thinking) {
            thinkingContent = String(thinking).trim();
            thinkingBlockFound = !!thinkingContent;
            actualResponse = rawResponse ? String(rawResponse).trim() : '';
        } else if (rawResponse) {
            try {
                const thinkMatch = rawResponse.match(/<think>([\s\S]*?)<\/think>/i);
                if (thinkMatch) {
                    thinkingBlockFound = true;
                    thinkingContent = thinkMatch[1].trim();
                    const remainingContent = rawResponse.replace(/<think>[\s\S]*?<\/think>/i, '');
                    actualResponse = remainingContent.trim();
                } else {
                    actualResponse = rawResponse.trim();
                }
            } catch (error) {
                actualResponse = rawResponse ? rawResponse.trim() : '';
                console.warn('Error parsing thinking block:', error);
            }
        }

        if (thinkingBlockFound) {
            modalThinking.textContent = thinkingContent || '(empty thinking block)';
            modalThinkingSection.style.display = '';
        } else {
            modalThinkingSection.style.display = 'none';
        }

        modalResponse.textContent = actualResponse || '(No response content.)';
        modalOverlay.classList.add('open');
    }

    function closeModal() {
        modalOverlay.classList.remove('open');
    }

    modalClose.addEventListener('click', closeModal);
    modalOverlay.addEventListener('click', (e) => {
        if (e.target === modalOverlay) closeModal();
    });

    // Logging helpers
    function logToTerminal(message, type = 'info', targetTerminal = 'general') {
        const timestamp = new Date().toLocaleTimeString();
        const line = document.createElement('div');
        line.className = `console-line ${type}`;
        
        const timeSpan = document.createElement('span');
        timeSpan.className = 'console-line timestamp';
        timeSpan.textContent = `[${timestamp}]`;
        
        const contentSpan = document.createElement('span');
        contentSpan.textContent = message;
        
        line.appendChild(timeSpan);
        line.appendChild(contentSpan);
        
        const term = targetTerminal === 'shared' ? sharedConsoleTerminal : consoleTerminal;
        term.appendChild(line);
        term.scrollTop = term.scrollHeight;
    }

    // Toggle Direct / Proxy Mode
    modeProxyBtn.addEventListener('click', () => {
        benchmarkMode = 'proxy';
        modeProxyBtn.classList.add('active');
        modeDirectBtn.classList.remove('active');
        logToTerminal("Switched mode to: Proxy (using alpaca-proxy)");
        logToTerminal("Switched mode to: Proxy (using alpaca-proxy)", 'info', 'shared');
    });

    modeDirectBtn.addEventListener('click', () => {
        benchmarkMode = 'direct';
        modeDirectBtn.classList.add('active');
        modeProxyBtn.classList.remove('active');
        logToTerminal("Switched mode to: Direct (using direct llama-servers)");
        logToTerminal("Switched mode to: Direct (using direct llama-servers)", 'info', 'shared');
    });

    selectAllBtn.addEventListener('click', () => {
        const boxes = modelCheckboxes.querySelectorAll('input[type="checkbox"]');
        boxes.forEach(box => box.checked = true);
        logToTerminal("All models selected");
    });

    const selectNewBtn = document.getElementById('btn-select-new');
    if (selectNewBtn) {
        selectNewBtn.addEventListener('click', () => {
            const items = modelCheckboxes.querySelectorAll('.checkbox-item');
            let selectedCount = 0;
            items.forEach(item => {
                const box = item.querySelector('input[type="checkbox"]');
                if (item.getAttribute('data-status') === 'new') {
                    if (box) { box.checked = true; selectedCount++; }
                } else if (box) {
                    box.checked = false;
                }
            });
            logToTerminal(`Selected ${selectedCount} newly added unbenchmarked model(s)`);
        });
    }

    deselectAllBtn.addEventListener('click', () => {
        const boxes = modelCheckboxes.querySelectorAll('input[type="checkbox"]');
        boxes.forEach(box => box.checked = false);
        logToTerminal("All models deselected");
    });

    // Model Filter Tabs (All, New, Benchmarked)
    function applyModelListFilter(filterType) {
        const items = modelCheckboxes.querySelectorAll('.checkbox-item');
        const sep = modelCheckboxes.querySelector('.online-models-separator');
        
        items.forEach(item => {
            const status = item.getAttribute('data-status') || 'new';
            if (filterType === 'all') {
                item.style.display = 'flex';
            } else if (filterType === 'new') {
                item.style.display = status === 'new' ? 'flex' : 'none';
            } else if (filterType === 'benchmarked') {
                item.style.display = status === 'benchmarked' ? 'flex' : 'none';
            }
        });

        // Update active tab button style
        ['all', 'new', 'benchmarked'].forEach(type => {
            const btn = document.getElementById(`filter-models-${type}`);
            if (btn) {
                if (type === filterType) {
                    btn.classList.add('active');
                    btn.style.background = '#1e293b';
                    btn.style.borderColor = type === 'new' ? '#34d399' : (type === 'benchmarked' ? '#818cf8' : 'var(--color-primary)');
                } else {
                    btn.classList.remove('active');
                    btn.style.background = '#090d16';
                    btn.style.borderColor = 'var(--border-color)';
                }
            }
        });
    }

    const filterAllBtn = document.getElementById('filter-models-all');
    const filterNewBtn = document.getElementById('filter-models-new');
    const filterBenchBtn = document.getElementById('filter-models-benchmarked');
    if (filterAllBtn) filterAllBtn.addEventListener('click', () => applyModelListFilter('all'));
    if (filterNewBtn) filterNewBtn.addEventListener('click', () => applyModelListFilter('new'));
    if (filterBenchBtn) filterBenchBtn.addEventListener('click', () => applyModelListFilter('benchmarked'));

    selectAllTestsBtn.addEventListener('click', () => {
        const boxes = testCheckboxes.querySelectorAll('input[type="checkbox"]');
        boxes.forEach(box => box.checked = true);
        logToTerminal("All tests selected");
    });

    deselectAllTestsBtn.addEventListener('click', () => {
        const boxes = testCheckboxes.querySelectorAll('input[type="checkbox"]');
        boxes.forEach(box => box.checked = false);
        logToTerminal("All tests deselected");
    });

    function getSelectedTests() {
        const boxes = testCheckboxes.querySelectorAll('input[type="checkbox"]:checked');
        return Array.from(boxes).map(box => box.value);
    }

    function getSelectedGroups() {
        if (!groupCheckboxes) return [];
        const boxes = groupCheckboxes.querySelectorAll('input[type="checkbox"]:checked');
        return Array.from(boxes).map(box => box.value);
    }

    function getSelectedSharedTests() {
        const container = document.getElementById('shared-test-checkboxes');
        if (!container) return [];
        const boxes = container.querySelectorAll('input[type="checkbox"]:checked');
        return Array.from(boxes).map(box => box.value);
    }

    function getSelectedMultistepWorkflows() {
        const container = document.getElementById('multistep-workflow-checkboxes');
        if (!container) return [];
        const boxes = container.querySelectorAll('input[type="checkbox"]:checked');
        return Array.from(boxes).map(box => box.value);
    }

    function getSelectedModels() {
        const boxes = modelCheckboxes.querySelectorAll('input[type="checkbox"]:checked');
        return Array.from(boxes).map(box => box.value);
    }

    // Polling Stable Diffusion Status
    async function pollSDStatus() {
        const sdStatusCard = document.getElementById('sd-status-card');
        const sdConnectionTitle = document.getElementById('sd-connection-title');
        const sdConnectionSubtitle = document.getElementById('sd-connection-subtitle');
        const sdActiveModelBadge = document.getElementById('sd-active-model-badge');
        const sdStatusBadge = document.getElementById('sd-status-badge');

        if (!sdStatusCard) return;

        try {
            const res = await fetch('/api/sd/status', { signal: AbortSignal.timeout(3000) });
            const data = await res.json();

            if (!data.online) {
                sdStatusCard.style.borderLeftColor = 'var(--color-danger)';
                sdConnectionTitle.textContent = "Stable Diffusion Backend Offline";
                sdConnectionSubtitle.textContent = `Error connecting to Stable Diffusion proxy: ${data.error || 'Server unreachable'}`;
                sdActiveModelBadge.textContent = "Model: None";
                sdStatusBadge.className = "badge badge-danger";
                sdStatusBadge.textContent = "Offline";
                return;
            }

            // Proxy is reachable - check if sd-server is still booting (model swap in progress)
            if (data.online && !data.sd_server_healthy) {
                sdStatusCard.style.borderLeftColor = 'var(--color-warning)';
                sdConnectionTitle.textContent = "Stable Diffusion Backend [Loading...]";
                sdConnectionSubtitle.textContent = "SD-Server is starting up - auto-loading a model. This may take up to 60 seconds.";
                sdActiveModelBadge.textContent = data.active_model ? `Loading: ${data.active_model.split('/').pop()}` : "Model: Loading...";
                sdStatusBadge.className = "badge badge-warning";
                sdStatusBadge.textContent = "Loading";
                return;
            }

            sdStatusCard.style.borderLeftColor = 'var(--color-success)';
            sdConnectionTitle.textContent = "Stable Diffusion Backend [Online]";
            sdStatusBadge.className = "badge badge-success";
            sdStatusBadge.textContent = "Online";

            if (data.active_model) {
                const modelName = data.active_model.split('/').pop();
                sdActiveModelBadge.textContent = `Model: ${modelName}`;
                sdConnectionSubtitle.textContent = `SD-Server is active. Queue depth: ${data.queue_depth || 0}. GPU VRAM: ${data.vram_used_mb}MB / ${data.vram_total_mb}MB`;
            } else {
                sdActiveModelBadge.textContent = "Model: None (Idle)";
                sdConnectionSubtitle.textContent = "SD-Server is active and idling. Waiting for generation requests.";
            }

        } catch (err) {
            // Swallow fetch errors (timeout / network) - proxy may be momentarily busy during model swap
            if (err.name !== 'AbortError' && err.name !== 'TimeoutError') {
                console.error("SD Poller error:", err);
            }
        }
    }

    // Polling System Metrics from proxy
    async function pollProxyStatus() {
        try {
            pollSDStatus();
            const res = await fetch('/api/proxy/status', { signal: AbortSignal.timeout(4000) });
            const data = await res.json();
            
            if (!data.online) {
                // Proxy Offline
                proxyStatusCard.style.borderLeftColor = 'var(--color-danger)';
                proxyConnectionTitle.textContent = "Server Monitor Offline";
                proxyConnectionSubtitle.textContent = `Error connecting to proxy: ${data.error || 'Server unreachable'}`;
                proxyUptimeBadge.className = "badge badge-danger";
                proxyUptimeBadge.textContent = "Offline";
                
                const downloadLogsBtn = document.getElementById('btn-download-logs');
                if (downloadLogsBtn) downloadLogsBtn.style.display = 'none';
                
                // Clear hardware gauges
                if (cpuPercent) cpuPercent.textContent = "0%";
                if (cpuBar) cpuBar.style.width = "0%";
                if (ramUsageText) ramUsageText.textContent = "0GB / 0GB (0%)";
                if (ramBar) ramBar.style.width = "0%";
                if (vramUsageText) vramUsageText.textContent = "0MB / 0MB (0%)";
                if (vramBar) vramBar.style.width = "0%";
                if (loadedModelName) loadedModelName.textContent = "Offline";
                currentModelName = null;

                const pCpuPct = document.getElementById('profile-cpu-percent');
                const pCpuBar = document.getElementById('profile-cpu-bar');
                const pRamUsage = document.getElementById('profile-ram-usage-text');
                const pRamBar = document.getElementById('profile-ram-bar');
                const pVramUsage = document.getElementById('profile-vram-usage-text');
                const pVramBar = document.getElementById('profile-vram-bar');

                if (pCpuPct) pCpuPct.textContent = "0%";
                if (pCpuBar) pCpuBar.style.width = "0%";
                if (pRamUsage) pRamUsage.textContent = "0GB / 0GB (0%)";
                if (pRamBar) pRamBar.style.width = "0%";
                if (pVramUsage) pVramUsage.textContent = "0MB / 0MB (0%)";
                if (pVramBar) pVramBar.style.width = "0%";
                
                return;
            }

            // Proxy Online
            proxyStatusCard.style.borderLeftColor = 'var(--color-success)';
            proxyConnectionTitle.textContent = `Alpaca Proxy Monitor [Online]`;
            
            const downloadLogsBtn = document.getElementById('btn-download-logs');
            if (downloadLogsBtn) downloadLogsBtn.style.display = 'flex';
            
            // Format uptime
            const uptimeSec = data.metrics.uptime_seconds || 0;
            const uptimeHr = Math.floor(uptimeSec / 3600);
            const uptimeMin = Math.floor((uptimeSec % 3600) / 60);
            proxyConnectionSubtitle.textContent = `Proxy running on primary network node. Host: ${data.system.hostname || 'unknown'}`;
            proxyUptimeBadge.className = "badge badge-success";
            proxyUptimeBadge.textContent = uptimeHr > 0 ? `Uptime: ${uptimeHr}h ${uptimeMin}m` : `Uptime: ${uptimeMin}m`;

            const pCpuPct = document.getElementById('profile-cpu-percent');
            const pCpuBar = document.getElementById('profile-cpu-bar');
            const pRamUsage = document.getElementById('profile-ram-usage-text');
            const pRamBar = document.getElementById('profile-ram-bar');
            const pVramUsage = document.getElementById('profile-vram-usage-text');
            const pVramBar = document.getElementById('profile-vram-bar');

            // 1. Hardware Utilization Gauges
            if (data.system.cpu_usage) {
                const cpu = Math.round(data.system.cpu_usage.percent || 0);
                if (cpuPercent) cpuPercent.textContent = `${cpu}%`;
                if (cpuBar) cpuBar.style.width = `${cpu}%`;
                if (pCpuPct) pCpuPct.textContent = `${cpu}%`;
                if (pCpuBar) pCpuBar.style.width = `${cpu}%`;
            }
            if (data.system.ram_usage) {
                const ram = data.system.ram_usage;
                const ramStr = `${ram.used_gb}GB / ${ram.total_gb}GB (${ram.used_pct}%)`;
                if (ramUsageText) ramUsageText.textContent = ramStr;
                if (ramBar) ramBar.style.width = `${ram.used_pct}%`;
                if (pRamUsage) pRamUsage.textContent = ramStr;
                if (pRamBar) pRamBar.style.width = `${ram.used_pct}%`;
            }
            if (data.system.gpus && data.system.gpus.length > 0) {
                const gpu = data.system.gpus[0]; // primary GPU
                const total = gpu.total_mb;
                const used = gpu.used_mb;
                const pct = Math.round((used / Math.max(total, 1)) * 100);
                const vramStr = `${used}MB / ${total}MB (${pct}%)`;
                if (vramUsageText) vramUsageText.textContent = vramStr;
                if (vramBar) vramBar.style.width = `${pct}%`;
                if (pVramUsage) pVramUsage.textContent = vramStr;
                if (pVramBar) pVramBar.style.width = `${pct}%`;
            }

            // 2. Currently Loaded Model Details
            const loaded = data.runtime.loaded_models || [];
            let activeModelName = null;
            const runningSettingsContainer = document.getElementById('loaded-model-running-settings');
            const syncBadge = document.getElementById('loaded-model-sync-badge');
            const peakReqs = document.getElementById('loaded-model-peak-requests');
            const totalReqs = document.getElementById('loaded-model-total-requests');
            
            if (runningSettingsContainer) {
                runningSettingsContainer.innerHTML = '';
            }
            if (syncBadge) {
                syncBadge.classList.add('d-none');
            }
            
            if (loaded.length > 0) {
                const activeModel = loaded[0];
                activeModelName = activeModel.name;
                currentModelName = activeModel.name;
                loadedModelName.textContent = activeModel.name;
                loadedModelRequests.textContent = activeModel.active_requests || 0;
                if (peakReqs) peakReqs.textContent = activeModel.peak_active_requests || 0;
                if (totalReqs) totalReqs.textContent = activeModel.total_requests_processed || 0;
                
                // Determine context length - prefer running_settings ctx-size over
                // the model record's context_length, which may be stale/default
                const runningSettings = activeModel.running_settings || {};
                let ctxLength = runningSettings['ctx-size'] || activeModel.context_length;
                if (!ctxLength && data.system && data.system.llama_server_props) {
                    ctxLength = data.system.llama_server_props.n_ctx;
                }
                if (!ctxLength) {
                    ctxLength = '?';
                }
                loadedModelContext.textContent = ctxLength !== '?' ? `${Number(ctxLength).toLocaleString()} tokens` : 'Unknown';
                
                if (activeModel.expires_at.startsWith('9999') || activeModel.expires_at.startsWith('0001')) {
                    loadedModelTtl.textContent = "Persistent (Never Evict)";
                } else {
                    const expiry = new Date(activeModel.expires_at);
                    const now = new Date();
                    const ttlMin = Math.max(0, Math.round((expiry - now) / 60000));
                    loadedModelTtl.textContent = `Unloads in ${ttlMin} mins`;
                }

                // Render running settings
                const settings = activeModel.running_settings || {};
                const keysToDisplay = {
                    'ctx-size': 'Context Size',
                    'n-gpu-layers': 'GPU Layers',
                    'flash-attn': 'Flash Attention',
                    'cache-type-k': 'KV Key Cache',
                    'cache-type-v': 'KV Value Cache',
                    'kv-unified': 'KV Unified',
                    'spec-type': 'Speculative Type',
                    'spec-draft-n-max': 'Spec Draft Max',
                    'n-cpu-moe': 'CPU MoE Threads'
                };
                
                if (runningSettingsContainer) {
                    Object.entries(keysToDisplay).forEach(([key, label]) => {
                        if (settings[key] !== undefined) {
                            const val = settings[key];
                            const div = document.createElement('div');
                            div.className = 'flex-space';
                            div.innerHTML = `<span style="font-size:0.75rem; color:var(--text-muted);">${label}</span><span style="font-size:0.8rem; color:white;">${val}</span>`;
                            runningSettingsContainer.appendChild(div);
                        }
                    });
                }
                
                // Compare with Disk logic
                if (syncBadge && activeModel.backend_model && modelProfiles[activeModel.backend_model]) {
                    const profileSettings = modelProfiles[activeModel.backend_model];
                    let outOfSync = false;
                    const fieldsToCompare = ['ctx-size', 'n-gpu-layers', 'cache-type-k', 'cache-type-v', 'flash-attn', 'kv-unified', 'spec-type', 'spec-draft-n-max', 'n-cpu-moe'];
                    for (const f of fieldsToCompare) {
                        const runVal = settings[f];
                        const diskVal = profileSettings[f];
                        if (runVal !== undefined && diskVal !== undefined) {
                            if (String(runVal) !== String(diskVal)) {
                                outOfSync = true;
                                break;
                            }
                        }
                    }
                    if (outOfSync) {
                        syncBadge.classList.remove('d-none');
                    }
                }
            } else {
                const loading = data.runtime.loading_models || [];
                currentModelName = null;
                if (loading.length > 0) {
                    loadedModelName.innerHTML = `<span style="color:var(--color-secondary); animation: pulse 1.5s infinite;">Attempting to load: ${loading[0].name} (${loading[0].elapsed_seconds}s)</span>`;
                    loadedModelRequests.textContent = "0";
                    if (peakReqs) peakReqs.textContent = "0";
                    if (totalReqs) totalReqs.textContent = "0";
                    loadedModelContext.textContent = "Loading...";
                    loadedModelTtl.textContent = "In progress...";
                } else {
                    loadedModelName.textContent = "No model active (Evicted/Idle)";
                    loadedModelRequests.textContent = "0";
                    if (peakReqs) peakReqs.textContent = "0";
                    if (totalReqs) totalReqs.textContent = "0";
                    loadedModelContext.textContent = "-";
                    loadedModelTtl.textContent = "-";
                }
            }

            // 3. Performance Metrics Counters
            proxyTotalRequests.textContent = data.metrics.requests_total || 0;
            proxyAvgLatency.textContent = data.metrics.avg_latency_ms ? `${Math.round(data.metrics.avg_latency_ms)} ms` : '0 ms';
            proxyPromptTokens.textContent = data.metrics.tokens_prompted || 0;
            proxyGenTokens.textContent = data.metrics.tokens_generated || 0;

            // 4. Connected Clients Parsing
            const logLines = data.logs || [];
            const clients = {};
            const requestTypes = {};
            
            // Regex to parse log statements: Hit: POST /v1/chat/completions | Origin: browser/ui | IP: 172.22.0.1
            const logRegex = /Hit: (\w+ \S+) \| Origin: ([^|]+) \| IP: ([^|]+)/;
            
            logLines.forEach(line => {
                const match = logRegex.exec(line);
                if (match) {
                    const endpoint = match[1];
                    const origin = match[2].trim();
                    const ip = match[3].trim();
                    
                    // Client totals
                    if (!clients[origin]) {
                        clients[origin] = { ip: ip, count: 0 };
                    }
                    clients[origin].count += 1;
                    
                    // Endpoints count
                    requestTypes[endpoint] = (requestTypes[endpoint] || 0) + 1;
                }
            });

            // Display Clients
            monitorClients.innerHTML = '';
            const clientKeys = Object.keys(clients);
            if (clientKeys.length === 0) {
                monitorClients.innerHTML = `<div style="color:var(--text-muted);font-size:0.75rem;padding:0.5rem;text-align:center;">No clients detected in buffer</div>`;
            } else {
                clientKeys.forEach(origin => {
                    const item = document.createElement('div');
                    item.className = 'monitor-list-item';
                    
                    const left = document.createElement('div');
                    left.className = 'monitor-list-item-left';
                    const title = document.createElement('div');
                    title.className = 'monitor-list-item-title';
                    title.textContent = origin.toUpperCase();
                    const sub = document.createElement('div');
                    sub.className = 'monitor-list-item-sub';
                    sub.textContent = `IP: ${clients[origin].ip}`;
                    
                    left.appendChild(title);
                    left.appendChild(sub);
                    
                    const badge = document.createElement('span');
                    badge.className = 'monitor-list-item-badge';
                    badge.textContent = `${clients[origin].count} reqs`;
                    
                    item.appendChild(left);
                    item.appendChild(badge);
                    monitorClients.appendChild(item);
                });
            }

            // Display Request Types
            monitorEndpoints.innerHTML = '';
            const endpointKeys = Object.keys(requestTypes);
            if (endpointKeys.length === 0) {
                monitorEndpoints.innerHTML = `<div style="color:var(--text-muted);font-size:0.75rem;padding:0.5rem;text-align:center;">No requests in buffer</div>`;
            } else {
                endpointKeys.forEach(endpoint => {
                    const item = document.createElement('div');
                    item.className = 'monitor-list-item';
                    
                    const left = document.createElement('div');
                    left.className = 'monitor-list-item-left';
                    const title = document.createElement('div');
                    title.className = 'monitor-list-item-title';
                    title.textContent = endpoint;
                    left.appendChild(title);
                    
                    const badge = document.createElement('span');
                    badge.className = 'monitor-list-item-badge';
                    badge.textContent = `${requestTypes[endpoint]} hits`;
                    
                    item.appendChild(left);
                    item.appendChild(badge);
                    monitorEndpoints.appendChild(item);
                });
            }

            // 5. Server Slots Grid (llama.cpp)
            slotsGridContainer.innerHTML = '';
            const slots = data.slots.slots || [];
            if (slots.length === 0) {
                slotsGridContainer.innerHTML = `<div style="color:var(--text-muted);font-size:0.8rem;text-align:center;padding:2rem;width:100%;">No server slots detected</div>`;
            } else {
                slots.forEach(slot => {
                    const isBusy = slot.is_processing || slot.alpaca?.is_busy || false;
                    const usedPct = slot.alpaca?.context_used_pct || 0;
                    
                    const card = document.createElement('div');
                    card.className = `slot-card ${isBusy ? 'busy' : 'idle'}`;
                    
                    const header = document.createElement('div');
                    header.className = 'slot-header';
                    const slotId = document.createElement('span');
                    slotId.className = 'slot-id';
                    slotId.textContent = `Slot #${slot.id}`;
                    const statusText = document.createElement('span');
                    statusText.className = `slot-status-text ${isBusy ? 'busy' : 'idle'}`;
                    statusText.textContent = isBusy ? 'Processing' : 'Idle';
                    
                    header.appendChild(slotId);
                    header.appendChild(statusText);
                    
                    const ctxItem = document.createElement('div');
                    ctxItem.className = 'slot-detail-item';
                    ctxItem.innerHTML = `<span>Context Util</span><strong>${usedPct}%</strong>`;
                    
                    const ctxTokensItem = document.createElement('div');
                    ctxTokensItem.className = 'slot-detail-item';
                    ctxTokensItem.innerHTML = `<span>Context Tokens</span><strong>${slot.n_past || 0} / ${slot.n_ctx || 0}</strong>`;
                    
                    const tokensItem = document.createElement('div');
                    tokensItem.className = 'slot-detail-item';
                    tokensItem.innerHTML = `<span>Tokens Gen</span><strong>${slot.n_written || slot.n_decoded || 0}</strong>`;
                    
                    const hitTokens = slot.n_prompt_tokens_cache || 0;
                    const totalTokens = slot.n_prompt_tokens || 0;
                    const hitRate = totalTokens > 0 ? Math.round((hitTokens / totalTokens) * 100) : 0;
                    const cacheHitItem = document.createElement('div');
                    cacheHitItem.className = 'slot-detail-item';
                    cacheHitItem.innerHTML = `<span>Cache Hit Rate</span><strong>${hitTokens} / ${totalTokens} (${hitRate}%)</strong>`;
                    
                    card.appendChild(header);
                    card.appendChild(ctxItem);
                    card.appendChild(ctxTokensItem);
                    card.appendChild(tokensItem);
                    card.appendChild(cacheHitItem);

                    // Optional Speculative parameters
                    const specType = slot.params && slot.params["speculative.types"];
                    if (specType && specType !== "none") {
                        const specItem = document.createElement('div');
                        specItem.className = 'slot-detail-item';
                        specItem.innerHTML = `<span>Spec Mode</span><strong>${specType}</strong>`;
                        card.appendChild(specItem);
                    }
                    
                    slotsGridContainer.appendChild(card);
                });
            }

            // Update OOM Telemetry and Config suggestions
            updateTelemetryAndRecommendations(activeModelName);

        } catch (err) {
            console.error("Metrics Poller error: ", err);
        }
    }

    function startMonitorPolling() {
        if (!monitorIntervalId) {
            pollProxyStatus(); // immediate load
            // Visibility-aware: skip network churn while the page is hidden.
            monitorIntervalId = setInterval(() => { if (!document.hidden) pollProxyStatus(); }, 2000);
        }
    }

    function stopMonitorPolling() {
        if (monitorIntervalId) {
            clearInterval(monitorIntervalId);
            monitorIntervalId = null;
        }
    }

    let requestsIntervalId = null;
    let selectedRequestId = null;
    let allRequestsMap = {};
    let lastActiveList = [];
    let lastCompletedList = [];
    let requestsServerContext = null;

    function startRequestsPolling() {
        if (!requestsIntervalId) {
            setupRequestsControls();
            pollRequestsStatus(); // immediate load
            // Visibility-aware: skip network churn while the page is hidden.
            requestsIntervalId = setInterval(() => { if (!document.hidden) pollRequestsStatus(); }, 2000);
        }
    }

    function stopRequestsPolling() {
        if (requestsIntervalId) {
            clearInterval(requestsIntervalId);
            requestsIntervalId = null;
        }
    }

    function showToast(message, type = 'info', opts = {}) {
        const toast = document.createElement('div');
        const colors = {
            success: { bg: 'rgba(16, 185, 129, 0.15)', border: 'rgba(16, 185, 129, 0.3)', color: '#6ee7b7' },
            error: { bg: 'rgba(239, 68, 68, 0.15)', border: 'rgba(239, 68, 68, 0.3)', color: '#fca5a5' },
            info: { bg: 'rgba(59, 130, 246, 0.15)', border: 'rgba(59, 130, 246, 0.3)', color: '#93c5fd' }
        };
        const c = colors[type] || colors.info;
        Object.assign(toast.style, {
            position: 'fixed', bottom: '1rem', right: '1rem', padding: '0.6rem 1rem',
            background: c.bg, border: `1px solid ${c.border}`, borderRadius: '8px',
            color: c.color, fontSize: '0.8rem', zIndex: '10000',
            fontFamily: 'system-ui, sans-serif', boxShadow: '0 4px 12px rgba(0,0,0,0.3)',
            display: 'flex', alignItems: 'center', gap: '0.75rem', maxWidth: 'min(92vw, 420px)'
        });
        const msgSpan = document.createElement('span');
        msgSpan.textContent = message;
        msgSpan.style.wordBreak = 'break-word';
        toast.appendChild(msgSpan);
        if (opts.actionLabel && typeof opts.onAction === 'function') {
            const actBtn = document.createElement('button');
            actBtn.className = 'btn btn-sm btn-ghost';
            actBtn.textContent = opts.actionLabel;
            actBtn.style.flex = '0 0 auto';
            actBtn.addEventListener('click', () => { opts.onAction(); dismissToast(); });
            toast.appendChild(actBtn);
        }
        let dismissed = false;
        function dismissToast() {
            if (dismissed) return;
            dismissed = true;
            toast.style.opacity = '0';
            toast.style.transition = 'opacity 0.3s ease';
            setTimeout(() => toast.remove(), 300);
        }
        document.body.appendChild(toast);
        setTimeout(dismissToast, opts.duration || 3000);
    }


    async function pollRequestsStatus() {
        try {
            const res = await fetch('/api/requests');
            if (!res.ok) {
                return;
            }
            const data = await res.json();
            
            lastActiveList = data.active_requests || [];
            lastCompletedList = data.completed_requests || [];
            requestsServerContext = data.server_context || null;
            
            const newRequestsMap = {};
            lastActiveList.forEach(r => {
                newRequestsMap[r.request_id] = r;
            });
            lastCompletedList.forEach(r => {
                newRequestsMap[r.request_id] = r;
            });
            allRequestsMap = newRequestsMap;

            renderRequestsLists(lastActiveList, lastCompletedList);
            
            if (selectedRequestId && allRequestsMap[selectedRequestId]) {
                updateInspectorDetails(allRequestsMap[selectedRequestId]);
            }
        } catch (err) {
            console.error("Requests Poller error:", err);
        }
    }

    function formatInitiatedTime(unixTimestamp) {
        const timezoneSelect = document.getElementById('requests-timezone-select');
        const selectedTz = timezoneSelect ? timezoneSelect.value : 'local';
        
        const date = new Date(unixTimestamp * 1000);
        
        const options = {
            year: 'numeric',
            month: '2-digit',
            day: '2-digit',
            hour: '2-digit',
            minute: '2-digit',
            second: '2-digit',
            hour12: false,
            hourCycle: 'h23'
        };
        
        if (selectedTz !== 'local') {
            options.timeZone = selectedTz;
        }
        
        try {
            const formatter = new Intl.DateTimeFormat('en-US', options);
            const parts = formatter.formatToParts(date);
            const partMap = {};
            parts.forEach(p => {
                partMap[p.type] = p.value;
            });
            return `${partMap.year}-${partMap.month}-${partMap.day} ${partMap.hour}:${partMap.minute}:${partMap.second}`;
        } catch (e) {
            // Fallback
            return date.toISOString().replace('T', ' ').substring(0, 19);
        }
    }

    function renderRequestsLists(activeList, completedList) {
        const activeContainer = document.getElementById('active-requests-list');
        const completedContainer = document.getElementById('completed-requests-list');
        
        if (!activeContainer || !completedContainer) return;

        const query = (document.getElementById('requests-search-input')?.value || '').toLowerCase();
        
        let filteredActive = activeList;
        let filteredCompleted = completedList;
        
        if (query) {
            filteredActive = activeList.filter(req => 
                (req.model || '').toLowerCase().includes(query) || 
                (req.prompt || '').toLowerCase().includes(query)
            );
            filteredCompleted = completedList.filter(req => 
                (req.model || '').toLowerCase().includes(query) || 
                (req.prompt || '').toLowerCase().includes(query)
            );
        }

        // Render Active Requests
        if (filteredActive.length === 0) {
            activeContainer.innerHTML = `<div style="color:var(--text-muted);font-size:0.75rem;padding:0.5rem;text-align:center;background:#0f172a;border-radius:6px;">No active requests</div>`;
        } else {
            activeContainer.innerHTML = '';
            filteredActive.forEach(req => {
                const item = createRequestRow(req, true);
                activeContainer.appendChild(item);
            });
        }
        
        // Render Completed Requests
        if (filteredCompleted.length === 0) {
            completedContainer.innerHTML = `<div style="color:var(--text-muted);font-size:0.75rem;padding:0.5rem;text-align:center;background:#0f172a;border-radius:6px;">No requests in history</div>`;
        } else {
            completedContainer.innerHTML = '';
            const sortedCompleted = [...filteredCompleted].reverse();
            sortedCompleted.forEach(req => {
                const item = createRequestRow(req, false);
                completedContainer.appendChild(item);
            });
        }
    }

    function createRequestRow(req, isActive) {
        const div = document.createElement('div');
        div.className = `request-item-row ${selectedRequestId === req.request_id ? 'active' : ''}`;
        div.style.cssText = `
            padding: 0.6rem;
            background: #0f172a;
            border: 1px solid ${selectedRequestId === req.request_id ? 'var(--color-primary)' : 'var(--border-color)'};
            border-radius: 6px;
            cursor: pointer;
            display: flex;
            flex-direction: column;
            gap: 0.25rem;
            transition: all 0.2s ease;
        `;
        
        div.addEventListener('mouseenter', () => {
            if (selectedRequestId !== req.request_id) {
                div.style.borderColor = 'rgba(139, 92, 246, 0.4)';
            }
        });
        div.addEventListener('mouseleave', () => {
            if (selectedRequestId !== req.request_id) {
                div.style.borderColor = 'var(--border-color)';
            }
        });

        div.addEventListener('click', () => {
            selectedRequestId = req.request_id;
            
            document.querySelectorAll('.request-item-row').forEach(row => {
                row.classList.remove('active');
                row.style.borderColor = 'var(--border-color)';
            });
            div.classList.add('active');
            div.style.borderColor = 'var(--color-primary)';
            
            const emptyEl = document.getElementById('request-inspector-empty');
            const detailsEl = document.getElementById('request-inspector-details');
            if (emptyEl) emptyEl.classList.add('d-none');
            if (detailsEl) detailsEl.classList.remove('d-none');
            
            updateInspectorDetails(req);
        });

        const headerDiv = document.createElement('div');
        headerDiv.style.cssText = 'display:flex; justify-content:space-between; align-items:center;';
        
        const typeBadge = document.createElement('span');
        typeBadge.className = `badge ${isActive ? 'badge-success' : 'badge-secondary'}`;
        typeBadge.style.fontSize = '0.6rem';
        typeBadge.style.padding = '0.1rem 0.3rem';
        typeBadge.textContent = req.type || 'unknown';

        if (req.online) {
            typeBadge.textContent = `ONLINE · ${req.type || 'online'}`;
            typeBadge.style.background = 'rgba(139, 92, 246, 0.25)';
            typeBadge.style.color = '#c4b5fd';
            typeBadge.style.border = '1px solid rgba(139, 92, 246, 0.5)';
            div.style.borderLeft = '3px solid rgba(139, 92, 246, 0.7)';
        }

        // Failed request marker: red FAILED chip + accent border so empty-output
        // rows are instantly recognizable as errors, with the reason in the inspector.
        const failedBadge = document.createElement('span');
        if (!isActive && req.error) {
            failedBadge.className = 'badge badge-danger';
            failedBadge.style.fontSize = '0.6rem';
            failedBadge.style.padding = '0.1rem 0.35rem';
            failedBadge.textContent = 'FAILED';
            failedBadge.title = String(req.error);
            div.style.borderLeft = '3px solid var(--color-danger, #ef4444)';
        }

        const timeSpan = document.createElement('span');
        timeSpan.style.cssText = 'font-size: 0.65rem; color: var(--text-muted);';
        
        if (isActive) {
            const elapsed = Math.round(Date.now() / 1000 - req.started_at);
            timeSpan.textContent = `Running: ${elapsed}s`;
        } else {
            timeSpan.textContent = `${req.duration_seconds || 0}s`;
        }

        headerDiv.appendChild(typeBadge);
        if (!isActive && req.error) headerDiv.appendChild(failedBadge);
        headerDiv.appendChild(timeSpan);

        const modelDiv = document.createElement('div');
        modelDiv.style.cssText = 'font-size: 0.75rem; font-weight: 600; color: white; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;';
        modelDiv.title = req.model || 'Unknown Model'; // full name in native tooltip
        modelDiv.textContent = truncateModelName(req.model) || 'Unknown Model';

        const idDiv = document.createElement('div');
        idDiv.style.cssText = 'font-size: 0.65rem; color: var(--text-muted); font-family: monospace;';
        idDiv.textContent = `ID: ${req.request_id}`;

        const detailsRow = document.createElement('div');
        detailsRow.style.cssText = 'display:flex; justify-content:space-between; align-items:center; font-size:0.65rem; color:var(--text-muted); margin-top:0.15rem;';
        
        const originSpan = document.createElement('span');
        originSpan.style.color = '#38bdf8';
        originSpan.style.fontWeight = '500';
        originSpan.textContent = `Origin: ${req.request_source || 'unknown'}`;
        
        const metricsSpan = document.createElement('span');
        if (req.tps) {
            metricsSpan.textContent = `${req.tps} tps | ${req.ttft_seconds || 0}s ttft`;
        } else {
            metricsSpan.textContent = '';
        }
        
        detailsRow.appendChild(originSpan);
        detailsRow.appendChild(metricsSpan);

        div.appendChild(headerDiv);
        div.appendChild(modelDiv);

        // Small preview thumbnail for image-generation requests in the Active Requests list.
        if (req.type === 'image_generation' && req.images && req.images.length) {
            const first = req.images[0];
            const src = first.type === 'url' ? first.data : `data:image/png;base64,${first.data}`;
            const thumb = document.createElement('img');
            thumb.src = src;
            thumb.alt = 'preview';
            thumb.title = `Preview (${req.images.length} image(s))`;
            thumb.style.cssText = 'max-width:100%; max-height:90px; border-radius:6px; border:1px solid var(--border-color); margin-top:0.25rem; cursor:pointer;';
            thumb.addEventListener('click', (e) => {
                e.stopPropagation();
                window.open(src, '_blank');
            });
            div.appendChild(thumb);
        }

        div.appendChild(idDiv);

        const initTimeDiv = document.createElement('div');
        initTimeDiv.style.cssText = 'font-size: 0.65rem; color: var(--text-muted); display: flex; align-items: center; gap: 0.25rem; margin-top: 0.1rem;';
        initTimeDiv.innerHTML = `<span>🕒</span> <span>Initiated: ${formatInitiatedTime(req.started_at)}</span>`;
        div.appendChild(initTimeDiv);

        div.appendChild(detailsRow);
        const actionBtns = document.createElement('div');
        actionBtns.style.cssText = 'display:flex; gap:0.25rem; justify-content:flex-end; margin-top:0.25rem;';
        
        if (isActive) {
            const cancelBtn = document.createElement('button');
            cancelBtn.className = 'req-action-btn';
            cancelBtn.textContent = 'Cancel';
            cancelBtn.style.cssText = `
                font-size: 0.6rem;
                padding: 0.15rem 0.4rem;
                background: rgba(239,68,68,0.15);
                border: 1px solid rgba(239,68,68,0.3);
                border-radius: 4px;
                color: #fca5a5;
                cursor: pointer;
            `;
            cancelBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                cancelBtn.textContent = 'Cancelling...';
                cancelBtn.disabled = true;
                fetch('/api/requests/cancel', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({request_id: req.request_id})
                })
                .then(r => r.json())
                .then(data => {
                    if (data.error) {
                        showToast('Error: ' + data.error, 'error');
                        cancelBtn.textContent = 'Cancel';
                        cancelBtn.disabled = false;
                    } else {
                        showToast('Request cancelled', 'success');
                        pollRequestsStatus();
                    }
                })
                .catch(err => {
                    showToast('Cancel failed: ' + err.message, 'error');
                    cancelBtn.textContent = 'Cancel';
                    cancelBtn.disabled = false;
                });
            });
            actionBtns.appendChild(cancelBtn);
        } else {
            const resubmitBtn = document.createElement('button');
            resubmitBtn.className = 'req-action-btn';
            resubmitBtn.textContent = 'Resubmit';
            resubmitBtn.style.cssText = `
                font-size: 0.6rem;
                padding: 0.15rem 0.4rem;
                background: rgba(251,191,36,0.15);
                border: 1px solid rgba(251,191,36,0.3);
                border-radius: 4px;
                color: #fde68a;
                cursor: pointer;
            `;
            resubmitBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                resubmitBtn.textContent = 'Resubmitting...';
                resubmitBtn.disabled = true;
                fetch('/api/requests/resubmit/' + req.request_id, {
                    method: 'POST'
                })
                .then(r => r.json())
                .then(data => {
                    if (data.error) {
                        showToast('Error: ' + data.error, 'error');
                        resubmitBtn.textContent = 'Resubmit';
                        resubmitBtn.disabled = false;
                    } else {
                        showToast('Request resubmitted', 'success');
                        pollRequestsStatus();
                    }
                })
                .catch(err => {
                    showToast('Resubmit failed: ' + err.message, 'error');
                    resubmitBtn.textContent = 'Resubmit';
                    resubmitBtn.disabled = false;
                });
            });
            actionBtns.appendChild(resubmitBtn);
        }
        
        div.appendChild(actionBtns);


        return div;
    }

    function updateInspectorDetails(req) {
        const idEl = document.getElementById('inspect-id');
        const modelEl = document.getElementById('inspect-model');
        const typeEl = document.getElementById('inspect-type');
        const durationEl = document.getElementById('inspect-duration');
        const promptEl = document.getElementById('inspect-prompt');
        const thinkingContainer = document.getElementById('inspect-thinking-container');
        const thinkingEl = document.getElementById('inspect-thinking');
        const responseEl = document.getElementById('inspect-response');
        
        const originEl = document.getElementById('inspect-origin');
        const ipEl = document.getElementById('inspect-ip');
        const ttftEl = document.getElementById('inspect-ttft');
        const tpsEl = document.getElementById('inspect-tps');

        if (idEl) idEl.textContent = req.request_id;
        if (modelEl) modelEl.textContent = req.model;
        
        const initiatedEl = document.getElementById('inspect-initiated');
        if (initiatedEl) initiatedEl.textContent = formatInitiatedTime(req.started_at);
        
        if (typeEl) typeEl.textContent = req.type;
        if (originEl) originEl.textContent = req.request_source || 'unknown';
        if (ipEl) ipEl.textContent = req.client_ip || 'unknown';
        const clientHostEl = document.getElementById('inspect-client-host');
        const hostIpsEl = document.getElementById('inspect-host-ip');
        const externalIpEl = document.getElementById('inspect-external-ip');
        if (clientHostEl) clientHostEl.textContent = req.client_host || '-';
        if (hostIpsEl) {
            hostIpsEl.textContent =
                requestsServerContext && Array.isArray(requestsServerContext.host_ips)
                    ? requestsServerContext.host_ips.join(', ')
                    : '-';
        }
        if (externalIpEl) externalIpEl.textContent = requestsServerContext?.external_ip || '-';
        if (ttftEl) ttftEl.textContent = req.ttft_seconds ? `${req.ttft_seconds}s` : '-';
        if (tpsEl) tpsEl.textContent = req.tps ? `${req.tps} tok/s` : '-';
        
        if (durationEl) {
            if (req.completed_at) {
                durationEl.textContent = `${req.duration_seconds}s (Finished)`;
                durationEl.style.color = 'var(--color-success)';
            } else {
                const elapsed = Math.round(Date.now() / 1000 - req.started_at);
                durationEl.textContent = `${elapsed}s (Active)`;
                durationEl.style.color = 'var(--color-secondary)';
            }
        }
        
        if (promptEl) promptEl.textContent = req.prompt || '(Empty Prompt)';
        
        if (thinkingContainer && thinkingEl) {
            if (req.thinking) {
                thinkingContainer.classList.remove('d-none');
                const isNearBottomThinking = thinkingEl.scrollHeight - thinkingEl.clientHeight - thinkingEl.scrollTop < 40;
                thinkingEl.textContent = req.thinking;
                if (isNearBottomThinking || !req.completed_at) {
                    thinkingEl.scrollTop = thinkingEl.scrollHeight;
                }
            } else {
                thinkingContainer.classList.add('d-none');
                thinkingEl.textContent = '';
            }
        }
        
        if (responseEl) {
            if (req.type === 'image_generation' && req.images && req.images.length) {
                // Render a preview gallery of generated images (thumbnails, not full-size).
                let html = `<div style="display:flex; flex-wrap:wrap; gap:0.5rem;">`;
                req.images.forEach((img, i) => {
                    const src = img.type === 'url' ? img.data : `data:image/png;base64,${img.data}`;
                    html += `<img src="${src}" alt="generated ${i + 1}" title="Generated image ${i + 1}" ` +
                        `style="max-width:220px; max-height:220px; border-radius:6px; border:1px solid var(--border-color); cursor:pointer;" ` +
                        `onclick="window.open('${src}', '_blank')" />`;
                });
                html += `</div>`;
                responseEl.innerHTML = html;
            } else {
                const isNearBottom = responseEl.scrollHeight - responseEl.clientHeight - responseEl.scrollTop < 40;
                if (req.response) {
                    responseEl.textContent = req.response;
                } else if (req.error) {
                    // Failed request: surface WHY there is no output instead of a bare placeholder.
                    const reason = String(req.error).replace(/[<>&]/g, c => ({'<': '&lt;', '>': '&gt;', '&': '&amp;'}[c]));
                    responseEl.innerHTML =
                        `<div class="request-error-note" style="border:1px solid var(--color-danger, #ef4444); background:rgba(239,68,68,0.08); color:var(--color-danger, #f87171); border-radius:6px; padding:0.5rem 0.75rem; font-size:0.85rem;">` +
                        `<strong style="text-transform:uppercase; letter-spacing:0.04em;">Request failed</strong>` +
                        `<div style="margin-top:0.25rem; white-space:pre-wrap; word-break:break-word;">${reason}</div></div>`;
                } else {
                    responseEl.textContent = req.completed_at ? '(No Output)' : 'Generating output...';
                }
                if (isNearBottom || !req.completed_at) {
                    responseEl.scrollTop = responseEl.scrollHeight;
                }
            }
        }
    }

    let requestsControlsSetup = false;
    function setupRequestsControls() {
        const timezoneSelect = document.getElementById('requests-timezone-select');
        if (!timezoneSelect) return; // DOM not ready yet; retry next call

        // Populate options once. The guard resets if the select is ever found
        // empty again (e.g. markup replaced), so restore/listener are re-applied.
        if (timezoneSelect.options.length === 0) {
            requestsControlsSetup = false;
        }
        if (requestsControlsSetup && timezoneSelect.dataset.tzWired === '1') return;
        requestsControlsSetup = true;
        timezoneSelect.dataset.tzWired = '1';
        // Add default browser local timezone
        const localOption = document.createElement('option');
        localOption.value = 'local';
        localOption.textContent = `Local Time (${Intl.DateTimeFormat().resolvedOptions().timeZone})`;
        timezoneSelect.appendChild(localOption);

        // Add UTC option
        const utcOption = document.createElement('option');
        utcOption.value = 'UTC';
        utcOption.textContent = 'UTC';
        timezoneSelect.appendChild(utcOption);

        // Add all other supported timezones
        try {
            const timezones = Intl.supportedValuesOf('timeZone');
            timezones.forEach(tz => {
                if (tz !== 'UTC') { // Already added UTC
                    const opt = document.createElement('option');
                    opt.value = tz;
                    opt.textContent = tz;
                    timezoneSelect.appendChild(opt);
                }
            });
        } catch (e) {
            // Fallback common timezones if Intl.supportedValuesOf is not supported
            const fallbackTz = [
                "America/New_York", "America/Chicago", "America/Denver", "America/Los_Angeles",
                "Europe/London", "Europe/Paris", "Asia/Tokyo", "Asia/Shanghai", "Asia/Kolkata", "Australia/Sydney"
            ];
            fallbackTz.forEach(tz => {
                const opt = document.createElement('option');
                opt.value = tz;
                opt.textContent = tz;
                timezoneSelect.appendChild(opt);
            });
        }

        // Restore selection from localStorage; drop stale keys whose saved
        // value no longer matches any option so the select stays usable.
        const savedTz = localStorage.getItem('alpaca_requests_timezone');
        if (savedTz) {
            if ([...timezoneSelect.options].some(o => o.value === savedTz)) {
                timezoneSelect.value = savedTz;
            } else {
                localStorage.removeItem('alpaca_requests_timezone');
            }
        }

        // Delegated listener: matches by id so it keeps working even if the
        // select element is ever replaced after setup ran.
        document.addEventListener('change', (ev) => {
            if (!ev.target || ev.target.id !== 'requests-timezone-select') return;
            const sel = document.getElementById('requests-timezone-select');
            localStorage.setItem('alpaca_requests_timezone', sel ? sel.value : 'local');
            renderRequestsLists(lastActiveList, lastCompletedList);
            if (selectedRequestId && allRequestsMap[selectedRequestId]) {
                updateInspectorDetails(allRequestsMap[selectedRequestId]);
            }
        });

        document.getElementById('requests-search-input')?.addEventListener('input', () => {
            renderRequestsLists(lastActiveList, lastCompletedList);
        });

        document.getElementById('btn-clear-requests')?.addEventListener('click', async () => {
            if (!confirm('Are you sure you want to clear completed requests history?')) {
                return;
            }
            try {
                const res = await fetch('/api/requests/clear', { method: 'POST' });
                if (res.ok) {
                    selectedRequestId = null;
                    const emptyEl = document.getElementById('request-inspector-empty');
                    const detailsEl = document.getElementById('request-inspector-details');
                    if (emptyEl) emptyEl.classList.remove('d-none');
                    if (detailsEl) detailsEl.classList.add('d-none');
                    pollRequestsStatus();
                }
            } catch (err) {
                console.error("Failed to clear requests history:", err);
            }
        });

        function setupCopyButton(btnId, targetId) {
            document.getElementById(btnId)?.addEventListener('click', () => {
                const el = document.getElementById(targetId);
                const btn = document.getElementById(btnId);
                if (!el || !btn) return;
                
                navigator.clipboard.writeText(el.textContent).then(() => {
                    const originalText = btn.textContent;
                    btn.textContent = '✅ Copied!';
                    setTimeout(() => {
                        btn.textContent = originalText;
                    }, 1500);
                }).catch(err => {
                    console.error("Clipboard copy failed:", err);
                });
            });
        }

        setupCopyButton('btn-copy-prompt', 'inspect-prompt');
        setupCopyButton('btn-copy-thinking', 'inspect-thinking');
        setupCopyButton('btn-copy-response', 'inspect-response');

        document.getElementById('btn-copy-request-details')?.addEventListener('click', async () => {
            const req = allRequestsMap[selectedRequestId];
            if (!req) return;
            const cap = (text, max = 8000) => {
                const s = String(text ?? '');
                return s.length > max ? `${s.slice(0, max)}... [truncated]` : s;
            };
            const ctx = requestsServerContext || {};
            const lines = [
                `Request ID: ${req.request_id}`,
                `Type: ${req.type}`,
                `Model: ${req.model || '-'}`,
                `Initiated: ${formatInitiatedTime(req.started_at)}`,
                req.completed_at
                    ? `Duration: ${req.duration_seconds}s (Finished)`
                    : `Duration: ${Math.round(Date.now() / 1000 - (req.started_at || 0))}s (Active)`,
                `Origin: ${req.request_source || 'unknown'}`,
                `Client Host: ${req.client_host || '-'}`,
                `Client IP: ${req.client_ip || '-'}`,
                `Host IPs: ${(Array.isArray(ctx.host_ips) ? ctx.host_ips.join(', ') : '') || '-'}`,
                `External IP: ${ctx.external_ip || '-'}`,
                `TTFT: ${req.ttft_seconds ? `${req.ttft_seconds}s` : '-'}`,
                `TPS: ${req.tps ? `${req.tps} tok/s` : '-'}`,
            ];
            if (req.error) {
                lines.push(`Error: ${cap(req.error)}`);
            }
            lines.push('', '--- Prompt ---', cap(req.prompt || '(Empty Prompt)'));
            if (req.thinking) {
                lines.push('', '--- Thinking ---', cap(req.thinking));
            }
            lines.push('', '--- Response ---', cap(req.response || '(No Response)'));
            const text = lines.join('\n');
            try {
                await navigator.clipboard.writeText(text);
            } catch (err) {
                // Clipboard API may be unavailable (insecure origin/permissions); fall back.
                const ta = document.createElement('textarea');
                ta.value = text;
                ta.style.position = 'fixed';
                ta.style.opacity = '0';
                document.body.appendChild(ta);
                ta.select();
                document.execCommand('copy');
                ta.remove();
            }
            showToast('Request details copied to clipboard', 'success');
        });
    }

    // Fetch and display available models in configurations sidebar with lifecycle tracking
    async function loadModels() {
        try {
            logToTerminal("Fetching available models & benchmark tracking...");
            const [modelsRes, trackingRes, onlineRes] = await Promise.allSettled([
                fetch('/api/models'),
                fetch('/api/models/tracking'),
                fetch('/api/models/online')
            ]);

            const modelsData = modelsRes.status === 'fulfilled' ? await modelsRes.value.json() : { models: [] };
            const trackingData = trackingRes.status === 'fulfilled' ? await trackingRes.value.json() : { all_tracked: {}, counts: {} };
            const onlineData = onlineRes.status === 'fulfilled' ? await onlineRes.value.json() : { models: [] };

            availableModels = modelsData.models || [];
            const trackedMap = trackingData.all_tracked || {};
            const onlineModels = onlineData.models || [];

            modelCheckboxes.innerHTML = '';
            
            if (availableModels.length === 0 && onlineModels.length === 0) {
                modelCheckboxes.innerHTML = `<div style="color:var(--text-muted);font-size:0.8rem;padding:0.5rem;">No models detected</div>`;
                return;
            }

            let newCount = 0;
            let benchCount = 0;

            function getTrackedMeta(modelId) {
                if (!modelId) return {};
                if (trackedMap[modelId]) return trackedMap[modelId];
                const clean = String(modelId).trim();
                const isOnline = isOnlineModelName(clean);

                if (isOnline) {
                    // For online models, strictly match online provider keys
                    const withoutPrefix = clean.includes(':') ? clean.substring(clean.indexOf(':') + 1) : clean;
                    for (const [k, v] of Object.entries(trackedMap)) {
                        if (!isOnlineModelName(k)) continue;
                        if (k === clean || k === withoutPrefix) return v;
                        if (k.includes(':') && k.substring(k.indexOf(':') + 1) === withoutPrefix) return v;
                    }
                    return {};
                }

                // For local models, strictly match local keys
                for (const [k, v] of Object.entries(trackedMap)) {
                    if (isOnlineModelName(k)) continue;
                    if (k === clean) return v;
                    if (k.replace(/--/g, ':') === clean.replace(/--/g, ':')) return v;
                    if (k.replace(/[:/.]/g, '_') === clean.replace(/[:/.]/g, '_')) return v;
                    if (k.replace(/:latest$/, '') === clean.replace(/:latest$/, '')) return v;
                }
                return {};
            }

            function createModelCheckboxItem(id, displayName, isOnline = false, isFree = false) {
                const meta = getTrackedMeta(id);
                const isBenchmarked = !!(meta.benchmark_count && meta.benchmark_count > 0);
                const isNew = !isBenchmarked;

                if (isNew) newCount++;
                if (isBenchmarked) benchCount++;

                const item = document.createElement('label');
                item.className = 'checkbox-item';
                item.setAttribute('data-status', isNew ? 'new' : 'benchmarked');

                const input = document.createElement('input');
                input.type = 'checkbox';
                input.value = id;
                input.checked = false;

                const span = document.createElement('span');
                span.className = 'checkbox-label';
                span.style.display = 'flex';
                span.style.alignItems = 'center';
                span.style.justifyContent = 'space-between';
                span.style.width = '100%';
                span.style.gap = '0.5rem';

                let badgesHtml = '';
                if (isFree) {
                    badgesHtml += '<span style="font-size:0.62rem; background:rgba(34,197,94,0.15); color:#22c55e; padding:1px 4px; border-radius:3px; margin-left:4px; font-weight:700;">FREE</span>';
                }
                if (isNew) {
                    badgesHtml += '<span style="font-size:0.62rem; background:rgba(16,185,129,0.15); color:#34d399; border:1px solid rgba(16,185,129,0.3); padding:1px 4px; border-radius:3px; margin-left:4px; font-weight:700;" title="Newly added model awaiting benchmark">🆕 NEW</span>';
                } else if (isBenchmarked) {
                    const score = meta.latest_score !== undefined && meta.latest_score !== null ? `${meta.latest_score}%` : 'Done';
                    badgesHtml += `<span class="badge-benchmarked-score" data-model="${id}" style="font-size:0.62rem; background:rgba(99,102,241,0.22); color:#a5b4fc; border:1px solid rgba(99,102,241,0.45); padding:1px 6px; border-radius:4px; margin-left:4px; font-weight:700; cursor:pointer; display:inline-flex; align-items:center; gap:2px; transition:all 0.15s ease-in-out;" title="Click to view latest test results for ${displayName} (${meta.benchmark_count}x runs | Last: ${meta.last_benchmarked_at || 'Unknown'})">📊 ${score} ↗</span>`;
                    badgesHtml += `<span class="badge-delete-benchmarks" data-model="${id}" style="font-size:0.62rem; background:rgba(239,68,68,0.15); color:#f87171; border:1px solid rgba(239,68,68,0.4); padding:1px 4px; border-radius:4px; margin-left:2px; font-weight:700; cursor:pointer; display:inline-flex; align-items:center; gap:2px; transition:all 0.15s ease-in-out;" title="Delete all saved benchmark results for ${displayName} (model is kept)">🗑</span>`;
                }

                span.innerHTML = `<span style="overflow:hidden; text-overflow:ellipsis; white-space:nowrap; flex:1; min-width:0;" title="${escapeHtml(displayName)}">${escapeHtml(displayName)}</span><span style="display:flex; gap:2px; flex-shrink:0;">${badgesHtml}</span>`;

                const badgeEl = span.querySelector('.badge-benchmarked-score');
                if (badgeEl) {
                    badgeEl.addEventListener('mouseenter', () => {
                        badgeEl.style.background = 'rgba(99,102,241,0.45)';
                        badgeEl.style.borderColor = '#818cf8';
                        badgeEl.style.color = '#ffffff';
                        badgeEl.style.transform = 'scale(1.05)';
                    });
                    badgeEl.addEventListener('mouseleave', () => {
                        badgeEl.style.background = 'rgba(99,102,241,0.22)';
                        badgeEl.style.borderColor = 'rgba(99,102,241,0.45)';
                        badgeEl.style.color = '#a5b4fc';
                        badgeEl.style.transform = 'none';
                    });
                    badgeEl.addEventListener('click', (e) => {
                        e.preventDefault();
                        e.stopPropagation();
                        navigateToModelTests(id);
                    });
                }

                const deleteBadge = span.querySelector('.badge-delete-benchmarks');
                if (deleteBadge) {
                    deleteBadge.addEventListener('mouseenter', () => {
                        deleteBadge.style.background = 'rgba(239,68,68,0.35)';
                        deleteBadge.style.borderColor = '#fca5a5';
                        deleteBadge.style.color = '#fecaca';
                        deleteBadge.style.transform = 'scale(1.05)';
                    });
                    deleteBadge.addEventListener('mouseleave', () => {
                        deleteBadge.style.background = 'rgba(239,68,68,0.15)';
                        deleteBadge.style.borderColor = 'rgba(239,68,68,0.4)';
                        deleteBadge.style.color = '#f87171';
                        deleteBadge.style.transform = 'none';
                    });
                    deleteBadge.addEventListener('click', (e) => {
                        e.preventDefault();
                        e.stopPropagation();
                        deleteModelBenchmarks(id, displayName);
                    });
                }

                item.appendChild(input);
                item.appendChild(span);
                return item;
            }

            // Local Models
            availableModels.forEach((model) => {
                const item = createModelCheckboxItem(model, model, false, false);
                modelCheckboxes.appendChild(item);
            });

            // Online Models (both active selected and any past benchmarked online models)
            const activeOnlineIds = new Set(onlineModels.map((m) => m.id));
            const pastOnlineModels = [];
            for (const [trackedId, tMeta] of Object.entries(trackedMap)) {
                if (
                    isOnlineModelName(trackedId) &&
                    !activeOnlineIds.has(trackedId) &&
                    tMeta.benchmark_count > 0
                ) {
                    pastOnlineModels.push({
                        id: trackedId,
                        name: trackedId,
                        label: trackedId,
                        free: false,
                    });
                }
            }

            const allDisplayOnline = [...onlineModels, ...pastOnlineModels];
            if (allDisplayOnline.length > 0) {
                const sep = document.createElement('div');
                sep.className = 'online-models-separator';
                sep.style.fontSize = '0.7rem';
                sep.style.fontWeight = '700';
                sep.style.color = 'var(--color-primary)';
                sep.style.margin = '0.75rem 0 0.35rem 0';
                sep.style.textTransform = 'uppercase';
                sep.style.letterSpacing = '0.05em';
                sep.textContent = '- Online Provider Models -';
                modelCheckboxes.appendChild(sep);

                allDisplayOnline.forEach((m) => {
                    const item = createModelCheckboxItem(m.id, m.label || m.name, true, !!m.free);
                    modelCheckboxes.appendChild(item);
                });
            }

            // Update UI count badges
            const totalModels = availableModels.length + allDisplayOnline.length;
            const cntAll = document.getElementById('cnt-models-all');
            const cntNew = document.getElementById('cnt-models-new');
            const cntBench = document.getElementById('cnt-models-benchmarked');
            if (cntAll) cntAll.textContent = totalModels;
            if (cntNew) cntNew.textContent = newCount;
            if (cntBench) cntBench.textContent = benchCount;
            
            populateModelSwitcher(availableModels);
            logToTerminal(`Discovered ${availableModels.length} local + ${onlineModels.length} online models (${newCount} new, ${benchCount} benchmarked)`, 'success');
        } catch (err) {
            logToTerminal(`Failed to discover models: ${err.message}`, 'error');
        }
    }

    // Fetch and display available tests in configurations sidebar
    async function loadTests() {
        try {
            logToTerminal("Fetching available test cases...");
            const res = await fetch('/api/tests');
            const data = await res.json();
            
            const tests = data.tests || [];
            testCheckboxes.innerHTML = '';
            
            if (tests.length === 0) {
                testCheckboxes.innerHTML = `<div style="color:var(--text-muted);font-size:0.8rem;padding:0.5rem;">No test cases detected</div>`;
                return;
            }

            tests.forEach((test) => {
                const item = document.createElement('label');
                item.className = 'checkbox-item';
                
                const input = document.createElement('input');
                input.type = 'checkbox';
                input.value = test.id;
                input.checked = true;
                
                const span = document.createElement('span');
                span.className = 'checkbox-label';
                span.textContent = `${test.category.toUpperCase()}: ${test.label}`;
                
                item.appendChild(input);
                item.appendChild(span);
                testCheckboxes.appendChild(item);
            });

            await applyResumeDeselect();

            logToTerminal(`Loaded ${tests.length} benchmark test cases`, 'success');
        } catch (err) {
            logToTerminal(`Failed to load tests: ${err.message}`, 'error');
        }
    }

    // Populate the benchmark GROUP multi-select from the backend's universal grader.
    async function loadBenchmarkGroups() {
        if (!groupCheckboxes) return;
        try {
            const res = await fetch('/api/benchmark/groups', { signal: AbortSignal.timeout(5000) });
            if (!res.ok) throw new Error(`status ${res.status}`);
            const data = await res.json();
            const groups = Array.isArray(data) ? data : (data.groups || []);
            groupCheckboxes.innerHTML = '';
            if (groups.length === 0) {
                groupCheckboxes.innerHTML = `<div style="color:var(--text-muted);font-size:0.8rem;padding:0.5rem;">No groups available</div>`;
                return;
            }
            groups.forEach((grp) => {
                const item = document.createElement('label');
                item.className = 'checkbox-item';
                const input = document.createElement('input');
                input.type = 'checkbox';
                input.value = grp;
                input.checked = false; // default: run all groups
                const span = document.createElement('span');
                span.className = 'checkbox-label';
                span.textContent = grp.replace(/_/g, ' ');
                item.appendChild(input);
                item.appendChild(span);
                groupCheckboxes.appendChild(item);
            });
            logToTerminal(`Loaded ${groups.length} benchmark groups`, 'success');
        } catch (err) {
            logToTerminal(`Failed to load benchmark groups: ${err.message}`, 'error');
            groupCheckboxes.innerHTML = `<div style="color:var(--text-muted);font-size:0.8rem;padding:0.5rem;">Could not load groups</div>`;
        }
    }

    // Pre-deselect tests already completed for the selected model(s) when Resume is on.
    // Completed = present in the model's per-model result file (pass or fail). Re-checking
    // a test (or turning Resume off) re-includes it for a (re)run that overwrites it.
    async function applyResumeDeselect() {
        const resumeChk = document.getElementById('chk-resume');
        if (!resumeChk || !resumeChk.checked) return;
        const selected = getSelectedModels();
        if (selected.length !== 1) return; // only meaningful for a single-model run
        const model = selected[0];
        try {
            const res = await fetch(`/api/benchmarks/completed?model=${encodeURIComponent(model)}`);
            if (!res.ok) return;
            const data = await res.json();
            const done = new Set(data.completed || []);
            if (done.size === 0) return;
            let deselected = 0;
            testCheckboxes.querySelectorAll('input[type="checkbox"]').forEach((box) => {
                if (done.has(box.value)) {
                    box.checked = false;
                    deselected++;
                }
            });
            if (deselected > 0) {
                logToTerminal(`Resume: pre-deselected ${deselected} already-completed tests for ${model}`, 'info');
            }
        } catch (err) {
            /* non-fatal: resume still enforced server-side */
        }
    }

    function wireResumeDeselect() {
        const resumeChk = document.getElementById('chk-resume');
        if (resumeChk) {
            resumeChk.addEventListener('change', () => {
                if (resumeChk.checked) {
                    applyResumeDeselect();
                } else {
                    // Re-enable every test when resume is turned off.
                    testCheckboxes.querySelectorAll('input[type="checkbox"]').forEach((box) => { box.checked = true; });
                }
            });
            // Re-apply whenever the model selection changes.
            modelCheckboxes.querySelectorAll('input[type="checkbox"]').forEach((box) => {
                box.addEventListener('change', () => applyResumeDeselect());
            });
        }
    }
    function escapeHtml(s) {
        return String(s == null ? '' : s)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#39;');
    }

    function gradeForScore(score) {
        if (score === undefined || score === null || Number.isNaN(Number(score))) return '—';
        const p = Number(score);
        if (p >= 90) return 'A';
        if (p >= 80) return 'B';
        if (p >= 70) return 'C';
        if (p >= 60) return 'D';
        return 'F';
    }

    let ALL_TESTS = [];
    let TEST_BROWSER_FILTER = { q: '', kind: 'all', status: 'all', model: '' };

    const TEST_KIND_ICON = { text: '📝', image: '🖼️', html: '🌐', node: '⚡' };

    function _testCardHtml(t) {
        const kind = (t.kind || 'text').toLowerCase();
        const icon = TEST_KIND_ICON[kind] || '📝';
        const atts = t.attachments || [];
        const attBadges = atts.length
            ? `<span class="test-card-att">📎 ${atts.length}</span>`
            : '';

        let runsBadge = '';
        if (t.models_tested_count > 0) {
            const passedInfo = `${t.models_passed_count}/${t.models_tested_count} passed`;
            runsBadge = `<span class="test-card-runs tested" title="Tested on ${t.models_tested_count} model(s): ${escapeHtml((t.models_tested || []).join(', '))}">⚡ ${t.models_tested_count} model${t.models_tested_count === 1 ? '' : 's'} (${passedInfo})</span>`;
        } else {
            runsBadge = `<span class="test-card-runs none" title="Not tested on any models yet">⚪ No runs</span>`;
        }

        const outdatedBadge = t.is_out_of_date
            ? `<span class="test-card-outdated" title="Test was updated since previous benchmark runs (${escapeHtml((t.out_of_date_models || []).join(', '))})">⚠️ Outdated</span>`
            : '';

        const ratingsHtml = _testCardRatingsHtml(t);

        return `<div class="test-card" data-test-id="${t.id}" role="button" tabindex="0">
            <div class="test-card-top">
                <span class="kind-badge kind-${kind}">${icon} ${kind}</span>
                ${attBadges}
                ${outdatedBadge}
            </div>
            <div class="test-card-label" title="${escapeHtml(t.label)}">${escapeHtml(t.label)}</div>
            ${ratingsHtml}
            <div class="test-card-footer">
                <div class="test-card-cat">${escapeHtml((t.category || '').toUpperCase())}</div>
                ${runsBadge}
                <button class="test-card-run-btn" title="Run the winning model's code in an expanded window / terminal">▶ Run</button>
            </div>
        </div>`;
    }

    function _testCardRatingsHtml(t) {
        const tested = t.models_tested || [];
        const scores = t.models_scores || {};
        if (!tested.length) return '';
        const saved = _loadHumanRatings(t.id) || {};
        const tokens = t.models_tokens || {};
        const latencies = t.models_latency || {};
        const speeds = t.models_speed || {};
        const passed = t.models_passed || [];
        // A "physical" score is a concrete benchmark score that clears the
        // passing bar — only then is a model genuinely the winner. If no model
        // has passed (all attempts failed the expectations / lint / run), the
        // benchmark is still "undefeated": show that state instead of crowning
        // a winner that doesn't exist.
        const winners = tested.filter((m) => passed.includes(m));
        if (!winners.length) {
            const bestScore = tested
                .map((m) => ({ m, s: Number(scores[m]) || 0 }))
                .sort((a, b) => b.s - a.s)[0];
            const bestTxt = bestScore ? `${Math.round(bestScore.s)}` : '—';
            return `<div class="test-card-ratings">
                <div class="top-rated undefeated" title="No model has passed this benchmark yet — it's still undefeated">
                    <span class="top-rated-label">🛡️ Unbeaten</span>
                    <span class="top-rated-score">best attempt ${bestTxt}</span>
                </div>
            </div>`;
        }
        // Winning model: highest total points — benchmark score plus the
        // human star bonus (30/star, 15/half star, 150 for all five).
        // Unrated models rank on their plain benchmark score.
        const best = winners.slice().sort((a, b) =>
            _modelTotalPoints(b, scores, saved) - _modelTotalPoints(a, scores, saved)
        )[0];
        const bestScore = scores[best] != null ? Math.round(Number(scores[best])) : '—';
        const starCount = saved[best] || 0;
        const totalPts = _modelTotalPoints(best, scores, saved);
        const starsHtml = starCount > 0
            ? `<span class="top-rated-human" title="Human rating: ${starCount}/5 (+${_starPoints(starCount)} pts)">${_starDisplayHtml(starCount)}</span>`
            : `<span class="top-rated-human unrated" title="No human rating yet">☆☆☆☆☆</span>`;
        const scoreHtml = starCount > 0
            ? `<span class="top-rated-score" title="${bestScore} code + ${_starPoints(starCount)} star points = ${totalPts} ranking points">${totalPts} pts</span>`
            : `<span class="top-rated-score" title="Benchmark score (no human rating yet)">score ${bestScore}</span>`;
        const specParts = [];
        if (tokens[best] != null && Number.isFinite(Number(tokens[best]))) {
            specParts.push(`⚡ ${Number(tokens[best]).toLocaleString()} tok`);
        }
        const timeSec = latencies[best] != null ? Number(latencies[best]) : null;
        if (timeSec != null && Number.isFinite(timeSec)) {
            specParts.push(`⏱ ${timeSec >= 60 ? (timeSec / 60).toFixed(1) + 'm' : timeSec.toFixed(1) + 's'}`);
        }
        if (speeds[best] != null && Number.isFinite(Number(speeds[best]))) {
            specParts.push(`🚀 ${Number(speeds[best]).toFixed(1)} tok/s`);
        }
        const specsHtml = specParts.length
            ? `<span class="top-rated-specs">${specParts.join(' · ')}</span>`
            : '';
        return `<div class="test-card-ratings">
            <div class="top-rated" title="Winning model: ${escapeHtml(best)} — click to preview/rate/view all models">
                <span class="top-rated-label">🏆 ${escapeHtml(truncateModelName(best))}</span>
                ${scoreHtml}
                ${starsHtml}
            </div>
            ${specsHtml}
        </div>`;
    }

    function _loadHumanRatings(testId) {
        try {
            const raw = localStorage.getItem('alpaca_human_ratings');
            if (!raw) return {};
            const all = JSON.parse(raw);
            return all[testId] || {};
        } catch (e) {
            return {};
        }
    }

    // Human star ratings convert into ranking points on top of the benchmark
    // (code) score: each full star is worth 30 points, a half star 15, so all
    // 5 stars add up to 150. A model's ranking total for a benchmark is its
    // benchmark score plus its star points; unrated models simply keep their
    // raw code score.
    const STAR_POINTS_PER_STAR = 30;

    function _starPoints(stars) {
        return Math.round((Number(stars) || 0) * STAR_POINTS_PER_STAR);
    }

    function _modelTotalPoints(model, scores, ratings) {
        const codeScore = Number(scores ? scores[model] : 0) || 0;
        return Math.round(codeScore) + _starPoints(ratings ? ratings[model] : 0);
    }

    // Run the winning model's generated code from a Test Browser card directly,
    // mirroring the featured "winner" action inside the preview modal: HTML/JS UI
    // opens in a new browser window, non-HTML UI launches the sandbox viewer,
    // and CLI code runs in the terminal. Rerunning a benchmark is done from the
    // preview modal ("Rerun for selected models") where the stats table lives.
    async function runWinningFromCard(t, anchorBtn) {
        const originalLabel = anchorBtn.textContent;
        anchorBtn.disabled = true;
        anchorBtn.innerHTML = '<span class="loader"></span>';
        try {
            const res = await fetch(`/api/tests/${encodeURIComponent(t.id)}/responses`);
            if (!res.ok) throw new Error('Failed to load stored results');
            const data = await res.json();
            const responses = Array.isArray(data) ? data : (data.responses || []);
            const scores = t.models_scores || {};
            // Same ranking as the card winner: benchmark score + star bonus.
            const savedRatings = _loadHumanRatings(t.id) || {};
            let winnerName = null;
            (t.models_tested || []).forEach((m) => {
                if (winnerName === null) winnerName = m;
                else if (_modelTotalPoints(m, scores, savedRatings) > _modelTotalPoints(winnerName, scores, savedRatings)) winnerName = m;
            });
            const winnerRow = winnerName === null || !responses.length
                ? null
                : responses.find((r) => r.model === winnerName) || responses[0];
            if (!winnerRow || !winnerRow.response) {
                showToast('No stored result to run for this benchmark yet.', 'error');
                return;
            }
            const isUi = t.type === 'ui';
            const isHtml = !!winnerRow.is_html;
            if (isHtml) {
                openExpandedRunner(winnerRow.response, winnerRow.model, winnerRow.thinking);
            } else if (isUi) {
                openUiViewer(winnerRow.response, winnerRow.model, winnerRow.thinking);
            } else {
                openExpandedRunner(winnerRow.response, winnerRow.model, winnerRow.thinking);
            }
        } catch (err) {
            showToast(`Failed to run winning result: ${err.message}`, 'error');
        } finally {
            anchorBtn.disabled = false;
            anchorBtn.textContent = originalLabel;
        }
    }

    function _saveHumanRating(testId, model, val) {
        try {
            let all = {};
            const raw = localStorage.getItem('alpaca_human_ratings');
            if (raw) all = JSON.parse(raw) || {};
            if (!all[testId]) all[testId] = {};
            all[testId][model] = val;
            localStorage.setItem('alpaca_human_ratings', JSON.stringify(all));
        } catch (e) { /* storage unavailable — ignore */ }
    }

    function _loadAllHumanRatings() {
        try {
            const raw = localStorage.getItem('alpaca_human_ratings');
            return raw ? JSON.parse(raw) || {} : {};
        } catch (e) {
            return {};
        }
    }

    // Render 5 star slots with fractional fill (0.0–5.0 in 0.5 steps).
    // Each slot draws an SVG star twice: a gray base plus a gold overlay
    // clipped with clip-path inset, so a 0.5 rating is an exact half star
    // (font-metrics independent — text-clipping rendered ~2/3 instead).
    const STAR_SVG_PATH = 'M12 2.2l2.98 6.04 6.67.97-4.83 4.7 1.14 6.64L12 17.42l-5.96 3.13 1.14-6.64-4.83-4.7 6.67-.97z';

    function _starDisplayHtml(rating, interactive) {
        const r = Math.max(0, Math.min(5, Number(rating) || 0));
        const slots = [];
        for (let pos = 1; pos <= 5; pos++) {
            const frac = Math.max(0, Math.min(1, r - (pos - 1)));
            const attrs = interactive
                ? ' role="button" tabindex="0" title="Rate 1–5 (click left half for .5)"'
                : '';
            slots.push(
                `<span class="star-slot" data-pos="${pos}"${attrs} aria-label="${r}/5">` +
                    `<svg class="star-svg" viewBox="0 0 24 24" aria-hidden="true">` +
                        `<path class="star-bg" d="${STAR_SVG_PATH}"></path>` +
                        (frac > 0
                            ? `<path class="star-fill" style="clip-path:inset(0 ${(100 - frac * 100).toFixed(2)}% 0 0)" d="${STAR_SVG_PATH}"></path>`
                            : '') +
                    `</svg>` +
                `</span>`
            );
        }
        return slots.join('');
    }

    // Wire click/keyboard interaction on .star-slot elements inside a container.
    // Clicking the left half of a star sets that star minus 0.5; the right half
    // sets the full star. onChange receives the new rating (0.5 increments).
    function _bindStarSlots(container, onChange) {
        container.querySelectorAll('.star-slot').forEach((slot) => {
            const pos = Number(slot.dataset.pos);
            const set = (val) => onChange(Math.max(0, Math.min(5, val)));
            const click = (e) => {
                e.stopPropagation();
                const rect = slot.getBoundingClientRect();
                const half = (e.clientX - rect.left) < rect.width / 2;
                set(half ? pos - 0.5 : pos);
            };
            slot.addEventListener('click', click);
            slot.addEventListener('keydown', (e) => {
                if (e.key === 'ArrowLeft') { e.preventDefault(); set(pos - 0.5); }
                else if (e.key === 'ArrowRight') { e.preventDefault(); set(pos); }
                else if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); set(pos); }
            });
        });
    }

    // Active preview-modal context: stored so a rating change can re-rank and
    // re-render the featured winning result (and its screenshot) in place.
    let ACTIVE_PREVIEW = null; // { testId, responses }

    function renderTestPreviewRatings(t) {
        const el = document.getElementById('test-preview-ratings');
        if (!el) return;
        const tested = t.models_tested || [];
        const scores = t.models_scores || {};
        if (!tested.length) {
            el.innerHTML = '';
            return;
        }
const saved = _loadHumanRatings(t.id) || {};
        // Rank by total points (benchmark score + star bonus, 30/star).
        // Unrated models keep their plain benchmark score.
        const ranked = tested.slice().sort((a, b) =>
            _modelTotalPoints(b, scores, saved) - _modelTotalPoints(a, scores, saved));
        const rows = ranked.map((m, rIdx) => {
            const rating = saved[m] || 0;
            const score = scores[m] != null ? `${Math.round(Number(scores[m]))}` : '—';
            const starPts = _starPoints(rating);
            const totalPts = _modelTotalPoints(m, scores, saved);
            const stars = _starDisplayHtml(rating, true);
            const rank = rIdx === 0 ? '🥇' : rIdx === 1 ? '🥈' : rIdx === 2 ? '🥉' : `${rIdx + 1}`;
            const ptsHtml = rating > 0
                ? `<span class="rating-score" title="${score} code + ${starPts} star points">${totalPts} pts</span>`
                : `<span class="rating-score" title="Benchmark score (no human rating yet)">${score}</span>`;
            return `<div class="rating-row" data-model="${escapeHtml(m)}">
                <span class="rating-rank">${rank}</span>
                <span class="rating-model" title="${escapeHtml(m)}">${escapeHtml(m)}</span>
                <span class="rating-stars">${stars}</span>
                ${ptsHtml}
            </div>`;
        }).join('');

        el.innerHTML = `<div class="modal-section-title">Human Aesthetic Rating <span style="color:var(--text-muted);font-size:0.72rem;font-weight:400;">(per model · 30 pts/star — rated models rank on code score + stars)</span></div>
            <div class="rating-list">${rows}</div>`;
        el.querySelectorAll('.rating-row').forEach((row) => {
            const model = row.dataset.model;
            _bindStarSlots(row, (val) => {
                _saveHumanRating(t.id, model, val);
                renderTestPreviewRatings(t);
                // Stars can flip the #1 — refresh the featured winning
                // result (and its screenshot) inside the open dialog.
                if (ACTIVE_PREVIEW && ACTIVE_PREVIEW.testId === t.id) {
                    renderFeaturedWinner(t, ACTIVE_PREVIEW.responses);
                }
                renderRatingsBoard();
                renderTestBrowser();
            });
        });
    }

    // Build / rebuild the featured "🏆 Winning result" block of the preview
    // modal. Winner = highest total points (code score + star bonus), the
    // same ranking shown on the test card. secOverride lets the initial
    // render target a not-yet-attached section.
    function renderFeaturedWinner(t, responses, secOverride) {
        const sec = secOverride || document.querySelector('#test-preview-overlay .model-outputs-section');
        if (!sec) return;
        const prev = sec.querySelector('.winner-result');
        if (prev) prev.remove();
        const passedList = t.models_passed || [];
        const scores = t.models_scores || {};
        const savedRatings = _loadHumanRatings(t.id) || {};
        const candidates = (t.models_tested || []).filter((m) => passedList.includes(m));
        if (!candidates.length) candidates.push(...(t.models_tested || []));
        let winnerName = null;
        candidates.forEach((m) => {
            if (winnerName === null) winnerName = m;
            else if (_modelTotalPoints(m, scores, savedRatings) > _modelTotalPoints(winnerName, scores, savedRatings)) winnerName = m;
        });
        const winnerRow = (winnerName === null || !responses.length)
            ? null
            : (responses.find((r) => r.model === winnerName) || responses[0]);
        const hasPassingWinner = winnerRow && passedList.includes(winnerRow.model);
        if (!winnerRow) return;
        const isUi = t.type === 'ui';
        const isHtml = !!winnerRow.is_html;
        const starPts = _starPoints(savedRatings[winnerRow.model]);
        const winnerWrap = document.createElement('div');
        winnerWrap.className = hasPassingWinner ? 'winner-result' : 'winner-result undefeated';
        winnerWrap.dataset.winnerModel = winnerRow.model;
        const winnerTitle = document.createElement('div');
        winnerTitle.className = 'test-preview-note';
        winnerTitle.textContent = hasPassingWinner
            ? `🏆 Winning result: ${winnerRow.model}` +
                (winnerRow.score != null
                    ? ` (${Math.round(Number(winnerRow.score))} code` + (starPts > 0 ? ` + ${starPts}★ = ${_modelTotalPoints(winnerRow.model, scores, savedRatings)} pts` : '') + ')'
                    : '') +
                (isUi ? ` — ${isHtml ? 'HTML/JS UI' : 'UI'} result` : ' — CLI result')
            : `🛡️ Unbeaten — no model has passed this benchmark yet (best attempt: ${winnerRow.model})`;
        winnerWrap.appendChild(winnerTitle);
        const winnerBar = document.createElement('div');
        winnerBar.className = 'att-run-bar';
        const winRun = document.createElement('button');
        winRun.className = 'btn-run-code';
        if (isHtml) {
            winRun.textContent = '▶ Play (expanded)';
            winRun.addEventListener('click', () => openExpandedRunner(winnerRow.response, winnerRow.model, winnerRow.thinking));
        } else if (isUi) {
            winRun.textContent = '🖥 View UI (expanded)';
            winRun.addEventListener('click', () => openUiViewer(winnerRow.response, winnerRow.model, winnerRow.thinking));
        } else {
            winRun.textContent = '▶ Run (terminal)';
            winRun.addEventListener('click', () => openExpandedRunner(winnerRow.response, winnerRow.model, winnerRow.thinking));
        }
        winnerBar.appendChild(winRun);
        if (winnerRow.last_run) {
            const winDate = document.createElement('span');
            winDate.style.marginLeft = '0.5rem';
            winDate.style.color = 'var(--text-muted)';
            winDate.style.fontSize = '0.72rem';
            winDate.textContent = `ran ${String(winnerRow.last_run).replace('T', ' ').slice(0, 16)}`;
            winnerBar.appendChild(winDate);
        }
        winnerWrap.appendChild(winnerBar);
        if (isUi && winnerRow.screenshot) {
            const shot = document.createElement('img');
            shot.className = 'test-preview-screenshot';
            shot.src = `data:image/png;base64,${winnerRow.screenshot}`;
            shot.alt = `${winnerRow.model} screenshot`;
            shot.title = 'Click to expand';
            shot.addEventListener('click', () => openScreenshotLightbox(`data:image/png;base64,${winnerRow.screenshot}`));
            winnerWrap.appendChild(shot);
            const shotHint = document.createElement('div');
            shotHint.className = 'test-preview-note';
            shotHint.style.fontSize = '0.68rem';
            shotHint.style.color = 'var(--text-muted)';
            shotHint.textContent = 'Click the screenshot to open it in an expanded window.';
            winnerWrap.appendChild(shotHint);
        } else if (winnerRow.code_output) {
            const outPre = document.createElement('pre');
            outPre.className = 'code-display-block test-preview-code';
            outPre.textContent = winnerRow.code_output.slice(0, 4000) +
                (winnerRow.code_output.length > 4000 ? '\n… (truncated)' : '');
            winnerWrap.appendChild(outPre);
        }
        sec.insertBefore(winnerWrap, sec.firstChild);
    }
    function renderRatingsBoard() {
        const board = document.getElementById('test-ratings-board');
        if (!board) return;
        const all = _loadAllHumanRatings();
        const catAcc = {};  // category -> { model -> {sum, count, scoreSum} }
        let ratedAny = false;

        ALL_TESTS.forEach((t) => {
            const perTest = all[t.id];
            if (!perTest) return;
            const cat = (t.category || 'other').toUpperCase();
            const scores = t.models_scores || {};
            Object.entries(perTest).forEach(([model, val]) => {
                const r = Number(val);
                if (!r) return;
                ratedAny = true;
                if (!catAcc[cat]) catAcc[cat] = {};
                if (!catAcc[cat][model]) catAcc[cat][model] = { sum: 0, count: 0, scoreSum: 0 };
                catAcc[cat][model].sum += r;
                catAcc[cat][model].count += 1;
                const sc = Number(scores[model]);
                if (!Number.isNaN(sc)) catAcc[cat][model].scoreSum += sc;
            });
        });

        if (!ratedAny) {
            board.innerHTML = `<h4>🏆 Top Rated</h4>
                <div class="ratings-board-overall"><span class="overall-label">No human ratings yet</span>
                <span class="overall-count">Click the ★ stars on any test card to rate each model's output.</span></div>`;
            return;
        }

        const catWinners = Object.entries(catAcc).map(([cat, models]) => {
            const best = Object.entries(models)
                .map(([model, a]) => ({
                    model,
                    rating: a.sum / a.count,
                    score: a.count ? a.scoreSum / a.count : null,
                    count: a.count,
                }))
                .sort((x, y) => y.rating - x.rating || (y.score || 0) - (x.score || 0))[0];
            return { cat, ...best };
        }).sort((a, b) => b.rating - a.rating || (b.score || 0) - (a.score || 0));

        const overall = (() => {
            const agg = {};
            Object.values(catAcc).forEach((models) => {
                Object.entries(models).forEach(([model, a]) => {
                    if (!agg[model]) agg[model] = { sum: 0, count: 0, scoreSum: 0 };
                    agg[model].sum += a.sum;
                    agg[model].count += a.count;
                    agg[model].scoreSum += a.scoreSum;
                });
            });
            return Object.entries(agg).map(([model, a]) => ({
                model,
                rating: a.sum / a.count,
                score: a.count ? a.scoreSum / a.count : null,
                count: a.count,
            })).sort((x, y) => y.rating - x.rating || (y.score || 0) - (x.score || 0))[0];
        })();

        const starsFor = (rating) => _starDisplayHtml(rating);
        const scoreFor = (s) => (s != null ? `${Math.round(s)}` : '—');

        board.innerHTML = `<h4>🏆 Top Rated</h4>
            <div class="ratings-board-overall">
                <span class="overall-label">🏆 Overall: ${escapeHtml(truncateModelName(overall.model))}</span>
                <span class="top-rated-score">score ${scoreFor(overall.score)}</span>
                <span class="top-rated-human">${starsFor(overall.rating)} (${overall.rating.toFixed(1)})</span>
                <span class="overall-count">${overall.count} rated across ${catWinners.length} categories</span>
            </div>
            <div class="ratings-board-cats">
                ${catWinners.map((w) => `<div class="ratings-board-cat">
                    <span class="ratings-board-cat-name">${escapeHtml(w.cat)}</span>
                    <span class="ratings-board-cat-entry">
                        <span class="top-rated-label" title="${escapeHtml(w.model)}">${escapeHtml(truncateModelName(w.model))}</span>
                        <span class="top-rated-score">score ${scoreFor(w.score)}</span>
                        <span class="top-rated-human">${starsFor(w.rating)}</span>
                    </span>
                </div>`).join('')}
            </div>`;
    }

    function renderTestBrowser() {
        const grid = document.getElementById('test-browser-grid');
        if (!grid) return;
        const q = (TEST_BROWSER_FILTER.q || '').trim().toLowerCase();
        const kind = TEST_BROWSER_FILTER.kind || 'all';
        const status = TEST_BROWSER_FILTER.status || 'all';
        const modelFilter = TEST_BROWSER_FILTER.model || '';
        const filtered = ALL_TESTS.filter((t) => {
            if (kind !== 'all' && (t.kind || 'text').toLowerCase() !== kind) return false;
            if (status === 'tested' && (!t.models_tested_count || t.models_tested_count === 0)) return false;
            if (status === 'untested' && (t.models_tested_count > 0)) return false;
            if (status === 'outdated' && !t.is_out_of_date) return false;
            if (modelFilter) {
                const tested = (t.models_tested || []);
                const hit = tested.some(m => m === modelFilter || m.includes(modelFilter) || modelFilter.includes(m));
                if (!hit) return false;
            }
            if (!q) return true;
            return (
                (t.label || '').toLowerCase().includes(q) ||
                (t.id || '').toLowerCase().includes(q) ||
                (t.category || '').toLowerCase().includes(q)
            );
        });
        const countEl = document.getElementById('test-browser-count');
        if (countEl) countEl.textContent = `${filtered.length} test${filtered.length === 1 ? '' : 's'}`;
        renderRatingsBoard();
        if (!filtered.length) {
            grid.innerHTML = `<div class="empty-state">No tests match your filter.</div>`;
            return;
        }
        grid.innerHTML = filtered.map(_testCardHtml).join('');
        grid.querySelectorAll('.test-card').forEach((card) => {
            const id = card.dataset.testId;
            card.addEventListener('click', (e) => {
                if (e.target.closest('.test-card-run-btn')) return;
                openTestPreview(id);
            });
            card.addEventListener('keydown', (e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault();
                    openTestPreview(id);
                }
            });
            const runBtn = card.querySelector('.test-card-run-btn');
            if (runBtn) {
                runBtn.addEventListener('click', (e) => {
                    e.stopPropagation();
                    e.preventDefault();
                    const t = ALL_TESTS.find((x) => x.id === id);
                    if (t) runWinningFromCard(t, runBtn);
                });
            }
        });
    }

    async function loadTestBrowser() {
        const grid = document.getElementById('test-browser-grid');
        if (!grid) return;
        grid.innerHTML = `<div class="loading-spinner">Loading tests…</div>`;
        try {
            const res = await fetch('/api/tests');
            const data = await res.json();
            ALL_TESTS = data.tests || [];
            populateTestModelSelect();
            renderTestBrowser();
        } catch (err) {
            grid.innerHTML = `<div class="empty-state">Failed to load tests: ${escapeHtml(err.message)}</div>`;
        }
    }

    function populateTestModelSelect() {
        const sel = document.getElementById('test-model-select');
        if (!sel) return;
        const models = new Set();
        ALL_TESTS.forEach(t => (t.models_tested || []).forEach(m => models.add(m)));
        const sorted = [...models].sort((a, b) => a.localeCompare(b));
        const current = TEST_BROWSER_FILTER.model || '';
        sel.innerHTML = '<option value="">All Models</option>' + sorted.map(m => `<option value="${escapeHtml(m)}">${escapeHtml(m)}</option>`).join('');
        if (current && sorted.includes(current)) sel.value = current;
    }

    // Execute a code attachment (node/js) in a throwaway sandboxed iframe and pipe
    // its console output back into outEl. This is what makes the Test Browser "run"
    // complete code artifacts in the card, not just preview them statically.
    // --- Code sandbox terminal -------------------------------------------------
    // Text-based languages (Python/Node) run in a short-lived Docker container
    // (alpaca-sandbox image) streamed over SocketIO, giving a small interactive
    // terminal where multiple inputs/outputs can be exercised. HTML stays as a
    // live iframe.
    function detectLang(code) {
        const low = (code || '').toLowerCase();
        const py = (low.match(/def\s|import\s|print\(|self\./g) || []).length;
        const js = (low.match(/console\.log|function\s|=>>|require\(|document\./g) || []).length;
        if (py === 0 && js === 0) return 'python';
        return py >= js ? 'python' : 'node';
    }

    let _termBuilt = false;
    let _termOut = null;
    let _termInput = null;
    let _termLang = null;
    let _termOpen = false;
    // Readline-style input history shared by every terminal session:
    // Enter executes + clears, ArrowUp/ArrowDown walk through past inputs
    // (unsent draft is preserved), like a real shell.
    const _termHistory = [];
    let _termHistIdx = -1; // -1 = live input line, otherwise index into _termHistory
    let _termDraft = '';

    function buildTerminal() {
        if (_termBuilt) return;
        const overlay = document.createElement('div');
        overlay.id = 'sandbox-terminal-overlay';
        overlay.className = 'overlay-hidden';
        const panel = document.createElement('div');
        panel.className = 'sandbox-terminal-panel';
        const bar = document.createElement('div');
        bar.className = 'sandbox-terminal-bar';
        const title = document.createElement('span');
        title.className = 'sandbox-terminal-title';
        title.textContent = 'Terminal';
        _termLang = document.createElement('span');
        _termLang.className = 'sandbox-terminal-lang';
        const stopBtn = document.createElement('button');
        stopBtn.className = 'btn-run-code';
        stopBtn.textContent = 'Stop';
        stopBtn.addEventListener('click', () => { socket.emit('sandbox_kill'); appendTerm('\n[stopped]\n'); });
        const closeBtn = document.createElement('button');
        closeBtn.className = 'btn-run-code';
        closeBtn.textContent = 'Close';
        closeBtn.addEventListener('click', () => closeTerminal());
        bar.appendChild(title);
        bar.appendChild(_termLang);
        bar.appendChild(stopBtn);
        bar.appendChild(closeBtn);
        _termOut = document.createElement('pre');
        _termOut.className = 'sandbox-terminal-out';
        _termInput = document.createElement('input');
        _termInput.className = 'sandbox-terminal-in';
        _termInput.type = 'text';
        _termInput.placeholder = 'Type input and press Enter (multiple interactions supported)';
        _termInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') {
                const v = _termInput.value;
                if (v.trim() !== '') _termHistory.push(v);
                _termHistIdx = -1;
                _termDraft = '';
                _termInput.value = '';
                appendTerm('> ' + v + '\n');
                socket.emit('sandbox_input', { text: v + '\n' });
            } else if (e.key === 'ArrowUp') {
                e.preventDefault();
                if (!_termHistory.length) return;
                if (_termHistIdx === -1) {
                    _termDraft = _termInput.value;
                    _termHistIdx = _termHistory.length - 1;
                } else if (_termHistIdx > 0) {
                    _termHistIdx -= 1;
                }
                _termInput.value = _termHistory[_termHistIdx];
            } else if (e.key === 'ArrowDown') {
                e.preventDefault();
                if (_termHistIdx === -1) return;
                _termHistIdx += 1;
                if (_termHistIdx >= _termHistory.length) {
                    _termHistIdx = -1;
                    _termInput.value = _termDraft;
                    _termDraft = '';
                } else {
                    _termInput.value = _termHistory[_termHistIdx];
                }
            }
        });
        panel.appendChild(bar);
        panel.appendChild(_termOut);
        panel.appendChild(_termInput);
        overlay.appendChild(panel);
        document.body.appendChild(overlay);
        _termBuilt = true;
    }

    function appendTerm(text, isErr) {
        if (!_termOut) return;
        const span = document.createElement('span');
        span.textContent = text;
        if (isErr) span.style.color = '#ff7b72';
        _termOut.appendChild(span);
        _termOut.scrollTop = _termOut.scrollHeight;
    }

    // Pull the real code out of a model response. Model outputs are frequently
    // wrapped in markdown prose with a fenced code block; running the whole
    // response as code fails. Extract the fenced block (longest match wins),
    // falling back to the raw text when no fence is present.
    function extractRunnableCode(text) {
        const fences = [...text.matchAll(/```[^\n]*\n([\s\S]*?)```/g)];
        if (fences.length) {
            let best = '';
            for (const f of fences) {
                if (f[1].trim().length > best.length) best = f[1].trim();
            }
            return best;
        }
        return text;
    }

    // Pull the actual HTML document out of a model response. Model responses
    // frequently wrap the page in prose, markdown fences and stray comments
    // (e.g. "Here's the code:" or ```html fences); dumping that raw text into
    // an iframe srcdoc makes the page render with the preamble visible and the
    // markup in quirks mode. This finds the block that contains the real
    // <!doctype/<html> (or falls back to a <script>/<canvas> block) and trims
    // everything before the document start and after </html>.
    function extractHtmlDocument(text) {
        if (!text) return '';
        const fences = [...text.matchAll(/```[^\n]*\n([\s\S]*?)```/g)];
        let candidates = fences.map((f) => f[1].trim());
        if (!candidates.length) candidates = [text];
        let doc = candidates.find((c) => /<!doctype|<\s*html[\s>]/i.test(c)) || '';
        if (!doc) doc = candidates.find((c) => /<script|<\s*canvas/i.test(c)) || '';
        if (!doc) return '';
        const start = doc.search(/<!doctype|<\s*html[\s>]/i);
        if (start > 0) doc = doc.slice(start);
        const end = doc.search(/<\/html>[\s\S]*$/i);
        if (end >= 0) doc = doc.slice(0, end + 7);
        return doc.trim();
    }

    // Categories whose model responses typically contain runnable code/UI output.
    const CODE_UI_CATEGORIES = ['coding', 'gamedev', 'appdev', 'webdev', 'debugging', 'cpp', 'java', 'linux_admin', 'database'];

    function isCodeUiCategory(cat) {
        return CODE_UI_CATEGORIES.includes(cat);
    }

    // Infer which sandbox language to serve a response under.
    function inferServeLang(category, response) {
        if (category === 'webdev') return 'html';
        if (/\bdef\s|\bimport\s/.test(response)) return 'python';
        if (/console\.log|\bfunction\s/.test(response)) return 'node';
        return 'python';
    }

    // Graphical (X11) code — pygame/tkinter etc. — must be served through the noVNC
    // UI path (Xvfb + websockify), not serve_app, which expects an HTTP server on the
    // published port. serve_app on such code leaves nothing listening -> connection refused.
    function isGraphicalUiCode(response) {
        return /import\s+pygame|\bfrom\s+pygame\b|import\s+tkinter|\bfrom\s+tkinter\b|pygame\.init\s*\(|import\s+arcade|\bimport\s+pyglet/.test(response);
    }

    // Build a "Serve & View" button that launches the extracted code on a port and
    // opens it in a new browser tab. Also wires a Stop control bound to the container id.
    function createServeButton(response, category) {
        const btn = document.createElement('button');
        btn.className = 'btn btn-secondary btn-sm serve-view-btn';
        btn.textContent = '▶ Serve & View';
        btn.style.cssText = 'padding: 3px 10px; font-size: 0.7rem; cursor:pointer;';
        btn.addEventListener('click', async (e) => {
            e.stopPropagation();
            const code = extractRunnableCode(response || '');
            if (!code.trim()) {
                showToast('No runnable code found in response', 'error');
                return;
            }
            const lang = inferServeLang(category, code);
            btn.disabled = true;
            btn.textContent = '⏳ Serving...';
            try {
                const graphical = isGraphicalUiCode(code);
                if (graphical) {
                    // Pygame / other X11 apps: launch under Xvfb and stream via noVNC launcher.
                    const res = await fetch('/api/sandbox/serve_ui', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ code: code, lang: 'python' })
                    });
                    const data = await res.json();
                    if (!res.ok || data.error) {
                        throw new Error(data.error || `status ${res.status}`);
                    }
                    const launcherUrl = `/ui/launcher/${data.container_id}`;
                    showToast(`UI streaming on ${launcherUrl}`, 'success');
                    window.open(launcherUrl, '_blank');
                    addServeStopControl(btn, data.container_id);
                } else {
                    const res = await fetch('/api/sandbox/serve', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ code: code, lang: lang })
                    });
                    const data = await res.json();
                    if (!res.ok || data.error) {
                        throw new Error(data.error || `status ${res.status}`);
                    }
                    // Tunnel the app through the dashboard origin (port 5000) so it works
                    // from the LAN, a VPN, or an external/forwarded hostname alike — never
                    // assume a direct-to-sandbox-port URL (those are random and unreachable
                    // off the local machine).
                    const serveUrl = `/serve/${data.container_id}/`;
                    showToast(`Serving on ${serveUrl}`, 'success');
                    window.open(serveUrl, '_blank');
                    addServeStopControl(btn, data.container_id);
                }
            } catch (err) {
                showToast(`Serve failed: ${err.message}`, 'error');
            } finally {
                btn.disabled = false;
                btn.textContent = '▶ Serve & View';
            }
        });
        return btn;
    }

    function addServeStopControl(btn, containerId) {
        const parent = btn.parentNode;
        if (!parent) return;
        const existing = parent.querySelector('.serve-stop-btn');
        if (existing) existing.remove();
        const stopBtn = document.createElement('button');
        stopBtn.className = 'btn btn-danger btn-sm serve-stop-btn';
        stopBtn.textContent = '■ Stop';
        stopBtn.style.cssText = 'padding: 3px 10px; font-size: 0.7rem; cursor:pointer;';
        stopBtn.addEventListener('click', async (e) => {
            e.stopPropagation();
            try {
                const res = await fetch('/api/sandbox/stop_serve', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ container_id: containerId })
                });
                const data = await res.json();
                showToast(data.stopped ? 'Server stopped' : (`Stop failed: ${data.error || ''}`), data.stopped ? 'success' : 'error');
            } catch (err) {
                showToast(`Stop error: ${err.message}`, 'error');
            }
            stopBtn.remove();
        });
        parent.appendChild(stopBtn);
    }

    function openTerminal(code, langHint, initialInput) {
        buildTerminal();
        _termOut.textContent = '';
        const lang = langHint || detectLang(code);
        _termLang.textContent = lang;
        const ov = document.getElementById('sandbox-terminal-overlay');
        ov.classList.remove('overlay-hidden');
        ov.classList.add('open');
        _termOpen = true;
        _termInput.focus();
        socket.emit('sandbox_run', { code: code, lang: lang, input: initialInput || '' });
    }

    function closeTerminal() {
        const ov = document.getElementById('sandbox-terminal-overlay');
        if (ov) {
            ov.classList.add('overlay-hidden');
            ov.classList.remove('open');
        }
        _termOpen = false;
        socket.emit('sandbox_kill');
    }

    socket.on('sandbox_started', (d) => {
        if (_termLang && d && d.lang) _termLang.textContent = d.lang;
        appendTerm('[connected - ' + (d ? d.lang : '') + ' sandbox ready]\n');
    });
    socket.on('sandbox_out', (txt) => { appendTerm(txt || ''); });
    socket.on('sandbox_error', (txt) => { appendTerm('\n[error] ' + (txt || '') + '\n', true); });
    socket.on('sandbox_done', () => { appendTerm('\n[process exited]\n'); });

    // Run a model-produced response (finished code) directly in the browser. HTML/3js
    // renders live in a sandboxed iframe; node/js executes in a sandbox and streams its
    // console output. This satisfies "any finished code that is complete should be able
    // to be ran" for benchmark outputs, not just static test attachments.
    function openResponseRunner(responseText, label, thinkingText) {
        const overlay = document.getElementById('test-preview-overlay');
        const title = document.getElementById('test-preview-title');
        const meta = document.getElementById('test-preview-meta');
        const promptEl = document.getElementById('test-preview-prompt');
        const stage = document.getElementById('test-preview-stage');
        const expectedEl = document.getElementById('test-preview-expected');
        const downloadBtn = document.getElementById('btn-download-test');
        title.textContent = '▶ Run: ' + (label || 'response');
        meta.innerHTML = '<span class="kind-badge kind-node">RUN</span>';
        promptEl.textContent = 'Model-produced code (runnable in browser):';
        expectedEl.textContent = '';
        stage.innerHTML = '';
        const txt = responseText || '';
        const thinking = thinkingText || '';
        if (thinking) {
            const thWrap = document.createElement('div');
            thWrap.className = 'att-run-wrap thinking-wrap';
            const thBar = document.createElement('div');
            thBar.className = 'att-run-bar';
            const thBtn = document.createElement('button');
            thBtn.className = 'btn-run-code';
            thBtn.textContent = '🧠 Show thinking';
            const thPre = document.createElement('pre');
            thPre.className = 'code-display-block thinking-block';
            thPre.style.display = 'none';
            thPre.textContent = thinking;
            thBtn.addEventListener('click', () => {
                const hidden = thPre.style.display === 'none';
                thPre.style.display = hidden ? 'block' : 'none';
                thBtn.textContent = hidden ? '🧠 Hide thinking' : '🧠 Show thinking';
            });
            thBar.appendChild(thBtn);
            thWrap.appendChild(thBar);
            thWrap.appendChild(thPre);
            stage.appendChild(thWrap);
        }
        const isHtml = /<!doctype|<html|<script/i.test(txt);
        if (isHtml) {
            const note = document.createElement('div');
            note.className = 'test-preview-note';
            note.textContent = '▶ Running live HTML/3js output';
            stage.appendChild(note);
            const htmlDoc = extractHtmlDocument(txt) || extractRunnableCode(txt);
            const frame = document.createElement('iframe');
            frame.className = 'test-preview-iframe';
            frame.setAttribute('sandbox', 'allow-scripts allow-same-origin allow-popups');
            frame.srcdoc = htmlDoc;
            stage.appendChild(frame);
            const openTabBtn = document.createElement('button');
            openTabBtn.className = 'btn btn-secondary btn-sm';
            openTabBtn.textContent = '↗ Open in new tab';
            openTabBtn.style.marginTop = '0.5rem';
            openTabBtn.addEventListener('click', () => {
                const blob = new Blob([htmlDoc], { type: 'text/html' });
                const url = URL.createObjectURL(blob);
                window.open(url, '_blank');
                setTimeout(() => URL.revokeObjectURL(url), 60000);
            });
            stage.appendChild(openTabBtn);
        } else {
            const wrap = document.createElement('div');
            wrap.className = 'att-run-wrap';
            const bar = document.createElement('div');
            bar.className = 'att-run-bar';
            const runBtn = document.createElement('button');
            runBtn.className = 'btn-run-code';
            runBtn.textContent = '▶ Run';
            const uiBtn = document.createElement('button');
            uiBtn.className = 'btn-run-code';
            uiBtn.style.marginLeft = '0.5rem';
            uiBtn.textContent = '🖥 View UI';
            const out = document.createElement('div');
            out.className = 'code-run-output';
            out.textContent = 'Output appears here after Run.';
            const pre = document.createElement('pre');
            pre.className = 'code-display-block test-preview-code';
            const code = extractRunnableCode(txt);
            pre.textContent = code;
            const stdin = document.createElement('textarea');
            stdin.className = 'code-input-stdin';
            stdin.placeholder = 'Input (stdin) - one line per prompt()/input() call';
            stdin.rows = 2;
            bar.appendChild(runBtn);
            bar.appendChild(uiBtn);
            wrap.appendChild(bar);
            wrap.appendChild(stdin);
            wrap.appendChild(pre);
            wrap.appendChild(out);
            stage.appendChild(wrap);
            runBtn.addEventListener('click', () => openTerminal(code, null, stdin.value));
            uiBtn.addEventListener('click', () => openUiViewer(code, label, thinking));
        }
        if (downloadBtn) {
            downloadBtn.onclick = () => {
                const blob = new Blob([txt], { type: 'text/plain' });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = (label || 'response') + '.txt';
                a.click();
                URL.revokeObjectURL(url);
            };
        }
        overlay.classList.add('open');
    }

    // Open a model-produced result in an ACTUAL expanded window. HTML/3js games
    // are opened as a blob in a new browser window (fully playable, larger);
    // CLI/code results launch the full-screen sandbox terminal instead of the
    // in-modal preview. This backs the "▶ Play (expanded)" / "▶ Run (expanded)"
    // actions on the Test Browser winner/response rows.
    function openExpandedRunner(responseText, label, thinkingText) {
        const txt = responseText || '';
        const isHtml = /<!doctype|<html|<script/i.test(txt);
        if (isHtml) {
            const htmlDoc = extractHtmlDocument(txt) || extractRunnableCode(txt);
            const blob = new Blob([htmlDoc], { type: 'text/html' });
            const url = URL.createObjectURL(blob);
            const win = window.open(url, '_blank');
            setTimeout(() => URL.revokeObjectURL(url), 60000);
            if (!win) showToast('Popup blocked — allow popups for this site.', 'error');
        } else {
            const code = extractRunnableCode(txt);
            openTerminal(code, null, '');
        }
    }

    function openUiViewer(responseText, label, thinkingText) {
        const overlay = document.getElementById('test-preview-overlay');
        const title = document.getElementById('test-preview-title');
        const meta = document.getElementById('test-preview-meta');
        const promptEl = document.getElementById('test-preview-prompt');
        const stage = document.getElementById('test-preview-stage');
        const expectedEl = document.getElementById('test-preview-expected');
        const downloadBtn = document.getElementById('btn-download-test');
        title.textContent = '🖥 UI Launcher: ' + (label || 'app');
        meta.innerHTML = '<span class="kind-badge kind-node">UI</span>';
        promptEl.textContent = 'Model-produced UI code running live in a sandbox container (noVNC). Use the tools below to troubleshoot.';
        expectedEl.textContent = '';
        stage.innerHTML = '';
        const txt = responseText || '';
        if (thinkingText) {
            const thWrap = document.createElement('div');
            thWrap.className = 'att-run-wrap thinking-wrap';
            const thBar = document.createElement('div');
            thBar.className = 'att-run-bar';
            const thBtn = document.createElement('button');
            thBtn.className = 'btn-run-code';
            thBtn.textContent = '🧠 Show thinking';
            const thPre = document.createElement('pre');
            thPre.className = 'code-display-block thinking-block';
            thPre.style.display = 'none';
            thPre.textContent = thinkingText;
            thBtn.addEventListener('click', () => {
                const hidden = thPre.style.display === 'none';
                thPre.style.display = hidden ? 'block' : 'none';
                thBtn.textContent = hidden ? '🧠 Hide thinking' : '🧠 Show thinking';
            });
            thBar.appendChild(thBtn);
            thWrap.appendChild(thBar);
            thWrap.appendChild(thPre);
            stage.appendChild(thWrap);
        }
        const code = extractRunnableCode(txt);
        const note = document.createElement('div');
        note.className = 'test-preview-note';
        note.textContent = '▶ Starting X11 app in sandbox (Xvfb + x11vnc + websockify)…';
        stage.appendChild(note);
        const status = document.createElement('div');
        status.className = 'code-run-output';
        status.textContent = 'Contacting sandbox…';
        stage.appendChild(status);
        const frame = document.createElement('iframe');
        frame.className = 'test-preview-iframe ui-live-iframe';
        frame.setAttribute('allowfullscreen', 'true');
        frame.setAttribute('allow', 'fullscreen');
        frame.style.display = 'none';
        stage.appendChild(frame);

        const toolRow = document.createElement('div');
        toolRow.className = 'ui-toolbar';
        stage.appendChild(toolRow);
        const termPanel = document.createElement('div');
        termPanel.className = 'ui-tool-panel';
        termPanel.style.display = 'none';
        stage.appendChild(termPanel);
        const logPanel = document.createElement('div');
        logPanel.className = 'ui-tool-panel';
        logPanel.style.display = 'none';
        stage.appendChild(logPanel);
        const shotPanel = document.createElement('div');
        shotPanel.className = 'ui-tool-panel';
        shotPanel.style.display = 'none';
        stage.appendChild(shotPanel);

        const modalContainer = overlay.querySelector('.modal-container');
        const expandBtn = document.createElement('button');
        expandBtn.className = 'btn-run-code ui-tool-btn';
        expandBtn.textContent = '⛶ Expand';
        expandBtn.title = 'Open the noVNC stream in a new tab (full screen)';
        expandBtn.style.display = 'none';
        expandBtn.addEventListener('click', () => {
            if (containerId) {
                window.open(`/ui/launcher/${containerId}`, '_blank');
            }
        });
        toolRow.appendChild(expandBtn);

        const pre = document.createElement('pre');
        pre.className = 'code-display-block test-preview-code';
        pre.textContent = code;
        stage.appendChild(pre);

        const stopBtn = document.createElement('button');
        stopBtn.className = 'btn-run-code';
        stopBtn.style.display = 'none';
        stopBtn.textContent = '⏹ Stop UI';
        stage.appendChild(stopBtn);

        let containerId = null;
        const uiFetch = (path, body) =>
            fetch(path, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body),
            }).then((r) => r.json().then((j) => ({ ok: r.ok, j })));

        const mkTool = (text, titleAttr, handler) => {
            const b = document.createElement('button');
            b.className = 'btn-run-code ui-tool-btn';
            b.textContent = text;
            b.title = titleAttr || '';
            b.addEventListener('click', handler);
            toolRow.appendChild(b);
            return b;
        };
        const showPanel = (panel) => {
            [termPanel, logPanel, shotPanel].forEach((p) => { p.style.display = p === panel ? 'block' : 'none'; });
        };

        fetch('/api/sandbox/serve_ui', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ code: code, lang: 'python' }),
        })
            .then((r) => r.json().then((j) => ({ ok: r.ok, j })))
            .then(({ ok, j }) => {
                if (!ok || j.error) throw new Error(j.error || 'serve_ui failed');
                containerId = j.container_id;
                note.textContent = '▶ Live X11 app streaming below (noVNC).';
                status.textContent = 'Connected. Interact via mouse/keyboard in the window.';
                frame.style.display = 'block';
                frame.src = `/serve/${j.container_id}/vnc.html?autoconnect=true&resize=scale&path=/serve/ws/${j.container_id}/websockify`;
                stopBtn.style.display = 'inline-block';
                expandBtn.style.display = 'inline-block';

                mkTool('💻 Terminal', 'Run shell commands inside the container', () => showPanel(termPanel));
                mkTool('📜 App Log', 'Show app stdout/exit code', () => showPanel(logPanel));
                mkTool('📷 Screenshot', 'Capture the current Xvfb framebuffer', () => showPanel(shotPanel));
                mkTool('🔄 Restart App', 'Kill and relaunch the app', () => {
                    status.textContent = 'Restarting app…';
                    uiFetch('/api/sandbox/ui/restart', { container_id: containerId })
                        .then(({ ok, j }) => {
                            if (!ok || j.error) throw new Error(j.error || 'restart failed');
                            status.textContent = 'App relaunched.';
                        })
                        .catch((e) => { status.textContent = `Restart failed: ${e.message}`; });
                });

                const cmdInput = document.createElement('input');
                cmdInput.className = 'ui-term-input';
                cmdInput.placeholder = 'e.g. ls /tmp; cat /tmp/ui_stdout.txt; DISPLAY=:99 xdpyinfo | head';
                cmdInput.addEventListener('keydown', (ev) => {
                    if (ev.key === 'Enter') cmdRunBtn.click();
                });
                const cmdRunBtn = mkTool('▶ Run', 'Execute the command', () => {
                    const cmd = cmdInput.value.trim();
                    if (!cmd) return;
                    const outPre = termPanel.querySelector('pre.ui-term-out');
                    outPre.textContent += `\n$ ${cmd}\n`;
                    outPre.scrollTop = outPre.scrollHeight;
                    uiFetch('/api/sandbox/ui/exec', { container_id: containerId, command: cmd })
                        .then(({ ok, j }) => {
                            if (!ok || j.error) throw new Error(j.error || 'exec failed');
                            outPre.textContent += (j.output || '(no output)') + '\n';
                            outPre.scrollTop = outPre.scrollHeight;
                        })
                        .catch((e) => { outPre.textContent += `[error] ${e.message}\n`; });
                });
                cmdRunBtn.disabled = true;
                cmdInput.disabled = true;
                const outPre = document.createElement('pre');
                outPre.className = 'code-display-block ui-term-out';
                outPre.textContent = '# sandbox terminal (runs as sandbox user, DISPLAY=:99)\n';
                termPanel.appendChild(cmdInput);
                termPanel.appendChild(cmdRunBtn);
                termPanel.appendChild(outPre);
                cmdInput.disabled = false;
                cmdRunBtn.disabled = false;
                cmdInput.focus();

                const logPre = document.createElement('pre');
                logPre.className = 'code-display-block ui-term-out';
                logPre.textContent = '# app runtime state\n';
                logPanel.appendChild(logPre);
                const logBtn = mkTool('🔄 Refresh Log', '', () => {
                    uiFetch('/api/sandbox/ui/status', { container_id: containerId })
                        .then(({ ok, j }) => {
                            if (!ok || j.error) throw new Error(j.error || 'status failed');
                            logPre.textContent =
                                `running: ${j.running}\n` +
                                `app_pid: ${j.app_pid || 'n/a'}\n` +
                                `app_exitcode: ${j.app_exitcode === null ? 'still running' : j.app_exitcode}\n` +
                                `\n--- stdout (tail) ---\n${j.stdout_tail || '(no output)'}`;
                        })
                        .catch((e) => { logPre.textContent += `\n[error] ${e.message}\n`; });
                });

                const shotImg = document.createElement('img');
                shotImg.className = 'test-preview-image ui-shot-img';
                shotPanel.appendChild(shotImg);
                const shotBtn = mkTool('📷 Capture', '', () => {
                    shotImg.removeAttribute('src');
                    uiFetch('/api/sandbox/ui/screenshot', { container_id: containerId })
                        .then(({ ok, j }) => {
                            if (!ok || j.error) throw new Error(j.error || 'screenshot failed');
                            if (!j.image) throw new Error('no image returned');
                            shotImg.src = 'data:image/png;base64,' + j.image;
                        })
                        .catch((e) => { shotPanel.insertAdjacentHTML('beforeend', `<div class="code-run-output">[error] ${escapeHtml(e.message)}</div>`); });
                });
            })
            .catch((e) => {
                status.textContent = `UI launch failed: ${e.message}`;
            });
        stopBtn.addEventListener('click', () => {
            stopBtn.disabled = true;
            if (containerId) {
                fetch('/api/sandbox/stop_serve', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ container_id: containerId }),
                }).catch(() => {});
            }
            status.textContent = 'UI stopped.';
            frame.style.display = 'none';
            expandBtn.style.display = 'none';
        });
        if (downloadBtn) {
            downloadBtn.onclick = () => {
                const blob = new Blob([txt], { type: 'text/plain' });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = (label || 'response') + '.txt';
                a.click();
                URL.revokeObjectURL(url);
            };
        }
        overlay.classList.add('open');
    }

    function openTestPreview(testId) {
        const t = ALL_TESTS.find((x) => x.id === testId);
        if (!t) return;
        const kind = (t.kind || 'text').toLowerCase();
        const icon = TEST_KIND_ICON[kind] || '📝';
        const overlay = document.getElementById('test-preview-overlay');
        const title = document.getElementById('test-preview-title');
        const meta = document.getElementById('test-preview-meta');
        const promptEl = document.getElementById('test-preview-prompt');
        const stage = document.getElementById('test-preview-stage');
        const expectedEl = document.getElementById('test-preview-expected');
        const downloadBtn = document.getElementById('btn-download-test');
        const outdatedAlert = document.getElementById('test-preview-outdated-alert');
        const runsDetails = document.getElementById('test-preview-runs-details');

        title.textContent = `${icon} ${t.label || t.id}`;

        if (outdatedAlert) {
            if (t.is_out_of_date) {
                outdatedAlert.classList.remove('d-none');
                outdatedAlert.innerHTML = `<strong>⚠️ Test Definition Modified (Results Out of Date)</strong>` +
                    `<p>This test definition was modified after benchmarks were recorded on ${t.out_of_date_count || 'some'} model(s) (${escapeHtml((t.out_of_date_models || []).join(', '))}). Re-running the benchmark for these models will evaluate against the current test specification.</p>`;
            } else {
                outdatedAlert.classList.add('d-none');
            }
        }

        if (runsDetails) {
            if (t.models_tested_count > 0) {
                runsDetails.classList.remove('d-none');
                const modelList = (t.models_tested || []).map((m) => {
                    const passed = (t.models_passed || []).includes(m);
                    return `<span class="test-runs-pill ${passed ? 'passed' : 'failed'}">${passed ? '✓' : '✗'} ${escapeHtml(m)}</span>`;
                }).join(' ');
                runsDetails.innerHTML = `<div><strong>Tested on ${t.models_tested_count} model${t.models_tested_count === 1 ? '' : 's'}:</strong>${t.last_run ? ` <span style="color:var(--text-muted); margin-left:0.5rem;">(Last run: ${escapeHtml(t.last_run)})</span>` : ''}</div>` +
                    `<div style="display:flex; flex-wrap:wrap; gap:0.4rem; margin-top:0.35rem;">${modelList}</div>`;
            } else {
                runsDetails.classList.remove('d-none');
                runsDetails.innerHTML = `<span style="color:var(--text-muted);">⚪ This test has not been executed on any models yet.</span>`;
            }
        }

        renderTestPreviewRatings(t);

        const statsEl = document.getElementById('test-preview-stats');
        if (statsEl) {
            const tested = t.models_tested || [];
            if (tested.length === 0) {
                statsEl.classList.add('d-none');
                statsEl.innerHTML = '';
            } else {
                statsEl.classList.remove('d-none');
                const scores = t.models_scores || {};
                const lint = t.models_lint || {};
                const lastRun = t.models_last_run || {};
                const passedList = t.models_passed || [];
                const runCount = t.models_run_count || {};
                const failCount = t.models_fail_count || {};
                const tokens = t.models_tokens || {};
                const latencies = t.models_latency || {};
                const speeds = t.models_speed || {};
                const breakdowns = t.models_breakdown || {};
                // Rank order: total points (benchmark score + star bonus,
                // 30/star) — unrated models keep their plain code score.
                const savedRatings = _loadHumanRatings(t.id) || {};
                const rankedTested = tested.slice().sort((a, b) =>
                    _modelTotalPoints(b, scores, savedRatings) - _modelTotalPoints(a, scores, savedRatings));
                const rows = rankedTested.map((m, rIdx) => {
                    const sc = scores[m];
                    const scoreTxt = (sc === undefined || sc === null) ? '—' : `${Math.round(sc)}`;
                    const rating = savedRatings[m] || 0;
                    const starPts = _starPoints(rating);
                    const totalPts = _modelTotalPoints(m, scores, savedRatings);
                    const ptsCell = rating > 0
                        ? `<span title="${scoreTxt} code + ${starPts} star points">${totalPts} pts</span>`
                        : `<span title="Benchmark score (no human rating yet)">${scoreTxt === '—' ? '—' : `${totalPts} pts`}</span>`;
                    const rank = rIdx === 0 ? '🥇' : rIdx === 1 ? '🥈' : rIdx === 2 ? '🥉' : `${rIdx + 1}`;                    const passed = passedList.includes(m);
                    const lp = lint[m];
                    let lintCell;
                    if (lp === true) lintCell = '<span class="test-stats-lint-ok">✓ passed</span>';
                    else if (lp === false) lintCell = '<span class="test-stats-lint-fail">✗ FAILED</span>';
                    else lintCell = '<span style="color:var(--text-muted);">—</span>';
                    const lr = lastRun[m];
                    const dateCell = lr
                        ? `<span title="${escapeHtml(String(lr))}" style="color:var(--text-muted);white-space:nowrap;">${escapeHtml(String(lr).replace('T', ' ').slice(0, 16))}</span>`
                        : '<span style="color:var(--text-muted);white-space:nowrap;">—</span>';
                    const rcount = runCount[m] != null ? Number(runCount[m]) : 1;
                    const fcount = failCount[m] != null ? Number(failCount[m]) : (passed ? 0 : 1);
                    const tcount = tokens[m] != null ? Number(tokens[m]) : null;
                    const timeSec = latencies[m] != null ? Number(latencies[m]) : null;
                    const tokPerSec = speeds[m] != null ? Number(speeds[m]) : null;
                    const tokensCell = (tcount !== null && Number.isFinite(tcount))
                        ? `<span title="${tcount.toLocaleString()} tokens generated">${tcount.toLocaleString()}</span>`
                        : '<span style="color:var(--text-muted);">—</span>';
                    const timeCell = (timeSec !== null && Number.isFinite(timeSec))
                        ? `<span title="${timeSec.toFixed(1)}s to finish">${timeSec >= 60 ? (timeSec / 60).toFixed(1) + 'm' : timeSec.toFixed(1) + 's'}</span>`
                        : '<span style="color:var(--text-muted);">—</span>';
                    const speedCell = (tokPerSec !== null && Number.isFinite(tokPerSec) && tokPerSec > 0)
                        ? `${tokPerSec.toFixed(1)} tok/s`
                        : '<span style="color:var(--text-muted);">—</span>';
                    // Score breakdown panel (why this score): execution outcome,
                    // functional check, code quality notes, watermark, error.
                    const bd = breakdowns[m] || {};
                    const yesno = (v) => (v === true ? '✓' : v === false ? '✗' : '—');
                    const bdScore = (bd.score !== undefined && bd.score !== null) ? `${Math.round(Number(bd.score))}/100` : '—';
                    const bdGrade = (bd.score !== undefined && bd.score !== null) ? gradeForScore(Number(bd.score)) : '—';
                    const bdFp = bd.functional_pass;
                    const bdRan = bd.code_ran;
                    const bdNotes = Array.isArray(bd.code_quality_notes) && bd.code_quality_notes.length
                        ? bd.code_quality_notes.map((n) => `<li>${escapeHtml(n)}</li>`).join('')
                        : '<li>—</li>';
                    const bdWater = Array.isArray(bd.watermark_flags) && bd.watermark_flags.length
                        ? bd.watermark_flags.map((f) => `<li>${escapeHtml(f)}</li>`).join('')
                        : '<li>none detected</li>';
                    const bdErr = (bd.error || '').trim();
                    // Rubric per-criterion results (scoring rubric from the test's
                    // "rubric" field merged with the engine default). Each entry is
                    // {label, passed, points, checks}.
                    const rubric = bd.rubric || {};
                    const crits = Array.isArray(rubric.criteria) ? rubric.criteria : [];
                    const rubricPassed = crits.filter((c) => c.passed).length;
                    const rubricScore = (rubric.score !== undefined && rubric.score !== null) ? `${Math.round(Number(rubric.score))}%` : '—';
                    const rubricHtml = crits.length
                        ? `<details class="test-stats-rubric">
                            <summary><span class="ttl">Scoring rubric</span> <span class="val">${rubricPassed}/${crits.length} criteria <em>(${rubricScore})</em></span></summary>
                            <ul>
                                ${crits.map((c) => `<li class="${c.passed ? 'rubric-ok' : 'rubric-miss'}">${c.passed ? '✓' : '✗'} <strong>${escapeHtml(c.label)}</strong> <span class="val">${Number(c.points) || 0} pts</span></li>`).join('')}
                            </ul>
                        </details>`
                        : '';
                    const bdQScore = (bd.code_quality !== undefined && bd.code_quality !== null)
                        ? `${Math.round(Number(bd.code_quality))}/100${bd.code_quality_lang ? ` (${escapeHtml(bd.code_quality_lang)})` : ''}`
                        : '—';
                    const bdCS = (bd.code_score !== undefined && bd.code_score !== null) ? `${Math.round(Number(bd.code_score))}` : '—';
                    const detailRow = `<tr class="test-stats-detail-row" id="stats-detail-${rIdx}" style="display:none;">
                        <td colspan="11">
                            <div class="test-stats-detail">
                                <div class="test-stats-detail-grid">
                                    <div><span class="ttl">Final score</span><span class="val">${bdScore} <em>(${bdGrade})</em></span></div>
                                    <div><span class="ttl">Code ran</span><span class="val">${yesno(bdRan)}</span></div>
                                    <div><span class="ttl">Functional pass</span><span class="val">${yesno(bdFp)}</span></div>
                                    <div><span class="ttl">Lint / compile</span><span class="val">${yesno(bd.lint_passed)}</span></div>
                                    <div><span class="ttl">Run score</span><span class="val">${bdCS}</span></div>
                                    <div><span class="ttl">Code quality</span><span class="val">${bdQScore}</span></div>
                                    <div><span class="ttl">AI watermark</span><span class="val">${bd.watermark !== undefined && bd.watermark !== null ? Math.round(Number(bd.watermark)) : '—'}</span></div>
                                </div>
                                <div class="test-stats-detail-col"><span class="ttl">Code-quality notes</span>
                                    <ul>${bdNotes}</ul>
                                </div>
                                <div class="test-stats-detail-col"><span class="ttl">Watermark flags</span>
                                    <ul>${bdWater}</ul>
                                </div>
                                ${rubricHtml ? `<div class="test-stats-detail-col">${rubricHtml}</div>` : ''}
                                ${bdErr ? `<div class="test-stats-detail-col test-stats-detail-err"><span class="ttl">Error / notes</span><span class="val">${escapeHtml(bdErr)}</span></div>` : ''}
                            </div>
                        </td>
                    </tr>`;
                    return `<tr class="test-stats-main-row">
                        <td><input type="checkbox" class="test-stats-rerun" value="${escapeHtml(m)}" checked title="Tick to include this model in a rerun"></td>
                        <td class="test-stats-model">${rank} ${escapeHtml(m)}</td>
                        <td class="${passed ? 'test-stats-pass' : 'test-stats-fail'}">${passed ? '✓ Pass' : '✗ Fail'}</td>
                        <td><button class="btn-stats-score" type="button" title="Why this score? Click for breakdown" data-detail="${rIdx}">${scoreTxt} ${scoreTxt !== '—' ? 'ⓘ' : ''}</button></td>
                        <td>${ptsCell}</td>
                        <td>${lintCell}</td>
                        <td><span title="${rcount} run(s), ${fcount} failed">${rcount}<span style="color:var(--text-muted);"> / ${fcount} fail</span></span></td>
                        <td>${tokensCell}</td>
                        <td>${timeCell}</td>
                        <td>${speedCell}</td>
                        <td>${dateCell}</td>
                    </tr>${detailRow}`;
                }).join('');
                // Model weaknesses: aggregate rubric criteria across every model's
                // breakdown and rank the most-missed criteria, so the panel shows
                // which requested features models commonly fail to deliver.
                const gapCounts = new Map();
                const gapTotal = new Map();
                Object.values(breakdowns).forEach((bd) => {
                    const crits = (bd.rubric && Array.isArray(bd.rubric.criteria)) ? bd.rubric.criteria : [];
                    crits.forEach((c) => {
                        gapTotal.set(c.label, (gapTotal.get(c.label) || 0) + 1);
                        if (!c.passed) gapCounts.set(c.label, (gapCounts.get(c.label) || 0) + 1);
                    });
                });
                const gaps = [...gapCounts.entries()]
                    .map(([label, miss]) => ({ label, miss, total: gapTotal.get(label) || 0 }))
                    .filter((g) => g.miss > 0)
                    .sort((a, b) => (b.miss / b.total) - (a.miss / a.total) || b.miss - a.miss)
                    .slice(0, 6);
                const gapsHtml = gaps.length
                    ? `<div class="test-stats-gaps">
                        <div class="modal-section-title" style="margin-bottom:0.4rem;">Common rubric gaps <span style="color:var(--text-muted);font-size:0.72rem;font-weight:400;">(most-missed criteria across models)</span></div>
                        <ul>${gaps.map((g) => `<li class="${g.miss === g.total ? 'rubric-miss' : ''}"><strong>${escapeHtml(g.label)}</strong> — missed by ${g.miss}/${g.total} model${g.miss === 1 ? '' : 's'}</li>`).join('')}</ul>
                    </div>`
                    : '';
                statsEl.innerHTML = `<div class="modal-section-title">Per-Model Stats <span style="color:var(--text-muted);font-size:0.72rem;font-weight:400;">(ranked by points — code score + 30/star bonus; tick models to rerun)</span>
                    <button id="btn-rerun-stats" class="btn btn-secondary btn-sm" title="Re-run this benchmark for the models ticked above" style="float:right;margin-top:-0.2rem;">🔄 Rerun selected</button></div>
                    <table class="test-stats-table">
                        <thead><tr><th>Rerun</th><th>Model</th><th>Pass</th><th>Code Score</th><th>Points</th><th>Lint / Compile</th><th>Runs <span style="font-weight:400;color:var(--text-muted);">(fails)</span></th><th>Tokens</th><th>Time</th><th>Speed</th><th>Run Date</th></tr></thead>
                        <tbody>${rows}</tbody>
                    </table>
                    ${gapsHtml}`;
            }
        }

        const rerunBtn = document.getElementById('btn-rerun-test');
        if (rerunBtn) {
            rerunBtn.onclick = () => rerunTestFromBrowser(t.id);
        }
        const rerunStatsBtn = document.getElementById('btn-rerun-stats');
        if (rerunStatsBtn) {
            rerunStatsBtn.onclick = () => rerunTestFromBrowser(t.id);
        }
        statsEl.querySelectorAll('.btn-stats-score').forEach((btn) => {
            btn.addEventListener('click', () => {
                const row = document.getElementById(`stats-detail-${btn.dataset.detail}`);
                if (row) {
                    const hidden = row.style.display === 'none' || !row.style.display;
                    row.style.display = hidden ? '' : 'none';
                }
            });
        });

        meta.innerHTML = `<span class="kind-badge kind-${kind}">${kind}</span>` +
            `<span class="test-meta-cat">${escapeHtml((t.category || '').toUpperCase())}</span>` +
            (t.type ? `<span class="test-meta-type">${escapeHtml(t.type)}</span>` : '');
        promptEl.textContent = t.prompt || '(no prompt)';

        stage.innerHTML = '';
        const atts = t.attachments || [];
        if (kind === 'image' && atts.length) {
            const note = document.createElement('div');
            note.className = 'test-preview-note';
            note.textContent = 'Source image (attachment) - shown to confirm it is a readable image';
            stage.appendChild(note);
            atts.forEach((a) => {
                const img = document.createElement('img');
                img.className = 'test-preview-image';
                img.src = `/api/tests/${encodeURIComponent(t.id)}/attachment/${encodeURIComponent(a.name)}`;
                img.alt = a.name;
                stage.appendChild(img);
            });
        } else if (kind === 'html' && atts.length) {
            const note = document.createElement('div');
            note.className = 'test-preview-note';
            note.textContent = '▶ Running live HTML/3js preview';
            stage.appendChild(note);
            const frame = document.createElement('iframe');
            frame.className = 'test-preview-iframe';
            frame.setAttribute('sandbox', 'allow-scripts');
            frame.srcdoc = '';
            stage.appendChild(frame);
            fetch(`/api/tests/${encodeURIComponent(t.id)}/attachment/${encodeURIComponent(atts[0].name)}`)
                .then((r) => r.text())
                .then((txt) => { frame.srcdoc = txt; })
                .catch(() => {});
        } else if (atts.length) {
            // code attachments (node/js): show source plus a Run button that executes
            // the code in a sandbox and streams its console output back into the card.
            atts.forEach((a) => {
                const wrap = document.createElement('div');
                wrap.className = 'att-run-wrap';
                const bar = document.createElement('div');
                bar.className = 'att-run-bar';
                const runBtn = document.createElement('button');
                runBtn.className = 'btn-run-code';
                runBtn.textContent = '▶ Run';
                const out = document.createElement('div');
                out.className = 'code-run-output';
                out.textContent = 'Output appears here after Run.';
                const pre = document.createElement('pre');
                pre.className = 'code-display-block test-preview-code';
                pre.textContent = `Loading ${a.name}…`;
                const stdin = document.createElement('textarea');
                stdin.className = 'code-input-stdin';
                stdin.placeholder = 'Input (stdin) - one line per prompt()/input() call';
                stdin.rows = 2;
                bar.appendChild(runBtn);
                wrap.appendChild(bar);
                wrap.appendChild(stdin);
                wrap.appendChild(pre);
                wrap.appendChild(out);
                stage.appendChild(wrap);
                fetch(`/api/tests/${encodeURIComponent(t.id)}/attachment/${encodeURIComponent(a.name)}`)
                    .then((r) => r.text())
                    .then((txt) => {
                        pre.textContent = txt;
                        runBtn.addEventListener('click', () => openTerminal(txt, null, stdin.value));
                    })
                    .catch((e) => { pre.textContent = `Failed to load ${a.name}: ${e.message}`; });
            });
        } else {
            stage.innerHTML = `<div class="empty-state">This test has no attachments to preview.</div>`;
        }

        // Model-produced outputs (e.g. games the model wrote): every response is
        // downloadable, and web-based ones (HTML/3js) are playable in the card.
        fetch(`/api/tests/${encodeURIComponent(t.id)}/responses`)
            .then((r) => r.json())
            .then((data) => {
                const responses = data.responses || [];
                if (!responses.length) return;
                const sec = document.createElement('div');
                sec.className = 'model-outputs-section';
                const h = document.createElement('div');
                h.className = 'test-preview-note';
                h.textContent = `Model-produced outputs (${responses.length}): download, or play if web-based`;
                sec.appendChild(h);

                // Featured: the winning model's result (same total-points
                // ranking as the card — stars included). Re-rendered in place
                // when a rating flips the #1.
                ACTIVE_PREVIEW = { testId: t.id, responses };
                renderFeaturedWinner(t, responses, sec);

                responses.forEach((resp) => {
                    const row = document.createElement('div');
                    row.className = 'att-run-wrap';
                    const bar = document.createElement('div');
                    bar.className = 'att-run-bar';
                    const isUi = t.type === 'ui';
                    const isHtml = !!resp.is_html;
                    const lbl = document.createElement('span');
                    lbl.className = 'model-output-label';
                    lbl.textContent = `${resp.model}${resp.passed === false ? ' (failed)' : ''} - ${resp.response_len} chars`;
                    if (isHtml) {
                        const playBtn = document.createElement('button');
                        playBtn.className = 'btn-run-code';
                        playBtn.textContent = '▶ Play';
                        playBtn.addEventListener('click', () => openResponseRunner(resp.response, resp.model, resp.thinking));
                        bar.appendChild(playBtn);
                    } else if (isUi) {
                        const runBtn = document.createElement('button');
                        runBtn.className = 'btn-run-code';
                        runBtn.textContent = '▶ Run UI';
                        runBtn.addEventListener('click', () => openResponseRunner(resp.response, resp.model, resp.thinking));
                        bar.appendChild(runBtn);
                        const uiBtn = document.createElement('button');
                        uiBtn.className = 'btn-run-code';
                        uiBtn.style.marginLeft = '0.5rem';
                        uiBtn.textContent = '🖥 View UI';
                        uiBtn.addEventListener('click', () => openUiViewer(resp.response, resp.model, resp.thinking));
                        bar.appendChild(uiBtn);
                    } else {
                        const runBtn = document.createElement('button');
                        runBtn.className = 'btn-run-code';
                        runBtn.textContent = '▶ Run (terminal)';
                        runBtn.addEventListener('click', () => openResponseRunner(resp.response, resp.model, resp.thinking));
                        bar.appendChild(runBtn);
                    }
                    const expandBtn = document.createElement('button');
                    expandBtn.className = 'btn-run-code';
                    expandBtn.style.marginLeft = '0.5rem';
                    if (isHtml) {
                        expandBtn.textContent = '⤢ Play (expanded)';
                        expandBtn.addEventListener('click', () => openExpandedRunner(resp.response, resp.model, resp.thinking));
                    } else if (isUi) {
                        expandBtn.textContent = '⤢ View UI (expanded)';
                        expandBtn.addEventListener('click', () => openUiViewer(resp.response, resp.model, resp.thinking));
                    } else {
                        expandBtn.textContent = '⤢ Run (terminal)';
                        expandBtn.addEventListener('click', () => openExpandedRunner(resp.response, resp.model, resp.thinking));
                    }
                    bar.appendChild(expandBtn);
                    const dlBtn = document.createElement('button');
                    dlBtn.className = 'btn-run-code';
                    dlBtn.style.marginLeft = '0.5rem';
                    dlBtn.textContent = '⬇ Download';
                    dlBtn.addEventListener('click', () => {
                        const blob = new Blob([resp.response], { type: 'text/plain' });
                        const url = URL.createObjectURL(blob);
                        const a = document.createElement('a');
                        a.href = url;
                        a.download = `${t.id}_${resp.model.replace(/[^a-z0-9]/gi, '_')}.txt`;
                        a.click();
                        URL.revokeObjectURL(url);
                    });
                    bar.appendChild(dlBtn);
                    bar.appendChild(lbl);
                    const pre = document.createElement('pre');
                    pre.className = 'code-display-block test-preview-code';
                    pre.textContent = resp.response;
                    row.appendChild(bar);
                    row.appendChild(pre);
                    sec.appendChild(row);
                });
                stage.appendChild(sec);
            })
            .catch(() => {});

        expectedEl.textContent = t.expected ? `Expected: ${t.expected}` : '';
        downloadBtn.onclick = () => {
            window.location.href = `/api/tests/${encodeURIComponent(t.id)}/download`;
        };
        overlay.classList.add('open');
    }

    function closeTestPreview() {
        const overlay = document.getElementById('test-preview-overlay');
        if (overlay) overlay.classList.remove('open');
    }

    async function rerunTestFromBrowser(testId) {
        // Prefer models ticked in the card's stats table; fall back to the
        // sidebar model checklist so the button also works without a prior run.
        const checkedBoxes = document.querySelectorAll('#test-preview-stats .test-stats-rerun:checked');
        const models = Array.from(checkedBoxes).map((cb) => cb.value);
        if (models.length === 0) {
            const sidebarModels = getSelectedModels();
            if (sidebarModels.length === 0) {
                showToast('Please tick at least one model in the stats panel, or select one in the sidebar.', 'error');
                return;
            }
            models.push(...sidebarModels);
        }
        const btn = document.getElementById('btn-rerun-test');
        if (btn) {
            btn.disabled = true;
            btn.innerHTML = `<span class="loader"></span> Starting...`;
        }
        try {
            const payload = {
                models: models,
                use_proxy: (typeof benchmarkMode === 'undefined' ? true : benchmarkMode === 'proxy'),
                test_ids: [testId],
                resume: false,
            };
            const res = await fetch('/api/run', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
            });
            if (res.status === 409) {
                const data = await res.json();
                showToast(data.error, 'error');
                return;
            }
            if (!res.ok) {
                const errorText = await res.text();
                throw new Error(errorText || 'Server error starting benchmark');
            }
            const resData = await res.json().catch(() => null);
            if (resData && resData.status === 'No outdated benchmarks') {
                showToast(resData.message || 'All benchmark definitions are up to date.', 'success');
                return;
            }
            logToTerminal(`Re-running benchmark "${testId}" for models: ${models.join(', ')}...`, 'info', 'general');
            showToast(`Benchmark "${testId}" started for ${models.length} model${models.length === 1 ? '' : 's'}.`, 'success');
            closeTestPreview();
            switchTab('general');
            setRunnerState('running');
        } catch (err) {
            logToTerminal(`Failed to start benchmark: ${err.message}`, 'error', 'general');
            showToast(`Failed to start benchmark: ${err.message}`, 'error');
        } finally {
            if (btn) {
                btn.disabled = false;
                btn.innerHTML = '🔄 Rerun for selected models';
            }
        }
    }

    // Wire up Test Browser toolbar + preview modal
    (function wireTestBrowser() {
        const search = document.getElementById('test-search');
        if (search) {
            search.addEventListener('input', (e) => {
                TEST_BROWSER_FILTER.q = e.target.value;
                renderTestBrowser();
            });
        }
        const filters = document.getElementById('test-kind-filters');
        if (filters) {
            filters.querySelectorAll('.kind-filter').forEach((btn) => {
                btn.addEventListener('click', () => {
                    filters.querySelectorAll('.kind-filter').forEach((b) => b.classList.remove('active'));
                    btn.classList.add('active');
                    TEST_BROWSER_FILTER.kind = btn.dataset.kind;
                    renderTestBrowser();
                });
            });
        }
        const statusFilters = document.getElementById('test-status-filters');
        if (statusFilters) {
            statusFilters.querySelectorAll('.kind-filter').forEach((btn) => {
                btn.addEventListener('click', () => {
                    statusFilters.querySelectorAll('.kind-filter').forEach((b) => b.classList.remove('active'));
                    btn.classList.add('active');
                    TEST_BROWSER_FILTER.status = btn.dataset.status;
                    renderTestBrowser();
                });
            });
        }
        const modelSelect = document.getElementById('test-model-select');
        if (modelSelect) {
            modelSelect.addEventListener('change', (e) => {
                TEST_BROWSER_FILTER.model = e.target.value;
                renderTestBrowser();
            });
        }
        const closeBtn = document.getElementById('test-preview-close');
        if (closeBtn) closeBtn.addEventListener('click', closeTestPreview);
        const overlay = document.getElementById('test-preview-overlay');
        if (overlay) {
            overlay.addEventListener('click', (e) => {
                if (e.target === overlay) closeTestPreview();
            });
        }
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') closeTestPreview();
        });
    })();

    async function loadSharedTests() {
        const container = document.getElementById('shared-test-checkboxes');
        if (!container) return;
        try {
            const res = await fetch('/api/tests/shared_llm');
            const data = await res.json();
            const tests = data.tests || [];
            container.innerHTML = '';

            if (tests.length === 0) {
                container.innerHTML = `<div style="color:var(--text-muted);font-size:0.8rem;padding:0.5rem;">No SharedLLM tasks found</div>`;
                return;
            }

            tests.forEach((test) => {
                const item = document.createElement('label');
                item.className = 'checkbox-item';

                const input = document.createElement('input');
                input.type = 'checkbox';
                input.value = test.id;
                input.checked = true;

                const span = document.createElement('span');
                span.className = 'checkbox-label';
                span.textContent = `${test.category}: ${test.label}`;

                item.appendChild(input);
                item.appendChild(span);
                container.appendChild(item);
            });
        } catch (err) {
            container.innerHTML = `<div style="color:var(--text-muted);font-size:0.8rem;padding:0.5rem;">Failed to load tasks</div>`;
        }
    }

    async function loadMultistepWorkflows() {
        const container = document.getElementById('multistep-workflow-checkboxes');
        if (!container) return;
        try {
            const res = await fetch('/api/tests/multistep');
            const data = await res.json();
            const workflows = data.tests || [];
            container.innerHTML = '';

            if (workflows.length === 0) {
                container.innerHTML = `<div style="color:var(--text-muted);font-size:0.8rem;padding:0.5rem;">No MultiStep workflows found</div>`;
                return;
            }

            workflows.forEach((wf) => {
                const item = document.createElement('label');
                item.className = 'checkbox-item';

                const input = document.createElement('input');
                input.type = 'checkbox';
                input.value = wf.id;
                input.checked = true;

                const span = document.createElement('span');
                span.className = 'checkbox-label';
                span.textContent = `${wf.category}: ${wf.label} (${wf.steps} steps)` + (wf.description ? ` — ${wf.description}` : '');

                item.appendChild(input);
                item.appendChild(span);
                container.appendChild(item);
            });
        } catch (err) {
            container.innerHTML = `<div style="color:var(--text-muted);font-size:0.8rem;padding:0.5rem;">Failed to load workflows</div>`;
        }
    }

    // Load past reports list and auto-restore comparison view on refresh
    async function loadHistory() {
        try {
            const res = await fetch('/api/results');
            const data = await res.json();
            
            historyList.innerHTML = '';
            const results = data.results || [];
            
            if (results.length === 0) {
                historyList.innerHTML = `<div style="color:var(--text-muted);font-size:0.8rem;text-align:center;padding:1rem;">No past runs found</div>`;
                return;
            }

            // Auto-restore: merge ALL saved results into the comparison views so the
            // full cross-run comparison is visible immediately after a page refresh.
            // This mirrors what the live benchmark runner does as results stream in.
            const generalResults = [];
            const sharedResults  = [];
            let latestGeneralData = null;
            let latestSharedData  = null;
            TREND_DATA.general = [];
            TREND_DATA.shared = [];

            await Promise.all(results.map(async (result) => {
                try {
                    const dr = await fetch(`/api/results/${result.filename}`);
                    const detail = await dr.json();
                    // Run snapshots (not per-model aggregates) feed the trend charts:
                    // each file is a point-in-time run with its own timestamp.
                    if (!result.per_model && result.generated_at) {
                        const label = result.generated_at.replace('T', ' ').slice(0, 16);
                        (detail.results || []).forEach(modelRecord => {
                            const bucket = result.type === 'shared_llm' ? TREND_DATA.shared : TREND_DATA.general;
                            const row = result.type === 'shared_llm' ? computeSharedRow(modelRecord) : computeGeneralRow(modelRecord);
                            bucket.push({ label, filename: result.filename, model: modelRecord.model, score: row.score });
                        });
                    }
                    if (result.type === 'shared_llm') {
                        (detail.results || []).forEach(modelRecord => {
                            const existing = sharedResults.find(r => r.model === modelRecord.model);
                            if (result.per_model) {
                                // Per-model files are authoritative: overwrite stale run-file data
                                if (existing) sharedResults[sharedResults.indexOf(existing)] = modelRecord;
                                else sharedResults.push(modelRecord);
                            } else if (!existing) {
                                sharedResults.push(modelRecord);
                            }
                        });
                        if (!latestSharedData || result.filename > (latestSharedData._filename || '')) {
                            latestSharedData = { ...detail, _filename: result.filename };
                        }
                    } else {
                        (detail.results || []).forEach(modelRecord => {
                            const existing = generalResults.find(r => r.model === modelRecord.model);
                            if (result.per_model) {
                                // Per-model files are authoritative: overwrite stale run-file data
                                if (existing) generalResults[generalResults.indexOf(existing)] = modelRecord;
                                else generalResults.push(modelRecord);
                            } else if (!existing) {
                                generalResults.push(modelRecord);
                            }
                        });
                        if (!latestGeneralData || result.filename > (latestGeneralData._filename || '')) {
                            latestGeneralData = { ...detail, _filename: result.filename };
                        }
                    }
                } catch (_) { /* skip unreadable files */ }
            }));

            if (generalResults.length > 0) {
                currentResults = generalResults;
                updateOverviewMetrics({ results: generalResults });
                renderChartsFromData(generalResults);
                renderDetailsSection(generalResults);
                logToTerminal(`Restored ${generalResults.length} general benchmark model(s) from history`, 'success');
            }
            if (sharedResults.length > 0) {
                currentSharedResults = sharedResults;
                if (latestSharedData) updateSharedOverviewMetrics({ ...latestSharedData, results: sharedResults });
                renderSharedChartsFromData(sharedResults);
                renderSharedDetailsSection(sharedResults);
                logToTerminal(`Restored ${sharedResults.length} SharedLLM benchmark model(s) from history`, 'success');
            }

            // Merged view active again -> hide single-run banner and refresh trends.
            const snapBar = document.getElementById('snapshot-mode-bar');
            if (snapBar) snapBar.classList.add('d-none');
            renderTrendChart('general');
            renderTrendChart('shared');
            populateDiffSelects(results.filter(r => !r.per_model));

            results.forEach(result => {
                // Per-model files are an internal storage detail; only show run snapshots in history.
                if (result.per_model) {
                    return;
                }
                const item = document.createElement('div');
                item.className = 'history-item active'; // all files contribute to merged view
                
                const title = document.createElement('div');
                title.className = 'history-title';
                title.textContent = result.filename.replace('benchmarks_', '').replace('shared_llm_', '').replace('.json', '');
                
                const meta = document.createElement('div');
                meta.className = 'history-meta';
                
                const runTypeBadge = result.type === 'shared_llm' ? 'SharedLLM' : result.type === 'multistep' ? 'Multi-Step' : 'General';
                const typeText = document.createElement('span');
                typeText.textContent = `${runTypeBadge} (${result.benchmark_type.toUpperCase()})`;
                
                const dateText = document.createElement('span');
                dateText.textContent = result.generated_at ? result.generated_at.split('T')[0] : '';
                
                meta.appendChild(typeText);
                meta.appendChild(dateText);
                
                const badges = document.createElement('div');
                badges.className = 'history-models-badges';
                result.models.forEach(model => {
                    const badge = document.createElement('span');
                    badge.className = 'model-mini-badge';
                    badge.textContent = model;
                    badges.appendChild(badge);
                });
                
                item.appendChild(title);
                item.appendChild(meta);
                item.appendChild(badges);
                
                // Add Delete button
                item.style.position = 'relative';
                const deleteBtn = document.createElement('span');
                deleteBtn.textContent = 'Delete';
                deleteBtn.className = 'history-delete-btn';
                deleteBtn.style.cssText = 'position: absolute; right: 8px; top: 8px; z-index: 100; text-transform: uppercase;';
                deleteBtn.title = 'Delete result file';
                deleteBtn.addEventListener('click', async (e) => {
                    e.stopPropagation();
                    if (!confirm(`Are you sure you want to permanently delete result file "${result.filename}"?`)) {
                        return;
                    }
                    try {
                        const deleteRes = await fetch(`/api/results/${result.filename}`, { method: 'DELETE' });
                        if (deleteRes.ok) {
                            logToTerminal(`Deleted result file ${result.filename}`, 'success');
                            loadHistory();
                        } else {
                            const errData = await deleteRes.json();
                            logToTerminal(errData.error || "Failed to delete result file", "error");
                        }
                    } catch (err) {
                        logToTerminal(`Error deleting result file: ${err.message}`, "error");
                    }
                });
                item.appendChild(deleteBtn);
                
                item.addEventListener('click', () => {
                    document.querySelectorAll('.history-item').forEach(el => el.classList.remove('active'));
                    item.classList.add('active');
                    loadBenchmarkDetail(result.filename, result.type);
                });
                
                historyList.appendChild(item);
            });
        } catch (err) {
            logToTerminal(`Error fetching history list: ${err.message}`, 'error');
        }
    }

    // Wire "Show All Runs" (restore merged multi-run view from single-run snapshot mode)
    const btnShowAllRuns = document.getElementById('btn-show-all-runs');
    if (btnShowAllRuns) {
        btnShowAllRuns.addEventListener('click', () => loadHistory());
    }

    // Wire run-to-run diff
    const btnDiffRuns = document.getElementById('btn-diff-runs');
    if (btnDiffRuns) {
        btnDiffRuns.addEventListener('click', runDiff);
    }

    // Wire Clear All History button
    const btnClearAllHistory = document.getElementById('btn-clear-all-history');
    if (btnClearAllHistory) {        btnClearAllHistory.addEventListener('click', async (e) => {
            e.stopPropagation();
            if (!confirm('Are you sure you want to permanently clear ALL benchmark history reports, result files, and artifacts across all models? This cannot be undone.')) {
                return;
            }
            try {
                const res = await fetch('/api/benchmarks/clear', { method: 'POST' });
                const data = await res.json();
                if (res.ok) {
                    logToTerminal(data.message || 'All benchmark data cleared.', 'success');
                    showToast(data.message || 'All benchmark data cleared.', 'success');
                    await loadHistory();
                    await loadModels();
                    if (typeof loadRoutingMatrix === 'function') await loadRoutingMatrix();
                } else {
                    logToTerminal(`Failed to clear benchmarks: ${data.error || 'Unknown error'}`, 'error');
                    showToast(`Failed to clear benchmarks: ${data.error || 'Unknown error'}`, 'error');
                }
            } catch (err) {
                logToTerminal(`Clear all benchmarks error: ${err.message}`, 'error');
                showToast(`Clear error: ${err.message}`, 'error');
            }
        });
    }

    // Load details of selected file
    async function loadBenchmarkDetail(filename, type) {
        try {
            logToTerminal(`Loading benchmark file: ${filename}...`);
            const res = await fetch(`/api/results/${filename}`);
            const data = await res.json();

            // Single-run snapshot mode: banner tells the user the merged view is
            // replaced and offers one-click restore.
            const snapBar = document.getElementById('snapshot-mode-bar');
            if (snapBar) {
                document.getElementById('snapshot-mode-filename').textContent = filename;
                snapBar.classList.remove('d-none');
            }

            if (type === 'shared_llm' || type === 'multistep') {
                currentSharedResults = data.results || [];
                logToTerminal(
                    `Loaded ${type === 'multistep' ? 'Multi-Step' : 'SharedLLM'} results for ${currentSharedResults.length} models`,
                    'success'
                );
                switchTab('shared');
                updateSharedOverviewMetrics(data);
                renderSharedChartsFromData(currentSharedResults);
                renderSharedDetailsSection(currentSharedResults);
            } else {
                currentResults = data.results || [];
                logToTerminal(`Loaded general results for ${currentResults.length} models`, 'success');
                switchTab('general');
                updateOverviewMetrics(data);
                renderChartsFromData(currentResults);
                renderDetailsSection(currentResults);
            }
        } catch (err) {
            logToTerminal(`Error loading benchmark details: ${err.message}`, 'error');
        }
    }

    // Line chart of per-model score across run snapshots (oldest -> newest).
    function renderTrendChart(kind) {
        const canvas = document.getElementById(`${kind}-trend-chart`);
        if (!canvas || typeof Chart === 'undefined') return;
        const entries = TREND_DATA[kind] || [];
        if (entries.length === 0) return;

        const labels = [...new Set(entries.map(e => e.label))].sort();
        const models = [...new Set(entries.map(e => e.model))].map(m => ({
            model: m,
            last: entries.filter(e => e.model === m).sort((a, b) => b.label.localeCompare(a.label))[0].score,
        })).sort((a, b) => b.last - a.last);

        if (TREND_CHARTS[kind]) { TREND_CHARTS[kind].destroy(); }
        TREND_CHARTS[kind] = new Chart(canvas.getContext('2d'), {
            type: 'line',
            data: {
                labels,
                datasets: models.map((m, i) => {
                    const color = TREND_COLORS[i % TREND_COLORS.length];
                    const byLabel = Object.fromEntries(
                        entries.filter(e => e.model === m.model).map(e => [e.label, e.score])
                    );
                    return {
                        label: m.model,
                        data: labels.map(l => byLabel[l] !== undefined ? byLabel[l] : null),
                        borderColor: color,
                        backgroundColor: color + '33',
                        spanGaps: true,
                        tension: 0.25,
                        pointRadius: 4,
                        pointHoverRadius: 6,
                    };
                }),
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: { beginAtZero: true, max: 100, title: { display: true, text: 'Score' } },
                    x: { ticks: { maxRotation: 45, minRotation: 30, font: { size: 10 } } },
                },
                plugins: { legend: { position: 'bottom', labels: { boxWidth: 12, font: { size: 10 } } } },
            },
        });
    }

    // Run-to-run diff: pick two snapshots, show per-model score deltas.
    function populateDiffSelects(runResults) {
        const selA = document.getElementById('diff-run-a');
        const selB = document.getElementById('diff-run-b');
        if (!selA || !selB) return;
        const runs = runResults.filter(r => r.type !== 'shared_llm');
        const opts = runs.map(r => {
            const d = r.generated_at ? ` (${r.generated_at.slice(0, 10)})` : '';
            return `<option value="${escapeHtml(r.filename)}">${escapeHtml(r.filename.replace('benchmarks_', '').replace('.json', ''))}${d}</option>`;
        }).join('');
        selA.innerHTML = opts;
        selB.innerHTML = opts;
        if (runs.length > 1) selB.selectedIndex = 1; // default: latest vs previous
    }

    async function runDiff() {
        const fa = document.getElementById('diff-run-a')?.value;
        const fb = document.getElementById('diff-run-b')?.value;
        const out = document.getElementById('diff-results');
        if (!fa || !fb || !out) return;
        out.innerHTML = '<div style="color:var(--text-muted);font-size:0.8rem;">Comparing…</div>';
        try {
            const [da, db] = await Promise.all([
                fetch(`/api/results/${fa}`).then(r => r.json()),
                fetch(`/api/results/${fb}`).then(r => r.json()),
            ]);
            const rowsA = new Map((da.results || []).map(m => [m.model, computeGeneralRow(m).score]));
            const rowsB = new Map((db.results || []).map(m => [m.model, computeGeneralRow(m).score]));
            const models = [...new Set([...rowsA.keys(), ...rowsB.keys()])];
            const html = `<table class="diff-table"><thead><tr>
                <th>Model</th><th>${escapeHtml(fa.replace('benchmarks_', '').replace('.json', ''))}</th>
                <th>${escapeHtml(fb.replace('benchmarks_', '').replace('.json', ''))}</th><th>Δ</th></tr></thead><tbody>` +
                models.map(m => {
                    const a = rowsA.get(m), b = rowsB.get(m);
                    const d = (a != null && b != null) ? b - a : null;
                    const delta = d == null ? '<span class="muted">—</span>'
                        : `<span class="${d > 0 ? 'delta-up' : d < 0 ? 'delta-down' : 'muted'}">${d > 0 ? '▲ +' : d < 0 ? '▼ ' : '± '}${d}</span>`;
                    return `<tr><td class="model-cell" title="${escapeHtml(m)}">${escapeHtml(truncateModelName(m))}</td>
                        <td>${a != null ? a : '<span class="muted">—</span>'}</td>
                        <td>${b != null ? b : '<span class="muted">—</span>'}</td><td>${delta}</td></tr>`;
                }).join('') + '</tbody></table>';
            out.innerHTML = html;
        } catch (err) {
            out.innerHTML = `<div style="color:var(--color-danger);font-size:0.8rem;">Diff failed: ${escapeHtml(err.message)}</div>`;
        }
    }

    // Direct Navigation to Model's Latest Test Results
    async function navigateToModelTests(modelId) {
        logToTerminal(`Navigating to latest test results for "${modelId}"...`);
        
        // 1. Check if model is in currentSharedResults
        const inShared = currentSharedResults.find(r => r.model === modelId || r.model.includes(modelId) || modelId.includes(r.model));
        if (inShared) {
            switchTab('shared');
            renderSharedDetailsSection(currentSharedResults, inShared.model);
            const targetSection = document.getElementById('shared-model-tabs') || document.querySelector('#view-shared .results-details-section');
            if (targetSection) {
                targetSection.scrollIntoView({ behavior: 'smooth', block: 'center' });
            }
            return;
        }

        // 2. Check if model is in currentResults (General)
        const inGeneral = currentResults.find(r => r.model === modelId || r.model.includes(modelId) || modelId.includes(r.model));
        if (inGeneral) {
            switchTab('general');
            renderDetailsSection(currentResults, inGeneral.model);
            const targetSection = document.getElementById('model-tabs') || document.querySelector('#view-general .results-details-section');
            if (targetSection) {
                targetSection.scrollIntoView({ behavior: 'smooth', block: 'center' });
            }
            return;
        }

        // 3. Search history list for the most recent result file containing this model
        try {
            const res = await fetch('/api/results');
            const historyData = await res.json();
            const results = historyData.results || [];
            
            const matchingFile = results.find(r => (r.models || []).some(m => m === modelId || m.includes(modelId) || modelId.includes(m)));
            if (matchingFile) {
                await loadBenchmarkDetail(matchingFile.filename, matchingFile.type);
                if (matchingFile.type === 'shared_llm') {
                    switchTab('shared');
                    const target = currentSharedResults.find(r => r.model === modelId || r.model.includes(modelId) || modelId.includes(r.model));
                    if (target) renderSharedDetailsSection(currentSharedResults, target.model);
                    const targetSection = document.getElementById('shared-model-tabs') || document.querySelector('#view-shared .results-details-section');
                    if (targetSection) targetSection.scrollIntoView({ behavior: 'smooth', block: 'center' });
                } else {
                    switchTab('general');
                    const target = currentResults.find(r => r.model === modelId || r.model.includes(modelId) || modelId.includes(r.model));
                    if (target) renderDetailsSection(currentResults, target.model);
                    const targetSection = document.getElementById('model-tabs') || document.querySelector('#view-general .results-details-section');
                    if (targetSection) targetSection.scrollIntoView({ behavior: 'smooth', block: 'center' });
                }
                return;
            }
        } catch (err) {
            console.error('Error navigating to model tests:', err);
        }

        // Fallback: switch to SharedLLM tab
        switchTab('shared');
    }

    // Calculate General Overview card metrics
    function updateOverviewMetrics(fullData) {
        const results = getFilteredResults(fullData.results || [], 'general');
        if (results.length === 0) {
            metricTps.textContent = '0 tok/s';
            metricTtft.textContent = '0 ms';
            metricSuccess.textContent = '0%';
            metricCount.textContent = '0 Models';
            return;
        }

        let totalTps = 0, tpsCount = 0;
        let totalTtft = 0, ttftCount = 0;
        let totalPassed = 0, totalRun = 0;

        results.forEach(m => {
            const categories = ['coding', 'reasoning', 'instruction', 'creative', 'home_automation', 'gamedev', 'appdev', 'linux_admin', 'webdev', 'database', 'cpp', 'java', 'debugging', 'logic', 'retrogames', 'threedprint', 'languages', 'tvdev', 'uiux', 'office', 'life', 'biblical', 'metacog'];
            categories.forEach(cat => {
                const catData = m[`category_${cat}`];
                if (catData) {
                    totalPassed += catData.tests_passed || 0;
                    totalRun += catData.tests_run || 0;
                    if (catData.avg_tokens_per_sec > 0) {
                        totalTps += catData.avg_tokens_per_sec;
                        tpsCount++;
                    }
                    if (catData.avg_ttft_ms > 0) {
                        totalTtft += catData.avg_ttft_ms;
                        ttftCount++;
                    }
                }
            });
        });

        const avgTps = tpsCount > 0 ? (totalTps / tpsCount).toFixed(1) : '0';
        const avgTtft = ttftCount > 0 ? (totalTtft / ttftCount).toFixed(0) : '0';
        const successRate = totalRun > 0 ? ((totalPassed / totalRun) * 100).toFixed(0) : '0';

        metricTps.textContent = `${avgTps} tok/s`;
        metricTtft.textContent = `${avgTtft} ms`;
        metricSuccess.textContent = `${successRate}%`;
        metricCount.textContent = `${results.length} Models`;
    }

    // Calculate SharedLLM Overview card metrics
    function updateSharedOverviewMetrics(fullData) {
        const results = getFilteredResults(fullData.results || [], 'shared');
        if (results.length === 0) {
            sharedMetricFastpath.textContent = '0 ms';
            sharedMetricLibrarian.textContent = '0%';
            sharedMetricRaven.textContent = '0%';
            sharedMetricCount.textContent = '0 Models';
            return;
        }

        let fastPathTotalLat = 0, fastPathCount = 0;
        let toolSuccessCount = 0, toolTotalCount = 0;
        let codeGenAstSuccess = 0, codeGenCount = 0;

        results.forEach(m => {
            m.tasks.forEach(task => {
                if (task.test_id === 'fast_path') {
                    if (task.success) {
                        fastPathTotalLat += task.latency;
                        fastPathCount++;
                    }
                } else if (task.test_id === 'tool_use') {
                    toolTotalCount++;
                    if (task.success) toolSuccessCount++;
                } else if (task.test_id === 'code_gen') {
                    codeGenCount++;
                    if (task.success) codeGenAstSuccess++;
                }
            });
        });

        const fastPathLat = fastPathCount > 0 ? Math.round((fastPathTotalLat / fastPathCount) * 1000) : 0;
        const toolRate = toolTotalCount > 0 ? Math.round((toolSuccessCount / toolTotalCount) * 100) : 0;
        const ravenRate = codeGenCount > 0 ? Math.round((codeGenAstSuccess / codeGenCount) * 100) : 0;

        sharedMetricFastpath.textContent = `${fastPathLat} ms`;
        sharedMetricLibrarian.textContent = `${toolRate}%`;
        sharedMetricRaven.textContent = `${ravenRate}%`;
        sharedMetricCount.textContent = `${results.length} Models`;
    }

    // Model comparison filter (graphs & stats)
    function isOnlineModelName(model) {
        return /^(openrouter|huggingface|hf|cloudflare|opencode_zen|groq|gemini):/i.test(model || '');
    }

    function getFilteredResults(results, type) {
        if (!results || !filterInitialized[type]) return results;
        const sel = filterSelection[type];
        if (!sel || sel.size === 0) return [];
        return results.filter(r => sel.has(r.model));
    }

    function setFilterGroupState(type, group, state) {
        const btn = document.querySelector(`.model-filter-group[data-filter-type="${type}"][data-group="${group}"]`);
        if (!btn) return;
        btn.classList.remove('mf-active', 'mf-partial');
        if (state === 'active') btn.classList.add('mf-active');
        else if (state === 'partial') btn.classList.add('mf-partial');
    }

    function updateFilterGroupState(type) {
        const all = filterAllModels[type] || [];
        const sel = filterSelection[type] || new Set();
        const local = all.filter(m => !isOnlineModelName(m));
        const online = all.filter(m => isOnlineModelName(m));
        const groupState = (list) => {
            if (list.length === 0) return 'none';
            if (list.every(m => sel.has(m))) return 'active';
            if (list.some(m => sel.has(m))) return 'partial';
            return 'none';
        };
        setFilterGroupState(type, 'all', groupState(all));
        setFilterGroupState(type, 'local', groupState(local));
        setFilterGroupState(type, 'online', groupState(online));
        const countEl = document.getElementById(`model-filter-count-${type}`);
        if (countEl) {
            countEl.textContent = all.length > 0 ? `${sel.size} / ${all.length} models shown` : '';
        }
    }

    function addFilterCheckbox(model, type, container) {
        const label = document.createElement('label');
        label.className = 'checkbox-item';
        label.style.cssText = 'padding:0.15rem 0.5rem;margin:0;border-radius:4px;font-size:0.72rem;display:inline-flex;align-items:center;gap:0.3rem;';

        const input = document.createElement('input');
        input.type = 'checkbox';
        input.checked = true;
        filterCheckboxInputs[type][model] = input;

        input.addEventListener('change', () => {
            if (input.checked) filterSelection[type].add(model);
            else filterSelection[type].delete(model);
            updateFilterGroupState(type);
            onModelFilterChange(type);
        });

        const isOnline = isOnlineModelName(model);
        const span = document.createElement('span');
        span.textContent = truncateModelName(model);
        if (isOnline) span.style.color = '#c4b5fd';

        const badge = document.createElement('span');
        badge.textContent = isOnline ? 'ONLINE' : 'LOCAL';
        badge.style.cssText = `font-size:0.55rem;font-weight:700;padding:1px 4px;border-radius:3px;${
            isOnline ? 'background:rgba(139,92,246,0.2);color:#c4b5fd;' : 'background:rgba(6,182,212,0.15);color:#67e8f9;'
        }`;

        label.appendChild(input);
        label.appendChild(span);
        label.appendChild(badge);
        container.appendChild(label);
    }

    function ensureFilterModels(results, type) {
        const container = document.getElementById(`model-filter-models-${type}`);
        if (!container || !results || results.length === 0) return;
        const all = filterAllModels[type];
        const sel = filterSelection[type];
        const known = new Set(all);
        let changed = false;

        results.forEach(r => {
            if (!known.has(r.model)) {
                known.add(r.model);
                all.push(r.model);
                sel.add(r.model);
                changed = true;
                const placeholder = container.querySelector('.model-filter-placeholder');
                if (placeholder) placeholder.remove();
                addFilterCheckbox(r.model, type, container);
            }
        });

        if (!filterInitialized[type] && all.length > 0) {
            filterInitialized[type] = true;
            changed = true;
        }
        if (changed) updateFilterGroupState(type);
    }

    function syncFilterCheckboxUI(type) {
        Object.entries(filterCheckboxInputs[type] || {}).forEach(([model, input]) => {
            input.checked = filterSelection[type].has(model);
        });
    }

    function onModelFilterChange(type) {
        if (type === 'shared') {
            updateSharedOverviewMetrics({ results: currentSharedResults });
            renderSharedChartsFromData(currentSharedResults);
        } else {
            updateOverviewMetrics({ results: currentResults });
            renderChartsFromData(currentResults);
        }
    }

    // Render General Charts
    function renderChartsFromData(results) {
        ensureFilterModels(results, 'general');
        results = getFilteredResults(results, 'general');
        renderGeneralLeaderboard(results);
        if (results.length === 0) {
            tpsChart.data.datasets = [];
            ttftChart.data.datasets = [];
            categoryChart.data.datasets = [];
            tpsChart.update();
            ttftChart.update();
            categoryChart.update();
            return;
        }
        const models = results.map(r => r.model);
        const displayNames = models.map(model => truncateModelName(model));

        // TPS chart - one dataset per model so legend renders model names cleanly
        const tpsDatasets = results.map((r, idx) => {
            let totalTps = 0, tpsCount = 0;
            const categories = ['coding', 'reasoning', 'instruction', 'creative', 'home_automation', 'gamedev', 'appdev', 'linux_admin', 'webdev', 'database', 'cpp', 'java', 'debugging', 'logic', 'retrogames', 'threedprint', 'languages', 'tvdev', 'uiux', 'office', 'life', 'biblical', 'metacog'];
            categories.forEach(cat => {
                const catData = r[`category_${cat}`];
                if (catData && catData.avg_tokens_per_sec > 0) {
                    totalTps += catData.avg_tokens_per_sec;
                    tpsCount++;
                }
            });
            const avgTps = tpsCount > 0 ? (totalTps / tpsCount) : 0;
            return {
                label: displayNames[idx],
                originalLabel: r.model,
                data: [avgTps],
                backgroundColor: modelColor(idx, 0.6),
                borderColor: modelColor(idx, 1),
                borderWidth: 1,
                borderRadius: 4
            };
        });
        tpsChart.data.labels = [''];
        tpsChart.data.originalLabels = [''];
        tpsChart.data.datasets = tpsDatasets;
        tpsChart.update();

        // TTFT chart - one dataset per model
        const ttftDatasets = results.map((r, idx) => {
            let totalTtft = 0, ttftCount = 0;
            const categories = ['coding', 'reasoning', 'instruction', 'creative', 'home_automation', 'gamedev', 'appdev', 'linux_admin', 'webdev', 'database', 'cpp', 'java', 'debugging', 'logic', 'retrogames', 'threedprint', 'languages', 'tvdev', 'uiux', 'office', 'life', 'biblical', 'metacog'];
            categories.forEach(cat => {
                const catData = r[`category_${cat}`];
                if (catData && catData.avg_ttft_ms > 0) {
                    totalTtft += catData.avg_ttft_ms;
                    ttftCount++;
                }
            });
            const avgTtft = ttftCount > 0 ? (totalTtft / ttftCount) : 0;
            return {
                label: displayNames[idx],
                originalLabel: r.model,
                data: [avgTtft],
                backgroundColor: modelColor(idx, 0.6),
                borderColor: modelColor(idx, 1),
                borderWidth: 1,
                borderRadius: 4
            };
        });
        ttftChart.data.labels = [''];
        ttftChart.data.originalLabels = [''];
        ttftChart.data.datasets = ttftDatasets;
        ttftChart.update();

        // Category success rate chart - one dataset per model, 5 categories
        const datasets = [];
        results.forEach((r, idx) => {
            const categories = ['coding', 'reasoning', 'instruction', 'creative', 'home_automation', 'gamedev', 'appdev', 'linux_admin', 'webdev', 'database', 'cpp', 'java', 'debugging', 'logic', 'retrogames', 'threedprint', 'languages', 'tvdev', 'uiux', 'office', 'life', 'biblical', 'metacog'];
            const data = categories.map(c => {
                const catData = r[`category_${c}`];
                if (!catData) return 0;
                return catData.tests_run > 0 ? ((catData.tests_passed / catData.tests_run) * 100) : 0;
            });
            const displayName = truncateModelName(r.model);
            datasets.push({
                label: displayName,
                originalLabel: r.model,
                data: data,
                backgroundColor: modelColor(idx, 0.7),
                borderRadius: 4,
                borderWidth: 0
            });
        });

        categoryChart.data.datasets = datasets;
        categoryChart.update();
    }

    // Render SharedLLM Charts
    function renderSharedChartsFromData(results) {
        ensureFilterModels(results, 'shared');
        results = getFilteredResults(results, 'shared');
        renderSharedLeaderboard(results);
        if (results.length === 0) {
            sharedLatencyChart.data.datasets.forEach(d => d.data = []);
            sharedAstChart.data.datasets = [];
            sharedLatencyChart.update();
            sharedAstChart.update();
            return;
        }
        const models = results.map(r => r.model);
        
        // 1. Latency Chart per tier
        const fastpathLats = [];
        const librarianLats = [];
        const ravenLats = [];
        const troubleshootLats = [];
        const mediaDocsLats = [];
        const chainingLats = [];
        
        results.forEach(m => {
            const getAvgLat = (catMatcher) => {
                const matching = (m.tasks || []).filter(t => catMatcher(t.test_category || '', t.test_id || ''));
                if (matching.length === 0) return 0;
                const sum = matching.reduce((acc, cur) => acc + (cur.latency || 0), 0);
                return +(sum / matching.length).toFixed(2);
            };

            fastpathLats.push(getAvgLat((cat) => cat.includes('FastPath')));
            librarianLats.push(getAvgLat((cat) => cat.includes('Librarian')));
            ravenLats.push(getAvgLat((cat) => cat.includes('Raven Code')));
            troubleshootLats.push(getAvgLat((cat) => cat.includes('Troubleshoot') || cat.includes('Planning')));
            mediaDocsLats.push(getAvgLat((cat) => cat.includes('Media') || cat.includes('Word Processing')));
            chainingLats.push(getAvgLat((cat) => cat.includes('Chaining')));
        });

        const displayModels = models.map(m => truncateModelName(m));

        sharedLatencyChart.data.labels = displayModels;
        sharedLatencyChart.data.originalLabels = models;
        sharedLatencyChart.data.datasets[0].data = fastpathLats;
        sharedLatencyChart.data.datasets[1].data = librarianLats;
        sharedLatencyChart.data.datasets[2].data = ravenLats;
        if (sharedLatencyChart.data.datasets[3]) sharedLatencyChart.data.datasets[3].data = troubleshootLats;
        if (sharedLatencyChart.data.datasets[4]) sharedLatencyChart.data.datasets[4].data = mediaDocsLats;
        if (sharedLatencyChart.data.datasets[5]) sharedLatencyChart.data.datasets[5].data = chainingLats;
        sharedLatencyChart.update();

        // 2. AST compliance rates
        const datasets = [];

        results.forEach((m, idx) => {
            let syntaxCount = 0, schemaCount = 0, contractCount = 0, totalPassCount = 0;
            const tasks = m.tasks || [];
            const totalTasks = tasks.length || 1;

            tasks.forEach(t => {
                const v = t.validation || {};
                if (t.success || v.valid_syntax || v.valid_json || v.valid_patch_format || v.has_headings) syntaxCount++;
                if (t.success || v.has_model || v.has_class || v.has_required_keys || v.has_table) schemaCount++;
                if (t.success || v.correct_intent || v.has_func || v.tool_match || v.needle_found) contractCount++;
                if (t.success) totalPassCount++;
            });

            const syntaxPct = Math.round((syntaxCount / totalTasks) * 100);
            const schemaPct = Math.round((schemaCount / totalTasks) * 100);
            const contractPct = Math.round((contractCount / totalTasks) * 100);
            const overallPct = Math.round((totalPassCount / totalTasks) * 100);

            datasets.push({
                label: truncateModelName(m.model),
                originalLabel: m.model,
                data: [syntaxPct, schemaPct, contractPct, overallPct],
                backgroundColor: modelColor(idx, 0.75),
                borderRadius: 4
            });
        });

        sharedAstChart.data.datasets = datasets;
        sharedAstChart.update();
    }

    // Benchmark Leaderboard
    // Sortable, color-coded comparison table inspired by common OSS LLM
    // benchmark dashboards (leaderboard table, score classes, best-value
    // highlighting).
    const LEADERBOARD_SORTS = {};

    // Procedurally generate a distinct color per model index so charts and
    // leaderboards stay legible with dozens of benchmarked models (no fixed
    // palette caps). Golden-angle hue spacing avoids adjacent similar hues.
    function modelColor(idx, alpha) {
        const hue = (idx * 137.508) % 360;
        const sat = 68 + ((idx * 7) % 20);
        const light = 58 + ((idx * 5) % 14);
        return `hsla(${hue.toFixed(0)}, ${sat}%, ${light}%, ${alpha})`;
    }

    function scoreClass(score) {
        if (score >= 80) return 'score-good';
        if (score >= 60) return 'score-mid';
        return 'score-bad';
    }

    function scoreBar(score, maxScore) {
        const pct = maxScore > 0 ? Math.max(0, Math.min(100, (score / maxScore) * 100)) : 0;
        let cls = 'lb-bar-bad';
        if (pct >= 80) cls = 'lb-bar-good';
        else if (pct >= 55) cls = 'lb-bar-mid';
        return `<span class="mini-scorebar ${cls}" style="width:${Math.max(18, pct)}px;"></span>`;
    }

    function rankBadge(idx) {
        if (idx === 0) return '<td class="rank-cell rank-gold">🥇</td>';
        if (idx === 1) return '<td class="rank-cell rank-silver">🥈</td>';
        if (idx === 2) return '<td class="rank-cell rank-bronze">🥉</td>';
        return `<td class="rank-cell">${idx + 1}</td>`;
    }

    function computeGeneralRow(m) {
        const cats = ['coding', 'reasoning', 'instruction', 'creative', 'home_automation', 'gamedev', 'appdev', 'linux_admin', 'webdev', 'database', 'cpp', 'java', 'debugging', 'logic', 'retrogames', 'threedprint', 'languages', 'tvdev', 'uiux', 'office', 'life', 'biblical', 'metacog'];
        let passed = 0, run = 0, tpsSum = 0, tpsN = 0, ttftSum = 0, ttftN = 0, tokSum = 0, tokN = 0;
        cats.forEach(cat => {
            const cd = m[`category_${cat}`];
            if (!cd) return;
            passed += cd.tests_passed || 0;
            run += cd.tests_run || 0;
            if (cd.avg_tokens_per_sec > 0) { tpsSum += cd.avg_tokens_per_sec; tpsN++; }
            if (cd.avg_ttft_ms > 0) { ttftSum += cd.avg_ttft_ms; ttftN++; }
            if (cd.avg_tokens_generated > 0) { tokSum += cd.avg_tokens_generated; tokN++; }
        });
        const success = run > 0 ? (passed / run) * 100 : 0;
        const tps = tpsN > 0 ? tpsSum / tpsN : 0;
        const ttft = ttftN > 0 ? ttftSum / ttftN : 0;
        const tokens = tokN > 0 ? tokSum / tokN : 0;
        const score = Math.round((success * 0.8) + (Math.min(100, tps / 10) * 0.2));
        return { model: m.model, score, success, tps, ttft, tokens, tests: run };
    }

    function computeSharedRow(m) {
        const tasks = m.tasks || [];
        let passed = 0, latSum = 0, latN = 0, tokSum = 0, tokN = 0;
        tasks.forEach(t => {
            if (t.success) passed++;
            if (typeof t.latency === 'number') { latSum += t.latency; latN++; }
            if (t.tokens_generated > 0) { tokSum += t.tokens_generated; tokN++; }
        });
        const success = tasks.length > 0 ? (passed / tasks.length) * 100 : 0;
        const latency = latN > 0 ? latSum / latN : 0;
        const tokens = tokN > 0 ? tokSum / tokN : 0;
        return { model: m.model, score: Math.round(success), success, latency, tokens, tests: tasks.length };
    }

    function bindLeaderboardSorts(tableEl, bodyEl, rows) {
        const setSort = (th) => {
            const key = th.dataset.sort;
            const dir = LEADERBOARD_SORTS[key] === 'asc' ? 'desc' : 'asc';
            LEADERBOARD_SORTS[key] = dir;
            tableEl.querySelectorAll('th.sortable').forEach(h => {
                h.classList.remove('sorted-asc', 'sorted-desc');
            });
            th.classList.add(dir === 'asc' ? 'sorted-asc' : 'sorted-desc');
            const sorted = [...rows].sort((a, b) => {
                let av = a[key], bv = b[key];
                if (key === 'rank' || key === 'model') {
                    return dir === 'asc' ? String(av).localeCompare(String(bv)) : String(bv).localeCompare(String(av));
                }
                av = Number(av) || 0; bv = Number(bv) || 0;
                return dir === 'asc' ? av - bv : bv - av;
            });
            renderLeaderboardRows(bodyEl, sorted, key);
        };
        tableEl.querySelectorAll('th.sortable').forEach(th => {
            th.addEventListener('click', () => setSort(th));
        });
    }

    function renderGeneralLeaderboard(results) {
        const body = document.getElementById('general-leaderboard-body');
        const table = document.getElementById('general-leaderboard');
        if (!body || !table) return;
        if (!results || results.length === 0) {
            body.innerHTML = '<tr class="empty-state"><td colspan="9">No benchmark results loaded.</td></tr>';
            return;
        }
        const rows = results.map(computeGeneralRow);
        bindLeaderboardSorts(table, body, rows);
        renderLeaderboardRows(body, rows, 'score', true);
    }

    function renderSharedLeaderboard(results) {
        const body = document.getElementById('shared-leaderboard-body');
        const table = document.getElementById('shared-leaderboard');
        if (!body || !table) return;
        if (!results || results.length === 0) {
            body.innerHTML = '<tr class="empty-state"><td colspan="8">No SharedLLM benchmark results loaded.</td></tr>';
            return;
        }
        const rows = results.map(computeSharedRow);
        bindLeaderboardSorts(table, body, rows);
        renderLeaderboardRows(body, rows, 'score', false);
    }

    // Per-leaderboard selection sets (model name -> row data) and view mode
    const LB_SELECTION = { general: new Map(), shared: new Map() };
    const LB_MODE = { general: 'top', shared: 'top' }; // 'top' (default) | 'all'
    const LB_TOP_N = 10;

    function lbViewMode(isGeneral) { return isGeneral ? LB_MODE.general : LB_MODE.shared; }

    function lbRowsForBody(rows, isGeneral) {
        const mode = lbViewMode(isGeneral);
        return mode === 'all' ? rows : rows.slice(0, LB_TOP_N);
    }

    function lbSelectionKey(isGeneral) { return isGeneral ? 'general' : 'shared'; }

    function updateLbCompareBar(isGeneral) {
        const key = lbSelectionKey(isGeneral);
        const bar = document.getElementById(`${key}-lb-compare-bar`);
        const countEl = document.getElementById(`${key}-lb-compare-count`);
        const goBtn = document.getElementById(`${key}-lb-compare-go`);
        const n = LB_SELECTION[key].size;
        if (bar) bar.style.display = n > 0 ? 'flex' : 'none';
        if (countEl) countEl.textContent = `${n} selected`;
        if (goBtn) goBtn.textContent = `⇄ Compare (${n})`;
    }

    function renderLeaderboardComparePanel(isGeneral) {
        const key = lbSelectionKey(isGeneral);
        const panel = document.getElementById(`${key}-lb-compare-panel`);
        if (!panel) return;
        const selected = [...LB_SELECTION[key].values()];
        if (selected.length === 0) { panel.innerHTML = ''; return; }

        // Determine best value per metric for winner highlighting
        const metrics = isGeneral
            ? ['score', 'success', 'tps', 'ttft', 'tokens']
            : ['score', 'success', 'latency', 'tokens'];
        const best = {};
        metrics.forEach(m => {
            const lowerBetter = (m === 'ttft' || m === 'latency');
            let val = lowerBetter ? Infinity : -Infinity;
            selected.forEach(r => {
                const v = Number(r[m]) || 0;
                if (v <= 0) return;
                if (lowerBetter ? v < val : v > val) val = v;
            });
            best[m] = val;
        });
        const topScore = Math.max(...selected.map(r => Number(r.score) || 0), 0);

        const label = (m) => ({
            score: 'Score', success: isGeneral ? 'Success %' : 'Pass %', tps: 'Tokens/s',
            ttft: 'TTFT (ms)', tokens: 'Avg Tokens', latency: 'Avg Latency (s)'
        })[m];
        const fmt = (m, v) => {
            v = Number(v) || 0;
            if (m === 'score' || m === 'success') return `${Math.round(v)}`;
            if (m === 'ttft') return `${Math.round(v)} ms`;
            if (m === 'latency') return `${v.toFixed(2)}s`;
            return v > 0 ? v.toFixed(1) : '-';
        };
        panel.innerHTML = `<div class="compare-grid">${selected.map(r => {
            const isWinner = Number(r.score) === topScore && topScore > 0;
            const cells = metrics.map(m => {
                const v = Number(r[m]) || 0;
                const isBest = v > 0 && v === best[m];
                return `<div class="compare-metric">
                    <span class="m-label">${label(m)}</span>
                    <span class="m-value ${isBest ? 'best-value' : ''}">${fmt(m, v)}</span>
                </div>`;
            }).join('');
            return `<div class="compare-card ${isWinner ? 'winner' : ''}">
                <div class="compare-card-header">
                    <span class="model-cell" title="${escapeHtml(r.model)}">${escapeHtml(truncateModelName(r.model))}</span>
                    ${isWinner ? '<span class="compare-winner-badge">🏆 Best</span>' : ''}
                </div>
                ${cells}
            </div>`;
        }).join('')}</div>${selected.length >= 2
            ? `<div class="radar-wrapper chart-wrapper"><canvas id="${key}-radar-chart"></canvas></div>`
            : ''}`;
        renderCompareRadar(isGeneral, key, selected);
    }

    // Radar/spider profile of per-category success % for the compared models.
    function renderCompareRadar(isGeneral, key, selected) {
        const canvas = document.getElementById(`${key}-radar-chart`);
        if (!canvas || typeof Chart === 'undefined' || selected.length < 2) return;
        const source = isGeneral ? currentResults : currentSharedResults;
        const profiles = selected.map(r =>
            source.find(rec => rec.model === r.model)
        ).filter(Boolean);
        if (profiles.length < 2) return;

        // Category -> success% extraction.
        let axes = [];
        const values = [];
        if (isGeneral) {
            const catKeys = [...new Set(profiles.flatMap(p => Object.keys(p).filter(k => k.startsWith('category_'))))];
            axes = catKeys.map(k => k.replace('category_', '').replace(/_/g, ' '));
            profiles.forEach(p => {
                values.push(catKeys.map(k => {
                    const cd = p[k];
                    return cd && cd.tests_run > 0 ? Math.round((cd.tests_passed / cd.tests_run) * 100) : 0;
                }));
            });
        } else {
            const catSet = new Set();
            profiles.forEach(p => (p.tasks || []).forEach(t => { if (t.category) catSet.add(t.category); }));
            axes = [...catSet].map(c => String(c).replace(/_/g, ' '));
            profiles.forEach(p => {
                const byCat = {};
                (p.tasks || []).forEach(t => {
                    if (!t.category) return;
                    byCat[t.category] = byCat[t.category] || { run: 0, passed: 0 };
                    byCat[t.category].run++;
                    if (t.success) byCat[t.category].passed++;
                });
                values.push(axes.map((_, i) => {
                    const c = [...catSet][i];
                    const b = byCat[c];
                    return b && b.run > 0 ? Math.round((b.passed / b.run) * 100) : 0;
                }));
            });
        }
        if (axes.length < 3) return; // radar needs at least a triangle

        const id = `${key}-radar-chart`;
        if (RADAR_CHARTS[id]) RADAR_CHARTS[id].destroy();
        RADAR_CHARTS[id] = new Chart(canvas.getContext('2d'), {
            type: 'radar',
            data: {
                labels: axes,
                datasets: profiles.map((p, i) => ({
                    label: truncateModelName(p.model),
                    data: values[i],
                    borderColor: TREND_COLORS[i % TREND_COLORS.length],
                    backgroundColor: TREND_COLORS[i % TREND_COLORS.length] + '22',
                    pointRadius: 2,
                    borderWidth: 2,
                })),
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: { r: { min: 0, max: 100, ticks: { stepSize: 25, font: { size: 9 } }, pointLabels: { font: { size: 9 } } } },
                plugins: { legend: { position: 'bottom', labels: { boxWidth: 12, font: { size: 10 } } } },
            },
        });
    }

    function renderLeaderboardRows(body, rows, sortKey, isGeneral) {
        const ordered = [...rows].sort((a, b) => (Number(b[sortKey]) || 0) - (Number(a[sortKey]) || 0));
        const visibleRows = lbRowsForBody(ordered, isGeneral);
        const maxScore = Math.max(...ordered.map(r => r.score), 1);
        const maxTps = Math.max(...ordered.map(r => r.tps), 1);
        const maxTokens = Math.max(...ordered.map(r => r.tokens), 1);

        // Best (or for TTFT/latency: lowest) value per column for highlighting
        let bestTps = maxTps, bestTokens = maxTokens;
        let bestTtft = Infinity, bestLatency = Infinity;
        ordered.forEach(r => {
            if (r.ttft > 0 && r.ttft < bestTtft) bestTtft = r.ttft;
            if (r.latency > 0 && r.latency < bestLatency) bestLatency = r.latency;
        });

        const key = lbSelectionKey(isGeneral);
        body.innerHTML = visibleRows.map((r, idx) => {
            const score = Math.round(r.score);
            const checked = LB_SELECTION[key].has(r.model) ? 'checked' : '';
            const scoreTxt = isGeneral
                ? `${scoreBar(score, maxScore)}${score}`
                : `${score}%`;
            const tpsTxt = isGeneral
                ? `${r.tps > 0 ? r.tps.toFixed(1) : '-'}`
                : '';
            const ttftTxt = isGeneral && r.ttft > 0 ? r.ttft.toFixed(0) : (isGeneral ? '-' : '');
            const latencyTxt = !isGeneral ? (r.latency > 0 ? r.latency.toFixed(2) : '-') : '';
            const successTxt = `${r.success.toFixed(0)}%`;
            const tokensTxt = r.tokens > 0 ? r.tokens.toFixed(0) : '-';
            const testsTxt = r.tests;

            const tpsCls = r.tps >= bestTps && r.tps > 0 ? 'best-value' : '';
            const tokensCls = r.tokens >= bestTokens && r.tokens > 0 ? 'best-value' : '';
            const ttftCls = r.ttft === bestTtft && r.ttft > 0 ? 'best-value' : '';
            const latencyCls = r.latency === bestLatency && r.latency > 0 ? 'best-value' : '';

            const cells = isGeneral
                ? `<td class="metric-cell ${tpsCls}">${tpsTxt}</td>
                   <td class="metric-cell ${ttftCls}">${ttftTxt}</td>
                   <td class="metric-cell">${successTxt}</td>
                   <td class="metric-cell ${tokensCls}">${tokensTxt}</td>
                   <td class="metric-cell">${testsTxt}</td>`
                : `<td class="metric-cell">${successTxt}</td>
                   <td class="metric-cell ${latencyCls}">${latencyTxt}</td>
                   <td class="metric-cell ${tokensCls}">${tokensTxt}</td>
                   <td class="metric-cell">${testsTxt}</td>`;

            return `<tr class="lb-row ${checked ? 'selected' : ''}" data-model="${r.model.replace(/"/g, '&quot;')}" data-score="${score}">
                <td class="lb-check-col"><input type="checkbox" class="lb-check" data-model="${r.model.replace(/"/g, '&quot;')}" ${checked}></td>
                ${rankBadge(idx)}
                <td class="model-cell" title="${r.model.replace(/"/g, '&quot;')}">${truncateModelName(r.model)}</td>
                <td class="score-cell ${scoreClass(score)}">${scoreTxt}</td>
                ${cells}
            </tr>`;
        }).join('');

        const selected = body.querySelectorAll('.lb-check');
        selected.forEach(cb => {
            cb.addEventListener('change', () => {
                const model = cb.dataset.model;
                if (cb.checked) {
                    const row = ordered.find(o => o.model === model);
                    if (row) LB_SELECTION[key].set(model, row);
                } else {
                    LB_SELECTION[key].delete(model);
                }
                const tr = cb.closest('tr');
                if (tr) tr.classList.toggle('selected', cb.checked);
                updateLbCompareBar(isGeneral);
                renderLeaderboardComparePanel(isGeneral);
            });
        });

        body.querySelectorAll('tr.lb-row').forEach(row => {
            row.addEventListener('click', (e) => {
                if (e.target.closest('.lb-check')) return;
                body.querySelectorAll('tr.lb-row').forEach(tr => tr.classList.remove('selected'));
                row.classList.add('selected');
                const model = row.dataset.model;
                if (isOnlineModelName(model)) return;
                const evt = new CustomEvent('leaderboard:select', { detail: { model, isGeneral } });
                document.dispatchEvent(evt);
            });
        });
    }

    // Render General Details section
    function renderDetailsSection(results, activeModelName = null) {
        modelTabs.innerHTML = '';
        detailedResultsBody.innerHTML = '';

        if (results.length === 0) {
            detailedResultsBody.innerHTML = `<tr><td colspan="10" style="text-align:center;color:var(--text-muted);">No detailed data available.</td></tr>`;
            return;
        }

        let activeIdx = 0;
        if (activeModelName) {
            const foundIdx = results.findIndex(m => m.model === activeModelName || m.model.includes(activeModelName) || activeModelName.includes(m.model));
            if (foundIdx !== -1) activeIdx = foundIdx;
        }

        results.forEach((modelData, idx) => {
            const opt = document.createElement('option');
            opt.value = modelData.model;
            opt.textContent = truncateModelName(modelData.model);
            if (idx === activeIdx) opt.selected = true;
            modelTabs.appendChild(opt);
        });

        const onSelect = () => {
            const sel = results.find(m => m.model === modelTabs.value);
            if (!sel) return;
            renderModelDetailedTable(sel);
            renderResultsSummary(sel);
        };

        modelTabs.addEventListener('change', onSelect);
        renderModelDetailedTable(results[activeIdx]);
        renderResultsSummary(results[activeIdx]);
    }

    // Render the overall + per-group benchmark summary for a single model record.
    function renderResultsSummary(modelData) {
        const card = document.getElementById('results-summary-card');
        const overallEl = document.getElementById('results-overall');
        const groupBody = document.getElementById('results-group-scores');
        const modelLabel = document.getElementById('results-summary-model');
        if (!card || !overallEl || !groupBody) return;

        const hasSummary = !!modelData &&
            ((modelData.overall_score !== undefined && modelData.overall_score !== null) ||
             (modelData.group_scores && modelData.group_scores.length > 0));
        if (!hasSummary) {
            card.style.display = 'none';
            return;
        }

        card.style.display = '';
        if (modelLabel) modelLabel.textContent = modelData.model ? `Model: ${modelData.model}` : '';

        // Overall card
        overallEl.innerHTML = '';
        const oCard = document.createElement('div');
        oCard.className = 'overall-score-card';
        const letterEl = document.createElement('div');
        const letterVal = modelData.overall_letter || '-';
        letterEl.className = `overall-letter letter-${String(letterVal).toLowerCase()}`;
        letterEl.textContent = letterVal;
        const infoEl = document.createElement('div');
        infoEl.className = 'overall-info';
        const scoreEl = document.createElement('div');
        scoreEl.className = 'overall-score-value';
        scoreEl.textContent = `${modelData.overall_score != null ? modelData.overall_score : '-'}%`;
        const starsEl = document.createElement('div');
        starsEl.className = 'overall-stars';
        starsEl.textContent = modelData.overall_stars || '';
        infoEl.appendChild(scoreEl);
        infoEl.appendChild(starsEl);
        oCard.appendChild(letterEl);
        oCard.appendChild(infoEl);
        overallEl.appendChild(oCard);

        // Per-group table (sorted by group name for easy comparison)
        groupBody.innerHTML = '';
        const groups = (modelData.group_scores || []).slice()
            .sort((a, b) => String(a.group).localeCompare(String(b.group)));
        if (groups.length === 0) {
            groupBody.innerHTML = `<tr><td colspan="5" style="text-align:center;color:var(--text-muted);">No group scores recorded.</td></tr>`;
            return;
        }
        groups.forEach(g => {
            const tr = document.createElement('tr');
            const tdG = document.createElement('td');
            tdG.textContent = String(g.group).replace(/_/g, ' ');
            const tdS = document.createElement('td');
            tdS.textContent = `${g.score != null ? g.score : '-'}%`;
            const tdL = document.createElement('td');
            tdL.className = `group-letter letter-${String(g.letter || '-').toLowerCase()}`;
            tdL.textContent = g.letter || '-';
            const tdSt = document.createElement('td');
            tdSt.className = 'group-stars';
            tdSt.textContent = g.stars || '';
            const tdP = document.createElement('td');
            tdP.textContent = `${g.tests_passed != null ? g.tests_passed : 0} / ${g.tests_run != null ? g.tests_run : 0}`;
            tr.appendChild(tdG);
            tr.appendChild(tdS);
            tr.appendChild(tdL);
            tr.appendChild(tdSt);
            tr.appendChild(tdP);
            groupBody.appendChild(tr);
        });
    }

    function setDetailedRating(tdRating, ratingKey, modelName, val) {
        _saveHumanRating(ratingKey, modelName, val);
        tdRating.innerHTML = _starDisplayHtml(val, true);
        _bindStarSlots(tdRating, (v) => setDetailedRating(tdRating, ratingKey, modelName, v));
        if (typeof renderRatingsBoard === 'function') renderRatingsBoard();
        if (typeof renderTestBrowser === 'function') renderTestBrowser();
    }

    function renderModelDetailedTable(modelData) {
        detailedResultsBody.innerHTML = '';
        const categories = ['coding', 'reasoning', 'instruction', 'creative', 'home_automation', 'gamedev', 'appdev', 'linux_admin', 'webdev', 'database', 'cpp', 'java', 'debugging', 'logic', 'retrogames', 'threedprint', 'languages', 'tvdev', 'uiux', 'office', 'life', 'biblical', 'metacog'];
        let hasRows = false;

        categories.forEach(catKey => {
            const catStats = modelData[`category_${catKey}`];
            if (!catStats || !catStats.tests) return;

            catStats.tests.forEach(test => {
                hasRows = true;
                const tr = document.createElement('tr');
                
                const tdCat = document.createElement('td');
                tdCat.className = 'td-category';
                tdCat.textContent = catKey.replace('_', ' ');
                
                const tdLabel = document.createElement('td');
                tdLabel.textContent = test.test_label;
                
                const tdStatus = document.createElement('td');
                const badge = document.createElement('span');
                badge.className = `td-badge ${test.success ? 'success' : 'fail'}`;
                badge.textContent = test.success ? 'Success' : 'Fail';
                tdStatus.appendChild(badge);

                const tdScore = document.createElement('td');
                tdScore.style.fontWeight = '600';
                tdScore.textContent = (test.score !== undefined && test.score !== null) ? `${test.score}` : '-';

                const tdLastRun = document.createElement('td');
                tdLastRun.style.color = 'var(--text-muted)';
                tdLastRun.textContent = test.last_run ? test.last_run.replace('T', ' ') : '-';

                const tdLat = document.createElement('td');
                let latVal = test.eval_duration && test.prompt_eval_duration ? (test.eval_duration + test.prompt_eval_duration) / 1e9 : test.latency;
                tdLat.textContent = latVal ? `${latVal.toFixed(2)}s` : '-';

                const tdSpeed = document.createElement('td');
                if (test.success && test.tokens_generated > 0) {
                    const duration = test.eval_duration && test.prompt_eval_duration ? (test.eval_duration + test.prompt_eval_duration) / 1e9 : test.latency;
                    const tps = duration > 0 ? (test.tokens_generated / duration) : 0;
                    tdSpeed.textContent = `${tps.toFixed(1)} tok/s`;
                } else {
                    tdSpeed.textContent = '-';
                }

                const tdView = document.createElement('td');
                const promptLink = document.createElement('span');
                promptLink.className = 'prompt-text';
                promptLink.textContent = 'View Prompt & Response';
                promptLink.addEventListener('click', () => {
                    const errorMsg = test.error ? `Error: ${test.error}` : 'No response.';
                    openModal(test.prompt || '(no prompt recorded)', test.response || errorMsg, test.thinking);
                });
                tdView.appendChild(promptLink);

                const runLink = document.createElement('span');
                runLink.className = 'prompt-text';
                runLink.textContent = '▶ Run code';
                runLink.style.marginLeft = '0.6rem';
                runLink.addEventListener('click', () => {
                    if (!test.response) { alert('No response to run.'); return; }
                    openResponseRunner(test.response, test.test_label, test.thinking);
                });
                tdView.appendChild(runLink);

                // Artifacts column: screenshot thumbnail + Serve & View for code/UI tests.
                const tdArt = document.createElement('td');
                tdArt.style.cssText = 'display:flex; gap:0.4rem; align-items:center; flex-wrap:wrap;';
                if (test.screenshot) {
                    const img = document.createElement('img');
                    img.src = `data:image/png;base64,${test.screenshot}`;
                    img.className = 'screenshot-thumb';
                    img.title = 'Click to enlarge';
                    img.addEventListener('click', (e) => {
                        e.stopPropagation();
                        openScreenshotLightbox(img.src);
                    });
                    tdArt.appendChild(img);
                }
                if (isCodeUiCategory(catKey) && test.response && extractRunnableCode(test.response).trim()) {
                    tdArt.appendChild(createServeButton(test.response, catKey));
                }

                // Human rating stars — same store/key as the Test Browser card
                // (localStorage 'alpaca_human_ratings' -> testId -> model -> 1..5),
                // so a rating given here shows on the card and vice versa.
                const tdRating = document.createElement('td');
                tdRating.className = 'rating-stars detail-rating';
                tdRating.style.cursor = 'pointer';
                const ratingKey = test.test_id || test.id || test.test_label;
                const saved = _loadHumanRatings(ratingKey) || {};
                const modelName = modelData.model;
                const curRating = saved[modelName] || 0;
                tdRating.setAttribute('data-test', ratingKey);
                tdRating.setAttribute('data-model', modelName);
                tdRating.innerHTML = _starDisplayHtml(curRating, true);
                _bindStarSlots(tdRating, (v) => setDetailedRating(tdRating, ratingKey, modelName, v));

                tr.appendChild(tdCat);
                tr.appendChild(tdLabel);
                tr.appendChild(tdStatus);
                tr.appendChild(tdScore);
                tr.appendChild(tdLastRun);
                tr.appendChild(tdLat);
                tr.appendChild(tdSpeed);
                tr.appendChild(tdView);
                tr.appendChild(tdArt);
                tr.appendChild(tdRating);
                
                let expandedRow = null;
                tr.style.cursor = 'pointer';
                tr.addEventListener('click', (e) => {
                    if (e.target.classList.contains('prompt-text')) return;
                    if (e.target.classList.contains('screenshot-thumb')) return;
                    if (e.target.classList.contains('serve-view-btn')) return;
                    if (e.target.classList.contains('serve-stop-btn')) return;
                    if (e.target.classList.contains('star') || e.target.closest('.star-slot') || e.target.closest('.detail-rating')) return;
                    
                    if (expandedRow) {
                        expandedRow.remove();
                        expandedRow = null;
                        tr.classList.remove('expanded-parent');
                    } else {
                        expandedRow = document.createElement('tr');
                        expandedRow.className = 'expanded-row';
                        const tdFull = document.createElement('td');
                        tdFull.colSpan = 10;
                        tdFull.style.cssText = 'background: rgba(15, 23, 42, 0.4); padding: 1rem; border-bottom: 1px solid rgba(255, 255, 255, 0.04);';
                        
                        const flex = document.createElement('div');
                        flex.style.cssText = 'display:flex; flex-direction:column; gap:0.5rem;';
                        
                        const title = document.createElement('div');
                        title.style.cssText = 'font-weight: 600; font-size: 0.75rem; color: var(--color-primary);';
                        title.textContent = 'Inline Result Preview:';
                        
                        const codeBlock = document.createElement('pre');
                        codeBlock.style.cssText = 'margin: 0; background: rgba(9, 15, 29, 0.95); border: 1px solid rgba(255, 255, 255, 0.05); padding: 0.75rem; border-radius: 6px; font-family: monospace; font-size: 0.75rem; overflow-x: auto; white-space: pre-wrap; word-break: break-word; color: #e2e8f0; max-height: 300px;';
                        codeBlock.textContent = test.response || test.error || 'No response recorded';

                        flex.appendChild(title);
                        flex.appendChild(codeBlock);

                        // Sandbox run output (code_ran / code_output / code_error) if present.
                        if (test.code_output || test.code_error || test.code_ran !== undefined) {
                            const ranNote = document.createElement('div');
                            ranNote.style.cssText = 'font-size:0.68rem; color:var(--text-muted);';
                            ranNote.textContent = `Code executed: ${test.code_ran ? 'yes' : 'no'}${test.code_score != null ? ` | code score: ${test.code_score}` : ''}`;
                            const sbTitle = document.createElement('div');
                            sbTitle.style.cssText = 'font-weight:600; font-size:0.72rem; color: var(--color-secondary); margin-top:0.25rem;';
                            sbTitle.textContent = 'Sandbox Output:';
                            const sbPre = document.createElement('pre');
                            sbPre.className = 'sandbox-output-pre';
                            sbPre.style.cssText = 'margin:0; background: rgba(9,15,29,0.95); border:1px solid rgba(255,255,255,0.05); padding:0.6rem; border-radius:6px; font-family:monospace; font-size:0.72rem; white-space:pre-wrap; word-break:break-word; color:#cbd5e1; max-height:200px; overflow:auto;';
                            sbPre.textContent = [test.code_output || '', test.code_error || ''].filter(Boolean).join('\n') || '(no output captured)';
                            flex.appendChild(ranNote);
                            flex.appendChild(sbTitle);
                            flex.appendChild(sbPre);
                        }

                        // Inline rendered HTML preview for webdev responses.
                        if (catKey === 'webdev' && test.response) {
                            const htmlCode = extractHtmlDocument(test.response) || extractRunnableCode(test.response);
                            if (htmlCode && /<!doctype|<html|<script/i.test(htmlCode)) {
                                const htmlNote = document.createElement('div');
                                htmlNote.style.cssText = 'font-weight:600; font-size:0.72rem; color: var(--color-secondary); margin-top:0.25rem;';
                                htmlNote.textContent = 'Inline Rendered Preview (webdev):';
                                const frame = document.createElement('iframe');
                                frame.className = 'test-preview-iframe';
                                frame.setAttribute('sandbox', 'allow-scripts');
                                frame.srcdoc = htmlCode;
                                flex.appendChild(htmlNote);
                                flex.appendChild(frame);
                            }
                        }

                        addArtifactButtons(flex, modelData.model, test.test_id, test.response);
                        tdFull.appendChild(flex);
                        expandedRow.appendChild(tdFull);
                        
                        tr.parentNode.insertBefore(expandedRow, tr.nextSibling);
                        tr.classList.add('expanded-parent');
                    }
                });
                
                detailedResultsBody.appendChild(tr);
            });
        });

        if (!hasRows) {
            detailedResultsBody.innerHTML = `<tr><td colspan="10" style="text-align:center;color:var(--text-muted);">No test logs for this model</td></tr>`;
        }
    }

    // Render SharedLLM Details Section
    function renderSharedDetailsSection(results, activeModelName = null) {
        sharedModelTabs.innerHTML = '';
        sharedDetailedResultsBody.innerHTML = '';

        if (results.length === 0) {
            sharedDetailedResultsBody.innerHTML = `<tr><td colspan="6" style="text-align:center;color:var(--text-muted);">No detailed data available.</td></tr>`;
            return;
        }

        let activeIdx = 0;
        if (activeModelName) {
            const foundIdx = results.findIndex(m => m.model === activeModelName || m.model.includes(activeModelName) || activeModelName.includes(m.model));
            if (foundIdx !== -1) activeIdx = foundIdx;
        }

        results.forEach((modelData, idx) => {
            const opt = document.createElement('option');
            opt.value = modelData.model;
            opt.textContent = truncateModelName(modelData.model);
            if (idx === activeIdx) opt.selected = true;
            sharedModelTabs.appendChild(opt);
        });

        const onSelect = () => {
            const sel = results.find(m => m.model === sharedModelTabs.value);
            if (!sel) return;
            renderSharedModelDetailedTable(sel);
        };

        sharedModelTabs.addEventListener('change', onSelect);
        renderSharedModelDetailedTable(results[activeIdx]);
    }

    function renderSharedModelDetailedTable(modelData) {
        sharedDetailedResultsBody.innerHTML = '';
        
        if (!modelData.tasks) {
            sharedDetailedResultsBody.innerHTML = `<tr><td colspan="6" style="text-align:center;color:var(--text-muted);">No detailed task data for this model.</td></tr>`;
            return;
        }

        modelData.tasks.forEach(task => {
            const tr = document.createElement('tr');
            
            const tdCat = document.createElement('td');
            tdCat.className = 'td-category';
            tdCat.textContent = task.test_category;
            
            const tdLabel = document.createElement('td');
            tdLabel.textContent = task.test_label;
            
            const tdStatus = document.createElement('td');
            const badge = document.createElement('span');
            badge.className = `td-badge ${task.success ? 'success' : 'fail'}`;
            badge.textContent = task.success ? 'Pass' : 'Fail';
            tdStatus.appendChild(badge);

            const tdLat = document.createElement('td');
            const latency = typeof task.latency === 'number' ? task.latency : 0;
            tdLat.textContent = `${latency.toFixed(2)}s`;

            // Custom Payload descriptions
            const tdPayload = document.createElement('td');
            const val = task.validation || {};
            
            if (task.test_id.startsWith('fast_path')) {
                tdPayload.textContent = `Intent: "${val.actual || ''}" (${val.correct_intent ? '✓ Match' : '✗ Expected: ' + (val.expected || '')})`;
            } else if (task.test_id.startsWith('tool_')) {
                tdPayload.textContent = `JSON: ${val.valid_json ? '✓' : '✗'} | Tool: "${val.parsed?.tool || 'None'}"`;
            } else if (task.test_id.startsWith('code_')) {
                const checks = document.createElement('div');
                checks.className = 'ast-check-list';
                
                const synBadge = document.createElement('span');
                synBadge.className = `ast-check-badge ${val.valid_syntax ? 'checked' : 'failed'}`;
                synBadge.textContent = val.valid_syntax ? '✓ Syntax' : '✗ Syntax';
                checks.appendChild(synBadge);

                if (val.has_class !== undefined) {
                    const classBadge = document.createElement('span');
                    classBadge.className = `ast-check-badge ${val.has_class ? 'checked' : 'failed'}`;
                    classBadge.textContent = val.has_class ? '✓ Class' : '✗ No Class';
                    checks.appendChild(classBadge);
                }
                if (val.has_acquire !== undefined) {
                    const acqBadge = document.createElement('span');
                    acqBadge.className = `ast-check-badge ${val.has_acquire ? 'checked' : 'failed'}`;
                    acqBadge.textContent = val.has_acquire ? '✓ acquire()' : '✗ acquire()';
                    checks.appendChild(acqBadge);
                }
                if (val.has_model !== undefined) {
                    const modelBadge = document.createElement('span');
                    modelBadge.className = `ast-check-badge ${val.has_model ? 'checked' : 'failed'}`;
                    modelBadge.textContent = val.has_model ? '✓ Pydantic' : '✗ No Model';
                    checks.appendChild(modelBadge);
                }
                if (val.has_func !== undefined) {
                    const funcBadge = document.createElement('span');
                    funcBadge.className = `ast-check-badge ${val.has_func ? 'checked' : 'failed'}`;
                    funcBadge.textContent = val.has_func ? '✓ Function' : '✗ No Function';
                    checks.appendChild(funcBadge);
                }
                tdPayload.appendChild(checks);
            } else if (task.test_id.startsWith('troubleshoot') || task.test_id.startsWith('chaining') || task.test_id.startsWith('media') || task.test_id.startsWith('raven')) {
                if (val.valid_patch_format !== undefined) {
                    tdPayload.textContent = `Unified Git Diff: ${val.valid_patch_format ? '✓ Valid Patch' : '✗ Invalid Patch Header'}`;
                } else if (val.valid_json) {
                    tdPayload.textContent = `Structured Schema: ✓ Complete (${Object.keys(val.parsed || {}).length} keys)`;
                } else {
                    tdPayload.textContent = `Validation: ${task.success ? '✓ Passed' : '✗ Incomplete'}`;
                }
            } else if (task.test_id.startsWith('wordproc')) {
                if (val.has_headings !== undefined) {
                    tdPayload.textContent = `Markdown: ${val.has_headings ? '✓ Headings' : '✗ Headings'} | ${val.has_table ? '✓ Table' : '✗ Table'}`;
                } else {
                    tdPayload.textContent = `Validation: ${task.success ? '✓ Passed' : '✗ Incomplete'}`;
                }
            } else if (task.test_id.startsWith('needle')) {
                tdPayload.textContent = `Secret Token: ${val.needle_found ? '✓ Found: ' + (val.expected || '') : '✗ Not Found'}`;
            } else if (task.test_category === 'multistep_gamedev' || task.artifact_url) {
                const bits = [];
                if (typeof task.score === 'number') bits.push(`Score: ${task.score.toFixed(1)}/100`);
                if (task.steps_completed !== undefined) bits.push(`Steps: ${task.steps_completed}/${task.steps_total}`);
                if (val.ui_render) {
                    bits.push(val.ui_render.ran ? '✓ Renders' : `✗ Render${val.ui_render.error ? ': ' + val.ui_render.error : ''}`);
                }
                tdPayload.textContent = bits.join(' | ') || `Status: ${task.success ? '✓ Pass' : '✗ Fail'}`;
            } else {
                tdPayload.textContent = `Status: ${task.success ? '✓ Pass' : '✗ Fail'}`;
            }

            const tdView = document.createElement('td');
            if (task.artifact_url) {
                const fullLink = document.createElement('span');
                fullLink.className = 'prompt-text';
                fullLink.style.marginRight = '0.75rem';
                fullLink.textContent = '🎮 View Full Game';
                fullLink.title = 'Open the complete generated HTML document';
                fullLink.addEventListener('click', () => window.open(task.artifact_url, '_blank'));
                tdView.appendChild(fullLink);
            }
            const promptLink = document.createElement('span');
            promptLink.className = 'prompt-text';
            promptLink.textContent = 'View Code / Payload';
            promptLink.addEventListener('click', async () => {
                const details = task.error ? `Error: ${task.error}` : '';
                let body = task.response || details;
                // Older snapshots stored only the last ~4000 chars of the
                // generated document; when the full artifact exists on disk,
                // fetch it so the payload modal shows the COMPLETE code.
                if (task.artifact_url) {
                    try {
                        const res = await fetch(task.artifact_url);
                        if (res.ok) body = await res.text();
                    } catch (e) { /* keep inline excerpt on failure */ }
                }
                openModal(task.prompt, body);
            });
            tdView.appendChild(promptLink);

            tr.appendChild(tdCat);
            tr.appendChild(tdLabel);
            tr.appendChild(tdStatus);
            tr.appendChild(tdLat);
            tr.appendChild(tdPayload);
            tr.appendChild(tdView);
            
            let expandedRow = null;
            tr.style.cursor = 'pointer';
            tr.addEventListener('click', (e) => {
                if (e.target.classList.contains('prompt-text')) return;
                
                if (expandedRow) {
                    expandedRow.remove();
                    expandedRow = null;
                    tr.classList.remove('expanded-parent');
                } else {
                    expandedRow = document.createElement('tr');
                    expandedRow.className = 'expanded-row';
                    const tdFull = document.createElement('td');
                    tdFull.colSpan = 6;
                    tdFull.style.cssText = 'background: rgba(15, 23, 42, 0.4); padding: 1rem; border-bottom: 1px solid rgba(255, 255, 255, 0.04);';
                    
                    const flex = document.createElement('div');
                    flex.style.cssText = 'display:flex; flex-direction:column; gap:0.5rem;';
                    
                    const title = document.createElement('div');
                    title.style.cssText = 'font-weight: 600; font-size: 0.75rem; color: var(--color-primary);';
                    title.textContent = 'Inline Result Preview:';
                    
                    const codeBlock = document.createElement('pre');
                    codeBlock.style.cssText = 'margin: 0; background: rgba(9, 15, 29, 0.95); border: 1px solid rgba(255, 255, 255, 0.05); padding: 0.75rem; border-radius: 6px; font-family: monospace; font-size: 0.75rem; overflow-x: auto; white-space: pre-wrap; word-break: break-word; color: #e2e8f0; max-height: 300px;';
                    codeBlock.textContent = task.response || task.error || 'No response recorded';

                    flex.appendChild(title);
                    if (task.artifact_url) {
                        const note = document.createElement('div');
                        note.style.cssText = 'font-size: 0.75rem; color: var(--text-muted);';
                        note.innerHTML = 'Preview below is an excerpt — <a href="' + task.artifact_url + '" target="_blank">open the FULL generated document</a>';
                        flex.appendChild(note);
                    }
                    flex.appendChild(codeBlock);
                    addArtifactButtons(flex, modelData.model, task.test_id, task.response);
                    tdFull.appendChild(flex);
                    expandedRow.appendChild(tdFull);
                    
                    tr.parentNode.insertBefore(expandedRow, tr.nextSibling);
                    tr.classList.add('expanded-parent');
                }
            });
            
            sharedDetailedResultsBody.appendChild(tr);
        });
    }

    // Trigger General Benchmarks
    btnRun.addEventListener('click', () => triggerBenchmark('/api/run'));

    // Trigger Outdated-Only Benchmarks: re-run just the tests whose
    // definitions have changed since each model's last run.
    btnRunOutdated.addEventListener('click', () => triggerBenchmark('/api/run', { outdatedOnly: true }));

    // Trigger SharedLLM Benchmarks
    btnRunShared.addEventListener('click', () => triggerBenchmark('/api/run/shared_llm'));

    // Trigger MultiStep (agentic) Benchmarks
    // Guarded: a stale cached template without the button must not break global init.
    if (btnRunMultistep) {
        btnRunMultistep.addEventListener('click', () => triggerBenchmark('/api/run/multistep', { multistep: true }));
    }

    async function triggerBenchmark(endpoint, options = {}) {
        const selected = getSelectedModels();
        if (selected.length === 0) {
            showToast('Please select at least one model to benchmark.', 'error');
            return;
        }

            const isShared = endpoint.endsWith('shared_llm');
            const isMultistep = endpoint.endsWith('multistep');
            if (options.outdatedOnly && isShared) {
                showToast('Outdated re-runs apply to General benchmarks only.', 'error');
                return;
            }
            const selectedTests = options.testIds && options.testIds.length
                ? options.testIds
                : (isMultistep ? getSelectedMultistepWorkflows() : (isShared ? getSelectedSharedTests() : getSelectedTests()));
            const selectedGroups = isShared || isMultistep ? [] : getSelectedGroups();
            if (isMultistep && selectedTests.length === 0) {
                showToast('Please select at least one MultiStep workflow to run.', 'error');
                return;
            }
            if (isShared && selectedTests.length === 0) {
                showToast('Please select at least one SharedLLM task to run.', 'error');
                return;
            }
            // A group selection is a valid scope on its own (no individual tests needed).
            if (!isShared && !isMultistep && selectedTests.length === 0 && selectedGroups.length === 0) {
                showToast('Please select at least one test case or benchmark group to run.', 'error');
                return;
            }

        btnRun.disabled = true;
        btnRunShared.disabled = true;
        btnRunMultistep.disabled = true;
        btnRunOutdated.disabled = true;

        if (isMultistep) {
            btnRunMultistep.innerHTML = `<span class="loader"></span> Starting...`;
        } else if (isShared) {
            btnRunShared.innerHTML = `<span class="loader"></span> Starting...`;
        } else if (options.outdatedOnly) {
            btnRunOutdated.innerHTML = `<span class="loader"></span> Starting...`;
        } else {
            btnRun.innerHTML = `<span class="loader"></span> Starting...`;
        }

        const termTarget = isShared || isMultistep ? 'shared' : 'general';
        
        try {
            logToTerminal(`Initiating benchmark for models: ${selected.join(', ')}...`, 'info', termTarget);
            
            const payload = {
                models: selected,
                use_proxy: (benchmarkMode === 'proxy')
            };
            if (!isShared) {
                if (options.outdatedOnly) {
                    payload.outdated_only = true;
                }
                // Benchmark GROUPS: only send when the user explicitly chose some.
                // Omitting (or emptying) tells the backend to run ALL groups.
                if (selectedGroups.length > 0) {
                    payload.groups = selectedGroups;
                }
                // Resume is on by default: when "run all" is intended (every test
                // checkbox checked) we omit test_ids so the backend skips already
                // completed tests. Only send test_ids when a subset is selected,
                // which forces those specific tests to (re)run and overwrite.
                const resumeChk = document.getElementById('chk-resume');
                payload.resume = resumeChk ? resumeChk.checked : true;
                // Advanced tier (long-running / complex tests) is opt-in so the
                // default run stays fast and runs every standard test.
                const advChk = document.getElementById('chk-advanced');
                payload.tiers = (advChk && advChk.checked) ? ['standard', 'advanced'] : ['standard'];
                if (!options.outdatedOnly) {
                    const totalTests = testCheckboxes.querySelectorAll('input[type="checkbox"]').length || 0;
                    if (selectedTests.length < totalTests) {
                        payload.test_ids = selectedTests;
                    }
                }
            } else if (isMultistep) {
                // Only pass workflow_ids if not all workflows selected (backend treats null as "all")
                const allWorkflows = document.getElementById('multistep-workflow-checkboxes')
                    ?.querySelectorAll('input[type="checkbox"]').length || 0;
                if (selectedTests.length < allWorkflows) {
                    payload.workflow_ids = selectedTests;
                }
            } else {
                // Only pass test_ids if not all tasks selected (backend treats null as "all")
                const allSharedTasks = document.getElementById('shared-test-checkboxes')
                    ?.querySelectorAll('input[type="checkbox"]').length || 0;
                if (selectedTests.length < allSharedTasks) {
                    payload.test_ids = selectedTests;
                }
            }
            const res = await fetch(endpoint, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            if (res.status === 409) {
                const data = await res.json();
                showToast(data.error, 'error');
                setRunnerState('idle');
                return;
            }

            if (!res.ok) {
                const errorText = await res.text();
                throw new Error(errorText || 'Server error starting benchmark');
            }

            const resData = await res.json().catch(() => null);
            if (options.outdatedOnly && resData && resData.status === 'No outdated benchmarks') {
                showToast(resData.message || 'All benchmark definitions are up to date.', 'success');
                logToTerminal(resData.message || 'All benchmark definitions are up to date — nothing to redo.', 'info', termTarget);
                setRunnerState('idle');
                return;
            }

            logToTerminal("Benchmark pipeline initialized successfully.", 'success', termTarget);
            setRunnerState('running');
            
            // Ensure SocketIO is connected before switching tabs to receive progress events
            if (socket.connected) {
                if (isShared || isMultistep) {
                    switchTab('shared');
                } else {
                    switchTab('general');
                }
            } else {
                // Wait for socket connection with timeout
                let attempts = 0;
                const maxAttempts = 50; // 5 seconds
                await new Promise((resolve) => {
                    const checkSocket = setInterval(() => {
                        attempts++;
                        if (socket.connected || attempts >= maxAttempts) {
                            clearInterval(checkSocket);
                            resolve();
                        }
                    }, 100);
                });
                if (isShared || isMultistep) {
                    switchTab('shared');
                } else {
                    switchTab('general');
                }
            }
        } catch (err) {
            logToTerminal(`Failed to start benchmark: ${err.message}`, 'error', termTarget);
            setRunnerState('idle');
        }
    }

    // Cancel Active Run
    btnCancel.addEventListener('click', async () => {
        btnCancel.disabled = true;
        btnCancel.textContent = 'Stopping...';
        
        try {
            logToTerminal("Sending cancel request...", "warn");
            logToTerminal("Sending cancel request...", "warn", "shared");
            const res = await fetch('/api/cancel', { method: 'POST' });
            if (!res.ok) {
                const data = await res.json();
                logToTerminal(`Cancellation error: ${data.error}`, "error");
                btnCancel.disabled = false;
                btnCancel.textContent = 'Cancel Run';
            }
        } catch (err) {
            logToTerminal(`Network error sending cancel: ${err.message}`, 'error');
            btnCancel.disabled = false;
            btnCancel.textContent = 'Cancel Run';
        }
    });

    function setRunnerState(state) {
        if (state === 'running') {
            runnerStatusBadge.className = 'badge badge-warning';
            runnerStatusBadge.innerHTML = `<span class="badge-dot"></span> RUNNING`;
            btnRun.disabled = true;
            btnRunShared.disabled = true;
            btnRunMultistep.disabled = true;
            btnRunOutdated.disabled = true;
            btnCancel.disabled = false;
            progressCard.classList.remove('d-none');

            progressPercent.textContent = '0%';
            progressBarFill.style.width = '0%';
            progressText.textContent = 'Initializing test run...';
        } else {
            runnerStatusBadge.className = 'badge badge-success';
            runnerStatusBadge.innerHTML = `<span class="badge-dot"></span> IDLE`;
            btnRun.disabled = false;
            btnRunShared.disabled = false;
            btnRunMultistep.disabled = false;
            btnRunOutdated.disabled = false;
            btnRun.innerHTML = 'Run General';
            btnRunShared.innerHTML = 'Run SharedLLM';
            btnRunMultistep.innerHTML = '🛩 Run MultiStep';
            btnRunOutdated.innerHTML = 'Run Outdated';
            btnCancel.disabled = true;
            btnCancel.textContent = 'Cancel Run';
            progressCard.classList.add('d-none');
        }
    }

    function syncRunnerState(data) {
        if (data.status === 'running') {
            setRunnerState('running');
            const completed = data.tests_completed || 0;
            const total = data.total_tests || 1;
            const pct = Math.round((completed / total) * 100);
            
            progressPercent.textContent = `${pct}%`;
            progressBarFill.style.width = `${pct}%`;
            progressText.textContent = `Running: ${data.current_model || 'Loading'}...`;
            
            statusModel.textContent = data.current_model || '-';
            statusTest.textContent = data.current_test || '-';
            statusCategory.textContent = data.current_category ? data.current_category.replace('_', ' ') : '-';
            
            if (data.results && data.results.length > 0) {
                if (data.type === 'shared_llm') {
                    currentSharedResults = data.results;
                    renderSharedChartsFromData(currentSharedResults);
                    renderSharedDetailsSection(currentSharedResults);
                    updateSharedOverviewMetrics(data);
                } else {
                    currentResults = data.results;
                    renderChartsFromData(currentResults);
                    renderDetailsSection(currentResults);
                    updateOverviewMetrics(data);
                }
            }
        } else {
            setRunnerState('idle');
        }
    }

    // Socket Event Handling
    socket.on('connect', () => {
        connectionStatusBadge.className = 'badge badge-success';
        connectionStatusBadge.innerHTML = `<span class="badge-dot"></span> Socket Connected`;
        logToTerminal("Socket.IO client connected.", "success");
        logToTerminal("Socket.IO client connected.", "success", "shared");
    });

    socket.on('disconnect', () => {
        connectionStatusBadge.className = 'badge badge-danger';
        connectionStatusBadge.innerHTML = `<span class="badge-dot"></span> Disconnected`;
        logToTerminal("Socket.IO client disconnected.", "error");
        logToTerminal("Socket.IO client disconnected.", "error", "shared");
        setRunnerState('idle');
    });

    socket.on('sync_status', (data) => {
        syncRunnerState(data);
    });

    socket.on('benchmark_start', (data) => {
        const term = data.type === 'shared_llm' ? 'shared' : 'general';
        const numModels = data.total_models !== undefined ? data.total_models : (data.models ? data.models.length : 0);
        logToTerminal(`Benchmark started: ${data.total_tests} tests across ${numModels} models.`, "info", term);
        setRunnerState('running');
        
        if (data.type === 'shared_llm') {
            sharedLatencyChart.data.labels = [];
            sharedLatencyChart.data.datasets.forEach(d => d.data = []);
            sharedLatencyChart.update();
            
            sharedAstChart.data.datasets = [];
            sharedAstChart.update();
            
            sharedDetailedResultsBody.innerHTML = `<tr><td colspan="6" style="text-align:center;">Waiting for first validation results...</td></tr>`;
            currentSharedResults = [];
        } else {
            tpsChart.data.labels = [];
            tpsChart.data.datasets[0].data = [];
            tpsChart.update();
            
            ttftChart.data.labels = [];
            ttftChart.data.datasets[0].data = [];
            ttftChart.update();
            
            categoryChart.data.datasets = [];
            categoryChart.update();
            
            detailedResultsBody.innerHTML = `<tr><td colspan="6" style="text-align:center;">Waiting for first test results...</td></tr>`;
            currentResults = [];
        }
    });

    socket.on('model_start', (data) => {
        statusModel.textContent = data.model;
    });

    socket.on('test_start', (data) => {
        statusTest.textContent = data.test_label;
        statusCategory.textContent = data.category;
    });

    socket.on('benchmark_step', (data) => {
        // Multi-step agentic workflows: one line per conversation turn.
        logToTerminal(data.message || 'next turn…', "info", "shared");
        if (typeof showToast === 'function') {
            showToast(data.message || 'Next turn…', 'info', 1500);
        }
    });

    socket.on('test_complete', (data) => {
        const res = data.result;
        const pct = data.progress.percentage;
        progressPercent.textContent = `${pct}%`;
        progressBarFill.style.width = `${pct}%`;

        // Live failure display: surface any failing test immediately in the terminal.
        if (res && res.success === false) {
            const why = res.error || 'incorrect result';
            logToTerminal(`FAIL  ${data.model} / [${data.category}] ${data.test_id}: ${why}`, "error", "general");
            if (typeof showToast === 'function') {
                showToast(`Failed: ${data.test_id}`, 'error', 1500);
            }
        }
    });

    socket.on('model_complete', (data) => {
        const isShared = !data.results.category_coding; // SharedLLM has a task list instead of category mappings
        const term = isShared ? 'shared' : 'general';
        logToTerminal(`Model ${data.model} benchmarking complete!`, "success", term);
        
        if (isShared) {
            const idx = currentSharedResults.findIndex(r => r.model === data.model);
            if (idx !== -1) currentSharedResults[idx] = data.results;
            else currentSharedResults.push(data.results);
            
            renderSharedChartsFromData(currentSharedResults);
            renderSharedDetailsSection(currentSharedResults);
            updateSharedOverviewMetrics({ results: currentSharedResults });
        } else {
            const idx = currentResults.findIndex(r => r.model === data.model);
            if (idx !== -1) currentResults[idx] = data.results;
            else currentResults.push(data.results);
            
            renderChartsFromData(currentResults);
            renderDetailsSection(currentResults);
            updateOverviewMetrics({ results: currentResults });
        }
    });

    socket.on('benchmark_complete', (data) => {
        const isShared = data.benchmark_version.startsWith('SharedLLM');
        const term = isShared ? 'shared' : 'general';
        
        if (data.status === 'cancelled') {
            logToTerminal(`Benchmark pipeline cancelled. Completed results saved.`, "warn", term);
        } else {
            logToTerminal(`Benchmark complete! Final report written to ${data.saved_as}`, "success", term);
        }
        
        setRunnerState('idle');
        loadHistory();
        if (typeof loadTestBrowser === 'function') loadTestBrowser();
        // Run-flow clarity: surface results from any tab (audit: results materialize
        // only in General/Shared while the user may be anywhere).
        showToast(
            data.status === 'cancelled' ? 'Benchmark cancelled — partial results saved.' : 'Benchmark complete — results saved.',
            data.status === 'cancelled' ? 'info' : 'success',
            { actionLabel: 'View Results', onAction: () => switchTab(isShared ? 'shared' : 'general'), duration: 8000 }
        );
    });

    socket.on('benchmark_cancelled', (data) => {
        logToTerminal(`Benchmark run cancelled: ${data.message}`, "warn");
        logToTerminal(`Benchmark run cancelled: ${data.message}`, "warn", "shared");
        setRunnerState('idle');
    });

    socket.on('benchmark_error', (data) => {
        logToTerminal(`Critical runner error: ${data.error}`, "error");
        logToTerminal(`Critical runner error: ${data.error}`, "error", "shared");
        showToast(`Runner Error: ${data.error}`, 'error');
        setRunnerState('idle');
    });

    // Profile presets variables
    let modelProfiles = {};

    function loadModelProfiles() {
        const select = document.getElementById('profile-section-select');
        const cardsGrid = document.getElementById('profile-cards-grid');
        if (!select) return;
        select.innerHTML = '<option value="">Loading profiles...</option>';
        if (cardsGrid) cardsGrid.innerHTML = '<div style="color:var(--text-muted);font-size:0.8rem;padding:0.5rem;">Loading profiles...</div>';

        fetch('/api/profiles')
            .then(res => res.json())
            .then(data => {
                if (data.error) {
                    logToTerminal("Error loading profiles: " + data.error, "error");
                    select.innerHTML = '<option value="">Error loading profiles</option>';
                    if (cardsGrid) cardsGrid.innerHTML = `<div style="color:var(--color-danger);font-size:0.8rem;padding:0.5rem;">Error: ${data.error}</div>`;
                    return;
                }
                modelProfiles = data.profiles || {};
                select.innerHTML = '';

                const placeholder = document.createElement('option');
                placeholder.value = '';
                placeholder.textContent = '-- Select a profile --';
                select.appendChild(placeholder);

                Object.keys(modelProfiles).forEach(section => {
                    const opt = document.createElement('option');
                    opt.value = section;
                    opt.textContent = section === '*' ? '[*] Defaults' : section;
                    select.appendChild(opt);
                });

                // Render profile cards
                if (cardsGrid) {
                    cardsGrid.innerHTML = '';
                    const sections = Object.keys(modelProfiles);
                    if (sections.length === 0) {
                        cardsGrid.innerHTML = '<div style="color:var(--text-muted);font-size:0.8rem;padding:0.5rem;">No profiles found in models.ini</div>';
                        return;
                    }
                    sections.forEach(section => {
                        const s = modelProfiles[section];
                        const isDefault = section === '*';
                        const label = isDefault ? '[*] Global Defaults' : section.replace(/--/g, ' / ');

                        const specBadge = s['spec-type'] && s['spec-type'] !== 'none'
                            ? `<span style="background:rgba(139,92,246,0.25);color:#a78bfa;padding:1px 6px;border-radius:4px;font-size:0.65rem;">${s['spec-type']}</span>` : '';
                        const flashBadge = s['flash-attn'] === 'on' || s['flash-attn'] === 'true'
                            ? `<span style="background:rgba(34,211,238,0.2);color:#22d3ee;padding:1px 6px;border-radius:4px;font-size:0.65rem;">FA✓</span>` : '';
                        
                        // Backend classification badge
                        const isSd = s['backend'] === 'stable-diffusion';
                        const backendBadge = isSd
                            ? `<span style="background:rgba(236,72,153,0.2);color:#f472b6;padding:1px 6px;border-radius:4px;font-size:0.65rem;font-weight:600;">🎨 Image SD</span>`
                            : `<span style="background:rgba(59,130,246,0.2);color:#60a5fa;padding:1px 6px;border-radius:4px;font-size:0.65rem;font-weight:600;">💬 Text LLM</span>`;

                        const card = document.createElement('div');
                        card.style.cssText = `background:var(--card-bg);border:1px solid var(--border-color);border-radius:8px;padding:0.85rem;cursor:pointer;transition:border-color 0.2s,box-shadow 0.2s;`;
                        card.innerHTML = `
                            <div style="font-size:0.72rem;font-weight:600;color:${isDefault ? '#f59e0b' : 'var(--color-primary)'};margin-bottom:0.5rem;word-break:break-all;line-height:1.3;">${label}</div>
                            <div style="display:flex;gap:4px;flex-wrap:wrap;margin-bottom:0.5rem;">${backendBadge}${specBadge}${flashBadge}</div>
                            <div style="display:grid;grid-template-columns:1fr 1fr;gap:2px 8px;font-size:0.68rem;">
                                ${s['ctx-size'] ? `<span style="color:var(--text-muted);">CTX</span><span style="color:white;">${Number(s['ctx-size']).toLocaleString()}</span>` : ''}
                                ${s['n-gpu-layers'] ? `<span style="color:var(--text-muted);">GPU Layers</span><span style="color:white;">${s['n-gpu-layers']}</span>` : ''}
                                ${s['cache-type-k'] ? `<span style="color:var(--text-muted);">KV-K</span><span style="color:white;">${s['cache-type-k']}</span>` : ''}
                                ${s['cache-type-v'] ? `<span style="color:var(--text-muted);">KV-V</span><span style="color:white;">${s['cache-type-v']}</span>` : ''}
                                ${s['n-cpu-moe'] ? `<span style="color:var(--text-muted);">MoE CPU</span><span style="color:white;">${s['n-cpu-moe']}</span>` : ''}
                            </div>`;
                        card.addEventListener('mouseenter', () => { card.style.borderColor = 'var(--color-primary)'; card.style.boxShadow = '0 0 0 1px var(--color-primary)33'; });
                        card.addEventListener('mouseleave', () => { card.style.borderColor = 'var(--border-color)'; card.style.boxShadow = 'none'; });
                        card.addEventListener('click', () => {
                            select.value = section;
                            select.dispatchEvent(new Event('change'));
                            // Scroll to editor
                            const form = document.getElementById('profile-edit-form');
                            if (form) form.scrollIntoView({ behavior: 'smooth', block: 'start' });
                        });
                        cardsGrid.appendChild(card);
                    });
                }
            })
            .catch(err => {
                console.error("Fetch profiles error:", err);
                select.innerHTML = '<option value="">Failed to connect to backend</option>';
                if (cardsGrid) cardsGrid.innerHTML = '<div style="color:var(--color-danger);font-size:0.8rem;padding:0.5rem;">Failed to connect to backend</div>';
            });
    }

    const profileSectionSelect = document.getElementById('profile-section-select');
    const profileEditForm = document.getElementById('profile-edit-form');
    const btnRestartServices = document.getElementById('btn-restart-services');
    let cachedCompanions = null;

    if (profileSectionSelect && profileEditForm) {
        profileSectionSelect.addEventListener('change', () => {
            const section = profileSectionSelect.value;
            const badgeEl = document.getElementById('profile-backend-badge');
            if (!section) {
                profileEditForm.reset();
                if (badgeEl) badgeEl.style.display = 'none';
                return;
            }
            
            const settings = modelProfiles[section] || {};

            // Dynamic warning and information badge based on backend type
            if (badgeEl) {
                if (settings['backend'] === 'stable-diffusion') {
                    badgeEl.style.display = 'block';
                    badgeEl.innerHTML = `
                        <div style="background:rgba(236,72,153,0.1); border:1px solid rgba(236,72,153,0.25); color:#f472b6; padding:0.6rem; border-radius:6px; font-size:0.75rem; line-height:1.4;">
                            <strong>🎨 Stable Diffusion Profile</strong><br>
                            Only <strong>GPU Layers</strong> (offloaded transformer layers) and <strong>CPU Threads</strong> (under MoE threads) apply to this model backend. Other settings are ignored.
                        </div>
                    `;
                } else if (section === '*') {
                    badgeEl.style.display = 'block';
                    badgeEl.innerHTML = `
                        <div style="background:rgba(245,158,11,0.1); border:1px solid rgba(245,158,11,0.25); color:#fbbf24; padding:0.6rem; border-radius:6px; font-size:0.75rem; line-height:1.4;">
                            <strong>⚙️ Global Default Presets</strong><br>
                            These parameters apply to all llama.cpp models unless overridden in their specific model profiles.
                        </div>
                    `;
                } else {
                    badgeEl.style.display = 'block';
                    badgeEl.innerHTML = `
                        <div style="background:rgba(59,130,246,0.1); border:1px solid rgba(59,130,246,0.25); color:#60a5fa; padding:0.6rem; border-radius:6px; font-size:0.75rem; line-height:1.4;">
                            <strong>💬 llama.cpp Language Model Profile</strong><br>
                            All parameters below are configurable for the llama.cpp inference engine.
                        </div>
                    `;
                }
            }
            
            const globalDefaults = modelProfiles['*'] || {};
            const isSd = settings['backend'] === 'stable-diffusion';

            // Show the field group that matches the model backend.
            const llmFields = document.getElementById('llm-fields');
            const sdFields = document.getElementById('sd-fields');
            if (llmFields) llmFields.style.display = isSd ? 'none' : 'block';
            if (sdFields) sdFields.style.display = isSd ? 'block' : 'none';

            const setNumberInput = (name, key, fallbackDesc) => {
                const input = profileEditForm.elements[name];
                if (!input) return;
                const sectionVal = settings[key];
                const globalVal = globalDefaults[key];

                if (sectionVal !== undefined && sectionVal !== null && sectionVal !== '') {
                    input.value = sectionVal;
                    input.placeholder = '';
                } else if (globalVal !== undefined && globalVal !== null && globalVal !== '' && section !== '*') {
                    input.value = '';
                    input.placeholder = `${globalVal} (Inherited)`;
                } else {
                    input.value = '';
                    input.placeholder = fallbackDesc;
                }
            };

            const setSelectInput = (name, key, fallbackVal) => {
                const select = profileEditForm.elements[name];
                if (!select) return;
                const sectionVal = settings[key];
                const globalVal = globalDefaults[key];

                if (sectionVal !== undefined && sectionVal !== null && sectionVal !== '') {
                    select.value = sectionVal;
                } else if (globalVal !== undefined && globalVal !== null && globalVal !== '' && section !== '*') {
                    select.value = globalVal;
                } else {
                    select.value = fallbackVal;
                }
            };

            setNumberInput('ctx-size', 'ctx-size', '4096 (Default)');
            setNumberInput('n-gpu-layers', 'n-gpu-layers', isSd ? '40 (Default)' : '99 (Default)');
            
            setSelectInput('cache-type-k', 'cache-type-k', 'f16');
            setSelectInput('cache-type-v', 'cache-type-v', 'f16');
            setSelectInput('flash-attn', 'flash-attn', 'on');
            setSelectInput('kv-unified', 'kv-unified', 'true');
            setSelectInput('spec-type', 'spec-type', 'none');

            setNumberInput('spec-draft-n-max', 'spec-draft-n-max', '0 (Disabled)');
            setNumberInput('n-cpu-moe', 'n-cpu-moe', isSd ? 'Auto (nproc - 2)' : 'Auto');
            setNumberInput('temperature', 'temperature', '0.6 (Recommended)');

            // SD / image model fields
            setSelectInput('model_family', 'model_family', 'qwen-image');
            setNumberInput('gpu_layers', 'gpu_layers', '40 (Default)');
            const setTextInput = (name, key) => {
                const el = profileEditForm.elements[name];
                if (!el) return;
                const v = settings[key];
                el.value = (v !== undefined && v !== null) ? v : '';
            };
            setTextInput('extra_args', 'extra_args');
            setNumberInput('threads', 'threads', 'Auto (nproc - 2)');
            setSelectInput('cache-mode', 'cache-mode', '');
            setTextInput('cache-option', 'cache-option');

            // SD companion models come from a dropdown populated with the
            // companion files discovered on disk (VAE / LLM / CLIP / T5XXL).
            const populateCompanion = (name, key) => {
                const sel = profileEditForm.elements[name];
                if (!sel) return;
                const cur = settings[key] || '';
                let opts = '<option value="">&lt;none&gt;</option>';
                (cachedCompanions || []).forEach(c => {
                    const selAttr = (c === cur) ? ' selected' : '';
                    opts += `<option value="${c}"${selAttr}>${c}</option>`;
                });
                sel.innerHTML = opts;
            };
            if (isSd) {
                if (cachedCompanions === null) {
                    fetch('/api/companions')
                        .then(r => r.json())
                        .then(d => {
                            cachedCompanions = (d.companions || []);
                            populateCompanion('vae', 'vae');
                            populateCompanion('llm', 'llm');
                            populateCompanion('clip_l', 'clip_l');
                            populateCompanion('t5xxl', 't5xxl');
                        })
                        .catch(() => {});
                } else {
                    populateCompanion('vae', 'vae');
                    populateCompanion('llm', 'llm');
                    populateCompanion('clip_l', 'clip_l');
                    populateCompanion('t5xxl', 't5xxl');
                }
            }
        });

        // Build the settings payload from the visible field group, including the
        // model backend so the server knows where to persist (models.ini vs the
        // image-model .profile.json overlay).
        const buildProfileSettings = (section) => {
            const isSd = (modelProfiles[section] || {})['backend'] === 'stable-diffusion';
            let settings, backend = 'llama.cpp';
            if (isSd) {
                backend = 'stable-diffusion';
                const get = (name) => {
                    const el = profileEditForm.elements[name];
                    return el ? (el.value || null) : null;
                };
                settings = {
                    'model_family': get('model_family'),
                    'gpu_layers': get('gpu_layers'),
                    'vae': get('vae'),
                    'llm': get('llm'),
                    'clip_l': get('clip_l'),
                    't5xxl': get('t5xxl'),
                    'extra_args': get('extra_args'),
                    'threads': get('threads'),
                    'cache-mode': get('cache-mode'),
                    'cache-option': get('cache-option')
                };
            } else {
                settings = {
                    'ctx-size': profileEditForm.elements['ctx-size'].value || null,
                    'n-gpu-layers': profileEditForm.elements['n-gpu-layers'].value || null,
                    'cache-type-k': profileEditForm.elements['cache-type-k'].value,
                    'cache-type-v': profileEditForm.elements['cache-type-v'].value,
                    'flash-attn': profileEditForm.elements['flash-attn'].value,
                    'kv-unified': profileEditForm.elements['kv-unified'].value,
                    'spec-type': profileEditForm.elements['spec-type'].value,
                    'spec-draft-n-max': profileEditForm.elements['spec-draft-n-max'].value || null,
                    'n-cpu-moe': profileEditForm.elements['n-cpu-moe'].value || null,
                    'temperature': profileEditForm.elements['temperature'].value || null
                };
            }
            Object.keys(settings).forEach(key => {
                if (settings[key] === null || settings[key] === '') {
                    delete settings[key];
                }
            });
            return { settings, backend };
        };

        profileEditForm.addEventListener('submit', (e) => {
            e.preventDefault();
            const section = profileSectionSelect.value;
            if (!section) {
                showToast('Please select a model profile section to save.', 'error');
                return;
            }

            const { settings, backend } = buildProfileSettings(section);

            fetch('/api/profiles/save', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ section, settings, backend })
            })
            .then(res => res.json())
            .then(data => {
                if (data.error) {
                    showToast('Failed to save settings: ' + data.error, 'error');
                } else {
                    modelProfiles[section] = Object.assign({}, modelProfiles[section], settings);
                    showToast('Settings saved successfully!', 'success');
                }
            })
            .catch(err => {
                console.error("Save profile error:", err);
                showToast('Failed to save profile settings', 'error');
            });
        });

        if (btnRestartServices) {
            btnRestartServices.addEventListener('click', () => {
                const section = profileSectionSelect.value;
                if (!section) {
                    showToast('Please select a model profile section first.', 'error');
                    return;
                }
                
                const confirmed = confirm('Are you sure you want to save the settings and restart the backend services (llama-server and alpaca-proxy)? This will temporarily interrupt any active inference sessions.');
                if (!confirmed) return;
                
                // Construct settings from form fields
                const { settings, backend } = buildProfileSettings(section);
                
                btnRestartServices.disabled = true;
                btnRestartServices.textContent = '🔄 Restarting Backend...';
                
                fetch('/api/profiles/save', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ section, settings, backend })
                })
                .then(res => res.json())
                .then(data => {
                    if (data.error) {
                        throw new Error(data.error);
                    }
                    modelProfiles[section] = Object.assign({}, modelProfiles[section], settings);
                    
                    // Trigger container restart
                    return fetch('/api/proxy/restart', { method: 'POST' });
                })
                .then(res => res.json())
                .then(data => {
                    if (data.error) {
                        showToast('Restart command failed: ' + data.error, 'error');
                    } else {
                        showToast('Backend restart sequence initiated. Reloading system monitor in 5 seconds...', 'success');
                        setTimeout(() => {
                            window.location.reload();
                        }, 5000);
                    }
                })
                .catch(err => {
                    console.error("Save and restart error:", err);
                    showToast('Failed to save settings or restart backend: ' + err.message, 'error');
                })
                .finally(() => {
                    btnRestartServices.disabled = false;
                    btnRestartServices.textContent = '🔄 Save & Restart Backend';
                });
            });
        }
    }

    // Wire Create Profile
    const btnCreateProfile = document.getElementById('btn-create-profile');
    if (btnCreateProfile) {
        btnCreateProfile.addEventListener('click', () => {
            const name = prompt('Enter a name for the new profile section:');
            if (!name) return;
            const sanitized = name.trim().replace(/[^A-Za-z0-9._-]/g, '-');
            if (!sanitized) {
                showToast('Invalid profile name.', 'error');
                return;
            }
            
            if (modelProfiles[sanitized]) {
                showToast(`Profile section [${sanitized}] already exists.`, 'error');
                profileSectionSelect.value = sanitized;
                profileSectionSelect.dispatchEvent(new Event('change'));
                return;
            }
            
            fetch('/api/profiles/save', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ section: sanitized, settings: {} })
            })
            .then(res => res.json())
            .then(data => {
                if (data.error) {
                    showToast('Failed to create profile: ' + data.error, 'error');
                } else {
                    modelProfiles[sanitized] = {};
                    
                    const select = document.getElementById('profile-section-select');
                    const opt = document.createElement('option');
                    opt.value = sanitized;
                    opt.textContent = sanitized;
                    select.appendChild(opt);
                    select.value = sanitized;
                    
                    profileSectionSelect.dispatchEvent(new Event('change'));
                    showToast(`Profile section [${sanitized}] created successfully!`, 'success');
                }
            })
            .catch(err => {
                console.error("Create profile error:", err);
                showToast('Failed to create profile section', 'error');
            });
        });
    }

    // Wire Delete Profile
    const btnDeleteProfile = document.getElementById('btn-delete-profile');
    if (btnDeleteProfile) {
        btnDeleteProfile.addEventListener('click', () => {
            const section = profileSectionSelect.value;
            if (!section) {
                showToast('Please select a profile section to delete.', 'error');
                return;
            }
            if (section === '*') {
                showToast('Cannot delete global defaults section [*].', 'error');
                return;
            }
            if (!confirm(`Are you sure you want to delete profile section [${section}]? This cannot be undone.`)) {
                return;
            }
            
            fetch('/api/profiles/delete', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ section })
            })
            .then(res => res.json())
            .then(data => {
                if (data.error) {
                    showToast('Failed to delete profile: ' + data.error, 'error');
                } else {
                    delete modelProfiles[section];
                    
                    const option = profileSectionSelect.querySelector(`option[value="${section}"]`);
                    if (option) option.remove();
                    profileSectionSelect.value = '';
                    profileSectionSelect.dispatchEvent(new Event('change'));
                    
                    showToast(`Profile section [${section}] deleted successfully!`, 'success');
                }
            })
            .catch(err => {
                console.error("Delete profile error:", err);
                showToast('Failed to delete profile section', 'error');
            });
        });
    }

    if (btnRestartServices) {
        btnRestartServices.addEventListener('click', () => {
            if (!confirm('Are you sure you want to save current settings and restart backend services? Active requests will be interrupted.')) {
                return;
            }
            
            const section = profileSectionSelect ? profileSectionSelect.value : null;
            if (section && profileEditForm) {
                const settings = {
                    'ctx-size': profileEditForm.elements['ctx-size'].value || null,
                    'n-gpu-layers': profileEditForm.elements['n-gpu-layers'].value || null,
                    'cache-type-k': profileEditForm.elements['cache-type-k'].value,
                    'cache-type-v': profileEditForm.elements['cache-type-v'].value,
                    'flash-attn': profileEditForm.elements['flash-attn'].value,
                    'kv-unified': profileEditForm.elements['kv-unified'].value,
                    'spec-type': profileEditForm.elements['spec-type'].value,
                    'spec-draft-n-max': profileEditForm.elements['spec-draft-n-max'].value || null,
                    'n-cpu-moe': profileEditForm.elements['n-cpu-moe'].value || null,
                    'temperature': profileEditForm.elements['temperature'].value || null
                };
                
                Object.keys(settings).forEach(key => {
                    if (settings[key] === null || settings[key] === '') {
                        delete settings[key];
                    }
                });
                
                fetch('/api/profiles/save', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ section, settings })
                })
                .then(res => res.json())
                .then(data => {
                    if (data.error) {
                        console.error("Save profile failed before restart:", data.error);
                    }
                    triggerRestart();
                })
                .catch(err => {
                    console.error("Save profile failed before restart:", err);
                    triggerRestart();
                });
            } else {
                triggerRestart();
            }
        });
    }

    function triggerRestart() {
        const overlay = document.getElementById('restart-overlay');
        const s1 = document.getElementById('restart-status-1');
        const s2 = document.getElementById('restart-status-2');
        const s3 = document.getElementById('restart-status-3');

        if (overlay) {
            s1.textContent = '⌛';
            s2.textContent = '⌛';
            s3.textContent = '⌛';
            overlay.classList.add('open');
        }

        fetch('/api/proxy/restart', { method: 'POST' })
            .then(res => res.json())
            .then(data => {
                if (s1) s1.textContent = '✅';
                startRestartPolling();
            })
            .catch(err => {
                console.error("Restart error:", err);
                if (s1) s1.textContent = '✅';
                startRestartPolling();
            });
    }

    function startRestartPolling() {
        const s2 = document.getElementById('restart-status-2');
        const s3 = document.getElementById('restart-status-3');
        const overlay = document.getElementById('restart-overlay');
        
        let wentOffline = false;
        let checks = 0;
        
        stopMonitorPolling();
        
        const interval = setInterval(async () => {
            checks++;
            try {
                const res = await fetch('/api/proxy/status');
                const data = await res.json();
                
                if (!data.online) {
                    wentOffline = true;
                    if (s2) s2.textContent = '✅';
                } else if (wentOffline) {
                    if (s3) s3.textContent = '✅';
                    clearInterval(interval);
                    setTimeout(() => {
                        if (overlay) overlay.classList.remove('open');
                        loadModelProfiles();
                        startMonitorPolling();
                    }, 1000);
                } else if (checks > 8) {
                    wentOffline = true;
                    if (s2) s2.textContent = '✅';
                }
            } catch (err) {
                wentOffline = true;
                if (s2) s2.textContent = '✅';
            }
            
            if (checks > 45) {
                clearInterval(interval);
                if (overlay) overlay.classList.remove('open');
                showToast('Restart sequence timed out or connection lost. Please refresh the page manually.', 'error');
                startMonitorPolling();
            }
        }, 2000);
    }

    // Model Switcher Functions
    let currentModelName = null;
    
    async function updateCurrentModel() {
        try {
            const res = await fetch('/api/proxy/status');
            const data = await res.json();
            const loaded = (data.runtime && data.runtime.loaded_models) || [];
            const loading = (data.runtime && data.runtime.loading_models) || [];
            
            if (data.online && loaded.length > 0) {
                currentModelName = loaded[0].name;
                let statusHTML = `<span style="color:var(--color-success);">Currently loaded: <strong>${currentModelName}</strong></span>`;
                if (loading.length > 0) {
                    statusHTML += ` <span style="color:var(--color-secondary); font-size:0.85rem; margin-left:0.5rem; animation: pulse 1.5s infinite;">(Switching to: ${loading[0].name}...)</span>`;
                }
                if (modelSwitcherStatus) {
                    modelSwitcherStatus.innerHTML = statusHTML;
                }
            } else if (data.online && loading.length > 0) {
                currentModelName = null;
                if (modelSwitcherStatus) {
                    modelSwitcherStatus.innerHTML = `<span style="color:var(--color-secondary); animation: pulse 1.5s infinite;">Attempting to load: <strong>${loading[0].name}</strong> (${loading[0].elapsed_seconds}s)</span>`;
                }
            } else {
                currentModelName = null;
                if (modelSwitcherStatus) {
                    modelSwitcherStatus.innerHTML = `<span style="color:var(--text-muted);">No model currently loaded</span>`;
                }
            }
        } catch (err) {
            console.error("Failed to get current model:", err);
        }
    }
    
    async function switchToModel(modelName) {
        if (!modelName) {
            if (modelSwitcherStatus) {
                modelSwitcherStatus.innerHTML = `<span style="color:var(--color-danger);">No model selected</span>`;
            }
            return;
        }
        
        if (modelSwitcherStatus) {
            modelSwitcherStatus.innerHTML = `<span style="color:var(--color-secondary);">Loading ${modelName}...</span>`;
        }
        
        try {
            const res = await fetch('/api/models/switch', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({model: modelName})
            });
            
            const data = await res.json();
            
            if (res.ok) {
                if (modelSwitcherStatus) {
                    modelSwitcherStatus.innerHTML = `<span style="color:var(--color-success);">✅ ${data.status}: ${modelName}</span>`;
                }
                await updateCurrentModel();
                logToTerminal(`Model switched to: ${modelName}`, 'success');
            } else {
                if (modelSwitcherStatus) {
                    modelSwitcherStatus.innerHTML = `<span style="color:var(--color-danger);">❌ ${escapeHtml(data.error || 'Failed to switch model')}</span>`;
                }
                logToTerminal(`Model switch failed: ${data.error}`, 'error');
            }
        } catch (err) {
            if (modelSwitcherStatus) {
                modelSwitcherStatus.innerHTML = `<span style="color:var(--color-danger);">❌ ${escapeHtml(err.message)}</span>`;
            }
            logToTerminal(`Model switch error: ${err.message}`, 'error');
        }
    }
    
    async function unloadCurrentModel() {
        if (!currentModelName) {
            if (modelSwitcherStatus) {
                modelSwitcherStatus.innerHTML = `<span style="color:var(--color-warning);">No model currently loaded</span>`;
            }
            return;
        }
        
        if (!confirm(`Unload model "${currentModelName}"?`)) {
            return;
        }
        
        if (modelSwitcherStatus) {
            modelSwitcherStatus.innerHTML = `<span style="color:var(--color-secondary);">Unloading ${escapeHtml(currentModelName)}...</span>`;
        }
        
        try {
            const res = await fetch('/api/models/unload', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({model: currentModelName})
            });
            
            const data = await res.json();
            
            if (res.ok) {
                const unloadedModel = currentModelName;
                if (modelSwitcherStatus) {
                    modelSwitcherStatus.innerHTML = `<span style="color:var(--color-success);">✅ Unloaded: ${unloadedModel}</span>`;
                }
                currentModelName = null;
                logToTerminal(`Model unloaded: ${unloadedModel}`, 'success');
            } else {
                if (modelSwitcherStatus) {
                    modelSwitcherStatus.innerHTML = `<span style="color:var(--color-danger);">❌ ${data.error || 'Failed to unload model'}</span>`;
                }
            }
        } catch (err) {
            if (modelSwitcherStatus) {
                modelSwitcherStatus.innerHTML = `<span style="color:var(--color-danger);">❌ ${err.message}</span>`;
            }
            logToTerminal(`Unload error: ${err.message}`, 'error');
        }
    }
    
    async function clearVram() {
        if (!confirm("Are you sure you want to FORCE clear VRAM? This will unload all active models and restart the llama-server.")) {
            return;
        }
        
        if (modelSwitcherStatus) {
            modelSwitcherStatus.innerHTML = `<span style="color:var(--color-secondary);">Clearing VRAM & restarting llama-server...</span>`;
        }
        
        try {
            const res = await fetch('/api/vram/clear', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'}
            });
            
            const data = await res.json();
            
            if (res.ok) {
                if (modelSwitcherStatus) {
                    modelSwitcherStatus.innerHTML = `<span style="color:var(--color-success);">✅ VRAM Cleared successfully</span>`;
                }
                currentModelName = null;
                logToTerminal(`VRAM Cleared successfully: ${data.message}`, 'success');
            } else {
                if (modelSwitcherStatus) {
                    modelSwitcherStatus.innerHTML = `<span style="color:var(--color-danger);">❌ ${data.error || 'Failed to clear VRAM'}</span>`;
                }
                logToTerminal(`VRAM Clear failure: ${data.error || 'Unknown error'}`, 'error');
            }
        } catch (err) {
            if (modelSwitcherStatus) {
                modelSwitcherStatus.innerHTML = `<span style="color:var(--color-danger);">❌ ${err.message}</span>`;
            }
            logToTerminal(`VRAM Clear error: ${err.message}`, 'error');
        }
    }

    async function deleteModel(modelName) {
        if (!modelName) {
            showToast("Please select a model to delete.", 'error');
            return;
        }
        if (!confirm(`Are you sure you want to permanently delete model "${modelName}"?\nThis will remove the manifest and all unshared blobs from disk. This action cannot be undone!`)) {
            return;
        }

        const hasBenchmarks =
            (Array.isArray(currentResults) && currentResults.some(r => r.model === modelName)) ||
            (Array.isArray(currentSharedResults) && currentSharedResults.some(r => r.model === modelName));
        let removeBenchmarks = false;
        if (hasBenchmarks) {
            removeBenchmarks = confirm(`Model "${modelName}" has saved benchmark results.\n\nOK = also delete its benchmark results\nCancel = delete the model but keep its benchmark history`);
        }

        if (modelSwitcherStatus) {
            modelSwitcherStatus.innerHTML = `<span style="color:var(--color-secondary);">Deleting ${modelName}...</span>`;
        }

        try {
            const res = await fetch('/api/models/delete', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ model: modelName, remove_benchmarks: removeBenchmarks })
            });

            const data = await res.json();

            if (res.ok) {
                if (modelSwitcherStatus) {
                    modelSwitcherStatus.innerHTML = `<span style="color:var(--color-success);">✅ Model "${modelName}" deleted successfully</span>`;
                }
                logToTerminal(`Model "${modelName}" deleted successfully.`, 'success');
                showToast(`Model "${modelName}" deleted successfully.`, 'success');

                if (currentModelName === modelName) {
                    currentModelName = null;
                }

                await loadModels();
                
                if (removeBenchmarks) {
                    if (typeof loadHistory === 'function') {
                        await loadHistory();
                    }
                }
                
                if (typeof loadRoutingMatrix === 'function') {
                    loadRoutingMatrix();
                }
            } else {
                if (modelSwitcherStatus) {
                    modelSwitcherStatus.innerHTML = `<span style="color:var(--color-danger);">❌ ${data.error || 'Failed to delete model'}</span>`;
                }
                logToTerminal(`Model delete failure: ${data.error || 'Unknown error'}`, 'error');
                showToast(`Failed to delete model: ${data.error || 'Unknown error'}`, 'error');
            }
        } catch (err) {
            if (modelSwitcherStatus) {
                modelSwitcherStatus.innerHTML = `<span style="color:var(--color-danger);">❌ ${err.message}</span>`;
            }
            logToTerminal(`Model delete error: ${err.message}`, 'error');
            showToast(`Model delete error: ${err.message}`, 'error');
        }
    }

    /**
     * Bulk-remove all saved benchmark results for a model that is NOT being
     * deleted from disk. The model itself stays (manifest, blobs, router link);
     * only its benchmark history (per-model result files + merged latest
     * snapshot entry + tracker record) is purged, so it reverts to "new" in
     * the tracking UI and can be re-benchmarked later.
     */
    async function deleteModelBenchmarks(modelId, displayName) {
        if (!modelId) {
            showToast("Please select a model to clear.", 'error');
            return;
        }
        if (!confirm(`Delete all saved benchmark results for "${displayName}"?\n\nThe model itself is kept on disk — only its benchmark history is removed. This action cannot be undone.`)) {
            return;
        }

        try {
            const res = await fetch(`/api/benchmarks/model/${encodeURIComponent(modelId)}`, {
                method: 'DELETE'
            });
            const data = await res.json();

            if (res.ok) {
                logToTerminal(`Benchmark history cleared for "${displayName}" (${data.removed || 0} file(s) removed).`, 'success');
                showToast(`Benchmark history cleared for "${displayName}"`, 'success');
                await loadModels();
                if (typeof loadHistory === 'function') {
                    await loadHistory();
                }
                if (typeof loadRoutingMatrix === 'function') {
                    await loadRoutingMatrix();
                }
            } else {
                logToTerminal(`Failed to clear benchmarks for "${displayName}": ${data.error || 'Unknown error'}`, 'error');
                showToast(`Failed to clear benchmarks: ${data.error || 'Unknown error'}`, 'error');
            }
        } catch (err) {
            logToTerminal(`Benchmark clear error: ${err.message}`, 'error');
            showToast(`Benchmark clear error: ${err.message}`, 'error');
        }
    }

    // TELEMETRY AND AUTO-TUNING INTEGRATION
    async function updateTelemetryAndRecommendations(modelName) {
        if (!modelName || modelName === 'None') {
            const modelBadge = document.getElementById('optimization-model-badge');
            if (modelBadge) modelBadge.textContent = 'None';
            const suggestionsEl = document.getElementById('optimization-suggestions');
            if (suggestionsEl) suggestionsEl.textContent = 'No active model running. Load a model to inspect telemetry recommendations.';
            const applyBtn = document.getElementById('btn-apply-optimizations');
            if (applyBtn) applyBtn.classList.add('d-none');
            const badge = document.getElementById('optimization-status-badge');
            if (badge) {
                badge.className = 'badge badge-secondary';
                badge.textContent = 'Idle';
            }
            
            if (memoryCreepChart) {
                memoryCreepChart.data.labels = [];
                memoryCreepChart.data.datasets[0].data = [];
                memoryCreepChart.data.datasets[1].data = [];
                memoryCreepChart.update();
            }
            return;
        }
        
        try {
            const strategy = document.getElementById('tuning-strategy-select')?.value || 'performance';
            
            // 1. Fetch History
            const histRes = await fetch(`/api/telemetry/history?model=${encodeURIComponent(modelName)}&limit=50`);
            const histData = await histRes.json();
            
            if (histData.history && histData.history.length > 0) {
                const timestamps = histData.history.map(p => {
                    const t = new Date(p.timestamp);
                    return t.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
                });
                const ramData = histData.history.map(p => p.system.ram_used_pct);
                const vramData = histData.history.map(p => {
                    const gpus = p.gpus || [];
                    return gpus.length > 0 ? gpus[0].vram_used_pct : 0;
                });
                
                if (memoryCreepChart) {
                    memoryCreepChart.data.labels = timestamps;
                    memoryCreepChart.data.datasets[0].data = ramData;
                    memoryCreepChart.data.datasets[1].data = vramData;
                    memoryCreepChart.update();
                }
            } else {
                if (memoryCreepChart) {
                    memoryCreepChart.data.labels = [];
                    memoryCreepChart.data.datasets[0].data = [];
                    memoryCreepChart.data.datasets[1].data = [];
                    memoryCreepChart.update();
                }
            }
            
            // 2. Fetch Recommendations
            const recRes = await fetch(`/api/telemetry/recommendations?model=${encodeURIComponent(modelName)}&strategy=${strategy}`);
            const recData = await recRes.json();
            
            const modelBadge = document.getElementById('optimization-model-badge');
            const statusBadge = document.getElementById('optimization-status-badge');
            const suggestionsEl = document.getElementById('optimization-suggestions');
            const applyBtn = document.getElementById('btn-apply-optimizations');
            
            if (modelBadge) modelBadge.textContent = truncateModelName(modelName);
            
            if (recData.status === 'insufficient_data') {
                if (statusBadge) {
                    statusBadge.className = 'badge badge-warning';
                    statusBadge.textContent = 'Collecting Data';
                }
                if (suggestionsEl) suggestionsEl.textContent = recData.explanation || 'Collecting telemetry... waiting for more data points.';
                if (applyBtn) applyBtn.classList.add('d-none');
                currentRecommendations = null;
            } else {
                currentRecommendations = recData.recommendations || {};
                
                // Status badge
                if (statusBadge) {
                    if (recData.status === 'critical') {
                        statusBadge.className = 'badge badge-danger';
                        statusBadge.textContent = 'Critical (OOM Risk)';
                    } else if (recData.status === 'warning') {
                        statusBadge.className = 'badge badge-warning';
                        statusBadge.textContent = 'Warning (High Usage)';
                    } else {
                        statusBadge.className = 'badge badge-success';
                        statusBadge.textContent = 'Optimal';
                    }
                }
                
                // Suggestions description
                if (suggestionsEl) {
                    if (recData.detected_issues && recData.detected_issues.length > 0 && Object.keys(currentRecommendations).length > 0) {
                        let html = `<strong style="color:white; display:block; margin-bottom:0.25rem;">Detected Issues:</strong>`;
                        html += `<ul style="margin: 0 0 0.5rem 1rem; padding: 0;">`;
                        recData.detected_issues.forEach(issue => {
                            html += `<li>${issue}</li>`;
                        });
                        html += `</ul>`;
                        html += `<strong style="color:white; display:block; margin-bottom:0.25rem;">Actions:</strong><br>`;
                        html += recData.explanation;
                        suggestionsEl.innerHTML = html;
                    } else {
                        suggestionsEl.textContent = recData.explanation || 'System resource usage is within safe operating margins.';
                    }
                }
                
                // Show/hide apply button
                if (applyBtn) {
                    if (Object.keys(currentRecommendations).length > 0) {
                        applyBtn.classList.remove('d-none');
                    } else {
                        applyBtn.classList.add('d-none');
                    }
                }
            }
        } catch (err) {
            console.error("Error updating telemetry/recommendations:", err);
        }
    }

    async function applyTuningOptimizations() {
        const activeModelName = currentModelName;
        if (!activeModelName) {
            showToast("No active model loaded.", "error");
            return;
        }
        
        if (!currentRecommendations || Object.keys(currentRecommendations).length === 0) {
            showToast("No recommendations to apply.", "warning");
            return;
        }
        
        try {
            const res = await fetch('/api/telemetry/recommendations/apply', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    model: activeModelName,
                    recommendations: currentRecommendations
                })
            });
            const data = await res.json();
            if (res.ok) {
                showToast(data.message || "Tuning configurations applied successfully!", "success");
                updateTelemetryAndRecommendations(activeModelName);
            } else {
                showToast(data.error || "Failed to apply configurations.", "error");
            }
        } catch (err) {
            console.error("Error applying tuning configurations:", err);
            showToast("Failed to apply configurations.", "error");
        }
    }

    // CAPABILITY ROUTING MATRIX INTEGRATION
    async function loadRoutingMatrix() {
        try {
            // 1. Fetch available models
            const modelsRes = await fetch('/api/models');
            const modelsData = await modelsRes.json();
            availableModels = modelsData.models || [];
            
            // 2. Fetch routing matrix mappings
            const matrixRes = await fetch('/api/routing/matrix');
            const matrixData = await matrixRes.json();
            
            renderRoutingMatrix(matrixData);
        } catch (err) {
            console.error("Error loading routing matrix:", err);
        }
    }
    
    function renderRoutingMatrix(matrix) {
        const tbody = document.getElementById('routing-matrix-body');
        if (!tbody) return;
        tbody.innerHTML = '';

        const entries = Object.entries(matrix || {});
        if (entries.length === 0) {
            const tr = document.createElement('tr');
            tr.innerHTML = `<td colspan="7" style="text-align:center; color:var(--text-muted); font-size:0.8rem; padding:1.5rem;">No routing matrix configured. Load models and save to populate.</td>`;
            tbody.appendChild(tr);
            return;
        }
        
        entries.forEach(([taskKey, config]) => {
            const tr = document.createElement('tr');
            
            // Task Key
            const tdTask = document.createElement('td');
            tdTask.style.fontWeight = 'bold';
            tdTask.style.color = 'white';
            tdTask.textContent = taskKey.split('_').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ');
            
            // Description
            const tdDesc = document.createElement('td');
            tdDesc.style.fontSize = '0.7rem';
            tdDesc.style.color = 'var(--text-muted)';
            tdDesc.textContent = config.description || '';
            
            // Model selector
            const tdModel = document.createElement('td');
            const select = document.createElement('select');
            select.className = 'routing-model-select';
            select.dataset.task = taskKey;
            select.style.cssText = 'background:#0f172a; color:white; border:1px solid var(--border-color); padding:0.3rem; border-radius:6px; font-size:0.75rem; width:100%;';
            
            availableModels.forEach(m => {
                const opt = document.createElement('option');
                opt.value = m;
                opt.textContent = truncateModelName(m);
                if (m === config.model) {
                    opt.selected = true;
                }
                select.appendChild(opt);
            });
            tdModel.appendChild(select);
            
            // Min TPS
            const tdMinTps = document.createElement('td');
            const inputTps = document.createElement('input');
            inputTps.type = 'number';
            inputTps.step = '0.1';
            inputTps.min = '0';
            inputTps.className = 'routing-tps-input';
            inputTps.dataset.task = taskKey;
            inputTps.value = config.min_tps || 0;
            inputTps.style.cssText = 'background:#0f172a; color:white; border:1px solid var(--border-color); padding:0.3rem; border-radius:6px; font-size:0.75rem; width:60px; text-align:center;';
            tdMinTps.appendChild(inputTps);
            
            // Max TTFT
            const tdMaxTtft = document.createElement('td');
            const inputTtft = document.createElement('input');
            inputTtft.type = 'number';
            inputTtft.min = '0';
            inputTtft.className = 'routing-ttft-input';
            inputTtft.dataset.task = taskKey;
            inputTtft.value = config.max_ttft_ms || 0;
            inputTtft.style.cssText = 'background:#0f172a; color:white; border:1px solid var(--border-color); padding:0.3rem; border-radius:6px; font-size:0.75rem; width:70px; text-align:center;';
            tdMaxTtft.appendChild(inputTtft);
            
            // Reasoning Required
            const tdReasoning = document.createElement('td');
            const checkbox = document.createElement('input');
            checkbox.type = 'checkbox';
            checkbox.className = 'routing-reasoning-checkbox';
            checkbox.dataset.task = taskKey;
            checkbox.checked = !!config.reasoning_required;
            checkbox.style.cssText = 'width: 16px; height: 16px; cursor: pointer;';
            tdReasoning.style.textAlign = 'center';
            tdReasoning.appendChild(checkbox);
            
            // Live Status
            const tdStatus = document.createElement('td');
            tdStatus.style.fontSize = '0.7rem';
            tdStatus.id = `routing-status-${taskKey}`;
            tdStatus.textContent = 'Checking benchmarks...';
            
            tr.appendChild(tdTask);
            tr.appendChild(tdDesc);
            tr.appendChild(tdModel);
            tr.appendChild(tdMinTps);
            tr.appendChild(tdMaxTtft);
            tr.appendChild(tdReasoning);
            tr.appendChild(tdStatus);
            tbody.appendChild(tr);
            
            updateRoutingStatus(taskKey, config.model);
            
            // Re-fetch status when target model changes
            select.addEventListener('change', () => {
                updateRoutingStatus(taskKey, select.value);
            });
        });
    }
    
    async function updateRoutingStatus(taskKey, modelName) {
        const td = document.getElementById(`routing-status-${taskKey}`);
        if (!td) return;
        if (!modelName) {
            td.innerHTML = `<span style="color:var(--text-muted);">No model selected</span>`;
            return;
        }
        
        try {
            const res = await fetch(`/api/telemetry/recommendations?model=${encodeURIComponent(modelName)}`);
            const data = await res.json();
            
            if (data.baseline_comparison && data.baseline_comparison.baseline_tps) {
                const tps = data.baseline_comparison.baseline_tps;
                const ttft = data.baseline_comparison.baseline_ttft_ms;
                td.innerHTML = `<span style="color:#10b981; font-weight:600;">⚡ Bench: ${tps} TPS</span><br><span style="color:#fbbf24; font-weight:600;">🕒 TTFT: ${ttft}ms</span>`;
            } else {
                td.innerHTML = `<span style="color:var(--text-muted);">No benchmark baseline</span>`;
            }
        } catch (err) {
            td.innerHTML = `<span style="color:var(--color-danger);">Error</span>`;
        }
    }
    
    async function saveRoutingMatrix() {
        const matrix = {};
        const rows = document.querySelectorAll('#routing-matrix-body tr');
        
        rows.forEach(tr => {
            const select = tr.querySelector('.routing-model-select');
            const inputTps = tr.querySelector('.routing-tps-input');
            const inputTtft = tr.querySelector('.routing-ttft-input');
            const checkbox = tr.querySelector('.routing-reasoning-checkbox');
            
            if (select) {
                const taskKey = select.dataset.task;
                const description = tr.cells[1].textContent;
                
                matrix[taskKey] = {
                    model: select.value,
                    description: description,
                    min_tps: parseFloat(inputTps.value) || 0,
                    max_ttft_ms: parseInt(inputTtft.value) || 0,
                    reasoning_required: checkbox.checked
                };
            }
        });
        
        try {
            const res = await fetch('/api/routing/matrix', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(matrix)
            });
            const data = await res.json();
            if (res.ok) {
                showToast("Routing matrix updated successfully!", "success");
                loadRoutingMatrix();
            } else {
                showToast(data.error || "Failed to update routing matrix.", "error");
            }
        } catch (err) {
            console.error("Error saving routing matrix:", err);
            showToast("Failed to save routing matrix.", "error");
        }
    }

    function populateModelSwitcher(models) {
        if (!modelSwitcherSelect) return;
        
        modelSwitcherSelect.innerHTML = '';
        
        if (!models || models.length === 0) {
            const option = document.createElement('option');
            option.value = '';
            option.textContent = 'No models available';
            modelSwitcherSelect.appendChild(option);
            return;
        }
        
        models.forEach(model => {
            const option = document.createElement('option');
            option.value = model;
            option.textContent = model;
            modelSwitcherSelect.appendChild(option);
        });
    }
    
    if (btnSwitchModel) {
        btnSwitchModel.addEventListener('click', () => {
            const selectedModel = modelSwitcherSelect?.value;
            switchToModel(selectedModel);
        });
    }
    
    if (btnUnloadCurrent) {
        btnUnloadCurrent.addEventListener('click', unloadCurrentModel);
    }
    
    const btnClearVram = document.getElementById('btn-clear-vram');
    if (btnClearVram) {
        btnClearVram.addEventListener('click', clearVram);
    }

    const btnDeleteModel = document.getElementById('btn-delete-model');
    if (btnDeleteModel) {
        btnDeleteModel.addEventListener('click', () => {
            const selectedModel = modelSwitcherSelect?.value;
            deleteModel(selectedModel);
        });
    }

    const tuningStrategySelect = document.getElementById('tuning-strategy-select');
    if (tuningStrategySelect) {
        tuningStrategySelect.addEventListener('change', () => {
            const activeModelName = currentModelName;
            if (activeModelName) {
                updateTelemetryAndRecommendations(activeModelName);
            }
        });
    }
    
    const btnApplyOptimizations = document.getElementById('btn-apply-optimizations');
    if (btnApplyOptimizations) {
        btnApplyOptimizations.addEventListener('click', applyTuningOptimizations);
    }

    // RESOURCE ANALYSIS
    async function analyzeAllModels() {
        const btn = document.getElementById('btn-analyze-all');
        const resultsEl = document.getElementById('resource-analysis-results');
        const strategy = document.getElementById('analysis-strategy-select')?.value || 'performance';

        if (!btn || !resultsEl) return;
        btn.disabled = true;
        btn.textContent = 'Analyzing...';
        resultsEl.innerHTML = `<div style="text-align:center;padding:2rem;color:var(--text-muted);">⏳ Running resource analysis across all models...</div>`;

        try {
            const res = await fetch(`/api/analyze/all?strategy=${strategy}`);
            const data = await res.json();

            if (data.error) {
                resultsEl.innerHTML = `<div style="color:var(--color-danger);padding:1rem;">❌ ${data.error}</div>`;
                return;
            }

            const { results, models_analyzed, models_skipped } = data;

            if (!results || results.length === 0) {
                resultsEl.innerHTML = `<div style="text-align:center;padding:2rem;color:var(--text-muted);">No telemetry data found. Ensure models have been run with telemetry active.</div>`;
                return;
            }

            const statusColors = { ok: 'var(--color-success)', warning: 'var(--color-warning)', critical: 'var(--color-danger)' };
            const statusIcons = { ok: '✅', warning: '⚠️', critical: '🔴' };

            let html = `<div style="margin-bottom:0.75rem;font-size:0.7rem;color:var(--text-muted);">Analyzed <strong style="color:var(--text-primary)">${models_analyzed}</strong> models. Skipped: ${models_skipped.join(', ') || 'none'}.</div>`;

            results.forEach(r => {
                const statusColor = statusColors[r.status] || 'var(--text-muted)';
                const statusIcon = statusIcons[r.status] || '✅';
                const hasRecs = r.recommendations && Object.keys(r.recommendations).length > 0;
                const vram = r.vram_summary;
                const ram = r.ram_summary;

                // VRAM bar
                const vramPct = Math.min(100, vram.max_pct || 0);
                const vramBarColor = vramPct > 85 ? 'var(--color-danger)' : vramPct > 60 ? 'var(--color-warning)' : 'var(--color-success)';

                html += `
                <div style="border:1px solid var(--border-color);border-radius:8px;padding:0.75rem;margin-bottom:0.75rem;background:rgba(255,255,255,0.02);">
                    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.5rem;">
                        <span style="font-weight:600;color:var(--text-primary);font-family:monospace;font-size:0.78rem;">${r.model_alias}</span>
                        <span style="color:${statusColor};font-size:0.7rem;">${statusIcon} ${r.status.toUpperCase()}</span>
                    </div>
                    <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.4rem;margin-bottom:0.5rem;font-size:0.68rem;">
                        <div>
                            <div style="color:var(--text-muted);margin-bottom:2px;">VRAM: ${vram.used_mb}MB / ${vram.total_mb}MB (${vramPct}%)</div>
                            <div style="background:var(--border-color);border-radius:4px;height:6px;overflow:hidden;">
                                <div style="width:${vramPct}%;height:100%;background:${vramBarColor};border-radius:4px;transition:width 0.3s;"></div>
                            </div>
                        </div>
                        <div>
                            <div style="color:var(--text-muted);margin-bottom:2px;">RAM Peak: ${ram.max_pct}% &nbsp;|&nbsp; GPU Util: ${r.gpu_util_pct?.max || 0}%</div>
                            <div style="color:var(--text-muted);">VRAM Headroom: <strong style="color:var(--text-primary)">${vram.headroom_mb}MB free</strong></div>
                        </div>
                    </div>
                    ${r.detected_issues && r.detected_issues[0] !== 'No resource utilization issues detected.' ? `
                    <div style="color:${statusColor};font-size:0.68rem;margin-bottom:0.4rem;">⚠ ${r.detected_issues[0]}</div>` : ''}
                    ${hasRecs ? `
                    <div style="background:rgba(99,102,241,0.08);border:1px solid rgba(99,102,241,0.2);border-radius:6px;padding:0.5rem;margin-top:0.4rem;">
                        <div style="color:var(--color-primary);font-weight:600;font-size:0.68rem;margin-bottom:0.3rem;">💡 Suggested Settings:</div>
                        <div style="font-family:monospace;font-size:0.67rem;color:var(--text-secondary);margin-bottom:0.4rem;">${Object.entries(r.recommendations).map(([k,v]) => `${k} = ${v}`).join(' &nbsp;|&nbsp; ')}</div>
                        <div style="font-size:0.67rem;color:var(--text-muted);margin-bottom:0.5rem;line-height:1.4;">${r.explanation}</div>
                        <button class="btn btn-primary" style="padding:0.25rem 0.6rem;font-size:0.65rem;margin:0;"
                            onclick="applyAnalysisRec('${r.model_alias}', ${JSON.stringify(r.recommendations).replace(/"/g, '&quot;')})">
                            Apply to Profile
                        </button>
                    </div>` : `<div style="color:var(--color-success);font-size:0.68rem;">✅ No optimizations needed. Settings are well-configured.</div>`}
                </div>`;
            });

            resultsEl.innerHTML = html;
        } catch (err) {
            resultsEl.innerHTML = `<div style="color:var(--color-danger);padding:1rem;">❌ Analysis failed: ${err.message}</div>`;
        } finally {
            btn.disabled = false;
            btn.textContent = 'Analyze All Models';
        }
    }

    async function applyAnalysisRec(modelAlias, recommendations) {
        try {
            const res = await fetch('/api/telemetry/recommendations/apply', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ model: modelAlias, recommendations })
            });
            const data = await res.json();
            if (data.status === 'success') {
                showToast(`✅ Applied settings for ${modelAlias}. Reload model to activate.`, 'success');
            } else {
                showToast(`❌ Failed to apply: ${data.error || 'Unknown error'}`, 'error');
            }
        } catch (err) {
            showToast(`❌ Apply failed: ${err.message}`, 'error');
        }
    }

    const btnAnalyzeAll = document.getElementById('btn-analyze-all');
    if (btnAnalyzeAll) {
        btnAnalyzeAll.addEventListener('click', analyzeAllModels);
    }

    // MODEL ERROR LOG
    const ERROR_TYPE_STYLES = {
        context_overflow: { color: '#f97316', label: 'CTX OVERFLOW' },
        oom:              { color: '#ef4444', label: 'OOM' },
        slot_unavailable: { color: '#eab308', label: 'SLOT BUSY' },
        connection_error: { color: '#8b5cf6', label: 'CONN ERROR' },
        inference_error:  { color: '#64748b', label: 'INFERENCE' },
        bad_request:      { color: '#06b6d4', label: 'BAD REQUEST' },
        model_not_found:  { color: '#ec4899', label: 'NOT FOUND' },
        upstream_error:   { color: '#f43f5e', label: 'UPSTREAM' },
    };

    async function loadErrorLog() {
        const listEl = document.getElementById('error-log-list');
        const summaryEl = document.getElementById('error-log-summary');
        const filterType = document.getElementById('error-log-filter-type')?.value || '';
        if (!listEl) return;

        try {
            const params = new URLSearchParams({ limit: 100 });
            if (filterType) params.set('error_type', filterType);
            const res = await fetch(`/api/errors?${params}`);
            const data = await res.json();

            if (data.error) {
                listEl.innerHTML = `<div style="color:var(--color-danger);padding:0.75rem;">⚠ ${data.error}</div>`;
                return;
            }

            const errors = data.errors || [];
            const counts = data.error_type_counts || {};

            // Summary badges
            if (summaryEl) {
                summaryEl.innerHTML = Object.entries(counts).map(([type, cnt]) => {
                    const style = ERROR_TYPE_STYLES[type] || { color: '#64748b', label: type.toUpperCase() };
                    return `<span style="background:${style.color}22;border:1px solid ${style.color}55;color:${style.color};padding:0.15rem 0.5rem;border-radius:4px;font-size:0.65rem;font-weight:600;">${style.label} &times;${cnt}</span>`;
                }).join('') || '<span style="color:var(--text-muted);">No errors</span>';
            }

            if (errors.length === 0) {
                listEl.innerHTML = `<div style="color:var(--text-muted);text-align:center;padding:1.5rem;">No errors recorded${filterType ? ` for type "${filterType}"` : ''}.</div>`;
                return;
            }

            listEl.innerHTML = errors.map(e => {
                const style = ERROR_TYPE_STYLES[e.error_type] || { color: '#64748b', label: (e.error_type || 'unknown').toUpperCase() };
                const ts = e.timestamp || '';
                const model = e.model || 'unknown';
                const msg = (e.message || '').replace(/</g, '&lt;').replace(/>/g, '&gt;');
                const extras = [];
                if (e.n_prompt_tokens) extras.push(`prompt_tokens: ${e.n_prompt_tokens.toLocaleString()}`);
                if (e.n_ctx) extras.push(`ctx_size: ${e.n_ctx.toLocaleString()}`);
                if (e.http_status) extras.push(`HTTP ${e.http_status}`);
                return `
                <div style="display:flex;gap:0.6rem;align-items:flex-start;padding:0.5rem 0;border-bottom:1px solid rgba(255,255,255,0.04);">
                    <span style="background:${style.color}22;border:1px solid ${style.color}55;color:${style.color};padding:0.1rem 0.4rem;border-radius:4px;font-size:0.6rem;font-weight:700;white-space:nowrap;margin-top:1px;">${style.label}</span>
                    <div style="flex:1;min-width:0;">
                        <div style="display:flex;gap:0.5rem;align-items:center;margin-bottom:0.2rem;">
                            <span style="color:var(--text-primary);font-family:monospace;font-size:0.7rem;font-weight:600;">${model}</span>
                            <span style="color:var(--text-muted);font-size:0.62rem;">${ts}</span>
                            ${extras.map(x => `<span style="color:var(--text-muted);font-size:0.62rem;background:rgba(255,255,255,0.05);padding:0 0.3rem;border-radius:3px;">${x}</span>`).join('')}
                        </div>
                        <div style="color:var(--text-secondary);font-size:0.68rem;line-height:1.35;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;max-width:100%;" title="${msg}">${msg}</div>
                    </div>
                </div>`;
            }).join('');

        } catch (err) {
            if (listEl) listEl.innerHTML = `<div style="color:var(--color-danger);padding:0.75rem;">Failed to load error log: ${err.message}</div>`;
        }
    }

    // Auto-refresh error log every 30s when on Monitor tab
    let errorLogInterval = null;
    document.addEventListener('tabChanged', (e) => {
        if (e.detail === 'monitor') {
            if (!errorLogInterval) {
                loadErrorLog();
                // Visibility-aware: skip network churn while the page is hidden.
                errorLogInterval = setInterval(() => { if (!document.hidden) loadErrorLog(); }, 30000);
            }
        } else {
            if (errorLogInterval) {
                clearInterval(errorLogInterval);
                errorLogInterval = null;
            }
        }
    });

    const btnRefreshErrors = document.getElementById('btn-refresh-errors');
    if (btnRefreshErrors) btnRefreshErrors.addEventListener('click', loadErrorLog);

    const errorFilterType = document.getElementById('error-log-filter-type');
    if (errorFilterType) errorFilterType.addEventListener('change', loadErrorLog);

    const btnClearErrors = document.getElementById('btn-clear-errors');
    if (btnClearErrors) {
        btnClearErrors.addEventListener('click', async () => {
            try {
                await fetch('/api/errors/clear', { method: 'POST' });
                await loadErrorLog();
                showToast('Error log cleared', 'success');
            } catch (err) {
                showToast(`Failed to clear: ${err.message}`, 'error');
            }
        });
    }

    // Load error log on initial monitor tab view
    loadErrorLog();

    const btnSaveRoutingMatrix = document.getElementById('btn-save-routing-matrix');
    if (btnSaveRoutingMatrix) {
        btnSaveRoutingMatrix.addEventListener('click', saveRoutingMatrix);
    }
    
    // Periodically update current model in switcher (skips while tab hidden)
    setInterval(() => { if (!document.hidden) updateCurrentModel(); }, 5000);

    // MODEL DISCOVERY SEARCH AND PULL INTEGRATION
    const btnOpenSearchModal = document.getElementById('btn-open-search-modal');
    const searchPullOverlay = document.getElementById('search-pull-overlay');
    const searchPullClose = document.getElementById('search-pull-close');
    const modelSearchQuery = document.getElementById('model-search-query');
    const modelSearchSource = document.getElementById('model-search-source');
    const btnRunModelSearch = document.getElementById('btn-run-model-search');
    const searchLoading = document.getElementById('search-loading');
    const searchResultsContainer = document.getElementById('search-results-container');
    const hfFilesContainer = document.getElementById('hf-files-container');
    const hfFilesTitle = document.getElementById('hf-files-title');
    const hfFilesList = document.getElementById('hf-files-list');
    const btnBackToSearch = document.getElementById('btn-back-to-search');
    const pullProgressContainer = document.getElementById('pull-progress-container');
    const pullModelName = document.getElementById('pull-model-name');
    const pullStatusBadge = document.getElementById('pull-status-badge');
    const pullConsoleLog = document.getElementById('pull-console-log');
    const btnPullStop = document.getElementById('btn-pull-stop');
    const btnPullCancel = document.getElementById('btn-pull-cancel');
    const modelSearchType = document.getElementById('model-search-type');
    const searchResultCount = document.getElementById('search-result-count');

    let currentHfRepo = "";

    if (btnOpenSearchModal) {
        btnOpenSearchModal.addEventListener('click', () => {
            if (searchPullOverlay) searchPullOverlay.classList.add('open');
            if (modelSearchQuery) {
                modelSearchQuery.value = '';
                modelSearchQuery.focus();
            }
            if (searchResultsContainer) {
                searchResultsContainer.innerHTML = `<div style="text-align:center; color:var(--text-muted); font-size:0.85rem; padding:2rem;">Search for models above to discover from Ollama Library and Hugging Face.</div>`;
            }
            if (hfFilesContainer) hfFilesContainer.classList.add('d-none');
            if (pullProgressContainer) pullProgressContainer.classList.add('d-none');
            loadActivePulls();
        });
    }

    if (searchPullClose) {
        searchPullClose.addEventListener('click', () => {
            if (searchPullOverlay) searchPullOverlay.classList.remove('open');
        });
    }

    if (searchPullOverlay) {
        searchPullOverlay.addEventListener('click', (e) => {
            if (e.target === searchPullOverlay) {
                searchPullOverlay.classList.remove('open');
            }
        });
    }

    if (btnBackToSearch) {
        btnBackToSearch.addEventListener('click', () => {
            if (hfFilesContainer) hfFilesContainer.classList.add('d-none');
            if (searchResultsContainer) searchResultsContainer.classList.remove('d-none');
        });
    }

    async function executeModelSearch() {
        const query = modelSearchQuery?.value.trim();
        if (!query) {
            showToast("Please enter a search query.", 'error');
            return;
        }

        if (searchLoading) searchLoading.classList.remove('d-none');
        if (searchResultsContainer) searchResultsContainer.innerHTML = '';
        if (hfFilesContainer) hfFilesContainer.classList.add('d-none');
        if (searchResultCount) searchResultCount.textContent = '';

        try {
            const res = await fetch('/api/models/search', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    query: query,
                    source: modelSearchSource?.value || 'all',
                    type: modelSearchType?.value || 'all'
                })
            });

            const data = await res.json();
            if (searchLoading) searchLoading.classList.add('d-none');

            if (!res.ok) {
                if (searchResultsContainer) {
                    searchResultsContainer.innerHTML = `<div style="text-align:center; color:var(--color-danger); padding:1rem;">${data.error || 'Failed to search models'}</div>`;
                }
                return;
            }

            const results = data.results || [];
            if (searchResultCount) searchResultCount.textContent = `${results.length} result${results.length !== 1 ? 's' : ''}`;

            if (results.length === 0) {
                if (searchResultsContainer) {
                    searchResultsContainer.innerHTML = `<div style="text-align:center; color:var(--text-muted); padding:2rem;">No matching models found. Try a different query or type filter.</div>`;
                }
                return;
            }

            if (searchResultsContainer) {
                searchResultsContainer.style.cssText = 'display: grid; grid-template-columns: repeat(auto-fill, minmax(340px, 1fr)); gap: 1rem; padding: 0.25rem; max-height: 350px; overflow-y: auto;';
                searchResultsContainer.innerHTML = '';

                results.forEach(item => {
                    const isSd = item.type === 'stable-diffusion';

                    // Card base - tinted differently for SD vs LLM
                    const card = document.createElement('div');
                    card.style.cssText = isSd
                        ? 'background: rgba(234, 88, 12, 0.07); border: 1px solid rgba(234, 88, 12, 0.18); border-radius: 10px; padding: 1rem; display: flex; flex-direction: column; justify-content: space-between; gap: 0.75rem; transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;'
                        : 'background: rgba(30, 41, 59, 0.45); border: 1px solid rgba(255, 255, 255, 0.07); border-radius: 10px; padding: 1rem; display: flex; flex-direction: column; justify-content: space-between; gap: 0.75rem; transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;';

                    const hoverBorder = isSd ? 'rgba(251, 146, 60, 0.4)' : 'rgba(139, 92, 246, 0.3)';
                    const hoverShadow = isSd
                        ? '0 8px 20px rgba(0,0,0,0.3), 0 0 10px rgba(234,88,12,0.15)'
                        : '0 8px 20px rgba(0,0,0,0.3), 0 0 10px rgba(139,92,246,0.1)';
                    card.addEventListener('mouseenter', () => {
                        card.style.transform = 'translateY(-2px)';
                        card.style.borderColor = hoverBorder;
                        card.style.boxShadow = hoverShadow;
                    });
                    card.addEventListener('mouseleave', () => {
                        card.style.transform = 'none';
                        card.style.borderColor = isSd ? 'rgba(234,88,12,0.18)' : 'rgba(255,255,255,0.07)';
                        card.style.boxShadow = 'none';
                    });

                    // Header: name + badges
                    const header = document.createElement('div');
                    header.style.cssText = 'display: flex; justify-content: space-between; align-items: flex-start; gap: 0.5rem;';

                    const titleWrapper = document.createElement('div');
                    titleWrapper.style.cssText = 'display: flex; flex-direction: column; min-width: 0; flex: 1; gap: 0.25rem;';

                    const nameSpan = document.createElement('span');
                    nameSpan.style.cssText = 'color: white; font-weight: 600; font-size: 0.85rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; display: block;';
                    nameSpan.textContent = item.name;
                    nameSpan.title = item.name;
                    titleWrapper.appendChild(nameSpan);

                    // Type badge (LLM or Stable Diffusion)
                    const typeBadge = document.createElement('span');
                    if (isSd) {
                        typeBadge.style.cssText = 'background: rgba(234,88,12,0.15); color: #fb923c; border: 1px solid rgba(234,88,12,0.3); font-size: 0.6rem; padding: 0.1rem 0.35rem; border-radius: 20px; font-weight: 600; display: inline-flex; align-items: center; gap: 0.2rem; width: fit-content;';
                        typeBadge.innerHTML = '🎨 Stable Diffusion';
                    } else {
                        typeBadge.style.cssText = 'background: rgba(99,102,241,0.12); color: #a5b4fc; border: 1px solid rgba(99,102,241,0.25); font-size: 0.6rem; padding: 0.1rem 0.35rem; border-radius: 20px; font-weight: 600; display: inline-flex; align-items: center; gap: 0.2rem; width: fit-content;';
                        typeBadge.innerHTML = '🤖 Language Model';
                    }
                    titleWrapper.appendChild(typeBadge);

                    // Source badge (Ollama / HF)
                    const sourceBadge = document.createElement('span');
                    sourceBadge.style.cssText = item.source === 'ollama'
                        ? 'background: rgba(16,185,129,0.15); color: #10b981; border: 1px solid rgba(16,185,129,0.25); font-size: 0.65rem; padding: 0.15rem 0.4rem; border-radius: 20px; display: inline-flex; align-items: center; gap: 0.25rem; font-weight: 500; white-space: nowrap;'
                        : 'background: rgba(37,99,235,0.15); color: #60a5fa; border: 1px solid rgba(37,99,235,0.25); font-size: 0.65rem; padding: 0.15rem 0.4rem; border-radius: 20px; display: inline-flex; align-items: center; gap: 0.25rem; font-weight: 500; white-space: nowrap;';
                    sourceBadge.textContent = item.source === 'ollama' ? '🦙 Ollama' : '🤗 HF';

                    header.appendChild(titleWrapper);
                    header.appendChild(sourceBadge);

                    // Body: description + stats pills
                    const body = document.createElement('div');
                    body.style.cssText = 'flex: 1; display: flex; flex-direction: column; gap: 0.4rem;';

                    const descDiv = document.createElement('p');
                    descDiv.style.cssText = 'font-size: 0.75rem; color: var(--text-muted); line-height: 1.45; margin: 0; overflow: hidden; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical;';

                    if (item.source === 'huggingface') {
                        // Author line
                        const author = item.author || 'HF Author';
                        descDiv.textContent = item.description.startsWith('[Direct Match]')
                            ? `⭐ Direct match · by ${author}`
                            : `Repository by ${author}`;

                        const statsWrapper = document.createElement('div');
                        statsWrapper.style.cssText = 'display: flex; flex-wrap: wrap; gap: 0.3rem; margin-top: 0.1rem;';

                        const pillStyle = 'background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.06); padding: 0.1rem 0.3rem; border-radius: 4px; font-size: 0.65rem; color: #94a3b8; display: inline-flex; align-items: center; gap: 0.2rem;';

                        if (item.downloads != null) {
                            const dlPill = document.createElement('span');
                            dlPill.style.cssText = pillStyle;
                            dlPill.innerHTML = `📥 <span style="color:#cbd5e1;font-weight:500;">${Number(item.downloads).toLocaleString()}</span>`;
                            statsWrapper.appendChild(dlPill);
                        }
                        if (item.likes != null) {
                            const likesPill = document.createElement('span');
                            likesPill.style.cssText = pillStyle;
                            likesPill.innerHTML = `❤️ <span style="color:#cbd5e1;font-weight:500;">${Number(item.likes).toLocaleString()}</span>`;
                            statsWrapper.appendChild(likesPill);
                        }
                        if (item.tags && item.tags.length > 0) {
                            item.tags.slice(0, 3).forEach(tag => {
                                const tagPill = document.createElement('span');
                                tagPill.style.cssText = pillStyle + ' max-width:110px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;';
                                tagPill.textContent = `🏷️ ${tag}`;
                                statsWrapper.appendChild(tagPill);
                            });
                        }

                        body.appendChild(descDiv);
                        body.appendChild(statsWrapper);
                    } else {
                        descDiv.textContent = item.description || 'No description provided.';
                        body.appendChild(descDiv);
                    }

                    // Footer: action button
                    const footer = document.createElement('div');
                    footer.style.cssText = 'display: flex; justify-content: flex-end; align-items: center; margin-top: auto; padding-top: 0.5rem; border-top: 1px solid rgba(255,255,255,0.05);';

                    const actionBtn = document.createElement('button');
                    actionBtn.className = 'btn';

                    const sdBtnStyle = 'font-size:0.75rem; padding:0.35rem 0.75rem; background:#c2410c; border-color:#c2410c; color:white; border-radius:6px; font-weight:500; width:100%; text-align:center; display:flex; align-items:center; justify-content:center; gap:0.25rem; cursor:pointer; transition: background 0.2s, transform 0.1s;';
                    const llmOllamaBtnStyle = 'font-size:0.75rem; padding:0.35rem 0.75rem; background:#059669; border-color:#059669; color:white; border-radius:6px; font-weight:500; width:100%; text-align:center; display:flex; align-items:center; justify-content:center; gap:0.25rem; cursor:pointer; transition: background 0.2s, transform 0.1s;';
                    const llmHfBtnStyle = 'font-size:0.75rem; padding:0.35rem 0.75rem; background:#2563eb; border-color:#2563eb; color:white; border-radius:6px; font-weight:500; width:100%; text-align:center; display:flex; align-items:center; justify-content:center; gap:0.25rem; cursor:pointer; transition: background 0.2s, transform 0.1s;';

                    if (item.source === 'ollama') {
                        actionBtn.style.cssText = llmOllamaBtnStyle;
                        actionBtn.innerHTML = '📂 View Available Tags';
                        actionBtn.addEventListener('click', () => showOllamaModelTags(item.name));
                        actionBtn.addEventListener('mouseenter', () => { actionBtn.style.background = '#047857'; actionBtn.style.transform = 'scale(1.01)'; });
                        actionBtn.addEventListener('mouseleave', () => { actionBtn.style.background = '#059669'; actionBtn.style.transform = 'none'; });
                    } else if (isSd) {
                        actionBtn.style.cssText = sdBtnStyle;
                        actionBtn.innerHTML = '🎨 Browse Model Files';
                        actionBtn.addEventListener('click', () => showHfRepoFiles(item.name, 'stable-diffusion'));
                        actionBtn.addEventListener('mouseenter', () => { actionBtn.style.background = '#9a3412'; actionBtn.style.transform = 'scale(1.01)'; });
                        actionBtn.addEventListener('mouseleave', () => { actionBtn.style.background = '#c2410c'; actionBtn.style.transform = 'none'; });
                    } else {
                        actionBtn.style.cssText = llmHfBtnStyle;
                        actionBtn.innerHTML = '📂 View GGUF Files';
                        actionBtn.addEventListener('click', () => showHfRepoFiles(item.name, 'llm'));
                        actionBtn.addEventListener('mouseenter', () => { actionBtn.style.background = '#1d4ed8'; actionBtn.style.transform = 'scale(1.01)'; });
                        actionBtn.addEventListener('mouseleave', () => { actionBtn.style.background = '#2563eb'; actionBtn.style.transform = 'none'; });
                    }

                    footer.appendChild(actionBtn);
                    card.appendChild(header);
                    card.appendChild(body);
                    card.appendChild(footer);
                    searchResultsContainer.appendChild(card);
                });
            }

        } catch (err) {
            if (searchLoading) searchLoading.classList.add('d-none');
            if (searchResultsContainer) {
                searchResultsContainer.innerHTML = `<div style="text-align:center; color:var(--color-danger); padding:1rem;">Search error: ${err.message}</div>`;
            }
        }
    }

    if (btnRunModelSearch) {
        btnRunModelSearch.addEventListener('click', executeModelSearch);
    }
    if (modelSearchQuery) {
        modelSearchQuery.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') executeModelSearch();
        });
    }

    async function showOllamaModelTags(modelName) {
        if (searchResultsContainer) searchResultsContainer.classList.add('d-none');
        if (hfFilesContainer) {
            hfFilesContainer.classList.remove('d-none');
        }
        if (hfFilesTitle) {
            hfFilesTitle.textContent = `Model Tags: ${modelName}`;
        }
        if (hfFilesList) {
            hfFilesList.innerHTML = `<div style="text-align:center; padding:1.5rem; color:var(--text-muted);"><div class="loader" style="width:20px; height:20px; border-width:2px; display:inline-block;"></div><br>Fetching tags from Ollama Registry...</div>`;
        }

        try {
            const res = await fetch(`/api/models/ollama/tags?model=${encodeURIComponent(modelName)}`);
            const data = await res.json();

            if (!res.ok) {
                if (hfFilesList) {
                    hfFilesList.innerHTML = `<div style="text-align:center; color:var(--color-danger); padding:1rem;">${data.error || 'Failed to list tags'}</div>`;
                }
                return;
            }

            const tags = data.tags || [];
            if (tags.length === 0) {
                if (hfFilesList) {
                    hfFilesList.innerHTML = `<div style="text-align:center; color:var(--text-muted); padding:1rem;">No tags found for this model.</div>`;
                }
                return;
            }

            if (hfFilesList) {
                hfFilesList.style.cssText = 'display: grid; grid-template-columns: repeat(auto-fill, minmax(340px, 1fr)); gap: 0.75rem; padding: 0.25rem; max-height: 250px; overflow-y: auto;';
                hfFilesList.innerHTML = '';
                tags.forEach(tag => {
                    const card = document.createElement('div');
                    card.style.cssText = 'background: rgba(30, 41, 59, 0.45); border: 1px solid rgba(255, 255, 255, 0.07); border-radius: 8px; padding: 0.75rem; display: flex; flex-direction: column; gap: 0.5rem; transition: border-color 0.2s, transform 0.2s;';
                    
                    const top = document.createElement('div');
                    top.style.cssText = 'display:flex; justify-content:space-between; align-items:center; gap:0.5rem;';
                    
                    const nameSpan = document.createElement('span');
                    nameSpan.style.cssText = 'color: white; font-size: 0.75rem; font-family: monospace; font-weight: 500; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; flex: 1;';
                    nameSpan.textContent = `${modelName}:${tag}`;
                    nameSpan.title = `${modelName}:${tag}`;
                    
                    const tagSpan = document.createElement('span');
                    tagSpan.style.cssText = 'font-size: 0.65rem; color: #10b981; background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.2); padding: 0.1rem 0.35rem; border-radius: 4px; font-weight: 500;';
                    tagSpan.textContent = tag;
                    
                    top.appendChild(nameSpan);
                    top.appendChild(tagSpan);
                    
                    const footer = document.createElement('div');
                    footer.style.cssText = 'display:flex; justify-content:flex-end; border-top: 1px solid rgba(255, 255, 255, 0.04); padding-top: 0.4rem; margin-top: auto;';
                    
                    const pullBtn = document.createElement('button');
                    pullBtn.className = 'btn btn-primary';
                    pullBtn.style.cssText = 'font-size: 0.7rem; padding: 0.3rem 0.65rem; background: #059669; border-color: #059669; color: white; border-radius: 4px; display: inline-flex; align-items: center; gap: 0.25rem; font-weight: 500; cursor: pointer; transition: background 0.2s, transform 0.1s; width: 100%; justify-content: center;';
                    pullBtn.innerHTML = '📥 Pull Model Variant';
                    
                    pullBtn.addEventListener('mouseenter', () => {
                        pullBtn.style.transform = 'scale(1.01)';
                        pullBtn.style.background = '#047857';
                    });
                    pullBtn.addEventListener('mouseleave', () => {
                        pullBtn.style.transform = 'none';
                        pullBtn.style.background = '#059669';
                    });
                    
                    pullBtn.addEventListener('click', () => {
                        pullModel(`${modelName}:${tag}`, 'ollama');
                    });
                    
                    footer.appendChild(pullBtn);
                    card.appendChild(top);
                    card.appendChild(footer);
                    hfFilesList.appendChild(card);
                });
            }

        } catch (err) {
            if (hfFilesList) {
                hfFilesList.innerHTML = `<div style="text-align:center; color:var(--color-danger); padding:1rem;">Error listing tags: ${err.message}</div>`;
            }
        }
    }

    async function showHfRepoFiles(repoName, hintType = 'llm') {
        currentHfRepo = repoName;
        if (searchResultsContainer) searchResultsContainer.classList.add('d-none');
        if (hfFilesContainer) hfFilesContainer.classList.remove('d-none');
        if (hfFilesTitle) {
            const typeLabel = hintType === 'stable-diffusion' ? '🎨 Stable Diffusion' : '🤖 GGUF';
            hfFilesTitle.textContent = `Repository: ${repoName}`;
            hfFilesTitle.title = `Type: ${typeLabel}`;
        }
        if (hfFilesList) {
            hfFilesList.innerHTML = `<div style="text-align:center; padding:1.5rem; color:var(--text-muted);"><div class="loader" style="width:20px; height:20px; border-width:2px; display:inline-block;"></div><br>Fetching files from Hugging Face...</div>`;
        }

        try {
            const res = await fetch(`/api/models/huggingface/files?repo=${encodeURIComponent(repoName)}`);
            const data = await res.json();

            if (!res.ok) {
                if (hfFilesList) {
                    hfFilesList.innerHTML = `<div style="text-align:center; color:var(--color-danger); padding:1rem;">${data.error || 'Failed to list files'}</div>`;
                }
                return;
            }

            const files = data.files || [];
            const repoType = data.repo_type || hintType;  // authoritative type from backend
            const isSDRepo = repoType === 'stable-diffusion';

            // Update title with confirmed type
            if (hfFilesTitle) {
                const typeBadgeHtml = isSDRepo
                    ? '<span style="font-size:0.65rem; background:rgba(234,88,12,0.15); color:#fb923c; border:1px solid rgba(234,88,12,0.3); padding:0.1rem 0.35rem; border-radius:20px; font-weight:600; margin-left:0.5rem;">🎨 Stable Diffusion</span>'
                    : '<span style="font-size:0.65rem; background:rgba(99,102,241,0.12); color:#a5b4fc; border:1px solid rgba(99,102,241,0.25); padding:0.1rem 0.35rem; border-radius:20px; font-weight:600; margin-left:0.5rem;">🤖 Language Model</span>';
                hfFilesTitle.innerHTML = `Repository: <span style="color:white;">${escapeHtml(repoName)}</span>${typeBadgeHtml}`;
            }

            if (files.length === 0) {
                if (hfFilesList) {
                    hfFilesList.innerHTML = `<div style="text-align:center; color:var(--text-muted); padding:1rem;">No compatible model files found in this repository.</div>`;
                }
                return;
            }

            if (hfFilesList) {
                hfFilesList.style.cssText = 'display: grid; grid-template-columns: repeat(auto-fill, minmax(340px, 1fr)); gap: 0.75rem; padding: 0.25rem; max-height: 250px; overflow-y: auto;';
                hfFilesList.innerHTML = '';

                files.forEach(file => {
                    const isFileSd = file.type === 'stable-diffusion';
                    const isSafetensors = file.format === 'safetensors';

                    const card = document.createElement('div');
                    card.style.cssText = isFileSd
                        ? 'background: rgba(234,88,12,0.06); border: 1px solid rgba(234,88,12,0.18); border-radius: 8px; padding: 0.75rem; display: flex; flex-direction: column; gap: 0.5rem; transition: border-color 0.2s, transform 0.2s;'
                        : 'background: rgba(30,41,59,0.45); border: 1px solid rgba(255,255,255,0.07); border-radius: 8px; padding: 0.75rem; display: flex; flex-direction: column; gap: 0.5rem; transition: border-color 0.2s, transform 0.2s;';

                    const top = document.createElement('div');
                    top.style.cssText = 'display:flex; justify-content:space-between; align-items:center; gap:0.5rem;';

                    const nameSpan = document.createElement('span');
                    nameSpan.style.cssText = 'color: white; font-size: 0.75rem; font-family: monospace; font-weight: 500; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; flex: 1;';
                    nameSpan.textContent = file.filename;
                    nameSpan.title = file.filename;

                    const badgesDiv = document.createElement('div');
                    badgesDiv.style.cssText = 'display:flex; gap:0.25rem; align-items:center; flex-shrink:0;';

                    // Format badge (.gguf vs .safetensors)
                    const fmtBadge = document.createElement('span');
                    if (isSafetensors) {
                        fmtBadge.style.cssText = 'font-size:0.6rem; color:#fb923c; background:rgba(234,88,12,0.15); border:1px solid rgba(234,88,12,0.3); padding:0.1rem 0.3rem; border-radius:4px; font-weight:600;';
                        fmtBadge.textContent = '.safetensors';
                    } else {
                        fmtBadge.style.cssText = 'font-size:0.6rem; color:#a5b4fc; background:rgba(99,102,241,0.12); border:1px solid rgba(99,102,241,0.25); padding:0.1rem 0.3rem; border-radius:4px; font-weight:600;';
                        fmtBadge.textContent = '.gguf';
                    }
                    badgesDiv.appendChild(fmtBadge);

                    // Size badge
                    if (file.size) {
                        const sizeSpan = document.createElement('span');
                        sizeSpan.style.cssText = 'font-size:0.65rem; color:#94a3b8; background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.06); padding:0.1rem 0.35rem; border-radius:4px; font-weight:500;';
                        sizeSpan.textContent = file.size;
                        badgesDiv.appendChild(sizeSpan);
                    }

                    top.appendChild(nameSpan);
                    top.appendChild(badgesDiv);

                    const footer = document.createElement('div');
                    footer.style.cssText = 'display:flex; justify-content:flex-end; border-top:1px solid rgba(255,255,255,0.04); padding-top:0.4rem; margin-top:auto;';

                    const pullBtn = document.createElement('button');
                    pullBtn.className = 'btn btn-primary';

                    if (isFileSd) {
                        pullBtn.style.cssText = 'font-size:0.7rem; padding:0.3rem 0.65rem; background:#c2410c; border-color:#c2410c; color:white; border-radius:4px; display:inline-flex; align-items:center; gap:0.25rem; font-weight:500; cursor:pointer; width:100%; justify-content:center;';
                        pullBtn.innerHTML = '🎨 Download for Diffusion';
                        pullBtn.addEventListener('mouseenter', () => { pullBtn.style.background = '#9a3412'; });
                        pullBtn.addEventListener('mouseleave', () => { pullBtn.style.background = '#c2410c'; });
                    } else {
                        pullBtn.style.cssText = 'font-size:0.7rem; padding:0.3rem 0.65rem; background:#059669; border-color:#059669; color:white; border-radius:4px; display:inline-flex; align-items:center; gap:0.25rem; font-weight:500; cursor:pointer; width:100%; justify-content:center;';
                        pullBtn.innerHTML = '📥 Download for LLM';
                        pullBtn.addEventListener('mouseenter', () => { pullBtn.style.background = '#047857'; });
                        pullBtn.addEventListener('mouseleave', () => { pullBtn.style.background = '#059669'; });
                    }

                    pullBtn.addEventListener('click', () => {
                        const baseName = file.filename.replace(/\.(gguf|safetensors)$/i, '').toLowerCase().replace(/[^a-z0-9\-]/g, '-');
                        let alias;
                        if (isFileSd) {
                            alias = prompt(`Enter a local name for this Stable Diffusion model:\n(Leave empty to use: "${baseName}")`);
                        } else {
                            alias = prompt(`Enter a friendly local name/alias for this model:\n(Leave empty to use: "${baseName}")`);
                        }
                        if (alias === null) return;  // user pressed Cancel

                        const ref = `hf://${repoName}/${file.filename}`;
                        // Always use 'huggingface' as source - puller auto-detects SD vs LLM from the file
                        pullModel(ref, 'huggingface', alias || baseName);
                    });

                    footer.appendChild(pullBtn);
                    card.appendChild(top);
                    card.appendChild(footer);
                    hfFilesList.appendChild(card);
                });
            }

        } catch (err) {
            if (hfFilesList) {
                hfFilesList.innerHTML = `<div style="text-align:center; color:var(--color-danger); padding:1rem;">Error listing files: ${err.message}</div>`;
            }
        }
    }

    async function pullModel(modelName, source, localName = "") {
        if (pullProgressContainer) pullProgressContainer.classList.remove('d-none');
        if (pullModelName) pullModelName.textContent = `Pulling: ${modelName}`;
        if (pullStatusBadge) {
            pullStatusBadge.className = 'badge badge-warning';
            pullStatusBadge.textContent = 'Running';
        }
        if (pullConsoleLog) {
            pullConsoleLog.textContent = `[System] Spawning download process for ${modelName}...\n`;
        }

        try {
            const res = await fetch('/api/models/pull', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    model: modelName,
                    source: source,
                    local_name: localName
                })
            });

            const data = await res.json();
            if (!res.ok) {
                if (pullStatusBadge) {
                    pullStatusBadge.className = 'badge badge-danger';
                    pullStatusBadge.textContent = 'Error';
                }
                if (pullConsoleLog) {
                    pullConsoleLog.textContent += `[Error] ${data.error || 'Failed to start download.'}\n`;
                }
            } else {
                if (pullConsoleLog) {
                    pullConsoleLog.textContent += `[System] ${data.message}\n`;
                }
            }
        } catch (err) {
            if (pullStatusBadge) {
                pullStatusBadge.className = 'badge badge-danger';
                pullStatusBadge.textContent = 'Error';
            }
            if (pullConsoleLog) {
                pullConsoleLog.textContent += `[Error] Connection error: ${err.message}\n`;
            }
        }
    }

    socket.on('pull_log', (data) => {
        if (pullConsoleLog) {
            pullConsoleLog.textContent += data.line + '\n';
            pullConsoleLog.scrollTop = pullConsoleLog.scrollHeight;
        }
    });

    socket.on('pull_status', (data) => {
        if (pullStatusBadge) {
            if (data.status === 'success') {
                pullStatusBadge.className = 'badge badge-success';
                pullStatusBadge.textContent = 'Success';
                showToast(`Model pull successful!`, 'success');
                loadModels();
            } else if (data.status === 'stopping') {
                pullStatusBadge.className = 'badge badge-warning';
                pullStatusBadge.textContent = 'Stopping...';
            } else if (data.status === 'stopped') {
                pullStatusBadge.className = 'badge badge-secondary';
                pullStatusBadge.textContent = 'Stopped';
                showToast('Download stopped', 'warning');
            } else if (data.status === 'cancelled') {
                pullStatusBadge.className = 'badge badge-danger';
                pullStatusBadge.textContent = 'Cancelled';
            } else {
                pullStatusBadge.className = 'badge badge-danger';
                pullStatusBadge.textContent = 'Failed';
                showToast(`Model pull failed: ${data.error || 'Unknown error'}`, 'error');
            }
        }
        if (pullConsoleLog) {
            pullConsoleLog.textContent += `\n[System] Pull process completed with status: ${data.status.toUpperCase()} ${data.error ? '(' + data.error + ')' : ''}\n`;
            pullConsoleLog.scrollTop = pullConsoleLog.scrollHeight;
        }
    });

    async function loadActivePulls() {
        try {
            const res = await fetch('/api/models/pulls/active');
            if (!res.ok) return;
            const data = await res.json();
            const pulls = data.active_pulls || {};
            
            const activeModels = Object.keys(pulls);
            if (activeModels.length > 0) {
                const activeModel = activeModels[0];
                const pullInfo = pulls[activeModel];
                
                if (pullProgressContainer) pullProgressContainer.classList.remove('d-none');
                if (pullModelName) pullModelName.textContent = `Pulling: ${pullInfo.model}`;
                
                let showButtons = false;
                if (pullStatusBadge) {
                    if (pullInfo.status === 'running') {
                        pullStatusBadge.className = 'badge badge-warning';
                        pullStatusBadge.textContent = 'Running';
                        showButtons = true;
                    } else if (pullInfo.status === 'stopping') {
                        pullStatusBadge.className = 'badge badge-warning';
                        pullStatusBadge.textContent = 'Stopping...';
                        showButtons = false;
                    } else if (pullInfo.status === 'cancelled') {
                        pullStatusBadge.className = 'badge badge-danger';
                        pullStatusBadge.textContent = 'Cancelled';
                        showButtons = false;
                    } else if (pullInfo.status === 'stopped') {
                        pullStatusBadge.className = 'badge badge-secondary';
                        pullStatusBadge.textContent = 'Stopped';
                        showButtons = false;
                    } else if (pullInfo.status === 'failed') {
                        pullStatusBadge.className = 'badge badge-danger';
                        pullStatusBadge.textContent = 'Failed';
                        showButtons = false;
                    } else {
                        pullStatusBadge.className = 'badge badge-secondary';
                        pullStatusBadge.textContent = pullInfo.status || 'Unknown';
                        showButtons = false;
                    }
                }
                if (btnPullStop) btnPullStop.style.display = showButtons ? 'inline-block' : 'none';
                if (btnPullCancel) btnPullCancel.style.display = showButtons ? 'inline-block' : 'none';
                if (pullConsoleLog) {
                    pullConsoleLog.textContent = pullInfo.logs.join('\n') + '\n';
                    pullConsoleLog.scrollTop = pullConsoleLog.scrollHeight;
                }
            } else {
                if (btnPullStop) btnPullStop.style.display = 'none';
                if (btnPullCancel) btnPullCancel.style.display = 'none';
                if (pullProgressContainer) pullProgressContainer.classList.add('d-none');
            }
        } catch (err) {
            console.error("Error loading active pulls:", err);
        }
    }

    // Stop/Cancel button handlers
    if (btnPullStop) {
        btnPullStop.addEventListener('click', async () => {
            try {
                const res = await fetch('/api/models/pulls/active', { method: 'GET' });
                if (!res.ok) return;
                const data = await res.json();
                const pulls = data.active_pulls || {};
                const activeModel = Object.keys(pulls)[0];
                if (!activeModel) return;
                
                const stopRes = await fetch(`/api/models/pulls/${encodeURIComponent(activeModel)}/stop`, {
                    method: 'POST'
                });
                if (stopRes.ok) {
                    if (pullStatusBadge) {
                        pullStatusBadge.className = 'badge badge-warning';
                        pullStatusBadge.textContent = 'Stopping...';
                    }
                    showToast('Stopping download...', 'warning');
                }
            } catch (err) {
                console.error("Failed to stop pull:", err);
                showToast('Failed to stop download', 'error');
            }
        });
    }

    if (btnPullCancel) {
        btnPullCancel.addEventListener('click', async () => {
            if (!confirm('Cancel this download and remove partial files?')) return;
            try {
                const res = await fetch('/api/models/pulls/active', { method: 'GET' });
                if (!res.ok) return;
                const data = await res.json();
                const pulls = data.active_pulls || {};
                const activeModel = Object.keys(pulls)[0];
                if (!activeModel) return;
                
                const cancelRes = await fetch(`/api/models/pulls/${encodeURIComponent(activeModel)}/cancel`, {
                    method: 'POST'
                });
                if (cancelRes.ok) {
                    if (pullStatusBadge) {
                        pullStatusBadge.className = 'badge badge-danger';
                        pullStatusBadge.textContent = 'Cancelled';
                    }
                    showToast('Download cancelled', 'warning');
                    setTimeout(loadActivePulls, 2000);
                }
            } catch (err) {
                console.error("Failed to cancel pull:", err);
                showToast('Failed to cancel download', 'error');
            }
        });
    }

    // Startup Tasks
    initCharts();

    // Model filter group toggle buttons (All / Local / Online)
    document.querySelectorAll('.model-filter-group').forEach(btn => {
        btn.addEventListener('click', () => {
            const type = btn.dataset.filterType;
            const group = btn.dataset.group;
            const all = filterAllModels[type] || [];
            const sel = filterSelection[type] || new Set();
            if (all.length === 0) return;

            if (group === 'all') {
                all.forEach(m => sel.add(m));
            } else {
                const groupModels = all.filter(m => (group === 'online') === isOnlineModelName(m));
                if (groupModels.length === 0) return;
                const allSelected = groupModels.every(m => sel.has(m));
                groupModels.forEach(m => { if (allSelected) sel.delete(m); else sel.add(m); });
            }
            syncFilterCheckboxUI(type);
            updateFilterGroupState(type);
            onModelFilterChange(type);
        });
    });

    loadModels();
    loadTests();
    loadBenchmarkGroups();
    wireResumeDeselect();
    loadSharedTests();
    loadMultistepWorkflows();
    loadModelProfiles();
    loadHistory();
    loadActivePulls();

    // SharedLLM test select-all / none
    const btnSelectAllSharedTests = document.getElementById('btn-select-all-shared-tests');
    const btnDeselectAllSharedTests = document.getElementById('btn-deselect-all-shared-tests');
    if (btnSelectAllSharedTests) {
        btnSelectAllSharedTests.addEventListener('click', () => {
            document.getElementById('shared-test-checkboxes')
                ?.querySelectorAll('input[type="checkbox"]')
                .forEach(cb => { cb.checked = true; });
        });
    }
    if (btnDeselectAllSharedTests) {
        btnDeselectAllSharedTests.addEventListener('click', () => {
            document.getElementById('shared-test-checkboxes')
                ?.querySelectorAll('input[type="checkbox"]')
                .forEach(cb => { cb.checked = false; });
        });
    }

    // MultiStep workflow select-all / none
    const btnSelectAllMultistep = document.getElementById('btn-select-all-multistep-workflows');
    const btnDeselectAllMultistep = document.getElementById('btn-deselect-all-multistep-workflows');
    if (btnSelectAllMultistep) {
        btnSelectAllMultistep.addEventListener('click', () => {
            document.getElementById('multistep-workflow-checkboxes')
                ?.querySelectorAll('input[type="checkbox"]')
                .forEach(cb => { cb.checked = true; });
        });
    }
    if (btnDeselectAllMultistep) {
        btnDeselectAllMultistep.addEventListener('click', () => {
            document.getElementById('multistep-workflow-checkboxes')
                ?.querySelectorAll('input[type="checkbox"]')
                .forEach(cb => { cb.checked = false; });
        });
    }

    // Benchmark group select-all / none
    const btnSelectAllGroups = document.getElementById('btn-select-all-groups');
    const btnDeselectAllGroups = document.getElementById('btn-deselect-all-groups');
    if (btnSelectAllGroups) {
        btnSelectAllGroups.addEventListener('click', () => {
            groupCheckboxes?.querySelectorAll('input[type="checkbox"]')
                .forEach(cb => { cb.checked = true; });
        });
    }
    if (btnDeselectAllGroups) {
        btnDeselectAllGroups.addEventListener('click', () => {
            groupCheckboxes?.querySelectorAll('input[type="checkbox"]')
                .forEach(cb => { cb.checked = false; });
        });
    }

    // Screenshot lightbox wiring
    const screenshotLightbox = document.getElementById('screenshot-lightbox');
    const screenshotLightboxImg = document.getElementById('screenshot-lightbox-img');
    const screenshotLightboxClose = document.getElementById('screenshot-lightbox-close');
    function openScreenshotLightbox(dataUrl) {
        if (!screenshotLightbox || !screenshotLightboxImg) return;
        screenshotLightboxImg.src = dataUrl;
        screenshotLightbox.classList.add('open');
    }
    function closeScreenshotLightbox() {
        if (!screenshotLightbox) return;
        screenshotLightbox.classList.remove('open');
        if (screenshotLightboxImg) screenshotLightboxImg.src = '';
    }
    if (screenshotLightboxClose) screenshotLightboxClose.addEventListener('click', closeScreenshotLightbox);
    if (screenshotLightbox) {
        screenshotLightbox.addEventListener('click', (e) => {
            if (e.target === screenshotLightbox) closeScreenshotLightbox();
        });
    }

    // Poll pull status periodically to keep UI in sync (skips while tab hidden)
    setInterval(() => {
        if (document.hidden) return;
        try {
            fetch('/api/models/pulls/active').then(res => res.ok ? res.json().then(data => {
                if (data.active_pulls && Object.keys(data.active_pulls).length > 0) {
                    loadActivePulls();
                }
            }) : null).catch(() => {});
        } catch(e) {}
    }, 3000);

    // Initial tab routing based on URL hash
    const initialHash = window.location.hash.substring(1);
    const validTabs = ['monitor', 'general', 'shared', 'tests', 'profiles', 'requests', 'docs', 'sd', 'audio'];
    if (validTabs.includes(initialHash)) {
        switchTab(initialHash);
    } else {
        switchTab('monitor'); // Start on System Monitor
    }

    function downloadFile(content, filename, contentType) {
        const blob = new Blob([content], { type: contentType });
        const a = document.createElement('a');
        a.href = URL.createObjectURL(blob);
        a.download = filename;
        a.click();
        URL.revokeObjectURL(a.href);
    }

    function extractCodeFromResponse(response) {
        if (!response) return '';
        const fence = response.match(/```(?:python|py)?\s*([\s\S]*?)```/i);
        if (fence && fence[1]) return fence[1].trim();
        return response.trim();
    }

    function sanitizeArtifactName(name) {
        return (name || 'model').replace(/[/:.]/g, '_');
    }

    async function saveArtifact(model, testId, content, type) {
        try {
            const resp = await fetch('/api/artifacts', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ model, test_id: testId, content, type })
            });
            const data = await resp.json();
            if (!resp.ok) throw new Error(data.error || 'Failed to save artifact');
            return data;
        } catch (err) {
            showToast(`Failed to save artifact: ${err.message}`, 'error');
            return null;
        }
    }

    function addArtifactButtons(container, model, testId, response) {
        const code = extractCodeFromResponse(response);
        if (!code) return;
        const btnRow = document.createElement('div');
        btnRow.style.cssText = 'display:flex; gap:0.5rem; margin-top:0.5rem;';

        const dlBtn = document.createElement('button');
        dlBtn.textContent = '⬇ Download .py';
        dlBtn.className = 'btn btn-secondary btn-sm';
        dlBtn.style.cssText = 'padding: 4px 12px; font-size: 0.75rem; cursor:pointer;';
        dlBtn.addEventListener('click', async (e) => {
            e.stopPropagation();
            const filename = `${sanitizeArtifactName(model)}__${testId}.py`;
            downloadFile(code, filename, 'text/x-python');
            showToast(`Downloading ${filename}`, 'success');
        });
        btnRow.appendChild(dlBtn);

        const hostBtn = document.createElement('button');
        hostBtn.textContent = '🖥 Host in Browser';
        hostBtn.className = 'btn btn-secondary btn-sm';
        hostBtn.style.cssText = 'padding: 4px 12px; font-size: 0.75rem; cursor:pointer;';
        hostBtn.addEventListener('click', async (e) => {
            e.stopPropagation();
            const result = await saveArtifact(model, testId, code, 'python');
            if (result && result.host_url) {
                showToast(`Artifact saved - opening hosted view`, 'success');
                window.open(result.host_url, '_blank');
            }
        });
        btnRow.appendChild(hostBtn);

        container.appendChild(btnRow);
    }

    function exportGeneralResults() {
        if (!currentResults || currentResults.length === 0) {
            showToast("No general benchmark results loaded to export", "error");
            return;
        }
        
        let md = `# General Benchmark Results Report\n\n`;
        md += `* **Generated At:** ${new Date().toLocaleString()}\n`;
        md += `* **Models Tested:** ${currentResults.length}\n\n`;
        
        currentResults.forEach(modelData => {
            const row = (typeof computeGeneralRow === 'function') ? computeGeneralRow(modelData) : null;
            md += `## Model: ${modelData.model}\n\n`;
            md += `* **Timestamp:** ${modelData.timestamp || 'N/A'}\n`;
            if (row) {
                md += `* **Overall Score:** ${row.score} / 100  (success ${row.success.toFixed(1)}%, ${row.tests} tests)\n`;
                md += `* **Avg Speed:** ${row.tps.toFixed(1)} TPS  |  **Avg TTFT:** ${row.ttft.toFixed(0)} ms  |  **Avg Tokens:** ${row.tokens.toFixed(0)}\n`;
            }
            if (modelData.performance_metrics) {
                const perf = modelData.performance_metrics;
                md += `* **Average Speed:** ${perf.avg_tps || 0} TPS\n`;
                md += `* **Average TTFT:** ${perf.avg_ttft_ms || 0} ms\n`;
                md += `* **Peak RAM:** ${perf.peak_ram_pct || 0}%\n`;
                md += `* **Peak VRAM:** ${perf.peak_vram_mb || 0} MB\n`;
            }
            md += `\n`;
            
            const categories = ['coding', 'reasoning', 'instruction', 'creative', 'home_automation', 'gamedev', 'appdev', 'linux_admin', 'webdev', 'database', 'cpp', 'java', 'debugging', 'logic', 'retrogames', 'threedprint', 'languages', 'tvdev', 'uiux', 'office', 'life', 'biblical', 'metacog'];
            categories.forEach(catKey => {
                const catStats = modelData[`category_${catKey}`];
                if (!catStats || !catStats.tests) return;
                
                md += `### Category: ${catKey.replace('_', ' ').toUpperCase()}\n\n`;
                md += `| Test Scenario | Status | Code Quality | Watermark | Latency | Speed |\n`;
                md += `| --- | --- | --- | --- | --- | --- |\n`;
                
                catStats.tests.forEach(test => {
                    const latVal = test.eval_duration && test.prompt_eval_duration ? (test.eval_duration + test.prompt_eval_duration) / 1e9 : test.latency;
                    const duration = latVal || 0;
                    const tps = (test.success && test.tokens_generated > 0 && duration > 0) ? (test.tokens_generated / duration) : 0;
                    const cq = test.code_quality ? test.code_quality.score : '-';
                    const wm = test.watermark ? test.watermark.score : '-';
                    
                    md += `| ${test.test_label} | ${test.success ? '✅ Success' : '❌ Fail'} | ${cq} | ${wm} | ${duration.toFixed(2)}s | ${tps > 0 ? tps.toFixed(1) + ' TPS' : '-'} |\n`;
                });
                md += `\n`;
                
                catStats.tests.forEach(test => {
                    md += `#### ${test.test_label} Detailed Log\n\n`;
                    md += `**Prompt (Question):**\n\`\`\`\n${test.prompt || '(none)'}\n\`\`\`\n\n`;
                    
                    let thinking = test.thinking || '';
                    let response = test.response || '';
                    if (!thinking && response) {
                        const match = response.match(/<(think|thinking)>([\s\S]*?)<\/\1>/i);
                        if (match) {
                            thinking = match[2].trim();
                        }
                    }
                    
                    if (thinking) {
                        md += `**Thinking Process:**\n\`\`\`\n${thinking}\n\`\`\`\n\n`;
                    }
                    
                    md += `**Response:**\n\`\`\`\n${response || test.error || '(no response)'}\n\`\`\`\n\n`;
                    md += `* * * * *\n\n`;
                });
            });
        });
        
        downloadFile(md, `general_benchmark_report_${new Date().toISOString().slice(0,10)}.md`, 'text/markdown');
    }

    function exportSharedResults() {
        if (!currentSharedResults || currentSharedResults.length === 0) {
            showToast("No SharedLLM benchmark results loaded to export", "error");
            return;
        }
        
        let md = `# SharedLLM Benchmark Results Report\n\n`;
        md += `* **Generated At:** ${new Date().toLocaleString()}\n`;
        md += `* **Models Tested:** ${currentSharedResults.length}\n\n`;
        
        currentSharedResults.forEach(modelData => {
            md += `## Model: ${modelData.model}\n\n`;
            md += `* **Timestamp:** ${modelData.timestamp || 'N/A'}\n`;
            if (modelData.performance_metrics) {
                const perf = modelData.performance_metrics;
                md += `* **Average Speed:** ${perf.avg_tps || 0} TPS\n`;
                md += `* **Average TTFT:** ${perf.avg_ttft_ms || 0} ms\n`;
            }
            md += `\n`;
            
            if (modelData.tasks) {
                md += `### Verification Audits\n\n`;
                md += `| Test Scope | Status | Latency | Payload Details |\n`;
                md += `| --- | --- | --- | --- |\n`;
                
                modelData.tasks.forEach(task => {
                    const latency = typeof task.latency === 'number' ? task.latency : 0;
                    const val = task.validation || {};
                    let details = '';
                    if (task.test_id === 'fast_path') {
                        details = `Intent: "${val.actual || ''}" (${val.correct_intent ? 'Correct' : 'Incorrect'})`;
                    } else if (task.test_id === 'tool_use') {
                        details = `Valid JSON: ${val.valid_json ? 'Yes' : 'No'} | Tool: "${(val.parsed && val.parsed.tool) || ''}"`;
                    } else if (task.test_id === 'code_gen') {
                        details = `Class: ${val.has_class ? 'Yes' : 'No'} | acquire: ${val.has_acquire ? 'Yes' : 'No'} | release: ${val.has_release ? 'Yes' : 'No'}`;
                    }
                    
                    md += `| ${task.test_label} | ${task.success ? '✅ Pass' : '❌ Fail'} | ${latency.toFixed(2)}s | ${details} |\n`;
                });
                md += `\n`;
                
                modelData.tasks.forEach(task => {
                    md += `#### ${task.test_label} Detailed Log\n\n`;
                    md += `**Prompt (Question):**\n\`\`\`\n${task.prompt || '(none)'}\n\`\`\`\n\n`;
                    
                    let thinking = task.thinking || '';
                    let response = task.response || '';
                    if (!thinking && response) {
                        const match = response.match(/<(think|thinking)>([\s\S]*?)<\/\1>/i);
                        if (match) {
                            thinking = match[2].trim();
                        }
                    }
                    
                    if (thinking) {
                        md += `**Thinking Process:**\n\`\`\`\n${thinking}\n\`\`\`\n\n`;
                    }
                    
                    md += `**Response / Generated Code:**\n\`\`\`python\n${response || task.error || '(no response)'}\n\`\`\`\n\n`;
                    md += `* * * * *\n\n`;
                });
            }
        });
        
        downloadFile(md, `shared_llm_benchmark_report_${new Date().toISOString().slice(0,10)}.md`, 'text/markdown');
    }

    const btnExportGeneral = document.getElementById('btn-export-general');
    if (btnExportGeneral) {
        btnExportGeneral.addEventListener('click', exportGeneralResults);
    }
    const btnExportCsv = document.getElementById('btn-export-csv');
    if (btnExportCsv) {
        btnExportCsv.addEventListener('click', () => {
            const url = '/api/benchmarks/export?format=csv';
            const a = document.createElement('a');
            a.href = url;
            a.download = `benchmarks_export_${new Date().toISOString().slice(0,10)}.csv`;
            document.body.appendChild(a);
            a.click();
            a.remove();
        });
    }
    const btnExportShared = document.getElementById('btn-export-shared');
    if (btnExportShared) {
        btnExportShared.addEventListener('click', exportSharedResults);
    }

    // API KEYS & PROVIDER SETTINGS CONTROLLER
    const btnOpenApiKeys = document.getElementById('btn-open-api-keys');
    const apiKeysModal = document.getElementById('api-keys-modal');
    const apiKeysClose = document.getElementById('api-keys-close');
    const btnCancelApiKeys = document.getElementById('btn-cancel-api-keys');
    const btnSaveApiKeys = document.getElementById('btn-save-api-keys');

    const inputAlpacaKey = document.getElementById('input-alpaca-key');
    const btnGenerateAlpacaToken = document.getElementById('btn-generate-alpaca-token');
    const btnCopyAlpacaToken = document.getElementById('btn-copy-alpaca-token');
    const btnClearAlpacaToken = document.getElementById('btn-clear-alpaca-token');
    const badgeStatusAlpaca = document.getElementById('badge-status-alpaca');
    const snippetApiKeyVal = document.getElementById('snippet-api-key-val');

    const inputOpenrouterKey = document.getElementById('input-openrouter-key');
    const btnTestOpenrouter = document.getElementById('btn-test-openrouter');
    const testResultOpenrouter = document.getElementById('test-result-openrouter');
    const badgeStatusOpenrouter = document.getElementById('badge-status-openrouter');

    const inputHuggingfaceToken = document.getElementById('input-huggingface-token');
    const btnTestHuggingface = document.getElementById('btn-test-huggingface');
    const testResultHuggingface = document.getElementById('test-result-huggingface');
    const badgeStatusHuggingface = document.getElementById('badge-status-huggingface');

    const inputCloudflareToken = document.getElementById('input-cloudflare-token');
    const inputCloudflareAccount = document.getElementById('input-cloudflare-account');
    const btnTestCloudflare = document.getElementById('btn-test-cloudflare');
    const testResultCloudflare = document.getElementById('test-result-cloudflare');
    const badgeStatusCloudflare = document.getElementById('badge-status-cloudflare');

    const inputOpencodeBaseUrl = document.getElementById('input-opencode-base-url');
    const inputOpencodeKey = document.getElementById('input-opencode-key');
    const btnTestOpencode = document.getElementById('btn-test-opencode');
    const testResultOpencode = document.getElementById('test-result-opencode');
    const badgeStatusOpencode = document.getElementById('badge-status-opencode');

    const inputGroqKey = document.getElementById('input-groq-key');
    const btnTestGroq = document.getElementById('btn-test-groq');
    const testResultGroq = document.getElementById('test-result-groq');
    const badgeStatusGroq = document.getElementById('badge-status-groq');

    const inputGeminiKey = document.getElementById('input-gemini-key');
    const btnTestGemini = document.getElementById('btn-test-gemini');
    const testResultGemini = document.getElementById('test-result-gemini');
    const badgeStatusGemini = document.getElementById('badge-status-gemini');

    function updateAlpacaSnippet(token) {
        if (snippetApiKeyVal) {
            snippetApiKeyVal.textContent = token ? `"${token}"` : '"YOUR_TOKEN_HERE"';
        }
    }

    async function loadApiKeysStatus() {
        try {
            const res = await fetch('/api/online/providers');
            if (!res.ok) return;
            const data = await res.json();
            const providers = data.providers || {};

            if (providers.alpaca) {
                if (inputAlpacaKey && providers.alpaca.masked_key) {
                    inputAlpacaKey.placeholder = providers.alpaca.masked_key;
                }
                if (badgeStatusAlpaca) {
                    badgeStatusAlpaca.className = providers.alpaca.configured ? 'badge badge-success' : 'badge badge-secondary';
                    badgeStatusAlpaca.textContent = providers.alpaca.configured ? 'Protected (Token Required)' : 'Public / No Auth';
                }
                updateAlpacaSnippet(providers.alpaca.masked_key || '');
            }

            if (providers.openrouter) {
                if (inputOpenrouterKey && providers.openrouter.masked_key) {
                    inputOpenrouterKey.placeholder = providers.openrouter.masked_key;
                }
                if (badgeStatusOpenrouter) {
                    badgeStatusOpenrouter.className = providers.openrouter.configured ? 'badge badge-success' : 'badge badge-secondary';
                    badgeStatusOpenrouter.textContent = providers.openrouter.configured ? 'Configured' : 'Not Configured';
                }
            }

            if (providers.huggingface) {
                if (inputHuggingfaceToken && providers.huggingface.masked_key) {
                    inputHuggingfaceToken.placeholder = providers.huggingface.masked_key;
                }
                if (badgeStatusHuggingface) {
                    badgeStatusHuggingface.className = providers.huggingface.configured ? 'badge badge-success' : 'badge badge-secondary';
                    badgeStatusHuggingface.textContent = providers.huggingface.configured ? 'Configured' : 'Not Configured';
                }
            }

            if (providers.cloudflare) {
                if (inputCloudflareToken && providers.cloudflare.masked_token) {
                    inputCloudflareToken.placeholder = providers.cloudflare.masked_token;
                }
                if (inputCloudflareAccount && providers.cloudflare.account_id) {
                    inputCloudflareAccount.value = providers.cloudflare.account_id;
                }
                if (badgeStatusCloudflare) {
                    badgeStatusCloudflare.className = providers.cloudflare.configured ? 'badge badge-success' : 'badge badge-secondary';
                    badgeStatusCloudflare.textContent = providers.cloudflare.configured ? 'Configured' : 'Not Configured';
                }
            }

            if (providers.opencode_zen) {
                if (inputOpencodeBaseUrl && providers.opencode_zen.base_url) {
                    inputOpencodeBaseUrl.value = providers.opencode_zen.base_url;
                }
                if (inputOpencodeKey && providers.opencode_zen.masked_key) {
                    inputOpencodeKey.placeholder = providers.opencode_zen.masked_key;
                }
                if (badgeStatusOpencode) {
                    badgeStatusOpencode.className = 'badge badge-success';
                    badgeStatusOpencode.textContent = 'Ready';
                }
            }

            if (providers.groq) {
                if (inputGroqKey && providers.groq.masked_key) {
                    inputGroqKey.placeholder = providers.groq.masked_key;
                }
                if (badgeStatusGroq) {
                    badgeStatusGroq.className = providers.groq.configured ? 'badge badge-success' : 'badge badge-secondary';
                    badgeStatusGroq.textContent = providers.groq.configured ? 'Configured' : 'Not Configured';
                }
            }

            if (providers.gemini) {
                if (inputGeminiKey && providers.gemini.masked_key) {
                    inputGeminiKey.placeholder = providers.gemini.masked_key;
                }
                if (badgeStatusGemini) {
                    badgeStatusGemini.className = providers.gemini.configured ? 'badge badge-success' : 'badge badge-secondary';
                    badgeStatusGemini.textContent = providers.gemini.configured ? 'Configured' : 'Not Configured';
                }
            }
        } catch (err) {
            console.error('Error loading provider credentials:', err);
        }
    }

    if (btnGenerateAlpacaToken) {
        btnGenerateAlpacaToken.addEventListener('click', async () => {
            try {
                const res = await fetch('/api/online/providers/alpaca/generate', { method: 'POST' });
                const data = await res.json();
                if (data.token && inputAlpacaKey) {
                    inputAlpacaKey.value = data.token;
                    updateAlpacaSnippet(data.token);
                    if (badgeStatusAlpaca) {
                        badgeStatusAlpaca.className = 'badge badge-warning';
                        badgeStatusAlpaca.textContent = 'Unsaved Token';
                    }
                    showToast('Generated new Alpaca Proxy token! Click Save to apply.', 'info');
                }
            } catch (err) {
                showToast(`Failed to generate token: ${err.message}`, 'error');
            }
        });
    }

    if (btnCopyAlpacaToken) {
        btnCopyAlpacaToken.addEventListener('click', () => {
            const token = inputAlpacaKey?.value.trim();
            if (token) {
                navigator.clipboard.writeText(token).then(() => {
                    showToast('Alpaca API token copied to clipboard!', 'success');
                }).catch(() => {
                    showToast('Failed to copy token to clipboard', 'error');
                });
            } else {
                showToast('No token set to copy', 'warning');
            }
        });
    }

    if (btnClearAlpacaToken) {
        btnClearAlpacaToken.addEventListener('click', () => {
            if (inputAlpacaKey) {
                inputAlpacaKey.value = '';
                updateAlpacaSnippet('');
                if (badgeStatusAlpaca) {
                    badgeStatusAlpaca.className = 'badge badge-secondary';
                    badgeStatusAlpaca.textContent = 'Public / No Auth';
                }
                showToast('Alpaca token cleared. Click Save to set proxy to public access.', 'info');
            }
        });
    }

    if (inputAlpacaKey) {
        inputAlpacaKey.addEventListener('input', () => {
            updateAlpacaSnippet(inputAlpacaKey.value.trim());
        });
    }

    if (btnOpenApiKeys) {
        btnOpenApiKeys.addEventListener('click', () => {
            if (apiKeysModal) apiKeysModal.classList.add('open');
            loadApiKeysStatus();
        });
    }

    function closeApiKeysModal() {
        if (apiKeysModal) apiKeysModal.classList.remove('open');
    }

    if (apiKeysClose) apiKeysClose.addEventListener('click', closeApiKeysModal);
    if (btnCancelApiKeys) btnCancelApiKeys.addEventListener('click', closeApiKeysModal);
    if (apiKeysModal) {
        apiKeysModal.addEventListener('click', (e) => {
            if (e.target === apiKeysModal) closeApiKeysModal();
        });
    }

    // Helper to run connection tests with visual feedback
    async function runProviderTest(provider, customKeys, resultElem, badgeElem) {
        if (!resultElem) return;
        resultElem.style.display = 'block';
        resultElem.style.color = '#93c5fd';
        resultElem.innerHTML = '⏳ Testing connection...';

        try {
            const res = await fetch('/api/online/providers/test', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ provider: provider, keys: customKeys })
            });
            const data = await res.json();

            if (data.success) {
                resultElem.style.color = '#4ade80';
                resultElem.innerHTML = `✅ ${data.message || 'Connection successful!'}`;
                if (badgeElem) {
                    badgeElem.className = 'badge badge-success';
                    badgeElem.textContent = 'Active';
                }
            } else {
                resultElem.style.color = '#f87171';
                resultElem.innerHTML = `❌ ${data.error || 'Connection failed'}`;
                if (badgeElem) {
                    badgeElem.className = 'badge badge-danger';
                    badgeElem.textContent = 'Failed';
                }
            }
        } catch (err) {
            resultElem.style.color = '#f87171';
            resultElem.innerHTML = `❌ Network error: ${err.message}`;
        }
    }

    if (btnTestOpenrouter) {
        btnTestOpenrouter.addEventListener('click', () => {
            const key = inputOpenrouterKey?.value.trim();
            runProviderTest('openrouter', key ? { openrouter_api_key: key } : {}, testResultOpenrouter, badgeStatusOpenrouter);
        });
    }

    if (btnTestHuggingface) {
        btnTestHuggingface.addEventListener('click', () => {
            const key = inputHuggingfaceToken?.value.trim();
            runProviderTest('huggingface', key ? { huggingface_token: key } : {}, testResultHuggingface, badgeStatusHuggingface);
        });
    }

    if (btnTestCloudflare) {
        btnTestCloudflare.addEventListener('click', () => {
            const token = inputCloudflareToken?.value.trim();
            const account = inputCloudflareAccount?.value.trim();
            const keys = {};
            if (token) keys.cloudflare_api_token = token;
            if (account) keys.cloudflare_account_id = account;
            runProviderTest('cloudflare', keys, testResultCloudflare, badgeStatusCloudflare);
        });
    }

    if (btnTestOpencode) {
        btnTestOpencode.addEventListener('click', () => {
            const url = inputOpencodeBaseUrl?.value.trim();
            const key = inputOpencodeKey?.value.trim();
            const keys = {};
            if (url) keys.opencode_zen_base_url = url;
            if (key) keys.opencode_zen_api_key = key;
            runProviderTest('opencode_zen', keys, testResultOpencode, badgeStatusOpencode);
        });
    }

    if (btnTestGroq) {
        btnTestGroq.addEventListener('click', () => {
            const key = inputGroqKey?.value.trim();
            runProviderTest('groq', key ? { groq_api_key: key } : {}, testResultGroq, badgeStatusGroq);
        });
    }

    if (btnTestGemini) {
        btnTestGemini.addEventListener('click', () => {
            const key = inputGeminiKey?.value.trim();
            runProviderTest('gemini', key ? { gemini_api_key: key } : {}, testResultGemini, badgeStatusGemini);
        });
    }

    if (btnSaveApiKeys) {
        btnSaveApiKeys.addEventListener('click', async () => {
            const payload = {};
            if (inputAlpacaKey) payload.alpaca_api_key = inputAlpacaKey.value.trim();
            if (inputOpenrouterKey?.value.trim()) payload.openrouter_api_key = inputOpenrouterKey.value.trim();
            if (inputHuggingfaceToken?.value.trim()) payload.huggingface_token = inputHuggingfaceToken.value.trim();
            if (inputCloudflareToken?.value.trim()) payload.cloudflare_api_token = inputCloudflareToken.value.trim();
            if (inputCloudflareAccount?.value.trim()) payload.cloudflare_account_id = inputCloudflareAccount.value.trim();
            if (inputOpencodeBaseUrl?.value.trim()) payload.opencode_zen_base_url = inputOpencodeBaseUrl.value.trim();
            if (inputOpencodeKey?.value.trim()) payload.opencode_zen_api_key = inputOpencodeKey.value.trim();
            if (inputGroqKey?.value.trim()) payload.groq_api_key = inputGroqKey.value.trim();
            if (inputGeminiKey?.value.trim()) payload.gemini_api_key = inputGeminiKey.value.trim();

            try {
                btnSaveApiKeys.disabled = true;
                btnSaveApiKeys.textContent = 'Saving...';
                const res = await fetch('/api/online/providers/save', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                const data = await res.json();
                if (data.success) {
                    showToast('Credentials and security settings saved successfully!', 'success');
                    closeApiKeysModal();
                    await loadModels();
                } else {
                    showToast(`Failed to save credentials: ${data.error}`, 'error');
                }
            } catch (err) {
                showToast(`Error saving credentials: ${err.message}`, 'error');
            } finally {
                btnSaveApiKeys.disabled = false;
                btnSaveApiKeys.textContent = '💾 Save Credentials';
            }
        });
    }

    // LIVE ONLINE MODELS EXPLORER CONTROLLER
    const btnOpenOnlineModels = document.getElementById('btn-open-online-models');
    const onlineModelsModal = document.getElementById('online-models-modal');
    const onlineModelsClose = document.getElementById('online-models-close');
    const btnCancelOnlineModels = document.getElementById('btn-cancel-online-models');
    const btnSaveOnlineSelection = document.getElementById('btn-save-online-selection');
    const btnFetchLiveModels = document.getElementById('btn-fetch-live-models');
    const onlineProviderFilter = document.getElementById('online-provider-filter');
    const onlineModelsSearchInput = document.getElementById('online-models-search-input');
    const onlineFreeOnlyToggle = document.getElementById('online-free-only-toggle');
    const onlineModelsResultsContainer = document.getElementById('online-models-results-container');
    const onlineResultsCount = document.getElementById('online-results-count');
    const onlineSelectedCount = document.getElementById('online-selected-count');
    const btnSelectAllOnline = document.getElementById('btn-select-all-online');
    const btnClearOnlineSelection = document.getElementById('btn-clear-online-selection');

    let currentOnlineCatalog = [];
    let selectedOnlineModelMap = new Map(); // id -> model object

    async function loadSelectedOnlineModels() {
        try {
            const res = await fetch('/api/online/models/selected');
            if (res.ok) {
                const data = await res.json();
                selectedOnlineModelMap.clear();
                (data.models || []).forEach(m => {
                    if (m.id) selectedOnlineModelMap.set(m.id, m);
                });
                updateOnlineSelectionCounter();
            }
        } catch (err) {
            console.error('Error loading selected online models:', err);
        }
    }

    function updateOnlineSelectionCounter() {
        if (onlineSelectedCount) {
            onlineSelectedCount.textContent = selectedOnlineModelMap.size;
        }
    }

    async function fetchLiveOnlineModels() {
        if (!onlineModelsResultsContainer) return;
        onlineModelsResultsContainer.innerHTML = `
            <div style="text-align:center; color:var(--text-muted); font-size:0.85rem; padding:2.5rem;">
                <div class="loader" style="width:28px; height:28px; border-width:3px; display:inline-block; margin-bottom:0.5rem;"></div>
                <div>Fetching live models from remote providers...</div>
            </div>`;

        const provider = onlineProviderFilter?.value || 'all';
        const query = onlineModelsSearchInput?.value.trim() || '';
        const freeOnly = onlineFreeOnlyToggle?.checked || false;

        try {
            const url = `/api/online/models/search?provider=${encodeURIComponent(provider)}&query=${encodeURIComponent(query)}&free_only=${freeOnly}`;
            const res = await fetch(url);
            const data = await res.json();
            currentOnlineCatalog = data.models || [];

            if (onlineResultsCount) onlineResultsCount.textContent = currentOnlineCatalog.length;
            renderOnlineModelResults(currentOnlineCatalog);
        } catch (err) {
            onlineModelsResultsContainer.innerHTML = `
                <div style="text-align:center; color:#f87171; padding:2rem;">
                    Failed to discover online models: ${err.message}
                </div>`;
        }
    }

    function renderOnlineModelResults(models) {
        if (!onlineModelsResultsContainer) return;
        onlineModelsResultsContainer.innerHTML = '';

        if (!models || models.length === 0) {
            onlineModelsResultsContainer.innerHTML = `
                <div style="text-align:center; color:var(--text-muted); font-size:0.85rem; padding:2.5rem;">
                    No models matched the current filter. Try adjusting your search query or provider.
                </div>`;
            return;
        }

        models.forEach(m => {
            const isChecked = selectedOnlineModelMap.has(m.id);
            const card = document.createElement('div');
            card.style.cssText = `
                display: flex;
                align-items: flex-start;
                gap: 0.75rem;
                background: #090d16;
                border: 1px solid ${isChecked ? 'var(--color-primary)' : 'var(--border-color)'};
                border-radius: 8px;
                padding: 0.65rem 0.85rem;
                transition: all 0.15s ease;
                cursor: pointer;
            `;

            const chk = document.createElement('input');
            chk.type = 'checkbox';
            chk.checked = isChecked;
            chk.style.marginTop = '0.25rem';
            chk.style.accentColor = 'var(--color-primary)';
            chk.style.cursor = 'pointer';

            const info = document.createElement('div');
            info.style.flex = '1';
            info.style.display = 'flex';
            info.style.flexDirection = 'column';
            info.style.gap = '0.25rem';

            const headerRow = document.createElement('div');
            headerRow.style.display = 'flex';
            headerRow.style.alignItems = 'center';
            headerRow.style.gap = '0.5rem';
            headerRow.style.flexWrap = 'wrap';

            const title = document.createElement('strong');
            title.style.color = 'white';
            title.style.fontSize = '0.84rem';
            title.textContent = m.label || m.name;

            const provBadge = document.createElement('span');
            provBadge.style.fontSize = '0.65rem';
            provBadge.style.padding = '1px 6px';
            provBadge.style.borderRadius = '4px';
            provBadge.style.fontWeight = '600';
            if (m.provider === 'openrouter') {
                provBadge.style.background = 'rgba(99, 102, 241, 0.2)';
                provBadge.style.color = '#818cf8';
                provBadge.textContent = 'OpenRouter';
            } else if (m.provider === 'huggingface') {
                provBadge.style.background = 'rgba(234, 179, 8, 0.2)';
                provBadge.style.color = '#fde047';
                provBadge.textContent = 'Hugging Face';
            } else if (m.provider === 'cloudflare') {
                provBadge.style.background = 'rgba(249, 115, 22, 0.2)';
                provBadge.style.color = '#fb923c';
                provBadge.textContent = 'Cloudflare';
            } else if (m.provider === 'groq') {
                provBadge.style.background = 'rgba(255, 87, 34, 0.2)';
                provBadge.style.color = '#ffab91';
                provBadge.textContent = 'Groq';
            } else if (m.provider === 'gemini') {
                provBadge.style.background = 'rgba(56, 189, 248, 0.2)';
                provBadge.style.color = '#7dd3fc';
                provBadge.textContent = 'Gemini';
            } else {
                provBadge.style.background = 'rgba(168, 85, 247, 0.2)';
                provBadge.style.color = '#c084fc';
                provBadge.textContent = 'OpenCode Zen';
            }

            const freeBadge = document.createElement('span');
            freeBadge.style.fontSize = '0.65rem';
            freeBadge.style.padding = '1px 6px';
            freeBadge.style.borderRadius = '4px';
            freeBadge.style.fontWeight = '700';
            const tierText = m.free_tier || (m.free ? 'FREE' : (m.pricing_label || 'Paid Tier'));
            if (m.free) {
                freeBadge.style.background = 'rgba(34, 197, 94, 0.2)';
                freeBadge.style.color = '#4ade80';
                freeBadge.textContent = tierText;
            } else {
                freeBadge.style.background = 'rgba(56, 189, 248, 0.2)';
                freeBadge.style.color = '#38bdf8';
                freeBadge.textContent = tierText;
            }

            const ctxBadge = document.createElement('span');
            ctxBadge.style.fontSize = '0.65rem';
            ctxBadge.style.color = 'var(--text-muted)';
            if (m.context_length) {
                const kCtx = Math.round(m.context_length / 1024);
                ctxBadge.textContent = `${kCtx > 0 ? kCtx + 'k' : m.context_length} ctx`;
            }

            headerRow.appendChild(title);
            headerRow.appendChild(provBadge);
            headerRow.appendChild(freeBadge);
            if (ctxBadge.textContent) headerRow.appendChild(ctxBadge);

            const modelIdText = document.createElement('div');
            modelIdText.style.fontFamily = 'monospace';
            modelIdText.style.fontSize = '0.72rem';
            modelIdText.style.color = '#94a3b8';
            modelIdText.textContent = m.id;

            const desc = document.createElement('div');
            desc.style.fontSize = '0.74rem';
            desc.style.color = 'var(--text-muted)';
            desc.style.lineHeight = '1.35';
            desc.textContent = m.description || '';

            info.appendChild(headerRow);
            info.appendChild(modelIdText);
            if (m.description) info.appendChild(desc);

            card.appendChild(chk);
            card.appendChild(info);

            function toggleSelection() {
                if (selectedOnlineModelMap.has(m.id)) {
                    selectedOnlineModelMap.delete(m.id);
                    chk.checked = false;
                    card.style.borderColor = 'var(--border-color)';
                } else {
                    selectedOnlineModelMap.set(m.id, m);
                    chk.checked = true;
                    card.style.borderColor = 'var(--color-primary)';
                }
                updateOnlineSelectionCounter();
            }

            chk.addEventListener('change', (e) => {
                e.stopPropagation();
                if (chk.checked) {
                    selectedOnlineModelMap.set(m.id, m);
                    card.style.borderColor = 'var(--color-primary)';
                } else {
                    selectedOnlineModelMap.delete(m.id);
                    card.style.borderColor = 'var(--border-color)';
                }
                updateOnlineSelectionCounter();
            });

            card.addEventListener('click', toggleSelection);
            onlineModelsResultsContainer.appendChild(card);
        });
    }

    if (btnOpenOnlineModels) {
        btnOpenOnlineModels.addEventListener('click', async () => {
            if (onlineModelsModal) onlineModelsModal.classList.add('open');
            await loadSelectedOnlineModels();
            await fetchLiveOnlineModels();
        });
    }

    function closeOnlineModelsModal() {
        if (onlineModelsModal) onlineModelsModal.classList.remove('open');
    }

    if (onlineModelsClose) onlineModelsClose.addEventListener('click', closeOnlineModelsModal);
    if (btnCancelOnlineModels) btnCancelOnlineModels.addEventListener('click', closeOnlineModelsModal);
    if (onlineModelsModal) {
        onlineModelsModal.addEventListener('click', (e) => {
            if (e.target === onlineModelsModal) closeOnlineModelsModal();
        });
    }

    if (btnFetchLiveModels) btnFetchLiveModels.addEventListener('click', fetchLiveOnlineModels);
    if (onlineProviderFilter) onlineProviderFilter.addEventListener('change', fetchLiveOnlineModels);
    if (onlineFreeOnlyToggle) onlineFreeOnlyToggle.addEventListener('change', fetchLiveOnlineModels);

    let onlineSearchDebounce = null;
    if (onlineModelsSearchInput) {
        onlineModelsSearchInput.addEventListener('input', () => {
            clearTimeout(onlineSearchDebounce);
            onlineSearchDebounce = setTimeout(fetchLiveOnlineModels, 300);
        });
        onlineModelsSearchInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') {
                clearTimeout(onlineSearchDebounce);
                fetchLiveOnlineModels();
            }
        });
    }

    if (btnSelectAllOnline) {
        btnSelectAllOnline.addEventListener('click', () => {
            currentOnlineCatalog.forEach(m => {
                if (m.id) selectedOnlineModelMap.set(m.id, m);
            });
            renderOnlineModelResults(currentOnlineCatalog);
            updateOnlineSelectionCounter();
        });
    }

    if (btnClearOnlineSelection) {
        btnClearOnlineSelection.addEventListener('click', () => {
            selectedOnlineModelMap.clear();
            renderOnlineModelResults(currentOnlineCatalog);
            updateOnlineSelectionCounter();
        });
    }

    if (btnSaveOnlineSelection) {
        btnSaveOnlineSelection.addEventListener('click', async () => {
            const modelsToSave = Array.from(selectedOnlineModelMap.values());
            try {
                btnSaveOnlineSelection.disabled = true;
                btnSaveOnlineSelection.textContent = 'Saving...';
                const res = await fetch('/api/online/models/selected', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ models: modelsToSave })
                });
                const data = await res.json();
                if (data.success) {
                    showToast(`Updated benchmark models (${modelsToSave.length} online models selected)`, 'success');
                    closeOnlineModelsModal();
                    await loadModels();
                } else {
                    showToast(`Failed to save model selection: ${data.error}`, 'error');
                }
            } catch (err) {
                showToast(`Error saving model selection: ${err.message}`, 'error');
            } finally {
                btnSaveOnlineSelection.disabled = false;
                btnSaveOnlineSelection.textContent = '✅ Save Benchmark Selection';
            }
        });
    }

    // Listen for hashchange event to handle back/forward navigation
    window.addEventListener('hashchange', () => {
        const hash = window.location.hash.substring(1);
        if (validTabs.includes(hash)) {
            switchTab(hash);
        }
    });

    // Leaderboard row selection shows the corresponding detailed results tab
    document.addEventListener('leaderboard:select', (e) => {
        const { model, isGeneral } = e.detail || {};
        if (!model) return;
        try {
            if (isGeneral) {
                renderDetailsSection(currentResults, model);
                const sel = document.getElementById('model-tabs');
                if (sel) {
                    sel.value = [...sel.options].find(o => o.value === model || o.value.includes(model))?.value || '';
                    sel.dispatchEvent(new Event('change'));
                }
            } else {
                renderSharedDetailsSection(currentSharedResults, model);
                const sel = document.getElementById('shared-model-tabs');
                if (sel) {
                    sel.value = [...sel.options].find(o => o.value === model || o.value.includes(model))?.value || '';
                    sel.dispatchEvent(new Event('change'));
                }
            }
        } catch (_) { /* ignore */ }
    });

    // Leaderboard view mode toggle: Top 10 (default) vs Show All
    document.querySelectorAll('.lb-show-top, .lb-show-all').forEach(btn => {
        btn.addEventListener('click', () => {
            const isGeneral = btn.dataset.lb === 'general';
            const mode = btn.dataset.mode;
            if (isGeneral) LB_MODE.general = mode; else LB_MODE.shared = mode;
            const group = btn.parentElement;
            group.querySelectorAll('.lb-show-top, .lb-show-all').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            if (isGeneral) {
                renderGeneralLeaderboard(currentResults);
            } else {
                renderSharedLeaderboard(currentSharedResults);
            }
        });
    });

    // Compare bar controls
    [['general', true], ['shared', false]].forEach(([key, isGeneral]) => {
        const clearBtn = document.getElementById(`${key}-lb-compare-clear`);
        if (clearBtn) clearBtn.addEventListener('click', () => {
            LB_SELECTION[key].clear();
            updateLbCompareBar(isGeneral);
            renderLeaderboardComparePanel(isGeneral);
            if (isGeneral) renderGeneralLeaderboard(currentResults);
            else renderSharedLeaderboard(currentSharedResults);
        });
        const goBtn = document.getElementById(`${key}-lb-compare-go`);
        if (goBtn) goBtn.addEventListener('click', () => {
            const panel = document.getElementById(`${key}-lb-compare-panel`);
            if (panel) panel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        });
    });
});

// ═══════════════════════════ AUDIO STUDIO ═══════════════════════════
// TTS (Kokoro-82M) + Music (MusicGen-small) via the audio-server service.
// Models load on demand server-side and unload when idle, so VRAM stays
// available for llama-server.

let _audioStatusTimer = null;

function initAudioStudio() {
    if (!document.getElementById('btn-tts-run').dataset.wired) {
        wireAudioStudio();
    }
    refreshAudioStatus();
    if (_audioStatusTimer) clearInterval(_audioStatusTimer);
    _audioStatusTimer = setInterval(() => {
        if (document.hidden) return;
        if (!activeTab || activeTab !== 'audio') {
            clearInterval(_audioStatusTimer);
            _audioStatusTimer = null;
            return;
        }
        refreshAudioStatus();
    }, 15000);
}

function wireAudioStudio() {
    const ttsBtn = document.getElementById('btn-tts-run');
    const musicBtn = document.getElementById('btn-music-run');
    const unloadBtn = document.getElementById('btn-audio-unload');
    const speed = document.getElementById('tts-speed');
    const speedVal = document.getElementById('tts-speed-val');
    if (!ttsBtn) return;
    ttsBtn.dataset.wired = '1';

    speed.addEventListener('input', () => { speedVal.textContent = parseFloat(speed.value).toFixed(2).replace(/0$/, '') + '×'; });

    ttsBtn.addEventListener('click', async () => {
        const text = document.getElementById('tts-text').value.trim();
        const meta = document.getElementById('tts-meta');
        if (!text) { meta.textContent = 'Enter some text first.'; return; }
        ttsBtn.disabled = true; ttsBtn.textContent = '⏳ Synthesizing…';
        meta.textContent = '';
        try {
            const resp = await fetch('/api/audio/tts', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text, voice: document.getElementById('tts-voice').value, speed: parseFloat(speed.value) }),
            });
            const data = await resp.json();
            if (!resp.ok || data.error) throw new Error(data.error || `HTTP ${resp.status}`);
            showAudio(data.audio_b64, `alpaca-tts-${data.meta.voice}.wav`);
            meta.textContent = `✅ ${data.meta.duration_s}s · RTF ${data.meta.rtf} · ${data.meta.chunks} chunk(s) · ${data.meta.elapsed_s}s elapsed`;
            refreshAudioStatus();
        } catch (err) {
            meta.textContent = '❌ ' + err.message;
        } finally {
            ttsBtn.disabled = false; ttsBtn.textContent = '🔊 Generate Speech';
        }
    });

    musicBtn.addEventListener('click', async () => {
        const prompt = document.getElementById('music-prompt').value.trim();
        const meta = document.getElementById('music-meta');
        if (!prompt) { meta.textContent = 'Describe the music you want first.'; return; }
        const seedRaw = document.getElementById('music-seed').value;
        const body = {
            prompt,
            duration_s: parseInt(document.getElementById('music-duration').value, 10),
        };
        if (seedRaw) body.seed = parseInt(seedRaw, 10);
        musicBtn.disabled = true; musicBtn.textContent = '⏳ Composing…';
        meta.textContent = '';
        try {
            const resp = await fetch('/api/audio/music', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body),
            });
            const data = await resp.json();
            if (!resp.ok || data.error) throw new Error(data.error || `HTTP ${resp.status}`);
            showAudio(data.audio_b64, 'alpaca-music.wav');
            meta.textContent = `✅ ${data.meta.duration_s}s clip · ${data.meta.elapsed_s}s to render · RTF ${data.meta.rtf}`;
            refreshAudioStatus();
        } catch (err) {
            meta.textContent = '❌ ' + err.message;
        } finally {
            musicBtn.disabled = false; musicBtn.textContent = '🎶 Generate Music';
        }
    });

    unloadBtn.addEventListener('click', async () => {
        unloadBtn.disabled = true;
        try {
            await fetch('/api/audio/unload', { method: 'POST' });
            refreshAudioStatus();
        } finally {
            unloadBtn.disabled = false;
        }
    });

    // Preset chips for music prompts
    fetch('/api/audio/status').then(r => r.json()).then(st => {
        const presets = (st.music && st.music.presets) || [];
        const box = document.getElementById('music-presets');
        box.innerHTML = '';
        presets.forEach(p => {
            const chip = document.createElement('button');
            chip.type = 'button';
            chip.className = 'music-preset-chip';
            chip.textContent = p.split(',')[0];
            chip.title = p;
            chip.style.cssText = 'font-size:0.68rem;padding:0.25rem 0.55rem;border-radius:999px;border:1px solid rgba(255,255,255,0.14);background:#090d16;color:#94a3b8;cursor:pointer;';
            chip.addEventListener('mouseenter', () => chip.style.color = '#f59e0b');
            chip.addEventListener('mouseleave', () => chip.style.color = '#94a3b8');
            chip.addEventListener('click', () => {
                document.getElementById('music-prompt').value = p;
            });
            box.appendChild(chip);
        });
    }).catch(() => {});
}

async function refreshAudioStatus() {
    const chip = document.getElementById('audio-status-chip');
    const voiceSel = document.getElementById('tts-voice');
    try {
        const st = await (await fetch('/api/audio/status')).json();
        if (st.status === 'offline') {
            chip.innerHTML = '<span style="color:#ef4444;">● audio-server offline</span>';
            return;
        }
        const free = st.vram_free_mb != null ? `${(st.vram_free_mb / 1024).toFixed(1)} GB free` : 'VRAM n/a';
        const parts = [
            `<span style="color:${st.tts.loaded ? '#34d399' : '#64748b'};">${st.tts.loaded ? '●' : '○'} Kokoro</span>`,
            `<span style="color:${st.music.loaded ? '#34d399' : '#64748b'};">${st.music.loaded ? '●' : '○'} MusicGen</span>`,
            `<span style="color:#64748b;">${free}</span>`,
        ];
        chip.innerHTML = parts.join(' &nbsp; ');
        if (voiceSel.options.length === 0 && Array.isArray(st.tts.voices)) {
            st.tts.voices.forEach(v => {
                const opt = document.createElement('option');
                opt.value = v; opt.textContent = v;
                voiceSel.appendChild(opt);
            });
        }
    } catch (err) {
        chip.innerHTML = '<span style="color:#ef4444;">● audio-server unreachable</span>';
    }
}

function showAudio(b64, filename) {
    document.getElementById('audio-output-empty').style.display = 'none';
    const box = document.getElementById('audio-player-box');
    const player = document.getElementById('audio-player');
    const dl = document.getElementById('audio-download');
    player.src = `data:audio/wav;base64,${b64}`;
    dl.href = player.src;
    dl.setAttribute('download', filename);
    box.style.display = 'block';
}
