document.addEventListener('DOMContentLoaded', () => {
    // Socket initialization
    const socket = io();

    // Navigation Tabs Elements
    const tabBtnMonitor = document.getElementById('tab-btn-monitor');
    const tabBtnGeneral = document.getElementById('tab-btn-general');
    const tabBtnShared = document.getElementById('tab-btn-shared');
    const tabBtnProfiles = document.getElementById('tab-btn-profiles');
    const tabBtnRequests = document.getElementById('tab-btn-requests');
    const tabBtnDocs = document.getElementById('tab-btn-docs');
    const tabBtnSd = document.getElementById('tab-btn-sd');
    const viewMonitor = document.getElementById('view-monitor');
    const viewGeneral = document.getElementById('view-general');
    const viewShared = document.getElementById('view-shared');
    const viewProfiles = document.getElementById('view-profiles');
    const viewRequests = document.getElementById('view-requests');
    const viewDocs = document.getElementById('view-docs');
    const viewImageStudio = document.getElementById('view-image-studio');

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
    const btnCancel = document.getElementById('btn-cancel');
     const selectAllBtn = document.getElementById('btn-select-all');
    const deselectAllBtn = document.getElementById('btn-deselect-all');
    const testCheckboxes = document.getElementById('test-checkboxes');
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
    let currentSharedResults = [];
    let monitorIntervalId = null;

    // Utility function to truncate long model names
    function truncateModelName(name) {
        if (!name) return '';
        if (name.length <= 25) return name;
        
        let cleanName = name.replace(/\.gguf$/i, '');
        
        if (cleanName.includes('--')) {
            const parts = cleanName.split('--');
            cleanName = parts[parts.length - 1];
        }
        
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
        tabBtnProfiles.classList.remove('active');
        tabBtnRequests.classList.remove('active');
        tabBtnDocs.classList.remove('active');
        tabBtnSd.classList.remove('active');

        // Hide views
        viewMonitor.classList.add('d-none');
        viewGeneral.classList.add('d-none');
        viewShared.classList.add('d-none');
        viewProfiles.classList.add('d-none');
        viewRequests.classList.add('d-none');
        viewDocs.classList.add('d-none');
        viewImageStudio.classList.add('d-none');
        
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
        }
        document.dispatchEvent(new CustomEvent('tabChanged', { detail: tabName }));
    }

    tabBtnMonitor.addEventListener('click', () => switchTab('monitor'));
    tabBtnGeneral.addEventListener('click', () => switchTab('general'));
    tabBtnShared.addEventListener('click', () => switchTab('shared'));
    tabBtnProfiles.addEventListener('click', () => switchTab('profiles'));
    tabBtnRequests.addEventListener('click', () => switchTab('requests'));
    tabBtnDocs.addEventListener('click', () => switchTab('docs'));
    tabBtnSd.addEventListener('click', () => switchTab('sd'));

    // ─── Image Studio (Stable Diffusion) ───────────────────────────────────
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

            // Populate Vision & Synthesis model selectors in Image-to-Prompt Assistant
            const visionSel = document.getElementById('sd-promptgen-vision-model');
            const synthSel = document.getElementById('sd-promptgen-synth-model');

            // Humanize router alias names (e.g. "qwen2.5-vl--3b" → "qwen2.5-vl:3b")
            const humanizeModelName = (id) => id.replace(/--/g, ':');

            if (visionSel) {
                // Vision: VL multimodal models from router + Ollama models
                try {
                    const res = await fetch('/api/models/text');
                    const data = await res.json();
                    const models = data.models || [];
                    visionSel.innerHTML = models.length
                        ? models.map(m => `<option value="${m}">${humanizeModelName(m)}</option>`).join('')
                        : '<option value="" disabled>No models available</option>';
                } catch(err) {
                    console.warn('Could not populate vision model selector:', err);
                }
            }

            if (synthSel) {
                // Synthesis: Ollama text-only models (VL excluded — optimised for image understanding)
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

    // ── Helper to render SD result cards with Download & Send to Canvas ─────
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

    // ── Image Studio Mode Tabs & Enhanced Presets / Canvas Logic ────────────
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
                alert('Please upload an image or PDF file first.');
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
            alert('✅ Extracted text transferred to Flyer Creator!');
        });
    }

    // ── Image-to-Prompt Assistant Logic ────────────────────────────────────
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
                alert('Please upload an image first.');
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
                alert('Please provide both the base image description and desired modifications.');
                return;
            }

            if (promptgenStatus) promptgenStatus.textContent = '✨ Synthesizing master prompt...';
            promptgenSynthBtn.disabled = true;

            try {
                const payload = {
                    base_description: baseDesc,
                    desired_changes: changes,
                    style_preset: preset
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
                alert('Please synthesize a master prompt first.');
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

    // ── Flyer Creator Synthesizer Logic ────────────────────────────────────
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

    // ── Photo Realism Retouch Controls ─────────────────────────────────────
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

    // ── Quick Inspiration Chips & Dimension Presets ────────────────────────
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

    // ── Drag & Drop Photo Upload Zone ──────────────────────────────────────
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

    // ── Interactive Canvas Text & Typography Studio Engine ─────────────────
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

        // SharedLLM latency comparison chart — uses legend instead of X-axis labels
        // to avoid diagonal/overflowing model name text on the axis
        sharedLatencyChart = new Chart(ctxSharedLatency, {
            type: 'bar',
            data: {
                labels: [],
                datasets: [
                    { label: 'FastPath (Intent)', data: [], backgroundColor: 'rgba(139, 92, 246, 0.6)', borderRadius: 4 },
                    { label: 'Librarian (Tool)', data: [], backgroundColor: 'rgba(6, 182, 212, 0.6)', borderRadius: 4 },
                    { label: 'Raven (Code Gen)', data: [], backgroundColor: 'rgba(16, 185, 129, 0.6)', borderRadius: 4 }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { 
                    legend: { position: 'top', labels: { boxWidth: 12, padding: 10 } },
                    tooltip: {
                        callbacks: {
                            title: function(context) {
                                const index = context[0].dataIndex;
                                const originalLabels = context[0].chart.data.originalLabels;
                                return (originalLabels && originalLabels[index]) ? originalLabels[index] : context[0].label;
                            },
                            afterTitle: function(context) {
                                // Show model index as subtitle so bars are identifiable
                                return `Model ${context[0].dataIndex + 1}`;
                            }
                        }
                    }
                },
                scales: {
                    // Hide X labels — models are in the tooltip title and legend
                    x: {
                        ticks: { display: false },
                        grid: { display: false }
                    },
                    y: { beginAtZero: true, title: { display: true, text: 'Latency (seconds)' } }
                }
            }
        });

        // SharedLLM AST Compliance Breakdown chart
        sharedAstChart = new Chart(ctxSharedAst, {
            type: 'bar',
            data: {
                labels: ['Class Match', 'Acquire Method', 'Release Method', 'Completed Code'],
                datasets: []
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { position: 'top', labels: { boxWidth: 12 } } },
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

    // Modal Control — supports thinking models with <think>...</think> blocks
    const modalThinking = document.getElementById('modal-thinking');
    const modalThinkingSection = document.getElementById('modal-thinking-section');

    function openModal(prompt, rawResponse) {
        modalPrompt.textContent = prompt || 'No prompt recorded.';

        // Parse out <think>...</think> block if present with error handling
        let thinkingContent = '';
        let actualResponse = '';
        let thinkingBlockFound = false;
        
        if (rawResponse) {
            try {
                const thinkMatch = rawResponse.match(/<think>([\s\S]*?)<\/think>/i);
                if (thinkMatch) {
                    thinkingBlockFound = true;
                    thinkingContent = thinkMatch[1].trim();
                    // Actual response is everything after the closing </think>
                    const remainingContent = rawResponse.replace(/<think>[\s\S]*?<\/think>/i, '');
                    actualResponse = remainingContent.trim();
                } else {
                    // No thinking block, use entire response
                    actualResponse = rawResponse.trim();
                }
            } catch (error) {
                // If parsing fails, use the raw response as-is
                actualResponse = rawResponse ? rawResponse.trim() : '';
                console.warn('Error parsing thinking block:', error);
            }
        }

        // Fallback for thinking block
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

    deselectAllBtn.addEventListener('click', () => {
        const boxes = modelCheckboxes.querySelectorAll('input[type="checkbox"]');
        boxes.forEach(box => box.checked = false);
        logToTerminal("All models deselected");
    });

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

    function getSelectedSharedTests() {
        const container = document.getElementById('shared-test-checkboxes');
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

            // Proxy is reachable — check if sd-server is still booting (model swap in progress)
            if (data.online && !data.sd_server_healthy) {
                sdStatusCard.style.borderLeftColor = 'var(--color-warning)';
                sdConnectionTitle.textContent = "Stable Diffusion Backend [Loading...]";
                sdConnectionSubtitle.textContent = "SD-Server is starting up — auto-loading a model. This may take up to 60 seconds.";
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
            // Swallow fetch errors (timeout / network) — proxy may be momentarily busy during model swap
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
            if (downloadLogsBtn) downloadLogsBtn.style.display = 'inline-block';
            
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
                
                // Determine context length — prefer running_settings ctx-size over
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
            monitorIntervalId = setInterval(pollProxyStatus, 2000);
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

    function startRequestsPolling() {
        if (!requestsIntervalId) {
            setupRequestsControls();
            pollRequestsStatus(); // immediate load
            requestsIntervalId = setInterval(pollRequestsStatus, 2000);
        }
    }

    function stopRequestsPolling() {
        if (requestsIntervalId) {
            clearInterval(requestsIntervalId);
            requestsIntervalId = null;
        }
    }

    function showToast(message, type = 'info') {
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
            fontFamily: 'system-ui, sans-serif', boxShadow: '0 4px 12px rgba(0,0,0,0.3)'
        });
        toast.textContent = message;
        document.body.appendChild(toast);
        setTimeout(() => {
            toast.style.opacity = '0';
            toast.style.transition = 'opacity 0.3s ease';
            setTimeout(() => toast.remove(), 300);
        }, 3000);
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

        const timeSpan = document.createElement('span');
        timeSpan.style.cssText = 'font-size: 0.65rem; color: var(--text-muted);';
        
        if (isActive) {
            const elapsed = Math.round(Date.now() / 1000 - req.started_at);
            timeSpan.textContent = `Running: ${elapsed}s`;
        } else {
            timeSpan.textContent = `${req.duration_seconds || 0}s`;
        }

        headerDiv.appendChild(typeBadge);
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
                responseEl.textContent = req.response || (req.completed_at ? '(No Output)' : 'Generating output...');
                if (isNearBottom || !req.completed_at) {
                    responseEl.scrollTop = responseEl.scrollHeight;
                }
            }
        }
    }

    let requestsControlsSetup = false;
    function setupRequestsControls() {
        if (requestsControlsSetup) return;
        requestsControlsSetup = true;

        const timezoneSelect = document.getElementById('requests-timezone-select');
        if (timezoneSelect && timezoneSelect.options.length === 0) {
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

            // Restore selection from localStorage
            const savedTz = localStorage.getItem('alpaca_requests_timezone');
            if (savedTz) {
                timezoneSelect.value = savedTz;
            }
            
            timezoneSelect.addEventListener('change', () => {
                localStorage.setItem('alpaca_requests_timezone', timezoneSelect.value);
                renderRequestsLists(lastActiveList, lastCompletedList);
                if (selectedRequestId && allRequestsMap[selectedRequestId]) {
                    updateInspectorDetails(allRequestsMap[selectedRequestId]);
                }
            });
        }

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
    }

    // Fetch and display available models in configurations sidebar
    async function loadModels() {
        try {
            logToTerminal("Fetching available models...");
            const res = await fetch('/api/models');
            const data = await res.json();
            
            availableModels = data.models || [];
            modelCheckboxes.innerHTML = '';
            
            if (availableModels.length === 0) {
                modelCheckboxes.innerHTML = `<div style="color:var(--text-muted);font-size:0.8rem;padding:0.5rem;">No models detected</div>`;
                return;
            }

            availableModels.forEach((model, idx) => {
                const item = document.createElement('label');
                item.className = 'checkbox-item';
                
                const input = document.createElement('input');
                input.type = 'checkbox';
                input.value = model;
                input.checked = false;
                
                const span = document.createElement('span');
                span.className = 'checkbox-label';
                span.textContent = model;
                
                item.appendChild(input);
                item.appendChild(span);
                modelCheckboxes.appendChild(item);
            });
            
            populateModelSwitcher(availableModels);
            logToTerminal(`Discovered ${availableModels.length} models from servers`, 'success');
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
            
            logToTerminal(`Loaded ${tests.length} benchmark test cases`, 'success');
        } catch (err) {
            logToTerminal(`Failed to load tests: ${err.message}`, 'error');
        }
    }

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

            await Promise.all(results.map(async (result) => {
                try {
                    const dr = await fetch(`/api/results/${result.filename}`);
                    const detail = await dr.json();
                    if (result.type === 'shared_llm') {
                        (detail.results || []).forEach(modelRecord => {
                            if (!sharedResults.find(r => r.model === modelRecord.model)) {
                                sharedResults.push(modelRecord);
                            }
                        });
                        if (!latestSharedData || result.filename > (latestSharedData._filename || '')) {
                            latestSharedData = { ...detail, _filename: result.filename };
                        }
                    } else {
                        (detail.results || []).forEach(modelRecord => {
                            if (!generalResults.find(r => r.model === modelRecord.model)) {
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

            results.forEach(result => {
                const item = document.createElement('div');
                item.className = 'history-item active'; // all files contribute to merged view
                
                const title = document.createElement('div');
                title.className = 'history-title';
                title.textContent = result.filename.replace('benchmarks_', '').replace('shared_llm_', '').replace('.json', '');
                
                const meta = document.createElement('div');
                meta.className = 'history-meta';
                
                const runTypeBadge = result.type === 'shared_llm' ? 'SharedLLM' : 'General';
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
                deleteBtn.style.cssText = 'position: absolute; right: 8px; top: 8px; cursor: pointer; font-size: 0.65rem; color: #ef4444; background: rgba(239, 68, 68, 0.1); border: 1px solid rgba(239, 68, 68, 0.2); padding: 2px 6px; border-radius: 4px; font-weight: 600; text-transform: uppercase; transition: all 0.2s; z-index: 100;';
                deleteBtn.title = 'Delete result file';
                deleteBtn.addEventListener('mouseenter', () => {
                    deleteBtn.style.background = '#ef4444';
                    deleteBtn.style.color = '#ffffff';
                });
                deleteBtn.addEventListener('mouseleave', () => {
                    deleteBtn.style.background = 'rgba(239, 68, 68, 0.1)';
                    deleteBtn.style.color = '#ef4444';
                });
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

    // Load details of selected file
    async function loadBenchmarkDetail(filename, type) {
        try {
            logToTerminal(`Loading benchmark file: ${filename}...`);
            const res = await fetch(`/api/results/${filename}`);
            const data = await res.json();
            
            if (type === 'shared_llm') {
                currentSharedResults = data.results || [];
                logToTerminal(`Loaded SharedLLM results for ${currentSharedResults.length} models`, 'success');
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

    // Calculate General Overview card metrics
    function updateOverviewMetrics(fullData) {
        const results = fullData.results || [];
        if (results.length === 0) return;

        let totalTps = 0, tpsCount = 0;
        let totalTtft = 0, ttftCount = 0;
        let totalPassed = 0, totalRun = 0;

        results.forEach(m => {
            const categories = ['coding', 'reasoning', 'instruction', 'creative', 'home_automation'];
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
        const results = fullData.results || [];
        if (results.length === 0) return;

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

    // Render General Charts
    function renderChartsFromData(results) {
        if (results.length === 0) return;
        const models = results.map(r => r.model);
        const displayNames = models.map(model => truncateModelName(model));
        const colors = [
            { bg: 'rgba(139, 92, 246, 0.6)', border: '#8b5cf6' },
            { bg: 'rgba(6, 182, 212, 0.6)',   border: '#06b6d4' },
            { bg: 'rgba(16, 185, 129, 0.6)',  border: '#10b981' },
            { bg: 'rgba(245, 158, 11, 0.6)',  border: '#f59e0b' },
            { bg: 'rgba(59, 130, 246, 0.6)',  border: '#3b82f6' },
        ];

        // TPS chart — one dataset per model so legend renders model names cleanly
        const tpsDatasets = results.map((r, idx) => {
            let totalTps = 0, tpsCount = 0;
            const categories = ['coding', 'reasoning', 'instruction', 'creative', 'home_automation'];
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
                backgroundColor: colors[idx % colors.length].bg,
                borderColor: colors[idx % colors.length].border,
                borderWidth: 1,
                borderRadius: 4
            };
        });
        tpsChart.data.labels = [''];
        tpsChart.data.originalLabels = [''];
        tpsChart.data.datasets = tpsDatasets;
        tpsChart.update();

        // TTFT chart — one dataset per model
        const ttftDatasets = results.map((r, idx) => {
            let totalTtft = 0, ttftCount = 0;
            const categories = ['coding', 'reasoning', 'instruction', 'creative', 'home_automation'];
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
                backgroundColor: colors[idx % colors.length].bg,
                borderColor: colors[idx % colors.length].border,
                borderWidth: 1,
                borderRadius: 4
            };
        });
        ttftChart.data.labels = [''];
        ttftChart.data.originalLabels = [''];
        ttftChart.data.datasets = ttftDatasets;
        ttftChart.update();

        // Category success rate chart — one dataset per model, 5 categories
        const catColors = ['rgba(139, 92, 246, 0.7)', 'rgba(6, 182, 212, 0.7)', 'rgba(16, 185, 129, 0.7)', 'rgba(245, 158, 11, 0.7)', 'rgba(59, 130, 246, 0.7)'];
        const datasets = [];
        results.forEach((r, idx) => {
            const categories = ['coding', 'reasoning', 'instruction', 'creative', 'home_automation'];
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
                backgroundColor: catColors[idx % catColors.length],
                borderRadius: 4,
                borderWidth: 0
            });
        });

        categoryChart.data.datasets = datasets;
        categoryChart.update();
    }

    // Render SharedLLM Charts
    function renderSharedChartsFromData(results) {
        if (results.length === 0) return;
        const models = results.map(r => r.model);
        
        // 1. Latency Chart per tier
        const fastpathLats = [];
        const librarianLats = [];
        const ravenLats = [];
        
        results.forEach(m => {
            let fp = 0, lib = 0, rav = 0;
            m.tasks.forEach(t => {
                if (t.test_id === 'fast_path') fp = t.latency;
                else if (t.test_id === 'tool_use') lib = t.latency;
                else if (t.test_id === 'code_gen') rav = t.latency;
            });
            fastpathLats.push(fp);
            librarianLats.push(lib);
            ravenLats.push(rav);
        });

        const displayModels = models.map(m => truncateModelName(m));

        sharedLatencyChart.data.labels = displayModels;
        sharedLatencyChart.data.originalLabels = models;
        sharedLatencyChart.data.datasets[0].data = fastpathLats;
        sharedLatencyChart.data.datasets[1].data = librarianLats;
        sharedLatencyChart.data.datasets[2].data = ravenLats;
        sharedLatencyChart.update();

        // 2. AST compliance rates
        const datasets = [];
        const colors = ['rgba(139, 92, 246, 0.7)', 'rgba(6, 182, 212, 0.7)', 'rgba(16, 185, 129, 0.7)', 'rgba(245, 158, 11, 0.7)'];
        
        results.forEach((m, idx) => {
            let classMatch = 0, acquireMatch = 0, releaseMatch = 0, isComplete = 0;
            m.tasks.forEach(t => {
                if (t.test_id === 'code_gen') {
                    const v = t.validation || {};
                    classMatch = v.has_class ? 100 : 0;
                    acquireMatch = v.has_acquire ? 100 : 0;
                    releaseMatch = v.has_release ? 100 : 0;
                    isComplete = v.is_complete ? 100 : 0;
                }
            });

            datasets.push({
                label: truncateModelName(m.model),
                originalLabel: m.model,
                data: [classMatch, acquireMatch, releaseMatch, isComplete],
                backgroundColor: colors[idx % colors.length],
                borderRadius: 4
            });
        });

        sharedAstChart.data.datasets = datasets;
        sharedAstChart.update();
    }

    // Render General Details section
    function renderDetailsSection(results) {
        modelTabs.innerHTML = '';
        detailedResultsBody.innerHTML = '';

        if (results.length === 0) {
            detailedResultsBody.innerHTML = `<tr><td colspan="6" style="text-align:center;color:var(--text-muted);">No detailed data available.</td></tr>`;
            return;
        }

        results.forEach((modelData, idx) => {
            const btn = document.createElement('button');
            btn.className = 'tab-btn';
            const displayName = truncateModelName(modelData.model);
            btn.textContent = displayName;
            if (idx === 0) btn.classList.add('active');

            btn.addEventListener('click', () => {
                document.querySelectorAll('#model-tabs .tab-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                renderModelDetailedTable(modelData);
            });
            modelTabs.appendChild(btn);
        });

        renderModelDetailedTable(results[0]);
    }

    function renderModelDetailedTable(modelData) {
        detailedResultsBody.innerHTML = '';
        const categories = ['coding', 'reasoning', 'instruction', 'creative', 'home_automation'];
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
                    openModal(test.prompt || '(no prompt recorded)', test.response || errorMsg);
                });
                tdView.appendChild(promptLink);

                tr.appendChild(tdCat);
                tr.appendChild(tdLabel);
                tr.appendChild(tdStatus);
                tr.appendChild(tdLastRun);
                tr.appendChild(tdLat);
                tr.appendChild(tdSpeed);
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
                        codeBlock.textContent = test.response || test.error || 'No response recorded';
                        
                        flex.appendChild(title);
                        flex.appendChild(codeBlock);
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
            detailedResultsBody.innerHTML = `<tr><td colspan="6" style="text-align:center;color:var(--text-muted);">No test logs for this model</td></tr>`;
        }
    }

    // Render SharedLLM Details Section
    function renderSharedDetailsSection(results) {
        sharedModelTabs.innerHTML = '';
        sharedDetailedResultsBody.innerHTML = '';

        if (results.length === 0) {
            sharedDetailedResultsBody.innerHTML = `<tr><td colspan="6" style="text-align:center;color:var(--text-muted);">No detailed data available.</td></tr>`;
            return;
        }

        results.forEach((modelData, idx) => {
            const btn = document.createElement('button');
            btn.className = 'tab-btn';
            const displayName = truncateModelName(modelData.model);
            btn.textContent = displayName;
            if (idx === 0) btn.classList.add('active');

            btn.addEventListener('click', () => {
                document.querySelectorAll('#shared-model-tabs .tab-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                renderSharedModelDetailedTable(modelData);
            });
            sharedModelTabs.appendChild(btn);
        });

        renderSharedModelDetailedTable(results[0]);
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
            
            if (task.test_id === 'fast_path') {
                tdPayload.textContent = `Intent parsed: "${val.actual || ''}" (${val.correct_intent ? 'Correct' : 'Incorrect'})`;
            } else if (task.test_id === 'tool_use') {
                tdPayload.textContent = `Valid JSON: ${val.valid_json ? 'Yes' : 'No'} | Tool: "${val.parsed?.tool || ''}"`;
            } else if (task.test_id === 'code_gen') {
                const checks = document.createElement('div');
                checks.className = 'ast-check-list';
                
                const classBadge = document.createElement('span');
                classBadge.className = `ast-check-badge ${val.has_class ? 'checked' : 'failed'}`;
                classBadge.textContent = val.has_class ? '✓ Class Defined' : '✗ No Class';
                
                const acqBadge = document.createElement('span');
                acqBadge.className = `ast-check-badge ${val.has_acquire ? 'checked' : 'failed'}`;
                acqBadge.textContent = val.has_acquire ? '✓ acquire()' : '✗ no acquire()';
                
                const relBadge = document.createElement('span');
                relBadge.className = `ast-check-badge ${val.has_release ? 'checked' : 'failed'}`;
                relBadge.textContent = val.has_release ? '✓ release()' : '✗ no release()';
                
                checks.appendChild(classBadge);
                checks.appendChild(acqBadge);
                checks.appendChild(relBadge);
                tdPayload.appendChild(checks);
            }

            const tdView = document.createElement('td');
            const promptLink = document.createElement('span');
            promptLink.className = 'prompt-text';
            promptLink.textContent = 'View Code / Payload';
            promptLink.addEventListener('click', () => {
                const details = task.error ? `Error: ${task.error}` : '';
                openModal(task.prompt, task.response || details);
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
                    flex.appendChild(codeBlock);
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

    // Trigger SharedLLM Benchmarks
    btnRunShared.addEventListener('click', () => triggerBenchmark('/api/run/shared_llm'));

    async function triggerBenchmark(endpoint) {
        const selected = getSelectedModels();
        if (selected.length === 0) {
            alert('Please select at least one model to benchmark.');
            return;
        }

        const isShared = endpoint.endsWith('shared_llm');
        const selectedTests = isShared ? getSelectedSharedTests() : getSelectedTests();
        if (!isShared && selectedTests.length === 0) {
            alert('Please select at least one test case to run.');
            return;
        }
        if (isShared && selectedTests.length === 0) {
            alert('Please select at least one SharedLLM task to run.');
            return;
        }

        btnRun.disabled = true;
        btnRunShared.disabled = true;
        
        if (isShared) {
            btnRunShared.innerHTML = `<span class="loader"></span> Starting...`;
        } else {
            btnRun.innerHTML = `<span class="loader"></span> Starting...`;
        }
        
        const termTarget = isShared ? 'shared' : 'general';
        
        try {
            logToTerminal(`Initiating benchmark for models: ${selected.join(', ')}...`, 'info', termTarget);
            
            const payload = {
                models: selected,
                use_proxy: (benchmarkMode === 'proxy')
            };
            if (!isShared) {
                payload.test_ids = selectedTests;
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
                alert(data.error);
                setRunnerState('idle');
                return;
            }

            if (!res.ok) {
                const errorText = await res.text();
                throw new Error(errorText || 'Server error starting benchmark');
            }

            logToTerminal("Benchmark pipeline initialized successfully.", 'success', termTarget);
            setRunnerState('running');
            
            // Ensure SocketIO is connected before switching tabs to receive progress events
            if (socket.connected) {
                if (isShared) {
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
                if (isShared) {
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
            btnRun.innerHTML = 'Run General';
            btnRunShared.innerHTML = 'Run SharedLLM';
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
        logToTerminal(`Benchmark started: ${data.total_tests} tests across ${data.total_models} models.`, "info", term);
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

    socket.on('test_complete', (data) => {
        const res = data.result;
        const pct = data.progress.percentage;
        progressPercent.textContent = `${pct}%`;
        progressBarFill.style.width = `${pct}%`;
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
    });

    socket.on('benchmark_cancelled', (data) => {
        logToTerminal(`Benchmark run cancelled: ${data.message}`, "warn");
        logToTerminal(`Benchmark run cancelled: ${data.message}`, "warn", "shared");
        setRunnerState('idle');
    });

    socket.on('benchmark_error', (data) => {
        logToTerminal(`Critical runner error: ${data.error}`, "error");
        logToTerminal(`Critical runner error: ${data.error}`, "error", "shared");
        alert(`Runner Error: ${data.error}`);
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
                    'n-cpu-moe': profileEditForm.elements['n-cpu-moe'].value || null
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
                alert('Please select a model profile section to save.');
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
                    alert('Failed to save settings: ' + data.error);
                } else {
                    modelProfiles[section] = Object.assign({}, modelProfiles[section], settings);
                    alert('Settings saved successfully!');
                }
            })
            .catch(err => {
                console.error("Save profile error:", err);
                alert('Failed to save profile settings');
            });
        });

        if (btnRestartServices) {
            btnRestartServices.addEventListener('click', () => {
                const section = profileSectionSelect.value;
                if (!section) {
                    alert('Please select a model profile section first.');
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
                        alert('Restart command failed: ' + data.error);
                    } else {
                        showToast('Backend restart sequence initiated. Reloading system monitor in 5 seconds...', 'success');
                        setTimeout(() => {
                            window.location.reload();
                        }, 5000);
                    }
                })
                .catch(err => {
                    console.error("Save and restart error:", err);
                    alert('Failed to save settings or restart backend: ' + err.message);
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
                alert('Invalid profile name.');
                return;
            }
            
            if (modelProfiles[sanitized]) {
                alert(`Profile section [${sanitized}] already exists.`);
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
                    alert('Failed to create profile: ' + data.error);
                } else {
                    modelProfiles[sanitized] = {};
                    
                    const select = document.getElementById('profile-section-select');
                    const opt = document.createElement('option');
                    opt.value = sanitized;
                    opt.textContent = sanitized;
                    select.appendChild(opt);
                    select.value = sanitized;
                    
                    profileSectionSelect.dispatchEvent(new Event('change'));
                    alert(`Profile section [${sanitized}] created successfully!`);
                }
            })
            .catch(err => {
                console.error("Create profile error:", err);
                alert('Failed to create profile section');
            });
        });
    }

    // Wire Delete Profile
    const btnDeleteProfile = document.getElementById('btn-delete-profile');
    if (btnDeleteProfile) {
        btnDeleteProfile.addEventListener('click', () => {
            const section = profileSectionSelect.value;
            if (!section) {
                alert('Please select a profile section to delete.');
                return;
            }
            if (section === '*') {
                alert('Cannot delete global defaults section [*].');
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
                    alert('Failed to delete profile: ' + data.error);
                } else {
                    delete modelProfiles[section];
                    
                    const option = profileSectionSelect.querySelector(`option[value="${section}"]`);
                    if (option) option.remove();
                    profileSectionSelect.value = '';
                    profileSectionSelect.dispatchEvent(new Event('change'));
                    
                    alert(`Profile section [${section}] deleted successfully!`);
                }
            })
            .catch(err => {
                console.error("Delete profile error:", err);
                alert('Failed to delete profile section');
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
                    'n-cpu-moe': profileEditForm.elements['n-cpu-moe'].value || null
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
            overlay.style.display = 'flex';
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
                        if (overlay) overlay.style.display = 'none';
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
                if (overlay) overlay.style.display = 'none';
                alert('Restart sequence timed out or connection lost. Please refresh the page manually.');
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
                    modelSwitcherStatus.innerHTML = `<span style="color:var(--color-danger);">❌ ${data.error || 'Failed to switch model'}</span>`;
                }
                logToTerminal(`Model switch failed: ${data.error}`, 'error');
            }
        } catch (err) {
            if (modelSwitcherStatus) {
                modelSwitcherStatus.innerHTML = `<span style="color:var(--color-danger);">❌ ${err.message}</span>`;
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
            modelSwitcherStatus.innerHTML = `<span style="color:var(--color-secondary);">Unloading ${currentModelName}...</span>`;
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
            alert("Please select a model to delete.");
            return;
        }
        if (!confirm(`Are you sure you want to permanently delete model "${modelName}"?\nThis will remove the manifest and all unshared blobs from disk. This action cannot be undone!`)) {
            return;
        }

        if (modelSwitcherStatus) {
            modelSwitcherStatus.innerHTML = `<span style="color:var(--color-secondary);">Deleting ${modelName}...</span>`;
        }

        try {
            const res = await fetch('/api/models/delete', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ model: modelName })
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
    
    // ──── TELEMETRY AND AUTO-TUNING INTEGRATION ────
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

    // ──── CAPABILITY ROUTING MATRIX INTEGRATION ────
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

    // ──── RESOURCE ANALYSIS ────
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

    // ──── MODEL ERROR LOG ────
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
                errorLogInterval = setInterval(loadErrorLog, 30000);
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
    
    // Periodically update current model in switcher
    setInterval(updateCurrentModel, 5000);

    // ──── MODEL DISCOVERY SEARCH AND PULL INTEGRATION ────
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
            alert("Please enter a search query.");
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

                    // Card base — tinted differently for SD vs LLM
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

                    // ── Header: name + badges ──
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

                    // ── Body: description + stats pills ──
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

                    // ── Footer: action button ──
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
                hfFilesTitle.innerHTML = `Repository: <span style="color:white;">${repoName}</span>${typeBadgeHtml}`;
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
                        // Always use 'huggingface' as source — puller auto-detects SD vs LLM from the file
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
    loadModels();
    loadTests();
    loadSharedTests();
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

    // Poll pull status periodically to keep UI in sync
    setInterval(() => {
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
    const validTabs = ['monitor', 'general', 'shared', 'profiles', 'requests', 'docs', 'sd'];
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

    function exportGeneralResults() {
        if (!currentResults || currentResults.length === 0) {
            showToast("No general benchmark results loaded to export", "error");
            return;
        }
        
        let md = `# General Benchmark Results Report\n\n`;
        md += `* **Generated At:** ${new Date().toLocaleString()}\n`;
        md += `* **Models Tested:** ${currentResults.length}\n\n`;
        
        currentResults.forEach(modelData => {
            md += `## Model: ${modelData.model}\n\n`;
            md += `* **Timestamp:** ${modelData.timestamp || 'N/A'}\n`;
            if (modelData.performance_metrics) {
                const perf = modelData.performance_metrics;
                md += `* **Average Speed:** ${perf.avg_tps || 0} TPS\n`;
                md += `* **Average TTFT:** ${perf.avg_ttft_ms || 0} ms\n`;
                md += `* **Peak RAM:** ${perf.peak_ram_pct || 0}%\n`;
                md += `* **Peak VRAM:** ${perf.peak_vram_mb || 0} MB\n`;
            }
            md += `\n`;
            
            const categories = ['coding', 'reasoning', 'instruction', 'creative', 'home_automation'];
            categories.forEach(catKey => {
                const catStats = modelData[`category_${catKey}`];
                if (!catStats || !catStats.tests) return;
                
                md += `### Category: ${catKey.replace('_', ' ').toUpperCase()}\n\n`;
                md += `| Test Scenario | Status | Latency | Speed |\n`;
                md += `| --- | --- | --- | --- |\n`;
                
                catStats.tests.forEach(test => {
                    const latVal = test.eval_duration && test.prompt_eval_duration ? (test.eval_duration + test.prompt_eval_duration) / 1e9 : test.latency;
                    const duration = latVal || 0;
                    const tps = (test.success && test.tokens_generated > 0 && duration > 0) ? (test.tokens_generated / duration) : 0;
                    
                    md += `| ${test.test_label} | ${test.success ? '✅ Success' : '❌ Fail'} | ${duration.toFixed(2)}s | ${tps > 0 ? tps.toFixed(1) + ' TPS' : '-'} |\n`;
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
    const btnExportShared = document.getElementById('btn-export-shared');
    if (btnExportShared) {
        btnExportShared.addEventListener('click', exportSharedResults);
    }

    // Listen for hashchange event to handle back/forward navigation
    window.addEventListener('hashchange', () => {
        const hash = window.location.hash.substring(1);
        if (validTabs.includes(hash)) {
            switchTab(hash);
        }
    });
});
