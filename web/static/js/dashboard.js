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
    const viewMonitor = document.getElementById('view-monitor');
    const viewGeneral = document.getElementById('view-general');
    const viewShared = document.getElementById('view-shared');
    const viewProfiles = document.getElementById('view-profiles');
    const viewRequests = document.getElementById('view-requests');
    const viewDocs = document.getElementById('view-docs');

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
        
        // Update tab buttons
        tabBtnMonitor.classList.remove('active');
        tabBtnGeneral.classList.remove('active');
        tabBtnShared.classList.remove('active');
        tabBtnProfiles.classList.remove('active');
        tabBtnRequests.classList.remove('active');
        tabBtnDocs.classList.remove('active');
        
        // Hide views
        viewMonitor.classList.add('d-none');
        viewGeneral.classList.add('d-none');
        viewShared.classList.add('d-none');
        viewProfiles.classList.add('d-none');
        viewRequests.classList.add('d-none');
        viewDocs.classList.add('d-none');
        
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
        }
    }

    tabBtnMonitor.addEventListener('click', () => switchTab('monitor'));
    tabBtnGeneral.addEventListener('click', () => switchTab('general'));
    tabBtnShared.addEventListener('click', () => switchTab('shared'));
    tabBtnProfiles.addEventListener('click', () => switchTab('profiles'));
    tabBtnRequests.addEventListener('click', () => switchTab('requests'));
    tabBtnDocs.addEventListener('click', () => switchTab('docs'));

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

    function getSelectedModels() {
        const boxes = modelCheckboxes.querySelectorAll('input[type="checkbox"]:checked');
        return Array.from(boxes).map(box => box.value);
    }

    // Polling System Metrics from proxy
    async function pollProxyStatus() {
        try {
            const res = await fetch('/api/proxy/status');
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
            pollRequestsStatus(); // immediate load
            requestsIntervalId = setInterval(pollRequestsStatus, 2000);
            setupRequestsControls();
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
        div.appendChild(idDiv);
        div.appendChild(detailsRow);
        const actionBtns = document.createElement('div');
        actionBtns.style.cssText = 'display:flex; gap:0.25rem; justify-content:flex-end; margin-top:0.25rem;';
        
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
            const isNearBottom = responseEl.scrollHeight - responseEl.clientHeight - responseEl.scrollTop < 40;
            responseEl.textContent = req.response || (req.completed_at ? '(No Output)' : 'Generating output...');
            if (isNearBottom || !req.completed_at) {
                responseEl.scrollTop = responseEl.scrollHeight;
            }
        }
    }

    let requestsControlsSetup = false;
    function setupRequestsControls() {
        if (requestsControlsSetup) return;
        requestsControlsSetup = true;

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
                input.checked = (idx === 0);
                
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
                tr.appendChild(tdLat);
                tr.appendChild(tdSpeed);
                tr.appendChild(tdView);
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
            tdLat.textContent = `${task.latency.toFixed(2)}s`;

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

        btnRun.disabled = true;
        btnRunShared.disabled = true;
        
        const isShared = endpoint.endsWith('shared_llm');
        if (isShared) {
            btnRunShared.innerHTML = `<span class="loader"></span> Starting...`;
        } else {
            btnRun.innerHTML = `<span class="loader"></span> Starting...`;
        }
        
        const termTarget = isShared ? 'shared' : 'general';
        
        try {
            logToTerminal(`Initiating benchmark for models: ${selected.join(', ')}...`, 'info', termTarget);
            const res = await fetch(endpoint, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    models: selected,
                    use_proxy: (benchmarkMode === 'proxy')
                })
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

                        const card = document.createElement('div');
                        card.style.cssText = `background:var(--card-bg);border:1px solid var(--border-color);border-radius:8px;padding:0.85rem;cursor:pointer;transition:border-color 0.2s,box-shadow 0.2s;`;
                        card.innerHTML = `
                            <div style="font-size:0.72rem;font-weight:600;color:${isDefault ? '#f59e0b' : 'var(--color-primary)'};margin-bottom:0.5rem;word-break:break-all;line-height:1.3;">${label}</div>
                            <div style="display:flex;gap:4px;flex-wrap:wrap;margin-bottom:0.5rem;">${specBadge}${flashBadge}</div>
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

    if (profileSectionSelect && profileEditForm) {
        profileSectionSelect.addEventListener('change', () => {
            const section = profileSectionSelect.value;
            if (!section) {
                profileEditForm.reset();
                return;
            }
            
            const settings = modelProfiles[section] || {};
            
            profileEditForm.elements['ctx-size'].value = settings['ctx-size'] || '';
            profileEditForm.elements['n-gpu-layers'].value = settings['n-gpu-layers'] || '';
            profileEditForm.elements['cache-type-k'].value = settings['cache-type-k'] || 'q4_0';
            profileEditForm.elements['cache-type-v'].value = settings['cache-type-v'] || 'q4_0';
            profileEditForm.elements['flash-attn'].value = settings['flash-attn'] || 'on';
            profileEditForm.elements['kv-unified'].value = settings['kv-unified'] || 'true';
            profileEditForm.elements['spec-type'].value = settings['spec-type'] || 'none';
            profileEditForm.elements['spec-draft-n-max'].value = settings['spec-draft-n-max'] || '';
            profileEditForm.elements['n-cpu-moe'].value = settings['n-cpu-moe'] || '';
        });

        profileEditForm.addEventListener('submit', (e) => {
            e.preventDefault();
            const section = profileSectionSelect.value;
            if (!section) {
                alert('Please select a model profile section to save.');
                return;
            }
            
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
                    alert('Failed to save settings: ' + data.error);
                } else {
                    modelProfiles[section] = settings;
                    alert('Settings saved successfully!');
                }
            })
            .catch(err => {
                console.error("Save profile error:", err);
                alert('Failed to save profile settings');
            });
        });
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
        const activeModelName = document.getElementById('loaded-model-name').textContent;
        if (!activeModelName || activeModelName === 'None' || activeModelName === 'Offline') {
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

    const tuningStrategySelect = document.getElementById('tuning-strategy-select');
    if (tuningStrategySelect) {
        tuningStrategySelect.addEventListener('change', () => {
            const activeModelName = document.getElementById('loaded-model-name').textContent;
            if (activeModelName && activeModelName !== 'None' && activeModelName !== 'Offline' && activeModelName !== 'No model active (Evicted/Idle)') {
                updateTelemetryAndRecommendations(activeModelName);
            }
        });
    }
    
    const btnApplyOptimizations = document.getElementById('btn-apply-optimizations');
    if (btnApplyOptimizations) {
        btnApplyOptimizations.addEventListener('click', applyTuningOptimizations);
    }
    
    const btnSaveRoutingMatrix = document.getElementById('btn-save-routing-matrix');
    if (btnSaveRoutingMatrix) {
        btnSaveRoutingMatrix.addEventListener('click', saveRoutingMatrix);
    }
    
    // Periodically update current model in switcher
    setInterval(updateCurrentModel, 5000);

    // Startup Tasks
    initCharts();
    loadModels();
    loadModelProfiles();
    loadHistory();
    switchTab('monitor'); // Start on System Monitor
});
