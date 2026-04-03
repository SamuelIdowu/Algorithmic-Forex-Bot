/**
 * ENSOTRADE Insights Portal - Frontend Logic
 */

const API_BASE = window.location.origin;
const WS_BASE = `ws://${window.location.host}/ws/intelligence`;

// --- State ---
let state = {
    consensus: [],
    reliability: [],
    news: [],
    movers: { gainers: [], losers: [] },
    status: { status: 'OFFLINE', mode: 'backtest' },
    thoughtLogs: ["Waiting for intelligence stream..."],
    activeFilter: 'all',
    searchTerm: '',
    selectedSymbol: null,
    chart: null
};

// --- Initialization ---
document.addEventListener('DOMContentLoaded', () => {
    // Initialize Lucide icons
    if (window.lucide) {
        window.lucide.createIcons();
    }

    // Start clocks
    setInterval(updateClocks, 1000);
    updateClocks();

    // Initial data fetch
    fetchData();
    setInterval(fetchData, 5000); // Poll every 5s for real-time feel

    // Setup WebSocket
    setupWebSocket();

    // Setup Terminal
    const terminalInput = document.getElementById('terminal-input');
    if (terminalInput) {
        terminalInput.addEventListener('keydown', handleTerminalInput);
    }

    // Set up search
    const searchInput = document.getElementById('matrix-search');
    if (searchInput) {
        searchInput.addEventListener('input', (e) => {
            state.searchTerm = e.target.value;
            renderConsensus();
        });
    }

    // Set up filters
    const filterButtons = document.querySelectorAll('.filter-btn');
    filterButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            // Update UI
            filterButtons.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');

            // Update State
            state.activeFilter = btn.dataset.filter;
            renderConsensus();
        });
    });

    // Close detail view
    const closeBtn = document.getElementById('close-detail');
    if (closeBtn) {
        closeBtn.addEventListener('click', closeDetailView);
    }
});

// --- UI Updates ---

function updateClocks() {
    const now = new Date();
    document.getElementById('utc-clock').textContent = `UTC: ${formatTime(now, 'UTC')}`;
    document.getElementById('ny-clock').textContent = `NY: ${formatTime(now, 'America/New_York')}`;
    document.getElementById('ldn-clock').textContent = `LDN: ${formatTime(now, 'Europe/London')}`;
}

function formatTime(date, tz) {
    return date.toLocaleTimeString('en-US', {
        timeZone: tz,
        hour12: false,
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit'
    });
}

async function fetchData() {
    try {
        const [consensusRes, reliabilityRes, statusRes, newsRes, moversRes] = await Promise.all([
            fetch(`${API_BASE}/api/consensus`),
            fetch(`${API_BASE}/api/reliability`),
            fetch(`${API_BASE}/api/status`),
            fetch(`${API_BASE}/api/news`),
            fetch(`${API_BASE}/api/top_movers`)
        ]);

        state.consensus = await consensusRes.json();
        state.reliability = await reliabilityRes.json();
        state.status = await statusRes.json();
        state.news = await newsRes.json();
        state.movers = await moversRes.json();

        renderConsensus();
        renderReliability();
        updateSystemStatus();
        renderNews();
        renderTopMovers();
    } catch (err) {
        console.error("Failed to fetch dashboard data:", err);
    }
}

function renderConsensus() {
    const container = document.getElementById('consensus-matrix');
    if (!state.consensus.length) return;

    // Apply filters
    let filteredData = state.consensus;

    // Category filter
    if (state.activeFilter !== 'all') {
        filteredData = filteredData.filter(item => item.category.includes(state.activeFilter));
    }

    // Search filter
    if (state.searchTerm) {
        const term = state.searchTerm.toLowerCase();
        filteredData = filteredData.filter(item =>
            item.symbol.toLowerCase().includes(term) ||
            item.category.toLowerCase().includes(term)
        );
    }

    // Sort by confidence
    filteredData.sort((a, b) => b.confidence - a.confidence);

    container.innerHTML = filteredData.map(item => `
        <div class="matrix-row" onclick="openDetailView('${item.symbol}', '${item.category}')">
            <div class="symbol">${item.symbol}</div>
            <div class="cell">
                <span class="label">Quant</span>
                <span class="value ${item.quant.toLowerCase()}">${item.quant}</span>
            </div>
            <div class="cell">
                <span class="label">Sentiment</span>
                <span class="value ${item.sentiment.toLowerCase()}">${item.sentiment}</span>
            </div>
            <div class="cell">
                <span class="label">Fund.</span>
                <span class="value ${item.fundamentals.toLowerCase()}">${item.fundamentals}</span>
            </div>
            <div class="cell">
                <span class="label">Confidence</span>
                <span class="value">${(item.confidence * 100).toFixed(1)}%</span>
            </div>
            <div class="cell">
                <span class="action-badge bg-${getActionColor(item.consensus)}">${item.consensus}</span>
            </div>
        </div>
    `).join('');
}

function getActionColor(action) {
    if (action === 'BUY') return 'success';
    if (action === 'SELL') return 'danger';
    return 'warning';
}

function renderReliability() {
    const container = document.getElementById('reliability-meter');
    // Group by analyst and average
    const analysts = ['quant', 'sentiment', 'fundamentals'];

    container.innerHTML = analysts.map(analyst => {
        const scores = state.reliability.filter(r => r.analyst === analyst);
        const avgAcc = scores.length ? scores.reduce((a, b) => a + b.accuracy, 0) / scores.length : 0.33;

        return `
            <div class="rel-item">
                <div class="rel-header">
                    <span>${analyst.toUpperCase()}</span>
                    <span>${(avgAcc * 100).toFixed(1)}% Accuracy</span>
                </div>
                <div class="bar-bg">
                    <div class="bar-fill" style="width: ${avgAcc * 100}%"></div>
                </div>
            </div>
        `;
    }).join('');
}

function renderNews() {
    const container = document.getElementById('news-feed');
    if (!state.news.length) return;

    container.innerHTML = state.news.map(item => `
        <div class="news-item">
            <span class="headline">${item.title}</span>
            <div class="meta">
                <span>${item.source}</span>
                <span>${item.time}</span>
                <span class="sentiment-tag ${item.sentiment}">${item.sentiment.toUpperCase()}</span>
            </div>
        </div>
    `).join('');
}

function renderTopMovers() {
    const gainersContainer = document.getElementById('gainers-list');
    const losersContainer = document.getElementById('losers-list');

    if (!state.movers.gainers || !state.movers.losers) return;

    gainersContainer.innerHTML = state.movers.gainers.map(item => `
        <div class="mover-item">
            <span class="sym">${item.symbol}</span>
            <span class="pct positive">+${item.change.toFixed(2)}%</span>
        </div>
    `).join('');

    losersContainer.innerHTML = state.movers.losers.map(item => `
        <div class="mover-item">
            <span class="sym">${item.symbol}</span>
            <span class="pct negative">${item.change.toFixed(2)}%</span>
        </div>
    `).join('');
}

function updateSystemStatus() {
    const badge = document.getElementById('system-status');
    const color = state.status.status === 'ONLINE' ? '#00ff00' : '#ff3e3e';
    badge.querySelector('span').style.backgroundColor = color;
    badge.querySelector('span').style.boxShadow = `0 0 10px ${color}`;
}

// --- WebSocket ---

function setupWebSocket() {
    const socket = new WebSocket(WS_BASE);

    socket.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.type === 'thought') {
            addThoughtLog(data.message);
        }
    };

    socket.onclose = () => {
        console.log("WebSocket disconnected. Retrying...");
        setTimeout(setupWebSocket, 5000);
    };
}

function addThoughtLog(msg) {
    const logContainer = document.getElementById('thought-log');
    const timestamp = new Date().toLocaleTimeString('en-US', { hour12: false });
    logContainer.innerHTML = `[${timestamp}] ${msg}`;

    // Add to state for history (could implement a scrollback later)
    state.thoughtLogs.push(`[${timestamp}] ${msg}`);
}

// --- Terminal ---

async function handleTerminalInput(e) {
    if (e.key === 'Enter') {
        const input = e.target.value.trim();
        if (!input) return;

        addThoughtLog(`> ${input}`);
        e.target.value = '';

        // Handle commands
        if (input.startsWith('analyze ')) {
            const symbol = input.replace('analyze ', '').split(' ')[0];
            await triggerAnalysis(symbol);
        } else {
            addThoughtLog(`Unknown command: ${input}`);
        }
    }
}

async function triggerAnalysis(symbol) {
    try {
        const res = await fetch(`${API_BASE}/api/analyze/${symbol}`, { method: 'POST' });
        const data = await res.json();
        addThoughtLog(`System: ${data.status} for ${symbol.toUpperCase()}`);
    } catch (err) {
        addThoughtLog(`Error: Failed to reach analyst engine.`);
    }
}
async function openDetailView(symbol, category) {
    state.selectedSymbol = symbol;
    document.getElementById('detail-overlay').classList.remove('hidden');
    document.getElementById('detail-symbol').textContent = symbol;
    document.getElementById('detail-category').textContent = category;

    // Clear previous data
    document.getElementById('detail-price').textContent = 'Loading...';
    document.getElementById('detail-memo').textContent = 'Fetching AI analysis...';
    document.getElementById('detail-trades').innerHTML = '<div class="loading">Fetching trades...</div>';

    try {
        const res = await fetch(`${API_BASE}/api/details/${symbol}`);
        const data = await res.json();
        renderDetailView(data);
    } catch (err) {
        console.error("Failed to fetch detail data:", err);
    }
}

function closeDetailView() {
    state.selectedSymbol = null;
    document.getElementById('detail-overlay').classList.add('hidden');
    if (state.priceChart) {
        state.priceChart.destroy();
        state.priceChart = null;
    }
    if (state.signalChart) {
        state.signalChart.destroy();
        state.signalChart = null;
    }
}

function renderDetailView(data) {
    document.getElementById('detail-price').textContent = data.current_price ? `$${data.current_price.toLocaleString()}` : 'N/A';
    document.getElementById('detail-memo').textContent = data.consensus.memo || 'No specific reasoning provided for the latest cycle.';
    document.getElementById('detail-consensus').textContent = `${data.consensus.action} (${(data.consensus.confidence * 100).toFixed(1)}%)`;
    document.getElementById('detail-consensus').className = `value ${data.consensus.action.toLowerCase()}`;

    // Render Signal Breakdown
    const qEl = document.getElementById('signal-quant');
    const sEl = document.getElementById('signal-sentiment');
    const fEl = document.getElementById('signal-fundamentals');

    qEl.textContent = data.consensus.quant || 'NEUTRAL';
    qEl.className = `value ${(data.consensus.quant || 'neutral').toLowerCase()}`;

    sEl.textContent = data.consensus.sentiment || 'NEUTRAL';
    sEl.className = `value ${(data.consensus.sentiment || 'neutral').toLowerCase()}`;

    fEl.textContent = data.consensus.fundamentals || 'NEUTRAL';
    fEl.className = `value ${(data.consensus.fundamentals || 'neutral').toLowerCase()}`;

    document.getElementById('indicator-rsi').textContent = data.indicators.rsi ? data.indicators.rsi.toFixed(2) : '--';
    document.getElementById('indicator-sma').textContent = data.indicators.sma20 ? `$${data.indicators.sma20.toLocaleString()}` : '--';

    // Render Market Bias
    const biasFill = document.getElementById('bias-fill');
    const biasText = document.getElementById('detail-bias-text');
    const biasPercentage = ((data.bias + 1) / 2) * 100; // Map -1..1 to 0..100
    biasFill.style.width = `${biasPercentage}%`;

    if (data.bias > 0.2) biasText.textContent = 'BULLISH BIAS';
    else if (data.bias < -0.2) biasText.textContent = 'BEARISH BIAS';
    else biasText.textContent = 'NEUTRAL BIAS';

    // Render trades with Setup Triggers
    const tradesContainer = document.getElementById('detail-trades');
    if (data.trades.length > 0) {
        tradesContainer.innerHTML = data.trades.map(trade => `
            <div class="trade-mini-item ${trade.action.toLowerCase()}">
                <div class="trade-info">
                    <span>${new Date(trade.timestamp).toLocaleDateString()}</span>
                    <strong>${trade.action}</strong>
                    <span>${trade.price ? '$' + trade.price.toLocaleString() : 'N/A'}</span>
                    <span class="${trade.pnl >= 0 ? 'positive' : 'negative'}">${trade.pnl ? trade.pnl.toFixed(2) : '-'}</span>
                </div>
                <div class="setup-tags">
                   ${trade.setup.quant ? `<span class="setup-tag ${trade.setup.quant.toLowerCase()}">Q:${trade.setup.quant}</span>` : ''}
                   ${trade.setup.sentiment ? `<span class="setup-tag ${trade.setup.sentiment.toLowerCase()}">S:${trade.setup.sentiment}</span>` : ''}
                   ${trade.setup.fundamentals ? `<span class="setup-tag ${trade.setup.fundamentals.toLowerCase()}">F:${trade.setup.fundamentals}</span>` : ''}
                </div>
            </div>
        `).join('');
    } else {
        tradesContainer.innerHTML = '<div class="small-text">No recent trade activity.</div>';
    }

    // Render Charts
    initCharts(data.history, data.signal_history);
}

function initCharts(priceHistory, signalHistory) {
    if (state.priceChart) state.priceChart.destroy();
    if (state.signalChart) state.signalChart.destroy();

    const priceCtx = document.getElementById('price-chart').getContext('2d');
    const signalCtx = document.getElementById('signal-chart').getContext('2d');

    // 1. Price Chart
    state.priceChart = new Chart(priceCtx, {
        type: 'line',
        data: {
            labels: priceHistory.map(d => new Date(d.date).toLocaleDateString()),
            datasets: [{
                label: 'Close Price',
                data: priceHistory.map(d => d.close),
                borderColor: '#00f3ff',
                backgroundColor: 'rgba(0, 243, 255, 0.1)',
                borderWidth: 2,
                fill: true,
                tension: 0.1,
                pointRadius: 0
            }]
        },
        options: getChartOptions()
    });

    // 2. Signal History Chart
    state.signalChart = new Chart(signalCtx, {
        type: 'bar',
        data: {
            labels: signalHistory.map(d => new Date(d.timestamp).toLocaleTimeString()),
            datasets: [{
                label: 'AI Confidence Score',
                data: signalHistory.map(d => d.score),
                backgroundColor: signalHistory.map(d => d.score >= 0 ? 'rgba(35, 134, 54, 0.6)' : 'rgba(218, 54, 51, 0.6)'),
                borderColor: signalHistory.map(d => d.score >= 0 ? '#238636' : '#da3633'),
                borderWidth: 1
            }]
        },
        options: {
            ...getChartOptions(),
            scales: {
                x: { display: false },
                y: { min: -1, max: 1, grid: { color: 'rgba(255, 255, 255, 0.05)' } }
            }
        }
    });
}

function getChartOptions() {
    return {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
            x: {
                grid: { color: 'rgba(255, 255, 255, 0.05)' },
                ticks: { color: '#8b949e', maxRotation: 0, autoSkip: true, maxTicksLimit: 10 }
            },
            y: {
                grid: { color: 'rgba(255, 255, 255, 0.05)' },
                ticks: { color: '#8b949e' }
            }
        },
        plugins: {
            legend: { display: false },
            tooltip: {
                mode: 'index',
                intersect: false,
                backgroundColor: 'rgba(5, 7, 10, 0.9)',
                titleColor: '#00f3ff'
            }
        }
    };
}
