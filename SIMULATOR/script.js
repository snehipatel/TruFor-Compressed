/**
 * Platform Compression & Forensic Simulator — Frontend Logic
 * ==========================================================
 * Matches the reference mockup exactly with instant pre-loaded
 * demo state, full interactivity, live compression, TruFor forensic
 * scoring, zoom inspector, and batch dataset generation.
 */

// ============================================================================
// STATE
// ============================================================================
let selectedPlatform = 'whatsapp';
let qualityMode = 'standard';
let uploadedFile = null;
let isUsingDemo = true;
let currentResult = null;

const platformLabels = {
    whatsapp: 'WhatsApp',
    instagram: 'Instagram',
    facebook: 'Facebook',
    telegram: 'Telegram',
};

const platformButtonColors = {
    whatsapp: '#115e59', // Deep pine green (matching mockup)
    instagram: '#833ab4', // Instagram purple/pink
    facebook: '#1877f2', // Facebook blue
    telegram: '#0284c7', // Telegram blue
};

// ============================================================================
// INITIALIZATION
// ============================================================================
document.addEventListener('DOMContentLoaded', () => {
    setupDropZone();
    setupBatchDropZone();
    loadDemo();
});

// ============================================================================
// DEMO INITIALIZATION
// ============================================================================
async function loadDemo() {
    try {
        const res = await fetch(`/api/demo?platform=${selectedPlatform}&quality_mode=${qualityMode}`);
        if (!res.ok) return;
        const data = await res.json();
        if (data.success) {
            renderSimulationResults(data);
        }
    } catch (err) {
        console.warn('Could not load demo state:', err);
    }
}

// ============================================================================
// TAB NAVIGATION
// ============================================================================
function switchTab(tabId) {
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.tab === tabId);
    });
    document.querySelectorAll('.tab-panel').forEach(panel => {
        panel.classList.toggle('active', panel.id === `panel-${tabId}`);
    });
}

function scrollToUpload() {
    const el = document.getElementById('upload-section');
    if (el) el.scrollIntoView({ behavior: 'smooth' });
}

function onHeaderRunSimulation() {
    // 1. Switch to the interactive simulation tab
    switchTab('interactive');
    // 2. Trigger the platform compression simulation
    runSimulation();
    // 3. Smoothly scroll down to the results section so the user sees the output immediately
    setTimeout(() => {
        const resEl = document.getElementById('resultsSection');
        if (resEl) {
            resEl.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
    }, 250);
}

// ============================================================================
// PLATFORM SELECTION
// ============================================================================
function selectPlatform(platform) {
    selectedPlatform = platform;

    // Update active card styling
    document.querySelectorAll('.platform-card').forEach(card => {
        card.classList.toggle('selected', card.dataset.platform === platform);
    });

    updateSimulateButton();

    // If in demo mode, immediately re-simulate demo for that platform
    if (isUsingDemo) {
        runSimulation();
    }
}

// ============================================================================
// QUALITY MODE
// ============================================================================
function setQualityMode(mode) {
    qualityMode = mode;

    const btnStandard = document.getElementById('btnStandard');
    const radioHd = document.getElementById('radioHd');

    if (mode === 'standard') {
        btnStandard.classList.add('active');
        radioHd.checked = false;
    } else {
        btnStandard.classList.remove('active');
        radioHd.checked = true;
    }

    if (isUsingDemo) {
        runSimulation();
    }
}

function updateSimulateButton() {
    const btn = document.getElementById('btnSimulate');
    const btnText = document.getElementById('btnSimulateText');
    const pName = platformLabels[selectedPlatform] || 'WhatsApp';

    btnText.textContent = `Simulate for ${pName} →`;
    btn.style.background = platformButtonColors[selectedPlatform] || '#115e59';
}

// ============================================================================
// FILE UPLOAD (DRAG & DROP + BROWSE)
// ============================================================================
function setupDropZone() {
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');

    dropZone.addEventListener('click', () => fileInput.click());

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleUserFile(e.target.files[0]);
        }
    });

    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('drag-over');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('drag-over');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('drag-over');
        if (e.dataTransfer.files.length > 0) {
            handleUserFile(e.dataTransfer.files[0]);
        }
    });
}

function handleUserFile(file) {
    const validExts = /\.(jpg|jpeg|png|webp|bmp|tif|tiff)$/i;
    if (!validExts.test(file.name)) {
        alert('Please upload an image file (JPG, PNG, WEBP, BMP, TIFF).');
        return;
    }

    if (file.size > 50 * 1024 * 1024) {
        alert('File is too large. Maximum supported size is 50 MB.');
        return;
    }

    uploadedFile = file;
    isUsingDemo = false;

    // Show preview card
    const reader = new FileReader();
    reader.onload = (e) => {
        const img = new Image();
        img.onload = () => {
            document.getElementById('fileThumb').src = e.target.result;
            document.getElementById('fileName').textContent = file.name;
            document.getElementById('fileMeta').textContent =
                `${formatBytes(file.size)}    ${img.width} × ${img.height}`;
            document.getElementById('filePreview').classList.add('visible');

            // Automatically run simulation on newly uploaded file
            runSimulation();
        };
        img.src = e.target.result;
    };
    reader.readAsDataURL(file);
}

function removeUploadedFile(e) {
    e.stopPropagation();
    uploadedFile = null;
    isUsingDemo = true;
    document.getElementById('fileInput').value = '';
    // Revert to demo mountains.jpg
    loadDemo();
}

// ============================================================================
// SIMULATION EXECUTION
// ============================================================================
async function runSimulation() {
    const btn = document.getElementById('btnSimulate');
    const spinner = document.getElementById('simSpinner');
    const btnText = document.getElementById('btnSimulateText');

    btn.disabled = true;
    spinner.classList.add('active');

    try {
        let data;
        if (isUsingDemo || !uploadedFile) {
            const res = await fetch(`/api/demo?platform=${selectedPlatform}&quality_mode=${qualityMode}`);
            data = await res.json();
        } else {
            const formData = new FormData();
            formData.append('image', uploadedFile);
            formData.append('platform', selectedPlatform);
            formData.append('quality_mode', qualityMode);

            const res = await fetch('/api/simulate', {
                method: 'POST',
                body: formData,
            });
            data = await res.json();
        }

        if (data.success) {
            currentResult = data;
            renderSimulationResults(data);
        } else {
            alert('Simulation error: ' + (data.error || 'Unknown error'));
        }
    } catch (err) {
        console.error('Simulation request failed:', err);
    } finally {
        btn.disabled = false;
        spinner.classList.remove('active');
        btnText.style.display = 'inline';
    }
}

// ============================================================================
// RENDER SIMULATION RESULTS
// ============================================================================
function renderSimulationResults(data) {
    const pName = platformLabels[data.platform] || 'WhatsApp';
    const qMode = data.quality_mode === 'hd' ? 'HD' : 'Standard';

    // Titles
    document.getElementById('resultsTitle').textContent = `Comparison Results (${pName} Simulation)`;
    document.getElementById('compressedCardTitle').textContent = `Simulated ${pName} Compressed Image`;
    document.getElementById('statsBarTitle').textContent = `Compression Statistics (${pName} - ${qMode})`;

    // Image previews
    document.getElementById('imgOriginal').src = data.original.url;
    document.getElementById('imgCompressed').src = data.compressed.url;
    document.getElementById('imgDifference').src = data.difference.url;

    // Card 1: Original stats
    document.getElementById('statOrigSize').textContent = data.original.display_size || formatBytes(data.original.size);
    document.getElementById('statOrigRes').textContent = `${data.original.width} × ${data.original.height}`;
    document.getElementById('statOrigFormat').textContent = data.original.format;

    // Card 2: Compressed stats
    document.getElementById('statCompSize').textContent = data.compressed.display_size || `~ ${formatBytes(data.compressed.size)}`;
    document.getElementById('statCompRes').textContent = `${data.compressed.width} × ${data.compressed.height}`;
    document.getElementById('statCompFormat').textContent = data.compressed.format;

    // Bottom Stats Bar (6 columns)
    const stats = data.stats;
    document.getElementById('sbOrigSize').textContent = data.original.display_size || formatBytes(data.original.size);
    document.getElementById('sbCompSize').textContent = data.compressed.display_size || `~ ${formatBytes(data.compressed.size)}`;
    
    // Compression Ratio
    document.getElementById('sbRatio').textContent = `~ ${stats.compression_ratio}`;
    document.getElementById('sbRatioPct').textContent = `(= ${Math.round(stats.reduction_pct)}% reduction)`;

    // Resolution Change
    document.getElementById('sbResChange').textContent = stats.resolution_change;

    // Size Reduction
    document.getElementById('sbReduction').textContent = stats.display_reduction || `~ ${formatBytes(stats.size_reduction)}`;
    document.getElementById('sbReductionPct').textContent = `(${Math.round(stats.reduction_pct)}% smaller)`;

    // Visual Difference
    const visEl = document.getElementById('sbVisual');
    visEl.textContent = stats.visual_difference;
    if (stats.visual_difference === 'Minimal' || stats.visual_difference === 'Low') {
        visEl.className = 'stat-cell-val green-text';
    } else if (stats.visual_difference === 'Moderate') {
        visEl.className = 'stat-cell-val amber-text';
    } else {
        visEl.className = 'stat-cell-val'
        visEl.style.color = '#ef4444';
    }
    document.getElementById('sbVisualDesc').textContent = stats.visual_description || '';
}


// ============================================================================
// METHODOLOGY & ABOUT MODALS
// ============================================================================
function openMethodologyModal() {
    const m = document.getElementById('methodologyModal');
    if (m) m.classList.add('open');
}

function closeMethodologyModal(e) {
    if (e && e.target && e.target !== e.currentTarget) return;
    const m = document.getElementById('methodologyModal');
    if (m) m.classList.remove('open');
}

function openAboutModal() {
    const m = document.getElementById('aboutModal');
    if (m) m.classList.add('open');
}

function closeAboutModal(e) {
    if (e && e.target && e.target !== e.currentTarget) return;
    const m = document.getElementById('aboutModal');
    if (m) m.classList.remove('open');
}

// Backward-compatible alias
function openDetailsModal() {
    openMethodologyModal();
}

function closeDetailsModal(e) {
    closeMethodologyModal(e);
}

// ============================================================================
// ZOOM INSPECTOR MODAL
// ============================================================================
function openZoomModal(imgElementId, title) {
    const src = document.getElementById(imgElementId).src;
    if (!src) return;

    document.getElementById('zoomImg').src = src;
    document.getElementById('zoomTitle').textContent = title || 'Image Inspector';
    document.getElementById('zoomModal').classList.add('open');
}

function closeZoomModal() {
    document.getElementById('zoomModal').classList.remove('open');
}

// ============================================================================
// BATCH PROCESSING (DUAL MODE: UNIVERSAL UPLOAD & SERVER FOLDER)
// ============================================================================
let currentBatchMode = 'upload';
let selectedBatchFiles = [];

function setBatchMode(mode) {
    currentBatchMode = mode;
    const btnUpload = document.getElementById('btnModeUpload');
    const btnServer = document.getElementById('btnModeServer');
    const uploadContainer = document.getElementById('batchUploadModeContainer');
    const serverContainer = document.getElementById('batchServerModeContainer');

    if (mode === 'upload') {
        if (btnUpload) btnUpload.classList.add('active');
        if (btnServer) btnServer.classList.remove('active');
        if (uploadContainer) uploadContainer.style.display = 'block';
        if (serverContainer) serverContainer.style.display = 'none';
    } else {
        if (btnUpload) btnUpload.classList.remove('active');
        if (btnServer) btnServer.classList.add('active');
        if (uploadContainer) uploadContainer.style.display = 'none';
        if (serverContainer) serverContainer.style.display = 'block';
    }
}

function handleBatchFilesSelect(e) {
    if (e.target.files && e.target.files.length > 0) {
        handleBatchFilesList(Array.from(e.target.files));
    }
}

function handleBatchFolderSelect(e) {
    if (e.target.files && e.target.files.length > 0) {
        handleBatchFilesList(Array.from(e.target.files));
    }
}

function handleBatchFilesList(fileList) {
    const validExts = /\.(jpg|jpeg|png|webp|bmp|tif|tiff)$/i;
    selectedBatchFiles = fileList.filter(f => validExts.test(f.name));

    const badge = document.getElementById('batchFilesCountBadge');
    if (!badge) return;

    if (selectedBatchFiles.length === 0) {
        badge.style.display = 'none';
        alert('No valid image files (JPG, PNG, WEBP, BMP, TIFF) found in selection.');
        return;
    }

    badge.style.display = 'inline-block';
    badge.textContent = `📁 ${selectedBatchFiles.length} images selected (ready to compress & zip)`;
}

function setupBatchDropZone() {
    const dropZone = document.getElementById('batchUploadDropZone');
    if (!dropZone) return;

    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.style.borderColor = '#2563eb';
        dropZone.style.background = '#eff6ff';
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.style.borderColor = '#cbd5e1';
        dropZone.style.background = '#f8fafc';
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.style.borderColor = '#cbd5e1';
        dropZone.style.background = '#f8fafc';
        if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files.length > 0) {
            handleBatchFilesList(Array.from(e.dataTransfer.files));
        }
    });
}

async function runActiveBatchMode() {
    if (currentBatchMode === 'server') {
        return runBatchProcessing();
    }

    // Direct Browser Multi-File / Folder Upload Mode
    if (!selectedBatchFiles || selectedBatchFiles.length === 0) {
        alert('Please choose or drag & drop images or a folder first.');
        return;
    }

    const platform = document.getElementById('batchPlatform').value;
    const quality = document.getElementById('batchQuality').value;

    const btn = document.getElementById('btnBatchRun');
    const spinner = document.getElementById('batchSpinner');
    const btnText = document.getElementById('batchBtnText');

    btn.disabled = true;
    spinner.classList.add('active');
    btnText.textContent = `Compressing ${selectedBatchFiles.length} files...`;

    try {
        const formData = new FormData();
        selectedBatchFiles.forEach(f => formData.append('images', f));
        formData.append('platform', platform);
        formData.append('quality_mode', quality);

        const res = await fetch('/api/batch-upload', {
            method: 'POST',
            body: formData,
        });

        const data = await res.json();
        if (data.success) {
            renderBatchResults(data);
        } else {
            alert('Batch upload error: ' + (data.error || 'Failed'));
        }
    } catch (err) {
        alert('Batch upload request failed: ' + err.message);
    } finally {
        btn.disabled = false;
        spinner.classList.remove('active');
        btnText.textContent = 'Generate Dataset →';
    }
}

async function runBatchProcessing() {
    const inputFolder = document.getElementById('batchInput').value.trim();
    const outputFolder = document.getElementById('batchOutput').value.trim();
    const platform = document.getElementById('batchPlatform').value;
    const quality = document.getElementById('batchQuality').value;

    if (!inputFolder) {
        alert('Please specify a server input folder path.');
        return;
    }
    if (!outputFolder) {
        alert('Please specify a server output folder path.');
        return;
    }

    const btn = document.getElementById('btnBatchRun');
    const spinner = document.getElementById('batchSpinner');
    const btnText = document.getElementById('batchBtnText');

    btn.disabled = true;
    spinner.classList.add('active');
    btnText.textContent = 'Compressing...';

    try {
        const res = await fetch('/api/batch', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                input_folder: inputFolder,
                output_folder: outputFolder,
                platform: platform,
                quality_mode: quality,
            }),
        });

        const data = await res.json();
        if (data.success) {
            renderBatchResults(data);
        } else {
            alert('Batch error: ' + (data.error || 'Failed'));
        }
    } catch (err) {
        alert('Batch processing request failed: ' + err.message);
    } finally {
        btn.disabled = false;
        spinner.classList.remove('active');
        btnText.textContent = 'Generate Dataset →';
    }
}

function renderBatchResults(data) {
    const container = document.getElementById('batchResults');
    container.style.display = 'block';

    const summary = document.getElementById('batchSummary');
    const locationInfo = data.manifest_path ? `Manifest saved to ${data.manifest_path}` : 'ZIP dataset ready for download';
    summary.textContent = `Batch Completed: ${data.successful} / ${data.total_images} images compressed for ${data.profile_label}. ${locationInfo} (Average size reduction: ${data.overall_reduction_pct}%).`;

    // Show ZIP download button if available
    const zipArea = document.getElementById('batchZipDownloadArea');
    const zipBtn = document.getElementById('btnDownloadBatchZip');
    const zipText = document.getElementById('btnDownloadZipText');
    if (data.download_url && zipArea && zipBtn) {
        zipArea.style.display = 'block';
        zipBtn.href = data.download_url;
        const zipSizeStr = data.zip_size ? ` [${formatBytes(data.zip_size)}]` : '';
        if (zipText) {
            zipText.textContent = `Download Processed Dataset (.zip)${zipSizeStr}`;
        }
    } else if (zipArea) {
        zipArea.style.display = 'none';
    }

    const tbody = document.getElementById('batchTableBody');
    tbody.innerHTML = '';

    data.results.forEach((item, idx) => {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td>${idx + 1}</td>
            <td><strong>${item.original_filename}</strong></td>
            <td>${item.compressed_filename}</td>
            <td>${formatBytes(item.original_size)}</td>
            <td>${formatBytes(item.compressed_size)}</td>
            <td>${item.compression_ratio}</td>
            <td style="color:#16a34a; font-weight:600;">-${item.reduction_pct}%</td>
            <td><span style="color:#16a34a;">● ${item.status}</span></td>
        `;
        tbody.appendChild(row);
    });

    container.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// ============================================================================
// UTILITIES
// ============================================================================
function formatBytes(bytes, decimals = 1) {
    if (!bytes || bytes === 0) return '0 B';
    const k = 1024;
    const dm = decimals < 0 ? 0 : decimals;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
}
