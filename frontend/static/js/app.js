/**
 * app.js
 * ──────
 * Conversational Image Recognition Frontend Controller
 */

// Application State
const state = {
  sessionId: null,
  detectionData: null,
  sceneData: null,
  annotatedImage: null,
  cleanImage: null,
  viewMode: 'annotated', // 'annotated' | 'clean'
  confidence: 18,
  activeSampleFile: null,
  activeUploadedFile: null,
  isProcessing: false,
  highlightedElementId: null
};

// Color mapping for dominant color dots
const COLOR_HEX_MAP = {
  "Black": "#1e1e1e",
  "White": "#f5f5f5",
  "Dark Gray": "#4b5563",
  "Light Gray": "#9ca3af",
  "Red": "#ef4444",
  "Dark Red": "#991b1b",
  "Blue": "#3b82f6",
  "Navy Blue": "#1e3a8a",
  "Light Blue": "#38bdf8",
  "Green": "#22c55e",
  "Dark Green": "#166534",
  "Yellow": "#eab308",
  "Orange": "#f97316",
  "Purple": "#a855f7",
  "Pink": "#ec4899",
  "Brown": "#78350f",
  "Beige": "#d6d3d1",
  "Unknown": "#64748b"
};

// DOM Elements
const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('file-input');
const viewerContainer = document.getElementById('viewer-container');
const mainDisplayImage = document.getElementById('main-display-image');
const boundingBoxesOverlay = document.getElementById('bounding-boxes-overlay');
const viewModeToggle = document.getElementById('view-mode-toggle');
const btnViewAnnotated = document.getElementById('btn-view-annotated');
const btnViewClean = document.getElementById('btn-view-clean');
const metaResolution = document.getElementById('meta-resolution');
const metaCount = document.getElementById('meta-count');
const metaScene = document.getElementById('meta-scene');
const confSlider = document.getElementById('conf-slider');
const confVal = document.getElementById('conf-val');
const elementsBadge = document.getElementById('elements-badge');
const elementsListContainer = document.getElementById('elements-list-container');
const chatStream = document.getElementById('chat-stream');
const chatForm = document.getElementById('chat-form');
const userInput = document.getElementById('user-input');
const btnSend = document.getElementById('btn-send');
const suggestionsBar = document.getElementById('suggestions-bar');
const suggestionsContainer = document.getElementById('suggestions-container');
const btnReset = document.getElementById('btn-reset-session');
const btnClearChat = document.getElementById('btn-clear-chat');

// ── Initialization ──────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  setupEventListeners();
});

function setupEventListeners() {
  // File Upload & Drag-and-Drop
  dropzone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropzone.classList.add('drag-over');
  });

  dropzone.addEventListener('dragleave', () => {
    dropzone.classList.remove('drag-over');
  });

  dropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropzone.classList.remove('drag-over');
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileUpload(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files[0]) {
      handleFileUpload(e.target.files[0]);
    }
  });

  // Sample Chips
  document.querySelectorAll('.sample-chip').forEach(btn => {
    btn.addEventListener('click', (e) => {
      e.stopPropagation();
      const sampleName = btn.dataset.sample;
      handleSampleSelect(sampleName);
    });
  });

  // View Mode Buttons (Annotated vs Clean)
  btnViewAnnotated.addEventListener('click', () => setViewMode('annotated'));
  btnViewClean.addEventListener('click', () => setViewMode('clean'));

  // Sensitivity Slider
  confSlider.addEventListener('input', (e) => {
    const val = e.target.value;
    state.confidence = parseInt(val, 10);
    confVal.textContent = `${val}% ${val < 20 ? '(Exhaustive Scan)' : val < 40 ? '(Balanced)' : '(High Confidence)'}`;
  });

  confSlider.addEventListener('change', () => {
    // Re-run detection on slider release if an image is loaded
    if (state.activeSampleFile) {
      handleSampleSelect(state.activeSampleFile);
    } else if (state.activeUploadedFile) {
      handleFileUpload(state.activeUploadedFile);
    }
  });

  // Chat Form Submit
  chatForm.addEventListener('submit', (e) => {
    e.preventDefault();
    const query = userInput.value.trim();
    if (query && !state.isProcessing) {
      sendUserQuestion(query);
    }
  });

  // Reset & Clear
  btnReset.addEventListener('click', resetAll);
  btnClearChat.addEventListener('click', clearChat);
}

// ── Upload Handlers ─────────────────────────────────────────
async function handleFileUpload(file) {
  state.activeUploadedFile = file;
  state.activeSampleFile = null;

  const formData = new FormData();
  formData.append('file', file);
  formData.append('confidence', (state.confidence / 100).toString());
  if (state.sessionId) {
    formData.append('session_id', state.sessionId);
  }

  showUploadLoading(`Uploading and analyzing each and every element with YOLOv8...`);

  try {
    const res = await fetch('/api/upload', {
      method: 'POST',
      body: formData
    });
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();
    applyAnalysisData(data);
  } catch (err) {
    showErrorMessage(`Failed to analyze image: ${err.message}`);
  }
}

async function handleSampleSelect(sampleName) {
  state.activeSampleFile = sampleName;
  state.activeUploadedFile = null;

  showUploadLoading(`Loading sample '${sampleName}' and identifying all elements...`);

  try {
    const res = await fetch(`/api/sample/${sampleName}?confidence=${(state.confidence / 100)}`, {
      method: 'POST'
    });
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();
    applyAnalysisData(data);
  } catch (err) {
    showErrorMessage(`Failed to load sample: ${err.message}`);
  }
}

function showUploadLoading(msgText) {
  state.isProcessing = true;
  userInput.disabled = true;
  btnSend.disabled = true;
  appendBotTypingIndicator(msgText);
}

function applyAnalysisData(data) {
  state.isProcessing = false;
  state.sessionId = data.session_id;
  state.detectionData = data.detection;
  state.sceneData = data.scene;
  state.annotatedImage = data.annotated_image;
  state.cleanImage = data.original_image;

  removeTypingIndicator();

  // Show Canvas Viewer
  dropzone.style.display = 'none';
  viewerContainer.style.display = 'flex';
  viewModeToggle.style.display = 'flex';

  // Update Canvas Image
  setViewMode(state.viewMode);

  // Update Meta Bar
  const dims = data.detection.image_dimensions;
  metaResolution.textContent = `${dims.width} × ${dims.height}px`;
  metaCount.textContent = `🎯 ${data.detection.num_objects} Elements Identified`;
  metaScene.textContent = `🌄 Scene: ${data.scene.primary_scene_guess}`;

  // Render Elements Catalog Tray
  renderElementsCatalog(data.detection.detections);

  // Enable Chat Input
  userInput.disabled = false;
  btnSend.disabled = false;
  userInput.focus();

  // Append AI Initial Summary
  appendBotMessage(data.initial_message);

  // Populate Suggestions
  renderSuggestions(data.suggested_questions || []);
}

// ── View Mode Switching ─────────────────────────────────────
function setViewMode(mode) {
  state.viewMode = mode;
  if (mode === 'annotated') {
    btnViewAnnotated.classList.add('active');
    btnViewClean.classList.remove('active');
    mainDisplayImage.src = state.annotatedImage;
  } else {
    btnViewClean.classList.add('active');
    btnViewAnnotated.classList.remove('active');
    mainDisplayImage.src = state.cleanImage;
  }
  renderBoundingBoxesOverlay();
}

// ── Interactive Bounding Boxes Overlay ───────────────────────
function renderBoundingBoxesOverlay() {
  boundingBoxesOverlay.innerHTML = '';
  if (!state.detectionData || state.viewMode !== 'clean') return;

  const detections = state.detectionData.detections;
  detections.forEach(item => {
    const [normX1, normY1, normX2, normY2] = item.normalized_bbox;
    const box = document.createElement('div');
    box.className = 'interactive-bbox';
    box.id = `bbox-${item.rank}`;
    box.style.left = `${normX1 * 100}%`;
    box.style.top = `${normY1 * 100}%`;
    box.style.width = `${(normX2 - normX1) * 100}%`;
    box.style.height = `${(normY2 - normY1) * 100}%`;

    box.innerHTML = `
      <span class="bbox-label">#${item.rank} ${item.label} (${item.confidence_pct}%)</span>
    `;

    box.addEventListener('click', () => {
      highlightElement(item.rank);
      sendUserQuestion(`Tell me about object #${item.rank}`);
    });

    boundingBoxesOverlay.appendChild(box);
  });
}

// ── Elements Catalog Tray ────────────────────────────────────
function renderElementsCatalog(detections) {
  elementsBadge.textContent = `${detections.length} items`;
  elementsListContainer.innerHTML = '';

  if (detections.length === 0) {
    elementsListContainer.innerHTML = `
      <div class="empty-elements-placeholder">
        <p>No elements detected at the current sensitivity.</p>
      </div>
    `;
    return;
  }

  detections.forEach(item => {
    const card = document.createElement('div');
    card.className = 'element-card';
    card.id = `elem-card-${item.rank}`;

    const colorHex = COLOR_HEX_MAP[item.dominant_color] || '#64748b';
    const promClass = item.prominence.includes('Foreground') ? 'prom-foreground' :
                      item.prominence.includes('Midground') ? 'prom-midground' : 'prom-background';

    card.innerHTML = `
      <div class="elem-left">
        <span class="elem-rank">#${item.rank}</span>
        <div class="elem-info">
          <h4>
            <span class="elem-color-dot" style="background-color: ${colorHex};" title="${item.dominant_color}"></span>
            ${item.label}
          </h4>
          <span class="elem-sub">${item.category} • ${item.dominant_color} • ${item.position.location_desc}</span>
        </div>
      </div>
      <div class="elem-right">
        <span class="elem-conf">${item.confidence_pct}%</span>
        <span class="elem-prominence-tag ${promClass}">${item.prominence.split('/')[0].trim()}</span>
      </div>
    `;

    card.addEventListener('click', () => {
      highlightElement(item.rank);
      sendUserQuestion(`Tell me about object #${item.rank}`);
    });

    elementsListContainer.appendChild(card);
  });
}

function highlightElement(rank) {
  state.highlightedElementId = rank;
  // Highlight card in tray
  document.querySelectorAll('.element-card').forEach(c => c.classList.remove('active-highlight'));
  const card = document.getElementById(`elem-card-${rank}`);
  if (card) {
    card.classList.add('active-highlight');
    card.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }

  // Highlight bounding box on clean image if present
  document.querySelectorAll('.interactive-bbox').forEach(b => b.classList.remove('bbox-pulse'));
  const bbox = document.getElementById(`bbox-${rank}`);
  if (bbox) {
    bbox.classList.add('bbox-pulse');
  }
}

// ── Chat Functions ──────────────────────────────────────────
async function sendUserQuestion(questionText) {
  if (state.isProcessing) return;
  appendUserMessage(questionText);
  userInput.value = '';

  state.isProcessing = true;
  userInput.disabled = true;
  btnSend.disabled = true;
  appendBotTypingIndicator();

  try {
    const res = await fetch('/api/ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        question: questionText,
        session_id: state.sessionId || 'default'
      })
    });

    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();

    removeTypingIndicator();
    appendBotMessage(data.answer);

    if (data.highlight_id) {
      highlightElement(data.highlight_id);
    }
  } catch (err) {
    removeTypingIndicator();
    showErrorMessage(`Sorry, I couldn't answer that: ${err.message}`);
  } finally {
    state.isProcessing = false;
    userInput.disabled = false;
    btnSend.disabled = false;
    userInput.focus();
  }
}

function appendUserMessage(text) {
  const msg = document.createElement('div');
  msg.className = 'chat-message user-message';
  msg.innerHTML = `
    <div class="msg-avatar">👤</div>
    <div class="msg-body">
      <div class="msg-author">You</div>
      <div class="msg-text">${escapeHtml(text)}</div>
    </div>
  `;
  chatStream.appendChild(msg);
  scrollChatToBottom();
}

function appendBotMessage(markdownText) {
  const msg = document.createElement('div');
  msg.className = 'chat-message bot-message animate-in';
  
  // Render markdown safely using marked library
  const renderedHtml = (typeof marked !== 'undefined') ? marked.parse(markdownText) : markdownText;

  msg.innerHTML = `
    <div class="msg-avatar">🤖</div>
    <div class="msg-body">
      <div class="msg-author">Vision Assistant</div>
      <div class="msg-text">${renderedHtml}</div>
    </div>
  `;
  chatStream.appendChild(msg);
  scrollChatToBottom();
}

function appendBotTypingIndicator(customStatus) {
  removeTypingIndicator();
  const typing = document.createElement('div');
  typing.id = 'bot-typing-indicator';
  typing.className = 'chat-message bot-message';
  typing.innerHTML = `
    <div class="msg-avatar">🤖</div>
    <div class="msg-body">
      <div class="msg-author">Vision Assistant</div>
      <div class="msg-text">
        <div class="typing-dots">
          <span></span><span></span><span></span>
        </div>
        ${customStatus ? `<span style="font-size: 0.78rem; color: #94a3b8; margin-top: 4px; display: block;">${customStatus}</span>` : ''}
      </div>
    </div>
  `;
  chatStream.appendChild(typing);
  scrollChatToBottom();
}

function removeTypingIndicator() {
  const el = document.getElementById('bot-typing-indicator');
  if (el) el.remove();
}

function showErrorMessage(errorText) {
  removeTypingIndicator();
  const msg = document.createElement('div');
  msg.className = 'chat-message bot-message';
  msg.innerHTML = `
    <div class="msg-avatar">⚠️</div>
    <div class="msg-body">
      <div class="msg-author">System Notice</div>
      <div class="msg-text" style="border-color: rgba(244, 63, 94, 0.4); background: rgba(244, 63, 94, 0.1);">
        ${escapeHtml(errorText)}
      </div>
    </div>
  `;
  chatStream.appendChild(msg);
  scrollChatToBottom();
}

function renderSuggestions(suggestions) {
  if (!suggestions || suggestions.length === 0) {
    suggestionsBar.style.display = 'none';
    return;
  }
  suggestionsContainer.innerHTML = '';
  suggestions.forEach(text => {
    const pill = document.createElement('button');
    pill.type = 'button';
    pill.className = 'sugg-pill';
    pill.textContent = text;
    pill.addEventListener('click', () => sendUserQuestion(text));
    suggestionsContainer.appendChild(pill);
  });
  suggestionsBar.style.display = 'block';
}

function scrollChatToBottom() {
  chatStream.scrollTop = chatStream.scrollHeight;
}

function clearChat() {
  chatStream.innerHTML = `
    <div class="chat-message bot-message">
      <div class="msg-avatar">🤖</div>
      <div class="msg-body">
        <div class="msg-author">Vision Assistant</div>
        <div class="msg-text">Chat cleared. You can continue asking questions about the active image!</div>
      </div>
    </div>
  `;
}

function resetAll() {
  state.sessionId = null;
  state.detectionData = null;
  state.sceneData = null;
  state.annotatedImage = null;
  state.cleanImage = null;
  state.activeSampleFile = null;
  state.activeUploadedFile = null;
  state.isProcessing = false;
  state.highlightedElementId = null;

  fileInput.value = '';
  viewerContainer.style.display = 'none';
  viewModeToggle.style.display = 'none';
  dropzone.style.display = 'block';
  suggestionsBar.style.display = 'none';

  elementsBadge.textContent = '0 items';
  elementsListContainer.innerHTML = `
    <div class="empty-elements-placeholder">
      <p>Upload an image above to catalog every detected element</p>
    </div>
  `;

  userInput.disabled = true;
  btnSend.disabled = true;
  clearChat();
}

function escapeHtml(str) {
  return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}
