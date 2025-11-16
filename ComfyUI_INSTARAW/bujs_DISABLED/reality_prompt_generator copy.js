// ---
// Filename: ../ComfyUI_INSTARAW/js/reality_prompt_generator.js
// Reality Prompt Generator (RPG) - Full JavaScript UI Implementation
// Following AdvancedImageLoader patterns exactly
// ---

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const DEFAULT_RPG_SYSTEM_PROMPT = "You are an expert AI prompt engineer specializing in INSTARAW workflows.";
const REMOTE_PROMPTS_DB_URL = "https://instara.s3.us-east-1.amazonaws.com/prompts.db.json";
const CREATIVE_MODEL_OPTIONS = [
	{ value: "gemini-2.5-pro", label: "Gemini 2.5 Pro" },
	{ value: "gemini-flash-latest", label: "Gemini Flash Latest" },
	{ value: "grok-4-fast-reasoning", label: "Grok 4 Fast (Reasoning)" },
	{ value: "grok-4-fast-non-reasoning", label: "Grok 4 Fast (Non-Reasoning)" },
	{ value: "grok-4-0709", label: "Grok 4 0709" },
	{ value: "grok-3-mini", label: "Grok 3 Mini" },
];

app.registerExtension({
	name: "Comfy.INSTARAW.RealityPromptGenerator",

	async beforeRegisterNodeDef(nodeType, nodeData, app) {
		if (nodeData.name === "INSTARAW_RealityPromptGenerator") {
			const onNodeCreated = nodeType.prototype.onNodeCreated;
			nodeType.prototype.onNodeCreated = function () {
				onNodeCreated?.apply(this, arguments);

				// Initialize properties (following AIL pattern)
				if (!this.properties.prompt_queue_data) {
					this.properties.prompt_queue_data = JSON.stringify([]);
				}
				if (!this.properties.prompts_db_cache) {
					this.properties.prompts_db_cache = null;
				}
				if (!this.properties.bookmarks) {
					this.properties.bookmarks = JSON.stringify([]);
				}
				if (!this.properties.active_tab) {
					this.properties.active_tab = "library";
				}
				if (!this.properties.library_filters) {
					this.properties.library_filters = JSON.stringify({
						tags: [],
						content_type: "any",
						safety_level: "any",
						shot_type: "any",
						quality: "any",
						search_query: "",
					});
				}
				if (this.properties.creative_system_prompt === undefined) {
					this.properties.creative_system_prompt = DEFAULT_RPG_SYSTEM_PROMPT;
				}
				if (this.properties.creative_temperature === undefined) {
					this.properties.creative_temperature = 0.9;
				}
				if (this.properties.creative_top_p === undefined) {
					this.properties.creative_top_p = 0.9;
				}

				const node = this;
				let cachedHeight = 400; // Initial height (AIL pattern)
				let isUpdatingHeight = false; // Prevent concurrent updates (AIL pattern)

				// Database state
				let promptsDatabase = null;
				let isDatabaseLoading = false;
				let databaseLoadProgress = 0;

				// AIL sync state
				node._linkedAILNodeId = null;
				node._linkedImages = [];
				node._linkedImageCount = 0;

				// Pagination state
				let currentPage = 0;
				const itemsPerPage = 50;

				// Container setup (exact AIL pattern)
				const container = document.createElement("div");
				container.className = "instaraw-rpg-container";
				container.style.width = "100%";
				container.style.boxSizing = "border-box";
				container.style.overflow = "hidden";
				container.style.height = `${cachedHeight}px`;

				// === Height Management (Exact AIL Pattern) ===
				const updateCachedHeight = () => {
					if (isUpdatingHeight) return;
					isUpdatingHeight = true;
					container.style.overflow = "visible";
					container.style.height = "auto";
					requestAnimationFrame(() => {
						requestAnimationFrame(() => {
							const newHeight = container.offsetHeight;
							if (newHeight > 0 && Math.abs(newHeight - cachedHeight) > 2) {
								cachedHeight = newHeight;
								container.style.height = `${newHeight}px`;
								const sz = node.computeSize();
								node.size[1] = sz[1];
								node.onResize?.(sz);
								app.graph.setDirtyCanvas(true, false);
							}
							container.style.overflow = "hidden";
							isUpdatingHeight = false;
						});
					});
				};

				// === State Sync (Exact AIL Pattern) ===
				const syncPromptBatchWidget = () => {
					const promptQueueWidget = node.widgets?.find((w) => w.name === "prompt_queue_data");
					if (promptQueueWidget) {
						promptQueueWidget.value = node.properties.prompt_queue_data;
					} else {
						const widget = node.addWidget("text", "prompt_queue_data", node.properties.prompt_queue_data, () => {}, { serialize: true });
						widget.hidden = true;
					}
				};

				const parsePromptBatch = () => {
					try {
						const raw = node.properties.prompt_queue_data ?? "[]";
						if (Array.isArray(raw)) {
							return raw;
						}
						if (typeof raw === "string") {
							if (!raw.trim()) return [];
							return JSON.parse(raw);
						}
						return [];
					} catch (error) {
						console.warn("[RPG] Failed to parse prompt batch data, resetting to []", error);
						node.properties.prompt_queue_data = "[]";
						syncPromptBatchWidget();
						return [];
					}
				};

				const setPromptBatchData = (promptQueue) => {
					const normalized = Array.isArray(promptQueue) ? promptQueue : [];
					node.properties.prompt_queue_data = JSON.stringify(normalized);
					syncPromptBatchWidget();
					return normalized;
				};

				// === Database Loading with IndexedDB ===
				const loadPromptsDatabase = async () => {
					if (isDatabaseLoading) return;
					isDatabaseLoading = true;

					try {
						// Try IndexedDB first
						const cachedDB = await getFromIndexedDB("prompts_db_cache");
						if (cachedDB && cachedDB.version === "1.0") {
							promptsDatabase = cachedDB.data;
							isDatabaseLoading = false;
							renderUI();
							return;
						}

						// Fetch from server
						databaseLoadProgress = 0;
						renderUI(); // Show loading state
						updateLoadingProgressUI();

						const response = await fetchPromptsDatabase();
						const reader = response.body.getReader();
						const contentLength = +response.headers.get("Content-Length");
						let receivedLength = 0;
						const chunks = [];

						while (true) {
							const { done, value } = await reader.read();
							if (done) break;
							chunks.push(value);
							receivedLength += value.length;
							databaseLoadProgress = contentLength
								? Math.min(99, Math.round((receivedLength / contentLength) * 100))
								: 99;
							updateLoadingProgressUI();
						}

						const chunksAll = new Uint8Array(receivedLength);
						let position = 0;
						for (const chunk of chunks) {
							chunksAll.set(chunk, position);
							position += chunk.length;
						}

						const text = new TextDecoder("utf-8").decode(chunksAll);
						promptsDatabase = JSON.parse(text);

						// Store in IndexedDB
						await saveToIndexedDB("prompts_db_cache", {
							version: "1.0",
							data: promptsDatabase,
						});

						isDatabaseLoading = false;
						databaseLoadProgress = 100;
						renderUI();
					} catch (error) {
						console.error("[RPG] Database loading error:", error);
						isDatabaseLoading = false;
						renderUI();
					}
				};

				const updateLoadingProgressUI = () => {
					const fill = container.querySelector(".instaraw-rpg-progress-fill");
					const text = container.querySelector(".instaraw-rpg-progress-text");
					if (fill) fill.style.width = `${databaseLoadProgress}%`;
					if (text) text.textContent = `${databaseLoadProgress}% (22MB)`;
				};

				const fetchPromptsDatabase = async () => {
					const response = await fetch(REMOTE_PROMPTS_DB_URL, { cache: "no-store" });
					if (!response.ok) throw new Error(`Remote prompts DB error ${response.status}`);
					return response;
				};

				const autoResizeTextarea = (textarea) => {
					if (!textarea) return;
					textarea.style.height = "auto";
					textarea.style.height = `${textarea.scrollHeight}px`;
				};

				const parseJSONResponse = async (response) => {
					const text = await response.text();
					try {
						return JSON.parse(text);
					} catch (error) {
						throw new Error(`Unexpected response (${response.status}): ${text.slice(0, 500)}`);
					}
				};

				// === IndexedDB Helpers ===
				const getFromIndexedDB = (key) => {
					return new Promise((resolve) => {
						const request = indexedDB.open("INSTARAW_RPG", 1);
						request.onupgradeneeded = (e) => {
							const db = e.target.result;
							if (!db.objectStoreNames.contains("cache")) {
								db.createObjectStore("cache");
							}
						};
						request.onsuccess = (e) => {
							const db = e.target.result;
							const tx = db.transaction("cache", "readonly");
							const store = tx.objectStore("cache");
							const get = store.get(key);
							get.onsuccess = () => resolve(get.result);
							get.onerror = () => resolve(null);
						};
						request.onerror = () => resolve(null);
					});
				};

				const saveToIndexedDB = (key, value) => {
					return new Promise((resolve) => {
						const request = indexedDB.open("INSTARAW_RPG", 1);
						request.onupgradeneeded = (e) => {
							const db = e.target.result;
							if (!db.objectStoreNames.contains("cache")) {
								db.createObjectStore("cache");
							}
						};
						request.onsuccess = (e) => {
							const db = e.target.result;
							const tx = db.transaction("cache", "readwrite");
							const store = tx.objectStore("cache");
							const put = store.put(value, key);
							put.onsuccess = () => resolve(true);
							put.onerror = () => resolve(false);
						};
						request.onerror = () => resolve(false);
					});
				};

				// === Main Render Function ===
				const renderUI = () => {
					const activeTab = node.properties.active_tab || "library";
					const promptQueue = parsePromptBatch();
					const modeWidget = node.widgets?.find((w) => w.name === "mode");
					const currentMode = modeWidget?.value || "auto";
					const resolvedModeWidget = node.widgets?.find((w) => w.name === "resolved_mode");
					const resolvedMode = resolvedModeWidget?.value || "txt2img";
					const generationModeWidget = node.widgets?.find((w) => w.name === "generation_mode");
					const generationMode = generationModeWidget?.value || "sum_repeat_counts";
					const creativeModelWidget = node.widgets?.find((w) => w.name === "creative_model");
					if (this.properties.creative_model === undefined && creativeModelWidget) {
						this.properties.creative_model = creativeModelWidget.value;
					}
					const currentCreativeModel = creativeModelWidget?.value || this.properties.creative_model || "gemini-2.5-pro";
					const creativeSystemPrompt = node.properties.creative_system_prompt || DEFAULT_RPG_SYSTEM_PROMPT;
					const currentCreativeTemperature = parseFloat(node.properties.creative_temperature ?? 0.9) || 0.9;
					const currentCreativeTopP = parseFloat(node.properties.creative_top_p ?? 0.9) || 0.9;

					const totalGenerations =
						generationMode === "one_per_entry"
							? promptQueue.length
							: promptQueue.reduce((sum, entry) => sum + (entry.repeat_count || 1), 0);

					const tabs = [
						{ id: "library", label: "Library", icon: "📚" },
						{ id: "creative", label: "Creative Mix", icon: "✨" },
						{ id: "character", label: "Character", icon: "👤" },
					];

					const generationModeLabel = generationMode === "one_per_entry" ? "1 image per entry" : "Respect repeat counts";
					const linkedImages = node._linkedImageCount || 0;
					const modeHint =
						currentMode === "img2img"
							? "Generate from prompt batch & linked images"
							: currentMode === "txt2img"
								? "Generate fresh images from prompt batch"
								: "Let RPG decide per workflow wiring";

					const tabButtons = tabs
						.map(
							(tab) => `
							<button class="instaraw-rpg-tab ${activeTab === tab.id ? "active" : ""}" data-tab="${tab.id}">
								<span>${tab.icon}</span>
								${tab.label}
							</button>
						`,
						)
						.join("");

					const uiState = {
						promptQueue,
						currentCreativeModel,
						creativeSystemPrompt,
						currentCreativeTemperature,
						currentCreativeTopP,
					};
					const tabContent = renderActiveTabContent(activeTab, uiState);
					const imagePreview = renderImagePreview(resolvedMode, totalGenerations);

					container.innerHTML = `
						<div class="instaraw-rpg-topbar">
							<div class="instaraw-rpg-mode-card">
								<div class="instaraw-rpg-mode-selector">
									<label>Mode</label>
									<select class="instaraw-rpg-mode-dropdown">
										<option value="auto" ${currentMode === "auto" ? "selected" : ""}>🎯 Auto</option>
										<option value="img2img" ${currentMode === "img2img" ? "selected" : ""}>🖼️ Img2Img</option>
										<option value="txt2img" ${currentMode === "txt2img" ? "selected" : ""}>📝 Txt2Img</option>
									</select>
								</div>
								<div class="instaraw-rpg-mode-meta">
									<span class="instaraw-rpg-mode-hint">${modeHint}</span>
									<span class="instaraw-rpg-mode-pill">Resolved: ${resolvedMode}</span>
									<span class="instaraw-rpg-mode-pill">${generationModeLabel}</span>
								</div>
							</div>
							<div class="instaraw-rpg-kpi-row">
								<div class="instaraw-rpg-kpi">
									<span>Prompt Queue</span>
									<strong>${promptQueue.length}</strong>
								</div>
								<div class="instaraw-rpg-kpi">
									<span>Total Generations</span>
									<strong>${totalGenerations}</strong>
								</div>
								<div class="instaraw-rpg-kpi">
									<span>Linked Images</span>
									<strong>${linkedImages}</strong>
								</div>
							</div>
						</div>

						${imagePreview}

						<div class="instaraw-rpg-content">
							<div class="instaraw-rpg-main-panel">
								<div class="instaraw-rpg-tabs">
									${tabButtons}
								</div>
								<div class="instaraw-rpg-panel-card">
									${tabContent}
								</div>
							</div>
							<div class="instaraw-rpg-batch-panel">
								${renderBatchPanel(promptQueue, totalGenerations)}
							</div>
						</div>

						<div class="instaraw-rpg-footer">
							<div class="instaraw-rpg-stats">
								<span class="instaraw-rpg-stat-badge">Total Generations: ${totalGenerations}</span>
								<span class="instaraw-rpg-stat-label">Mode: ${resolvedMode}</span>
								${node._linkedAILNodeId ? `<span class="instaraw-rpg-stat-label">Linked AIL: Node #${node._linkedAILNodeId}</span>` : `<span class="instaraw-rpg-stat-label">No Advanced Loader linked</span>`}
							</div>
						</div>
					`;

					setupEventHandlers();
					setupDragAndDrop();
					updateCachedHeight();
				};

				// === Tab Content Rendering ===
				const renderActiveTabContent = (activeTab, uiState) => {
					if (isDatabaseLoading) {
						return `
							<div class="instaraw-rpg-loading">
								<div class="instaraw-rpg-loading-spinner"></div>
								<p>Loading Prompts Database...</p>
								<div class="instaraw-rpg-progress-bar">
									<div class="instaraw-rpg-progress-fill" style="width: ${databaseLoadProgress}%"></div>
								</div>
								<p class="instaraw-rpg-progress-text">${databaseLoadProgress}% (22MB)</p>
							</div>
						`;
					}

					if (!promptsDatabase) {
						return `
							<div class="instaraw-rpg-empty">
								<p>Database not loaded</p>
								<button class="instaraw-rpg-btn-primary instaraw-rpg-reload-db-btn">🔄 Load Database</button>
							</div>
						`;
					}

					switch (activeTab) {
						case "library":
							return renderLibraryTab();
						case "creative":
							return renderCreativeTab(uiState);
						case "character":
							return renderCharacterTab();
						default:
							return "";
					}
				};

				// === Library Tab ===
				const renderLibraryTab = () => {
					const filters = JSON.parse(node.properties.library_filters || "{}");
					const bookmarks = JSON.parse(node.properties.bookmarks || "[]");

					// Apply filters
					let filteredPrompts = filterPrompts(promptsDatabase, filters);

					// Pagination
					const totalPages = Math.ceil(filteredPrompts.length / itemsPerPage);
					const startIdx = currentPage * itemsPerPage;
					const endIdx = startIdx + itemsPerPage;
					const pagePrompts = filteredPrompts.slice(startIdx, endIdx);

					return `
						<div class="instaraw-rpg-library">
							<div class="instaraw-rpg-filters">
								<input type="text" class="instaraw-rpg-search-input" placeholder="🔍 Search prompts..." value="${filters.search_query || ""}" />
								<div class="instaraw-rpg-filter-row">
									<select class="instaraw-rpg-filter-dropdown" data-filter="content_type">
										<option value="any">All Content Types</option>
										<option value="person" ${filters.content_type === "person" ? "selected" : ""}>Person</option>
										<option value="landscape" ${filters.content_type === "landscape" ? "selected" : ""}>Landscape</option>
										<option value="object" ${filters.content_type === "object" ? "selected" : ""}>Object</option>
									</select>
									<select class="instaraw-rpg-filter-dropdown" data-filter="safety_level">
										<option value="any">All Safety Levels</option>
										<option value="sfw" ${filters.safety_level === "sfw" ? "selected" : ""}>SFW</option>
										<option value="suggestive" ${filters.safety_level === "suggestive" ? "selected" : ""}>Suggestive</option>
										<option value="nsfw" ${filters.safety_level === "nsfw" ? "selected" : ""}>NSFW</option>
									</select>
									<select class="instaraw-rpg-filter-dropdown" data-filter="shot_type">
										<option value="any">All Shot Types</option>
										<option value="portrait" ${filters.shot_type === "portrait" ? "selected" : ""}>Portrait</option>
										<option value="full_body" ${filters.shot_type === "full_body" ? "selected" : ""}>Full Body</option>
										<option value="close_up" ${filters.shot_type === "close_up" ? "selected" : ""}>Close Up</option>
									</select>
									<button class="instaraw-rpg-btn-secondary instaraw-rpg-clear-filters-btn">✖ Clear</button>
								</div>
							</div>

							<div class="instaraw-rpg-library-grid">
								${
									pagePrompts.length === 0
										? `<div class="instaraw-rpg-empty"><p>No prompts found</p><p class="instaraw-rpg-hint">Try adjusting your filters</p></div>`
										: pagePrompts
												.map(
													(prompt) => `
									<div class="instaraw-rpg-library-card" data-id="${prompt.id}">
										<div class="instaraw-rpg-library-card-header">
											<button class="instaraw-rpg-bookmark-btn ${bookmarks.includes(prompt.id) ? "bookmarked" : ""}" data-id="${prompt.id}">
												${bookmarks.includes(prompt.id) ? "⭐" : "☆"}
											</button>
											<button class="instaraw-rpg-add-to-batch-btn" data-id="${prompt.id}">+ Add</button>
										</div>
										<div class="instaraw-rpg-library-card-content">
											<div class="instaraw-rpg-prompt-preview">${escapeHtml(prompt.prompt?.positive || "")}</div>
											<div class="instaraw-rpg-library-card-tags">
												${(prompt.tags || [])
													.slice(0, 5)
													.map((tag) => `<span class="instaraw-rpg-tag">${tag}</span>`)
													.join("")}
												${prompt.tags?.length > 5 ? `<span class="instaraw-rpg-tag-more">+${prompt.tags.length - 5}</span>` : ""}
											</div>
										</div>
									</div>
								`
												)
												.join("")
								}
							</div>

							${
								totalPages > 1
									? `
								<div class="instaraw-rpg-pagination">
									<button class="instaraw-rpg-btn-secondary instaraw-rpg-prev-page-btn" ${currentPage === 0 ? "disabled" : ""}>← Prev</button>
									<span class="instaraw-rpg-page-info">Page ${currentPage + 1} / ${totalPages} (${filteredPrompts.length} prompts)</span>
									<button class="instaraw-rpg-btn-secondary instaraw-rpg-next-page-btn" ${currentPage >= totalPages - 1 ? "disabled" : ""}>Next →</button>
								</div>
							`
									: ""
							}
						</div>
					`;
				};

				// === Creative Tab ===
				const renderCreativeTab = (uiState) => {
					const promptQueue = uiState?.promptQueue || [];
					const selectedForInspiration = promptQueue.filter((p) => p.source_id).slice(0, 5);
					const modelOptionsHtml = CREATIVE_MODEL_OPTIONS
						.map(
							(opt) => `<option value="${opt.value}" ${opt.value === (uiState?.currentCreativeModel || "gemini-2.5-pro") ? "selected" : ""}>${opt.label}</option>`
						)
						.join("");
					const temperature = uiState?.currentCreativeTemperature ?? 0.9;
					const topP = uiState?.currentCreativeTopP ?? 0.9;
					const systemPrompt = uiState?.creativeSystemPrompt || DEFAULT_RPG_SYSTEM_PROMPT;

					return `
						<div class="instaraw-rpg-model-settings">
							<div class="instaraw-rpg-model-row">
								<label>Creative Model</label>
								<select class="instaraw-rpg-model-select">
									${modelOptionsHtml}
								</select>
							</div>
							<div class="instaraw-rpg-model-grid">
								<div class="instaraw-rpg-model-control">
									<label>Temperature</label>
									<input type="number" class="instaraw-rpg-model-temp" value="${temperature}" min="0" max="2" step="0.01" />
								</div>
								<div class="instaraw-rpg-model-control">
									<label>Top P</label>
									<input type="number" class="instaraw-rpg-model-top-p" value="${topP}" min="0" max="1" step="0.01" />
								</div>
							</div>
							<div class="instaraw-rpg-model-row">
								<label>System Prompt</label>
								<textarea class="instaraw-rpg-system-prompt" rows="4">${escapeHtml(systemPrompt)}</textarea>
							</div>
						</div>
						<div class="instaraw-rpg-creative">
							<div class="instaraw-rpg-creative-header">
								<h3>Creative Prompt Generation</h3>
								<p>Generate variations based on Library prompts or create new prompts from scratch</p>
							</div>

							<div class="instaraw-rpg-inspiration-section">
								<label>Inspiration Sources (${selectedForInspiration.length})</label>
								<div class="instaraw-rpg-inspiration-list">
									${
										selectedForInspiration.length === 0
											? `<p class="instaraw-rpg-hint">Add prompts from Library to use as inspiration</p>`
											: selectedForInspiration
													.map(
														(p) => `
										<div class="instaraw-rpg-inspiration-item">
											<span class="instaraw-rpg-inspiration-text">${escapeHtml(p.positive_prompt || "")}</span>
										</div>
									`
													)
													.join("")
									}
								</div>
							</div>

							<div class="instaraw-rpg-creative-controls">
								<div class="instaraw-rpg-control-group">
									<label>Generation Count</label>
									<input type="number" class="instaraw-rpg-number-input instaraw-rpg-gen-count-input" value="5" min="1" max="50" />
								</div>
								<div class="instaraw-rpg-control-group">
									<label>Inspiration Count</label>
									<input type="number" class="instaraw-rpg-number-input instaraw-rpg-inspiration-count-input" value="3" min="0" max="${selectedForInspiration.length}" />
								</div>
								<div class="instaraw-rpg-control-group">
									<label>
										<input type="checkbox" class="instaraw-rpg-checkbox instaraw-rpg-is-sdxl-checkbox" />
										SDXL Mode
									</label>
								</div>
							</div>

							<button class="instaraw-rpg-btn-primary instaraw-rpg-generate-creative-btn">✨ Generate & Add to Batch</button>

							<div class="instaraw-rpg-creative-preview" style="display: none;">
								<h4>Generated Prompts Preview</h4>
								<div class="instaraw-rpg-creative-preview-list"></div>
								<button class="instaraw-rpg-btn-primary instaraw-rpg-accept-creative-btn">✓ Accept All</button>
								<button class="instaraw-rpg-btn-secondary instaraw-rpg-cancel-creative-btn">✖ Cancel</button>
							</div>
						</div>
					`;
				};

				// === Character Tab ===
				const renderCharacterTab = () => {
					return `
						<div class="instaraw-rpg-character">
							<div class="instaraw-rpg-creative-header">
								<h3>Character-Consistent Generation</h3>
								<p>Generate prompts with character reference for consistent results</p>
							</div>

							<div class="instaraw-rpg-control-group">
								<label>Character Reference</label>
								<textarea class="instaraw-rpg-character-ref-input" placeholder="e.g., 1girl_character_lora, blonde hair, blue eyes, athletic build..." rows="4"></textarea>
							</div>

							<div class="instaraw-rpg-creative-controls">
								<div class="instaraw-rpg-control-group">
									<label>Generation Count</label>
									<input type="number" class="instaraw-rpg-number-input instaraw-rpg-char-gen-count-input" value="5" min="1" max="50" />
								</div>
								<div class="instaraw-rpg-control-group">
									<label>
										<input type="checkbox" class="instaraw-rpg-checkbox instaraw-rpg-char-is-sdxl-checkbox" />
										SDXL Mode
									</label>
								</div>
							</div>

							<button class="instaraw-rpg-btn-primary instaraw-rpg-generate-character-btn">👤 Generate Character Prompts</button>

							<div class="instaraw-rpg-creative-preview" style="display: none;">
								<h4>Generated Character Prompts Preview</h4>
								<div class="instaraw-rpg-character-preview-list"></div>
								<button class="instaraw-rpg-btn-primary instaraw-rpg-accept-character-btn">✓ Accept All</button>
								<button class="instaraw-rpg-btn-secondary instaraw-rpg-cancel-character-btn">✖ Cancel</button>
							</div>
						</div>
					`;
				};

				// === Batch Panel (AIL Item Style) ===
				const renderBatchPanel = (promptQueue, totalGenerations) => {
					const gridContent =
						promptQueue.length === 0
							? `<div class="instaraw-rpg-empty"><p>No prompts in batch</p><p class="instaraw-rpg-hint">Add prompts from Library or Creative mode</p></div>`
							: promptQueue
									.map(
										(entry, idx) => `
							<div class="instaraw-rpg-batch-item" data-id="${entry.id}" data-idx="${idx}" draggable="true">
								<div class="instaraw-rpg-batch-item-header">
									<span class="instaraw-rpg-batch-item-number">#${idx + 1}</span>
									<div class="instaraw-rpg-batch-item-controls">
										<label>Repeat:</label>
										<input type="number" class="instaraw-rpg-repeat-input" data-id="${entry.id}" value="${entry.repeat_count || 1}" min="1" max="99" />
										<button class="instaraw-rpg-batch-delete-btn" data-id="${entry.id}" title="Remove prompt">×</button>
									</div>
								</div>
								<div class="instaraw-rpg-batch-item-content">
									<label>Positive Prompt</label>
									<textarea class="instaraw-rpg-prompt-textarea instaraw-rpg-positive-textarea" data-id="${entry.id}" rows="3">${entry.positive_prompt || ""}</textarea>

									<div class="instaraw-rpg-negative-toggle">
										<label>
											<input type="checkbox" class="instaraw-rpg-checkbox instaraw-rpg-negative-checkbox" data-id="${entry.id}" ${entry.negative_prompt ? "checked" : ""} />
											Custom Negative
										</label>
									</div>
									${
										entry.negative_prompt
											? `<textarea class="instaraw-rpg-prompt-textarea instaraw-rpg-negative-textarea" data-id="${entry.id}" rows="2">${entry.negative_prompt || ""}</textarea>`
											: ""
									}

									${
										entry.tags && entry.tags.length > 0
											? `
										<div class="instaraw-rpg-batch-item-tags">
											${entry.tags
												.slice(0, 3)
												.map((tag) => `<span class="instaraw-rpg-tag">${tag}</span>`)
												.join("")}
											${entry.tags.length > 3 ? `<span class="instaraw-rpg-tag-more">+${entry.tags.length - 3}</span>` : ""}
										</div>
									`
											: ""
									}
								</div>
							</div>
						`,
									)
									.join("");

					return `
						<div class="instaraw-rpg-batch-header">
							<div>
								<h3>Generation Batch</h3>
								<p class="instaraw-rpg-batch-subtitle">Drag prompts to sync with Advanced Image Loader order</p>
							</div>
							<div class="instaraw-rpg-batch-actions">
								<span class="instaraw-rpg-batch-count">${totalGenerations} generation${totalGenerations !== 1 ? "s" : ""}</span>
								${promptQueue.length > 0 ? `<button class="instaraw-rpg-btn-secondary instaraw-rpg-clear-batch-btn">🗑️ Clear All</button>` : ""}
							</div>
						</div>
						<div class="instaraw-rpg-batch-grid">
							${gridContent}
						</div>
					`;
				};

				// === Image Preview (AIL Sync) ===
				const renderImagePreview = (resolvedMode, totalGenerations) => {
					if (!node._linkedAILNodeId) {
						return `
							<div class="instaraw-rpg-image-preview-section">
								<div class="instaraw-rpg-image-preview-empty">
									<p>No image source linked</p>
									<p class="instaraw-rpg-hint">Connect an Advanced Image Loader to see preview</p>
								</div>
							</div>
						`;
					}

					const imageCount = node._linkedImageCount || 0;
					const images = node._linkedImages || [];
					const displayImages = images.slice(0, 10);
					const isMatch = totalGenerations === imageCount;
					const matchClass = isMatch ? "match" : "mismatch";
					const matchIcon = isMatch ? "✅" : "⚠️";

					return `
						<div class="instaraw-rpg-image-preview-section">
							<div class="instaraw-rpg-image-preview-header">
								<span>🖼️ Linked to Advanced Image Loader (Node #${node._linkedAILNodeId})</span>
								<span class="instaraw-rpg-validation-badge instaraw-rpg-validation-${matchClass}">
									${matchIcon} ${totalGenerations} prompts ↔ ${imageCount} images
								</span>
							</div>
							<div class="instaraw-rpg-image-preview-grid">
								${displayImages
									.map(
										(img) => `
									<div class="instaraw-rpg-preview-thumb">
										<img src="${img.url}" alt="Preview" />
									</div>
								`
									)
									.join("")}
								${images.length > 10 ? `<div class="instaraw-rpg-preview-more">+${images.length - 10} more</div>` : ""}
							</div>
						</div>
					`;
				};

				// === Filter Prompts ===
				const filterPrompts = (database, filters) => {
					if (!database) return [];

					let filtered = database;

					// Search query
					if (filters.search_query && filters.search_query.trim() !== "") {
						const query = filters.search_query.toLowerCase();
						filtered = filtered.filter((p) => {
							const positive = (p.prompt?.positive || "").toLowerCase();
							const tags = (p.tags || []).join(" ").toLowerCase();
							return positive.includes(query) || tags.includes(query);
						});
					}

					// Content type
					if (filters.content_type && filters.content_type !== "any") {
						filtered = filtered.filter((p) => p.classification?.content_type === filters.content_type);
					}

					// Safety level
					if (filters.safety_level && filters.safety_level !== "any") {
						filtered = filtered.filter((p) => p.classification?.safety_level === filters.safety_level);
					}

					// Shot type
					if (filters.shot_type && filters.shot_type !== "any") {
						filtered = filtered.filter((p) => p.classification?.shot_type === filters.shot_type);
					}

					return filtered;
				};

				// === Helper Functions ===
				const escapeHtml = (text) => {
					if (!text) return "";
					return text
						.replace(/&/g, "&amp;")
						.replace(/</g, "&lt;")
						.replace(/>/g, "&gt;")
						.replace(/"/g, "&quot;")
						.replace(/'/g, "&#39;");
				};

				const generateUniqueId = () => {
					return `${Date.now()}_${Math.random().toString(36).substring(2, 9)}`;
				};

				// === Add Prompt to Batch ===
					const addPromptToBatch = (promptData) => {
						const promptQueue = parsePromptBatch();
						const newEntry = {
							id: generateUniqueId(),
							positive_prompt: promptData.prompt?.positive || "",
							negative_prompt: promptData.prompt?.negative || "",
						repeat_count: 1,
						tags: promptData.tags || [],
						source_id: promptData.id || null,
					};
					promptQueue.push(newEntry);
					setPromptBatchData(promptQueue);
					renderUI();
					};

					// === Update Prompt in Batch ===
					const updatePromptInBatch = (id, field, value) => {
						const promptQueue = parsePromptBatch();
						const entry = promptQueue.find((p) => p.id === id);
						if (entry) {
							entry[field] = value;
							node.properties.prompt_queue_data = JSON.stringify(promptQueue);
							syncPromptBatchWidget();
						// Don't re-render for text edits to avoid losing focus
						if (field !== "positive_prompt" && field !== "negative_prompt") {
							renderUI();
						}
					}
					};

					// === Delete Prompt from Batch ===
					const deletePromptFromBatch = (id) => {
						const promptQueue = parsePromptBatch();
						const filtered = promptQueue.filter((p) => p.id !== id);
						setPromptBatchData(filtered);
						renderUI();
					};

				// === Clear Batch ===
				const clearBatch = () => {
					if (!confirm("Clear all prompts from batch?")) return;
					setPromptBatchData([]);
					renderUI();
				};

				// === Toggle Bookmark ===
				const toggleBookmark = (promptId) => {
					const bookmarks = JSON.parse(node.properties.bookmarks || "[]");
					const idx = bookmarks.indexOf(promptId);
					if (idx >= 0) {
						bookmarks.splice(idx, 1);
					} else {
						bookmarks.push(promptId);
					}
					node.properties.bookmarks = JSON.stringify(bookmarks);
					renderUI();
				};

				// === Update Filters ===
				const updateFilter = (filterName, value) => {
					const filters = JSON.parse(node.properties.library_filters || "{}");
					filters[filterName] = value;
					node.properties.library_filters = JSON.stringify(filters);
					currentPage = 0; // Reset to first page
					renderUI();
				};

				// === Clear Filters ===
				const clearFilters = () => {
					node.properties.library_filters = JSON.stringify({
						tags: [],
						content_type: "any",
						safety_level: "any",
						shot_type: "any",
						quality: "any",
						search_query: "",
					});
					currentPage = 0;
					renderUI();
				};

				// === Generate Creative Prompts ===
				const generateCreativePrompts = async () => {
					const genCountInput = container.querySelector(".instaraw-rpg-gen-count-input");
					const inspirationCountInput = container.querySelector(".instaraw-rpg-inspiration-count-input");
					const isSDXLCheckbox = container.querySelector(".instaraw-rpg-is-sdxl-checkbox");
					const generateBtn = container.querySelector(".instaraw-rpg-generate-creative-btn");

					const generationCount = parseInt(genCountInput?.value || "5");
					const inspirationCount = parseInt(inspirationCountInput?.value || "3");
					const isSDXL = isSDXLCheckbox?.checked || false;

					const promptQueue = parsePromptBatch();
					const sourcePrompts = promptQueue.filter((p) => p.source_id).slice(0, inspirationCount);

					const modelWidget = node.widgets?.find((w) => w.name === "creative_model");
					const model = modelWidget?.value || node.properties.creative_model || "gemini-2.5-pro";
					const systemPrompt = node.properties.creative_system_prompt || DEFAULT_RPG_SYSTEM_PROMPT;
					const temperatureValue = parseFloat(node.properties.creative_temperature ?? 0.9) || 0.9;
					const topPValue = parseFloat(node.properties.creative_top_p ?? 0.9) || 0.9;

					// Get API keys from widgets
					const geminiKeyWidget = node.widgets?.find((w) => w.name === "gemini_api_key");
					const grokKeyWidget = node.widgets?.find((w) => w.name === "grok_api_key");
					const geminiApiKey = (geminiKeyWidget?.value || "").trim() || window.INSTARAW_GEMINI_KEY || "";
					const grokApiKey = (grokKeyWidget?.value || "").trim() || window.INSTARAW_GROK_KEY || "";

					// Disable button and show loading
					if (generateBtn) {
						generateBtn.disabled = true;
						generateBtn.textContent = "⏳ Generating...";
					}

					try {
					const response = await api.fetchApi("/instaraw/generate_creative_prompts", {
							method: "POST",
							headers: { "Content-Type": "application/json" },
							body: JSON.stringify({
								source_prompts: sourcePrompts.map((p) => ({
									id: p.source_id,
									prompt: {
										positive: p.positive_prompt,
										negative: p.negative_prompt,
									},
								})),
								generation_count: generationCount,
								inspiration_count: inspirationCount,
								is_sdxl: isSDXL,
								character_reference: "",
								model: model,
								gemini_api_key: geminiApiKey,
								grok_api_key: grokApiKey,
								system_prompt: systemPrompt,
								temperature: temperatureValue,
								top_p: topPValue,
							}),
						});

						const result = await parseJSONResponse(response);

						if (!response.ok) {
							throw new Error(result?.error || `Creative API error ${response.status}`);
						}

						if (result.success && result.prompts) {
							// Show preview
							const previewSection = container.querySelector(".instaraw-rpg-creative-preview");
							const previewList = container.querySelector(".instaraw-rpg-creative-preview-list");

							if (previewSection && previewList) {
								previewList.innerHTML = result.prompts
									.map(
										(p, idx) => `
									<div class="instaraw-rpg-preview-item">
										<strong>#${idx + 1}</strong>
										<p>${escapeHtml(p.positive || "")}</p>
									</div>
								`
									)
									.join("");

								previewSection.style.display = "block";
								// Store generated prompts temporarily
								node._generatedCreativePrompts = result.prompts;
								setupEventHandlers();
							}
						} else {
							throw new Error(result.error || "Unknown error");
						}
					} catch (error) {
						alert(`Creative generation error: ${error.message || error}`);
					} finally {
						if (generateBtn) {
							generateBtn.disabled = false;
							generateBtn.textContent = "✨ Generate & Add to Batch";
						}
					}
				};

				// === Accept Creative Prompts ===
				const acceptCreativePrompts = () => {
					if (!node._generatedCreativePrompts) return;

						const promptQueue = parsePromptBatch();
					node._generatedCreativePrompts.forEach((p) => {
						promptQueue.push({
							id: generateUniqueId(),
							positive_prompt: p.positive || "",
							negative_prompt: p.negative || "",
							repeat_count: 1,
							tags: p.tags || [],
							source_id: null,
						});
					});

						setPromptBatchData(promptQueue);
					delete node._generatedCreativePrompts;
					renderUI();
				};

				// === Cancel Creative Prompts ===
				const cancelCreativePrompts = () => {
					delete node._generatedCreativePrompts;
					renderUI();
				};

				// === Generate Character Prompts ===
				const generateCharacterPrompts = async () => {
					const charRefInput = container.querySelector(".instaraw-rpg-character-ref-input");
					const genCountInput = container.querySelector(".instaraw-rpg-char-gen-count-input");
					const isSDXLCheckbox = container.querySelector(".instaraw-rpg-char-is-sdxl-checkbox");
					const generateBtn = container.querySelector(".instaraw-rpg-generate-character-btn");

					const characterReference = charRefInput?.value || "";
					const generationCount = parseInt(genCountInput?.value || "5");
					const isSDXL = isSDXLCheckbox?.checked || false;

					const modelWidget = node.widgets?.find((w) => w.name === "creative_model");
					const model = modelWidget?.value || node.properties.creative_model || "gemini-2.5-pro";
					const systemPrompt = node.properties.creative_system_prompt || DEFAULT_RPG_SYSTEM_PROMPT;
					const temperatureValue = parseFloat(node.properties.creative_temperature ?? 0.9) || 0.9;
					const topPValue = parseFloat(node.properties.creative_top_p ?? 0.9) || 0.9;

					// Get API keys from widgets
					const geminiKeyWidget = node.widgets?.find((w) => w.name === "gemini_api_key");
					const grokKeyWidget = node.widgets?.find((w) => w.name === "grok_api_key");
					const geminiApiKey = (geminiKeyWidget?.value || "").trim() || window.INSTARAW_GEMINI_KEY || "";
					const grokApiKey = (grokKeyWidget?.value || "").trim() || window.INSTARAW_GROK_KEY || "";

					if (!characterReference.trim()) {
						alert("Please enter a character reference");
						return;
					}

					// Disable button and show loading
					if (generateBtn) {
						generateBtn.disabled = true;
						generateBtn.textContent = "⏳ Generating...";
					}

					try {
					const response = await api.fetchApi("/instaraw/generate_creative_prompts", {
							method: "POST",
							headers: { "Content-Type": "application/json" },
							body: JSON.stringify({
								source_prompts: [],
								generation_count: generationCount,
								inspiration_count: 0,
								is_sdxl: isSDXL,
								character_reference: characterReference,
								model: model,
								gemini_api_key: geminiApiKey,
								grok_api_key: grokApiKey,
								system_prompt: systemPrompt,
								temperature: temperatureValue,
								top_p: topPValue,
							}),
						});

						const result = await parseJSONResponse(response);

						if (!response.ok) {
							throw new Error(result?.error || `Creative API error ${response.status}`);
						}

						if (result.success && result.prompts) {
							// Show preview
							const previewSection = container.querySelector(".instaraw-rpg-creative-preview");
							const previewList = container.querySelector(".instaraw-rpg-character-preview-list");

							if (previewSection && previewList) {
								previewList.innerHTML = result.prompts
									.map(
										(p, idx) => `
									<div class="instaraw-rpg-preview-item">
										<strong>#${idx + 1}</strong>
										<p>${escapeHtml(p.positive || "")}</p>
									</div>
								`
									)
									.join("");

								previewSection.style.display = "block";
								// Store generated prompts temporarily
								node._generatedCharacterPrompts = result.prompts;
								setupEventHandlers();
							}
						} else {
							throw new Error(result.error || "Unknown error");
						}
					} catch (error) {
						alert(`Character generation error: ${error.message || error}`);
					} finally {
						if (generateBtn) {
							generateBtn.disabled = false;
							generateBtn.textContent = "👤 Generate Character Prompts";
						}
					}
				};

				// === Accept Character Prompts ===
				const acceptCharacterPrompts = () => {
					if (!node._generatedCharacterPrompts) return;

						const promptQueue = parsePromptBatch();
					node._generatedCharacterPrompts.forEach((p) => {
						promptQueue.push({
							id: generateUniqueId(),
							positive_prompt: p.positive || "",
							negative_prompt: p.negative || "",
							repeat_count: 1,
							tags: p.tags || [],
							source_id: null,
						});
					});

						setPromptBatchData(promptQueue);
					delete node._generatedCharacterPrompts;
					renderUI();
				};

				// === Cancel Character Prompts ===
				const cancelCharacterPrompts = () => {
					delete node._generatedCharacterPrompts;
					renderUI();
				};

				// === Event Handlers Setup (Following AIL Pattern) ===
				const setupEventHandlers = () => {
					// Tab switching
					container.querySelectorAll(".instaraw-rpg-tab").forEach((tab) => {
						tab.onclick = () => {
							node.properties.active_tab = tab.dataset.tab;
							renderUI();
						};
					});

					// Mode dropdown
					const modeDropdown = container.querySelector(".instaraw-rpg-mode-dropdown");
					if (modeDropdown) {
						modeDropdown.onchange = (e) => {
							const modeWidget = node.widgets?.find((w) => w.name === "mode");
							if (modeWidget) {
								modeWidget.value = e.target.value;
								renderUI();
							}
						};
					}

					// Creative model settings
					const creativeModelSelect = container.querySelector(".instaraw-rpg-model-select");
					if (creativeModelSelect) {
						creativeModelSelect.onchange = (e) => {
							node.properties.creative_model = e.target.value;
							const widget = node.widgets?.find((w) => w.name === "creative_model");
							if (widget) widget.value = e.target.value;
							renderUI();
						};
					}

					const creativeTempInput = container.querySelector(".instaraw-rpg-model-temp");
					if (creativeTempInput) {
						creativeTempInput.onchange = (e) => {
							const val = parseFloat(e.target.value) || 0.9;
							node.properties.creative_temperature = val;
							app.graph.setDirtyCanvas(true, true);
						};
					}

					const creativeTopPInput = container.querySelector(".instaraw-rpg-model-top-p");
					if (creativeTopPInput) {
						creativeTopPInput.onchange = (e) => {
							const val = parseFloat(e.target.value) || 0.9;
							node.properties.creative_top_p = val;
							app.graph.setDirtyCanvas(true, true);
						};
					}

					const systemPromptInput = container.querySelector(".instaraw-rpg-system-prompt");
					if (systemPromptInput) {
						systemPromptInput.oninput = (e) => {
							node.properties.creative_system_prompt = e.target.value;
							app.graph.setDirtyCanvas(true, true);
						};
					}

					// Reload database button
					const reloadDBBtn = container.querySelector(".instaraw-rpg-reload-db-btn");
					if (reloadDBBtn) {
						reloadDBBtn.onclick = () => loadPromptsDatabase();
					}

					// Library search
					const searchInput = container.querySelector(".instaraw-rpg-search-input");
					if (searchInput) {
						searchInput.oninput = (e) => {
							clearTimeout(node._searchTimeout);
							node._searchTimeout = setTimeout(() => {
								updateFilter("search_query", e.target.value);
							}, 300);
						};
					}

					// Library filters
					container.querySelectorAll(".instaraw-rpg-filter-dropdown").forEach((dropdown) => {
						dropdown.onchange = (e) => {
							updateFilter(dropdown.dataset.filter, e.target.value);
						};
					});

					// Clear filters
					const clearFiltersBtn = container.querySelector(".instaraw-rpg-clear-filters-btn");
					if (clearFiltersBtn) {
						clearFiltersBtn.onclick = clearFilters;
					}

					// Add to batch buttons
					container.querySelectorAll(".instaraw-rpg-add-to-batch-btn").forEach((btn) => {
						btn.onclick = () => {
							const promptId = btn.dataset.id;
							const promptData = promptsDatabase.find((p) => p.id === promptId);
							if (promptData) addPromptToBatch(promptData);
						};
					});

					// Bookmark buttons
					container.querySelectorAll(".instaraw-rpg-bookmark-btn").forEach((btn) => {
						btn.onclick = (e) => {
							e.stopPropagation();
							toggleBookmark(btn.dataset.id);
						};
					});

					// Pagination
					const prevPageBtn = container.querySelector(".instaraw-rpg-prev-page-btn");
					if (prevPageBtn) {
						prevPageBtn.onclick = () => {
							if (currentPage > 0) {
								currentPage--;
								renderUI();
							}
						};
					}

					const nextPageBtn = container.querySelector(".instaraw-rpg-next-page-btn");
					if (nextPageBtn) {
						nextPageBtn.onclick = () => {
							currentPage++;
							renderUI();
						};
					}

					// Batch item controls
					container.querySelectorAll(".instaraw-rpg-positive-textarea").forEach((textarea) => {
						autoResizeTextarea(textarea);
						textarea.oninput = (e) => autoResizeTextarea(textarea);
						textarea.onchange = (e) => {
							updatePromptInBatch(textarea.dataset.id, "positive_prompt", e.target.value);
						};
						textarea.onmousedown = (e) => e.stopPropagation();
					});

					container.querySelectorAll(".instaraw-rpg-negative-textarea").forEach((textarea) => {
						autoResizeTextarea(textarea);
						textarea.oninput = (e) => autoResizeTextarea(textarea);
						textarea.onchange = (e) => {
							updatePromptInBatch(textarea.dataset.id, "negative_prompt", e.target.value);
						};
						textarea.onmousedown = (e) => e.stopPropagation();
					});

					container.querySelectorAll(".instaraw-rpg-negative-checkbox").forEach((checkbox) => {
						checkbox.onchange = (e) => {
							if (!e.target.checked) {
								updatePromptInBatch(checkbox.dataset.id, "negative_prompt", "");
							} else {
								renderUI(); // Re-render to show textarea
							}
						};
					});

					container.querySelectorAll(".instaraw-rpg-repeat-input").forEach((input) => {
						input.onchange = (e) => {
							updatePromptInBatch(input.dataset.id, "repeat_count", parseInt(e.target.value) || 1);
						};
						input.onmousedown = (e) => e.stopPropagation();
					});

					container.querySelectorAll(".instaraw-rpg-batch-delete-btn").forEach((btn) => {
						btn.onclick = (e) => {
							e.stopPropagation();
							deletePromptFromBatch(btn.dataset.id);
						};
					});

					// Clear batch button
					const clearBatchBtn = container.querySelector(".instaraw-rpg-clear-batch-btn");
					if (clearBatchBtn) {
						clearBatchBtn.onclick = clearBatch;
					}

					// Creative mode buttons
					const generateCreativeBtn = container.querySelector(".instaraw-rpg-generate-creative-btn");
					if (generateCreativeBtn) {
						generateCreativeBtn.onclick = generateCreativePrompts;
					}

					const acceptCreativeBtn = container.querySelector(".instaraw-rpg-accept-creative-btn");
					if (acceptCreativeBtn) {
						acceptCreativeBtn.onclick = acceptCreativePrompts;
					}

					const cancelCreativeBtn = container.querySelector(".instaraw-rpg-cancel-creative-btn");
					if (cancelCreativeBtn) {
						cancelCreativeBtn.onclick = cancelCreativePrompts;
					}

					// Character mode buttons
					const generateCharacterBtn = container.querySelector(".instaraw-rpg-generate-character-btn");
					if (generateCharacterBtn) {
						generateCharacterBtn.onclick = generateCharacterPrompts;
					}

					const acceptCharacterBtn = container.querySelector(".instaraw-rpg-accept-character-btn");
					if (acceptCharacterBtn) {
						acceptCharacterBtn.onclick = acceptCharacterPrompts;
					}

					const cancelCharacterBtn = container.querySelector(".instaraw-rpg-cancel-character-btn");
					if (cancelCharacterBtn) {
						cancelCharacterBtn.onclick = cancelCharacterPrompts;
					}
				};

				// === Drag and Drop (Exact AIL Pattern) ===
				const setupDragAndDrop = () => {
					const items = container.querySelectorAll(".instaraw-rpg-batch-item");
					let draggedItem = null;

					items.forEach((item) => {
						item.addEventListener("dragstart", (e) => {
							draggedItem = item;
							item.style.opacity = "0.5";
							e.dataTransfer.effectAllowed = "move";
							e.stopPropagation();
							e.dataTransfer.setData("text/plain", "instaraw-rpg-reorder");
						});

						item.addEventListener("dragend", () => {
							item.style.opacity = "1";
							items.forEach((i) => i.classList.remove("instaraw-rpg-drop-before", "instaraw-rpg-drop-after"));
						});

						item.addEventListener("dragover", (e) => {
							e.preventDefault();
							if (draggedItem === item) return;
							e.dataTransfer.dropEffect = "move";
							const rect = item.getBoundingClientRect();
							const midpoint = rect.top + rect.height / 2;
							items.forEach((i) => i.classList.remove("instaraw-rpg-drop-before", "instaraw-rpg-drop-after"));
							item.classList.add(e.clientY < midpoint ? "instaraw-rpg-drop-before" : "instaraw-rpg-drop-after");
						});

						item.addEventListener("drop", (e) => {
							e.preventDefault();
							if (draggedItem === item) return;

							const draggedId = draggedItem.dataset.id;
							const targetId = item.dataset.id;

								const promptQueue = parsePromptBatch();
							const draggedIndex = promptQueue.findIndex((p) => p.id === draggedId);
							const targetIndex = promptQueue.findIndex((p) => p.id === targetId);

							if (draggedIndex === -1 || targetIndex === -1) return;

							const [draggedEntry] = promptQueue.splice(draggedIndex, 1);
							const rect = item.getBoundingClientRect();
							const insertAfter = e.clientY > rect.top + rect.height / 2;

							const newTargetIndex = promptQueue.findIndex((p) => p.id === targetId);
							promptQueue.splice(insertAfter ? newTargetIndex + 1 : newTargetIndex, 0, draggedEntry);

								setPromptBatchData(promptQueue);
							renderUI();
						});
					});
				};

				// === AIL Update Listener ===
				window.addEventListener("INSTARAW_AIL_UPDATED", (event) => {
					const { nodeId, images, total } = event.detail;
					node._linkedAILNodeId = nodeId;
					node._linkedImages = images;
					node._linkedImageCount = total;

					// Update expected_image_count widget
					const widget = node.widgets?.find((w) => w.name === "expected_image_count");
					if (widget) widget.value = total;

					renderUI();
				});

				// === Add DOM Widget (Exact AIL Pattern) ===
				const widget = node.addDOMWidget("rpg_display", "rpgpromptmanager", container, {
					getValue: () => node.properties.prompt_queue_data,
					setValue: (v) => {
						node.properties.prompt_queue_data = v;
						renderUI();
					},
					serialize: false,
				});

				widget.computeSize = (width) => [width, cachedHeight + 2];

				// Store references for lifecycle hooks
				node._updateCachedHeight = updateCachedHeight;
				node._renderUI = renderUI;

				// Initial setup
				setTimeout(() => {
					syncPromptBatchWidget();
					loadPromptsDatabase();
					renderUI();
				}, 100);
			};

			// === onResize Hook (Exact AIL Pattern) ===
			const onResize = nodeType.prototype.onResize;
			nodeType.prototype.onResize = function (size) {
				onResize?.apply(this, arguments);
				if (this._updateCachedHeight) {
					clearTimeout(this._resizeTimeout);
					this._resizeTimeout = setTimeout(() => this._updateCachedHeight(), 50);
				}
			};

			// === onConfigure Hook (Exact AIL Pattern) ===
			const onConfigure = nodeType.prototype.onConfigure;
			nodeType.prototype.onConfigure = function (data) {
				onConfigure?.apply(this, arguments);
				setTimeout(() => {
					const promptQueueWidget = this.widgets?.find((w) => w.name === "prompt_queue_data");
					if (promptQueueWidget) promptQueueWidget.value = this.properties.prompt_queue_data || "[]";
					if (this._renderUI) this._renderUI();
				}, 200);
			};
		}
	},
});