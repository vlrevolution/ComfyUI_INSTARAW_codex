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
				node._linkedLatents = [];
				node._linkedImageCount = 0;
				node._linkedAILMode = null;

				// Pagination state
				let currentPage = 0;
				const itemsPerPage = 6;
				let reorderModeEnabled = false;
				let sdxlModeEnabled = false;

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

				// Get target output dimensions from aspect ratio selector
				const getTargetDimensions = () => {
					const widthWidget = node.widgets?.find(w => w.name === "output_width");
					const heightWidget = node.widgets?.find(w => w.name === "output_height");
					const aspectWidget = node.widgets?.find(w => w.name === "aspect_label");

					return {
						width: widthWidget?.value || 1024,
						height: heightWidget?.value || 1024,
						aspect_label: aspectWidget?.value || "1:1"
					};
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
						{ id: "generate", label: "Generate", icon: "🎯" },
					];

					const generationModeLabel = generationMode === "one_per_entry" ? "1 image per entry" : "Respect repeat counts";
					const linkedImages = node._linkedImageCount || 0;

					// Use detected mode from AIL if available
					const detectedMode = node._linkedAILMode || resolvedMode;
					const isDetectedFromAIL = node._linkedAILMode !== null;

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
								<div class="instaraw-rpg-mode-indicator-container">
									<span class="instaraw-rpg-mode-badge ${detectedMode === 'img2img' ? 'instaraw-rpg-mode-img2img' : 'instaraw-rpg-mode-txt2img'}">
										${detectedMode === 'img2img' ? '🖼️ IMG2IMG MODE' : '🎨 TXT2IMG MODE'}
									</span>
									${isDetectedFromAIL ? `<span class="instaraw-rpg-mode-source">From AIL #${node._linkedAILNodeId}</span>` : ''}
								</div>
								<div class="instaraw-rpg-mode-meta">
									<span class="instaraw-rpg-mode-pill">${generationModeLabel}</span>
								</div>
							</div>
							<div class="instaraw-rpg-kpi-row">
								<div class="instaraw-rpg-kpi">
									<span>Queue</span>
									<strong>${promptQueue.length}</strong>
								</div>
								<div class="instaraw-rpg-kpi">
									<span>Generations</span>
									<strong>${totalGenerations}</strong>
								</div>
								<div class="instaraw-rpg-kpi">
									<span>Images</span>
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
								<span class="instaraw-rpg-stat-badge">Gen: ${totalGenerations}</span>
								<span class="instaraw-rpg-stat-label">${resolvedMode}</span>
								${node._linkedAILNodeId ? `<span class="instaraw-rpg-stat-label">AIL #${node._linkedAILNodeId}</span>` : `<span class="instaraw-rpg-stat-label">No AIL</span>`}
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
						case "generate":
							return renderGenerateTab(uiState);
						case "creative":  // LEGACY - Fallback for old workflows
							return renderCreativeTab(uiState);
						case "character":  // LEGACY - Fallback for old workflows
							return renderCharacterTab();
						default:
							return "";
					}
				};

					// === Library Tab ===
				const renderLibraryTab = () => {
					const filters = JSON.parse(node.properties.library_filters || "{}");
					const bookmarks = JSON.parse(node.properties.bookmarks || "[]");
					const promptQueue = parsePromptBatch();
					const batchSourceIds = new Set(promptQueue.map(p => p.source_id).filter(Boolean));

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
									</select>
									<label class="instaraw-rpg-checkbox-label" title="Show only bookmarked prompts">
										<input type="checkbox" class="instaraw-rpg-show-bookmarked-checkbox" ${filters.show_bookmarked ? "checked" : ""} />
										⭐ Favorites Only
									</label>
									<button class="instaraw-rpg-btn-secondary instaraw-rpg-clear-filters-btn">✖ Clear</button>
								</div>
							</div>

							<div class="instaraw-rpg-library-header">
								<div style="display: flex; align-items: center; gap: 12px;">
									<span class="instaraw-rpg-result-count">
										${filteredPrompts.length} prompt${filteredPrompts.length === 1 ? '' : 's'} found
										${filters.show_bookmarked ? ' (favorites only)' : ''}
										${totalPages > 1 ? ` • Page ${currentPage + 1} of ${totalPages}` : ''}
									</span>
									<label class="instaraw-rpg-sdxl-toggle" title="SDXL mode - show tags as main content">
										<input type="checkbox" class="instaraw-rpg-sdxl-mode-checkbox" ${filters.sdxl_mode ? "checked" : ""} />
										🎨 SDXL
									</label>
								</div>
							</div>

							${
								totalPages > 1
									? `
								<div class="instaraw-rpg-pagination instaraw-rpg-pagination-top">
									<button class="instaraw-rpg-btn-secondary instaraw-rpg-prev-page-btn" ${currentPage === 0 ? "disabled" : ""}>← Prev</button>
									<span class="instaraw-rpg-page-info">Page ${currentPage + 1} / ${totalPages}</span>
									<button class="instaraw-rpg-btn-secondary instaraw-rpg-next-page-btn" ${currentPage >= totalPages - 1 ? "disabled" : ""}>Next →</button>
								</div>
								`
									: ""
							}

							<div class="instaraw-rpg-library-grid">
								${
									pagePrompts.length === 0
										? `<div class="instaraw-rpg-empty"><p>No prompts found</p><p class="instaraw-rpg-hint">Try adjusting your filters</p></div>`
										: pagePrompts
												.map(
													(prompt) => {
														const batchCount = promptQueue.filter(p => p.source_id === prompt.id).length;
														const positive = prompt.prompt?.positive || "";
														const negative = prompt.prompt?.negative || "";

														// Debug: Log if positive is empty but negative isn't
														if (!positive && negative) {
															console.warn(`[RPG] Prompt ${prompt.id} has empty positive but has negative:`, {
																id: prompt.id,
																positive: positive,
																negative: negative,
																fullPrompt: prompt.prompt,
																tags: prompt.tags
															});
														}

														// Show positive, or if empty show negative with warning, or show placeholder
														const displayText = positive
															? positive
															: negative
																? `[No positive prompt] ${negative}`
																: "[Empty prompt]";

														const searchQuery = filters.search_query?.trim() || "";
														const sdxlMode = filters.sdxl_mode || false;
														const matchType = prompt._matchType;
														const matchBadge = matchType === 'both' ? '📝🏷️' : matchType === 'prompt' ? '📝' : matchType === 'tags' ? '🏷️' : '';

														const allTags = prompt.tags || [];
														const autoExpand = matchType === 'tags' || matchType === 'both'; // Auto-expand if tags match

														return `
									<div class="instaraw-rpg-library-card ${batchCount > 0 ? 'in-batch' : ''}" data-id="${prompt.id}">
										<div class="instaraw-rpg-library-card-header">
											<button class="instaraw-rpg-bookmark-btn ${bookmarks.includes(prompt.id) ? "bookmarked" : ""}" data-id="${prompt.id}">
												${bookmarks.includes(prompt.id) ? "⭐" : "☆"}
											</button>
											<div class="instaraw-rpg-batch-controls">
												<button class="instaraw-rpg-add-to-batch-btn" data-id="${prompt.id}">+ Add</button>
												${batchCount > 0 ? `<button class="instaraw-rpg-undo-batch-btn" data-id="${prompt.id}">↶ ${batchCount}</button>` : ''}
											</div>
										</div>
										<div class="instaraw-rpg-library-card-content">
											${matchBadge ? `<div class="instaraw-rpg-match-badge">${matchBadge} Match</div>` : ''}

											${sdxlMode ? `
												<!-- SDXL Mode: Show tags as comma-separated text -->
												<div class="instaraw-rpg-prompt-preview">
													${allTags.map((tag) => highlightSearchTerm(tag, searchQuery)).join(", ")}
												</div>
											` : `
												<!-- Normal Mode: Prompt primary, tags secondary with expand/collapse -->
												<div class="instaraw-rpg-prompt-preview ${!positive ? 'instaraw-rpg-error-text' : ''}">${highlightSearchTerm(displayText, searchQuery)}</div>
												<div class="instaraw-rpg-library-card-tags" data-expanded="${autoExpand}" data-prompt-id="${prompt.id}">
													${autoExpand || allTags.length <= 5
														? allTags.map((tag) => `<span class="instaraw-rpg-tag">${highlightSearchTerm(tag, searchQuery)}</span>`).join("")
														: allTags.slice(0, 5).map((tag) => `<span class="instaraw-rpg-tag">${highlightSearchTerm(tag, searchQuery)}</span>`).join("")
													}
													${allTags.length > 5 ? `<button class="instaraw-rpg-toggle-tags-btn" data-id="${prompt.id}">${autoExpand ? 'Show less' : '+' + (allTags.length - 5)}</button>` : ""}
												</div>
											`}
										</div>
									</div>
								`;
													}
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
								<div class="instaraw-rpg-control-group">
									<label title="Bypass cache and generate fresh results (costs API credits)">
										<input type="checkbox" class="instaraw-rpg-checkbox instaraw-rpg-force-regenerate-checkbox" />
										🔄 Force Regenerate
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

				// === Unified Generate Tab ===
				const renderGenerateTab = (uiState) => {
					const promptQueue = uiState?.promptQueue || [];
					const detectedMode = node._linkedAILMode || "img2img";
					const modelOptionsHtml = CREATIVE_MODEL_OPTIONS
						.map(
							(opt) => `<option value="${opt.value}" ${opt.value === (uiState?.currentCreativeModel || "gemini-2.5-pro") ? "selected" : ""}>${opt.label}</option>`
						)
						.join("");
					const temperature = uiState?.currentCreativeTemperature ?? 0.9;
					const topP = uiState?.currentCreativeTopP ?? 0.9;

					// Character likeness state
					const characterDescription = node.properties.cached_character_description || "";
					const characterCacheKey = node.properties.character_cache_key || "";
					const isCached = characterDescription && characterCacheKey;

					return `
						<div class="instaraw-rpg-generate-unified">
							<!-- Character Likeness Section -->
							<div class="instaraw-rpg-section">
								<div class="instaraw-rpg-section-header">
									<label class="instaraw-rpg-checkbox-label">
										<input type="checkbox" class="instaraw-rpg-checkbox instaraw-rpg-enable-character-checkbox" ${node.properties.use_character_likeness ? 'checked' : ''} />
										<span>Character Likeness</span>
									</label>
									<span class="instaraw-rpg-hint-badge">${isCached ? '✅ Cached' : '⚪ Not generated'}</span>
								</div>
								<div class="instaraw-rpg-character-section" style="display: ${node.properties.use_character_likeness ? 'block' : 'none'};">
									<div class="instaraw-rpg-control-group">
										<label>Character Description</label>
										<textarea class="instaraw-rpg-character-text-input" placeholder="Describe your character or leave empty to use character image from input..." rows="3">${escapeHtml(node.properties.character_text_input || "")}</textarea>
									</div>
									<div class="instaraw-rpg-character-actions">
										<button class="instaraw-rpg-btn-secondary instaraw-rpg-generate-character-desc-btn">
											${isCached ? '🔄 Refresh' : '✨ Generate'} Character Description
										</button>
										${isCached ? `<button class="instaraw-rpg-btn-text instaraw-rpg-view-character-desc-btn">👁️ View</button>` : ''}
									</div>
									${characterDescription ? `
										<div class="instaraw-rpg-character-preview">
											<strong>Current Description:</strong>
											<p class="instaraw-rpg-character-desc-text">${escapeHtml(characterDescription.substring(0, 150))}${characterDescription.length > 150 ? '...' : ''}</p>
										</div>
									` : ''}
								</div>
							</div>

							<!-- Mode Detection & Settings -->
							<div class="instaraw-rpg-section">
								<div class="instaraw-rpg-section-header">
									<span class="instaraw-rpg-mode-badge ${detectedMode === 'img2img' ? 'instaraw-rpg-mode-img2img' : 'instaraw-rpg-mode-txt2img'}">
										${detectedMode === 'img2img' ? '🖼️ IMG2IMG' : '🎨 TXT2IMG'}
									</span>
									<span class="instaraw-rpg-hint-text">Detected from ${node._linkedAILNodeId ? `AIL #${node._linkedAILNodeId}` : 'Advanced Image Loader'}</span>
								</div>

								<!-- img2img: Affect Elements -->
								${detectedMode === 'img2img' ? `
									<div class="instaraw-rpg-affect-elements">
										<label class="instaraw-rpg-section-label">Affect Elements (unchecked = describe as-is)</label>
										<div class="instaraw-rpg-checkbox-grid">
											<label class="instaraw-rpg-checkbox-label">
												<input type="checkbox" class="instaraw-rpg-checkbox instaraw-rpg-affect-background" />
												<span>Background</span>
											</label>
											<label class="instaraw-rpg-checkbox-label">
												<input type="checkbox" class="instaraw-rpg-checkbox instaraw-rpg-affect-outfit" />
												<span>Outfit</span>
											</label>
											<label class="instaraw-rpg-checkbox-label">
												<input type="checkbox" class="instaraw-rpg-checkbox instaraw-rpg-affect-pose" />
												<span>Pose</span>
											</label>
											<label class="instaraw-rpg-checkbox-label">
												<input type="checkbox" class="instaraw-rpg-checkbox instaraw-rpg-affect-lighting" />
												<span>Lighting</span>
											</label>
										</div>
									</div>
								` : `
									<!-- txt2img: User Input & Library Inspiration -->
									<div class="instaraw-rpg-txt2img-settings">
										<div class="instaraw-rpg-control-group">
											<label>User Input (optional)</label>
											<textarea class="instaraw-rpg-user-text-input" placeholder="Describe what you want to generate or leave empty to use only library prompts..." rows="3">${escapeHtml(node.properties.user_text_input || "")}</textarea>
										</div>
										<div class="instaraw-rpg-library-controls">
											<label>Library Inspiration</label>
											<div class="instaraw-rpg-control-row">
												<input type="number" class="instaraw-rpg-number-input instaraw-rpg-inspiration-count" value="${node.properties.inspiration_count || 3}" min="0" max="10" />
												<span>random prompts</span>
												<button class="instaraw-rpg-btn-text instaraw-rpg-preview-random-btn">🎲 Preview</button>
											</div>
										</div>
									</div>
								`}
							</div>

							<!-- Model Settings -->
							<div class="instaraw-rpg-section">
								<div class="instaraw-rpg-section-header">
									<span class="instaraw-rpg-section-label">Model Settings</span>
								</div>
								<div class="instaraw-rpg-model-settings">
									<div class="instaraw-rpg-model-row">
										<label>Model</label>
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
									<div class="instaraw-rpg-checkbox-row">
										<label class="instaraw-rpg-checkbox-label">
											<input type="checkbox" class="instaraw-rpg-checkbox instaraw-rpg-is-sdxl-checkbox" ${node.properties.is_sdxl ? 'checked' : ''} />
											<span>SDXL Mode</span>
										</label>
										<label class="instaraw-rpg-checkbox-label" title="Bypass cache and generate fresh results">
											<input type="checkbox" class="instaraw-rpg-checkbox instaraw-rpg-force-regenerate-checkbox" />
											<span>🔄 Force Regenerate</span>
										</label>
									</div>
								</div>
							</div>

							<!-- Generation Count -->
							<div class="instaraw-rpg-section">
								<div class="instaraw-rpg-control-group">
									<label>Generation Count</label>
									<input type="number" class="instaraw-rpg-number-input instaraw-rpg-gen-count-input" value="${node.properties.generation_count || 5}" min="1" max="50" />
								</div>
							</div>

							<!-- Generate Button -->
							<button class="instaraw-rpg-btn-primary instaraw-rpg-generate-unified-btn">
								${detectedMode === 'img2img' ? '🖼️' : '🎨'} Generate Prompts
							</button>

							<!-- Preview Section -->
							<div class="instaraw-rpg-generate-preview" style="display: none;">
								<h4>Generated Prompts Preview</h4>
								<div class="instaraw-rpg-generate-preview-list"></div>
								<button class="instaraw-rpg-btn-primary instaraw-rpg-accept-generated-btn">✓ Add to Batch</button>
								<button class="instaraw-rpg-btn-secondary instaraw-rpg-cancel-generated-btn">✖ Cancel</button>
							</div>
						</div>
					`;
				};

				// === Character Tab (LEGACY - Will be removed) ===
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
								<div class="instaraw-rpg-control-group">
									<label title="Bypass cache and generate fresh results (costs API credits)">
										<input type="checkbox" class="instaraw-rpg-checkbox instaraw-rpg-char-force-regenerate-checkbox" />
										🔄 Force Regenerate
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
					const filters = JSON.parse(node.properties.library_filters || "{}");
					const sdxlMode = filters.sdxl_mode || false;

					// Get linked images/latents for thumbnails
					const detectedMode = node._linkedAILMode || "img2img";
					const linkedImages = node._linkedImages || [];
					const linkedLatents = node._linkedLatents || [];
					const hasAILLink = node._linkedAILNodeId !== null;

					const gridContent =
						promptQueue.length === 0
							? `<div class="instaraw-rpg-empty"><p>No prompts in batch</p><p class="instaraw-rpg-hint">Add prompts from Library or Creative mode</p></div>`
							: promptQueue
									.map(
										(entry, idx) => {
											const sourceType = entry.source_id ? 'from-library' : 'from-ai';
											const sourceBadgeText = entry.source_id ? '📚 Library' : '✨ AI Generated';

											// Get linked thumbnail for this index
											const linkedItem = detectedMode === "img2img" ? linkedImages[idx] : linkedLatents[idx];
											const hasThumbnail = linkedItem !== undefined;

											let thumbnailHtml = '';
											if (hasThumbnail) {
												if (detectedMode === "img2img") {
													// Show image thumbnail with target output dimensions
													const targetDims = getTargetDimensions();
													const targetAspectRatio = targetDims.width / targetDims.height;
													thumbnailHtml = `
														<label class="instaraw-rpg-thumbnail-label">IMG2IMG Input Image → Output: ${targetDims.aspect_label} (${targetDims.width}×${targetDims.height})</label>
														<div class="instaraw-rpg-batch-thumbnail">
															<span class="instaraw-rpg-batch-thumbnail-index">#${idx + 1}</span>
															<div class="instaraw-rpg-batch-aspect-preview" style="aspect-ratio: ${targetAspectRatio};">
																<img src="${linkedItem.url}" alt="Linked image ${idx + 1}" style="width: 100%; height: 100%; object-fit: cover;" />
																<div class="instaraw-rpg-crop-indicator">Center Crop</div>
															</div>
														</div>
													`;
												} else {
													// Show latent preview with aspect ratio box (like AIL)
													const aspectRatio = linkedItem.width / linkedItem.height;
													thumbnailHtml = `
														<label class="instaraw-rpg-thumbnail-label">TXT2IMG Empty Latent</label>
														<div class="instaraw-rpg-batch-thumbnail instaraw-rpg-batch-thumbnail-latent">
															<span class="instaraw-rpg-batch-thumbnail-index">#${idx + 1}</span>
															<div class="instaraw-rpg-batch-aspect-preview" style="aspect-ratio: ${aspectRatio};">
																<div class="instaraw-rpg-batch-aspect-content">
																	<div style="font-size: 24px;">📐</div>
																	<div style="font-size: 14px; font-weight: 600;">${linkedItem.aspect_label || '1:1'}</div>
																</div>
															</div>
														</div>
													`;
												}
											} else {
												// Show placeholder
												const promptQueueTemp = parsePromptBatch();
												const totalMissing = promptQueueTemp.length - (detectedMode === "img2img" ? linkedImages.length : linkedLatents.length);
												thumbnailHtml = `
													<div class="instaraw-rpg-batch-thumbnail instaraw-rpg-batch-thumbnail-missing">
														<span class="instaraw-rpg-batch-thumbnail-index">#${idx + 1}</span>
														<div class="instaraw-rpg-batch-thumbnail-placeholder">
															<div style="font-size: 24px; opacity: 0.3;">⚠️</div>
															<div style="font-size: 11px; color: #f59e0b; font-weight: 600; margin-top: 4px;">
																${hasAILLink ? 'Missing Link' : 'No AIL'}
															</div>
															${hasAILLink && detectedMode === "txt2img" ? `
																<div style="font-size: 9px; color: #9ca3af; margin-top: 4px;">
																	Click "Sync AIL"
																</div>
															` : hasAILLink && detectedMode === "img2img" && totalMissing > 0 ? `
																<div style="font-size: 9px; color: #9ca3af; margin-top: 4px;">
																	Upload ${totalMissing} more image${totalMissing !== 1 ? 's' : ''} to AIL
																</div>
															` : ''}
														</div>
													</div>
												`;
											}

											// Get repeat count comparison
											const promptRepeat = entry.repeat_count || 1;
											const ailRepeat = linkedItem ? (linkedItem.repeat_count || 1) : null;
											const repeatMismatch = ailRepeat !== null && promptRepeat !== ailRepeat;

											return `
							<div class="instaraw-rpg-batch-item" data-id="${entry.id}" data-idx="${idx}" draggable="${reorderModeEnabled}">
								<div class="instaraw-rpg-batch-item-header">
									<div style="display: flex; align-items: center; gap: 8px;">
										<span class="instaraw-rpg-batch-item-number">#${idx + 1}</span>
										<span class="instaraw-rpg-source-badge ${sourceType}">${sourceBadgeText}</span>
										${ailRepeat !== null ? `
											<span class="instaraw-rpg-repeat-status ${repeatMismatch ? 'instaraw-rpg-repeat-mismatch' : 'instaraw-rpg-repeat-match'}" title="${repeatMismatch ? 'Repeat counts do not match! Click Sync Repeats to fix.' : 'Repeat counts match'}">
												${repeatMismatch ? '⚠️ ' : '✓ '}Prompt: ×${promptRepeat} | AIL: ×${ailRepeat}
											</span>
										` : ''}
									</div>
									<div class="instaraw-rpg-batch-item-controls">
										<label>Repeat:</label>
										<input type="number" class="instaraw-rpg-repeat-input" data-id="${entry.id}" value="${entry.repeat_count || 1}" min="1" max="99" />
										<button class="instaraw-rpg-batch-delete-btn" data-id="${entry.id}" title="Remove prompt">×</button>
									</div>
								</div>

								<!-- Thumbnail Section -->
								${thumbnailHtml}

								<div class="instaraw-rpg-batch-item-content">
									${sdxlMode && entry.tags && entry.tags.length > 0 ? `
										<!-- SDXL Mode: Editable SDXL prompt (tags) -->
										<label>SDXL Prompt (Tags)</label>
										<textarea class="instaraw-rpg-prompt-textarea instaraw-rpg-positive-textarea" data-id="${entry.id}" rows="3" draggable="false">${entry.tags.join(", ")}</textarea>

										<label>Negative Prompt</label>
										<textarea class="instaraw-rpg-prompt-textarea instaraw-rpg-negative-textarea" data-id="${entry.id}" rows="2" draggable="false">${entry.negative_prompt || ""}</textarea>
									` : `
										<!-- Normal Mode: Show positive and negative textareas -->
										<label>Positive Prompt</label>
										<textarea class="instaraw-rpg-prompt-textarea instaraw-rpg-positive-textarea" data-id="${entry.id}" rows="3" draggable="false">${entry.positive_prompt || ""}</textarea>

										<label>Negative Prompt</label>
										<textarea class="instaraw-rpg-prompt-textarea instaraw-rpg-negative-textarea" data-id="${entry.id}" rows="2" draggable="false">${entry.negative_prompt || ""}</textarea>
									`}

									${entry.tags && entry.tags.length > 0 && !sdxlMode
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
						`;
										}
									)
									.join("");

					const libraryCount = promptQueue.filter(p => p.source_id).length;
					const aiCount = promptQueue.filter(p => !p.source_id).length;

					// Show sync button if: AIL linked + txt2img mode + has prompts
					const showSyncButton = hasAILLink && detectedMode === "txt2img" && promptQueue.length > 0;

					// Check for repeat count mismatches
					const hasRepeatMismatch = promptQueue.some((p, idx) => {
						const linkedItem = detectedMode === "img2img" ? linkedImages[idx] : linkedLatents[idx];
						return linkedItem && (p.repeat_count || 1) !== (linkedItem.repeat_count || 1);
					});

					return `
						<div class="instaraw-rpg-batch-header">
							<div>
								<h3>Generation Batch</h3>
								<p class="instaraw-rpg-batch-subtitle">
									${promptQueue.length > 0 ? `📚 ${libraryCount} from Library  |  ✨ ${aiCount} AI Generated` : 'Drag prompts to sync with Advanced Image Loader order'}
								</p>
							</div>
							<div class="instaraw-rpg-batch-actions">
								${showSyncButton ? `
									<button class="instaraw-rpg-btn-primary instaraw-rpg-sync-ail-btn" title="Create ${totalGenerations} empty latents in AIL #${node._linkedAILNodeId}">
										📤 Sync AIL (${totalGenerations})
									</button>
								` : ''}
								${hasAILLink && promptQueue.length > 0 ? `
									<button class="instaraw-rpg-btn-secondary instaraw-rpg-sync-repeats-btn ${hasRepeatMismatch ? 'instaraw-rpg-btn-warning' : ''}" title="Sync repeat counts from prompts to AIL">
										${hasRepeatMismatch ? '⚠️ ' : '🔄 '}Sync Repeats
									</button>
								` : ''}
								<button class="instaraw-rpg-btn-secondary instaraw-rpg-reorder-toggle-btn">
									${reorderModeEnabled ? '🔓 Reorder ON' : '🔒 Reorder OFF'}
								</button>
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
									<p>No source linked</p>
									<p class="instaraw-rpg-hint">Connect an Advanced Image Loader to see preview</p>
								</div>
							</div>
						`;
					}

					const itemCount = node._linkedImageCount || 0;
					const detectedMode = node._linkedAILMode || "img2img";
					const images = node._linkedImages || [];
					const latents = node._linkedLatents || [];
					const isImg2Img = detectedMode === "img2img";
					const items = isImg2Img ? images : latents;
					const displayItems = items.slice(0, 10);
					const itemLabel = isImg2Img ? "images" : "latents";

					const isMatch = totalGenerations === itemCount;
					const matchClass = isMatch ? "match" : "mismatch";
					const matchIcon = isMatch ? "✅" : "⚠️";

					return `
						<div class="instaraw-rpg-image-preview-section">
							<div class="instaraw-rpg-image-preview-header">
								<span>${isImg2Img ? '🖼️ IMG2IMG Input Images' : '📐 TXT2IMG Empty Latents'} (AIL Node #${node._linkedAILNodeId})</span>
								<span class="instaraw-rpg-validation-badge instaraw-rpg-validation-${matchClass}">
									${matchIcon} ${totalGenerations} prompts ↔ ${itemCount} ${itemLabel}
								</span>
							</div>
							<div class="instaraw-rpg-image-preview-grid">
								${displayItems
									.map((item, idx) => {
										if (isImg2Img) {
											// Display image thumbnail with target output aspect ratio and center crop
											const targetDims = getTargetDimensions();
											const targetAspectRatio = targetDims.width / targetDims.height;
											return `
												<div class="instaraw-rpg-preview-thumb">
													<span class="instaraw-rpg-preview-index">#${idx + 1}</span>
													<div class="instaraw-rpg-preview-aspect-box" style="aspect-ratio: ${targetAspectRatio}; position: relative;">
														<img src="${item.url}" alt="Preview ${idx + 1}" style="width: 100%; height: 100%; object-fit: cover;" />
														<div class="instaraw-rpg-crop-indicator">Center Crop</div>
													</div>
													<span class="instaraw-rpg-preview-aspect-label">→ ${targetDims.aspect_label} (${targetDims.width}×${targetDims.height})</span>
													${item.repeat_count && item.repeat_count > 1 ? `<span class="instaraw-rpg-preview-repeat">×${item.repeat_count}</span>` : ''}
												</div>
											`;
										} else {
											// Display latent placeholder with aspect preview box
											const aspectRatio = item.width / item.height;
											return `
												<div class="instaraw-rpg-preview-latent">
													<span class="instaraw-rpg-preview-index">#${idx + 1}</span>
													<div class="instaraw-rpg-preview-aspect-box" style="aspect-ratio: ${aspectRatio};">
														<div class="instaraw-rpg-preview-aspect-content">
															<div style="font-size: 20px;">📐</div>
															<div style="font-size: 11px; font-weight: 600;">${item.aspect_label || '1:1'}</div>
														</div>
													</div>
													${item.repeat_count && item.repeat_count > 1 ? `<span class="instaraw-rpg-preview-repeat">×${item.repeat_count}</span>` : ''}
												</div>
											`;
										}
									})
									.join("")}
								${items.length > 10 ? `<div class="instaraw-rpg-preview-more">+${items.length - 10} more</div>` : ""}
							</div>
						</div>
					`;
				};

				// === Filter Prompts ===
				const filterPrompts = (database, filters) => {
					if (!database) return [];

					let filtered = database;

					// Search query with match tracking
					if (filters.search_query && filters.search_query.trim() !== "") {
						const query = filters.search_query.toLowerCase();
						filtered = filtered.filter((p) => {
							const positive = (p.prompt?.positive || "").toLowerCase();
							const tags = (p.tags || []).join(" ").toLowerCase();
							const matchInPrompt = positive.includes(query);
							const matchInTags = tags.includes(query);
							// Store match type for highlighting
							p._matchType = matchInPrompt ? (matchInTags ? 'both' : 'prompt') : (matchInTags ? 'tags' : null);
							return matchInPrompt || matchInTags;
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

					// Bookmarked only
					if (filters.show_bookmarked) {
						const bookmarks = JSON.parse(node.properties.bookmarks || "[]");
						filtered = filtered.filter((p) => bookmarks.includes(p.id));
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

				const highlightSearchTerm = (text, searchQuery) => {
					if (!text || !searchQuery) return escapeHtml(text);
					const regex = new RegExp(`(${searchQuery.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')})`, 'gi');
					return escapeHtml(text).replace(regex, '<mark class="instaraw-rpg-highlight">$1</mark>');
				};

				const generateUniqueId = () => {
					return `${Date.now()}_${Math.random().toString(36).substring(2, 9)}`;
				};

				/**
				 * Gets the value from an input, handling both widgets on the current node
				 * and values from connected nodes (like Primitive string nodes).
				 * @param {string} inputName The name of the input.
				 * @param {any} defaultValue The value to return if no input is found.
				 * @returns {any} The found value or the default.
				 */
				const getFinalInputValue = (inputName, defaultValue) => {
					console.group(`[RPG] 🔍 Getting input value for: "${inputName}"`);
					console.log(`[RPG] Node ID: ${node.id}, Node Type: ${node.type}`);

					// Log the entire node structure for debugging
					console.log(`[RPG] Node has ${node.inputs?.length || 0} inputs total`);
					console.log(`[RPG] Node has ${node.widgets?.length || 0} widgets total`);

					if (node.inputs && node.inputs.length > 0) {
						console.log(`[RPG] All inputs on this node:`, node.inputs.map(i => ({
							name: i.name,
							type: i.type,
							link: i.link,
							has_link: i.link != null
						})));
					}

					if (node.widgets && node.widgets.length > 0) {
						console.log(`[RPG] All widgets on this node:`, node.widgets.map(w => ({
							name: w.name,
							type: w.type,
							value: w.value ? "[HAS VALUE]" : "[EMPTY]"
						})));
					}

					// Check if node has inputs array
					if (!node.inputs || node.inputs.length === 0) {
						console.warn(`[RPG] ⚠️ No inputs found on node!`);
						// Fallback to widget
						const widget = node.widgets?.find(w => w.name === inputName);
						if (widget) {
							console.log(`[RPG] ✅ Found value in local widget:`, widget.value);
							console.groupEnd();
							return widget.value;
						}
						console.log(`[RPG] ❌ No widget found, returning default:`, defaultValue);
						console.groupEnd();
						return defaultValue;
					}

					// Find the input by name
					const input = node.inputs.find(i => i.name === inputName);

					if (!input) {
						console.error(`[RPG] ❌ Input "${inputName}" not found! Available inputs:`, node.inputs.map(i => i.name));
						console.groupEnd();
						return defaultValue;
					}

					console.log(`[RPG] ✅ Input found:`, {
						name: input.name,
						type: input.type,
						link: input.link,
						has_link: input.link != null
					});

					// Check if input is connected (has a link)
					if (input.link == null) {
						console.warn(`[RPG] ⚠️ Input "${inputName}" is not connected (link is null)`);

						// Check if there's a widget as fallback
						const widget = node.widgets?.find(w => w.name === inputName);
						if (widget) {
							console.log(`[RPG] ✅ Found fallback value in local widget:`, widget.value);
							console.groupEnd();
							return widget.value;
						}

						console.log(`[RPG] ❌ No fallback widget, returning default:`, defaultValue);
						console.groupEnd();
						return defaultValue;
					}

					// Get the link from the graph
					const link = app.graph.links[input.link];

					if (!link) {
						console.error(`[RPG] ❌ Link ${input.link} not found in app.graph.links!`);
						console.log(`[RPG] Available links in graph:`, Object.keys(app.graph.links || {}));
						console.groupEnd();
						return defaultValue;
					}

					console.log(`[RPG] ✅ Link found:`, {
						id: link.id,
						origin_id: link.origin_id,
						origin_slot: link.origin_slot,
						target_id: link.target_id,
						target_slot: link.target_slot,
						type: link.type
					});

					// Get the origin node
					const originNode = app.graph.getNodeById(link.origin_id);

					if (!originNode) {
						console.error(`[RPG] ❌ Origin node ${link.origin_id} not found!`);
						console.log(`[RPG] Available nodes in graph:`, app.graph._nodes.map(n => ({id: n.id, type: n.type})));
						console.groupEnd();
						return defaultValue;
					}

					console.log(`[RPG] ✅ Origin node found:`, {
						id: originNode.id,
						type: originNode.type,
						title: originNode.title,
						has_widgets: originNode.widgets && originNode.widgets.length > 0,
						widget_count: originNode.widgets?.length || 0,
						has_properties: originNode.properties !== undefined
					});

					// Log all widgets on the origin node
					if (originNode.widgets && originNode.widgets.length > 0) {
						console.log(`[RPG] Origin node widgets:`, originNode.widgets.map((w, idx) => ({
							index: idx,
							name: w.name,
							type: w.type,
							value: typeof w.value === 'string' && w.value.length > 50
								? `[STRING LENGTH: ${w.value.length}]`
								: w.value
						})));

						const value = originNode.widgets[0].value;
						console.log(`[RPG] ✅ Returning value from origin node widget[0]:`,
							typeof value === 'string' && value.length > 50
								? `[STRING LENGTH: ${value.length}, Preview: ${value.substring(0, 20)}...]`
								: value
						);
						console.groupEnd();
						return value;
					}

					// Fallback: check properties (some node types store value here)
					if (originNode.properties) {
						console.log(`[RPG] Origin node properties:`, originNode.properties);

						if (originNode.properties.value !== undefined) {
							const value = originNode.properties.value;
							console.log(`[RPG] ✅ Found value in origin node properties:`, value);
							console.groupEnd();
							return value;
						}
					}

					console.warn(`[RPG] ⚠️ Origin node has no widgets or properties with value`);
					console.log(`[RPG] Origin node full structure:`, originNode);

					// Last resort: check if there's a widget on this node
					const widget = node.widgets?.find(w => w.name === inputName);
					if (widget) {
						console.log(`[RPG] ✅ Found fallback value in local widget:`, widget.value);
						console.groupEnd();
						return widget.value;
					}

					console.log(`[RPG] ❌ No value found anywhere, returning default:`, defaultValue);
					console.groupEnd();
					return defaultValue;
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
						show_bookmarked: false,
						sdxl_mode: false,
					});
					currentPage = 0;
					renderUI();
				};

				// Debug helper - expose for testing in browser console
				node.debugGetInput = (inputName) => {
					return getFinalInputValue(inputName, "NOT_FOUND");
				};

				// Comprehensive graph dump for debugging
				node.debugDumpGraph = () => {
					console.group("[RPG] 📊 Complete Graph Dump");
					console.log("=== Current Node ===");
					console.log("Node ID:", node.id);
					console.log("Node Type:", node.type);
					console.log("Node Title:", node.title);
					console.log("Node Inputs:", node.inputs);
					console.log("Node Widgets:", node.widgets);
					console.log("Node Properties:", node.properties);

					console.log("\n=== All Nodes in Graph ===");
					console.table(app.graph._nodes.map(n => ({
						id: n.id,
						type: n.type,
						title: n.title,
						inputs: n.inputs?.length || 0,
						outputs: n.outputs?.length || 0,
						widgets: n.widgets?.length || 0
					})));

					console.log("\n=== All Links in Graph ===");
					const links = app.graph.links || {};
					console.table(Object.values(links).map(l => ({
						id: l.id,
						origin_id: l.origin_id,
						origin_slot: l.origin_slot,
						target_id: l.target_id,
						target_slot: l.target_slot,
						type: l.type
					})));

					console.log("\n=== Links Connected to This Node ===");
					const connectedLinks = Object.values(links).filter(l =>
						l.target_id === node.id || l.origin_id === node.id
					);
					console.log("Connected links:", connectedLinks);

					console.log("\n=== Nodes Connected to This Node ===");
					const connectedNodeIds = new Set();
					connectedLinks.forEach(l => {
						connectedNodeIds.add(l.origin_id);
						connectedNodeIds.add(l.target_id);
					});
					const connectedNodes = app.graph._nodes.filter(n => connectedNodeIds.has(n.id));
					connectedNodes.forEach(n => {
						console.group(`Node ${n.id}: ${n.type} (${n.title})`);
						console.log("Widgets:", n.widgets);
						console.log("Properties:", n.properties);
						console.groupEnd();
					});

					console.groupEnd();
					return {
						node: {
							id: node.id,
							type: node.type,
							inputs: node.inputs,
							widgets: node.widgets
						},
						connectedNodes,
						connectedLinks
					};
				};

				// === Generate Creative Prompts ===
				const generateCreativePrompts = async () => {
					console.group("[RPG] 🎨 Generate Creative Prompts - START");
					console.log("[RPG] Timestamp:", new Date().toISOString());

					const genCountInput = container.querySelector(".instaraw-rpg-gen-count-input");
					const inspirationCountInput = container.querySelector(".instaraw-rpg-inspiration-count-input");
					const isSDXLCheckbox = container.querySelector(".instaraw-rpg-is-sdxl-checkbox");
					const forceRegenerateCheckbox = container.querySelector(".instaraw-rpg-force-regenerate-checkbox");
					const generateBtn = container.querySelector(".instaraw-rpg-generate-creative-btn");

					const generationCount = parseInt(genCountInput?.value || "5");
					const inspirationCount = parseInt(inspirationCountInput?.value || "3");
					const isSDXL = isSDXLCheckbox?.checked || false;
					const forceRegenerate = forceRegenerateCheckbox?.checked || false;

					const promptQueue = parsePromptBatch();
					const sourcePrompts = promptQueue.filter((p) => p.source_id).slice(0, inspirationCount);

					const modelWidget = node.widgets?.find((w) => w.name === "creative_model");
					const model = modelWidget?.value || node.properties.creative_model || "gemini-2.5-pro";
					const systemPrompt = node.properties.creative_system_prompt || DEFAULT_RPG_SYSTEM_PROMPT;
					const temperatureValue = parseFloat(node.properties.creative_temperature ?? 0.9) || 0.9;
					const topPValue = parseFloat(node.properties.creative_top_p ?? 0.9) || 0.9;

					console.log("[RPG] Configuration:", {
						generationCount,
						inspirationCount,
						isSDXL,
						forceRegenerate,
						model,
						temperature: temperatureValue,
						topP: topPValue,
						sourcePromptsCount: sourcePrompts.length
					});

					console.log("[RPG] About to retrieve API keys from connected nodes...");

					// Get API keys by traversing the graph if inputs are connected
					const geminiApiKey = (getFinalInputValue("gemini_api_key", "") || "").trim() || window.INSTARAW_GEMINI_KEY || "";
					const grokApiKey = (getFinalInputValue("grok_api_key", "") || "").trim() || window.INSTARAW_GROK_KEY || "";

					console.log(`[RPG] ✅ Resolved Gemini API Key: ${geminiApiKey ? `[KEY PRESENT - Length: ${geminiApiKey.length}]` : "[EMPTY]"}`);
					console.log(`[RPG] ✅ Resolved Grok API Key: ${grokApiKey ? `[KEY PRESENT - Length: ${grokApiKey.length}]` : "[EMPTY]"}`);

					console.log("[RPG] Window fallback keys:", {
						INSTARAW_GEMINI_KEY: window.INSTARAW_GEMINI_KEY ? "[PRESENT]" : "[EMPTY]",
						INSTARAW_GROK_KEY: window.INSTARAW_GROK_KEY ? "[PRESENT]" : "[EMPTY]"
					});

					// Validate API keys before proceeding
					if (!geminiApiKey && !grokApiKey) {
						console.error("[RPG] ❌ NO API KEYS FOUND!");
						console.log("[RPG] To fix this, connect a Primitive String node to either:");
						console.log("[RPG]   - gemini_api_key input");
						console.log("[RPG]   - grok_api_key input");
						console.log("[RPG] Or set window.INSTARAW_GEMINI_KEY or window.INSTARAW_GROK_KEY");
						console.groupEnd();
						alert("No API keys found! Please connect a Primitive String node with your Gemini or Grok API key to the 'gemini_api_key' or 'grok_api_key' input.");
						return;
					}

					// Disable button and show loading
					if (generateBtn) {
						generateBtn.disabled = true;
						generateBtn.textContent = "⏳ Generating...";
					}

					console.log("[RPG] Making API request to /instaraw/generate_creative_prompts");
					console.log("[RPG] Request payload:", {
						source_prompts_count: sourcePrompts.length,
						generation_count: generationCount,
						inspiration_count: inspirationCount,
						is_sdxl: isSDXL,
						force_regenerate: forceRegenerate,
						model: model,
						has_gemini_key: !!geminiApiKey,
						has_grok_key: !!grokApiKey,
						temperature: temperatureValue,
						top_p: topPValue,
					});

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
								force_regenerate: forceRegenerate,
							}),
						});

						console.log("[RPG] API response status:", response.status, response.statusText);

						const result = await parseJSONResponse(response);
						console.log("[RPG] API response parsed:", result);

						if (!response.ok) {
							throw new Error(result?.error || `Creative API error ${response.status}`);
						}

						if (result.success && result.prompts) {
							console.log(`[RPG] ✅ Success! Generated ${result.prompts.length} prompts`);
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
						console.error("[RPG] ❌ Error during creative prompt generation:", error);
						console.error("[RPG] Error stack:", error.stack);
						alert(`Creative generation error: ${error.message || error}`);
					} finally {
						if (generateBtn) {
							generateBtn.disabled = false;
							generateBtn.textContent = "✨ Generate & Add to Batch";
						}
						console.log("[RPG] Generate Creative Prompts - END");
						console.groupEnd();
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
					console.group("[RPG] 👤 Generate Character Prompts - START");
					console.log("[RPG] Timestamp:", new Date().toISOString());

					const charRefInput = container.querySelector(".instaraw-rpg-character-ref-input");
					const genCountInput = container.querySelector(".instaraw-rpg-char-gen-count-input");
					const isSDXLCheckbox = container.querySelector(".instaraw-rpg-char-is-sdxl-checkbox");
					const forceRegenerateCheckbox = container.querySelector(".instaraw-rpg-char-force-regenerate-checkbox");
					const generateBtn = container.querySelector(".instaraw-rpg-generate-character-btn");

					const characterReference = charRefInput?.value || "";
					const generationCount = parseInt(genCountInput?.value || "5");
					const isSDXL = isSDXLCheckbox?.checked || false;
					const forceRegenerate = forceRegenerateCheckbox?.checked || false;

					const modelWidget = node.widgets?.find((w) => w.name === "creative_model");
					const model = modelWidget?.value || node.properties.creative_model || "gemini-2.5-pro";
					const systemPrompt = node.properties.creative_system_prompt || DEFAULT_RPG_SYSTEM_PROMPT;
					const temperatureValue = parseFloat(node.properties.creative_temperature ?? 0.9) || 0.9;
					const topPValue = parseFloat(node.properties.creative_top_p ?? 0.9) || 0.9;

					console.log("[RPG] Configuration:", {
						generationCount,
						isSDXL,
						forceRegenerate,
						model,
						temperature: temperatureValue,
						topP: topPValue,
						characterReferenceLength: characterReference.length
					});

					console.log("[RPG] About to retrieve API keys from connected nodes...");

					// Get API keys by traversing the graph if inputs are connected
					const geminiApiKey = (getFinalInputValue("gemini_api_key", "") || "").trim() || window.INSTARAW_GEMINI_KEY || "";
					const grokApiKey = (getFinalInputValue("grok_api_key", "") || "").trim() || window.INSTARAW_GROK_KEY || "";

					console.log(`[RPG] ✅ Resolved Gemini API Key: ${geminiApiKey ? `[KEY PRESENT - Length: ${geminiApiKey.length}]` : "[EMPTY]"}`);
					console.log(`[RPG] ✅ Resolved Grok API Key: ${grokApiKey ? `[KEY PRESENT - Length: ${grokApiKey.length}]` : "[EMPTY]"}`);

					console.log("[RPG] Window fallback keys:", {
						INSTARAW_GEMINI_KEY: window.INSTARAW_GEMINI_KEY ? "[PRESENT]" : "[EMPTY]",
						INSTARAW_GROK_KEY: window.INSTARAW_GROK_KEY ? "[PRESENT]" : "[EMPTY]"
					});

					// Validate API keys before proceeding
					if (!geminiApiKey && !grokApiKey) {
						console.error("[RPG] ❌ NO API KEYS FOUND!");
						console.log("[RPG] To fix this, connect a Primitive String node to either:");
						console.log("[RPG]   - gemini_api_key input");
						console.log("[RPG]   - grok_api_key input");
						console.log("[RPG] Or set window.INSTARAW_GEMINI_KEY or window.INSTARAW_GROK_KEY");
						console.groupEnd();
						alert("No API keys found! Please connect a Primitive String node with your Gemini or Grok API key to the 'gemini_api_key' or 'grok_api_key' input.");
						return;
					}

					if (!characterReference.trim()) {
						console.error("[RPG] ❌ No character reference provided");
						console.groupEnd();
						alert("Please enter a character reference");
						return;
					}

					// Disable button and show loading
					if (generateBtn) {
						generateBtn.disabled = true;
						generateBtn.textContent = "⏳ Generating...";
					}

					console.log("[RPG] Making API request to /instaraw/generate_creative_prompts (character mode)");
					console.log("[RPG] Request payload:", {
						generation_count: generationCount,
						is_sdxl: isSDXL,
						force_regenerate: forceRegenerate,
						model: model,
						has_gemini_key: !!geminiApiKey,
						has_grok_key: !!grokApiKey,
						character_reference_length: characterReference.length,
						temperature: temperatureValue,
						top_p: topPValue,
					});

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
								force_regenerate: forceRegenerate,
							}),
						});

						console.log("[RPG] API response status:", response.status, response.statusText);

						const result = await parseJSONResponse(response);
						console.log("[RPG] API response parsed:", result);

						if (!response.ok) {
							throw new Error(result?.error || `Creative API error ${response.status}`);
						}

						if (result.success && result.prompts) {
							console.log(`[RPG] ✅ Success! Generated ${result.prompts.length} character prompts`);
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
						console.error("[RPG] ❌ Error during character prompt generation:", error);
						console.error("[RPG] Error stack:", error.stack);
						alert(`Character generation error: ${error.message || error}`);
					} finally {
						if (generateBtn) {
							generateBtn.disabled = false;
							generateBtn.textContent = "👤 Generate Character Prompts";
						}
						console.log("[RPG] Generate Character Prompts - END");
						console.groupEnd();
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

					// Show bookmarked checkbox
					const showBookmarkedCheckbox = container.querySelector(".instaraw-rpg-show-bookmarked-checkbox");
					if (showBookmarkedCheckbox) {
						showBookmarkedCheckbox.onchange = (e) => {
							updateFilter("show_bookmarked", e.target.checked);
						};
					}

					// SDXL mode checkbox
					const sdxlModeCheckbox = container.querySelector(".instaraw-rpg-sdxl-mode-checkbox");
					if (sdxlModeCheckbox) {
						sdxlModeCheckbox.onchange = (e) => {
							updateFilter("sdxl_mode", e.target.checked);
						};
					}

					// Clear filters
					const clearFiltersBtn = container.querySelector(".instaraw-rpg-clear-filters-btn");
					if (clearFiltersBtn) {
						clearFiltersBtn.onclick = clearFilters;
					}

					// Add to batch buttons (always adds)
					container.querySelectorAll(".instaraw-rpg-add-to-batch-btn").forEach((btn) => {
						btn.onclick = () => {
							const promptId = btn.dataset.id;
							const promptData = promptsDatabase.find((p) => p.id === promptId);
							if (promptData) addPromptToBatch(promptData);
						};
					});

					// Undo batch buttons (removes last instance)
					container.querySelectorAll(".instaraw-rpg-undo-batch-btn").forEach((btn) => {
						btn.onclick = () => {
							const promptId = btn.dataset.id;
							const promptQueue = parsePromptBatch();
							// Find all instances with this source_id
							const instances = promptQueue.filter(p => p.source_id === promptId);
							if (instances.length > 0) {
								// Remove the last one added
								const lastInstance = instances[instances.length - 1];
								deletePromptFromBatch(lastInstance.id);
							}
						};
					});

					// Bookmark buttons
					container.querySelectorAll(".instaraw-rpg-bookmark-btn").forEach((btn) => {
						btn.onclick = (e) => {
							e.stopPropagation();
							toggleBookmark(btn.dataset.id);
						};
					});

					// Toggle tags buttons (expand/collapse) - use event delegation
					container.addEventListener('click', (e) => {
						if (e.target.classList.contains('instaraw-rpg-toggle-tags-btn')) {
							e.stopPropagation();
							const promptId = e.target.dataset.id;
							const card = e.target.closest('.instaraw-rpg-library-card');
							const tagsContainer = card.querySelector('.instaraw-rpg-library-card-tags');
							const prompt = promptsDatabase.find(p => p.id === promptId);

							if (prompt && tagsContainer) {
								const isExpanded = tagsContainer.getAttribute('data-expanded') === 'true';
								const filters = JSON.parse(node.properties.library_filters || "{}");
								const searchQuery = filters.search_query?.trim() || "";

								if (isExpanded) {
									// Collapse - show only first 5
									tagsContainer.setAttribute('data-expanded', 'false');
									tagsContainer.innerHTML = prompt.tags.slice(0, 5)
										.map(tag => `<span class="instaraw-rpg-tag">${highlightSearchTerm(tag, searchQuery)}</span>`)
										.join("") + ` <button class="instaraw-rpg-toggle-tags-btn" data-id="${promptId}">+${prompt.tags.length - 5}</button>`;
								} else {
									// Expand - show all tags
									tagsContainer.setAttribute('data-expanded', 'true');
									tagsContainer.innerHTML = prompt.tags
										.map(tag => `<span class="instaraw-rpg-tag">${highlightSearchTerm(tag, searchQuery)}</span>`)
										.join("") + ` <button class="instaraw-rpg-toggle-tags-btn" data-id="${promptId}">Show less</button>`;
								}
							}
						}
					});

					// Pagination
					container.querySelectorAll(".instaraw-rpg-prev-page-btn").forEach((prevPageBtn) => {
						prevPageBtn.onclick = () => {
							if (currentPage > 0) {
								currentPage--;
								renderUI();
							}
						};
					});

					container.querySelectorAll(".instaraw-rpg-next-page-btn").forEach((nextPageBtn) => {
						nextPageBtn.onclick = () => {
							currentPage++;
							renderUI();
						};
					});

					// Batch item controls
					container.querySelectorAll(".instaraw-rpg-positive-textarea").forEach((textarea) => {
						autoResizeTextarea(textarea);
						textarea.oninput = (e) => autoResizeTextarea(textarea);
						textarea.onchange = (e) => {
							updatePromptInBatch(textarea.dataset.id, "positive_prompt", e.target.value);
						};
					});

					container.querySelectorAll(".instaraw-rpg-negative-textarea").forEach((textarea) => {
						autoResizeTextarea(textarea);
						textarea.oninput = (e) => autoResizeTextarea(textarea);
						textarea.onchange = (e) => {
							updatePromptInBatch(textarea.dataset.id, "negative_prompt", e.target.value);
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

					// Reorder toggle button
					const reorderToggleBtn = container.querySelector(".instaraw-rpg-reorder-toggle-btn");
					if (reorderToggleBtn) {
						reorderToggleBtn.onclick = () => {
							reorderModeEnabled = !reorderModeEnabled;
							renderUI();
						};
					}

					// Sync Repeats button
					const syncRepeatsBtn = container.querySelector(".instaraw-rpg-sync-repeats-btn");
					if (syncRepeatsBtn) {
						syncRepeatsBtn.onclick = () => {
							const promptQueue = parsePromptBatch();
							const detectedMode = node._linkedAILMode || "img2img";

							if (confirm(`This will update repeat counts in AIL Node #${node._linkedAILNodeId} to match your ${promptQueue.length} prompts.\n\nContinue?`)) {
								// Send sync event to AIL
								window.dispatchEvent(new CustomEvent("INSTARAW_SYNC_AIL_REPEATS", {
									detail: {
										targetNodeId: node._linkedAILNodeId,
										mode: detectedMode,
										repeats: promptQueue.map(p => p.repeat_count || 1)
									}
								}));

								console.log(`[INSTARAW RPG] Sent repeat sync request to AIL #${node._linkedAILNodeId}`);
							}
						};
					}

					// Sync AIL button
					const syncAilBtn = container.querySelector(".instaraw-rpg-sync-ail-btn");
					if (syncAilBtn) {
						syncAilBtn.onclick = () => {
							if (!node._linkedAILNodeId) {
								alert("No Advanced Image Loader detected. Connect AIL to RPG first.");
								return;
							}

							const promptQueue = parsePromptBatch();
							const totalGenerations = promptQueue.reduce((sum, p) => sum + (p.repeat_count || 1), 0);

							// Build latent specs with repeat counts
							const latentSpecs = promptQueue.map(p => ({
								repeat_count: p.repeat_count || 1
							}));

							if (confirm(`This will create ${promptQueue.length} latent${promptQueue.length !== 1 ? 's' : ''} (${totalGenerations} total generations) in AIL Node #${node._linkedAILNodeId} to match your prompts.\n\nContinue?`)) {
								// Get current dimensions from AIL (or use defaults)
								const dimensions = {
									width: 1024,
									height: 1024,
									aspect_label: "1:1"
								};

								// Dispatch event to AIL with latent specs
								window.dispatchEvent(new CustomEvent("INSTARAW_SYNC_AIL_LATENTS", {
									detail: {
										targetNodeId: node._linkedAILNodeId,
										latentSpecs: latentSpecs,
										dimensions: dimensions
									}
								}));

								console.log(`[INSTARAW RPG] Sent sync request to AIL #${node._linkedAILNodeId} for ${promptQueue.length} latents (${totalGenerations} generations)`);
							}
						};
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
					// Drag-and-drop reordering (only when enabled)
					if (reorderModeEnabled) {
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
					}
				};

				// === AIL Update Listener ===
				window.addEventListener("INSTARAW_AIL_UPDATED", (event) => {
					const { nodeId, images, latents, total, mode, enable_img2img } = event.detail;
					node._linkedAILNodeId = nodeId;
					node._linkedAILMode = mode || (enable_img2img ? "img2img" : "txt2img");

					// Store images OR latents depending on mode
					if (mode === "img2img" || enable_img2img) {
						node._linkedImages = images || [];
						node._linkedLatents = [];
					} else {
						node._linkedImages = [];
						node._linkedLatents = latents || [];
					}

					node._linkedImageCount = total;

					console.log(`[INSTARAW RPG] AIL update - Node: ${nodeId}, Mode: ${node._linkedAILMode}, Count: ${total}`);

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