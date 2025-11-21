// ---
// Filename: ../ComfyUI_INSTARAW/js/reality_prompt_generator.js
// Reality Prompt Generator (RPG) - Full JavaScript UI Implementation
// Following AdvancedImageLoader patterns exactly
// ---

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const DEFAULT_RPG_SYSTEM_PROMPT = `You are an expert AI prompt engineer specializing in photorealistic image generation.

Generate complete, detailed prompts in the following EXACT format (each field on a new line with prefix):

POSITIVE: [Highly detailed positive prompt with camera settings, lighting, composition, subject details, environment, mood, technical quality descriptors]
NEGATIVE: [Negative prompt listing unwanted elements: low quality, blurry, distorted, artifacts, bad anatomy, etc.]
CONTENT_TYPE: [person|landscape|architecture|object|animal|abstract|other]
SAFETY_LEVEL: [sfw|suggestive|nsfw]
SHOT_TYPE: [portrait|full_body|close_up|wide_angle|other]
TAGS: [SDXL-style comma-separated tags, MAX 50 words, most important tags only]

CRITICAL RULES:
- POSITIVE: 150-300 words, photorealistic, include camera/lens details, lighting, composition
- NEGATIVE: Common quality issues and artifacts to avoid
- TAGS: Maximum 50 words (≈250 chars), prioritize most visually important descriptors
- Use EXACT prefixes above (POSITIVE:, NEGATIVE:, etc.)
- Keep each field on ONE line (no line breaks within fields)
- Be specific, detailed, and technical for best image quality`;
const REMOTE_PROMPTS_DB_URL = "https://instara.s3.us-east-1.amazonaws.com/prompts.db.json";
const CREATIVE_MODEL_OPTIONS = [
	{ value: "gemini-2.5-pro", label: "Gemini 2.5 Pro" },
	{ value: "gemini-3-pro-preview", label: "Gemini 3.0 Pro Preview" },
	{ value: "gemini-flash-latest", label: "Gemini Flash Latest" },
	{ value: "grok-4-fast-reasoning", label: "Grok 4 Fast (Reasoning)" },
	{ value: "grok-4-fast-non-reasoning", label: "Grok 4 Fast (Non-Reasoning)" },
	{ value: "grok-4-0709", label: "Grok 4 0709" },
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
				if (this.properties.generation_style === undefined) {
					this.properties.generation_style = "reality"; // Default to reality mode
				}

				const node = this;
				let cachedHeight = 400; // Initial height (AIL pattern)
				let isUpdatingHeight = false; // Prevent concurrent updates (AIL pattern)

				// Database state
				let promptsDatabase = null;
				let isDatabaseLoading = false;
				let databaseLoadProgress = 0;

				// Generation lock state (crash prevention)
				let isGenerating = false;
				let currentAbortController = null;

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

				// Random mode state
				let showingRandomPrompts = false;
				let randomPrompts = [];
				let randomCount = 6;

				// User prompt edit mode state
				const editingPrompts = new Set(); // Track which prompt IDs are in edit mode
				const editingValues = {}; // Store temporary edit values

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

				// === Aspect Ratio Node Reading (Same as AIL) ===

				/**
				 * Reads output from WAN/SDXL aspect ratio nodes.
				 * Must match Python ASPECT_RATIOS dicts exactly.
				 */
				const getAspectRatioOutput = (aspectRatioNode, slotIndex) => {
					const selection = aspectRatioNode.widgets?.[0]?.value;
					if (!selection) {
						console.warn(`[RPG] Aspect ratio node has no selection`);
						return null;
					}

					const WAN_RATIOS = {
						"3:4 (Portrait)": { width: 720, height: 960, label: "3:4" },
						"9:16 (Tall Portrait)": { width: 540, height: 960, label: "9:16" },
						"1:1 (Square)": { width: 960, height: 960, label: "1:1" },
						"16:9 (Landscape)": { width: 960, height: 540, label: "16:9" }
					};

					const SDXL_RATIOS = {
						"3:4 (Portrait)": { width: 896, height: 1152, label: "3:4" },
						"9:16 (Tall Portrait)": { width: 768, height: 1344, label: "9:16" },
						"1:1 (Square)": { width: 1024, height: 1024, label: "1:1" },
						"16:9 (Landscape)": { width: 1344, height: 768, label: "16:9" }
					};

					const ratios = aspectRatioNode.type === "INSTARAW_WANAspectRatio" ? WAN_RATIOS : SDXL_RATIOS;
					const config = ratios[selection];
					if (!config) {
						console.warn(`[RPG] Unknown aspect ratio selection: ${selection}`);
						return null;
					}

					// console.log(`[RPG] Aspect ratio node output:`, {
					// 	selection,
					// 	slotIndex,
					// 	type: aspectRatioNode.type,
					// 	value: slotIndex === 0 ? config.width : slotIndex === 1 ? config.height : config.label
					// });

					// Return based on output slot (0=width, 1=height, 2=aspect_label)
					if (slotIndex === 0) return config.width;
					if (slotIndex === 1) return config.height;
					if (slotIndex === 2) return config.label;
					return null;
				};

				/**
				 * Retrieves the final value of an input by traversing connected nodes.
				 * Enhanced to properly handle multi-output nodes like aspect ratio nodes.
				 */
				const getFinalInputValue = (inputName, defaultValue) => {
					try {
						if (!node.inputs || node.inputs.length === 0) {
							const widget = node.widgets?.find(w => w.name === inputName);
							return widget ? widget.value : defaultValue;
						}

						const input = node.inputs.find(i => i.name === inputName);
						if (!input || input.link == null) {
							const widget = node.widgets?.find(w => w.name === inputName);
							return widget ? widget.value : defaultValue;
						}

						// Access app.graph safely
						if (typeof app === 'undefined' || !app.graph) {
							console.warn('[RPG] app.graph not available, using widget value');
							const widget = node.widgets?.find(w => w.name === inputName);
							return widget ? widget.value : defaultValue;
						}

						const link = app.graph.links[input.link];
						if (!link) return defaultValue;

						const originNode = app.graph.getNodeById(link.origin_id);
						if (!originNode) return defaultValue;

						// SPECIAL HANDLING: For aspect ratio nodes, compute the output locally
						if (originNode.type === "INSTARAW_WANAspectRatio" || originNode.type === "INSTARAW_SDXLAspectRatio") {
							const output = getAspectRatioOutput(originNode, link.origin_slot);
							if (output !== null) return output;
						}

						// For other nodes, read from widgets
						if (originNode.widgets && originNode.widgets.length > 0) {
							return originNode.widgets[0].value;
						}

						if (originNode.properties && originNode.properties.value !== undefined) {
							return originNode.properties.value;
						}

						return defaultValue;
					} catch (error) {
						console.error('[RPG] Error in getFinalInputValue:', error);
						return defaultValue;
					}
				};

				/**
				 * Get target output dimensions from aspect ratio selector.
				 * Now properly reads from connected aspect ratio nodes!
				 */
				const getTargetDimensions = () => {
					try {
						const width = getFinalInputValue("output_width", 1024);
						const height = getFinalInputValue("output_height", 1024);
						const aspect_label = getFinalInputValue("aspect_label", "1:1");

						const dims = {
							width: parseInt(width) || 1024,
							height: parseInt(height) || 1024,
							aspect_label: aspect_label || "1:1"
						};

						// console.log("[RPG] Target dimensions from aspect ratio node:", dims);
						return dims;
					} catch (error) {
						console.error('[RPG] Error in getTargetDimensions:', error);
						return { width: 1024, height: 1024, aspect_label: "1:1" };
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
							// Load user prompts and merge
							await loadUserPrompts();
							await loadGeneratedPrompts();
							mergeUserPromptsWithLibrary();
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

						// Load user prompts and merge
						await loadUserPrompts();
						mergeUserPromptsWithLibrary();

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

				// === User Prompts Management ===
				let userPrompts = []; // In-memory cache of user-created prompts

				const loadUserPrompts = async () => {
					try {
						const cached = await getFromIndexedDB("user_prompts");
						userPrompts = cached || [];
						console.log(`[RPG] Loaded ${userPrompts.length} user prompts from IndexedDB`);
						return userPrompts;
					} catch (error) {
						console.error("[RPG] Error loading user prompts:", error);
						userPrompts = [];
						return [];
					}
				};

				const saveUserPrompts = async (prompts) => {
					try {
						await saveToIndexedDB("user_prompts", prompts);
						userPrompts = prompts;
						console.log(`[RPG] Saved ${prompts.length} user prompts to IndexedDB`);
						return true;
					} catch (error) {
						console.error("[RPG] Error saving user prompts:", error);
						return false;
					}
				};

				const addUserPrompt = async (promptData) => {
					const newPrompt = {
						id: `user_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
						tags: promptData.tags || [],
						prompt: {
							positive: promptData.positive || "",
							negative: promptData.negative || ""
						},
						classification: {
							content_type: promptData.content_type || "person",
							safety_level: promptData.safety_level || "sfw",
							shot_type: promptData.shot_type || "portrait"
						},
						is_user_created: true, // Flag to identify user prompts
						created_at: Date.now()
					};

					userPrompts.unshift(newPrompt); // Add to beginning
					await saveUserPrompts(userPrompts);
					mergeUserPromptsWithLibrary();
					renderUI();
					return newPrompt;
				};

				const updateUserPrompt = async (id, updates) => {
					const index = userPrompts.findIndex(p => p.id === id);
					if (index !== -1) {
						userPrompts[index] = {
							...userPrompts[index],
							...updates,
							is_user_created: true, // Preserve flag
							updated_at: Date.now()
						};
						await saveUserPrompts(userPrompts);
						mergeUserPromptsWithLibrary();
						renderUI();
						return true;
					}
					return false;
				};

				const deleteUserPrompt = async (id) => {
					const filtered = userPrompts.filter(p => p.id !== id);
					if (filtered.length !== userPrompts.length) {
						await saveUserPrompts(filtered);
						mergeUserPromptsWithLibrary();
						renderUI();
						return true;
					}
					return false;
				};

			// === Generated Prompts Storage ===
			let generatedPrompts = [];
			const loadGeneratedPrompts = async () => {
				try {
					const cached = await getFromIndexedDB("generated_prompts");
					generatedPrompts = cached || [];
					console.log(`[RPG] Loaded ${generatedPrompts.length} generated prompts`);
					return generatedPrompts;
				} catch (e) { generatedPrompts = []; return []; }
			};
			const saveGeneratedPrompts = async (prompts) => {
				try {
					await saveToIndexedDB("generated_prompts", prompts);
					generatedPrompts = prompts;
					return true;
				} catch (e) { return false; }
			};
			const addGeneratedPrompt = async (data) => {
				const p = {
					id: `gen_${Date.now()}_${Math.random().toString(36).substr(2,9)}`,
					tags: data.tags || [],
					prompt: { positive: data.positive || "", negative: data.negative || "" },
					classification: data.classification || { content_type: "other", safety_level: "sfw", shot_type: "other" },
					is_ai_generated: true,
					created_at: Date.now()
				};
				generatedPrompts.unshift(p);
				await saveGeneratedPrompts(generatedPrompts);
				promptsDatabase = promptsDatabase.filter(x => !x.is_ai_generated);
				const userCount = promptsDatabase.filter(x => x.is_user_created).length;
				promptsDatabase.splice(userCount, 0, ...generatedPrompts);
				return p;
			};

				const exportUserPrompts = () => {
					const bookmarks = JSON.parse(node.properties.bookmarks || "[]");
					const exportData = {
						version: "1.0",
						exported_at: new Date().toISOString(),
						prompts: userPrompts,
						bookmarks: bookmarks
					};
					const dataStr = JSON.stringify(exportData, null, 2);
					const blob = new Blob([dataStr], { type: "application/json" });
					const url = URL.createObjectURL(blob);
					const a = document.createElement("a");
					a.href = url;
					a.download = `rpg_user_prompts_${Date.now()}.json`;
					document.body.appendChild(a);
					a.click();
					document.body.removeChild(a);
					URL.revokeObjectURL(url);
					console.log(`[RPG] Exported ${userPrompts.length} user prompts and ${bookmarks.length} bookmarks`);
				};

				const importUserPrompts = async (file) => {
					return new Promise((resolve, reject) => {
						const reader = new FileReader();
						reader.onload = async (e) => {
							try {
								const data = JSON.parse(e.target.result);
								const imported = data.prompts || [];
								const importedBookmarks = data.bookmarks || [];

								// Validate format
								if (!Array.isArray(imported)) {
									throw new Error("Invalid format: prompts must be an array");
								}

								// Merge with existing (avoid duplicates by ID)
								const existingIds = new Set(userPrompts.map(p => p.id));
								let added = 0;
								imported.forEach(prompt => {
									if (!existingIds.has(prompt.id)) {
										userPrompts.push({
											...prompt,
											is_user_created: true,
											imported_at: Date.now()
										});
										added++;
									}
								});

								// Merge bookmarks (avoid duplicates)
								const currentBookmarks = JSON.parse(node.properties.bookmarks || "[]");
								const mergedBookmarks = [...new Set([...currentBookmarks, ...importedBookmarks])];
								node.properties.bookmarks = JSON.stringify(mergedBookmarks);

								await saveUserPrompts(userPrompts);
								mergeUserPromptsWithLibrary();
								renderUI();
								console.log(`[RPG] Imported ${added} new user prompts (${imported.length - added} duplicates skipped) and ${importedBookmarks.length} bookmarks`);
								resolve({ added, skipped: imported.length - added, bookmarks: importedBookmarks.length });
							} catch (error) {
								console.error("[RPG] Error importing user prompts:", error);
								reject(error);
							}
						};
						reader.onerror = reject;
						reader.readAsText(file);
					});
				};

				const mergeUserPromptsWithLibrary = () => {
					// Remove old user prompts from promptsDatabase
					promptsDatabase = promptsDatabase.filter(p => !p.is_user_created);
					// Add current user prompts to the beginning
					promptsDatabase = [...userPrompts, ...promptsDatabase];
					console.log(`[RPG] Merged database: ${userPrompts.length} user + ${promptsDatabase.length - userPrompts.length} library = ${promptsDatabase.length} total`);
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
									<span class="instaraw-rpg-mode-badge ${detectedMode === 'img2img' ? 'instaraw-rpg-mode-img2img' : 'instaraw-rpg-mode-txt2img'}" style="font-size: 14px; padding: 8px 16px; font-weight: 700;">
										${detectedMode === 'img2img' ? '🖼️ IMG2IMG MODE' : '🎨 TXT2IMG MODE'}
									</span>
									${isDetectedFromAIL ? `<span class="instaraw-rpg-mode-source">From AIL #${node._linkedAILNodeId}</span>` : ''}
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
								<span class="instaraw-rpg-stat-label">${detectedMode === 'img2img' ? 'IMG2IMG' : 'TXT2IMG'}</span>
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

					// Determine which prompts to show
					let filteredPrompts;
					let pagePrompts;
					let totalPages;

					if (showingRandomPrompts && randomPrompts.length > 0) {
						// Random mode: show the fetched random prompts
						filteredPrompts = randomPrompts;
						pagePrompts = randomPrompts; // Show all random prompts (no pagination)
						totalPages = 1;
					} else {
						// Normal mode: apply filters
						filteredPrompts = filterPrompts(promptsDatabase, filters);

						// Pagination
						totalPages = Math.ceil(filteredPrompts.length / itemsPerPage);
						const startIdx = currentPage * itemsPerPage;
						const endIdx = startIdx + itemsPerPage;
						pagePrompts = filteredPrompts.slice(startIdx, endIdx);
					}

					return `
						<div class="instaraw-rpg-library">
							<div class="instaraw-rpg-filters">
								<input type="text" class="instaraw-rpg-search-input" placeholder="🔍 Search by prompt, tags, or ID..." value="${filters.search_query || ""}" />
								<div class="instaraw-rpg-filter-row">
									<select class="instaraw-rpg-filter-dropdown" data-filter="content_type">
										<option value="any">All Content Types</option>
										<option value="person" ${filters.content_type === "person" ? "selected" : ""}>Person</option>
										<option value="object" ${filters.content_type === "object" ? "selected" : ""}>Object</option>
										<option value="other" ${filters.content_type === "other" ? "selected" : ""}>Other</option>
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
										<option value="other" ${filters.shot_type === "other" ? "selected" : ""}>Other</option>
									</select>
									<select class="instaraw-rpg-filter-dropdown" data-filter="prompt_source">
										<option value="all" ${filters.prompt_source === "all" || !filters.prompt_source ? "selected" : ""}>📚 All Prompts</option>
										<option value="library" ${filters.prompt_source === "library" ? "selected" : ""}>📚 Library Only</option>
										<option value="user" ${filters.prompt_source === "user" ? "selected" : ""}>✏️ My Prompts</option>
										<option value="generated" ${filters.prompt_source === "generated" ? "selected" : ""}>✨ Generated</option>
									</select>
									<label class="instaraw-rpg-checkbox-label" title="Show only bookmarked prompts">
										<input type="checkbox" class="instaraw-rpg-show-bookmarked-checkbox" ${filters.show_bookmarked ? "checked" : ""} />
										⭐ Favorites Only
									</label>
									<button class="instaraw-rpg-btn-secondary instaraw-rpg-clear-filters-btn">✖ Clear</button>
								</div>
							</div>

							<div class="instaraw-rpg-library-header">
								<div style="display: flex; align-items: center; gap: 12px; flex-wrap: wrap;">
									<span class="instaraw-rpg-result-count">
										${showingRandomPrompts
											? `🎲 Showing ${randomPrompts.length} random prompt${randomPrompts.length === 1 ? '' : 's'}`
											: `${filteredPrompts.length} prompt${filteredPrompts.length === 1 ? '' : 's'} found${filters.show_bookmarked ? ' (favorites only)' : ''}${totalPages > 1 ? ` • Page ${currentPage + 1} of ${totalPages}` : ''}`
										}
									</span>
									<label class="instaraw-rpg-sdxl-toggle" title="SDXL mode - show tags as main content">
										<input type="checkbox" class="instaraw-rpg-sdxl-mode-checkbox" ${filters.sdxl_mode ? "checked" : ""} />
										🎨 SDXL
									</label>
									<div style="display: flex; align-items: center; gap: 6px; margin-left: auto;">
										${showingRandomPrompts ? `
											<!-- Random Mode: Show Add All, Reroll, and Exit buttons -->
											<button class="instaraw-rpg-btn-primary instaraw-rpg-add-all-random-btn" style="font-size: 12px; padding: 6px 12px;">
												✓ Add All ${randomPrompts.length} to Batch
											</button>
											<button class="instaraw-rpg-btn-secondary instaraw-rpg-reroll-random-btn" style="font-size: 12px; padding: 6px 12px;" title="Get different random prompts">
												🎲 Reroll
											</button>
											<button class="instaraw-rpg-btn-secondary instaraw-rpg-exit-random-btn" style="font-size: 12px; padding: 6px 12px;">
												← Back to Library
											</button>
										` : `
											<!-- Normal Mode: Show Create/Import/Export and Random controls -->
											<button class="instaraw-rpg-btn-secondary instaraw-rpg-create-prompt-btn" style="font-size: 12px; padding: 6px 12px;" title="Create new custom prompt">
												➕ Create
											</button>
											<button class="instaraw-rpg-btn-secondary instaraw-rpg-import-prompts-btn" style="font-size: 12px; padding: 6px 12px;" title="Import prompts from JSON file">
												📂 Import
											</button>
											<button class="instaraw-rpg-btn-secondary instaraw-rpg-export-prompts-btn" style="font-size: 12px; padding: 6px 12px;" title="Export my prompts and bookmarks to JSON file" ${userPrompts.length === 0 && bookmarks.length === 0 ? 'disabled' : ''}>
												💾 Export (${userPrompts.length + bookmarks.length})
											</button>
											<div style="width: 1px; height: 20px; background: #4b5563; margin: 0 4px;"></div>
											<label style="font-size: 11px; color: #9ca3af; margin-right: 4px;">Random:</label>
											<input type="number" class="instaraw-rpg-random-count-input" value="${randomCount}" min="1" max="50" style="width: 50px; padding: 4px 6px; border: 1px solid rgba(255, 255, 255, 0.1); background: rgba(255, 255, 255, 0.05); color: #f9fafb; border-radius: 4px; font-size: 12px;" title="How many random prompts to show (uses current filters)" />
											<button class="instaraw-rpg-btn-secondary instaraw-rpg-show-random-btn" style="font-size: 12px; padding: 6px 12px;">
												🎲 Show Random
											</button>
										`}
									</div>
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
														const sourceBadge = prompt.is_user_created ? '✏️ My Prompt' : '📚 Library';

														const allTags = prompt.tags || [];
														const autoExpand = matchType === 'tags' || matchType === 'both'; // Auto-expand if tags match

														return `
									<div class="instaraw-rpg-library-card ${batchCount > 0 ? 'in-batch' : ''} ${prompt.is_user_created ? 'user-prompt' : ''}" data-id="${prompt.id}" data-is-user="${prompt.is_user_created ? 'true' : 'false'}">
										<div class="instaraw-rpg-library-card-header">
											<button class="instaraw-rpg-bookmark-btn ${bookmarks.includes(prompt.id) ? "bookmarked" : ""}" data-id="${prompt.id}">
												${bookmarks.includes(prompt.id) ? "⭐" : "☆"}
											</button>
											<div class="instaraw-rpg-batch-controls">
												<button class="instaraw-rpg-add-to-batch-btn" data-id="${prompt.id}">+ Add</button>
												${batchCount > 0 ? `<button class="instaraw-rpg-undo-batch-btn" data-id="${prompt.id}">↶ ${batchCount}</button>` : ''}
												${prompt.is_user_created ? `<button class="instaraw-rpg-delete-user-prompt-btn" data-id="${prompt.id}" title="Delete this prompt">🗑️</button>` : ''}
											</div>
										</div>
										<div class="instaraw-rpg-library-card-content">
											<div style="display: flex; gap: 6px; flex-wrap: wrap; margin-bottom: 6px; align-items: center;">
												<div class="instaraw-rpg-source-badge ${prompt.is_user_created ? 'user' : 'library'}">${sourceBadge}</div>
												${matchBadge ? `<div class="instaraw-rpg-match-badge">${matchBadge} Match</div>` : ''}
												<div class="instaraw-rpg-id-badge-container">
													<span class="instaraw-rpg-id-badge" title="Prompt ID: ${prompt.id}">ID: ${prompt.id.substring(0, 8)}..</span>
													<button class="instaraw-rpg-id-copy-btn" data-id="${prompt.id}" title="Copy full ID">📄</button>
												</div>
											</div>

											${prompt.is_user_created ? `
												<!-- User Created Prompt -->
												${editingPrompts.has(prompt.id) ? `
													<!-- EDIT MODE: Show textareas with Save/Cancel -->
													<div style="display: flex; flex-direction: column; gap: 8px;">
														<div>
															<label style="font-size: 11px; font-weight: 500; color: rgba(249, 250, 251, 0.7); text-transform: uppercase; display: block; margin-bottom: 4px;">Positive Prompt</label>
															<textarea class="instaraw-rpg-prompt-textarea instaraw-rpg-user-prompt-edit-positive" data-id="${prompt.id}">${escapeHtml(editingValues[prompt.id]?.positive ?? positive)}</textarea>
														</div>
														<div>
															<label style="font-size: 11px; font-weight: 500; color: rgba(249, 250, 251, 0.7); text-transform: uppercase; display: block; margin-bottom: 4px;">Negative Prompt</label>
															<textarea class="instaraw-rpg-prompt-textarea instaraw-rpg-user-prompt-edit-negative" data-id="${prompt.id}">${escapeHtml(editingValues[prompt.id]?.negative ?? negative)}</textarea>
														</div>
														<div>
															<label style="font-size: 11px; font-weight: 500; color: rgba(249, 250, 251, 0.7); text-transform: uppercase; display: block; margin-bottom: 4px;">Tags (comma-separated)</label>
															<input type="text" class="instaraw-rpg-prompt-textarea instaraw-rpg-user-prompt-edit-tags" data-id="${prompt.id}" value="${editingValues[prompt.id]?.tags ?? allTags.join(", ")}" placeholder="tag1, tag2, tag3..." style="resize: none; min-height: auto; height: auto;" />
														</div>

														<!-- Classification Fields -->
														<div style="display: flex; flex-direction: column; gap: 8px; margin-top: 4px;">
															<div>
																<label style="font-size: 11px; font-weight: 500; color: rgba(249, 250, 251, 0.7); text-transform: uppercase; display: block; margin-bottom: 4px;">Content Type</label>
																<select class="instaraw-rpg-filter-dropdown instaraw-rpg-user-prompt-edit-content-type" data-id="${prompt.id}" style="width: 100%; padding: 6px 8px;">
																	<option value="person" ${(editingValues[prompt.id]?.content_type ?? prompt.classification?.content_type ?? 'person') === 'person' ? 'selected' : ''}>Person</option>
																	<option value="object" ${(editingValues[prompt.id]?.content_type ?? prompt.classification?.content_type ?? 'person') === 'object' ? 'selected' : ''}>Object</option>
																	<option value="other" ${(editingValues[prompt.id]?.content_type ?? prompt.classification?.content_type ?? 'person') === 'other' ? 'selected' : ''}>Other</option>
																</select>
															</div>
															<div>
																<label style="font-size: 11px; font-weight: 500; color: rgba(249, 250, 251, 0.7); text-transform: uppercase; display: block; margin-bottom: 4px;">Safety Level</label>
																<select class="instaraw-rpg-filter-dropdown instaraw-rpg-user-prompt-edit-safety-level" data-id="${prompt.id}" style="width: 100%; padding: 6px 8px;">
																	<option value="sfw" ${(editingValues[prompt.id]?.safety_level ?? prompt.classification?.safety_level ?? 'sfw') === 'sfw' ? 'selected' : ''}>SFW</option>
																	<option value="suggestive" ${(editingValues[prompt.id]?.safety_level ?? prompt.classification?.safety_level ?? 'sfw') === 'suggestive' ? 'selected' : ''}>Suggestive</option>
																	<option value="nsfw" ${(editingValues[prompt.id]?.safety_level ?? prompt.classification?.safety_level ?? 'sfw') === 'nsfw' ? 'selected' : ''}>NSFW</option>
																</select>
															</div>
															<div>
																<label style="font-size: 11px; font-weight: 500; color: rgba(249, 250, 251, 0.7); text-transform: uppercase; display: block; margin-bottom: 4px;">Shot Type</label>
																<select class="instaraw-rpg-filter-dropdown instaraw-rpg-user-prompt-edit-shot-type" data-id="${prompt.id}" style="width: 100%; padding: 6px 8px;">
																	<option value="portrait" ${(editingValues[prompt.id]?.shot_type ?? prompt.classification?.shot_type ?? 'portrait') === 'portrait' ? 'selected' : ''}>Portrait</option>
																	<option value="full_body" ${(editingValues[prompt.id]?.shot_type ?? prompt.classification?.shot_type ?? 'portrait') === 'full_body' ? 'selected' : ''}>Full Body</option>
																	<option value="other" ${(editingValues[prompt.id]?.shot_type ?? prompt.classification?.shot_type ?? 'portrait') === 'other' ? 'selected' : ''}>Other</option>
																</select>
															</div>
														</div>

														<div style="display: flex; gap: 6px; margin-top: 4px;">
															<button class="instaraw-rpg-btn-primary instaraw-rpg-save-user-prompt-btn" data-id="${prompt.id}" style="font-size: 11px; padding: 6px 12px; flex: 1;">
																💾 Save
															</button>
															<button class="instaraw-rpg-btn-secondary instaraw-rpg-cancel-edit-prompt-btn" data-id="${prompt.id}" style="font-size: 11px; padding: 6px 12px; flex: 1;">
																✖ Cancel
															</button>
														</div>
													</div>
												` : sdxlMode ? `
													<!-- VIEW MODE (SDXL): Show tags as comma-separated text -->
													<div class="instaraw-rpg-prompt-preview ${allTags.length === 0 ? 'instaraw-rpg-error-text' : ''}">
														${allTags.length > 0 ? allTags.map((tag) => highlightSearchTerm(tag, searchQuery)).join(", ") : "[Empty prompt]"}
													</div>
													<button class="instaraw-rpg-btn-secondary instaraw-rpg-edit-user-prompt-btn" data-id="${prompt.id}" style="font-size: 11px; padding: 4px 10px; margin-top: 8px; width: 100%;">
														✏️ Edit Prompt
													</button>
												` : `
													<!-- VIEW MODE (Normal): Show prompt with tags -->
													<div class="instaraw-rpg-prompt-preview ${!positive ? 'instaraw-rpg-error-text' : ''}">${highlightSearchTerm(displayText, searchQuery)}</div>
													${negative ? `<div class="instaraw-rpg-prompt-preview" style="font-size: 11px; color: #9ca3af; margin-top: 4px;"><strong>Negative:</strong> ${highlightSearchTerm(negative, searchQuery)}</div>` : ''}
													<div class="instaraw-rpg-library-card-tags" style="margin-top: 8px;">
														${allTags.map((tag) => `<span class="instaraw-rpg-tag">${highlightSearchTerm(tag, searchQuery)}</span>`).join("")}
													</div>
													<button class="instaraw-rpg-btn-secondary instaraw-rpg-edit-user-prompt-btn" data-id="${prompt.id}" style="font-size: 11px; padding: 4px 10px; margin-top: 8px; width: 100%;">
														✏️ Edit Prompt
													</button>
												`}
											` : sdxlMode ? `
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
				const detectedMode = node._linkedAILMode || "txt2img";
				const modelOptionsHtml = CREATIVE_MODEL_OPTIONS
					.map(
						(opt) => `<option value="${opt.value}" ${opt.value === (uiState?.currentCreativeModel || "gemini-2.5-pro") ? "selected" : ""}>${opt.label}</option>`
					)
					.join("");
				const temperature = uiState?.currentCreativeTemperature ?? 0.9;
				const topP = uiState?.currentCreativeTopP ?? 0.9;

				// Character description - generated descriptions populate the character_text_input
				const characterText = node.properties.character_text_input || "";
				// Check if character_image is connected AND the source node still exists
				const characterImageInput = node.inputs?.find(i => i.name === "character_image");
				let hasCharacterImage = false;
				if (characterImageInput && characterImageInput.link != null) {
					const link = app.graph.links[characterImageInput.link];
					if (link) {
						const sourceNode = app.graph.getNodeById(link.origin_id);
						hasCharacterImage = !!sourceNode; // Only true if node still exists
					}
				}

				return `
					<div class="instaraw-rpg-generate-unified">
						<!-- Model Settings (TOP - affects ALL generation including character) -->
						<div class="instaraw-rpg-section">
							<div class="instaraw-rpg-section-header">
								<span class="instaraw-rpg-section-label">⚙️ Model Settings</span>
								<span class="instaraw-rpg-hint-text" style="font-size: 11px; color: #9ca3af;">Used for all generation</span>
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
								</div>
							</div>
						</div>

						<!-- Character Consistency (Optional) -->
						<div class="instaraw-rpg-section">
							<div class="instaraw-rpg-section-header">
								<label class="instaraw-rpg-checkbox-label">
									<input type="checkbox" class="instaraw-rpg-checkbox instaraw-rpg-enable-character-checkbox" ${node.properties.use_character_likeness ? 'checked' : ''} />
									<span>🎭 Character Consistency</span>
								</label>
								<span class="instaraw-rpg-hint-badge">${
									!node.properties.use_character_likeness ? '⚪ Unused' :
									characterText.trim() ? '✅ Active' : '⚠️ Empty'
								}</span>
							</div>
							<div class="instaraw-rpg-character-section" style="display: ${node.properties.use_character_likeness ? 'block' : 'none'};">
								${hasCharacterImage ? `
									<div class="instaraw-rpg-info-banner" style="background: rgba(59, 130, 246, 0.1); border: 1px solid rgba(59, 130, 246, 0.3); padding: 8px 12px; border-radius: 4px; margin-bottom: 12px; font-size: 12px; color: #93c5fd;">
										📸 <strong>Character Image Connected</strong> - Click "Generate from Image" to create description
									</div>
								` : ''}
								<div class="instaraw-rpg-control-group">
									<label>Character Description</label>
									<textarea class="instaraw-rpg-character-text-input instaraw-rpg-prompt-textarea" placeholder="blonde hair, blue eyes, athletic build, fair skin, delicate features..." style="line-height: 1.42; min-height: 60px; resize: vertical;">${escapeHtml(characterText)}</textarea>
									<div class="instaraw-rpg-hint-text" style="margin-top: 8px; font-size: 11px; color: #9ca3af;">
										💡 This description will be included in ALL generated prompts for character consistency
									</div>
								</div>
								<div class="instaraw-rpg-character-generation-settings">
									<div class="instaraw-rpg-control-row" style="display: grid; grid-template-columns: auto 1fr; gap: 8px; align-items: center; margin-bottom: 12px;">
										<label style="margin: 0; font-size: 12px;">Complexity</label>
										<select class="instaraw-rpg-model-select instaraw-rpg-character-complexity">
											<option value="concise" ${(node.properties.character_complexity || 'balanced') === 'concise' ? 'selected' : ''}>Concise (50-75 words)</option>
											<option value="balanced" ${(node.properties.character_complexity || 'balanced') === 'balanced' ? 'selected' : ''}>Balanced (100-150 words)</option>
											<option value="detailed" ${(node.properties.character_complexity || 'balanced') === 'detailed' ? 'selected' : ''}>Detailed (200-250 words)</option>
										</select>
									</div>
									${hasCharacterImage ? `
										<div class="instaraw-rpg-character-actions" style="margin-bottom: 12px; display: flex; align-items: center; gap: 8px;">
											<button class="instaraw-rpg-btn-secondary instaraw-rpg-generate-character-desc-btn">
												✨ Generate from Image
											</button>
											<span class="instaraw-rpg-hint-text" style="font-size: 11px; color: #9ca3af;">Uses connected character_image and selected model above</span>
										</div>
									` : ''}
									<details class="instaraw-rpg-advanced-settings" style="margin-top: 12px; padding: 12px; background: #1f2937; border: 1px solid #4b5563; border-radius: 4px;">
										<summary style="cursor: pointer; font-weight: 500; font-size: 12px; color: #9ca3af; user-select: none;">⚙️ Advanced: Edit System Prompt</summary>
										<div style="margin-top: 12px;">
											<textarea class="instaraw-rpg-character-system-prompt instaraw-rpg-prompt-textarea" style="font-family: monospace; font-size: 11px; line-height: 1.5; min-height: 80px; resize: vertical; width: 100%;">${escapeHtml(node.properties.character_system_prompt || getCharacterSystemPrompt(node.properties.character_complexity || "balanced"))}</textarea>
											<div style="display: flex; align-items: center; justify-content: space-between; margin-top: 8px;">
												<div class="instaraw-rpg-hint-text" style="font-size: 10px; color: #9ca3af;">
													💡 Custom edits override complexity setting
												</div>
												<button class="instaraw-rpg-btn-text instaraw-rpg-reset-system-prompt-btn" style="font-size: 11px; padding: 4px 8px;">🔄 Reset</button>
											</div>
										</div>
									</details>
								</div>
							</div>
						</div>

						<!-- Mode Detection & Settings -->
						<div class="instaraw-rpg-section">
							<div class="instaraw-rpg-section-header">
								<span class="instaraw-rpg-mode-badge ${detectedMode === 'img2img' ? 'instaraw-rpg-mode-img2img' : 'instaraw-rpg-mode-txt2img'}">
									${detectedMode === 'img2img' ? '🖼️ IMG2IMG' : '🎨 TXT2IMG'}
								</span>
								<span class="instaraw-rpg-hint-text">Detected from ${node._linkedAILNodeId ? `AIL #${node._linkedAILNodeId}` : 'workflow'}</span>
							</div>

							<!-- Reality vs Creative Mode -->
							<div class="instaraw-rpg-generation-mode-selector" style="margin: 12px 0;">
								<label class="instaraw-rpg-section-label" style="margin-bottom: 8px; display: block;">Generation Mode</label>
								<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px;">
									<button class="instaraw-rpg-mode-toggle-btn ${node.properties.generation_style !== 'creative' ? 'active' : ''}" data-mode="reality" style="padding: 12px; border: 2px solid ${node.properties.generation_style !== 'creative' ? '#60a5fa' : '#4b5563'}; background: ${node.properties.generation_style !== 'creative' ? 'rgba(59, 130, 246, 0.1)' : 'transparent'}; border-radius: 6px; cursor: pointer; transition: all 0.2s;">
										<div style="font-weight: 600; font-size: 13px; color: ${node.properties.generation_style !== 'creative' ? '#60a5fa' : '#e5e7eb'};">🎯 Reality Mode</div>
										<div style="font-size: 11px; color: #9ca3af; margin-top: 4px;">Strict adherence to library prompts</div>
									</button>
									<button class="instaraw-rpg-mode-toggle-btn ${node.properties.generation_style === 'creative' ? 'active' : ''}" data-mode="creative" style="padding: 12px; border: 2px solid ${node.properties.generation_style === 'creative' ? '#8b5cf6' : '#4b5563'}; background: ${node.properties.generation_style === 'creative' ? 'rgba(139, 92, 246, 0.1)' : 'transparent'}; border-radius: 6px; cursor: pointer; transition: all 0.2s;">
										<div style="font-weight: 600; font-size: 13px; color: ${node.properties.generation_style === 'creative' ? '#8b5cf6' : '#e5e7eb'};">✨ Creative Mode</div>
										<div style="font-size: 11px; color: #9ca3af; margin-top: 4px;">Inspired by library prompts (flexible)</div>
									</button>
								</div>
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
										<textarea class="instaraw-rpg-user-text-input instaraw-rpg-prompt-textarea" placeholder="Describe what you want to generate or leave empty to use only library prompts..." rows="3" style="line-height: 1.42;">${escapeHtml(node.properties.user_text_input || "")}</textarea>
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

						<!-- Progress Tracking Section -->
						<div class="instaraw-rpg-generation-progress" style="display: none;">
							<div class="instaraw-rpg-progress-header">
								<h4>Generating Prompts...</h4>
								<button class="instaraw-rpg-btn-secondary instaraw-rpg-cancel-generation-btn">⏹ Cancel Generation</button>
							</div>
							<div class="instaraw-rpg-progress-items"></div>
						</div>

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
												// Both modes: Show aspect ratio box with content inside
												const targetDims = getTargetDimensions();
												const targetAspectRatio = targetDims.width / targetDims.height;

												if (detectedMode === "img2img") {
													// IMG2IMG: Show image in aspect ratio box with aspect ratio overlay
													thumbnailHtml = `
														<div class="instaraw-rpg-batch-thumbnail instaraw-rpg-batch-thumbnail-latent">
															<span class="instaraw-rpg-batch-thumbnail-index">#${idx + 1}</span>
															<div class="instaraw-rpg-batch-aspect-preview" style="aspect-ratio: ${targetAspectRatio}; position: relative; overflow: hidden;">
																<img src="${linkedItem.url}" alt="Linked image ${idx + 1}" style="width: 100%; height: 100%; object-fit: cover;" />
																<div class="instaraw-rpg-batch-aspect-content" style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; display: flex; align-items: center; justify-content: center; pointer-events: none;">
																	<div style="font-size: 14px; font-weight: 600; color: white; text-shadow: 0 2px 4px rgba(0,0,0,0.8); background: rgba(139, 92, 246, 0.75); padding: 4px 10px; border-radius: 4px;">${targetDims.aspect_label}</div>
																</div>
															</div>
														</div>
													`;
												} else {
													// TXT2IMG: Show empty latent with aspect ratio box
													thumbnailHtml = `
														<div class="instaraw-rpg-batch-thumbnail instaraw-rpg-batch-thumbnail-latent">
															<span class="instaraw-rpg-batch-thumbnail-index">#${idx + 1}</span>
															<div class="instaraw-rpg-batch-aspect-preview" style="aspect-ratio: ${targetAspectRatio};">
																<div class="instaraw-rpg-batch-aspect-content">
																	<div style="font-size: 24px;">📐</div>
																	<div style="font-size: 14px; font-weight: 600;">${targetDims.aspect_label}</div>
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
								<label class="instaraw-rpg-thumbnail-label">${detectedMode === "img2img" ? "IMG2IMG Input Image" : "TXT2IMG Empty Latent"}</label>
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

					// Check for repeat count mismatches
					const hasRepeatMismatch = promptQueue.some((p, idx) => {
						const linkedItem = detectedMode === "img2img" ? linkedImages[idx] : linkedLatents[idx];
						return linkedItem && (p.repeat_count || 1) !== (linkedItem.repeat_count || 1);
					});

					// Show smart sync button if: AIL linked + has prompts
					// Will handle both latent creation (txt2img) and repeat syncing (both modes)
					const showSyncButton = hasAILLink && promptQueue.length > 0;
					const needsLatentSync = detectedMode === "txt2img" && linkedLatents.length !== promptQueue.length;
					const needsRepeatSync = hasRepeatMismatch;

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
									<button class="instaraw-rpg-btn-${needsLatentSync || needsRepeatSync ? 'primary' : 'secondary'} instaraw-rpg-sync-ail-btn ${needsLatentSync || needsRepeatSync ? 'instaraw-rpg-btn-warning' : ''}"
										title="${needsLatentSync ? `Create ${totalGenerations} latents in AIL` : ''}${needsLatentSync && needsRepeatSync ? ' and ' : ''}${needsRepeatSync ? 'Sync repeat counts' : ''}${!needsLatentSync && !needsRepeatSync ? 'Everything synced!' : ''}">
										${needsLatentSync || needsRepeatSync ? '⚠️ ' : '✓ '}Sync AIL${needsLatentSync ? ` (${totalGenerations})` : ''}
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
					const displayItems = items; // Show ALL items, not just first 10
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
										// Both modes use same structure with target dimensions
										const targetDims = getTargetDimensions();
										const targetAspectRatio = targetDims.width / targetDims.height;

										if (isImg2Img) {
											// IMG2IMG: Show image in aspect ratio box
											return `
												<div class="instaraw-rpg-preview-thumb">
													<span class="instaraw-rpg-preview-index">#${idx + 1}</span>
													<img src="${item.url}" alt="Preview ${idx + 1}" />
													${item.repeat_count && item.repeat_count > 1 ? `<span class="instaraw-rpg-preview-repeat">×${item.repeat_count}</span>` : ''}
												</div>
											`;
										} else {
											// TXT2IMG: Show empty latent with emoji and aspect ratio below
											return `
												<div class="instaraw-rpg-preview-latent">
													<span class="instaraw-rpg-preview-index">#${idx + 1}</span>
													<div class="instaraw-rpg-preview-aspect-box" style="aspect-ratio: ${targetAspectRatio};">
														<div class="instaraw-rpg-preview-aspect-content">
															<div style="font-size: 20px;">📐</div>
															<div style="font-size: 11px; font-weight: 600; margin-top: 4px;">${targetDims.aspect_label}</div>
														</div>
													</div>
													${item.repeat_count && item.repeat_count > 1 ? `<span class="instaraw-rpg-preview-repeat">×${item.repeat_count}</span>` : ''}
												</div>
											`;
										}
									})
									.join("")}
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
							const id = (p.id || "").toLowerCase();
							const matchInPrompt = positive.includes(query);
							const matchInTags = tags.includes(query);
							const matchInId = id.includes(query);
							// Store match type for highlighting
							p._matchType = matchInPrompt ? (matchInTags ? 'both' : 'prompt') : (matchInTags ? 'tags' : (matchInId ? 'id' : null));
							return matchInPrompt || matchInTags || matchInId;
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

					// Prompt source (user vs library)
					if (filters.prompt_source && filters.prompt_source !== "all") {
						if (filters.prompt_source === "user") {
							filtered = filtered.filter((p) => p.is_user_created === true);
						} else if (filters.prompt_source === "generated") {
						filtered = filtered.filter((p) => p.is_ai_generated === true);
					} else if (filters.prompt_source === "library") {
							filtered = filtered.filter((p) => !p.is_user_created && !p.is_ai_generated);
						}
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
					// Generate ULID (Universally Unique Lexicographically Sortable Identifier)
					// 26 characters: 10 char timestamp + 16 char randomness
					const ENCODING = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"; // Crockford's Base32
					const ENCODING_LEN = ENCODING.length;
					const TIME_LEN = 10;
					const RANDOM_LEN = 16;

					const encodeTime = (now, len) => {
						let str = "";
						for (let i = len; i > 0; i--) {
							const mod = now % ENCODING_LEN;
							str = ENCODING.charAt(mod) + str;
							now = (now - mod) / ENCODING_LEN;
						}
						return str;
					};

					const encodeRandom = (len) => {
						let str = "";
						for (let i = 0; i < len; i++) {
							str += ENCODING.charAt(Math.floor(Math.random() * ENCODING_LEN));
						}
						return str;
					};

					const now = Date.now();
					return encodeTime(now, TIME_LEN) + encodeRandom(RANDOM_LEN);
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
					const generateBtn = container.querySelector(".instaraw-rpg-generate-creative-btn");

					const generationCount = parseInt(genCountInput?.value || "5");
					const inspirationCount = parseInt(inspirationCountInput?.value || "3");
					const isSDXL = isSDXLCheckbox?.checked || false;
					const forceRegenerate = true; // Always regenerate

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

					// Disable button and show loading progress bar
					if (generateBtn) {
						generateBtn.disabled = true;
						const originalText = generateBtn.textContent;
						generateBtn.style.position = 'relative';
						generateBtn.style.overflow = 'hidden';
						generateBtn.innerHTML = `
							${originalText}
							<div class="instaraw-rpg-progress-bar-loading"></div>
						`;
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
								// DON'T call setupEventHandlers() - already set up by renderUI()
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
					const generateBtn = container.querySelector(".instaraw-rpg-generate-character-btn");

					const characterReference = charRefInput?.value || "";
					const generationCount = parseInt(genCountInput?.value || "5");
					const isSDXL = isSDXLCheckbox?.checked || false;
					const forceRegenerate = true; // Always regenerate

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

					// Disable button and show loading progress bar
					if (generateBtn) {
						generateBtn.disabled = true;
						const originalText = generateBtn.textContent;
						generateBtn.style.position = 'relative';
						generateBtn.style.overflow = 'hidden';
						generateBtn.innerHTML = `
							${originalText}
							<div class="instaraw-rpg-progress-bar-loading"></div>
						`;
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
								// DON'T call setupEventHandlers() - already set up by renderUI()
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

				// ========================================
				// === UNIFIED GENERATE TAB FUNCTIONS ===
				// ========================================

				// === Get Default Character System Prompt (mirrors backend logic) ===
				const getCharacterSystemPrompt = (complexity = "balanced") => {
					const baseInstruction = `You are an expert at analyzing images and generating character descriptions for image generation prompts.

Generate a character description focusing on PERMANENT physical features:
- Facial features (face shape, eyes, nose, lips, skin tone)
- Hair (color, length, style, texture)
- Body type and build
- Age and ethnicity
- Distinctive features (scars, tattoos, piercings, etc.)

DO NOT include clothing, background, pose, or temporary features.
DO NOT use tags like "1girl, solo" or similar categorization prefixes.`;

					let lengthInstruction;
					if (complexity === "concise") {
						lengthInstruction = "\nOUTPUT: A concise description (50-75 words) focusing only on the most essential and distinctive physical features.";
					} else if (complexity === "detailed") {
						lengthInstruction = "\nOUTPUT: A comprehensive, detailed description (200-250 words) covering all physical aspects with nuanced detail and specific characteristics.";
					} else {  // balanced
						lengthInstruction = "\nOUTPUT: A balanced description (100-150 words) covering key physical features in natural language.";
					}

					return baseInstruction + lengthInstruction;
				};

				// === Generate Character Description (from image or text) ===
				const generateCharacterDescription = async () => {
					// CRASH PREVENTION: Check if already generating
					if (isGenerating) {
						console.warn("[RPG] ⚠️ Generation already in progress, ignoring request");
						return;
					}

					console.group("[RPG] 🎨 Generate Character Description - START");

					// Set generation lock
					isGenerating = true;

					const generateBtn = container.querySelector(".instaraw-rpg-generate-character-desc-btn");
					const characterTextInput = container.querySelector(".instaraw-rpg-character-text-input");

					const characterText = characterTextInput?.value?.trim() || "";

					// Get API keys
					const geminiApiKey = (getFinalInputValue("gemini_api_key", "") || "").trim() || window.INSTARAW_GEMINI_KEY || "";
					const grokApiKey = (getFinalInputValue("grok_api_key", "") || "").trim() || window.INSTARAW_GROK_KEY || "";

					if (!geminiApiKey && !grokApiKey) {
						alert("No API keys found! Please connect a Primitive String node with your Gemini or Grok API key.");
						console.groupEnd();
						return;
					}

					// Check if character_image input is connected
					const characterImageInput = node.inputs?.find(i => i.name === "character_image");
					let imageData = null;

					if (characterImageInput && characterImageInput.link != null) {
						console.log("[RPG] Character image input connected, reading image...");

						try {
							// Get the connected node
							const link = app.graph.links[characterImageInput.link];
							if (link) {
								const sourceNode = app.graph.getNodeById(link.origin_id);
								console.log("[RPG] Source node type:", sourceNode?.type);

								if (sourceNode && sourceNode.type === "LoadImage") {
									// Get the image filename from the LoadImage node
									const imageWidget = sourceNode.widgets?.find(w => w.name === "image");
									const filename = imageWidget?.value;

									if (filename) {
										console.log("[RPG] Loading image:", filename);

										// Fetch the image from ComfyUI's view endpoint
										const imageUrl = `/view?filename=${encodeURIComponent(filename)}&type=input`;
										const response = await fetch(imageUrl);
										const blob = await response.blob();

										// Convert to base64
										const base64 = await new Promise((resolve) => {
											const reader = new FileReader();
											reader.onloadend = () => resolve(reader.result);
											reader.readAsDataURL(blob);
										});

										imageData = base64;
										console.log("[RPG] ✅ Image loaded and converted to base64");
									} else {
										console.warn("[RPG] No image selected in LoadImage node");
									}
								} else if (sourceNode) {
									console.warn("[RPG] Connected node is not LoadImage, type:", sourceNode.type);
								}
							}
						} catch (error) {
							console.error("[RPG] Error reading character image:", error);
							alert(`Could not read character image: ${error.message}`);
							return;
						}
					}

					// Require either image or text
					if (!imageData && !characterText) {
						alert("Please either:\n1. Connect a Load Image node to character_image input, OR\n2. Enter character description text");
						console.groupEnd();
						return;
					}

					// Disable button and show loading progress bar
					if (generateBtn) {
						generateBtn.disabled = true;
						const originalText = generateBtn.textContent;
						generateBtn.style.position = 'relative';
						generateBtn.style.overflow = 'hidden';
						generateBtn.innerHTML = `
							${originalText}
							<div class="instaraw-rpg-progress-bar-loading"></div>
						`;
					}

					try {
						// Get complexity and custom system prompt from properties
						const complexity = node.properties.character_complexity || "balanced";
						const customSystemPrompt = node.properties.character_system_prompt?.trim() || "";

						console.log("[RPG] Sending character description request:", {
							hasImage: !!imageData,
							hasText: !!characterText,
							model: node.properties.creative_model || "gemini-2.5-pro",
							complexity: complexity,
							hasCustomSystemPrompt: !!customSystemPrompt
						});

						const response = await api.fetchApi("/instaraw/generate_character_description", {
							method: "POST",
							headers: { "Content-Type": "application/json" },
							body: JSON.stringify({
								character_image: imageData,
								character_text: characterText,
								gemini_api_key: geminiApiKey,
								grok_api_key: grokApiKey,
								model: node.properties.creative_model || "gemini-2.5-pro",
								complexity: complexity,
								custom_system_prompt: customSystemPrompt,
								force_regenerate: true  // Always force regenerate to bypass bad cache
							}),
						});

						const result = await parseJSONResponse(response);
						console.log("[RPG] Character description API response:", result);

						if (!response.ok) {
							throw new Error(result?.error || `Character description API error ${response.status}`);
						}

						if (result.success) {
							if (!result.description || result.description.trim() === "") {
								throw new Error("API returned empty description. Check backend logs for details.");
							}

							console.log(`[RPG] ✅ Character description generated successfully (${result.description.length} chars)`);

							// Populate the character text input with the generated description
							node.properties.character_text_input = result.description;

							// Update textarea directly (don't call renderUI to avoid event handler duplication)
							if (characterTextInput) {
								characterTextInput.value = result.description;
								autoResizeTextarea(characterTextInput);
							}

							// Mark canvas as dirty to save changes
							app.graph.setDirtyCanvas(true, true);
						} else {
							throw new Error(result.error || "API returned success=false");
						}
					} catch (error) {
						console.error("[RPG] ❌ Error during character description generation:", error);
						alert(`Character description error: ${error.message || error}`);
					} finally {
						// CRASH PREVENTION: Always release generation lock
						isGenerating = false;

						// Reset button
						if (generateBtn) {
							generateBtn.disabled = false;
							generateBtn.style.position = '';
							generateBtn.style.overflow = '';
							generateBtn.textContent = "✨ Generate from Image";
						}
						console.groupEnd();
					}
				};

				// === Generate Unified Prompts (Main Generate Button) - SEQUENTIAL VERSION ===
				let generationAbortController = null; // For cancel functionality

				const generateUnifiedPrompts = async () => {
					console.group("[RPG] 🎯 Generate Unified Prompts - START (Sequential)");
					console.log("[RPG] Timestamp:", new Date().toISOString());

					const generateBtn = container.querySelector(".instaraw-rpg-generate-unified-btn");
					const genCountInput = container.querySelector(".instaraw-rpg-gen-count-input");
					const isSDXLCheckbox = container.querySelector(".instaraw-rpg-is-sdxl-checkbox");
					const progressSection = container.querySelector(".instaraw-rpg-generation-progress");
					const progressItems = container.querySelector(".instaraw-rpg-progress-items");
					const previewSection = container.querySelector(".instaraw-rpg-generate-preview");

					// Character settings
					const enableCharacterCheckbox = container.querySelector(".instaraw-rpg-enable-character-checkbox");
					const characterTextInput = container.querySelector(".instaraw-rpg-character-text-input");
					const useCharacter = enableCharacterCheckbox?.checked || false;
					const characterText = characterTextInput?.value?.trim() || "";
					// Use character_text_input directly - generated descriptions populate this field
					const characterDescription = characterText;

					// Detect mode from AIL
					const detectedMode = node._linkedAILMode || "txt2img";
					const isImg2Img = detectedMode === "img2img";

					// Common settings
					const generationCount = parseInt(genCountInput?.value || "5");
					const forceRegenerate = true; // Always regenerate
					const isSDXL = isSDXLCheckbox?.checked || false;

					// Mode-specific settings
					let affectElements = [];
					let userInput = "";
					let inspirationCount = 0;

					if (isImg2Img) {
						// IMG2IMG: Collect affect elements
						const affectBackground = container.querySelector(".instaraw-rpg-affect-background")?.checked;
						const affectOutfit = container.querySelector(".instaraw-rpg-affect-outfit")?.checked;
						const affectPose = container.querySelector(".instaraw-rpg-affect-pose")?.checked;
						const affectLighting = container.querySelector(".instaraw-rpg-affect-lighting")?.checked;

						if (affectBackground) affectElements.push("background");
						if (affectOutfit) affectElements.push("outfit");
						if (affectPose) affectElements.push("pose");
						if (affectLighting) affectElements.push("lighting");
					} else {
						// TXT2IMG: Collect user input and inspiration
						const userTextInput = container.querySelector(".instaraw-rpg-user-text-input");
						const inspirationCountInput = container.querySelector(".instaraw-rpg-inspiration-count");

						userInput = userTextInput?.value?.trim() || "";
						inspirationCount = parseInt(inspirationCountInput?.value || "3");
					}

					const promptQueue = parsePromptBatch();
					const sourcePrompts = promptQueue.filter((p) => p.source_id).slice(0, inspirationCount);

					// Get model and settings
					const modelWidget = node.widgets?.find((w) => w.name === "creative_model");
					const model = modelWidget?.value || node.properties.creative_model || "gemini-2.5-pro";

					// FORCE USE NEW SYSTEM PROMPT - reset old cached value
					node.properties.creative_system_prompt = DEFAULT_RPG_SYSTEM_PROMPT;
					const systemPrompt = DEFAULT_RPG_SYSTEM_PROMPT;

					const temperatureValue = parseFloat(node.properties.creative_temperature ?? 0.9) || 0.9;
					const topPValue = parseFloat(node.properties.creative_top_p ?? 0.9) || 0.9;

					console.log(`[RPG] 🔧 Using system prompt (${systemPrompt.length} chars):`, systemPrompt.slice(0, 150) + "...");

					console.log("[RPG] Configuration:", {
						mode: detectedMode,
						generationCount,
						useCharacter,
						affectElements,
						userInput,
						inspirationCount,
						isSDXL,
						forceRegenerate,
						model,
						temperature: temperatureValue,
						topP: topPValue
					});

					// Get API keys
					const geminiApiKey = (getFinalInputValue("gemini_api_key", "") || "").trim() || window.INSTARAW_GEMINI_KEY || "";
					const grokApiKey = (getFinalInputValue("grok_api_key", "") || "").trim() || window.INSTARAW_GROK_KEY || "";

					if (!geminiApiKey && !grokApiKey) {
						console.error("[RPG] ❌ NO API KEYS FOUND!");
						console.groupEnd();
						alert("No API keys found! Please connect a Primitive String node with your Gemini or Grok API key.");
						return;
					}

					// Initialize AbortController for cancellation
					generationAbortController = new AbortController();
					const signal = generationAbortController.signal;

					// Disable generate button and show progress section
					if (generateBtn) {
						generateBtn.disabled = true;
						generateBtn.style.opacity = "0.5";
					}

					if (previewSection) {
						previewSection.style.display = "none";
					}

					if (progressSection) {
						progressSection.style.display = "block";
					}

					// Create progress items
					if (progressItems) {
						progressItems.innerHTML = "";
						for (let i = 0; i < generationCount; i++) {
							const progressItem = document.createElement("div");
							progressItem.className = "instaraw-rpg-progress-item";
							progressItem.dataset.index = i;
							progressItem.innerHTML = `
								<div class="instaraw-rpg-progress-item-header">
									<span class="instaraw-rpg-progress-item-label">Prompt ${i + 1}/${generationCount}</span>
									<span class="instaraw-rpg-progress-item-status pending">⏳ Pending</span>
								</div>
								<div class="instaraw-rpg-progress-item-bar">
									<div class="instaraw-rpg-progress-item-fill" style="width: 0%"></div>
								</div>
								<div class="instaraw-rpg-progress-item-message"></div>
							`;
							progressItems.appendChild(progressItem);
						}
						// Resize node to fit progress UI
						updateCachedHeight();
					}

					// Collect all generated prompts
					const allGeneratedPrompts = [];
					let cancelRequested = false;

					// Listen for cancel signal
					signal.addEventListener('abort', () => {
						cancelRequested = true;
						console.log("[RPG] 🛑 Cancel requested by user");
					});

					try {
						// PARALLEL generation with 222ms stagger
						console.log(`[RPG] 🚀 Launching ${generationCount} parallel requests with 222ms stagger...`);

						const generateSinglePrompt = async (index) => {
							// Check if cancelled
							if (cancelRequested) {
								console.log(`[RPG] ⏹ Generation cancelled at prompt ${i + 1}/${generationCount}`);
								break;
							}

							const progressItem = progressItems?.querySelector(`[data-index="${i}"]`);
							const statusBadge = progressItem?.querySelector(".instaraw-rpg-progress-item-status");
							const progressBar = progressItem?.querySelector(".instaraw-rpg-progress-item-fill");
							const messageDiv = progressItem?.querySelector(".instaraw-rpg-progress-item-message");

							// Update status to in-progress
							if (progressItem) progressItem.classList.add("in-progress");
							if (statusBadge) {
								statusBadge.className = "instaraw-rpg-progress-item-status in-progress";
								statusBadge.textContent = "⚡ Generating...";
							}
							if (progressBar) {
								progressBar.style.width = "100%";
								progressBar.classList.add("animating");
							}

							console.log(`[RPG] 📝 Generating prompt ${i + 1}/${generationCount}`);
							console.log(`[RPG] 🚀 About to make API request...`);

							// Build payload for single prompt generation
							const payload = {
								source_prompts: sourcePrompts.map((p) => ({
									id: p.source_id,
									prompt: {
										positive: p.positive_prompt,
										negative: p.negative_prompt,
									},
								})),
								generation_count: 1, // SEQUENTIAL: Generate 1 at a time
								inspiration_count: inspirationCount,
								is_sdxl: isSDXL,
								character_reference: useCharacter ? characterDescription : "",
								model: model,
								gemini_api_key: geminiApiKey,
								grok_api_key: grokApiKey,
								system_prompt: systemPrompt,
								temperature: temperatureValue,
								top_p: topPValue,
								force_regenerate: forceRegenerate,
								mode: detectedMode,
								affect_elements: affectElements,
								user_input: userInput,
								generation_style: node.properties.generation_style || "reality"
							};

							// Retry logic with exponential backoff
							const maxRetries = 3;
							let retryCount = 0;
							let success = false;
							let promptResult = null;
							let lastError = null;

							while (retryCount <= maxRetries && !success && !cancelRequested) {
								try {
									if (retryCount > 0) {
										if (statusBadge) {
											statusBadge.className = "instaraw-rpg-progress-item-status retrying";
											statusBadge.textContent = `🔄 Retry ${retryCount}/${maxRetries}`;
										}
										if (messageDiv) {
											messageDiv.textContent = `Rate limited, retrying in ${Math.pow(2, retryCount - 1)}s...`;
											messageDiv.className = "instaraw-rpg-progress-item-message";
										}
										console.log(`[RPG] ⏳ Retry ${retryCount}/${maxRetries} - waiting ${Math.pow(2, retryCount - 1)}s`);

										// Exponential backoff: 1s, 2s, 4s, 8s
										await new Promise(resolve => setTimeout(resolve, Math.pow(2, retryCount - 1) * 1000));

										if (cancelRequested) break;
									}

									// Make API request
									const response = await api.fetchApi("/instaraw/generate_creative_prompts", {
										method: "POST",
										headers: { "Content-Type": "application/json" },
										body: JSON.stringify(payload),
										signal: signal
									});

									const result = await parseJSONResponse(response);
									console.log(`[RPG] 🔍 API Response for prompt ${i + 1}:`, {
										status: response.status,
										ok: response.ok,
										resultSuccess: result?.success,
										promptsCount: result?.prompts?.length,
										error: result?.error
									});

									// Check for rate limiting (429 or error message contains "rate limit")
									if (response.status === 429 || (result.error && /rate.*limit/i.test(result.error))) {
										throw new Error("Rate limited");
									}

									if (!response.ok) {
										throw new Error(result?.error || `API error ${response.status}`);
									}

									if (result.success && result.prompts && result.prompts.length > 0) {
										const rawPrompt = result.prompts[0];
										console.log(`[RPG] 📄 Raw response from API:`, rawPrompt);

										// Parse line-based format with prefixes
										if (typeof rawPrompt === 'string') {
											const parseStructuredPrompt = (text) => {
												const lines = text.split('\n').map(l => l.trim()).filter(l => l);
												const parsed = {
													positive: "",
													negative: "",
													tags: [],
													classification: {
														content_type: "other",
														safety_level: "sfw",
														shot_type: "other"
													}
												};

												lines.forEach(line => {
													if (line.startsWith('POSITIVE:')) {
														parsed.positive = line.substring(9).trim();
													} else if (line.startsWith('NEGATIVE:')) {
														parsed.negative = line.substring(9).trim();
													} else if (line.startsWith('CONTENT_TYPE:')) {
														parsed.classification.content_type = line.substring(13).trim().toLowerCase();
													} else if (line.startsWith('SAFETY_LEVEL:')) {
														parsed.classification.safety_level = line.substring(13).trim().toLowerCase();
													} else if (line.startsWith('SHOT_TYPE:')) {
														parsed.classification.shot_type = line.substring(10).trim().toLowerCase();
													} else if (line.startsWith('TAGS:')) {
														const tagsStr = line.substring(5).trim();
														parsed.tags = tagsStr.split(',').map(t => t.trim()).filter(t => t);
													}
												});

												// Fallback: if no POSITIVE field found, use entire text as positive
												if (!parsed.positive && text) {
													parsed.positive = text;
												}

												return parsed;
											};

											promptResult = parseStructuredPrompt(rawPrompt);
											console.log(`[RPG] ✅ Prompt ${i + 1} parsed from structured format`);
										} else {
											// API returned an object (legacy format)
											promptResult = rawPrompt;
											console.log(`[RPG] ✅ Prompt ${i + 1} received as object`);
										}

										console.log(`[RPG] 📦 Prompt ${i + 1} final:`, {
											positiveLength: promptResult.positive?.length || 0,
											positivePreview: promptResult.positive?.slice(0, 80) + "...",
											negativeLength: promptResult.negative?.length || 0,
											tagsCount: promptResult.tags?.length || 0,
											classification: promptResult.classification
										});
										success = true;
									} else {
										console.error(`[RPG] ❌ Invalid result for prompt ${i + 1}:`, result);
										throw new Error(result.error || "No prompts returned");
									}

								} catch (error) {
									lastError = error;

									// Check if it's an abort error
									if (error.name === 'AbortError' || cancelRequested) {
										console.log(`[RPG] ⏹ Request aborted for prompt ${i + 1}`);
										break;
									}

									// Check if it's a rate limit error
									if (error.message.includes("Rate limited") || error.message.includes("429")) {
										console.log(`[RPG] ⚠️ Rate limited on prompt ${i + 1}, attempt ${retryCount + 1}/${maxRetries + 1}`);
										retryCount++;

										if (retryCount > maxRetries) {
											console.error(`[RPG] ❌ Max retries exceeded for prompt ${i + 1}`);
											throw new Error("Rate limit exceeded - max retries reached");
										}
									} else {
										// Other error - don't retry
										console.error(`[RPG] ❌ Error generating prompt ${i + 1}:`, error);
										throw error;
									}
								}
							}

							if (cancelRequested) {
								// Mark as cancelled
								if (progressItem) {
									progressItem.classList.remove("in-progress");
									progressItem.classList.add("error");
								}
								if (statusBadge) {
									statusBadge.className = "instaraw-rpg-progress-item-status error";
									statusBadge.textContent = "⏹ Cancelled";
								}
								if (progressBar) {
									progressBar.classList.remove("animating");
									progressBar.style.width = "0%";
								}
								if (messageDiv) {
									messageDiv.textContent = "Generation cancelled by user";
									messageDiv.className = "instaraw-rpg-progress-item-message";
								}
								break;
							}

							if (success && promptResult) {
								// Success!
								console.log(`[RPG] 💾 Saving prompt ${i + 1} to database:`, {
									positive: promptResult.positive?.slice(0, 50) + '...',
									positiveLength: promptResult.positive?.length || 0,
									negative: promptResult.negative?.slice(0, 50) + '...',
									negativeLength: promptResult.negative?.length || 0,
									tags: promptResult.tags,
									classification: promptResult.classification
								});
								allGeneratedPrompts.push(promptResult);
							// Save to database immediately with full data
							const savedPrompt = await addGeneratedPrompt({
								positive: promptResult.positive,
								negative: promptResult.negative,
								tags: promptResult.tags || [],
								classification: promptResult.classification || { content_type: "other", safety_level: "sfw", shot_type: "other" }
							});
							console.log(`[RPG] ✅ Saved prompt ${i + 1} with ID:`, savedPrompt.id);

								if (progressItem) {
									progressItem.classList.remove("in-progress");
									progressItem.classList.add("success");
								}
								if (statusBadge) {
									statusBadge.className = "instaraw-rpg-progress-item-status success";
									statusBadge.textContent = "✓ Complete";
								}
								if (progressBar) {
									progressBar.classList.remove("animating");
									progressBar.style.width = "100%";
								}
								if (messageDiv) {
									const preview = promptResult.positive?.slice(0, 80) || "No content";
									messageDiv.textContent = preview + (promptResult.positive?.length > 80 ? "..." : "");
									messageDiv.className = "instaraw-rpg-progress-item-message";
								}
							} else {
								// Failed after retries
								if (progressItem) {
									progressItem.classList.remove("in-progress");
									progressItem.classList.add("error");
								}
								if (statusBadge) {
									statusBadge.className = "instaraw-rpg-progress-item-status error";
									statusBadge.textContent = "✖ Failed";
								}
								if (progressBar) {
									progressBar.classList.remove("animating");
									progressBar.style.width = "0%";
								}
								if (messageDiv) {
									messageDiv.textContent = lastError?.message || "Generation failed";
									messageDiv.className = "instaraw-rpg-progress-item-message error";
								}
							}
						}

						// Show results
						if (allGeneratedPrompts.length > 0) {
							console.log(`[RPG] ✅ Generated ${allGeneratedPrompts.length}/${generationCount} prompts successfully`);

							// Update progress header with completion statistics
							const progressHeader = progressSection?.querySelector(".instaraw-rpg-progress-header h4");
							if (progressHeader) {
								const successCount = allGeneratedPrompts.length;
								const failedCount = generationCount - successCount;
								progressHeader.innerHTML = `
									✓ Generation Complete:
									<span style="color: #22c55e">${successCount} succeeded</span>
									${failedCount > 0 ? `<span style="color: #ef4444">, ${failedCount} failed</span>` : ''}
								`;
							}

							// Hide cancel button
							const cancelBtn = progressSection?.querySelector(".instaraw-rpg-cancel-generation-btn");
							if (cancelBtn) cancelBtn.style.display = "none";

							const previewList = container.querySelector(".instaraw-rpg-generate-preview-list");
							if (previewSection && previewList) {
								previewList.innerHTML = allGeneratedPrompts
									.map((p, idx) => `
										<div class="instaraw-rpg-preview-item">
											<strong>#${idx + 1}</strong>
											<p>${escapeHtml(p.positive || "")}</p>
										</div>
									`)
									.join("");

								// Show preview (keep progress visible for stats)
								previewSection.style.display = "block";

								// Store generated prompts temporarily
								node._generatedUnifiedPrompts = allGeneratedPrompts;
								setupEventHandlers();

								// Auto-hide progress section after 3 seconds if all succeeded
								if (allGeneratedPrompts.length === generationCount) {
									setTimeout(() => {
										if (progressSection) progressSection.style.display = "none";
									}, 3000);
								}
							}
						} else if (cancelRequested) {
							console.log("[RPG] ⏹ Generation cancelled - no prompts generated");

							// Update progress header
							const progressHeader = progressSection?.querySelector(".instaraw-rpg-progress-header h4");
							if (progressHeader) {
								progressHeader.innerHTML = '<span style="color: #fbbf24">⏹ Generation Cancelled</span>';
							}

							// Hide cancel button
							const cancelBtn = progressSection?.querySelector(".instaraw-rpg-cancel-generation-btn");
							if (cancelBtn) cancelBtn.style.display = "none";

							// Keep progress section visible to show cancellation status
						} else {
							throw new Error("No prompts were generated successfully");
						}

					} catch (error) {
						console.error("[RPG] ❌ Error during sequential prompt generation:", error);
						console.error("[RPG] Error stack:", error.stack);

						// Hide progress section, show error
						if (progressSection) progressSection.style.display = "none";
						alert(`Generation error: ${error.message || error}`);

					} finally {
						// Re-enable generate button
						if (generateBtn) {
							generateBtn.disabled = false;
							generateBtn.style.opacity = "1";
							const detectedMode = node._linkedAILMode || "txt2img";
							const emoji = detectedMode === 'img2img' ? '🖼️' : '🎨';
							generateBtn.textContent = `${emoji} Generate Prompts`;
						}

						// Clear abort controller
						generationAbortController = null;

						console.log("[RPG] Generate Unified Prompts - END (Sequential)");
						console.groupEnd();
					}
				};

				// === Accept Generated Prompts ===
				const acceptGeneratedPrompts = () => {
					if (!node._generatedUnifiedPrompts) return;

					const promptQueue = parsePromptBatch();
					node._generatedUnifiedPrompts.forEach((p) => {
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
					delete node._generatedUnifiedPrompts;
					renderUI();
				};

				// === Cancel Generated Prompts ===
				const cancelGeneratedPrompts = () => {
					delete node._generatedUnifiedPrompts;
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

					// Random count input - save value
					const randomCountInput = container.querySelector(".instaraw-rpg-random-count-input");
					if (randomCountInput) {
						randomCountInput.onchange = (e) => {
							randomCount = parseInt(e.target.value) || 6;
						};
					}

					// Show random prompts button
					const showRandomBtn = container.querySelector(".instaraw-rpg-show-random-btn");
					if (showRandomBtn) {
						showRandomBtn.onclick = async () => {
							const count = parseInt(randomCountInput?.value) || randomCount;
							const filters = JSON.parse(node.properties.library_filters || "{}");

							// Disable button and show loading state
							showRandomBtn.disabled = true;
							const originalText = showRandomBtn.innerHTML;
							showRandomBtn.innerHTML = '⏳ Selecting...';

							try {
								// OPTIMIZATION: Do random selection on frontend since we already have the database loaded!
								// No need to make API call and re-download/re-filter the database

								// Apply filters using existing filterPrompts function
								const filteredPrompts = filterPrompts(promptsDatabase, filters);

								if (filteredPrompts.length === 0) {
									throw new Error("No prompts match the current filters");
								}

								// Randomly select prompts
								const selectedCount = Math.min(count, filteredPrompts.length);
								const shuffled = [...filteredPrompts].sort(() => Math.random() - 0.5);
								const selected = shuffled.slice(0, selectedCount);

								// Store and display random prompts
								randomPrompts = selected;
								randomCount = count;
								showingRandomPrompts = true;
								currentPage = 0; // Reset pagination
								renderUI();
								console.log(`[RPG] Showing ${selected.length} random prompts (from ${filteredPrompts.length} filtered)`);

							} catch (error) {
								console.error("[RPG] Error selecting random prompts:", error);
								showRandomBtn.innerHTML = `✖ ${error.message}`;
								setTimeout(() => {
									showRandomBtn.innerHTML = originalText;
									showRandomBtn.disabled = false;
								}, 3000);
							}
						};
					}

					// Add all random prompts to batch
					const addAllRandomBtn = container.querySelector(".instaraw-rpg-add-all-random-btn");
					if (addAllRandomBtn) {
						addAllRandomBtn.onclick = () => {
							let added = 0;
							randomPrompts.forEach(promptData => {
								addPromptToBatch(promptData);
								added++;
							});
							console.log(`[RPG] Added ${added} random prompts to batch`);

							// Exit random mode
							showingRandomPrompts = false;
							randomPrompts = [];
							renderUI();
						};
					}

					// Reroll random prompts button
					const rerollRandomBtn = container.querySelector(".instaraw-rpg-reroll-random-btn");
					if (rerollRandomBtn) {
						rerollRandomBtn.onclick = () => {
							const count = randomCount; // Use same count as before
							const filters = JSON.parse(node.properties.library_filters || "{}");

							// Apply filters and randomly select
							const filteredPrompts = filterPrompts(promptsDatabase, filters);

							if (filteredPrompts.length === 0) {
								alert("No prompts match the current filters");
								return;
							}

							// Randomly select prompts (different from before)
							const selectedCount = Math.min(count, filteredPrompts.length);
							const shuffled = [...filteredPrompts].sort(() => Math.random() - 0.5);
							const selected = shuffled.slice(0, selectedCount);

							// Update and re-render
							randomPrompts = selected;
							renderUI();
							console.log(`[RPG] Rerolled ${selected.length} random prompts`);
						};
					}

					// Exit random mode button
					const exitRandomBtn = container.querySelector(".instaraw-rpg-exit-random-btn");
					if (exitRandomBtn) {
						exitRandomBtn.onclick = () => {
							showingRandomPrompts = false;
							randomPrompts = [];
							renderUI();
						};
					}

					// Create custom prompt button
					const createPromptBtn = container.querySelector(".instaraw-rpg-create-prompt-btn");
					if (createPromptBtn) {
						createPromptBtn.onclick = async () => {
							try {
								// Create empty prompt and add to user prompts
								const newPrompt = await addUserPrompt({
									positive: "",
									negative: "",
									tags: [],
									content_type: "person",
									safety_level: "sfw",
									shot_type: "portrait"
								});

								// Immediately put it in edit mode
								editingValues[newPrompt.id] = {
									positive: "",
									negative: "",
									tags: "",
									content_type: "person",
									safety_level: "sfw",
									shot_type: "portrait"
								};
								editingPrompts.add(newPrompt.id);
								renderUI();
								console.log("[RPG] Created new user prompt in edit mode");
							} catch (error) {
								console.error("[RPG] Error creating user prompt:", error);
								alert(`Error creating prompt: ${error.message}`);
							}
						};
					}

					// Import prompts button
					const importPromptsBtn = container.querySelector(".instaraw-rpg-import-prompts-btn");
					if (importPromptsBtn) {
						importPromptsBtn.onclick = () => {
							const input = document.createElement("input");
							input.type = "file";
							input.accept = ".json";
							input.onchange = async (e) => {
								const file = e.target.files[0];
								if (!file) return;

								try {
									const result = await importUserPrompts(file);
									alert(`Successfully imported ${result.added} prompts${result.skipped > 0 ? ` (${result.skipped} duplicates skipped)` : ''}`);
								} catch (error) {
									console.error("[RPG] Error importing prompts:", error);
									alert(`Error importing prompts: ${error.message}`);
								}
							};
							input.click();
						};
					}

					// Export prompts button
					const exportPromptsBtn = container.querySelector(".instaraw-rpg-export-prompts-btn");
					if (exportPromptsBtn) {
						exportPromptsBtn.onclick = () => {
							if (userPrompts.length === 0) {
								alert("No user prompts to export");
								return;
							}
							exportUserPrompts();
						};
					}

					// Delete user prompt buttons
					container.querySelectorAll(".instaraw-rpg-delete-user-prompt-btn").forEach((btn) => {
						btn.onclick = async (e) => {
							e.stopPropagation();
							const promptId = btn.dataset.id;
							const prompt = userPrompts.find(p => p.id === promptId);
							if (!prompt) return;

							const confirmMsg = `Delete this prompt?\n\nPositive: ${(prompt.prompt?.positive || '').substring(0, 100)}...`;
							if (!confirm(confirmMsg)) return;

							try {
								await deleteUserPrompt(promptId);
								console.log(`[RPG] Deleted user prompt ${promptId}`);

								// If in random mode, remove from randomPrompts array
								if (showingRandomPrompts) {
									randomPrompts = randomPrompts.filter(p => p.id !== promptId);
								}

								// Refresh UI
								renderUI();
							} catch (error) {
								console.error("[RPG] Error deleting user prompt:", error);
								alert(`Error deleting prompt: ${error.message}`);
							}
						};
					});

					// Edit button - enter edit mode
					container.querySelectorAll(".instaraw-rpg-edit-user-prompt-btn").forEach((btn) => {
						btn.onclick = () => {
							const promptId = btn.dataset.id;
							const prompt = userPrompts.find(p => p.id === promptId);
							if (prompt) {
								// Store current values in edit buffer
								editingValues[promptId] = {
									positive: prompt.prompt?.positive || "",
									negative: prompt.prompt?.negative || "",
									tags: prompt.tags?.join(", ") || "",
									content_type: prompt.classification?.content_type || "person",
									safety_level: prompt.classification?.safety_level || "sfw",
									shot_type: prompt.classification?.shot_type || "portrait"
								};
								editingPrompts.add(promptId);
								renderUI();
								console.log(`[RPG] Editing user prompt ${promptId}`);
							}
						};
					});

					// Save button - save changes and exit edit mode
					container.querySelectorAll(".instaraw-rpg-save-user-prompt-btn").forEach((btn) => {
						btn.onclick = async () => {
							const promptId = btn.dataset.id;
							const positiveTextarea = container.querySelector(`.instaraw-rpg-user-prompt-edit-positive[data-id="${promptId}"]`);
							const negativeTextarea = container.querySelector(`.instaraw-rpg-user-prompt-edit-negative[data-id="${promptId}"]`);
							const tagsInput = container.querySelector(`.instaraw-rpg-user-prompt-edit-tags[data-id="${promptId}"]`);
							const contentTypeSelect = container.querySelector(`.instaraw-rpg-user-prompt-edit-content-type[data-id="${promptId}"]`);
							const safetyLevelSelect = container.querySelector(`.instaraw-rpg-user-prompt-edit-safety-level[data-id="${promptId}"]`);
							const shotTypeSelect = container.querySelector(`.instaraw-rpg-user-prompt-edit-shot-type[data-id="${promptId}"]`);

							if (!positiveTextarea || !negativeTextarea || !tagsInput) return;

							const tagsArray = tagsInput.value.split(",").map(t => t.trim()).filter(Boolean);

							try {
								await updateUserPrompt(promptId, {
									prompt: {
										positive: positiveTextarea.value,
										negative: negativeTextarea.value
									},
									tags: tagsArray,
									classification: {
										content_type: contentTypeSelect?.value || "person",
										safety_level: safetyLevelSelect?.value || "sfw",
										shot_type: shotTypeSelect?.value || "portrait"
									}
								});
								console.log(`[RPG] Saved user prompt ${promptId}`);

								// Exit edit mode
								editingPrompts.delete(promptId);
								delete editingValues[promptId];
								renderUI();
							} catch (error) {
								console.error("[RPG] Error saving user prompt:", error);
								alert(`Error saving: ${error.message}`);
							}
						};
					});

					// Cancel button - discard changes and exit edit mode
					container.querySelectorAll(".instaraw-rpg-cancel-edit-prompt-btn").forEach((btn) => {
						btn.onclick = () => {
							const promptId = btn.dataset.id;
							editingPrompts.delete(promptId);
							delete editingValues[promptId];
							renderUI();
							console.log(`[RPG] Cancelled editing user prompt ${promptId}`);
						};
					});

					// Auto-resize textareas in edit mode
					container.querySelectorAll(".instaraw-rpg-user-prompt-edit-positive, .instaraw-rpg-user-prompt-edit-negative").forEach((textarea) => {
						autoResizeTextarea(textarea);
						textarea.oninput = () => autoResizeTextarea(textarea);
					});

					// Update editingValues when classification dropdowns change
					container.querySelectorAll(".instaraw-rpg-user-prompt-edit-content-type").forEach((select) => {
						select.onchange = () => {
							const promptId = select.dataset.id;
							if (editingValues[promptId]) {
								editingValues[promptId].content_type = select.value;
							}
						};
					});

					container.querySelectorAll(".instaraw-rpg-user-prompt-edit-safety-level").forEach((select) => {
						select.onchange = () => {
							const promptId = select.dataset.id;
							if (editingValues[promptId]) {
								editingValues[promptId].safety_level = select.value;
							}
						};
					});

					container.querySelectorAll(".instaraw-rpg-user-prompt-edit-shot-type").forEach((select) => {
						select.onchange = () => {
							const promptId = select.dataset.id;
							if (editingValues[promptId]) {
								editingValues[promptId].shot_type = select.value;
							}
						};
					});

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

					// ID copy buttons
					container.querySelectorAll(".instaraw-rpg-id-copy-btn").forEach((btn) => {
						btn.onclick = (e) => {
							e.stopPropagation();
							const promptId = btn.dataset.id;
							navigator.clipboard.writeText(promptId).then(() => {
								// Visual feedback
								const originalText = btn.textContent;
								btn.textContent = "✅";
								setTimeout(() => {
									btn.textContent = originalText;
								}, 1000);
							}).catch(err => {
								console.error("[RPG] Failed to copy ID:", err);
								alert("Failed to copy ID to clipboard");
							});
						};
					});

					// Toggle tags buttons (expand/collapse) - use event delegation
					// Only add listener once to prevent accumulation
					if (!container._hasToggleTagsListener) {
						container._hasToggleTagsListener = true;
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
					}

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

					// Smart Sync AIL button - handles both latent creation and repeat syncing
					const syncAilBtn = container.querySelector(".instaraw-rpg-sync-ail-btn");
					if (syncAilBtn) {
						syncAilBtn.onclick = () => {
							if (!node._linkedAILNodeId) {
								alert("No Advanced Image Loader detected. Connect AIL to RPG first.");
								return;
							}

							const promptQueue = parsePromptBatch();
							const detectedMode = node._linkedAILMode || "img2img";
							const totalGenerations = promptQueue.reduce((sum, p) => sum + (p.repeat_count || 1), 0);

							// Get target dimensions from aspect ratio selector
							const targetDims = getTargetDimensions();

							// Determine what needs syncing
							const linkedLatents = node._linkedLatents || [];
							const linkedImages = node._linkedImages || [];
							const needsLatentSync = detectedMode === "txt2img" && linkedLatents.length !== promptQueue.length;
							const needsRepeatSync = promptQueue.some((p, idx) => {
								const linkedItem = detectedMode === "img2img" ? linkedImages[idx] : linkedLatents[idx];
								return linkedItem && (p.repeat_count || 1) !== (linkedItem.repeat_count || 1);
							});

							let confirmMsg = "";
							if (needsLatentSync && needsRepeatSync) {
								confirmMsg = `This will:\n1. Create ${promptQueue.length} latents (${totalGenerations} total) at ${targetDims.aspect_label}\n2. Sync repeat counts\n\nContinue?`;
							} else if (needsLatentSync) {
								confirmMsg = `Create ${promptQueue.length} latents (${totalGenerations} total) at ${targetDims.aspect_label} in AIL #${node._linkedAILNodeId}?\n\nContinue?`;
							} else if (needsRepeatSync) {
								confirmMsg = `Sync repeat counts to AIL Node #${node._linkedAILNodeId}?\n\nContinue?`;
							} else {
								confirmMsg = `Everything is already synced! Sync anyway?\n\nContinue?`;
							}

							if (!confirm(confirmMsg)) return;

							// 1. Create/sync latents if needed (txt2img mode)
							if (needsLatentSync || detectedMode === "txt2img") {
								const latentSpecs = promptQueue.map(p => ({
									repeat_count: p.repeat_count || 1
								}));

								window.dispatchEvent(new CustomEvent("INSTARAW_SYNC_AIL_LATENTS", {
									detail: {
										targetNodeId: node._linkedAILNodeId,
										latentSpecs: latentSpecs,
										dimensions: targetDims
									}
								}));
								console.log(`[RPG] Synced ${promptQueue.length} latents to AIL`);
							}

							// 2. Sync repeat counts (both modes)
							window.dispatchEvent(new CustomEvent("INSTARAW_SYNC_AIL_REPEATS", {
								detail: {
									targetNodeId: node._linkedAILNodeId,
									mode: detectedMode,
									repeats: promptQueue.map(p => p.repeat_count || 1)
								}
							}));
							console.log(`[RPG] Synced repeat counts to AIL`);
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

					// ========================================
					// === UNIFIED GENERATE TAB HANDLERS ===
					// ========================================

					// Character description generation button
					const generateCharacterDescBtn = container.querySelector(".instaraw-rpg-generate-character-desc-btn");
					if (generateCharacterDescBtn) {
						generateCharacterDescBtn.onclick = generateCharacterDescription;
					}

					// Character likeness checkbox
					const enableCharacterCheckbox = container.querySelector(".instaraw-rpg-enable-character-checkbox");
					if (enableCharacterCheckbox) {
						enableCharacterCheckbox.onchange = (e) => {
							node.properties.use_character_likeness = e.target.checked;
							renderUI(); // Re-render to show/hide character section
						};
					}

					// Generation mode toggle buttons (Reality vs Creative)
					container.querySelectorAll(".instaraw-rpg-mode-toggle-btn").forEach((btn) => {
						btn.onclick = () => {
							const mode = btn.dataset.mode;
							node.properties.generation_style = mode;
							renderUI(); // Re-render to update button styles
							console.log(`[RPG] Switched to ${mode} mode`);
						};
					});

					// Character text input (save on change + auto-resize)
					const characterTextInput = container.querySelector(".instaraw-rpg-character-text-input");
					if (characterTextInput) {
						autoResizeTextarea(characterTextInput);
						characterTextInput.oninput = (e) => autoResizeTextarea(characterTextInput);
						characterTextInput.onchange = (e) => {
							node.properties.character_text_input = e.target.value;
							app.graph.setDirtyCanvas(true, true);
						};
					}

					// Character complexity dropdown
					const characterComplexitySelect = container.querySelector(".instaraw-rpg-character-complexity");
					if (characterComplexitySelect) {
						characterComplexitySelect.onchange = (e) => {
							node.properties.character_complexity = e.target.value;

							// If no custom system prompt, update the textarea to show new default
							const systemPromptTextarea = container.querySelector(".instaraw-rpg-character-system-prompt");
							if (systemPromptTextarea && !node.properties.character_system_prompt) {
								systemPromptTextarea.value = getCharacterSystemPrompt(e.target.value);
								autoResizeTextarea(systemPromptTextarea);
							}

							app.graph.setDirtyCanvas(true, true);
						};
					}

					// Character system prompt textarea (advanced)
					const characterSystemPromptInput = container.querySelector(".instaraw-rpg-character-system-prompt");
					if (characterSystemPromptInput) {
						autoResizeTextarea(characterSystemPromptInput);
						characterSystemPromptInput.oninput = (e) => autoResizeTextarea(characterSystemPromptInput);
						characterSystemPromptInput.onchange = (e) => {
							node.properties.character_system_prompt = e.target.value;
							app.graph.setDirtyCanvas(true, true);
						};
					}

					// Reset system prompt button
					const resetSystemPromptBtn = container.querySelector(".instaraw-rpg-reset-system-prompt-btn");
					if (resetSystemPromptBtn) {
						resetSystemPromptBtn.onclick = () => {
							const complexity = node.properties.character_complexity || "balanced";
							const defaultPrompt = getCharacterSystemPrompt(complexity);

							// Clear custom prompt and update textarea
							node.properties.character_system_prompt = "";
							if (characterSystemPromptInput) {
								characterSystemPromptInput.value = defaultPrompt;
								autoResizeTextarea(characterSystemPromptInput);
							}

							app.graph.setDirtyCanvas(true, true);
						};
					}

					// User text input (txt2img mode)
					const userTextInput = container.querySelector(".instaraw-rpg-user-text-input");
					if (userTextInput) {
						userTextInput.onchange = (e) => {
							node.properties.user_text_input = e.target.value;
							app.graph.setDirtyCanvas(true, true);
						};
					}

					// Inspiration count input (txt2img mode)
					const inspirationCountInput = container.querySelector(".instaraw-rpg-inspiration-count");
					if (inspirationCountInput) {
						inspirationCountInput.onchange = (e) => {
							node.properties.inspiration_count = parseInt(e.target.value) || 3;
							app.graph.setDirtyCanvas(true, true);
						};
					}

					// Generation count input (unified tab)
					const genCountInputUnified = container.querySelector(".instaraw-rpg-gen-count-input");
					if (genCountInputUnified) {
						genCountInputUnified.onchange = (e) => {
							node.properties.generation_count = parseInt(e.target.value) || 5;
							app.graph.setDirtyCanvas(true, true);
						};
					}

					// SDXL checkbox (unified tab)
					const isSDXLCheckboxUnified = container.querySelector(".instaraw-rpg-is-sdxl-checkbox");
					if (isSDXLCheckboxUnified) {
						isSDXLCheckboxUnified.onchange = (e) => {
							node.properties.is_sdxl = e.target.checked;
							app.graph.setDirtyCanvas(true, true);
						};
					}

					// Unified Generate button (MAIN HANDLER)
					const generateUnifiedBtn = container.querySelector(".instaraw-rpg-generate-unified-btn");
					if (generateUnifiedBtn) {
						generateUnifiedBtn.onclick = generateUnifiedPrompts;
					}

					// Cancel generation button (for aborting in-progress generation)
					const cancelGenerationBtn = container.querySelector(".instaraw-rpg-cancel-generation-btn");
					if (cancelGenerationBtn) {
						cancelGenerationBtn.onclick = () => {
							if (generationAbortController) {
								console.log("[RPG] 🛑 User clicked cancel - aborting generation");
								generationAbortController.abort();
							}
						};
					}

					// Accept generated prompts button
					const acceptGeneratedBtn = container.querySelector(".instaraw-rpg-accept-generated-btn");
					if (acceptGeneratedBtn) {
						acceptGeneratedBtn.onclick = acceptGeneratedPrompts;
					}

					// Cancel generated prompts button
					const cancelGeneratedBtn = container.querySelector(".instaraw-rpg-cancel-generated-btn");
					if (cancelGeneratedBtn) {
						cancelGeneratedBtn.onclick = cancelGeneratedPrompts;
					}
				};

				// === Drag and Drop (Exact AIL Pattern) ===
				const setupDragAndDrop = () => {
					// Drag-and-drop reordering (only when enabled)
					if (reorderModeEnabled) {
						const items = container.querySelectorAll(".instaraw-rpg-batch-item");
						let draggedItem = null;

						items.forEach((item) => {
						// Skip if already has drag listeners to prevent accumulation
						if (item._hasDragListeners) return;
						item._hasDragListeners = true;

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
				// Only add once to prevent accumulation
				if (!window._hasRPGAILListener) {
					window._hasRPGAILListener = true;
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
				}

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

				// Add widget change callbacks to automatically refresh UI when aspect ratio changes
				const setupWidgetCallbacks = () => {
					const widthWidget = node.widgets?.find((w) => w.name === "output_width");
					if (widthWidget && !widthWidget._instaraw_callback_added) {
						const originalCallback = widthWidget.callback;
						widthWidget.callback = function() {
							if (originalCallback) originalCallback.apply(this, arguments);
							renderUI();
						};
						widthWidget._instaraw_callback_added = true;
					}

					const heightWidget = node.widgets?.find((w) => w.name === "output_height");
					if (heightWidget && !heightWidget._instaraw_callback_added) {
						const originalCallback = heightWidget.callback;
						heightWidget.callback = function() {
							if (originalCallback) originalCallback.apply(this, arguments);
							renderUI();
						};
						heightWidget._instaraw_callback_added = true;
					}

					const aspectWidget = node.widgets?.find((w) => w.name === "aspect_label");
					if (aspectWidget && !aspectWidget._instaraw_callback_added) {
						const originalCallback = aspectWidget.callback;
						aspectWidget.callback = function() {
							if (originalCallback) originalCallback.apply(this, arguments);
							renderUI();
						};
						aspectWidget._instaraw_callback_added = true;
					}
				};

				// Periodic dimension check - checks every 2 seconds if dimensions changed
				let dimensionCheckInterval = null;
				let lastDimensions = null;
				const startDimensionCheck = () => {
					if (dimensionCheckInterval) clearInterval(dimensionCheckInterval);
					dimensionCheckInterval = setInterval(() => {
						const currentDims = getTargetDimensions();
						const dimsKey = `${currentDims.width}x${currentDims.height}:${currentDims.aspect_label}`;
						if (lastDimensions !== null && lastDimensions !== dimsKey) {
							console.log(`[RPG] Dimensions changed: ${lastDimensions} -> ${dimsKey}`);
							renderUI();
						}
						lastDimensions = dimsKey;
					}, 2000);
				};

				// Store references for lifecycle hooks
				node._updateCachedHeight = updateCachedHeight;
				node._renderUI = renderUI;
				node._setupWidgetCallbacks = setupWidgetCallbacks;

				// Initial setup
				setTimeout(() => {
					syncPromptBatchWidget();
					setupWidgetCallbacks();
					startDimensionCheck();
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
					if (this._setupWidgetCallbacks) this._setupWidgetCallbacks();
					if (this._renderUI) this._renderUI();
				}, 200);
			};

			// === onConnectionsChange Hook (for character_image reactivity) ===
			const onConnectionsChange = nodeType.prototype.onConnectionsChange;
			nodeType.prototype.onConnectionsChange = function(side, slot, connect, link_info, output) {
				const result = onConnectionsChange?.apply(this, arguments);
				// Re-render when character_image connection changes
				if (this._renderUI) {
					setTimeout(() => this._renderUI(), 50);
				}
				return result;
			};
		}
	},
});