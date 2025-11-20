// ---
// Filename: ../ComfyUI_INSTARAW/js/advanced_image_loader.js
// ---

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

app.registerExtension({
	name: "Comfy.INSTARAW.AdvancedImageLoader",

	async beforeRegisterNodeDef(nodeType, nodeData, app) {
		if (nodeData.name === "INSTARAW_AdvancedImageLoader") {
			const onNodeCreated = nodeType.prototype.onNodeCreated;
			nodeType.prototype.onNodeCreated = function () {
				onNodeCreated?.apply(this, arguments);

				if (!this.properties.batch_data) {
					this.properties.batch_data = JSON.stringify({ images: [], order: [], total_count: 0 });
				}
				if (!this.properties.txt2img_data_backup) {
					this.properties.txt2img_data_backup = JSON.stringify({ latents: [], order: [], total_count: 0 });
				}
				if (!this.properties.img2img_data_backup) {
					this.properties.img2img_data_backup = JSON.stringify({ images: [], order: [], total_count: 0 });
				}

				const node = this;
				let cachedHeight = 300;
				let isUpdatingHeight = false;
				let currentDetectedMode = null; // Track detected mode to minimize re-renders
				let modeCheckInterval = null; // For periodic mode checking

				const container = document.createElement("div");
				container.className = "instaraw-adv-loader-container";
				container.style.width = "100%";
				container.style.boxSizing = "border-box";
				container.style.overflow = "hidden";
				container.style.height = `${cachedHeight}px`;

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

				const syncBatchDataWidget = () => {
					const batchDataWidget = node.widgets?.find((w) => w.name === "batch_data");
					if (batchDataWidget) {
						batchDataWidget.value = node.properties.batch_data;
					} else {
						const widget = node.addWidget("text", "batch_data", node.properties.batch_data, () => {}, { serialize: true });
						widget.hidden = true;
					}
				};

				/**
				 * Computes aspect ratio node outputs locally (for multi-output nodes).
				 * Aspect ratio nodes compute width/height in Python, but we replicate that logic here
				 * to properly read values from the correct output slot.
				 */
				const getAspectRatioOutput = (aspectRatioNode, slotIndex) => {
					// Read the dropdown selection from the aspect ratio node
					const selection = aspectRatioNode.widgets?.[0]?.value;
					if (!selection) {
						console.warn(`[INSTARAW AIL ${node.id}] Aspect ratio node has no selection`);
						return null;
					}

					// Aspect ratio mappings (must match Python ASPECT_RATIOS dicts exactly)
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

					const ratios = aspectRatioNode.type === "INSTARAW_WANAspectRatio"
						? WAN_RATIOS
						: SDXL_RATIOS;

					const config = ratios[selection];
					if (!config) {
						console.warn(`[INSTARAW AIL ${node.id}] Unknown aspect ratio selection: ${selection}`);
						return null;
					}

					console.log(`[INSTARAW AIL ${node.id}] Aspect ratio node output:`, {
						selection,
						slotIndex,
						type: aspectRatioNode.type,
						value: slotIndex === 0 ? config.width : slotIndex === 1 ? config.height : config.label
					});

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
					if (!node.inputs || node.inputs.length === 0) {
						const widget = node.widgets?.find(w => w.name === inputName);
						return widget ? widget.value : defaultValue;
					}

					const input = node.inputs.find(i => i.name === inputName);
					if (!input || input.link == null) {
						const widget = node.widgets?.find(w => w.name === inputName);
						return widget ? widget.value : defaultValue;
					}

					const link = app.graph.links[input.link];
					if (!link) return defaultValue;

					const originNode = app.graph.getNodeById(link.origin_id);
					if (!originNode) return defaultValue;

					// SPECIAL HANDLING: For aspect ratio nodes, compute the output locally
					// because they have multiple outputs and don't store computed values in widgets
					if (originNode.type === "INSTARAW_WANAspectRatio" ||
					    originNode.type === "INSTARAW_SDXLAspectRatio") {
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
				};

				/**
				 * Detects if we're in txt2img mode by reading enable_img2img from connected nodes.
				 * Returns true for txt2img mode, false for img2img mode.
				 */
				const isTxt2ImgMode = () => {
					const enableImg2Img = getFinalInputValue("enable_img2img", true);
					console.log(`[INSTARAW AIL ${node.id}] enable_img2img value:`, enableImg2Img, `(type: ${typeof enableImg2Img})`);
					const result = enableImg2Img === false || enableImg2Img === "false";
					console.log(`[INSTARAW AIL ${node.id}] isTxt2ImgMode result:`, result);
					return result;
				};

				/**
				 * Gets width, height, and aspect_label from connected nodes for txt2img mode.
				 */
				const getTxt2ImgDimensions = () => {
					console.log(`[INSTARAW AIL ${node.id}] === Reading dimensions from connected nodes ===`);

					const widthRaw = getFinalInputValue("width", 960);
					const heightRaw = getFinalInputValue("height", 960);
					const aspect_label_raw = getFinalInputValue("aspect_label", null);

					console.log(`[INSTARAW AIL ${node.id}] Raw values:`, { widthRaw, heightRaw, aspect_label_raw });

					const width = parseInt(widthRaw) || 960;
					const height = parseInt(heightRaw) || 960;
					const aspect_label = aspect_label_raw || getAspectLabel(width, height);

					console.log(`[INSTARAW AIL ${node.id}] Final dimensions:`, { width, height, aspect_label });
					console.log(`[INSTARAW AIL ${node.id}] Expected tensor size: ${width}×${height} (${(width * height / 1000000).toFixed(2)}MP)`);

					return { width, height, aspect_label };
				};

				/**
				 * Generates a simple UUID v4.
				 */
				const generateUUID = () => {
					return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
						const r = Math.random() * 16 | 0;
						const v = c === 'x' ? r : (r & 0x3 | 0x8);
						return v.toString(16);
					});
				};

				/**
				 * Calculates aspect ratio label from width and height.
				 */
				const getAspectLabel = (width, height) => {
					const gcd = (a, b) => b === 0 ? a : gcd(b, a % b);
					const divisor = gcd(width, height);
					const w = width / divisor;
					const h = height / divisor;

					// Common aspect ratios
					if (w === h) return "1:1";
					if (w === 3 && h === 4) return "3:4";
					if (w === 4 && h === 3) return "4:3";
					if (w === 9 && h === 16) return "9:16";
					if (w === 16 && h === 9) return "16:9";
					if (w === 2 && h === 3) return "2:3";
					if (w === 3 && h === 2) return "3:2";

					return `${w}:${h}`;
				};

				const renderGallery = () => {
					const detectedMode = isTxt2ImgMode();
					console.log(`[INSTARAW AIL ${node.id}] renderGallery - currentDetectedMode:`, currentDetectedMode, `detectedMode:`, detectedMode);

					// Handle mode switching
					if (currentDetectedMode !== null && currentDetectedMode !== detectedMode) {
						console.log(`[INSTARAW AIL ${node.id}] MODE SWITCH DETECTED! Switching to ${detectedMode ? 'txt2img' : 'img2img'}`);
						if (detectedMode) {
							// Switching to txt2img mode
							node.properties.img2img_data_backup = node.properties.batch_data;
							node.properties.batch_data = node.properties.txt2img_data_backup || JSON.stringify({ latents: [], order: [], total_count: 0 });
						} else {
							// Switching to img2img mode
							node.properties.txt2img_data_backup = node.properties.batch_data;
							node.properties.batch_data = node.properties.img2img_data_backup || JSON.stringify({ images: [], order: [], total_count: 0 });
						}
						syncBatchDataWidget();
					}
					currentDetectedMode = detectedMode;

					console.log(`[INSTARAW AIL ${node.id}] Rendering ${detectedMode ? 'txt2img' : 'img2img'} gallery`);
					if (detectedMode) {
						renderTxt2ImgGallery();
					} else {
						renderImg2ImgGallery();
					}
				};

				const renderImg2ImgGallery = () => {
					const batchData = JSON.parse(node.properties.batch_data || "{}");
					const images = batchData.images || [];
					const order = batchData.order || [];
					const modeWidget = node.widgets?.find((w) => w.name === "mode");
					const currentMode = modeWidget?.value || "Batch Tensor";
					const batchIndexWidget = node.widgets?.find((w) => w.name === "batch_index");
					const currentIndex = node._processingIndex !== undefined ? node._processingIndex : batchIndexWidget?.value || 0;

					container.innerHTML = `
                        <div class="instaraw-adv-loader-mode-indicator">
                            <span class="instaraw-adv-loader-mode-badge instaraw-adv-loader-mode-img2img">🖼️ IMG2IMG MODE</span>
                        </div>
                        <div class="instaraw-adv-loader-mode-selector">
                            <label>Mode:</label>
                            <select class="instaraw-adv-loader-mode-dropdown">
                                <option value="Batch Tensor" ${currentMode === "Batch Tensor" ? "selected" : ""}>🎯 Batch Tensor</option>
                                <option value="Sequential" ${currentMode === "Sequential" ? "selected" : ""}>📑 Sequential</option>
                            </select>
                            ${currentMode === "Sequential" ? `<button class="instaraw-adv-loader-queue-all-btn" title="Queue all images">🎬 Queue All</button>` : ""}
                        </div>
                        ${currentMode === "Sequential" ? `<div class="instaraw-adv-loader-progress"><span class="instaraw-adv-loader-progress-label">Current Index:</span><span class="instaraw-adv-loader-progress-value">${currentIndex} / ${batchData.total_count || 0}</span></div>` : ""}
                        <div class="instaraw-adv-loader-header">
                            <div class="instaraw-adv-loader-actions">
                                <button class="instaraw-adv-loader-upload-btn" title="Upload images">📁 Upload Images</button>
                                ${images.length > 0 ? `<button class="instaraw-adv-loader-delete-all-btn" title="Delete all images">🗑️ Delete All</button>` : ""}
                            </div>
                            <div class="instaraw-adv-loader-stats">
                                <span class="instaraw-adv-loader-count">${images.length} image${images.length !== 1 ? "s" : ""}</span>
                                <span class="instaraw-adv-loader-total">Total: ${batchData.total_count || 0} (with repeats)</span>
                            </div>
                        </div>
                        <div class="instaraw-adv-loader-gallery">
                            ${order.length === 0 ? `<div class="instaraw-adv-loader-empty"><p>No images loaded</p><p class="instaraw-adv-loader-hint">Click "Upload Images" to get started</p></div>` : (() => {
								let currentIdx = 0;
								return order.map((imgId) => {
									const img = images.find((i) => i.id === imgId);
									if (!img) return "";
									const thumbUrl = `/view?filename=${img.thumbnail}&type=input&subfolder=INSTARAW_BatchUploads/${node.id}`;
									const repeatCount = img.repeat_count || 1;
									const startIdx = currentIdx;
									const endIdx = currentIdx + repeatCount - 1;
									currentIdx += repeatCount;
									const isActive = currentMode === "Sequential" && currentIndex >= startIdx && currentIndex <= endIdx;
									const isPast = currentMode === "Sequential" && currentIndex > endIdx;
									const isProcessing = node._isProcessing && isActive;
									return `<div class="instaraw-adv-loader-item ${isActive ? "instaraw-adv-loader-item-active" : ""} ${isPast ? "instaraw-adv-loader-item-done" : ""} ${isProcessing ? "instaraw-adv-loader-item-processing" : ""}" data-id="${img.id}" draggable="true">
                                        ${currentMode === "Sequential" ? `<div class="instaraw-adv-loader-index-badge">${repeatCount === 1 ? `#${startIdx}` : `#${startIdx}-${endIdx}`}</div>` : ""}
                                        <div class="instaraw-adv-loader-thumb">
                                            <img src="${thumbUrl}" alt="${img.original_name}" />
                                            ${isProcessing ? '<div class="instaraw-adv-loader-processing-indicator">⚡ PROCESSING...</div>' : isActive ? '<div class="instaraw-adv-loader-active-indicator">▶ NEXT</div>' : ""}
                                            ${isPast ? '<div class="instaraw-adv-loader-done-indicator">✓</div>' : ""}
                                        </div>
                                        <div class="instaraw-adv-loader-controls">
                                            <label>×</label>
                                            <input type="number" class="instaraw-adv-loader-repeat-input" value="${img.repeat_count || 1}" min="1" max="99" data-id="${img.id}" />
                                            <button class="instaraw-adv-loader-delete-btn" data-id="${img.id}" title="Delete">×</button>
                                        </div>
                                        <div class="instaraw-adv-loader-info">
                                            <div class="instaraw-adv-loader-filename" title="${img.original_name}">${img.original_name}</div>
                                            <div class="instaraw-adv-loader-dimensions">${img.width}×${img.height}</div>
                                        </div>
                                    </div>`;
								}).join("");
							})()}
                        </div>`;
					setupEventHandlers();
					setupDragAndDrop();
					setupFileDropZone();
					updateCachedHeight();

					// Dispatch update event for other nodes (e.g., RPG)
					window.dispatchEvent(new CustomEvent("INSTARAW_AIL_UPDATED", {
						detail: {
							nodeId: node.id,
							mode: "img2img",
							enable_img2img: true,  // NEW: explicit boolean for easier detection
							images: order.map(imgId => {
								const img = images.find(i => i.id === imgId);
								if (!img) return null;
								const thumbUrl = `/view?filename=${img.thumbnail}&type=input&subfolder=INSTARAW_BatchUploads/${node.id}`;
								const width = img.width || 1024;
								const height = img.height || 1024;
								return {
									url: thumbUrl,
									index: order.indexOf(imgId),
									id: imgId,
									repeat_count: img.repeat_count || 1,
									width: width,
									height: height,
									aspect_label: getAspectLabel(width, height)
								};
							}).filter(i => i !== null),
							total: batchData.total_count || 0
						}
					}));
				};

				const renderTxt2ImgGallery = () => {
					// Preserve batch count input value
					const existingInput = container.querySelector(".instaraw-adv-loader-batch-count-input");
					const preservedBatchCount = existingInput?.value || node._batchAddCount || 5;
					node._batchAddCount = preservedBatchCount;

					const batchData = JSON.parse(node.properties.batch_data || "{}");
					const latents = batchData.latents || [];
					const order = batchData.order || [];
					const modeWidget = node.widgets?.find((w) => w.name === "mode");
					const currentMode = modeWidget?.value || "Batch Tensor";
					const batchIndexWidget = node.widgets?.find((w) => w.name === "batch_index");
					const currentIndex = node._processingIndex !== undefined ? node._processingIndex : batchIndexWidget?.value || 0;

					// Get current dimensions from aspect ratio selector (for live updates)
					const dimensions = getTxt2ImgDimensions();

					// Debug: Log dimensions for debugging
					console.log(`[INSTARAW AIL ${node.id}] renderTxt2ImgGallery - dimensions:`, dimensions);

					container.innerHTML = `
                        <div class="instaraw-adv-loader-mode-indicator">
                            <span class="instaraw-adv-loader-mode-badge instaraw-adv-loader-mode-txt2img">🎨 TXT2IMG MODE</span>
                        </div>
                        <div class="instaraw-adv-loader-mode-selector">
                            <label>Mode:</label>
                            <select class="instaraw-adv-loader-mode-dropdown">
                                <option value="Batch Tensor" ${currentMode === "Batch Tensor" ? "selected" : ""}>🎯 Batch Tensor</option>
                                <option value="Sequential" ${currentMode === "Sequential" ? "selected" : ""}>📑 Sequential</option>
                            </select>
                            ${currentMode === "Sequential" ? `<button class="instaraw-adv-loader-queue-all-btn" title="Queue all latents">🎬 Queue All</button>` : ""}
                        </div>
                        ${currentMode === "Sequential" ? `<div class="instaraw-adv-loader-progress"><span class="instaraw-adv-loader-progress-label">Current Index:</span><span class="instaraw-adv-loader-progress-value">${currentIndex} / ${batchData.total_count || 0}</span></div>` : ""}
                        <div class="instaraw-adv-loader-header">
                            <div class="instaraw-adv-loader-actions">
                                <button class="instaraw-adv-loader-add-latent-btn" title="Add empty latent">🖼️ Add Empty Latent</button>
                                <div class="instaraw-adv-loader-batch-add-controls">
                                    <input type="number" class="instaraw-adv-loader-batch-count-input" value="${preservedBatchCount}" min="1" max="100" />
                                    <button class="instaraw-adv-loader-batch-add-btn" title="Batch add empty latents">📦 Add N</button>
                                </div>
                                ${latents.length > 0 ? `<button class="instaraw-adv-loader-delete-all-btn" title="Delete all latents">🗑️ Delete All</button>` : ""}
                            </div>
                            <div class="instaraw-adv-loader-stats">
                                <span class="instaraw-adv-loader-count">${latents.length} latent${latents.length !== 1 ? "s" : ""}</span>
                                <span class="instaraw-adv-loader-total">Total: ${batchData.total_count || 0} (with repeats)</span>
                            </div>
                        </div>
                        <div class="instaraw-adv-loader-gallery">
                            ${order.length === 0 ? `<div class="instaraw-adv-loader-empty"><p>No latents added</p><p class="instaraw-adv-loader-hint">Click "Add Empty Latent" to get started (txt2img mode)</p></div>` : (() => {
								let currentIdx = 0;
								return order.map((latentId) => {
									const latent = latents.find((l) => l.id === latentId);
									if (!latent) return "";

									// Debug: Log what latent contains
									console.log(`[INSTARAW AIL ${node.id}] Rendering latent:`, latent);

									const repeatCount = latent.repeat_count || 1;
									const startIdx = currentIdx;
									const endIdx = currentIdx + repeatCount - 1;
									currentIdx += repeatCount;
									const isActive = currentMode === "Sequential" && currentIndex >= startIdx && currentIndex <= endIdx;
									const isPast = currentMode === "Sequential" && currentIndex > endIdx;
									const isProcessing = node._isProcessing && isActive;

									// Use CURRENT dimensions from aspect ratio selector (not stored dimensions)
									// This ensures live updates when aspect ratio selector changes
									const width = dimensions.width;
									const height = dimensions.height;
									const aspectRatio = width / height;
									const aspectLabel = dimensions.aspect_label;

									return `<div class="instaraw-adv-loader-item ${isActive ? "instaraw-adv-loader-item-active" : ""} ${isPast ? "instaraw-adv-loader-item-done" : ""} ${isProcessing ? "instaraw-adv-loader-item-processing" : ""}" data-id="${latent.id}" draggable="true">
                                        ${currentMode === "Sequential" ? `<div class="instaraw-adv-loader-index-badge">${repeatCount === 1 ? `#${startIdx}` : `#${startIdx}-${endIdx}`}</div>` : ""}
                                        <div class="instaraw-adv-loader-latent-thumb">
                                            <div class="instaraw-adv-loader-aspect-preview" style="aspect-ratio: ${aspectRatio};">
                                                <div class="instaraw-adv-loader-aspect-content">
                                                    <div style="font-size: 24px;">📐</div>
                                                    <div style="font-size: 11px; font-weight: 600;">${aspectLabel}</div>
                                                </div>
                                            </div>
                                            ${isProcessing ? '<div class="instaraw-adv-loader-processing-indicator">⚡ PROCESSING...</div>' : isActive ? '<div class="instaraw-adv-loader-active-indicator">▶ NEXT</div>' : ""}
                                            ${isPast ? '<div class="instaraw-adv-loader-done-indicator">✓</div>' : ""}
                                        </div>
                                        <div class="instaraw-adv-loader-controls">
                                            <label>×</label>
                                            <input type="number" class="instaraw-adv-loader-repeat-input" value="${latent.repeat_count || 1}" min="1" max="99" data-id="${latent.id}" />
                                            <button class="instaraw-adv-loader-delete-btn" data-id="${latent.id}" title="Delete">×</button>
                                        </div>
                                        <div class="instaraw-adv-loader-info">
                                            <div class="instaraw-adv-loader-filename" title="${latent.id}">Latent ${latent.id.substring(0, 8)}</div>
                                            <div class="instaraw-adv-loader-dimensions">${width}×${height}</div>
                                        </div>
                                    </div>`;
								}).join("");
							})()}
                        </div>`;
					setupTxt2ImgEventHandlers();
					setupDragAndDrop();
					setupFileDropZone();
					updateCachedHeight();

					// Dispatch update event for other nodes (e.g., RPG)
					window.dispatchEvent(new CustomEvent("INSTARAW_AIL_UPDATED", {
						detail: {
							nodeId: node.id,
							mode: "txt2img",
							enable_img2img: false,  // NEW: explicit boolean for easier detection
							latents: order.map(latentId => {
								const latent = latents.find(l => l.id === latentId);
								if (!latent) return null;
								return {
									id: latent.id,
									width: latent.width,
									height: latent.height,
									aspect_label: latent.aspect_label,
									repeat_count: latent.repeat_count || 1,
									index: order.indexOf(latentId)
								};
							}).filter(l => l !== null),
							total: batchData.total_count || 0
						}
					}));
				};

				const addEmptyLatent = () => {
					const dimensions = getTxt2ImgDimensions();
					const batchData = JSON.parse(node.properties.batch_data || "{}");
					batchData.latents = batchData.latents || [];
					batchData.order = batchData.order || [];

					// Check for aspect ratio mismatch
					if (batchData.latents.length > 0) {
						const existingAspect = batchData.latents[0].aspect_label;
						if (existingAspect !== dimensions.aspect_label) {
							const confirmed = confirm(
								`Current batch has ${existingAspect} latents.\n` +
								`Switch to ${dimensions.aspect_label}?\n\n` +
								`This will clear all existing latents.`
							);
							if (!confirmed) {
								return; // Abort
							}
							// Clear existing latents
							batchData.latents = [];
							batchData.order = [];
						}
					}

					const newLatent = {
						id: generateUUID(),
						width: dimensions.width,
						height: dimensions.height,
						repeat_count: 1,
						aspect_label: dimensions.aspect_label  // Use aspect_label from connected node
					};

					console.log(`[INSTARAW AIL ${node.id}] Adding latent:`, newLatent);

					batchData.latents.push(newLatent);
					batchData.order.push(newLatent.id);
					batchData.total_count = batchData.latents.reduce((sum, l) => sum + (l.repeat_count || 1), 0);

					node.properties.batch_data = JSON.stringify(batchData);
					syncBatchDataWidget();
					renderGallery();
				};

				const batchAddLatents = (count) => {
					const dimensions = getTxt2ImgDimensions();
					const batchData = JSON.parse(node.properties.batch_data || "{}");
					batchData.latents = batchData.latents || [];
					batchData.order = batchData.order || [];

					// Check for aspect ratio mismatch
					if (batchData.latents.length > 0) {
						const existingAspect = batchData.latents[0].aspect_label;
						if (existingAspect !== dimensions.aspect_label) {
							const confirmed = confirm(
								`Current batch has ${existingAspect} latents.\n` +
								`Switch to ${dimensions.aspect_label}?\n\n` +
								`This will clear all existing latents.`
							);
							if (!confirmed) {
								return; // Abort
							}
							// Clear existing latents
							batchData.latents = [];
							batchData.order = [];
						}
					}

					console.log(`[INSTARAW AIL ${node.id}] Batch adding ${count} latents with dimensions:`, dimensions);

					for (let i = 0; i < count; i++) {
						const newLatent = {
							id: generateUUID(),
							width: dimensions.width,
							height: dimensions.height,
							repeat_count: 1,
							aspect_label: dimensions.aspect_label  // Use aspect_label from connected node
						};
						batchData.latents.push(newLatent);
						batchData.order.push(newLatent.id);
					}

					batchData.total_count = batchData.latents.reduce((sum, l) => sum + (l.repeat_count || 1), 0);
					node.properties.batch_data = JSON.stringify(batchData);
					syncBatchDataWidget();
					renderGallery();
				};

				const deleteLatent = (latentId) => {
					if (!confirm("Delete this latent?")) return;
					const batchData = JSON.parse(node.properties.batch_data || "{}");
					batchData.latents = batchData.latents.filter((l) => l.id !== latentId);
					batchData.order = batchData.order.filter((id) => id !== latentId);
					batchData.total_count = batchData.latents.reduce((sum, l) => sum + (l.repeat_count || 1), 0);
					node.properties.batch_data = JSON.stringify(batchData);
					syncBatchDataWidget();
					renderGallery();
				};

				const deleteAllLatents = () => {
					const batchData = JSON.parse(node.properties.batch_data || "{}");
					const latentCount = batchData.latents?.length || 0;
					if (latentCount === 0 || !confirm(`Delete all ${latentCount} latent${latentCount !== 1 ? "s" : ""}?`)) return;
					node.properties.batch_data = JSON.stringify({ latents: [], order: [], total_count: 0 });
					syncBatchDataWidget();
					renderGallery();
				};

				const updateLatentRepeatCount = (latentId, newCount) => {
					const batchData = JSON.parse(node.properties.batch_data || "{}");
					const latent = batchData.latents.find((l) => l.id === latentId);
					if (latent) {
						latent.repeat_count = Math.max(1, Math.min(99, newCount));
						batchData.total_count = batchData.latents.reduce((sum, l) => sum + (l.repeat_count || 1), 0);
						node.properties.batch_data = JSON.stringify(batchData);
						syncBatchDataWidget();
						const statsEl = container.querySelector(".instaraw-adv-loader-total");
						if (statsEl) statsEl.textContent = `Total: ${batchData.total_count} (with repeats)`;
						// Re-render to trigger update event for RPG
						renderGallery();
					}
				};

				const queueAllLatents = async () => {
					const batchData = JSON.parse(node.properties.batch_data || "{}");
					const totalCount = batchData.total_count || 0;
					if (totalCount === 0 || !confirm(`Queue ${totalCount} workflow executions?`)) return;
					try {
						const prompt = await app.graphToPrompt();
						for (let i = 0; i < totalCount; i++) {
							const promptCopy = JSON.parse(JSON.stringify(prompt.output));
							if (promptCopy[node.id] && promptCopy[node.id].inputs) {
								promptCopy[node.id].inputs.batch_index = i;
							}
							await fetch("/prompt", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ prompt: promptCopy, client_id: app.clientId }) });
							if (i < totalCount - 1) await new Promise((resolve) => setTimeout(resolve, 50));
						}
					} catch (error) {
						alert(`Queue error: ${error.message}`);
					}
				};

				const setupTxt2ImgEventHandlers = () => {
					const modeDropdown = container.querySelector(".instaraw-adv-loader-mode-dropdown");
					if (modeDropdown)
						modeDropdown.onchange = (e) => {
							const modeWidget = node.widgets?.find((w) => w.name === "mode");
							if (modeWidget) {
								modeWidget.value = e.target.value;
								renderGallery();
							}
						};
					const queueAllBtn = container.querySelector(".instaraw-adv-loader-queue-all-btn");
					if (queueAllBtn) queueAllBtn.onclick = queueAllLatents;
					const addLatentBtn = container.querySelector(".instaraw-adv-loader-add-latent-btn");
					if (addLatentBtn) addLatentBtn.onclick = addEmptyLatent;
					const batchAddBtn = container.querySelector(".instaraw-adv-loader-batch-add-btn");
					if (batchAddBtn) {
						batchAddBtn.onclick = () => {
							const countInput = container.querySelector(".instaraw-adv-loader-batch-count-input");
							const count = parseInt(countInput?.value || 5);
							if (count > 0 && count <= 100) {
								batchAddLatents(count);
							}
						};
					}
					const deleteAllBtn = container.querySelector(".instaraw-adv-loader-delete-all-btn");
					if (deleteAllBtn) deleteAllBtn.onclick = deleteAllLatents;
					container.querySelectorAll(".instaraw-adv-loader-delete-btn").forEach((btn) => (btn.onclick = (e) => { e.stopPropagation(); deleteLatent(btn.dataset.id); }));
					container.querySelectorAll(".instaraw-adv-loader-repeat-input").forEach((input) => {
						input.onchange = (e) => updateLatentRepeatCount(input.dataset.id, parseInt(input.value) || 1);
						input.onmousedown = (e) => e.stopPropagation();
					});
				};

				const handleFileSelect = async (e) => {
					const files = Array.from(e.target.files);
					if (files.length === 0) return;
					const uploadBtn = container.querySelector(".instaraw-adv-loader-upload-btn");
					const originalText = uploadBtn.textContent;
					uploadBtn.textContent = "⏳ Uploading...";
					uploadBtn.disabled = true;
					try {
						const formData = new FormData();
						formData.append("node_id", node.id);
						files.forEach((file) => formData.append("files", file));
						const response = await fetch("/instaraw/batch_upload", { method: "POST", body: formData });
						const result = await response.json();
						if (result.success) {
							const batchData = JSON.parse(node.properties.batch_data || "{}");
							batchData.images = batchData.images || [];
							batchData.order = batchData.order || [];
							result.images.forEach((img) => {
								batchData.images.push(img);
								batchData.order.push(img.id);
							});
							batchData.total_count = batchData.images.reduce((sum, img) => sum + (img.repeat_count || 1), 0);
							node.properties.batch_data = JSON.stringify(batchData);
							syncBatchDataWidget();
							renderGallery();
						} else {
							alert(`Upload failed: ${result.error}`);
						}
					} catch (error) {
						alert(`Upload error: ${error.message}`);
					} finally {
						uploadBtn.textContent = originalText;
						uploadBtn.disabled = false;
					}
				};

				const deleteImage = async (imageId) => {
					if (!confirm("Delete this image?")) return;
					try {
						await fetch(`/instaraw/batch_delete/${node.id}/${imageId}`, { method: "DELETE" });
						const batchData = JSON.parse(node.properties.batch_data || "{}");
						batchData.images = batchData.images.filter((img) => img.id !== imageId);
						batchData.order = batchData.order.filter((id) => id !== imageId);
						batchData.total_count = batchData.images.reduce((sum, img) => sum + (img.repeat_count || 1), 0);
						node.properties.batch_data = JSON.stringify(batchData);
						syncBatchDataWidget();
						renderGallery();
					} catch (error) {
						alert(`Delete error: ${error.message}`);
					}
				};

				const deleteAllImages = async () => {
					const batchData = JSON.parse(node.properties.batch_data || "{}");
					const imageCount = batchData.images?.length || 0;
					if (imageCount === 0 || !confirm(`Delete all ${imageCount} image${imageCount !== 1 ? "s" : ""}?`)) return;
					try {
						await Promise.all(batchData.images.map((img) => fetch(`/instaraw/batch_delete/${node.id}/${img.id}`, { method: "DELETE" })));
						node.properties.batch_data = JSON.stringify({ images: [], order: [], total_count: 0 });
						syncBatchDataWidget();
						renderGallery();
					} catch (error) {
						alert(`Delete all error: ${error.message}`);
					}
				};

				const queueAllImages = async () => {
					const batchData = JSON.parse(node.properties.batch_data || "{}");
					const totalCount = batchData.total_count || 0;
					if (totalCount === 0 || !confirm(`Queue ${totalCount} workflow executions?`)) return;
					try {
						const prompt = await app.graphToPrompt();
						for (let i = 0; i < totalCount; i++) {
							const promptCopy = JSON.parse(JSON.stringify(prompt.output));
							if (promptCopy[node.id] && promptCopy[node.id].inputs) {
								promptCopy[node.id].inputs.batch_index = i;
							}
							await fetch("/prompt", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ prompt: promptCopy, client_id: app.clientId }) });
							if (i < totalCount - 1) await new Promise((resolve) => setTimeout(resolve, 50));
						}
					} catch (error) {
						alert(`Queue error: ${error.message}`);
					}
				};

				const updateRepeatCount = (imageId, newCount) => {
					const batchData = JSON.parse(node.properties.batch_data || "{}");
					const img = batchData.images.find((i) => i.id === imageId);
					if (img) {
						img.repeat_count = Math.max(1, Math.min(99, newCount));
						batchData.total_count = batchData.images.reduce((sum, i) => sum + (i.repeat_count || 1), 0);
						node.properties.batch_data = JSON.stringify(batchData);
						syncBatchDataWidget();
						const statsEl = container.querySelector(".instaraw-adv-loader-total");
						if (statsEl) statsEl.textContent = `Total: ${batchData.total_count} (with repeats)`;
						// Trigger re-render to update RPG
						renderGallery();
					}
				};

				const setupEventHandlers = () => {
					const modeDropdown = container.querySelector(".instaraw-adv-loader-mode-dropdown");
					if (modeDropdown)
						modeDropdown.onchange = (e) => {
							const modeWidget = node.widgets?.find((w) => w.name === "mode");
							if (modeWidget) {
								modeWidget.value = e.target.value;
								renderGallery();
							}
						};
					const queueAllBtn = container.querySelector(".instaraw-adv-loader-queue-all-btn");
					if (queueAllBtn) queueAllBtn.onclick = queueAllImages;
					const uploadBtn = container.querySelector(".instaraw-adv-loader-upload-btn");
					if (uploadBtn)
						uploadBtn.onclick = () => {
							const input = document.createElement("input");
							input.type = "file";
							input.multiple = true;
							input.accept = "image/*";
							input.onchange = handleFileSelect;
							input.click();
						};
					const deleteAllBtn = container.querySelector(".instaraw-adv-loader-delete-all-btn");
					if (deleteAllBtn) deleteAllBtn.onclick = deleteAllImages;
					container.querySelectorAll(".instaraw-adv-loader-delete-btn").forEach((btn) => (btn.onclick = (e) => { e.stopPropagation(); deleteImage(btn.dataset.id); }));
					container.querySelectorAll(".instaraw-adv-loader-repeat-input").forEach((input) => {
						input.onchange = (e) => updateRepeatCount(input.dataset.id, parseInt(input.value) || 1);
						input.onmousedown = (e) => e.stopPropagation();
					});
				};

				const setupDragAndDrop = () => {
					const items = container.querySelectorAll(".instaraw-adv-loader-item");
					let draggedItem = null;
					items.forEach((item) => {
						item.addEventListener("dragstart", (e) => {
							draggedItem = item;
							item.style.opacity = "0.5";
							e.dataTransfer.effectAllowed = "move";
							e.stopPropagation();
							e.dataTransfer.setData("text/plain", "instaraw-reorder");
						});
						item.addEventListener("dragend", () => {
							item.style.opacity = "1";
							items.forEach((i) => i.classList.remove("instaraw-adv-loader-drop-before", "instaraw-adv-loader-drop-after"));
						});
						item.addEventListener("dragover", (e) => {
							e.preventDefault();
							if (draggedItem === item) return;
							e.dataTransfer.dropEffect = "move";
							const rect = item.getBoundingClientRect();
							const midpoint = rect.top + rect.height / 2;
							items.forEach((i) => i.classList.remove("instaraw-adv-loader-drop-before", "instaraw-adv-loader-drop-after"));
							item.classList.add(e.clientY < midpoint ? "instaraw-adv-loader-drop-before" : "instaraw-adv-loader-drop-after");
						});
						item.addEventListener("drop", (e) => {
							e.preventDefault();
							if (draggedItem === item) return;
							const draggedId = draggedItem.dataset.id;
							const targetId = item.dataset.id;
							const batchData = JSON.parse(node.properties.batch_data || "{}");
							const order = batchData.order;
							const draggedIndex = order.indexOf(draggedId);
							order.splice(draggedIndex, 1);
							const rect = item.getBoundingClientRect();
							const insertAfter = e.clientY > rect.top + rect.height / 2;
							const newTargetIndex = order.indexOf(targetId);
							order.splice(insertAfter ? newTargetIndex + 1 : newTargetIndex, 0, draggedId);
							node.properties.batch_data = JSON.stringify(batchData);
							syncBatchDataWidget();
							renderGallery();
						});
					});
				};

				const setupFileDropZone = () => {
					// Prevent duplicate listeners
					if (container._hasFileDropListeners) return;
					container._hasFileDropListeners = true;

					let dragCounter = 0; // Track nested drag events

					const handleFileDrop = async (files) => {
						if (files.length === 0) return;

						const uploadBtn = container.querySelector(".instaraw-adv-loader-upload-btn");
						if (!uploadBtn) return;

						const originalText = uploadBtn.textContent;
						uploadBtn.textContent = "⏳ Uploading...";
						uploadBtn.disabled = true;

						try {
							const formData = new FormData();
							formData.append("node_id", node.id);
							files.forEach((file) => formData.append("files", file));
							const response = await fetch("/instaraw/batch_upload", { method: "POST", body: formData });
							const result = await response.json();
							if (result.success) {
								const batchData = JSON.parse(node.properties.batch_data || "{}");
								batchData.images = batchData.images || [];
								batchData.order = batchData.order || [];
								result.images.forEach((img) => {
									batchData.images.push(img);
									batchData.order.push(img.id);
								});
								batchData.total_count = batchData.images.reduce((sum, img) => sum + (img.repeat_count || 1), 0);
								node.properties.batch_data = JSON.stringify(batchData);
								syncBatchDataWidget();
								renderGallery();
							} else {
								alert(`Upload failed: ${result.error}`);
							}
						} catch (error) {
							alert(`Upload error: ${error.message}`);
						} finally {
							uploadBtn.textContent = originalText;
							uploadBtn.disabled = false;
						}
					};

					container.addEventListener("dragenter", (e) => {
						e.preventDefault();
						e.stopPropagation();
						dragCounter++;
						if (e.dataTransfer.types.includes("Files")) {
							container.classList.add("instaraw-adv-loader-drag-over");
						}
					});

					container.addEventListener("dragover", (e) => {
						e.preventDefault();
						e.stopPropagation();
						if (e.dataTransfer.types.includes("Files")) {
							e.dataTransfer.dropEffect = "copy";
						}
					});

					container.addEventListener("dragleave", (e) => {
						e.preventDefault();
						e.stopPropagation();
						dragCounter--;
						if (dragCounter === 0) {
							container.classList.remove("instaraw-adv-loader-drag-over");
						}
					});

					container.addEventListener("drop", (e) => {
						e.preventDefault();
						e.stopPropagation();
						dragCounter = 0;
						container.classList.remove("instaraw-adv-loader-drag-over");

						// Only handle file drops, not reordering
						if (e.dataTransfer.getData("text/plain") === "instaraw-reorder") {
							return; // Let the reordering handler deal with this
						}

						const files = Array.from(e.dataTransfer.files).filter(file => file.type.startsWith("image/"));
						if (files.length > 0) {
							handleFileDrop(files);
						}
					});
				};

				const widget = node.addDOMWidget("batch_display", "batchloader", container, { getValue: () => node.properties.batch_data, setValue: (v) => { node.properties.batch_data = v; renderGallery(); }, serialize: false });
				widget.computeSize = (width) => [width, cachedHeight + 2];
				node._updateCachedHeight = updateCachedHeight;
				node._renderGallery = renderGallery;

				// Add widget change callbacks to automatically refresh
				const setupWidgetCallbacks = () => {
					const modeWidget = node.widgets?.find((w) => w.name === "mode");
					if (modeWidget && !modeWidget._instaraw_callback_added) {
						const originalCallback = modeWidget.callback;
						modeWidget.callback = function() {
							if (originalCallback) originalCallback.apply(this, arguments);
							renderGallery();
						};
						modeWidget._instaraw_callback_added = true;
					}

					const batchIndexWidget = node.widgets?.find((w) => w.name === "batch_index");
					if (batchIndexWidget && !batchIndexWidget._instaraw_callback_added) {
						const originalCallback = batchIndexWidget.callback;
						batchIndexWidget.callback = function() {
							if (originalCallback) originalCallback.apply(this, arguments);
							renderGallery();
						};
						batchIndexWidget._instaraw_callback_added = true;
					}

					// Aspect ratio widgets - re-render when dimensions change
					const widthWidget = node.widgets?.find((w) => w.name === "width");
					if (widthWidget && !widthWidget._instaraw_callback_added) {
						const originalCallback = widthWidget.callback;
						widthWidget.callback = function() {
							if (originalCallback) originalCallback.apply(this, arguments);
							renderGallery();
						};
						widthWidget._instaraw_callback_added = true;
					}

					const heightWidget = node.widgets?.find((w) => w.name === "height");
					if (heightWidget && !heightWidget._instaraw_callback_added) {
						const originalCallback = heightWidget.callback;
						heightWidget.callback = function() {
							if (originalCallback) originalCallback.apply(this, arguments);
							renderGallery();
						};
						heightWidget._instaraw_callback_added = true;
					}

					const aspectLabelWidget = node.widgets?.find((w) => w.name === "aspect_label");
					if (aspectLabelWidget && !aspectLabelWidget._instaraw_callback_added) {
						const originalCallback = aspectLabelWidget.callback;
						aspectLabelWidget.callback = function() {
							if (originalCallback) originalCallback.apply(this, arguments);
							renderGallery();
						};
						aspectLabelWidget._instaraw_callback_added = true;
					}
				};

				// Periodic mode check - checks every 2 seconds if mode changed
				const startModeCheck = () => {
					if (modeCheckInterval) clearInterval(modeCheckInterval);
					modeCheckInterval = setInterval(() => {
						const detectedMode = isTxt2ImgMode();
						if (currentDetectedMode !== null && currentDetectedMode !== detectedMode) {
							renderGallery();
						}
					}, 2000);
					// Store on node for cleanup
					node._modeCheckInterval = modeCheckInterval;
				};

				// Periodic dimension check - checks every 2 seconds if dimensions changed
				let dimensionCheckInterval = null;
				let lastDimensions = null;
				const startDimensionCheck = () => {
					if (dimensionCheckInterval) clearInterval(dimensionCheckInterval);
					dimensionCheckInterval = setInterval(() => {
						// Only check dimensions in txt2img mode (where aspect ratio matters for latent display)
						if (isTxt2ImgMode()) {
							const currentDims = getTxt2ImgDimensions();
							const dimsKey = `${currentDims.width}x${currentDims.height}:${currentDims.aspect_label}`;
							if (lastDimensions !== null && lastDimensions !== dimsKey) {
								console.log(`[INSTARAW AIL ${node.id}] Dimensions changed: ${lastDimensions} -> ${dimsKey}`);
								renderGallery();
							}
							lastDimensions = dimsKey;
						}
					}, 2000);
					// Store on node for cleanup
					node._dimensionCheckInterval = dimensionCheckInterval;
				};

				const handleBatchUpdate = (event) => {
					const data = event.detail;
					if (data && data.node_id == node.id) {
						const batchIndexWidget = node.widgets?.find((w) => w.name === "batch_index");
						if (batchIndexWidget) batchIndexWidget.value = data.next_index;
						node._processingIndex = data.next_index;
						node._isProcessing = false;
						if (node._renderGallery) node._renderGallery();
						app.graph.setDirtyCanvas(true, false);
					}
				};
				api.addEventListener("instaraw_adv_loader_update", handleBatchUpdate);

				// Listen for Sync requests from RPG
				window.addEventListener("INSTARAW_SYNC_AIL_LATENTS", (event) => {
					const { targetNodeId, latentSpecs, dimensions } = event.detail;
					if (node.id !== targetNodeId) return; // Not for this node

					console.log(`[INSTARAW AIL ${node.id}] Received sync request: Create ${latentSpecs.length} empty latents with repeat counts`);

					// Get current dimensions or use provided
					const currentDimensions = getTxt2ImgDimensions();
					const width = dimensions?.width || currentDimensions.width;
					const height = dimensions?.height || currentDimensions.height;
					const aspect_label = dimensions?.aspect_label || currentDimensions.aspect_label;

					// Clear existing latents and create new ones
					const batchData = JSON.parse(node.properties.batch_data || "{}");
					batchData.latents = [];
					batchData.order = [];

					// Create latents with repeat counts matching prompts
					let totalCount = 0;
					for (let i = 0; i < latentSpecs.length; i++) {
						const spec = latentSpecs[i];
						const newLatent = {
							id: generateUUID(),
							width: width,
							height: height,
							repeat_count: spec.repeat_count || 1,
							aspect_label: aspect_label
						};
						batchData.latents.push(newLatent);
						batchData.order.push(newLatent.id);
						totalCount += newLatent.repeat_count;
					}

					batchData.total_count = totalCount;
					node.properties.batch_data = JSON.stringify(batchData);

					syncBatchDataWidget();
					renderGallery();

					console.log(`[INSTARAW AIL ${node.id}] Created ${latentSpecs.length} latents (${totalCount} total generations) (${width}×${height})`);
				});

				// Listen for Repeat Sync requests from RPG
				window.addEventListener("INSTARAW_SYNC_AIL_REPEATS", (event) => {
					const { targetNodeId, mode, repeats } = event.detail;
					if (node.id !== targetNodeId) return; // Not for this node

					console.log(`[INSTARAW AIL ${node.id}] Received repeat sync request: Update ${repeats.length} items`);

					const batchData = JSON.parse(node.properties.batch_data || "{}");
					const items = mode === "img2img" ? batchData.images : batchData.latents;
					const order = batchData.order || [];

					if (!items || items.length === 0) {
						alert("No images/latents in AIL to sync!");
						return;
					}

					// Update repeat counts to match prompts, respecting display order
					let totalCount = 0;
					repeats.forEach((repeatCount, idx) => {
						// Find item by order, not by raw array index
						const itemId = order[idx];
						if (itemId) {
							const item = items.find(i => i.id === itemId);
							if (item) {
								item.repeat_count = repeatCount;
								totalCount += repeatCount;
							}
						}
					});

					batchData.total_count = totalCount;
					node.properties.batch_data = JSON.stringify(batchData);

					syncBatchDataWidget();
					renderGallery();

					console.log(`[INSTARAW AIL ${node.id}] Synced repeat counts: ${repeats.length} items, ${totalCount} total`);
				});

				setTimeout(() => {
					const batchIndexWidget = node.widgets?.find((w) => w.name === "batch_index");
					if (batchIndexWidget && batchIndexWidget.value !== undefined) node._processingIndex = batchIndexWidget.value;
					syncBatchDataWidget();
					setupWidgetCallbacks();
					startModeCheck();
					startDimensionCheck();
					renderGallery();
				}, 100);
			};

			const onResize = nodeType.prototype.onResize;
			nodeType.prototype.onResize = function (size) {
				onResize?.apply(this, arguments);
				if (this._updateCachedHeight) {
					clearTimeout(this._resizeTimeout);
					this._resizeTimeout = setTimeout(() => this._updateCachedHeight(), 50);
				}
			};

			const onConfigure = nodeType.prototype.onConfigure;
			nodeType.prototype.onConfigure = function (data) {
				onConfigure?.apply(this, arguments);
				setTimeout(() => {
					const batchDataWidget = this.widgets?.find((w) => w.name === "batch_data");
					if (batchDataWidget) batchDataWidget.value = this.properties.batch_data || "{}";
					const batchIndexWidget = this.widgets?.find((w) => w.name === "batch_index");
					if (batchIndexWidget && batchIndexWidget.value !== undefined) this._processingIndex = batchIndexWidget.value;
					if (this._renderGallery) this._renderGallery();
				}, 200);
			};

			const onRemoved = nodeType.prototype.onRemoved;
			nodeType.prototype.onRemoved = function () {
				// Clean up the periodic mode check interval
				if (this._modeCheckInterval) {
					clearInterval(this._modeCheckInterval);
					this._modeCheckInterval = null;
				}
				// Clean up the periodic dimension check interval
				if (this._dimensionCheckInterval) {
					clearInterval(this._dimensionCheckInterval);
					this._dimensionCheckInterval = null;
				}
				onRemoved?.apply(this, arguments);
			};
		}
	},
});