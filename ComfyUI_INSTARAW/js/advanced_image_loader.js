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

				const node = this;
				let cachedHeight = 300;
				let isUpdatingHeight = false;

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

				const renderGallery = () => {
					const batchData = JSON.parse(node.properties.batch_data || "{}");
					const images = batchData.images || [];
					const order = batchData.order || [];
					const modeWidget = node.widgets?.find((w) => w.name === "mode");
					const currentMode = modeWidget?.value || "Batch Tensor";
					const batchIndexWidget = node.widgets?.find((w) => w.name === "batch_index");
					const currentIndex = node._processingIndex !== undefined ? node._processingIndex : batchIndexWidget?.value || 0;

					container.innerHTML = `
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
					updateCachedHeight();

					// Dispatch update event for other nodes (e.g., RPG)
					window.dispatchEvent(new CustomEvent("INSTARAW_AIL_UPDATED", {
						detail: {
							nodeId: node.id,
							images: order.map(imgId => {
								const img = images.find(i => i.id === imgId);
								if (!img) return null;
								const thumbUrl = `/view?filename=${img.thumbnail}&type=input&subfolder=INSTARAW_BatchUploads/${node.id}`;
								return {url: thumbUrl, index: order.indexOf(imgId), id: imgId};
							}).filter(i => i !== null),
							total: batchData.total_count || 0
						}
					}));
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

				const widget = node.addDOMWidget("batch_display", "batchloader", container, { getValue: () => node.properties.batch_data, setValue: (v) => { node.properties.batch_data = v; renderGallery(); }, serialize: false });
				widget.computeSize = (width) => [width, cachedHeight + 2];
				node._updateCachedHeight = updateCachedHeight;
				node._renderGallery = renderGallery;

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

				setTimeout(() => {
					const batchIndexWidget = node.widgets?.find((w) => w.name === "batch_index");
					if (batchIndexWidget && batchIndexWidget.value !== undefined) node._processingIndex = batchIndexWidget.value;
					syncBatchDataWidget();
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
		}
	},
});