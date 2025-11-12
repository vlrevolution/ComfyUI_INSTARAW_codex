// --- Filename: ../ComfyUI_INSTARAW/js/boolean_bypass.js (FINAL, GROUP-BYPASS-AWARE VERSION) ---

import { app } from "../../scripts/app.js";

const MODE_ALWAYS = 0; // Corresponds to LiteGraph.ALWAYS
const MODE_BYPASS = 4;

app.registerExtension({
	name: "Comfy.INSTARAW.BooleanBypass.Final.V4", // ZWJ_WATERMARK_HERE

	async beforeRegisterNodeDef(nodeType, nodeData, app) {
		if (nodeData.name === "INSTARAW_BooleanBypass") {

			const onNodeCreated = nodeType.prototype.onNodeCreated;
			nodeType.prototype.onNodeCreated = function () {
				onNodeCreated?.apply(this, arguments);
				this.lastKnownState = undefined;
			};

			const onDrawForeground = nodeType.prototype.onDrawForeground;
			nodeType.prototype.onDrawForeground = function(ctx) {
				onDrawForeground?.apply(this, arguments);
				
				// --- THIS IS THE FIX ---
				// If this node itself is currently bypassed (e.g., by a group bypasser),
				// we must reset its internal state. This ensures that the next time it becomes
				// active, it will perform a full re-sync of its upstream nodes.
				if (this.mode !== MODE_ALWAYS) {
					this.lastKnownState = undefined;
					return; // Do nothing further while bypassed.
				}
				// --- END FIX ---

				// Find all our required widgets and inputs
				const booleanWidget = this.widgets.find(w => w.name === 'boolean');
				const invertWidget = this.widgets.find(w => w.name === 'invert_input');
				const booleanInput = this.inputs.find(i => i.name === 'boolean');
				if (!booleanWidget || !invertWidget || !booleanInput) return; // Safety check

				// 1. Determine the RAW boolean state from the controller (internal or external)
				let rawIsEnabled = false;
				if (booleanInput.link != null) {
					const link = app.graph.links[booleanInput.link];
					const originNode = link && app.graph.getNodeById(link.origin_id);
					if (originNode) {
						let originWidget = originNode.widgets?.find(w => w.name === (originNode.outputs[link.origin_slot]?.name));
						if (!originWidget && originNode.widgets?.length === 1) {
							originWidget = originNode.widgets[0];
						}
						
						if (originWidget) {
							rawIsEnabled = !!originWidget.value;
						} else {
							rawIsEnabled = originNode.outputs[link.origin_slot]?.value !== undefined ? !!originNode.outputs[link.origin_slot].value : false;
						}
					}
				} else {
					rawIsEnabled = booleanWidget.value;
				}

				// 2. Apply the inversion logic
				const shouldInvert = invertWidget.value;
				const finalIsEnabled = shouldInvert ? !rawIsEnabled : rawIsEnabled;

				// 3. Only proceed if the final state has changed (or if state was just reset)
				if (this.lastKnownState === finalIsEnabled) {
					return;
				}
				
				this.lastKnownState = finalIsEnabled;

				// 4. Update UI and apply bypass
				booleanWidget.value = rawIsEnabled;

				const newMode = finalIsEnabled ? MODE_ALWAYS : MODE_BYPASS;

				for (const input of this.inputs) {
					if (input.name === 'boolean' || input.name === 'invert_input') continue; // Skip control inputs
					if (input.link != null) {
						const link = app.graph.links[input.link];
						const connectedNode = link && app.graph.getNodeById(link.origin_id);
						if (connectedNode) {
							connectedNode.mode = newMode;
						}
					}
				}
			};
		}
	},
});