// --- Filename: ../ComfyUI_INSTARAW/js/group_bypass_detector.js ---

import { app } from "../../scripts/app.js";

const MODE_ALWAYS = 0; // LiteGraph.ALWAYS

app.registerExtension({
	name: "Comfy.INSTARAW.GroupBypassDetector",

	async beforeRegisterNodeDef(nodeType, nodeData, app) {
		if (nodeData.name === "INSTARAW_GroupBypassToBoolean") {

			// Store the last known state on the node instance to prevent unnecessary updates
			const onNodeCreated = nodeType.prototype.onNodeCreated;
			nodeType.prototype.onNodeCreated = function () {
				onNodeCreated?.apply(this, arguments);
				this.lastBypassState = undefined;
			};

			// Every frame, check if our bypass state has changed
			const onDrawForeground = nodeType.prototype.onDrawForeground;
			nodeType.prototype.onDrawForeground = function(ctx) {
				onDrawForeground?.apply(this, arguments);
				
				const widget = this.widgets.find(w => w.name === 'is_active');
				if (!widget) return;

				// The node's mode tells us if it's bypassed. 0 means active.
				const isActive = this.mode === MODE_ALWAYS;

				// Only update the widget if the state has actually changed
				if (this.lastBypassState !== isActive) {
					this.lastBypassState = isActive;
					widget.value = isActive;
					
					// Make sure the UI reflects the change immediately
					app.graph.setDirtyCanvas(true);
				}
			};
		}
	},
});