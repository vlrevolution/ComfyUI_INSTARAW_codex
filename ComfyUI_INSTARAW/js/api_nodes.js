import { app } from '../../scripts/app.js';

// _INSTARA_INJECT_

app.registerExtension({
	name: 'Comfy.INSTARAW.DynamicAPINodes',
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
		if (nodeData.name === 'INSTARAW_APITextToImage' || nodeData.name === 'INSTARAW_APIImageToImage') {
			
			const onNodeCreated = nodeType.prototype.onNodeCreated;
			nodeType.prototype.onNodeCreated = function () {
                onNodeCreated?.apply(this, arguments);
				this.last_model = null; // Cache property
			};

			const onDrawForeground = nodeType.prototype.onDrawForeground;
			nodeType.prototype.onDrawForeground = function(ctx) {
				onDrawForeground?.apply(this, arguments);

				const modelInput = this.inputs?.find((i) => i.name === 'model');
				let current_model = null;
				
				if (modelInput && modelInput.link != null) {
					const link = app.graph.links[modelInput.link];
					if (link) {
						const origin_node = app.graph.getNodeById(link.origin_id);
						const origin_widget = origin_node?.widgets.find(w => w.name === 'model');
						if (origin_widget) current_model = origin_widget.value;
					}
				}
				
				if (this.last_model === current_model) {
					return; // No change, do nothing
				}
				this.last_model = current_model;

				const isSeeDream = current_model === 'SeeDream v4';

				this.widgets.find(w => w.name === 'width').hidden = !isSeeDream;
				this.widgets.find(w => w.name === 'height').hidden = !isSeeDream;
				this.widgets.find(w => w.name === 'enable_safety_checker').hidden = !isSeeDream;
				
				const aspectRatioWidget = this.widgets.find(w => w.name === 'aspect_ratio');
				if (aspectRatioWidget) { 
					aspectRatioWidget.hidden = isSeeDream;
				}

				this.computeSize();
				this.setDirtyCanvas(true, true);
			};
		}
	},
});