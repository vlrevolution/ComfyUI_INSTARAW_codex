import { app } from '../../scripts/app.js';
import { ComfyWidgets } from '../../scripts/widgets.js';

app.registerExtension({
	name: 'Comfy.INSTARAW.APIModelSelector',
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
		if (nodeData.name === 'INSTARAW_API_ModelSelector') {
			
            const onNodeCreated = nodeType.prototype.onNodeCreated;
			nodeType.prototype.onNodeCreated = function () {
                onNodeCreated?.apply(this, arguments);

				// Create the warning widget for this node
				const warningWidget = ComfyWidgets["STRING"](this, "unreliable_provider_warning", ["STRING", { multiline: true }], app).widget;
				warningWidget.inputEl.readOnly = true;
				warningWidget.inputEl.style.backgroundColor = '#442222';
				warningWidget.inputEl.style.color = '#FFCCCC';
				warningWidget.inputEl.style.opacity = '0.8';
				warningWidget.value = '⚠️ For this model, the fal.ai provider is unreliable for Image-to-Image (Edit) tasks. wavespeed.ai recommended.';
				warningWidget.hidden = true; // Initially hidden

				const modelWidget = this.widgets.find((w) => w.name === 'model');
				if (modelWidget) {
					const originalCallback = modelWidget.callback;
					// Add our logic to the dropdown's callback
					modelWidget.callback = (value) => {
						originalCallback?.(value);
						this.updateWarningWidget();
					};
				}

				// Add a function to the node to update the widget
				this.updateWarningWidget = () => {
					const isProblematic = (modelWidget.value === 'Nano Banana');
					warningWidget.hidden = !isProblematic;
					this.computeSize();
					this.setDirtyCanvas(true, true);
				};
				
				// Run once on creation to set the initial state
				setTimeout(() => this.updateWarningWidget(), 1);
			};
		}
	},
});