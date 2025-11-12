import { app } from '../../scripts/app.js';

app.registerExtension({
	name: 'Comfy.INSTARAW.BrandingNode', // ZWJ_WATERMARK_HERE
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
		if (nodeData.name === 'INSTARAW_BrandingNode') {
			// This function is called once when the node is added to the graph.
			const onAdded = nodeType.prototype.onAdded;
			nodeType.prototype.onAdded = function () {
				onAdded?.apply(this, arguments);

				this.properties = this.properties || {};

				// Set default properties ONLY if they don't already exist.
				if (!this.properties.logo_url) {
					this.properties.logo_url = 'https://instara.s3.us-east-1.amazonaws.com/INSTARAW_logomark_only.svg';
				}
				if (this.properties.width == null) {
					this.properties.width = 120;
				}
				if (this.properties.height == null) {
					this.properties.height = 105;
				}

				this.logoImage = new Image();
				this.logoImage.onload = () => {
					this.setDirtyCanvas(true, false);
				};

				this.bgcolor = 'transparent';
				this.boxcolor = 'transparent';
			};

			// This function is called every frame to draw the node.
			const onDrawForeground = nodeType.prototype.onDrawForeground;
			nodeType.prototype.onDrawForeground = function (ctx) {
				onDrawForeground?.apply(this, arguments);

				// --- THIS IS THE FIX ---
				// We no longer force the title to be empty.
				// this.title = "";  <-- THIS LINE HAS BEEN REMOVED.

				const url = this.properties.logo_url;
				if (url && this.logoImage.src !== url) {
					this.logoImage.src = url;
				}

				if (this.size && this.logoImage.complete && this.logoImage.naturalWidth > 0) {
					const horizontalPadding = 10;
					const verticalPadding = 10; // New: Define vertical padding

					const nodeWidth = this.size[0];
					const nodeHeight = this.size[1];

					// The available area for drawing is now smaller
					const drawAreaWidth = nodeWidth - horizontalPadding * 2;
					const drawAreaHeight = nodeHeight - verticalPadding * 2; // Changed: Account for top and bottom padding

					const imgAspect = this.logoImage.naturalWidth / this.logoImage.naturalHeight;

					let drawWidth = drawAreaWidth;
					let drawHeight = drawWidth / imgAspect;

					if (drawHeight > drawAreaHeight) {
						drawHeight = drawAreaHeight;
						drawWidth = drawHeight * imgAspect;
					}

					// The x position calculation remains the same
					const x = horizontalPadding + (drawAreaWidth - drawWidth) / 2;

					// The y position now starts after the top padding
					const y = verticalPadding + (drawAreaHeight - drawHeight) / 2; // Changed: Add the top padding offset

					ctx.drawImage(this.logoImage, x, y, drawWidth, drawHeight);
				}
			};

			// This function controls the node's size.
			const computeSize = nodeType.prototype.computeSize;
			nodeType.prototype.computeSize = function () {
				const size = computeSize?.apply(this, arguments);
				// Get the size from our properties.
				const width = this.properties.width || 120;
				const height = this.properties.height || 105;
				// Ensure the node is big enough to show the title if it exists.
				return [Math.max(size[0], width), height];
			};
		}
	}
});
