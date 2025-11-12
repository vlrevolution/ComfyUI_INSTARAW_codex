import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { create } from "./utils.js";
import { popup } from "./popup.js";
import { ComfyWidgets } from "../../scripts/widgets.js";

const FILTER_TYPES = ["INSTARAW_ImageFilter", "INSTARAW_TextImageFilter", "INSTARAW_MaskImageFilter", "INSTARAW_Interactive_Crop"];

app.registerExtension({
    name: "Comfy.INSTARAW.InteractiveNodes", // ZWJ_WATERMARK_HERE
    settings: [
        {
            id: "INSTARAW.Interactive.Header",
            name: "INSTARAW Interactive Nodes",
            type: () => {
                const x = document.createElement('span');
                const a = document.createElement('a');
                a.innerText = "Based on original work by chrisgoringe (cg-image-filter)";
                a.href = "https://github.com/chrisgoringe/cg-image-filter";
                a.target = "_blank";
                a.style.paddingRight = "12px";
                x.appendChild(a);
                return x;
            },
        },
        { id: "INSTARAW.Interactive.PlaySound", name: "Play sound when activating", type: "boolean", defaultValue: true },
        { id: "INSTARAW.Interactive.EnlargeSmall", name: "Enlarge small images in grid", type: "boolean", defaultValue: true },
        { id: "INSTARAW.Interactive.ClickSends", name: "Clicking an image sends it", tooltip: "Use if you always want to send exactly one image.", type: "boolean", defaultValue: false },
        { id: "INSTARAW.Interactive.AutosendIdentical", name: "If all images are identical, autosend one", type: "boolean", defaultValue: false },
        { id: "INSTARAW.Interactive.StartZoomed", name: "Enter the Image Filter node with an image zoomed", type: "combo", options: [{ value: 0, text: "No" }, { value: "1", text: "first" }, { value: "-1", text: "last" }], default: 0 },
        { id: "INSTARAW.Interactive.SmallWindow", name: "Show a small popup instead of covering the screen", type: "boolean", tooltip: "Click the small popup to activate it", defaultValue: false },
        { id: "INSTARAW.Interactive.DetailedLogging", name: "Turn on detailed logging", tooltip: "If you are asked to for debugging!", type: "boolean", defaultValue: false },
        { id: "INSTARAW.Interactive.FPS", name: "Video Frames per Second", type: "int", defaultValue: 5 }
    ],
    setup() {
        create('link', null, document.getElementsByTagName('HEAD')[0], { 'rel': 'stylesheet', 'type': 'text/css', 'href': 'extensions/ComfyUI_INSTARAW/filter.css' });
        create('link', null, document.getElementsByTagName('HEAD')[0], { 'rel': 'stylesheet', 'type': 'text/css', 'href': 'extensions/ComfyUI_INSTARAW/floating_window.css' });
        create('link', null, document.getElementsByTagName('HEAD')[0], { 'rel': 'stylesheet', 'type': 'text/css', 'href': 'extensions/ComfyUI_INSTARAW/zoomed.css' });
        create('link', null, document.getElementsByTagName('HEAD')[0], { 'rel': 'stylesheet', 'type': 'text/css', 'href': 'extensions/ComfyUI_INSTARAW/advanced_image_loader.css' });

        api.addEventListener("execution_interrupted", popup.send_cancel.bind(popup));
        api.addEventListener("instaraw-interactive-images", popup.handle_message.bind(popup));
    },
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeType.comfyClass === "Pick from List" || nodeType.comfyClass === "INSTARAW_PickFromList") {
            const onConnectionsChange = nodeType.prototype.onConnectionsChange;
            nodeType.prototype.onConnectionsChange = function(side, slot, connect, link_info, output) {
                if (side == 1 && slot == 0 && link_info && connect) {
                    const originNode = this.graph.getNodeById(link_info.origin_id);
                    if (originNode?.outputs?.[link_info.origin_slot]) {
                        const type = originNode.outputs[link_info.origin_slot].type;
                        this.outputs[0].type = type;
                        this.inputs[0].type = type;
                    }
                } else if (side == 1 && slot == 0 && !connect) {
                    this.outputs[0].type = "*";
                    this.inputs[0].type = "*";
                }
                return onConnectionsChange?.apply(this, arguments);
            }
        }
        if (FILTER_TYPES.includes(nodeType.comfyClass)) {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {
                const onCreatedResult = onNodeCreated?.apply(this, arguments);

                this._ni_widget = this.widgets.find((n) => n.name == 'node_identifier');
                if (!this._ni_widget) {
                    this._ni_widget = ComfyWidgets["INT"](this, "node_identifier", ["INT", { "default": 0 }], app).widget;
                }
                this._ni_widget.hidden = true;
                this._ni_widget.computeSize = () => [0, 0];
                this._ni_widget.value = Math.floor(Math.random() * 1000000);

                if (this.comfyClass === "INSTARAW_TextImageFilter") {
                    
                    const buttonWidget = this.addWidget("button", "Clear Node Cache", null, async () => {
                        buttonWidget.name = "Clearing...";
                        this.disabled = true;

                        try {
                            const resp = await api.fetchApi('/instaraw/clear_text_filter_cache', {
                                method: 'POST',
                                headers: { 'Content-Type': 'application/json' },
                                body: JSON.stringify({ uid: this._ni_widget.value }),
                            });
                            
                            if (resp.status === 200) {
                                buttonWidget.success();
                            } else {
                                throw new Error(await resp.text());
                            }
                        } catch (e) {
                            console.error("INSTARAW: Failed to clear cache:", e);
                            buttonWidget.error();
                        }
                    });
                    
                    buttonWidget.name = "clear_cache_button";

                    buttonWidget.success = () => {
                        buttonWidget.name = "Cache Cleared!";
                        this.disabled = false;
                        setTimeout(() => {
                            buttonWidget.name = "Clear Node Cache";
                        }, 2000);
                    };

                    buttonWidget.error = () => {
                        buttonWidget.name = "Error Clearing!";
                        this.disabled = false;
                        setTimeout(() => {
                            buttonWidget.name = "Clear Node Cache";
                        }, 3000);
                    };
                }

                return onCreatedResult;
            }
        }
    },
});