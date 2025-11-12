import { app, ComfyApp } from '../../scripts/app.js';
import { api } from '../../scripts/api.js';

import {
	mask_editor_listen_for_cancel,
	mask_editor_showing,
	hide_mask_editor,
	press_maskeditor_cancel,
	press_maskeditor_save,
	new_editor
} from './mask_utils.js';
import { Log } from './log.js';
import { create } from './utils.js';
import { FloatingWindow } from './floating_window.js';

// Renamed to INSTARAW node class names
const POPUP_NODES = ['INSTARAW_ImageFilter', 'INSTARAW_TextImageFilter', 'INSTARAW_Interactive_Crop'];
const MASK_NODES = ['INSTARAW_MaskImageFilter'];

const REQUEST_RESHOW = '-1';
const CANCEL = '-3';

const GRID_IMAGE_SPACE = 10;

function get_full_url(url) {
	return api.apiURL(
		`/view?filename=${encodeURIComponent(url.filename ?? v)}&type=${url.type ?? 'input'}&subfolder=${url.subfolder ?? ''}&r=${Math.random()}`
	);
}

const State = Object.freeze({
	INACTIVE: 0,
	TINY: 1,
	MASK: 2,
	FILTER: 3,
	TEXT: 4,
	ZOOMED: 5,
	CROP: 6 // Our new state for the interactive cropper
});

class Popup extends HTMLElement {
	constructor() {
		super();
		// Path updated to point to our package's assets
		this.audio = new Audio('extensions/ComfyUI_INSTARAW/ding.mp3');

		this.classList.add('instaraw_popup');

		this.grid = create('span', 'grid', this);
		this.overlaygrid = create('span', 'grid overlaygrid', this);
		this.grid.addEventListener('click', this.on_click.bind(this));

		// --- NEW CROP UI ELEMENTS ---
		this.crop_container = create('div', 'crop_container', this);
		this.crop_image_bg = create('img', 'crop_image_bg', this.crop_container);
		this.crop_canvas = create('canvas', 'crop_canvas_overlay', this.crop_container);
		this.crop_ctx = this.crop_canvas.getContext('2d');
        this.crop_canvas.addEventListener('mousedown', this.on_crop_mouse_down.bind(this));
        // We now listen for moves on the canvas for cursor updates, and the whole document for dragging
        this.crop_canvas.addEventListener('mousemove', this.on_crop_mouse_move.bind(this));
        document.addEventListener('mouseup', this.on_crop_mouse_up.bind(this));
        document.addEventListener('mousemove', this.on_document_mouse_move.bind(this));
		this.crop_data = { active: false, box: null, mode: null };
		// --- END NEW CROP UI ---

		this.zoomed = create('span', 'zoomed', this);
		this.zoomed_prev = create('span', 'zoomed_prev', this.zoomed);
		this.zoomed_prev_arrow = create('span', 'zoomed_arrow', this.zoomed_prev, { innerHTML: '&#x21E6' });
		this.zoomed_image = create('img', 'zoomed_image', this.zoomed);
		this.zoomed_next = create('span', 'zoomed_next', this.zoomed);
		this.zoomed_number = create('span', 'zoomed_number', this.zoomed_next);
		this.zoomed_next_arrow = create('span', 'zoomed_arrow', this.zoomed_next, { innerHTML: '&#x21E8' });

		this.zoomed_prev_arrow.addEventListener('click', this.zoom_prev.bind(this));
		this.zoomed_next_arrow.addEventListener('click', this.zoom_next.bind(this));
		this.zoomed_image.addEventListener('click', this.click_zoomed.bind(this));

		this.tiny_window = new FloatingWindow('', 100, 100, null, this.tiny_moved.bind(this));
		this.tiny_window.classList.add('tiny');
		this.tiny_image = create('img', 'tiny_image', this.tiny_window.body);
		this.tiny_window.addEventListener('click', this.handle_deferred_message.bind(this));

		this.floating_window = new FloatingWindow('', 100, 100, null, this.floater_moved.bind(this));

		this.counter_row = create('span', 'counter row', this.floating_window.body);
		this.counter_reset_button = create('button', 'counter_reset', this.counter_row, { innerText: 'Reset' });
		this.counter_text = create('span', 'counter_text', this.counter_row);
		this.counter_reset_button.addEventListener('click', this.request_reset.bind(this));

		this.extras_row = create('span', 'extras row', this.floating_window.body);

		this.tip_row = create('span', 'tip row', this.floating_window.body);

		this.button_row = create('span', 'buttons row', this.floating_window.body);
		this.send_button = create('button', 'control', this.button_row, { innerText: 'Send' });
		this.cancel_button = create('button', 'control', this.button_row, { innerText: 'Cancel' });
		this.toggle_button = create('button', 'control', this.button_row, { innerText: 'Hide' });
		this.send_button.addEventListener('click', this.send_current_state.bind(this));
		this.cancel_button.addEventListener('click', this.send_cancel.bind(this));
		this.toggle_button.addEventListener('click', this.toggleHide.bind(this));

		this.mask_button_row = create('span', 'buttons row', this.floating_window.body);
		this.mask_send_button = create('button', 'control', this.mask_button_row, { innerText: 'Send' });
		this.mask_cancel_button = create('button', 'control', this.mask_button_row, { innerText: 'Cancel' });
		this.mask_send_button.addEventListener('click', press_maskeditor_save);
		this.mask_cancel_button.addEventListener('click', press_maskeditor_cancel);

		this.text_edit = create('textarea', 'text_edit row', this.floating_window.body);

		this.picked = new Set();

		document.addEventListener('keydown', this.on_key_down.bind(this));
		document.addEventListener('keypress', this.on_key_press.bind(this));

		document.body.appendChild(this);
		this.last_response_sent = 0;
		this.state = State.INACTIVE;
		this.hidden_by_toggle = false;
		this.render();
	}

	toggleHide() {
		this.hidden_by_toggle = !this.hidden_by_toggle;
		this.render();
	}

	floater_moved(x, y) {
		if (this.node?.properties) {
			this.node.properties['filter_floater_xy'] = { x: x, y: y };
		}
	}

	floater_position() {
		return this.node?.properties?.['filter_floater_xy'];
	}

	tiny_moved(x, y) {
		if (this.node?.properties) {
			this.node.properties['filter_tiny_xy'] = { x: x, y: y };
		}
	}

	tiny_position() {
		return this.node?.properties?.['filter_tiny_xy'];
	}

	visible(item, value) {
		if (value) item.classList.remove('hidden');
		else item.classList.add('hidden');
	}
	disabled(item, value) {
		item.disabled = value;
	}
	highlighted(item, value) {
		if (value) item.classList.add('highlighted');
		else item.classList.remove('highlighted');
	}

	render() {
		const state = this.state;
		// --- UPDATED RENDER LOGIC ---
		const isVisible =
			(state === State.FILTER || state === State.TEXT || state === State.ZOOMED || state === State.CROP) &&
			!this.hidden_by_toggle;
		this.visible(this, isVisible);

		this.visible(this.tiny_window, state === State.TINY);
		this.visible(this.zoomed, state === State.ZOOMED);

		this.visible(this.crop_container, state === State.CROP);

		const showFloater =
			state === State.FILTER ||
			state === State.ZOOMED ||
			state === State.TEXT ||
			state === State.MASK ||
			state === State.CROP;
		this.visible(this.floating_window, showFloater);

		this.visible(this.button_row, state !== State.MASK);

		// Disable send button if no images picked (filter) or no crop box drawn (crop)
        const sendDisabled =
            ((state === State.FILTER || state === State.ZOOMED) && this.picked.size === 0) ||
            (state === State.CROP && (!this.crop_data.box || this.crop_data.box.width <= 0 || this.crop_data.box.height <= 0));
		this.disabled(this.send_button, sendDisabled);

		this.visible(this.mask_button_row, state === State.MASK && new_editor());
		this.visible(this.extras_row, this.n_extras > 0);
		this.visible(this.tip_row, this.tip_row.innerHTML.length > 0);
		this.visible(this.text_edit, state === State.TEXT);

		if (state === State.ZOOMED) {
			const img_index = this.zoomed_image_holder.image_index;
			this.highlighted(this.zoomed, this.picked.has(`${img_index}`));
			this.zoomed_number.innerHTML = `${img_index + 1}/${this.n_images}`;
		}

		if (state !== State.MASK) hide_mask_editor();

		this.toggle_button.innerText = this.hidden_by_toggle ? 'Show' : 'Hide';
		const showToggleButton =
			state === State.FILTER || state === State.ZOOMED || state === State.TEXT || state === State.CROP;
		this.visible(this.toggle_button, showToggleButton);
	}

	_send_response(msg = {}, keep_open = false) {
		if (Date.now() - this.last_response_sent < 1000) {
			Log.message_out(msg, '(throttled)');
			return;
		}
		const unique = this.node?._ni_widget?.value;
		if (!unique) {
			if (this.node) Log.error(`Node ${this.node.id} has no _ni_widget when trying to send ${msg}`);
			else Log.error(`No node when trying to send ${msg}`);
			return;
		}
		msg.unique = `${unique}`;
		if (!msg.special) {
			if (this.n_extras > 0) {
				msg.extras = Array.from(this.extras_row.children).map((e) => e.value);
			}
			if (this.state === State.FILTER || this.state === State.ZOOMED) msg.selection = Array.from(this.picked);
			if (this.state === State.TEXT) msg.text = this.text_edit.value;
			// --- NEW: Add crop data to the response ---
			if (this.state === State.CROP && this.crop_data.box) {
				// Convert from canvas-space to original image-space before sending
				const { box, scale } = this.crop_data;
				// Send 'crop' at the top level, not nested under 'response'
				msg.crop = {
					x: Math.round(box.x / scale),
					y: Math.round(box.y / scale),
					width: Math.round(box.width / scale),
					height: Math.round(box.height / scale)
				};
			}
			this.last_response_sent = Date.now();
		}
		try {
			const body = new FormData();
			body.append('response', JSON.stringify(msg));
			api.fetchApi('/instaraw/interactive_message', { method: 'POST', body });
			Log.message_out(msg);
		} catch (e) {
			Log.error(e);
		} finally {
			if (!keep_open) this.close();
		}
	}

	send_current_state() {
		if (this.state == State.TEXT) {
			this._send_response();
		} else {
			this._send_response();
		}
	}

	send_cancel() {
		this._send_response({ special: CANCEL });
	}

	request_reset() {
		this._send_response({ special: REQUEST_RESHOW }, true);
	}

	close() {
		this.state = State.INACTIVE;
		this.render();
	}

	maybe_play_sound() {
		// Updated settings key
		if (app.ui.settings.getSettingValue('INSTARAW.Interactive.PlaySound')) this.audio.play();
	}

	handle_message(message) {
		Log.message_in(message);
		Log.log(this._handle_message(message, false));
		this.render();
	}

	handle_deferred_message(e) {
		Log.message_in(this.saved_message, '(deferred)');
		Log.log(this._handle_message(this.saved_message, true));
		this.render();
	}

	autosend() {
		// Updated settings key
		return app.ui.settings.getSettingValue('INSTARAW.Interactive.AutosendIdentical') && this.allsame;
	}

	on_new_node(nd) {
		this.node = nd;
		const fp = this.floater_position();
		if (fp) this.floating_window.move_to(fp.x, fp.y, true);
		const tp = this.tiny_position();
		if (tp) this.tiny_window.move_to(tp.x, tp.y, true);
	}

	find_node(uid) {
		const bits = uid.split(':');
		if (bits.length == 1) {
			return app.graph._nodes_by_id[uid];
		} else {
			var graph = app.graph;
			var node;
			bits.forEach((bit) => {
				node = graph._nodes_by_id[bit];
				graph = node.subgraph;
			});
		}
		return node;
	}

	_handle_message(message, using_saved) {
		const detail = message.detail;
		const uid = detail.uid;
		const the_node = this.find_node(uid);

		if (this.node != the_node) this.on_new_node(the_node);

		if (!this.node) return console.log(`Message was for ${uid} which doesn't exist`);
		if (this.node._ni_widget?.value != message.detail.unique) return console.log(`Message unique id wasn't mine`);

		if (detail.tick) {
			this.counter_text.innerText = `${detail.tick}s`;
			if (this.state == State.INACTIVE) this.request_reset();
			return;
		}

		if (detail.timeout) {
			this.close();
			return `Timeout`;
		}

		if (this.handling_message) return `Ignoring message because we're already handling a message`;

		this.set_title(this.node.title ?? 'INSTARAW Image Filter'); // Updated default title
		this.allsame = detail.allsame || false;
		if (detail.tip) this.tip_row.innerHTML = detail.tip.replace(/(?:\r\n|\r|\n)/g, '<br/>');
		else this.tip_row.innerHTML = '';

		// Updated settings key
		if (
			this.state == State.INACTIVE &&
			app.ui.settings.getSettingValue('INSTARAW.Interactive.SmallWindow') &&
			!using_saved &&
			!this.autosend()
		) {
			this.state = State.TINY;
			this.saved_message = message;
			this.tiny_image.src = get_full_url(message.detail.urls[message.detail.urls.length - 1]);
			this.maybe_play_sound();
			return `Deferring message and showing small window`;
		}

		try {
			this.handling_message = true;
			this.n_extras = detail.extras ? message.detail.extras.length : 0;
			this.extras_row.innerHTML = '';
			for (let i = 0; i < this.n_extras; i++) {
				create('input', 'extra', this.extras_row, { value: detail.extras[i] });
			}

			if (!using_saved && !this.autosend()) this.maybe_play_sound();

			if (detail.interactive_crop) {
				this.handle_crop(detail);
			} else if (detail.maskedit) {
				// existing logic
				this.handle_maskedit(detail);
			} else if (detail.urls) {
				// existing logic
				this.handle_urls(detail);
			}
		} finally {
			this.handling_message = false;
		}
	}

	window_not_showing(uid) {
		const node = this.find_node(uid);
		return (
			(POPUP_NODES.includes(node.type) && this.classList.contains('hidden')) ||
			(MASK_NODES.includes(node.type) && !mask_editor_showing())
		);
	}

	set_title(title) {
		this.floating_window.set_title(title);
		var pos = this.floater_position();
		if (pos) this.floating_window.move_to(pos.x, pos.y);
		pos = this.tiny_position();
		if (pos) this.tiny_window.move_to(pos.x, pos.y);
		this.tiny_window.set_title(title);
	}

	handle_maskedit(detail) {
		this.state = State.MASK;

		this.node.imgs = [];
		detail.urls.forEach((url, i) => {
			this.node.imgs.push(new Image());
			this.node.imgs[i].src = api.apiURL(
				`/view?filename=${encodeURIComponent(url.filename)}&type=${url.type}&subfolder=${url.subfolder}`
			);
		});
		ComfyApp.copyToClipspace(this.node);
		ComfyApp.clipspace_return_node = this.node;
		ComfyApp.open_maskeditor();
		this.seen_editor = false;
		setTimeout(this.wait_while_mask_editing.bind(this), 200);
	}

	wait_while_mask_editing() {
		if (!this.seen_editor && mask_editor_showing()) {
			mask_editor_listen_for_cancel(this.send_cancel.bind(this));
			this.render();
			this.seen_editor = true;
		}

		if (mask_editor_showing()) {
			setTimeout(this.wait_while_mask_editing.bind(this), 100);
		} else {
			this._send_response({ masked_image: this.extract_filename(this.node.imgs[0].src) });
		}
	}

	extract_filename(url_string) {
		return new URL(url_string).searchParams.get('filename');
	}

	handle_urls(detail) {
		this.video_frames = detail.video_frames || 1;

		if (this.autosend()) {
			return this._send_response({ selection: [0] });
		}

		this.autozoom_pending = false;
		if (detail.text != null) {
			this.state = State.TEXT;
			this.text_edit.value = detail.text;
			if (detail.textareaheight) this.text_edit.style.height = `${detail.textareaheight}px`;
		} else {
			// Updated settings key
			if (
				this.state != State.FILTER &&
				this.state != State.ZOOMED &&
				app.ui.settings.getSettingValue('INSTARAW.Interactive.StartZoomed') != 0
			) {
				this.autozoom_pending = true;
			}
			this.state = State.FILTER;
		}

		this.n_images = detail.urls?.length;
		this.laidOut = -1;
		this.picked = new Set();
		if (this.n_images == 1) this.picked.add('0');

		this.grid.innerHTML = '';
		this.overlaygrid.innerHTML = '';
		var latestImage = null;

		detail.urls.forEach((url, i) => {
			console.log(url);
			if (i % this.video_frames == 0) {
				const thisImage = create('img', null, this.grid, { src: get_full_url(url) });
				latestImage = thisImage;
				latestImage.onload = this.layout.bind(this);
				latestImage.image_index = i / this.video_frames;
				latestImage.addEventListener('mouseover', (e) => this.on_mouse_enter(thisImage));
				latestImage.addEventListener('mouseout', (e) => this.on_mouse_out(thisImage));
				latestImage.frames = [get_full_url(url)];
			} else {
				latestImage.frames.push(get_full_url(url));
			}
			if (detail.mask_urls) {
				create('img', null, this.overlaygrid, { src: get_full_url(detail.mask_urls[i]) });
			}
		});

		this.layout();

		if (this.video_frames > 1) {
			this.frame = 0;
			setTimeout(this.advance_videos.bind(this), 1000);
		}
	}

	advance_videos() {
		if (this.state == State.INACTIVE) return;

		this.frame = (this.frame + 1) % this.video_frames;
		Array.from(this.grid.children).forEach((img) => {
			img.src = img.frames[this.frame];
		});

		// Updated settings key
		const fps = app.ui.settings.getSettingValue('INSTARAW.Interactive.FPS');
		const delay = fps > 0 ? 1000 / fps : 1000;
		setTimeout(this.advance_videos.bind(this), delay);
	}

	on_mouse_enter(img) {
		this.mouse_is_over = img;
		this.redraw();
	}

	on_mouse_out(img) {
		this.mouse_is_over = null;
		this.redraw();
	}

	zoom_auto() {
		this.autozoom_pending = false;
		// Updated settings key
		const startZoomed = app.ui.settings.getSettingValue('INSTARAW.Interactive.StartZoomed');
		if (startZoomed == 1) {
			this.zoomed_image_holder = this.grid.firstChild;
		} else if (startZoomed == -1) {
			this.zoomed_image_holder = this.grid.lastChild;
		} else {
			return;
		}
		if (this.zoomed_image_holder.image_index >= 0) {
			this.state = State.ZOOMED;
			return this.show_zoomed();
		}
	}
	zoom_next() {
		this.zoomed_image_holder = this.zoomed_image_holder.nextSibling || this.zoomed_image_holder.parentNode.firstChild;
		this.show_zoomed();
	}
	zoom_prev() {
		this.zoomed_image_holder =
			this.zoomed_image_holder.previousSibling || this.zoomed_image_holder.parentNode.lastChild;
		this.show_zoomed();
	}
	click_zoomed() {
		const fake_event = { target: this.zoomed_image_holder };
		this.on_click(fake_event);
		this.show_zoomed();
	}
	show_zoomed() {
		this.zoomed_image.src = this.zoomed_image_holder.src;
		return this.render();
	}
	eat_event(e) {
		e.stopPropagation();
		e.preventDefault();
	}

	on_key_press(e) {
		if (document.activeElement?.type == 'text' || document.activeElement?.type == 'textarea') {
			if (this.floating_window.contains(document.activeElement) || this.contains(document.activeElement)) return;
		}
		if (this.state != State.INACTIVE && this.state != State.TINY) {
			this.eat_event(e);
		}
	}

	on_key_down(e) {
		if (document.activeElement?.type == 'text' || document.activeElement?.type == 'textarea') {
			if (this.floating_window.contains(document.activeElement) || this.contains(document.activeElement)) return;
			if (this.state == State.INACTIVE && this.state == State.TINY) return;
		}
		if (this.state == State.FILTER || this.state == State.TEXT) {
			if (e.key == 'Enter') {
				this.send_current_state();
				return this.eat_event(e);
			}
			if (e.key == 'Escape') {
				this.send_cancel();
				return this.eat_event(e);
			}
			if (`${parseInt(e.key)}` == e.key) {
				this.select_unselect(parseInt(e.key));
				this.render();
				return this.eat_event(e);
			}
		}

		if (this.state == State.FILTER) {
			if (e.key == ' ' && this.mouse_is_over) {
				this.state = State.ZOOMED;
				this.zoomed_image_holder = this.mouse_is_over;
				this.eat_event(e);
				return this.show_zoomed();
			}
			if (e.key == 'a' && e.ctrlKey) {
				if (this.picked.size > this.n_images / 2) {
					this.picked.clear();
					console.log('unselect all');
				} else {
					this.picked.clear();
					for (var i = 0; i < this.n_images; i++) {
						this.picked.add(`${i}`);
					}
					console.log('select all');
				}
				this.eat_event(e);
				return this.redraw();
			}
		}

		if (this.state == State.ZOOMED) {
			if (e.key == ' ') {
				this.state = State.FILTER;
				this.zoomed_image_holder = null;
				this.eat_event(e);
				return this.render();
			} else if (e.key == 'ArrowUp') {
				this.click_zoomed();
				return this.eat_event(e);
			} else if (e.key == 'ArrowDown') {
				// select or unselect
			} else if (e.key == 'ArrowRight') {
				this.zoom_next();
				return this.eat_event(e);
			} else if (e.key == 'ArrowLeft') {
				this.zoom_prev();
				return this.eat_event(e);
			}
		}
	}

	select_unselect(n) {
		if (n < 0 || n > this.n_images) {
			return;
		}
		const s = `${n}`;
		// Updated settings key
		if (app.ui.settings.getSettingValue('INSTARAW.Interactive.ClickSends')) {
			this.picked.add(s);
			this._send_response();
		} else {
			if (this.picked.has(s)) this.picked.delete(s);
			else this.picked.add(s);
			this.redraw();
		}
	}

	on_click(e) {
		if (e.target.image_index != undefined) {
			this.select_unselect(e.target.image_index);
		}
	}

	layout(norepeat) {
		const box = this.grid.getBoundingClientRect();
		if (this.laidOut == box.width) return;

		const im_w = this.grid.firstChild.naturalWidth;
		const im_h = this.grid.firstChild.naturalHeight;

		if (!im_w || !im_h || !box.width || !box.height) {
			if (!norepeat) setTimeout(this.layout.bind(this), 100, [true]);
			return;
		} else {
			var best_scale = 0;
			var best_pick;
			var per_row;
			for (per_row = 1; per_row <= this.n_images; per_row++) {
				const rows = Math.ceil(this.n_images / per_row);
				const scale = Math.min(box.width / (im_w * per_row), box.height / (im_h * rows));
				if (scale > best_scale) {
					best_scale = scale;
					best_pick = per_row;
				}
			}
			this.per_row = best_pick;
			this.laidOut = box.width;
		}

		this.rows = Math.ceil(this.n_images / this.per_row);
		const w = box.width / this.per_row - GRID_IMAGE_SPACE;
		const h = box.height / this.rows - GRID_IMAGE_SPACE;

		var template_columns = '';
		for (let i = 0; i < this.per_row; i++) template_columns += ` ${w + GRID_IMAGE_SPACE}px`;
		var template_rows = '';
		for (let i = 0; i < this.rows; i++) template_rows += ` ${h + GRID_IMAGE_SPACE}px`;
		this.grid.style.gridTemplateColumns = template_columns;
		this.grid.style.gridTemplateRows = template_rows;
		this.overlaygrid.style.gridTemplateColumns = template_columns;
		this.overlaygrid.style.gridTemplateRows = template_rows;

		Array.from(this.grid.children).forEach((c, i) => {
			c.style.gridArea = `${Math.floor(i / this.per_row) + 1} / ${(i % this.per_row) + 1} /  auto / auto`;
		});
		Array.from(this.overlaygrid.children).forEach((c, i) => {
			c.style.gridArea = `${Math.floor(i / this.per_row) + 1} / ${(i % this.per_row) + 1} /  auto / auto`;
		});

		this.redraw();
		setTimeout(this.rescale_images.bind(this), 100);

		if (this.autozoom_pending) {
			this.zoom_auto();
		}
	}

	rescale_images() {
		const box = this.grid.getBoundingClientRect();
		const sub = this.grid.firstChild.getBoundingClientRect();
		const w_used = ((sub.width + GRID_IMAGE_SPACE) * this.per_row) / box.width;
		const h_used = ((sub.height + GRID_IMAGE_SPACE) * this.rows) / box.height;
		const could_zoom = 1.0 / Math.max(w_used, h_used);
		// Updated settings key
		if (could_zoom > 1 && app.ui.settings.getSettingValue('INSTARAW.Interactive.EnlargeSmall')) {
			Array.from(this.grid.children).forEach((img) => {
				img.style.width = `${sub.width * could_zoom}px`;
			});
			Array.from(this.overlaygrid.children).forEach((img) => {
				img.style.width = `${sub.width * could_zoom}px`;
			});
		}
	}

	redraw() {
		Array.from(this.grid.children).forEach((c, i) => {
			if (this.picked.has(`${i}`)) c.classList.add('selected');
			else c.classList.remove('selected');

			if (c == this.mouse_is_over) c.classList.add('hover');
			else c.classList.remove('hover');
		});
	}

	handle_crop(detail) {
        console.log('DEBUG: handle_crop received detail object:', detail);

		this.state = State.CROP;
		this.crop_image_bg.onload = () => {
			const img = this.crop_image_bg;
			const container = this.crop_container;
			const containerRect = container.getBoundingClientRect();

			// Calculate the scaled (displayed) size of the image to fit the container
			const imgAspect = img.naturalWidth / img.naturalHeight;
			const containerAspect = containerRect.width / containerRect.height;

			let displayWidth, displayHeight;
			if (imgAspect > containerAspect) {
				displayWidth = containerRect.width;
				displayHeight = displayWidth / imgAspect;
			} else {
				displayHeight = containerRect.height;
				displayWidth = displayHeight * imgAspect;
			}

			// Position and size the canvas overlay to perfectly match the displayed image
			this.crop_canvas.width = displayWidth;
			this.crop_canvas.height = displayHeight;
			this.crop_canvas.style.left = `${(containerRect.width - displayWidth) / 2}px`;
			this.crop_canvas.style.top = `${(containerRect.height - displayHeight) / 2}px`;

			// Store the scaling factor. This is crucial for converting UI coordinates back to original image coordinates.
			this.crop_data.scale = displayWidth / img.naturalWidth;

			// If a proposed crop was sent from the backend, use it to initialize the crop box.
			if (detail.proposed_crop) {
				const proposal = detail.proposed_crop;
				const scale = this.crop_data.scale;

				// Convert the proposed crop (in original image coordinates) to canvas coordinates
				this.crop_data.box = {
					x: proposal.x * scale,
					y: proposal.y * scale,
					width: proposal.width * scale,
					height: proposal.height * scale
				};
			} else {
				// If there's no proposal, start with no crop box.
				this.crop_data.box = null;
			}

			// Initial draw of the UI
			this.draw_crop_box();

			// Update the main render loop to enable/disable the 'Send' button correctly
			this.render();
		};

		// Set the source of the background image, which triggers the onload event above
		this.crop_image_bg.src = get_full_url(detail.urls[0]);
	}

	// --- NEW: Drawing logic for the crop box ---
    draw_crop_box() {
        if (this.state !== State.CROP) return;
        const ctx = this.crop_ctx;
        const canvas = this.crop_canvas;
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
        ctx.beginPath();
        ctx.rect(0, 0, canvas.width, canvas.height);
        if (this.crop_data.box) {
            const { x, y, width, height } = this.get_normalized_box(this.crop_data.box);
            ctx.rect(x, y, width, height);
        }
        ctx.fill('evenodd');
        if (this.crop_data.box) {
            const { x, y, width, height } = this.get_normalized_box(this.crop_data.box);
            ctx.strokeStyle = '#00ff00';
            ctx.lineWidth = 2;
            ctx.strokeRect(x, y, width, height);
            const handleSize = 8;
            ctx.fillStyle = '#00ff00';
            const handles = this.get_crop_handles();
            for (const handle in handles) {
                ctx.fillRect(handles[handle].x - handleSize / 2, handles[handle].y - handleSize / 2, handleSize, handleSize);
            }
        }
    }

    get_normalized_box(box) {
        if (!box) return null;
        return {
            x: box.width < 0 ? box.x + box.width : box.x,
            y: box.height < 0 ? box.y + box.height : box.y,
            width: Math.abs(box.width),
            height: Math.abs(box.height),
        };
    }

    get_crop_handles() {
        const box = this.get_normalized_box(this.crop_data.box);
        if (!box) return {};
        const { x, y, width, height } = box;
        return {
            'top-left': { x, y }, 'top-right': { x: x + width, y },
            'bottom-left': { x, y: y + height }, 'bottom-right': { x: x + width, y: y + height },
            'top': { x: x + width / 2, y }, 'bottom': { x: x + width / 2, y: y + height },
            'left': { x, y: y + height / 2 }, 'right': { x: x + width, y: y + height / 2 }
        };
    }

    get_crop_hit_area(localPos) {
        const handleSize = 12;
        const handles = this.get_crop_handles();
        for (const mode in handles) {
            const handle = handles[mode];
            if (Math.abs(localPos.x - handle.x) < handleSize / 2 && Math.abs(localPos.y - handle.y) < handleSize / 2) {
                return mode;
            }
        }
        const box = this.get_normalized_box(this.crop_data.box);
        if (box) {
            const { x, y, width, height } = box;
            if (localPos.x > x && localPos.x < x + width && localPos.y > y && localPos.y < y + height) {
                return 'move';
            }
        }
        return 'new';
    }

    update_crop_cursor(e) {
        if (this.state !== State.CROP || this.crop_data.active) return;
        const rect = this.crop_canvas.getBoundingClientRect();
        const localPos = { x: e.clientX - rect.left, y: e.clientY - rect.top };
        const mode = this.get_crop_hit_area(localPos);
        
        const cursorMap = {
            'top-left': 'nwse-resize', 'bottom-right': 'nwse-resize',
            'top-right': 'nesw-resize', 'bottom-left': 'nesw-resize',
            'top': 'ns-resize', 'bottom': 'ns-resize',
            'left': 'ew-resize', 'right': 'ew-resize',
            'move': 'move',
            'new': 'crosshair'
        };
        this.crop_canvas.style.cursor = cursorMap[mode] || 'crosshair';
    }

    on_crop_mouse_down(e) {
        if (this.state !== State.CROP) return;
        const rect = this.crop_canvas.getBoundingClientRect();
        const localPos = { x: e.clientX - rect.left, y: e.clientY - rect.top };
        this.crop_data.active = true;
        this.crop_data.mode = this.get_crop_hit_area(localPos);
        this.crop_data.startPos = localPos;
        this.crop_data.originalBox = this.crop_data.box ? { ...this.get_normalized_box(this.crop_data.box) } : null;
        if (this.crop_data.mode === 'new') {
            this.crop_data.box = { x: localPos.x, y: localPos.y, width: 0, height: 0 };
            this.crop_data.originalBox = null;
        }
    }

    on_document_mouse_move(e) {
        if (this.state !== State.CROP || !this.crop_data.active) return;
        const rect = this.crop_canvas.getBoundingClientRect();
        let localPos = { x: e.clientX - rect.left, y: e.clientY - rect.top };
        
        localPos.x = Math.max(0, Math.min(localPos.x, this.crop_canvas.width));
        localPos.y = Math.max(0, Math.min(localPos.y, this.crop_canvas.height));

        const dx = localPos.x - this.crop_data.startPos.x;
        const dy = localPos.y - this.crop_data.startPos.y;
        const { mode, originalBox } = this.crop_data;
        let { x, y, width, height } = this.crop_data.box;

        if (mode === 'new') {
            width = localPos.x - x;
            height = localPos.y - y;
        } else if (mode === 'move') {
            x = originalBox.x + dx;
            y = originalBox.y + dy;
        } else if (mode.includes('right')) {
            width = originalBox.width + dx;
        } else if (mode.includes('left')) {
            x = originalBox.x + dx;
            width = originalBox.width - dx;
        }

        if (mode.includes('bottom')) {
            height = originalBox.height + dy;
        } else if (mode.includes('top')) {
            y = originalBox.y + dy;
            height = originalBox.height - dy;
        }

        this.crop_data.box = { x, y, width, height };
        this.draw_crop_box();
    }

    on_crop_mouse_move(e) {
        this.update_crop_cursor(e);
    }

    on_crop_mouse_up(e) {
        if (this.state !== State.CROP || !this.crop_data.active) return;
        this.crop_data.active = false;
        
        if (this.crop_data.box) {
            this.crop_data.box = this.get_normalized_box(this.crop_data.box);
        }

        this.draw_crop_box();
        this.render();
    }
}

customElements.define('instaraw-imgae-filter-popup', Popup); // Renamed custom element

export const popup = new Popup();
