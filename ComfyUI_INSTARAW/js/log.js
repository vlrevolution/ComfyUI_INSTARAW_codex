// ---
// Filename: ../ComfyUI_INSTARAW/js/log.js
// ---

import { app } from "../../scripts/app.js";

export class Log {
    static log(s) { if (s) console.log(s) }
    static error(e) { console.error(e) }
    static detail(s) {
        // Updated settings key
        if (app.ui.settings.getSettingValue("INSTARAW.Interactive.DetailedLogging")) Log.log(s)
    }
    static message_in(message, extra) {
        // Updated settings key
        if (!app.ui.settings.getSettingValue("INSTARAW.Interactive.DetailedLogging")) return
        if (message.detail && !message.detail.tick) Log.log(`--> ${JSON.stringify(message.detail)}` + (extra ? ` ${extra}` : ""))
        if (message.detail && message.detail.tick) Log.log(`--> tick`)
    }
    static message_out(response, extra) {
        // Updated settings key
        if (!app.ui.settings.getSettingValue("INSTARAW.Interactive.DetailedLogging")) return
        Log.log(`"<-- ${JSON.stringify(response)}` + (extra ? ` ${extra}` : ""))
    }
}