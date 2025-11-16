# Reality Prompt Generator - Debugging Guide

## 🎯 Overview

This guide explains how to debug API key retrieval issues in the Reality Prompt Generator node. The system has comprehensive logging to help identify exactly where things are going wrong when connecting Primitive String nodes for API keys.

---

## 🔧 Problem Being Solved

The Reality Prompt Generator node has two inputs:
- `gemini_api_key` (STRING input)
- `grok_api_key` (STRING input)

These are **inputs** on the Reality Prompt Generator node, not widgets. The actual API key values come from **connected nodes** (like Primitive String nodes), which have the widget containing the actual key value.

When you click "Generate Creative Prompts" or "Generate Character Prompts", the code needs to:
1. Find the input on the Reality Prompt Generator node
2. Follow the connection (link) to the connected node
3. Read the widget value from that connected node
4. Use that API key in the API request

---

## 🔍 Debugging Features Added

### 1. **Comprehensive Graph Traversal Logging**

The `getFinalInputValue()` function now logs every step:

```javascript
[RPG] 🔍 Getting input value for: "grok_api_key"
[RPG] Node ID: 42, Node Type: INSTARAW_RealityPromptGenerator
[RPG] Node has 7 inputs total
[RPG] Node has 10 widgets total
[RPG] All inputs on this node: [...]
[RPG] All widgets on this node: [...]
[RPG] ✅ Input found: {name: "grok_api_key", type: "STRING", link: 123, has_link: true}
[RPG] ✅ Link found: {id: 123, origin_id: 38, target_id: 42, ...}
[RPG] ✅ Origin node found: {id: 38, type: "PrimitiveNode", ...}
[RPG] Origin node widgets: [{index: 0, name: "value", type: "string", value: "[STRING LENGTH: 64]"}]
[RPG] ✅ Returning value from origin node widget[0]: [STRING LENGTH: 64, Preview: xai-abc123...]
```

**What to look for in these logs:**
- ❌ `No inputs found on node` → The Reality Prompt Generator node wasn't initialized with inputs
- ❌ `Input "grok_api_key" not found` → The input name doesn't match
- ❌ `Input is not connected (link is null)` → No Primitive String node connected
- ❌ `Link not found in app.graph.links` → Graph is corrupted or link ID is invalid
- ❌ `Origin node not found` → Connected node was deleted or graph is inconsistent
- ❌ `Origin node has no widgets` → Connected node doesn't have the expected structure

### 2. **Generation Function Logging**

When you click the generate buttons, you'll see:

```javascript
[RPG] 🎨 Generate Creative Prompts - START
[RPG] Timestamp: 2025-01-16T10:30:00.000Z
[RPG] Configuration: {generationCount: 5, model: "grok-4-fast-reasoning", ...}
[RPG] About to retrieve API keys from connected nodes...
  [RPG] 🔍 Getting input value for: "gemini_api_key"
  ... (detailed traversal logs)
  [RPG] 🔍 Getting input value for: "grok_api_key"
  ... (detailed traversal logs)
[RPG] ✅ Resolved Gemini API Key: [KEY PRESENT - Length: 58]
[RPG] ✅ Resolved Grok API Key: [KEY PRESENT - Length: 64]
[RPG] Window fallback keys: {INSTARAW_GEMINI_KEY: "[EMPTY]", ...}
[RPG] Making API request to /instaraw/generate_creative_prompts
[RPG] Request payload: {has_gemini_key: true, has_grok_key: true, ...}
[RPG] API response status: 200 OK
[RPG] API response parsed: {success: true, prompts: [...]}
[RPG] ✅ Success! Generated 5 prompts
[RPG] Generate Creative Prompts - END
```

### 3. **Helper Functions for Browser Console**

You can manually test from the browser console:

```javascript
// Find your Reality Prompt Generator node
const rpgNode = app.graph._nodes.find(n => n.type === "INSTARAW_RealityPromptGenerator");

// Test getting an API key
rpgNode.debugGetInput("grok_api_key");
rpgNode.debugGetInput("gemini_api_key");

// Dump the entire graph structure
rpgNode.debugDumpGraph();
```

The `debugDumpGraph()` function shows:
- Current node structure (inputs, widgets, properties)
- All nodes in the graph (table format)
- All links in the graph (table format)
- Nodes connected to this node (detailed view)

---

## 📋 How to Use This for Debugging

### Step 1: Open Browser Console
Press **F12** (or Cmd+Option+I on Mac) to open DevTools, then click the "Console" tab.

### Step 2: Reproduce the Issue
1. Connect a Primitive String node to the `grok_api_key` input
2. Enter your API key in the Primitive String node
3. Click "Generate Creative Prompts" or "Generate Character Prompts"

### Step 3: Analyze the Logs

Look for these patterns:

#### ✅ **Success Pattern**
```
[RPG] 🔍 Getting input value for: "grok_api_key"
[RPG] ✅ Input found: {has_link: true}
[RPG] ✅ Link found: {origin_id: 38}
[RPG] ✅ Origin node found: {type: "PrimitiveNode"}
[RPG] ✅ Returning value from origin node widget[0]: [STRING LENGTH: 64]
[RPG] ✅ Resolved Grok API Key: [KEY PRESENT - Length: 64]
```

#### ❌ **Failure Patterns**

**Pattern 1: No inputs on node**
```
[RPG] ⚠️ No inputs found on node!
```
**Fix:** The node wasn't initialized correctly. Refresh the page or recreate the node.

**Pattern 2: Input not connected**
```
[RPG] ⚠️ Input "grok_api_key" is not connected (link is null)
```
**Fix:** Connect a Primitive String node to the `grok_api_key` input.

**Pattern 3: Origin node has no widgets**
```
[RPG] ⚠️ Origin node has no widgets or properties with value
[RPG] Origin node full structure: {...}
```
**Fix:** The connected node isn't a Primitive node or doesn't have a value widget. Check the connected node type.

**Pattern 4: No API keys at all**
```
[RPG] ❌ NO API KEYS FOUND!
[RPG] To fix this, connect a Primitive String node to either:
[RPG]   - gemini_api_key input
[RPG]   - grok_api_key input
```
**Fix:** You need to connect at least one API key source.

### Step 4: Copy and Share Logs

If the issue persists:
1. **Copy all console logs** starting from the generate button click
2. **Take a screenshot** of your node graph showing the connections
3. **Share both** so we can diagnose the exact issue

The logs are grouped (you can collapse/expand them), so you can easily copy entire sections.

---

## 🛠️ Advanced Debugging

### Manually Inspect Node Structure

```javascript
const rpgNode = app.graph._nodes.find(n => n.type === "INSTARAW_RealityPromptGenerator");

// Check inputs
console.log("Inputs:", rpgNode.inputs);

// Find the grok_api_key input
const grokInput = rpgNode.inputs.find(i => i.name === "grok_api_key");
console.log("Grok Input:", grokInput);

// Check if it has a link
if (grokInput && grokInput.link) {
    const link = app.graph.links[grokInput.link];
    console.log("Link:", link);

    const originNode = app.graph.getNodeById(link.origin_id);
    console.log("Origin Node:", originNode);
    console.log("Origin Node Widgets:", originNode.widgets);
}
```

### Check Window Fallback Keys

```javascript
console.log("Gemini Key in window:", window.INSTARAW_GEMINI_KEY);
console.log("Grok Key in window:", window.INSTARAW_GROK_KEY);
```

If you want to set fallback keys globally:
```javascript
window.INSTARAW_GEMINI_KEY = "your-gemini-key";
window.INSTARAW_GROK_KEY = "your-grok-key";
```

---

## 📝 Expected Workflow

### Correct Setup:
1. Add **INSTARAW Reality Prompt Generator** node to canvas
2. Add **Primitive** node (ComfyUI built-in)
3. Set Primitive type to "STRING"
4. Enter your API key in the Primitive node's widget
5. Connect Primitive node output → Reality Prompt Generator `grok_api_key` input
6. Click "Generate Creative Prompts"
7. Check console for success logs

### What Should Happen:
```
[RPG] 🎨 Generate Creative Prompts - START
[RPG] About to retrieve API keys from connected nodes...
  [RPG] 🔍 Getting input value for: "gemini_api_key"
  [RPG] Input found...
  [RPG] Link found...
  [RPG] Origin node found: PrimitiveNode
  [RPG] ✅ Returning value from origin node widget[0]
[RPG] ✅ Resolved Grok API Key: [KEY PRESENT - Length: 64]
[RPG] Making API request...
[RPG] API response status: 200 OK
[RPG] ✅ Success! Generated 5 prompts
```

---

## 🐛 Common Issues & Solutions

| Issue | Symptom | Solution |
|-------|---------|----------|
| **No inputs on node** | `No inputs found on node` | Reload page or recreate node |
| **Wrong input name** | `Input "xxx" not found! Available inputs: [...]` | Check Python INPUT_TYPES matches JS code |
| **Not connected** | `Input is not connected (link is null)` | Connect Primitive String node |
| **Wrong node type** | `Origin node has no widgets` | Make sure connected node is Primitive/String type |
| **Empty key** | `[KEY PRESENT - Length: 0]` | Check Primitive node has value entered |
| **Link broken** | `Link not found in app.graph.links` | Reconnect the nodes |

---

## 📊 Log Legend

| Icon | Meaning |
|------|---------|
| 🔍 | Starting a search/lookup operation |
| ✅ | Success - found what we're looking for |
| ⚠️ | Warning - something is missing but we have a fallback |
| ❌ | Error - operation failed |
| 📊 | Data dump / comprehensive information |
| 🎨 | Creative prompt generation |
| 👤 | Character prompt generation |

---

## 🎯 Next Steps

Once you click the generate button:
1. Open browser console (F12)
2. Look for the grouped log sections
3. Expand the `[RPG] 🔍 Getting input value for: "grok_api_key"` section
4. Follow the path through the logs
5. Copy the logs and share them if you need help

The logs are designed to be self-explanatory and show exactly where in the graph traversal process things go wrong!
