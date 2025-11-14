# Browser-Based Model Inference

Run your fine-tuned Gemma3-270M model entirely in the browser using WebAssembly and wllama.

## Quick Start

### 1. GitHub Pages

This app is hosted on [GitHub Pages](https://vanpelt.github.io/summarizer/)!

### 2. Start a Local HTTP Server (optional)

**IMPORTANT**: You must access the site from localhost for browser security to function properly.

```bash
# From the project root directory
uv run python -m http.server 8000
# Or if you're into node
npx http-server -p 8000
```

**Supported Browsers**:
- ✅ Chrome/Edge 113+ (Recommended)
- ✅ Firefox 115+
- ✅ Safari 15.2+

### 3. Model Loading

The app intelligently handles model loading based on your device and connection:

**Desktop/Fast Connection:**
- Model loads automatically (242MB)
- Watch the progress bar
- Takes 30-60 seconds on first load
- Cached in browser (OPFS) for instant subsequent loads

**Mobile/Cellular/Slow Connection:**
- Green "Download Model" button appears
- Click to manually start download
- Prevents unwanted data usage on metered connections
- Respects browser "Save Data" mode

**Subsequent Visits:**
- Model loads instantly from cache (all devices)
- No re-download needed

### 4. Generate Text

1. Enter your prompt in the text area (enabled after model loads)
2. Adjust parameters (temperature, top-p, max tokens) if desired
3. Click "Generate" or press Enter
4. Watch tokens stream in real-time!

## Features

- **Entirely Client-Side**: No server needed after initial page load
- **Smart Loading**: Auto-detects device type and network speed
  - Desktop/WiFi: Auto-loads model
  - Mobile/Cellular: Requires manual confirmation
  - Respects "Save Data" browser setting
- **Prefill**: The app automatically computes the KV Cache ahead of time to make inference faster.
- **Streaming Output**: See tokens generated in real-time
- **Configurable**: Adjust temperature, top-p, max tokens
- **Persistent Caching**: Model cached in browser OPFS for instant reloads
- **Multi-threaded**: Uses all CPU cores for best performance

## Model Information

- **Model**: gemma3-270m-synthetic-v11
- **Huggingface**: [vanpelt/summarizer](https://huggingface.co/vanpelt/summarizer)
- **Format**: GGUF (Q4_K_M quantization)
- **Size**: 242MB
- **Architecture**: Gemma3 (Google)
- **Context Length**: 2048 tokens
- **Trained For**: JSON generation (task titles, branch names)

## Configuration

### Inference Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| **Temperature** | 0.70 | 0.0 - 2.0 | Controls randomness. Higher = more creative |
| **Top P** | 0.95 | 0.0 - 1.0 | Nucleus sampling. Lower = more focused |
| **Max Tokens** | 256 | 1 - 2048 | Maximum tokens to generate |

## Performance

### Expected Performance

| Hardware | Speed | Notes |
|----------|-------|-------|
| **Modern Desktop** | 10-20 tokens/sec | Best experience |
| **Laptop** | 5-15 tokens/sec | Good for most tasks |
| **Older Hardware** | 2-5 tokens/sec | Usable but slower |

### Optimization Tips

1. **Use Chrome/Edge**: Best WebAssembly performance
2. **Close Other Tabs**: Free up memory and CPU
3. **Reduce Context**: Lower `n_ctx` to 1024 for faster inference
4. **Fewer Threads**: Try `n_threads: 2` if CPU is thermal throttling
5. **Lower Max Tokens**: Generate shorter responses

## Troubleshooting

### Model Won't Load

**Error**: "Failed to fetch" or CORS errors

**Solution**:
- ✅ Ensure using HTTP server (not file://)
- ✅ Verify `summarizer-q4_k_m-v1.gguf` exists in web/ directory
- ✅ Check browser console for detailed error messages
- ✅ Ensure you're accessing via `http://localhost:8000/web/index.html`

### Out of Memory

**Issue**: Browser crashes or "Out of memory" errors

**Solutions**:
1. Close other tabs and applications
2. Use a machine with more RAM (recommend 8GB+ free)
3. Try Incognito/Private mode (no extensions)
4. Use a smaller model or different quantization

### Libraries Used

- **wllama**: WebAssembly port of llama.cpp
  - Version: 1.7.1
  - Repository: https://github.com/ngxson/wllama
  - License: MIT

- **Tailwind CSS**: Utility-first CSS framework
  - Loaded via CDN
  - Version: Latest
  - Website: https://tailwindcss.com

### Browser Requirements

**Required Features**:
- ES6+ JavaScript (modules, async/await)
- WebAssembly (WASM)
- SharedArrayBuffer (for multi-threading)
- IndexedDB (for model caching)
- Web Workers (for non-blocking inference)

**Security Context**:
- Requires "secure context" (HTTPS or localhost)
- Cross-Origin-Opener-Policy and Cross-Origin-Embedder-Policy headers may be needed for SharedArrayBuffer in production

### Model Format

The GGUF format is specifically designed for efficient inference:

- **Quantization**: Q4_K_M (4-bit with K-quant mixed precision)
- **Original Size**: ~1.08GB (FP16)
- **Quantized Size**: ~150MB (Q4_K_M)
- **Accuracy Loss**: Minimal (~1-2% perplexity increase)
- **Speed Gain**: 4-6x faster than FP16

## Customization

### Change Model

To use a different GGUF model:

1. Replace `summarizer-q4_k_m-v1.gguf` in the web/ directory with your model
2. Update the `MODEL_PATH` constant in `index.html`:
```javascript
const MODEL_PATH = 'your-model-name.gguf';
```

3. Update the model info display in the HTML (search for "gemma3-270m-synthetic-v11")

### Change UI Theme

Modify Tailwind classes in `index.html`:

- Background: `bg-gray-50` → `bg-blue-50`
- Buttons: `bg-blue-600` → `bg-purple-600`
- Text: `text-gray-800` → `text-slate-900`

### Add Custom Stop Tokens

Edit the `generateText()` function:

```javascript
for await (const chunk of wllama.createCompletion(formattedPrompt, {
    // ... other options ...
    stopTokens: ['<end_of_turn>', '<eos>', 'CUSTOM_STOP'],
})) {
```

### Change Chat Template

Modify the prompt formatting in `generateText()`:

```javascript
// Current (Gemma3 format)
const formattedPrompt = `<start_of_turn>user
${prompt}<end_of_turn>
<start_of_turn>model
`;

// For other models, adjust as needed
const formattedPrompt = `[INST] ${prompt} [/INST]`;  // Llama
const formattedPrompt = `### Instruction:\n${prompt}\n\n### Response:\n`;  // Alpaca
```

## Development

### Debugging

Enable verbose logging by opening browser DevTools (F12) and checking the Console tab.

Add debug statements:
```javascript
console.log('Model loaded:', isModelLoaded);
console.log('Generated token:', chunk);
```

### Testing

Test with various prompts:

- **Simple**: "Hello, how are you?"
- **JSON Task**: "Create a task for implementing user authentication"
- **Long Form**: "Write a detailed explanation of..."
- **Edge Cases**: Empty string, very long prompts, special characters

### Building from Source

The HTML is self-contained, but if you want to modify wllama:

1. Clone wllama: `git clone https://github.com/ngxson/wllama`
2. Build: `npm run build`
3. Update CDN paths in HTML to local paths
4. Serve both HTML and wllama build artifacts

## Comparison: Browser vs Server

### Browser Inference (This Implementation)

**Pros**:
- ✅ No server required
- ✅ Runs offline (after initial load)
- ✅ Privacy (no data leaves device)
- ✅ Zero server costs
- ✅ Self-contained HTML file

**Cons**:
- ❌ Slower than GPU inference
- ❌ Requires modern browser
- ❌ Large initial download
- ❌ Limited to smaller models (<1GB)
- ❌ CPU-only (no WebGPU support yet in wllama)

### Server Inference (vLLM/Ollama)

**Pros**:
- ✅ Much faster (GPU acceleration)
- ✅ Supports larger models
- ✅ Better for batch processing
- ✅ More control over infrastructure

**Cons**:
- ❌ Requires server running
- ❌ No offline support
- ❌ Privacy concerns (data sent to server)
- ❌ Server maintenance overhead

## Alternative: Ollama Backend

If browser inference is too slow, create a hybrid version:

### Option 1: Ollama API

Replace wllama with fetch calls to Ollama:

```javascript
const response = await fetch('http://localhost:11434/api/generate', {
    method: 'POST',
    body: JSON.stringify({
        model: 'gemma3-270m-synthetic-v11',
        prompt: prompt,
        stream: true
    })
});
```

See the parent directory's `justfile` for Ollama setup:
```bash
just serve  # Start Ollama server
```

### Option 2: Existing FastAPI

Use the project's existing FastAPI server:

```javascript
const response = await fetch('http://localhost:8080/generate', {
    method: 'POST',
    body: JSON.stringify({ prompt: prompt })
});
```

Start server:
```bash
just serve-api  # From project root
```

## Resources

- **wllama GitHub**: https://github.com/ngxson/wllama
- **llama.cpp**: https://github.com/ggerganov/llama.cpp
- **GGUF Format**: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
- **Gemma Models**: https://ai.google.dev/gemma
- **WebAssembly**: https://webassembly.org/

## License

This web interface is part of the summary-finetune project. Check the parent directory's LICENSE file.

## Support

For issues:
1. Check this README's Troubleshooting section
2. Open browser DevTools (F12) and check Console for errors
3. Verify model file exists and is accessible
4. Try the Ollama backend alternative if browser inference fails

## Credits

- **Model**: Fine-tuned gemma3-270m-synthetic-v11
- **Inference Engine**: wllama by @ngxson
- **Base Model**: Google Gemma3
- **UI Framework**: Tailwind CSS
