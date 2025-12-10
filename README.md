# VibeVoice Realtime TTS Extension

Lightweight realtime TTS for text generated in oobabooga's text-generation-webui using the VibeVoice streaming model.

## Quick features

- Streams audio while text is produced by the model
- Toggle on/off from the UI and refresh accumulated audio for playback

## Installation

1. **Important**: Get text-generation-webui text streaming PR ([#7348](https://github.com/oobabooga/text-generation-webui/pull/7348)):

    If you installed text-generation-webui via Git:

    ```bash
    cd path/to/text-generation-webui
    git fetch origin pull/7348/head:pr/streaming
    git switch pr/streaming
    ```

    If it errors, or you installed via a different method, you need a fresh install:

    ```bash
    git clone https://github.com/oobabooga/text-generation-webui.git
    cd text-generation-webui
    git fetch origin pull/7348/head:pr/streaming
    git switch pr/streaming
    ```

2. Build and install the [VibeVoice](https://github.com/microsoft/VibeVoice/blob/main/docs/vibevoice-realtime-0.5b.md#installation) package:

    ```bash
    git clone https://github.com/microsoft/VibeVoice.git
    cd VibeVoice
    pip install -e .
    ```

    Ensure required voice preset `.pt` files are present (VibeVoice demo voices directory). These will be fetched by the extension automatically.

3. Install the extension:

    ```bash
    cd path/to/text-generation-webui/extensions
    git clone https://github.com/Th-Underscore/vibevoice_realtime.git
    ```

    _(Optional, just for confirmation)_

    ```bash
    pip install -r vibevoice_realtime/requirements.txt
    ```

4. Launch the web UI with the extension:

    ```bash
    cd path/to/text-generation-webui
    python server.py --extensions vibevoice_realtime
    ```

    Or enable it in "Session → Extensions & flags" in text-generation-webui.

## Usage

1. Open text-generation-webui and go to the "Text generation" tab
2. Scroll down and enable "VibeVoice TTS"
3. Generate text! Audio is synthesized in realtime for emitted text chunks

<!-- Remaining configuration, implementation details and troubleshooting were removed for conciseness. See `script.py` for runtime options and voice preset locations. -->

## To-Do

- Refactor into multiple files for maintainability
- Ignore \<think> and "ASSISTANT:" (parse prompt template?)
- Stop on interruption (e.g. user input mid-stream)
- "Stop/Pause Generation/Stream" button
- Save as persistent attachment to assistant message
- Port-forwardable WebSocket (UPnP?)
- Detect 2nd difference between consecutive EOS probability tensors instead of using fixed threshold
- Restart on "Continue"
- "Generate audio from text" field
