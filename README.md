# VibeVoice Realtime TTS Extension

Lightweight realtime TTS for text generated in the text-generation-webui using the VibeVoice streaming model.

## Quick features

- Streams audio while text is produced by the model
- Toggle on/off from the UI and refresh accumulated audio for playback

## Installation

1. Build and install the [VibeVoice](https://github.com/microsoft/VibeVoice/blob/main/docs/vibevoice-realtime-0.5b.md#installation) package:

    ```bash
    git clone https://github.com/microsoft/VibeVoice.git
    pip install -e VibeVoice
    ```

    Ensure required voice preset `.pt` files are present (VibeVoice demo voices directory). These will be fetched by the extension automatically.

2. Launch the web UI with the extension:

    ```bash
    python server.py --extensions vibevoice_realtime
    ```

    Or enable it in "Session → Extensions & flags" in text-generation-webui.

## Usage

1. Open text-generation-webui and go to the "Text generation" tab
2. Scroll down and enable "VibeVoice TTS"
3. Generate text — Audio is synthesized in realtime for emitted text chunks

<!-- Remaining configuration, implementation details and troubleshooting were removed for conciseness. See `script.py` for runtime options and voice preset locations. -->

## To-Do

- Refactor into multiple files for maintainability
