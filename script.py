import asyncio
import copy
import json
import numpy as np
import os
import threading
import time
import torch
import traceback
from concurrent.futures import Future
from pathlib import Path
from queue import Queue, Empty

import gradio as gr
from modules.logging_colors import logger

try:
    from vibevoice.modular.modeling_vibevoice_streaming_inference import (
        VibeVoiceStreamingForConditionalGenerationInference,
        TTS_TEXT_WINDOW_SIZE,
        TTS_SPEECH_WINDOW_SIZE,
        _update_model_kwargs_for_generation
    )
    from vibevoice.processor.vibevoice_streaming_processor import VibeVoiceStreamingProcessor
    from vibevoice.modular.modular_vibevoice_tokenizer import VibeVoiceTokenizerStreamingCache
    VIBEVOICE_AVAILABLE = True
except ImportError:
    VIBEVOICE_AVAILABLE = False
    TTS_TEXT_WINDOW_SIZE = 5
    TTS_SPEECH_WINDOW_SIZE = 6

SCRIPT_DIR = Path(__file__).parent
SETTINGS_FILE = SCRIPT_DIR.parent / "config.json"
TGWUI_ROOT = ([parent for parent in SCRIPT_DIR.parents if parent.name == "text-generation-webui"] or [os.getcwd()])[0]

params = {
    "display_name": "VibeVoice Realtime TTS",
    "is_tab": False,

    "disable_flash_attn": False,
    "cfg_scale": 1.5,
    "inference_steps": 5,
    "eos_threshold": 0.0007,
    "default_voice_preset": "en-Davis_man",
}

def load_custom_settings():
    if SETTINGS_FILE.exists():
        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                saved_settings: dict = json.load(f)
                # Only update keys that exist in file
                params.update(saved_settings)
                logger.debug(f"[VibeVoice] Loaded config: {saved_settings}")
        except Exception as e:
            logger.error(f"[VibeVoice] Failed to load config.json: {e}")

load_custom_settings()

# Global State
_tts_service = None
_generator_instance = None
_enable_tts = False
_websocket_server = None
_websocket_port = 0
_server_ready_event = threading.Event()
_audio_chunks_for_ui = []
_audio_chunks_lock = threading.Lock()

_tts_queue = Queue()
_last_seen_len = 0

# Track client completion for each stream
_completion_events: dict[str, threading.Event] = {}
_completion_lock = threading.Lock()
_stream_id_counter = 0
_stream_id_lock = threading.Lock()


class VibeVoiceService:
    def __init__(self, model_path, device="cuda"):
        self.model_path = model_path
        self.device = device
        self._torch_device = torch.device(device)
        self.processor = None
        self.model = None
        self.voice_presets = {}
        self._voice_cache = {}
        self.default_voice_key = None
        self.inference_steps = 5

    def load(self):
        # Clean up existing model if this is a reload
        if self.model is not None:
            logger.info("[VibeVoice] Unloading existing model...")
            del self.model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        logger.info(f"[VibeVoice] Loading model from {self.model_path}...")
        self.processor = VibeVoiceStreamingProcessor.from_pretrained(self.model_path)

        load_dtype = torch.bfloat16 if self.device == "cuda" else torch.float32

        # Determine attention implementation
        attn_impl = "sdpa"
        if self.device == "cuda" and not params.get("disable_flash_attn", False):
            attn_impl = "flash_attention_2"

        logger.info(f"[VibeVoice] Attempting to use attention implementation: {attn_impl}")

        try:
            self.model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                self.model_path,
                torch_dtype=load_dtype,
                device_map=self.device,
                attn_implementation=attn_impl
            )
        except Exception as e:
            # Fallback if FA2 fails (e.g. not installed or GPU incompatibility)
            if attn_impl == "flash_attention_2":
                logger.warning(f"[VibeVoice] Flash Attention 2 failed to load ({e}). Falling back to SDPA.")
                self.model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                    self.model_path,
                    torch_dtype=load_dtype,
                    device_map=self.device,
                    attn_implementation="sdpa"
                )
            else:
                raise e

        self.model.eval()

        self.model.model.noise_scheduler = self.model.model.noise_scheduler.from_config(
            self.model.model.noise_scheduler.config,
            algorithm_type="sde-dpmsolver++",
            beta_schedule="squaredcos_cap_v2",
        )
        self.model.set_ddpm_inference_steps(num_steps=self.inference_steps)
        self._load_voice_presets()
        logger.info(f"[VibeVoice] Ready. (Attention: {self.model.model.config._attn_implementation})")

    def _load_voice_presets(self):
        import os
        import vibevoice
        from pathlib import Path

        vibevoice_path = Path(vibevoice.__file__).parent.parent
        possible_paths = [
            SCRIPT_DIR / Path("VibeVoice/demo/voices/streaming_model"),
            Path(vibevoice_path, "demo/voices/streaming_model"),
            Path("VibeVoice/demo/voices/streaming_model"),
            Path("user_data/VibeVoice/voices")
        ]

        found_dir = None
        for p in possible_paths:
            if p.exists():
                found_dir = p
                break

        if found_dir:
            for pt in found_dir.glob("*.pt"):
                self.voice_presets[pt.stem] = pt
            logger.info(f"[VibeVoice] Loaded {len(self.voice_presets)} voices from {os.path.relpath(found_dir, TGWUI_ROOT)}")
        else:
            logger.error("[VibeVoice] Could not find voices directory!")

    def _ensure_voice_cached(self, key: str):
        if key not in self._voice_cache:
            path = self.voice_presets.get(key)
            if not path:
                key = next(iter(self.voice_presets))
                path = self.voice_presets[key]
            logger.info(f"[VibeVoice] Caching voice: {key}")
            self._voice_cache[key] = torch.load(path, map_location=self._torch_device, weights_only=False)
        return self._voice_cache[key]

    def set_voice_key(self, key: str):
        if key not in self.voice_presets:
            raise RuntimeError(f"[VibeVoice] Voice preset not found: {key}")
        self._ensure_voice_cached(key)
        self.default_voice_key = key
        logger.info(f"[VibeVoice] Default voice set to: {key}")


class VibeVoiceIncrementalGenerator:
    def __init__(self, service: VibeVoiceService, voice_key: str, cfg_scale=1.5):
        self.service = service
        self.model = service.model
        self.processor = service.processor
        self.device = service._torch_device
        self.cfg_scale = cfg_scale
        self.voice_key = voice_key

        raw_preset = self.service._ensure_voice_cached(voice_key)
        self.prefilled_outputs = copy.deepcopy(raw_preset)
        self.acoustic_cache = VibeVoiceTokenizerStreamingCache()

        self.lm_outputs = self.prefilled_outputs["lm"]
        self.tts_lm_outputs = self.prefilled_outputs["tts_lm"]
        self.neg_lm_outputs = self.prefilled_outputs["neg_lm"]
        self.neg_tts_lm_outputs = self.prefilled_outputs["neg_tts_lm"]

        self.token_buffer = []
        self.finished = False

        # Token IDs
        self.eos_id = self.processor.tokenizer.eos_token_id
        if self.eos_id is None:
            self.eos_id = self.processor.tokenizer.encode(" ", add_special_tokens=False)[0]

        self.pad_id = self.processor.tokenizer.encode(" ", add_special_tokens=False)[0]

        self.lm_state = self._init_state(self.lm_outputs)
        self.tts_lm_state = self._init_state(self.tts_lm_outputs)
        self.neg_lm_state = self._init_state(self.neg_lm_outputs)
        self.neg_tts_lm_state = self._init_state(self.neg_tts_lm_outputs)

    def _init_state(self, outputs):
        seq_len = outputs.last_hidden_state.shape[1]
        return {
            "past_key_values": outputs.past_key_values,
            "seq_len": seq_len,
            "attention_mask": torch.ones((1, seq_len), dtype=torch.long, device=self.device)
        }

    def push_text(self, text_fragment):
        if not text_fragment or self.finished:
            return

        new_ids = self.processor.tokenizer.encode(text_fragment, add_special_tokens=False)
        self.token_buffer.extend(new_ids)

        while len(self.token_buffer) >= TTS_TEXT_WINDOW_SIZE:
            window_ids = self.token_buffer[:TTS_TEXT_WINDOW_SIZE]
            self.token_buffer = self.token_buffer[TTS_TEXT_WINDOW_SIZE:]
            yield from self._generate_window(window_ids)
            if self.finished:
                break

    def flush(self):
        if self.finished:
            return

        if self.token_buffer:
            while len(self.token_buffer) < TTS_TEXT_WINDOW_SIZE:
                self.token_buffer.append(self.pad_id)
            yield from self._generate_window(self.token_buffer)
            self.token_buffer = []

        max_audio_windows = 1000
        eos_threshold = params.get("eos_threshold", 0.0007)
        for _ in range(max_audio_windows):
            if self.finished:
                break
            # eos_threshold *= 0.96
            yield from self._generate_window(None, eos_threshold)

    @torch.inference_mode()
    def _generate_window(self, token_ids: np.ndarray | None = None, eos_threshold=0.001):
        if self.finished:
            return

        device = self.device

        # --- Text Prefill ---
        if token_ids is not None and len(token_ids) > 0:
            cur_input_tts_text_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
            window_size = cur_input_tts_text_ids.shape[1]

            lm_past, lm_len = self.lm_state["past_key_values"], self.lm_state["seq_len"]
            lm_pos = torch.arange(lm_len, lm_len + window_size, device=device)
            lm_mask = torch.cat([self.lm_state["attention_mask"], torch.ones((1, window_size), dtype=torch.long, device=device)], dim=-1)
            outputs = self.model.forward_lm(
                input_ids=cur_input_tts_text_ids, attention_mask=lm_mask, past_key_values=lm_past, cache_position=lm_pos,
                return_dict=True, output_attentions=False, output_hidden_states=False,
            )
            self.lm_state.update({"past_key_values": outputs.past_key_values, "seq_len": lm_len + window_size, "attention_mask": lm_mask})

            tts_past, tts_len = self.tts_lm_state["past_key_values"], self.tts_lm_state["seq_len"]
            tts_pos = torch.arange(tts_len, tts_len + window_size, device=device)
            tts_mask = torch.cat([self.tts_lm_state["attention_mask"], torch.ones((1, window_size), dtype=torch.long, device=device)], dim=-1)
            self.tts_lm_outputs = self.model.forward_tts_lm(
                input_ids=cur_input_tts_text_ids, attention_mask=tts_mask, past_key_values=tts_past, cache_position=tts_pos,
                tts_text_masks=torch.ones_like(cur_input_tts_text_ids[:, -1:]), lm_last_hidden_state=outputs.last_hidden_state,
                return_dict=True, output_attentions=False, output_hidden_states=False,
            )
            self.tts_lm_state.update({"past_key_values": self.tts_lm_outputs.past_key_values, "seq_len": tts_len + window_size, "attention_mask": tts_mask})

        # --- Speech Generation ---
        diffusion_indices = torch.LongTensor([0]).to(device)
        latents_buffer = []  # Accumulate to decode in one batch

        for _ in range(TTS_SPEECH_WINDOW_SIZE):
            pos_cond = self.tts_lm_outputs.last_hidden_state[diffusion_indices, -1, :]
            neg_cond = self.neg_tts_lm_outputs.last_hidden_state[diffusion_indices, -1, :]

            speech_latent = self.model.sample_speech_tokens(pos_cond, neg_cond, cfg_scale=self.cfg_scale).unsqueeze(1)

            latents_buffer.append(speech_latent)

            acoustic_embed = self.model.acoustic_connector(speech_latent)
            dummy_id = torch.ones((1, 1), dtype=torch.long, device=device)

            t_past, t_len = self.tts_lm_state["past_key_values"], self.tts_lm_state["seq_len"]
            t_pos, t_mask = torch.arange(t_len, t_len+1, device=device), torch.cat([self.tts_lm_state["attention_mask"], torch.ones((1, 1), device=device)], dim=-1)
            self.tts_lm_outputs = self.model.forward_tts_lm(
                input_ids=dummy_id, attention_mask=t_mask, past_key_values=t_past, cache_position=t_pos,
                tts_text_masks=torch.zeros_like(dummy_id), lm_last_hidden_state=acoustic_embed, return_dict=True, output_attentions=False, output_hidden_states=False
            )
            self.tts_lm_state.update({"past_key_values": self.tts_lm_outputs.past_key_values, "seq_len": t_len+1, "attention_mask": t_mask})

            tts_eos_logits = torch.sigmoid(self.model.tts_eos_classifier(self.tts_lm_outputs.last_hidden_state[diffusion_indices, -1, :]))
            # logger.debug(f"[VibeVoice] TTS EOS logits ({tts_eos_logits.shape}): {tts_eos_logits}")
            # logger.debug(f"[VibeVoice] TTS EOS-Worthy ({tts_eos_logits[0].item()} > {eos_threshold}) > {tts_eos_logits[0].item() > eos_threshold}")
            if tts_eos_logits[0].item() > eos_threshold:
                self.finished = True
                break

            n_past, n_len = self.neg_tts_lm_state["past_key_values"], self.neg_tts_lm_state["seq_len"]
            n_pos, n_mask = torch.arange(n_len, n_len+1, device=device), torch.cat([self.neg_tts_lm_state["attention_mask"], torch.ones((1, 1), device=device)], dim=-1)
            self.neg_tts_lm_outputs = self.model.forward_tts_lm(
                input_ids=dummy_id, attention_mask=n_mask, past_key_values=n_past, cache_position=n_pos,
                tts_text_masks=torch.zeros_like(dummy_id), lm_last_hidden_state=acoustic_embed, return_dict=True, output_attentions=False, output_hidden_states=False
            )
            self.neg_tts_lm_state.update({"past_key_values": self.neg_tts_lm_outputs.past_key_values, "seq_len": n_len+1, "attention_mask": n_mask})

        # --- Batch Decode ---
        if latents_buffer:
            combined_latents = torch.cat(latents_buffer, dim=1)

            scaled_latents = combined_latents / self.model.speech_scaling_factor.to(device) - self.model.speech_bias_factor.to(device)

            audio_chunk = self.model.acoustic_tokenizer.decode(
                scaled_latents.to(self.model.acoustic_tokenizer.device),
                cache=self.acoustic_cache,
                sample_indices=diffusion_indices,
                use_cache=True
            )

            if torch.is_tensor(audio_chunk):
                audio_np = audio_chunk.detach().cpu().float().numpy()
            else:
                audio_np = np.array(audio_chunk)

            if audio_np.ndim > 1:
                audio_np = audio_np.reshape(-1)

            yield audio_np


def _get_next_stream_id():
    global _stream_id_counter
    with _stream_id_lock:
        _stream_id_counter += 1
        return _stream_id_counter


def _tts_worker():
    global _generator_instance, _websocket_server, _tts_service
    logger.info("[VibeVoice] Background worker started")

    chunk_accumulator = []
    accumulated_samples = 0
    TARGET_CHUNK_SIZE = 24000 * 0.4
    current_stream_id = None

    def send_audio_chunk(audio_array: np.ndarray):
        pcm_data = _float32_to_pcm16(audio_array)

        if _websocket_server:
            _websocket_server.broadcast(pcm_data)

        with _audio_chunks_lock:
            _audio_chunks_for_ui.append(audio_array)

    while True:
        try:
            item = _tts_queue.get()
            if item == "STOP":
                break

            target_voice = None
            if _tts_service:
                target_voice = _tts_service.default_voice_key or params.get("default_voice_preset", "en-Davis_man")

            cmd, payload = item

            if cmd == "RESET":
                # Apply UI configuration
                cfg_val = params.get("cfg_scale", 1.5)
                steps_val = params.get("inference_steps", 5)

                if _tts_service and _tts_service.model:
                    _tts_service.model.set_ddpm_inference_steps(num_steps=steps_val)

                _generator_instance = VibeVoiceIncrementalGenerator(_tts_service, target_voice, cfg_scale=cfg_val)
                logger.debug(f"[VibeVoice] Generator initialized (CFG: {cfg_val}, Steps: {steps_val})")

                chunk_accumulator = []
                accumulated_samples = 0
                current_stream_id = _get_next_stream_id()

                with _audio_chunks_lock:
                    _audio_chunks_for_ui.clear()

                with _completion_lock:
                    _completion_events[current_stream_id] = threading.Event()

                if _websocket_server:
                    _websocket_server.broadcast(json.dumps({
                        "type": "stream_start",
                        "stream_id": current_stream_id
                    }))

            elif cmd == "TEXT":
                if not _generator_instance:
                    continue

                for audio_chunk in _generator_instance.push_text(payload):
                    chunk_accumulator.append(audio_chunk)
                    accumulated_samples += len(audio_chunk)

                    if accumulated_samples >= TARGET_CHUNK_SIZE:
                        full_arr = np.concatenate(chunk_accumulator)
                        send_audio_chunk(full_arr)
                        chunk_accumulator = []
                        accumulated_samples = 0

            elif cmd == "FLUSH":
                logger.debug(f"[VibeVoice] Flushing stream {current_stream_id}...")
                if _generator_instance:
                    for audio_chunk in _generator_instance.flush():
                        chunk_accumulator.append(audio_chunk)
                        accumulated_samples += len(audio_chunk)

                        if accumulated_samples >= TARGET_CHUNK_SIZE:
                            full_arr = np.concatenate(chunk_accumulator)
                            send_audio_chunk(full_arr)
                            chunk_accumulator = []
                            accumulated_samples = 0

                if chunk_accumulator:
                    full_arr = np.concatenate(chunk_accumulator)
                    send_audio_chunk(full_arr)
                    chunk_accumulator = []
                    accumulated_samples = 0

                silence = np.zeros(int(24000 * 0.5), dtype=np.float32)
                send_audio_chunk(silence)

                logger.debug(f"[VibeVoice] All audio sent for stream {current_stream_id}. Waiting for client completion...")

                if current_stream_id:
                    with _completion_lock:
                        completion_event = _completion_events.get(current_stream_id)

                    if completion_event:
                        if completion_event.wait(timeout=60):
                            logger.debug(f"[VibeVoice] Stream {current_stream_id} playback confirmed complete by client")
                        else:
                            logger.warning(f"[VibeVoice] Stream {current_stream_id} completion timeout (60s)")

                        with _completion_lock:
                            _completion_events.pop(current_stream_id, None)

                current_stream_id = None

        except Exception as e:
            logger.error(f"[VibeVoice Worker Error] {e}")
            traceback.print_exc()


def _float32_to_pcm16(chunk: np.ndarray):
    chunk = np.clip(chunk, -1.0, 1.0)
    return (chunk * 32767.0).astype(np.int16).tobytes()


# ==============================================================================
# Extension Hooks
# ==============================================================================

def setup():
    global _tts_service
    if not VIBEVOICE_AVAILABLE:
        logger.error("[VibeVoice] Import failed. Please install dependencies.")
        return

    _tts_service = VibeVoiceService("microsoft/VibeVoice-Realtime-0.5B")
    _tts_service.load()
    _start_websocket_server()
    threading.Thread(target=_tts_worker, daemon=True).start()


def state_modifier(state: dict):
    global _last_seen_len
    if _enable_tts:
        _last_seen_len = 0
        _tts_queue.put(("RESET", None))
    return state


def output_stream_modifier(string: str, state: dict, is_chat=False, is_final=False):
    global _enable_tts, _last_seen_len

    # TODO: Ignore reasoning and "ASSISTANT:" (account for prompt templates)
    # TODO: Stop on interruption

    if not all((_enable_tts, is_chat)):
        return string

    current_len = len(string)

    if current_len < _last_seen_len:
        diff = _last_seen_len - current_len
        if current_len < 10 or diff > 20:
            _last_seen_len = 0
            _tts_queue.put(("RESET", None))

    if current_len > _last_seen_len:
        new_text = string[_last_seen_len:]
        _last_seen_len = current_len
        if new_text:
            _tts_queue.put(("TEXT", new_text))

    if is_final:
        logger.debug("[VibeVoice] Final token. Scheduling flush...")

        def delayed_flush():
            timeout = 30
            start_time = time.time()

            while time.time() - start_time < timeout:
                queue_size = _tts_queue.qsize()

                if queue_size <= 1:
                    logger.debug("[VibeVoice] Queue drained, executing flush")
                    _tts_queue.put(("FLUSH", None))
                    break

                time.sleep(0.1)
            else:
                logger.warning("[VibeVoice] Flush timeout - forcing flush")
                _tts_queue.put(("FLUSH", None))

        threading.Thread(target=delayed_flush, daemon=True).start()

    return string


# ==============================================================================
# Websocket Server
# ==============================================================================


class SimpleAudioServer:
    def __init__(self):
        self.clients = {}
        self.client_counter = 0
        self.loop = None

    def get_client_ids(self):
        return list(self.clients.values())

    async def handler(self, websocket):
        self.client_counter += 1
        client_id = self.client_counter
        self.clients[websocket] = client_id

        logger.info(f"[VibeVoice] Client {client_id} connected")

        try:
            await websocket.send(json.dumps({
                "type": "ready",
                "sample_rate": 24000,
                "client_id": client_id
            }))

            async for message in websocket:
                try:
                    data: dict = json.loads(message)

                    if data.get("type") == "playback_complete":
                        stream_id = data.get("stream_id")
                        logger.debug(f"[VibeVoice] ✓ Client {client_id} finished playing stream {stream_id}")

                        with _completion_lock:
                            event = _completion_events.get(stream_id)
                            if event:
                                event.set()

                except json.JSONDecodeError:
                    logger.warning(f"[VibeVoice] Invalid JSON from client {client_id}")
                except Exception as e:
                    logger.error(f"[VibeVoice] Error processing message: {e}")

        except Exception as e:
            logger.warning(f"[VibeVoice] Client {client_id} error: {e}")
        finally:
            self.clients.pop(websocket, None)
            logger.info(f"[VibeVoice] Client {client_id} disconnected")

    def broadcast(self, data):
        if not self.clients or not self.loop:
            return []

        futures = []
        for ws in self.clients.keys():
            future = asyncio.run_coroutine_threadsafe(ws.send(data), self.loop)
            futures.append(future)
        return futures


def _start_websocket_server():
    global _websocket_server, _websocket_port

    _server_ready_event.clear()

    import websockets
    _websocket_server = SimpleAudioServer()

    def run():
        global _websocket_port
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        _websocket_server.loop = loop
        start_server = websockets.serve(
            _websocket_server.handler,
            "0.0.0.0",
            _websocket_port,
            origins=None,
            ping_interval=None
        )
        server = loop.run_until_complete(start_server)
        _websocket_port = server.sockets[0].getsockname()[1]
        logger.info(f"[VibeVoice] Streaming at ws://0.0.0.0:{_websocket_port}")
        _server_ready_event.set()
        loop.run_forever()

    threading.Thread(target=run, daemon=True).start()

# ==============================================================================
# UI
# ==============================================================================


def ui():
    global _tts_service, _enable_tts, _audio_chunks_for_ui

    if _tts_service is None:
        gr.Markdown("### VibeVoice Realtime TTS\n**Error:** Model not loaded.")
        return

    def save_current_settings(disable_fa, cfg, steps, eos_thres, voice):
        params["disable_flash_attn"] = disable_fa
        params["cfg_scale"] = cfg
        params["inference_steps"] = steps
        params["eos_threshold"] = eos_thres
        params["default_voice_preset"] = voice

        # Collect config values
        data_to_save = {
            "disable_flash_attn": disable_fa,
            "cfg_scale": cfg,
            "inference_steps": steps,
            "eos_threshold": eos_thres,
            "default_voice_preset": voice
        }

        try:
            with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
                json.dump(data_to_save, f, indent=4)
            return gr.Info(f"Configuration saved to {os.path.relpath(SETTINGS_FILE, TGWUI_ROOT)}")
        except Exception as e:
            return gr.Warning(f"Failed to save settings: {e}")

    with gr.Group():
        gr.Markdown("### VibeVoice Realtime TTS")

        with gr.Row():
            enable_checkbox = gr.Checkbox(label="Enable VibeVoice TTS", value=False, scale=1)

            def toggle_tts(enable):
                global _enable_tts
                _enable_tts = enable
                logger.info(f"[VibeVoice] TTS {'enabled' if enable else 'disabled'}")
                return enable
            enable_checkbox.change(toggle_tts, enable_checkbox, enable_checkbox)

        # --- Config Controls ---
        with gr.Row():
            disable_fa_chk = gr.Checkbox(
                label="Disable Flash Attention 2",
                value=params.get("disable_flash_attn", False),
                interactive=True,
                scale=2
            )
            reload_btn = gr.Button("Reload Model", scale=1, variant="secondary")

            def update_fa_param(x):
                params["disable_flash_attn"] = x
                return x

            def reload_model_trigger():
                if _tts_service:
                    try:
                        _tts_service.load()
                        return "Model Reloaded Successfully"
                    except Exception as e:
                        logger.error(f"[VibeVoice] Reload failed: {e}")
                        return f"Error: {e}"
                return "Service not initialized"

            disable_fa_chk.change(update_fa_param, disable_fa_chk, None)
            reload_btn.click(reload_model_trigger, None, None).then(
                fn=lambda: gr.Info("VibeVoice Model Reloaded"), outputs=None
            )

        with gr.Row():
            cfg_slider = gr.Slider(
                label="CFG Scale",
                minimum=1.0, maximum=3.0, step=0.05,
                value=params.get("cfg_scale", 1.5),
                scale=1, interactive=True
            )
            steps_slider = gr.Slider(
                label="Inference Steps",
                minimum=1, maximum=20, step=1,
                value=params.get("inference_steps", 5),
                scale=1, interactive=True
            )
            eos_thres_slider = gr.Slider(
                label="EOS Threshold",
                minimum=0.0, maximum=0.5, step=0.00005,
                value=params.get("eos_threshold", 0.0007),
                scale=1, interactive=True
            )

            def update_cfg(val):
                params["cfg_scale"] = val
                return val

            def update_steps(val):
                params["inference_steps"] = val
                return val

            def update_eos_thres(val):
                params["eos_threshold"] = val
                return val

            cfg_slider.change(update_cfg, cfg_slider, None)
            steps_slider.change(update_steps, steps_slider, None)
            eos_thres_slider.change(update_eos_thres, eos_thres_slider, None)

        with gr.Row():
            try:
                voice_choices = sorted(list(_tts_service.voice_presets.keys())) if _tts_service and _tts_service.voice_presets else []
            except Exception:
                voice_choices = []

            current_voice = (_tts_service.default_voice_key if _tts_service else None) or params.get("default_voice_preset", "en-Davis_man")

            voice_dropdown = gr.Dropdown(
                label="Voice Preset",
                choices=voice_choices,
                value=current_voice,
                interactive=True,
                scale=3
            )

            save_conf_btn = gr.Button("💾 Save Config", scale=1, variant="primary")

            def set_voice_ui(selected_key: str):
                global _tts_service
                if not _tts_service:
                    return "Error"
                _tts_service.set_voice_key(selected_key)
                params["default_voice_preset"] = selected_key
                return f"Voice set to {selected_key}"

            voice_dropdown.change(set_voice_ui, inputs=voice_dropdown)

            save_conf_btn.click(
                fn=save_current_settings,
                inputs=[disable_fa_chk, cfg_slider, steps_slider, eos_thres_slider, voice_dropdown],
                outputs=None
            )

        # --- Audio Output ---
        with gr.Row():
            def get_audio_output():
                global _audio_chunks_for_ui
                with _audio_chunks_lock:
                    if _audio_chunks_for_ui:
                        # Copy accumulated audio
                        chunks_snapshot = list(_audio_chunks_for_ui)

                        if not chunks_snapshot:
                            return None

                        audio_data = np.concatenate(chunks_snapshot, axis=0)

                        sr = 24000
                        try:
                            sr = _tts_service.processor.feature_extractor.sampling_rate
                        except AttributeError:
                            pass
                        return (sr, audio_data)
                return None

            audio_output = gr.Audio(label="Manual Refresh / Replay", interactive=False, scale=2)
            refresh_button = gr.Button("Refresh", scale=1)
            refresh_button.click(get_audio_output, outputs=audio_output)

        with gr.Row():
            gr.HTML("""
            <div style="border: 1px solid #ccc; padding: 10px; border-radius: 5px; background: rgba(0,0,0,0.05);">
                <h4>Realtime Stream</h4>
                <p style="font-size: 1.1em; font-family: monospace;">
                    Status: <span id="vibevoice-status" style="font-weight: bold;">Disconnected</span>
                </p>
                <div style="height: 5px; width: 0%; background-color: orange; transition: width 0.2s;" id="vibevoice-bar"></div>
            </div>
            """)


def custom_js():
    js = """
    (function() {
        let ws = null;
        let sampleRate = 24000;
        let audioContext = null;
        let nextStartTime = 0;
        let audioQueue = [];
        let hasStartedPlaying = false;
        let accumulatedSec = 0;
        const START_THRESHOLD_SEC = __START_THRESHOLD_SEC__;
        let currentStreamId = null;
        let lastScheduledSource = null;
        const NO_CHUNK_TIMEOUT = 1000; // 1 second of no chunks = stream ended

        // Timer Logic
        let playbackStartTime = 0;
        let timerInterval = null;

        function updateTimer() {
            if (!hasStartedPlaying || !audioContext) return;
            const elapsed = audioContext.currentTime - playbackStartTime;
            const mins = Math.floor(elapsed / 60).toString().padStart(2, '0');
            const secs = Math.floor(elapsed % 60).toString().padStart(2, '0');

            const statusSpan = document.getElementById('vibevoice-status');
            if (statusSpan) {
                statusSpan.textContent = `Playing (${mins}:${secs})`;
            }
        }

        function initAudioContext() {
            if (!audioContext) {
                audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: sampleRate });
            }
        }

        function scheduleQueue() {
            if (!audioContext) return;

            const currentTime = audioContext.currentTime;

            if (nextStartTime === 0 || nextStartTime < currentTime - 10.0) {
                nextStartTime = currentTime + 0.05;
            }

            while (audioQueue.length > 0) {
                const float32Data = audioQueue.shift();
                const buffer = audioContext.createBuffer(1, float32Data.length, sampleRate);
                buffer.getChannelData(0).set(float32Data);
                const source = audioContext.createBufferSource();
                source.buffer = buffer;
                source.connect(audioContext.destination);

                const startTime = Math.max(nextStartTime, currentTime);
                source.start(startTime);

                lastScheduledSource = source;

                source.onended = () => {
                    setTimeout(() => {
                        if (lastScheduledSource === source) {
                            console.debug(`[VibeVoice] ✓ Stream ended.`);

                            if (ws && ws.readyState === WebSocket.OPEN) {
                                ws.send(JSON.stringify({
                                    type: 'playback_complete',
                                    stream_id: currentStreamId
                                }));
                            }

                            const statusSpan = document.getElementById('vibevoice-status');
                            const bar = document.getElementById('vibevoice-bar');
                            if (statusSpan) {
                                statusSpan.textContent = 'Connected (Idle)';
                                statusSpan.style.color = 'green';
                            }
                            if (bar) bar.style.width = '0%';

                            if (timerInterval) clearInterval(timerInterval);
                            hasStartedPlaying = false;
                            lastScheduledSource = null;
                        }
                    }, NO_CHUNK_TIMEOUT);
                };

                nextStartTime = startTime + buffer.duration;
            }
        }

        function connectVibeVoiceStream() {
            const statusSpan = document.getElementById('vibevoice-status');
            const bar = document.getElementById('vibevoice-bar');

            const portPart = '__VIBEVOICE_WS_PORT__' !== '0' ? ':__VIBEVOICE_WS_PORT__' : (window.location.port ? ':' + window.location.port : '');
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const url = protocol + '//' + window.location.hostname + portPart;

            ws = new WebSocket(url);
            ws.binaryType = 'arraybuffer';

            ws.onopen = () => {
                if(statusSpan) { statusSpan.textContent = 'Connected'; statusSpan.style.color = 'green'; }
                initAudioContext();
            };

            ws.onmessage = (event) => {
                if (typeof event.data === 'string') {
                    const msg = JSON.parse(event.data);

                    if (msg.type === 'stream_start') {
                        currentStreamId = msg.stream_id;

                        nextStartTime = 0;
                        audioQueue = [];
                        hasStartedPlaying = false;
                        accumulatedSec = 0;
                        lastScheduledSource = null;

                        if (audioContext) {
                            nextStartTime = audioContext.currentTime + 0.1;
                        }

                        if(statusSpan) {
                            statusSpan.textContent = 'Buffering...';
                            statusSpan.style.color = 'yellow';
                        }
                        if (bar) bar.style.width = '10%';
                    }
                } else if (event.data instanceof ArrayBuffer) {

                    if (audioContext && audioContext.state === 'suspended') {
                        audioContext.resume();
                    }

                    const pcm16 = new Int16Array(event.data);
                    const float32 = new Float32Array(pcm16.length);
                    for (let i = 0; i < pcm16.length; i++) {
                        float32[i] = pcm16[i] / 32768.0;
                    }

                    audioQueue.push(float32);
                    accumulatedSec += float32.length / sampleRate;

                    if (!hasStartedPlaying) {
                        if (accumulatedSec >= START_THRESHOLD_SEC) {
                            hasStartedPlaying = true;
                            playbackStartTime = audioContext.currentTime;

                            const statusSpan = document.getElementById('vibevoice-status');
                            const bar = document.getElementById('vibevoice-bar');
                            if(statusSpan) {
                                statusSpan.style.color = 'green';
                            }
                            if(bar) bar.style.width = '100%';

                            if (timerInterval) clearInterval(timerInterval);
                            timerInterval = setInterval(updateTimer, 500);

                            nextStartTime = audioContext.currentTime + 0.1;
                            scheduleQueue();
                        }
                    } else {
                        scheduleQueue();
                    }
                }
            };

            ws.onclose = () => {
                if(statusSpan) { statusSpan.textContent = 'Disconnected'; statusSpan.style.color = 'red'; }
                if (timerInterval) clearInterval(timerInterval);
                setTimeout(connectVibeVoiceStream, 2000);
            };
        }

        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', connectVibeVoiceStream);
        } else {
            setTimeout(connectVibeVoiceStream, 500);
        }
    })();
    """

    if not _server_ready_event.wait(timeout=5.0):
        logger.error("[VibeVoice] Timed out waiting for WebSocket server!")
    return js.replace('__VIBEVOICE_WS_PORT__', str(_websocket_port)).replace('__START_THRESHOLD_SEC__', str(0.1))
