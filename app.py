import gradio as gr
import os
import glob
import shutil
import requests
import subprocess
import cv2
import random

# Dynamically resolve paths for absolute portability across systems
# Dynamically resolve paths for absolute portability
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SOVITS_DIR = os.path.join(BASE_DIR, "GPT-SoVITS-v2pro-20250604")
SOVITS_RUNTIME = os.path.join(SOVITS_DIR, "runtime")

# Add bundled FFmpeg/Python to PATH for this process
if os.path.exists(SOVITS_RUNTIME):
    os.environ["PATH"] = SOVITS_RUNTIME + os.pathsep + os.environ.get("PATH", "")

INPUT_DIR = os.path.join(BASE_DIR, "input")
WAV2LIP_DIR = os.path.join(BASE_DIR, "wav2lip-studio")

# Priority: wav2lip-studio Venv > Bundled SoVITS Python (most portable)
WAV2LIP_PYTHON = os.path.join(WAV2LIP_DIR, "venv", "Scripts", "python.exe")
if not os.path.exists(WAV2LIP_PYTHON):
    WAV2LIP_PYTHON = os.path.join(SOVITS_RUNTIME, "python.exe")

BRIDGE_SCRIPT = os.path.join(WAV2LIP_DIR, "auto_wav2lip.py")

DEFAULT_REF_AUDIO = os.path.join(SOVITS_DIR, "output", "slicer_opt", "LuckyV2", "有些事情好恐怖哦，是怎么回事呢，太好笑了，如果你们要找家庭看护工或家庭帮佣.wav")
DEFAULT_REF_TEXT = "有些事情好恐怖哦，是怎么回事呢，太好笑了，如果你们要找家庭看护工或家庭帮佣"
FALLBACK_VIDEO_DIR = os.path.join(BASE_DIR, "original", "video")
DEFAULT_VIDEO_FALLBACK = os.path.join(FALLBACK_VIDEO_DIR, "input_vid.mp4")

def is_valid_video(vid_path):
    try:
        cap = cv2.VideoCapture(vid_path)
        valid = cap.isOpened() and cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0
        cap.release()
        return valid
    except:
        return False

sovits_process = None

def start_sovits_server():
    global sovits_process
    try:
        requests.get("http://127.0.0.1:9880/", timeout=1, proxies={"http": None, "https": None})
        yield "Server connectivity verified! ✅", "[INFO] GPT-SoVITS server is already online!\n"
        return
    except:
        pass
        
    yield "Starting local GPT-SoVITS server... ⏳", "[INFO] Launching GPT-SoVITS in the background..."
    
    cmd = [os.path.join(SOVITS_RUNTIME, "python.exe"), "api_v2.py", "-a", "127.0.0.1", "-p", "9880"]
    env = os.environ.copy()
    env["PATH"] = SOVITS_RUNTIME + os.pathsep + env.get("PATH", "")
    
    try:
        sovits_process = subprocess.Popen(cmd, cwd=SOVITS_DIR, env=env)
        import time
        for attempt in range(15):
            try:
                requests.get("http://127.0.0.1:9880/", timeout=2, proxies={"http": None, "https": None})
                yield "Server started and ready! ✅", "[INFO] GPT-SoVITS launched successfully!\n"
                return
            except:
                yield f"Waiting for background AI server (Step {attempt+1}/15)... ⏳", f"[INFO] Waiting for port 9880..."
                time.sleep(2)
        yield "Server failed to start in time. ❌", "[ERROR] GPT-SoVITS took too long to start.\n"
    except Exception as e:
        yield f"Failed to start server: {e}", f"[ERROR] Failed to start GPT-SoVITS: {e}\n"

def stop_sovits_server():
    global sovits_process
    yield "Shutting down GPT-SoVITS to free VRAM... 🧹", "[INFO] Terminating GPT-SoVITS API Server..."
    if sovits_process is not None:
        try:
            sovits_process.terminate()
            sovits_process.wait(timeout=5)
        except:
            sovits_process.kill()
        sovits_process = None
        
    try:
        out = subprocess.check_output('netstat -ano | findstr :9880', shell=True).decode()
        for line in out.splitlines():
            if 'LISTENING' in line:
                pid = line.strip().split()[-1]
                subprocess.run(['taskkill', '/F', '/PID', pid], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                yield None, f"[INFO] Force killed process {pid} on port 9880."
    except:
        pass
    yield "GPT-SoVITS unloaded from memory! ✅", "[INFO] VRAM is now free for Wav2Lip."

def smart_video_looper(audio_path, video_path, folder_path):
    try:
        from pydub import AudioSegment
        from pydub.silence import detect_silence
    except ImportError:
        yield None, "[WARNING] pydub not installed, skipping smart-loop.", video_path
        return
        
    try:
        audio = AudioSegment.from_file(audio_path)
        audio_duration = len(audio) / 1000.0
    except Exception as e:
        yield None, f"[WARNING] pydub could not read audio: {e}", video_path
        return
        
    try:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_duration = frame_count / fps if fps > 0 else 0
        cap.release()
    except Exception as e:
        yield None, f"[WARNING] cv2 could not read video: {e}", video_path
        return
        
    if video_duration <= 0 or audio_duration <= video_duration:
        yield None, None, video_path
        return
        
    yield f"🔄 Audio is longer than video ({round(audio_duration, 1)}s > {round(video_duration, 1)}s). Smart-looping at pauses...", f"[SMART LOOP] Audio: {audio_duration}s, Video: {video_duration}s", None
    
    try:
        silences_ms = detect_silence(audio, min_silence_len=400, silence_thresh=audio.dBFS-16)
        pauses_s = [(start + end) / 2000.0 for start, end in silences_ms]
    except Exception as e:
        yield None, f"[WARNING] Failed to detect silence: {e}", None
        pauses_s = []

    current_audio_time = 0.0
    clips = []
    
    while current_audio_time < audio_duration:
        max_reach = current_audio_time + video_duration
        if max_reach >= audio_duration:
            clips.append(audio_duration - current_audio_time)
            break
            
        min_acceptable_pause = current_audio_time + (video_duration * 0.4)
        valid_pauses = [p for p in pauses_s if min_acceptable_pause <= p <= max_reach]
        
        if valid_pauses:
            chosen_pause = valid_pauses[-1]
            clip_len = chosen_pause - current_audio_time
            clips.append(clip_len)
            current_audio_time = chosen_pause
        else:
            clips.append(video_duration)
            current_audio_time += video_duration
            
    if not clips:
        yield None, None, video_path
        return
        
    import time
    unique_id = str(int(time.time()))
    concat_txt_path = os.path.join(folder_path, f"temp_concat_{unique_id}.txt")
    extended_video_path = os.path.join(folder_path, f"temp_extended_{unique_id}.mp4")
    
    try:
        with open(concat_txt_path, "w", encoding="utf-8") as f:
            for clip_len in clips:
                f.write(f"file '{os.path.abspath(video_path).replace(chr(92), '/')}'\n")
                f.write(f"outpoint {clip_len:.3f}\n")
                
        ffmpeg_cmd = [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0", 
            "-i", concat_txt_path, "-c", "copy", extended_video_path
        ]
        
        subprocess.run(ffmpeg_cmd, check=True, capture_output=True, text=True)
        try: os.remove(concat_txt_path)
        except: pass
        yield "✅ Smart-looping complete! Video extended seamlessly.", f"[SMART LOOP] Created {extended_video_path} with {len(clips)} chunks.", extended_video_path
    except Exception as e:
        try: os.remove(concat_txt_path)
        except: pass
        yield None, f"[WARNING] FFmpeg smart-looping failed: {e}", video_path


def process_folders(ref_audio, ref_text, ref_lang, target_lang, video_quality, restore_model, resize_factor, stop_video, random_cut, max_resolution_limit=1280):
    basic_log = ""
    raw_log = ""
    
    def append_log(basic_msg, raw_msg=None, is_error=False):
        nonlocal basic_log, raw_log
        
        if basic_msg:
            # HTML Formatting for Basic Log
            color = "#ef4444" if is_error else "#10b981" if "✅" in basic_msg else "#f8fafc"
            basic_log += f"<div style='color: {color}; margin-bottom: 4px;'>{basic_msg}</div>"
            
        if raw_msg:
            raw_log += raw_msg + "\n"
        elif basic_msg:
            raw_log += f"[SYSTEM] {basic_msg}\n"
            
        # Wrap basic log inside a scrolling container
        html_wrapper = f"<div style='background: #0f172a; border: 1px solid #334155; border-radius: 8px; padding: 15px; height: 250px; overflow-y: auto; font-family: monospace; font-size: 14px;'>{basic_log}</div>"
        
        return html_wrapper, raw_log

    if not os.path.exists(INPUT_DIR):
        yield append_log("🚨 The 'input' directory does not exist! Please create it.", f"[ERROR] The input directory '{INPUT_DIR}' does not exist.", is_error=True)
        return

    folders = [f.path for f in os.scandir(INPUT_DIR) if f.is_dir()]
    if not folders:
        yield append_log("No structural folders detected in 'input/' to process.", f"[WARNING] No folders found in '{INPUT_DIR}'.")
        return
        
    yield append_log(f"Detected {len(folders)} folders! Ready and organizing...", f"[INFO] Found {len(folders)} folders to process.")
    
    # ==========================================
    # PHASE 1: AUDIO GENERATION (SO-VITS)
    # ==========================================
    folders_needing_audio = []
    
    for folder in folders:
        # Check if already processed completely
        video_files = glob.glob(os.path.join(folder, "*.mp4"))
        generated_exists = any(os.path.basename(v).startswith("generated_video_") for v in video_files)
        if generated_exists:
            continue
            
        audio_output_path = os.path.join(folder, "generated_audio.wav")
        if not os.path.exists(audio_output_path):
            # Guardrail: Check text file integrity
            txts = glob.glob(os.path.join(folder, "*.txt"))
            if len(txts) == 0:
                yield append_log(f"🔴 Missing text script in folder [{os.path.basename(folder)}]!", f"[ERROR] No text file found in folder '{os.path.basename(folder)}'.", is_error=True)
                continue
            
            txt_path = txts[0]
            try:
                with open(txt_path, "r", encoding="utf-8") as f:
                    target_text = f.read().strip()
                if not target_text:
                    raise ValueError("Text string is empty.")
            except Exception as e:
                yield append_log(f"🔴 RED WARNING: '{os.path.basename(txt_path)}' is corrupted or entirely empty! Skipping folder.", f"[ERROR] Text file read failure: {str(e)}", is_error=True)
                continue
                
            folders_needing_audio.append((folder, target_text))

    if folders_needing_audio:
        yield append_log(f"<br><b>🔊 PHASE 1: Audio Generation ({len(folders_needing_audio)} folders)</b>", f"\n========== PHASE 1: AUDIO GENERATION ==========")
        
        server_ready = False
        for basic, raw in start_sovits_server():
            yield append_log(basic, raw)
            if "✅" in (basic or ""):
                server_ready = True
                
        if not server_ready:
            yield append_log("🚨 Failed to initialize So-VITS server. Stopping.", "[ERROR] Cannot proceed without TTS server.", is_error=True)
            return

        for folder, target_text in folders_needing_audio:
            folder_name = os.path.basename(folder)
            audio_output_path = os.path.join(folder, "generated_audio.wav")
            
            yield append_log(f"🎙 Generating voice for [{folder_name}]...", f"[INFO] Target Text: '{target_text}'\n-> Starting GPT-SoVITS TTS...")
            
            # Map target_lang logic
            tts_lang = "zh" if "Chinese" in target_lang else "ja" if "Japanese" in target_lang else "en"
            prompt_lang = "zh" if "Chinese" in ref_lang else "ja" if "Japanese" in ref_lang else "en"
            
            tts_payload = {
                "text": target_text,
                "text_lang": tts_lang,
                "ref_audio_path": ref_audio,
                "prompt_text": ref_text,
                "prompt_lang": prompt_lang,
                "media_type": "wav"
            }
            
            try:
                response = requests.post("http://127.0.0.1:9880/tts", json=tts_payload, timeout=180, proxies={"http": None, "https": None})
                if response.status_code != 200:
                    yield append_log(f"🔴 Audio engine network crashed for [{folder_name}].", f"[ERROR] TTS failed. Status {response.status_code}: {response.text}", is_error=True)
                    continue
                with open(audio_output_path, "wb") as f:
                    f.write(response.content)
                yield append_log("✅ Voice rendered successfully!", f"-> TTS Audio generated successfully: {audio_output_path}")
            except Exception as e:
                yield append_log(f"🔴 Fatal connection error to audio generator.", f"[ERROR] TTS request error: {str(e)}", is_error=True)
                continue
                
    else:
        yield append_log("<br><b>🔊 PHASE 1: Audio Generation Skipped</b>", "\n[INFO] All folders already have generated audio. Skipping So-VITS startup.")

    # Unconditionally ensure So-VITS is shut down before Phase 2 to guarantee VRAM is free
    for basic, raw in stop_sovits_server():
        if basic or raw:
            yield append_log(basic, raw)

    # ==========================================
    # PHASE 2: VIDEO GENERATION (WAV2LIP)
    # ==========================================
    yield append_log(f"<br><b>🎬 PHASE 2: Video Generation & Lipsync</b>", f"\n========== PHASE 2: VIDEO GENERATION ==========")
    
    for folder in folders:
        folder_name = os.path.basename(folder)
        
        # Check if already processed
        video_files = glob.glob(os.path.join(folder, "*.mp4"))
        generated_exists = any(os.path.basename(v).startswith("generated_video_") for v in video_files)
        
        if generated_exists:
            yield append_log(f"⚠️ Skipping Folder [{folder_name}] (Already generated previously)", f"[SKIP] Folder '{folder_name}' has already been processed.")
            continue
            
        videos = [v for v in video_files if not os.path.basename(v).startswith("generated_video_")]
        audio_output_path = os.path.join(folder, "generated_audio.wav")
        
        if not os.path.exists(audio_output_path):
            yield append_log(f"🔴 Missing audio file for [{folder_name}], skipping video phase.", f"[ERROR] Missing generated_audio.wav in {folder_name}.", is_error=True)
            continue
            
        yield append_log(f"<br><b>➡️ Working on Video for [{folder_name}]</b>", f"\n========== Processing Video: {folder_name} ==========")
        
        # Guardrail: Check video file integrity & fallback logic
        active_video_path = None
        if len(videos) > 0:
            active_video_path = videos[0]
            if not is_valid_video(active_video_path):
                yield append_log("⚠️ User-provided video was corrupted or unsupported. Intervening with default fallback...", f"[WARN] Video '{active_video_path}' failed cv2 validation.")
                active_video_path = None
                
        if not active_video_path:
            fallback_videos = []
            if os.path.exists(FALLBACK_VIDEO_DIR):
                fallback_videos = [v for v in glob.glob(os.path.join(FALLBACK_VIDEO_DIR, "*.mp4")) if not os.path.basename(v).startswith("generated_video_")]
            
            if fallback_videos:
                active_video_path = random.choice(fallback_videos)
                yield append_log(f"✅ Assigned random Fallback Video: {os.path.basename(active_video_path)}", f"[INFO] Hooking fallback video from directory: {active_video_path}")
            elif os.path.exists(DEFAULT_VIDEO_FALLBACK):
                active_video_path = DEFAULT_VIDEO_FALLBACK
                yield append_log("✅ Assigned verified Global Fallback Video.", f"[INFO] Hooking fallback video: {DEFAULT_VIDEO_FALLBACK}")
            else:
                yield append_log("🔴 RED WARNING: No valid input video AND fallback missing! Skipping folder.", f"[ERROR] No input video & Default Video missing.", is_error=True)
                continue
                
        video_name = os.path.basename(active_video_path)
        
        import time
        start_time = time.time()
        
        # 2b. RANDOM VIDEO CUTTER
        if random_cut:
            yield append_log(f"✂ Calculating algorithmic crop...", f"-> Applying Random Start Video Cutter...")
            import wave
            
            audio_duration = 0.0
            try:
                with wave.open(audio_output_path, 'rb') as w:
                    frames_w = w.getnframes()
                    rate = w.getframerate()
                    audio_duration = frames_w / float(rate)
            except Exception as e:
                yield append_log(None, f"[WARNING] Could not read audio duration: {e}")
                
            video_duration = 0.0
            try:
                cap = cv2.VideoCapture(active_video_path)
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if fps > 0:
                    video_duration = frame_count / fps
                cap.release()
            except Exception as e:
                yield append_log(None, f"[WARNING] Could not read video duration: {e}")
                
            if audio_duration > 0 and video_duration > 0:
                cut_length = audio_duration + 1.0 # 1.0 second padding to capture trailing syllables
                if video_duration > cut_length:
                    max_start = video_duration - cut_length
                    random_start = random.uniform(0.0, max_start)
                    
                    import time
                    unique_id = str(int(time.time()))
                    random_cut_path = os.path.join(folder, f"temp_cut_{unique_id}.mp4")
                    
                    ffmpeg_cmd = [
                        "ffmpeg", "-y", "-ss", str(random_start), "-i", active_video_path, 
                        "-t", str(cut_length), "-c:v", "copy", "-c:a", "copy", random_cut_path
                    ]
                    
                    try:
                        result = subprocess.run(ffmpeg_cmd, check=True, capture_output=True, text=True)
                        active_video_path = random_cut_path
                        yield append_log("✅ Subclip isolated!", f"   [Cutter] Sliced video from {round(random_start, 2)}s to {round(random_start+cut_length, 2)}s")
                    except subprocess.CalledProcessError as e:
                        yield append_log(None, f"[WARNING] ffmpeg cut failed, using original video. Error: {e.stderr}")
                    except Exception as e:
                        yield append_log(None, f"[WARNING] ffmpeg cut structural failure, using original video. Error: {e}")
                else:
                    yield append_log(None, f"   [Cutter] Video length is too short to cut ({round(video_duration,1)}s <= audio {round(audio_duration,1)}s). Using full.")
            else:
                yield append_log(None, f"[WARNING] Invalid duration fetched (V: {video_duration}, A: {audio_duration}), skipping random cut.")

        # 2c. VIDEO DOWN-SAMPLER / PREPROCESS
        if max_resolution_limit and max_resolution_limit > 0:
            try:
                cap = cv2.VideoCapture(active_video_path)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()
                
                if width > max_resolution_limit or height > max_resolution_limit:
                    scale = float(max_resolution_limit) / float(max(width, height))
                    new_width = (int(width * scale) // 2) * 2
                    new_height = (int(height * scale) // 2) * 2
                    
                    yield append_log(f"📐 Downscaling video to fit max {max_resolution_limit}px ({new_width}x{new_height})...", f"[PREPROCESS] Scaling {width}x{height} to {new_width}x{new_height}...")
                    
                    import time
                    unique_id = str(int(time.time()))
                    resized_path = os.path.join(folder, f"temp_resized_{unique_id}.mp4")
                    
                    ffmpeg_cmd = [
                        "ffmpeg", "-y", "-i", active_video_path,
                        "-vf", f"scale={new_width}:{new_height}",
                        "-c:v", "mpeg4", "-q:v", "2",
                        "-c:a", "copy", resized_path
                    ]
                    
                    try:
                        result = subprocess.run(ffmpeg_cmd, check=True, capture_output=True, text=True)
                        
                        # Clean up previous temporary cut file if it was created during random cut
                        if "temp_cut_" in active_video_path and os.path.exists(active_video_path):
                            try:
                                os.remove(active_video_path)
                            except Exception as e:
                                yield append_log(None, f"[WARNING] Could not delete temp cut clip during resize: {e}")
                                
                        active_video_path = resized_path
                        yield append_log(f"✅ Downscaling complete!", f"[PREPROCESS] Video successfully resized to {new_width}x{new_height} and saved to {resized_path}")
                    except subprocess.CalledProcessError as e:
                        yield append_log("⚠️ Downscaling failed, using original video size.", f"[WARNING] ffmpeg resize failed. Error: {e.stderr}")
                    except Exception as e:
                        yield append_log("⚠️ Downscaling failed, using original video size.", f"[WARNING] ffmpeg resize error: {e}")
            except Exception as e:
                yield append_log(None, f"[WARNING] Could not read video dimensions for resizing: {e}")
                
        # 2d. SMART VIDEO LOOPER
        try:
            for basic, raw, ret_vid in smart_video_looper(audio_output_path, active_video_path, folder):
                if basic or raw:
                    yield append_log(basic, raw)
                if ret_vid is not None:
                    if "temp_" in active_video_path and os.path.exists(active_video_path) and active_video_path != ret_vid:
                        try: os.remove(active_video_path)
                        except: pass
                    active_video_path = ret_vid
        except Exception as e:
            yield append_log(None, f"[WARNING] Smart looper error: {e}")
        
        # 3. RUN WAV2LIP
        yield append_log("🎬 Binding Video and Audio (Lip-syncing)... ETA ~3 Minutes", f"-> Starting Wav2Lip Studio Pipeline...")
        project_name = f"auto_project_{folder_name}"
        wav2lip_cmd = [
            WAV2LIP_PYTHON, "-u", BRIDGE_SCRIPT, 
            "--project", project_name, 
            "--video", active_video_path, 
            "--audio", audio_output_path,
            "--video_quality", video_quality,
            "--restore_model", restore_model,
            "--resize_factor", str(int(resize_factor))
        ]
        if stop_video:
            wav2lip_cmd.append("--stop_video")
            
        try:
            env_unbuffered = os.environ.copy()
            env_unbuffered["PYTHONUNBUFFERED"] = "1"
            process = subprocess.Popen(wav2lip_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=WAV2LIP_DIR, env=env_unbuffered)
            buffer = []
            while True:
                char = process.stdout.read(1)
                if not char:
                    break
                if char in ['\n', '\r']:
                    if buffer:
                        line = "".join(buffer).strip()
                        if line:
                            yield append_log(None, f"   [Wav2Lip] {line}")
                        buffer = []
                else:
                    buffer.append(char)
            process.wait()
            if process.returncode != 0:
                yield append_log("🔴 Wav2Lip integration engine completely crashed! Check real logs.", f"[ERROR] Wav2Lip failed with exit code {process.returncode}.", is_error=True)
                continue
        except Exception as e:
            yield append_log("🔴 Wav2Lip execution environment missing or dead.", f"[ERROR] Error executing Wav2Lip: {str(e)}", is_error=True)
            continue
            
        # 4. MOVE VIDEO BACK
        wav2lip_output = os.path.join(WAV2LIP_DIR, "projects", project_name, "wav2lip", "video.mp4")
        if os.path.exists(wav2lip_output):
            final_video_path = os.path.join(folder, f"generated_video_{video_name}")
            shutil.copy(wav2lip_output, final_video_path)
            
            end_time = time.time()
            elapsed = round(end_time - start_time, 2)
            yield append_log(f"✅ SUCCESSFULLY COMPLETED! Took {elapsed} seconds.", f"[COMPLETED] Successfully created: {final_video_path} (Took {elapsed} seconds)")
        else:
            yield append_log("🔴 Final generated output map was detached/missing.", f"[ERROR] Could not find Wav2Lip studio output at '{wav2lip_output}'.", is_error=True)
            
        # 5. CLEANUP TEMPS
        if "temp_" in active_video_path and os.path.exists(active_video_path):
            try:
                os.remove(active_video_path)
            except Exception as e:
                yield append_log(None, f"[WARNING] Could not delete temporary clip {active_video_path}: {e}")

    yield append_log("<br>🎉 <b>MASTER BATCH QUEUE CLEARED!</b>", "\n========== ALL FOLDERS PROCESSED ==========")

def process_simple(video_quality, max_resolution_limit):
    # Calling process_folders with strictly default values yielding two variables cleanly
    for basic_log, raw_log in process_folders(
        DEFAULT_REF_AUDIO, DEFAULT_REF_TEXT, "Chinese (zh)", "Chinese (zh)",
        video_quality=video_quality, restore_model="GFPGAN", resize_factor=1, stop_video=True, random_cut=True,
        max_resolution_limit=max_resolution_limit
    ): 
        yield basic_log, raw_log

# --- GUI DEFINITION ---
custom_css = """
body { font-family: 'Inter', sans-serif; }
.run-btn { background-image: linear-gradient(to right, #43e97b 0%, #38f9d7 100%); color: #000; border: none; padding: 15px; font-weight: bold; border-radius: 8px; transition: transform 0.2s; box-shadow: 0 4px 15px rgba(0,0,0,0.2); }
.run-btn:hover { transform: translateY(-2px); }
.gradio-container { background: #0f172a; } 
div.gradio-container { background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); color: #e2e8f0; }
"""

with gr.Blocks(title="AI Video Studio Automation", css=custom_css, theme=gr.themes.Base()) as demo:
    gr.HTML("<center><h1>🎙️ Auto TTS & Lipsync Studio 🎬</h1><p>Automate your workflow using GPT-SoVITS-v2 and Wav2Lip Studio seamlessly.</p></center>")
    
    with gr.Tabs():
        # SIMPLE MODE TAB
        with gr.TabItem("🚀 Simple Mode"):
            gr.Markdown("### Start the magic with a single click. Uses default optimized settings.")
            gr.Markdown("**Voice Reference**: LuckyV2 (Chinese) | **Face Restoration**: GFPGAN (if High Quality)")
            
            with gr.Row():
                video_quality_simple = gr.Radio(label="Video Quality", choices=["Fast", "High"], value="Fast")
                max_resolution_simple = gr.Slider(label="Maximum Video Resolution (Height/Width)", minimum=240, maximum=2160, step=40, value=1280)
            
            run_btn_simple = gr.Button("▶ START BATCH PROCESSING", elem_classes=["run-btn"])
            
            with gr.Column():
                gr.Markdown("### 👤 User Logs")
                logs_html_simple = gr.HTML()
                
            with gr.Column():    
                gr.Markdown("### 💻 Debug Raw Action Logs")
                logs_box_simple = gr.Textbox(label="Raw Application Logs", lines=10, max_lines=20, interactive=False)
            
            run_btn_simple.click(fn=process_simple, inputs=[video_quality_simple, max_resolution_simple], outputs=[logs_html_simple, logs_box_simple])

        # ADVANCED MODE TAB
        with gr.TabItem("⚙️ Advanced Mode"):
            with gr.Row():
                with gr.Column(scale=2):
                    gr.Markdown("### 🎙️ TTS Reference Config")
                    ref_audio_input = gr.Textbox(label="Reference Audio Path (.wav)", value=DEFAULT_REF_AUDIO, lines=2)
                    ref_text_input = gr.Textbox(label="Reference Text", value=DEFAULT_REF_TEXT, lines=2)
                    
                    with gr.Row():
                        ref_lang_input = gr.Dropdown(label="Reference Language", choices=["Chinese (zh)", "English (en)"], value="Chinese (zh)")
                        target_lang_input = gr.Dropdown(label="Target Spoken Language", choices=["Chinese (zh)", "English (en)", "Japanese (ja)"], value="Chinese (zh)")
                        
                with gr.Column(scale=2):
                    gr.Markdown("### 🎬 Wav2Lip Settings")
                    video_quality_input = gr.Dropdown(label="Video Quality", choices=["High", "Medium", "Fast"], value="High")
                    restore_model_input = gr.Dropdown(label="Face Restoration Model", choices=["GFPGAN", "CodeFormer", "None"], value="GFPGAN")
                    resize_factor_input = gr.Slider(label="Resize Factor (Downscale)", minimum=1, maximum=4, step=1, value=1)
                    max_resolution_adv = gr.Slider(label="Maximum Video Resolution (Height/Width)", minimum=240, maximum=2160, step=40, value=1280)
                    stop_video_input = gr.Checkbox(label="Cut Video exactly at Audio duration stop", value=True)
                    random_cut_input = gr.Checkbox(label="Enable Random Start Point Video Cutter", value=True)
                    
            run_btn_adv = gr.Button("▶ START BATCH PROCESSING", elem_classes=["run-btn"])
            
            with gr.Column():
                gr.Markdown("### 👤 User Logs")
                logs_html_adv = gr.HTML()
                
            with gr.Column():
                gr.Markdown("### 💻 Debug Raw Action Logs")
                logs_box_adv = gr.Textbox(label="Raw Application Logs", lines=10, max_lines=20, interactive=False)
                
            run_btn_adv.click(
                fn=process_folders, 
                inputs=[
                    ref_audio_input, ref_text_input, ref_lang_input, target_lang_input, 
                    video_quality_input, restore_model_input, resize_factor_input, 
                    stop_video_input, random_cut_input, max_resolution_adv
                ], 
                outputs=[logs_html_adv, logs_box_adv]
            )

if __name__ == "__main__":
    demo.launch(inbrowser=True, server_port=7860)
