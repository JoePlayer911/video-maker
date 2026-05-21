import argparse
import sys
import os

# Ensure local directories are prioritized in sys.path to avoid module shadowing on portable envs
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from scripts.analyse.analyse import Analyse
from scripts.wav2lip.w2l import W2l

def run(project_name, video_path, audio_path, video_quality, restore_model, resize_factor, stop_video):
    
    # Defaults based on ui.py initialization
    video_properties = {
        "project_name": project_name,
        "video_path": video_path,
        "face_swap_img": None,
        "resize_factor": resize_factor,
        "minimum_face_size": 30,
        "align_face": False,
        "mouth_only": False,
        "wav2lip_checkpoint": "wav2lip_gan" if video_quality != "Fast" else "wav2lip",
        "face_restore_model": restore_model,
        "code_former_weights": 0.75,
        "audio_type": "File",
        "audio_start_frame": 0,
        "audio_stop_video": stop_video,
        "audio_file": audio_path,
        "audio_generated": None,
        "language": "en",
        "translate": [],
        "whisper_model": "medium",
        "language_translate": "en",
        "nb_speakers": 1,
        "voice": "female_01.wav",
        "prompt": "",
        "faceswap_video": None,
        "wav2lip_video": None,
        "volume_amplifier": 1,
        "language": "en",
        "translate": []
    }
    
    # FIX: If the project workspace already exists from a previous run, old frame_*.png bleed over!
    # Delete the project directory completely to ensure a fresh session.
    import os, shutil
    wav2lip_studio = os.path.sep.join(os.path.abspath(__file__).split(os.path.sep)[:-1])
    project_folder = os.path.join(wav2lip_studio, "projects", project_name)
    if os.path.exists(project_folder):
        print(f"[auto_wav2lip] Wiping old project cache at {project_folder}...")
        try:
            shutil.rmtree(project_folder)
        except Exception as e:
            print(f"[ERROR] Could not clear old project cache: {e}")
            
    print(f"[auto_wav2lip] Starting Video Analysis...")
    analyse = Analyse(
        project_name, 
        video_path, 
        resize_factror=resize_factor, 
        face_swap=None, 
        only_mouth=False, 
        minimum_face_size=30,
        keyframes_on_speakers=False, 
        keyframe_on_scenes=False
    )
    frames, nb_frames, keyframes, all_frames, video_properties = analyse.execute(video_properties)
    
    # FIX: wav2lip-studio's Analyse.execute only copies the video to project/input if the path contains 'gradio'
    # We must explicitly copy the video into the project workspace for W2l class!
    target_video_path = video_properties["video_path"]
    if not os.path.exists(target_video_path):
        import shutil
        shutil.copyfile(video_path, target_video_path)
        
    analyse.save_project_properties(video_properties)
    print(f"[auto_wav2lip] Video analyzed. Frames: {nb_frames}")
    
    # 2. Wav2Lip Inference
    print("[auto_wav2lip] Starting Wav2lip inference...")
    w2l = W2l(project_name, video_quality, volume_amplifier=1)
    output_video_path, nb_frame = w2l.execute()
    video_properties["wav2lip_video"] = output_video_path
    analyse.save_project_properties(video_properties)
    
    print(f"[auto_wav2lip] DONE! Generated video at: {output_video_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--video_quality", default="High")
    parser.add_argument("--restore_model", default="GFPGAN")
    parser.add_argument("--resize_factor", type=int, default=1)
    parser.add_argument("--stop_video", action="store_true")
    args = parser.parse_args()
    
    run(args.project, args.video, args.audio, args.video_quality, args.restore_model, args.resize_factor, args.stop_video)
