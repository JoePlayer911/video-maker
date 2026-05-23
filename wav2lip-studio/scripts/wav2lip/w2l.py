import sys
import io
from contextlib import contextmanager
import numpy as np
import gc
import json
import shutil
import copy
import concurrent.futures
from pathlib import Path
from scipy.spatial import ConvexHull
from pydub import AudioSegment
import cv2, os, scripts.wav2lip.audio as audio
import subprocess
from types import SimpleNamespace
from imutils import face_utils
from modules.shared import opts
import modules.face_restoration as fr
from tqdm import tqdm
import torch
from scripts.utils.logger import Logger
from scripts.wav2lip.models import Wav2Lip
from pkg_resources import resource_filename


@contextmanager
def suppress_stdout():
    stdout_original = sys.stdout
    stderr_original = sys.stderr
    sys.stdout = sys.stderr = io.StringIO()
    sys.stdout = io.StringIO()
    try:
        yield
    finally:
        sys.stdout = stdout_original
        sys.stderr = stderr_original


class W2l:
    def __init__(self, project_name, video_quality, volume_amplifier):
        self.wav2lip_studio = os.path.sep.join(os.path.abspath(__file__).split(os.path.sep)[:-3])
        self.projects_folder = os.path.join(self.wav2lip_studio, "projects")
        self.project_folder = os.path.join(self.projects_folder, project_name)
        self.model_folder = os.path.join(self.wav2lip_studio, 'models')
        self.faceswap_output_folder = os.path.join(self.project_folder, 'faceswap')
        self.wav2lip_output_folder = os.path.join(self.project_folder, 'wav2lip')
        self.audio_folder = os.path.join(self.project_folder, 'audio')
        self.temp_folder = os.path.join(self.project_folder, 'temp')
        self.analyse_folder = os.path.join(self.project_folder, 'analyse')

        self.img_size = 96
        self.static = False
        #self.audio = audio_path
        self.video_quality = video_quality
        self.volume_amplifier = volume_amplifier
        # self.enhance_plus = enhance_plus
        self.mel_step_size = 16
        self.face_det_batch_size = 16
        self.wav2lip_batch_size = 128
        self.crop = [0, -1, 0, -1]

        full_frames, nb_frame, self.keyframes, self.frames, self.video_properties = self.load()
        self.face = self.video_properties['video_path']
        self.audio_file = self.video_properties['audio_file']
        self.audio_generated = self.video_properties['audio_generated']
        self.audio_type = self.video_properties['audio_type']
        self.logger = Logger()
        translates = [x for x in self.video_properties['translate'] if x['language'] == self.video_properties["language_translate"]]
        self.translated_audio = ""
        if translates and self.video_properties["audio_type"] == "Input Video":
            self.translated_audio = translates[0]["audio"]
            self.video_output = self.wav2lip_output_folder + '/video_translated_'+translates[0]["language"]+'.mp4'
        else:
            self.video_output = self.wav2lip_output_folder + '/video.mp4'

        self.face_swap_img = self.video_properties['face_swap_img']
        self.sound_start_frame = self.video_properties['audio_start_frame']
        self.stop_after_audio = self.video_properties['audio_stop_video']
        #self.nosmooth = self.video_properties['mouth_smooth']
        self.align_face = self.video_properties['align_face']
        self.only_mouth = self.video_properties["mouth_only"]
        self.resize_factor = self.video_properties["resize_factor"]
        self.checkpoint = self.video_properties["wav2lip_checkpoint"]
        self.checkpoint_path = self.model_folder + '/Wav2lip/' + self.video_properties["wav2lip_checkpoint"] + '.pth'
        self.video_no_sound = self.wav2lip_output_folder + '/video_no_sound.mp4'

        self.face_restore_model = "GFPGAN"
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.mouth = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
        self.jaw = face_utils.FACIAL_LANDMARKS_IDXS["jaw"]
        self.nose = face_utils.FACIAL_LANDMARKS_IDXS["nose"]
        self.logger.info("{0}{1}", "Device: ", self.device)
        self.ffmpeg_binary = self.find_ffmpeg_binary()

    @staticmethod
    def find_ffmpeg_binary():
        for package in ['imageio_ffmpeg', 'imageio-ffmpeg']:
            try:
                package_path = resource_filename(package, 'binaries')
                files = [os.path.join(package_path, f) for f in os.listdir(package_path) if f.startswith("ffmpeg-")]
                files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
                return files[0] if files else 'ffmpeg'
            except:
                return 'ffmpeg'

    @staticmethod
    def execute_command(command):
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            raise RuntimeError(stderr)

    def add_delay(self, audio_path):
        delay = str(int(self.sound_start_frame / self.video_properties['fps'] * 1000))
        self.logger.info("{0}{1}", 'Adding delay : ', delay)
        command = [self.ffmpeg_binary, "-y", "-i", audio_path, "-af", "adelay=" + delay + "|" + delay,
                   self.audio_folder + "/audio_delayed.wav"]
        self.execute_command(command)
        return self.audio_folder + "/audio_delayed.wav"

    def add_audio_to_video(self, audio_path):
        self.logger.info("{0}{1}", 'Adding audio to video', "")
        translation = [x for x in self.video_properties['translate'] if x['language'] == self.video_properties["language_translate"]]
        if self.video_properties["audio_type"] =='Input Video' and translation:
            piste = AudioSegment.from_file(translation[0]["audio"], format="wav")
            background = AudioSegment.from_file(self.video_properties['background'], format="wav")
            background = background.overlay(piste, position=0)
            background.export(self.temp_folder+"/audio.wav", format="wav")
            audio_path = self.temp_folder+"/audio.wav"

        command = [self.ffmpeg_binary, "-y", "-i", f"{self.video_no_sound}",
                   "-i", audio_path,
                   "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-strict",
                   "experimental", f"{self.video_output}"]
        self.execute_command(command)

    @staticmethod
    def get_smoothened_boxes(boxes, T):
        for i in range(len(boxes)):
            if i + T > len(boxes):
                window = boxes[len(boxes) - T:]
            else:
                window = boxes[i: i + T]
            boxes[i] = np.mean(window, axis=0)
        return boxes

    def face_detect(self, images):
        results = []
        targets = []
        self.logger.info("{0}{1}", 'Detecting faces', "")
        for i in range(len(images)):
            keyframe = [x for x in self.keyframes if x["frame"] == str(i + 1)]
            if keyframe:
                last_keyframe = keyframe[0]

            current_frame = self.frames[str(i % len(images) + 1)] if str(i % len(images) + 1) in self.frames else None

            if current_frame["faces"] > 0:
                speaker = [x for x in last_keyframe["faces_properties"] if x["speaker"] == True]
                if speaker and int(speaker[0]["id"]) < len(current_frame["target_face"]):
                    pady1, pady2, padx1, padx2 = speaker[0]["padding"]
                    target = current_frame["target_face"][int(speaker[0]["id"])]
                    y1 = int(max(0, target.bbox[1] - pady1))
                    y2 = int(min(images[i].shape[0], target.bbox[3] + pady2))
                    x1 = int(max(0, target.bbox[0] - padx1))
                    x2 = int(min(images[i].shape[1], target.bbox[2] + padx2))
                    #img = images[i][y1:y2, x1:x2]
                    if self.align_face and 'center' in target.__dict__:
                        M = cv2.getRotationMatrix2D(target.center, target.angle, 1)
                        images[i] = cv2.warpAffine(images[i], M, (images[i].shape[1], images[i].shape[0]))
                        #save image
                        #cv2.imwrite(f"{self.project_folder}/debug/image{i}.jpg", images[i])

                    results.append([x1, y1, x2, y2])
                    targets.append(target)
                else:
                    results.append([-1, -1, -1, -1])
                    targets.append(None)
            else:
                results.append([-1, -1, -1, -1])
                targets.append(None)
        boxes = np.array(results)
        #if not self.nosmooth:
        #    boxes = self.get_smoothened_boxes(boxes, T=5)

        results = [[image[y1: y2, x1:x2], (y1, y2, x1, x2), target] for image, (x1, y1, x2, y2), target in
                   zip(images, boxes, targets)]

        return results

    def datagen(self, frames, mels):
        self.logger.info("{0}{1}", 'Generating data', "")
        img_batch, mel_batch, frame_batch, coords_batch, targets_batch = [], [], [], [], []

        if not self.static:
            face_det_results = self.face_detect(frames)
        else:
            face_det_results = self.face_detect(frames)

        no_face_frames = []
        output = []
        last_mel_frame = 0
        for i, m in enumerate(mels):
            idx = 0 if self.static else i % len(frames)
            frame_to_save = frames[idx].copy()
            face, coords, target = face_det_results[idx].copy()

            if coords[0] != -1:
                #cv2.imwrite(f"{self.project_folder}/debug/face{i}.jpg", face)
                face = cv2.resize(face, (self.img_size, self.img_size))
                img_batch.append(face)
                mel_batch.append(m)
                frame_batch.append(frame_to_save)
                coords_batch.append(coords)
                targets_batch.append(target)
                last_mel_frame = i
            else:
                no_face_frames.append(i)

            if len(img_batch) >= self.wav2lip_batch_size:
                img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

                img_masked = img_batch.copy()
                img_masked[:, self.img_size // 2:] = 0

                img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
                mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

                output.append((img_batch, mel_batch, frame_batch, coords_batch, targets_batch))
                img_batch, mel_batch, frame_batch, coords_batch, targets_batch = [], [], [], [], []

        if len(img_batch) > 0:
            img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

            img_masked = img_batch.copy()
            img_masked[:, self.img_size // 2:] = 0

            img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
            mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

            output.append((img_batch, mel_batch, frame_batch, coords_batch, targets_batch))

        return output, no_face_frames, last_mel_frame

    def _load(self, checkpoint_path):
        self.logger.info("{0}{1}", 'model path: ', checkpoint_path)
        if self.device == 'cuda':
            checkpoint = torch.load(checkpoint_path)
        else:
            checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        return checkpoint

    def load_model(self, path):
        model = Wav2Lip()
        self.logger.info("{0}{1}", 'Loading wav2lip model', "")
        checkpoint = self._load(path)
        s = checkpoint["state_dict"]
        new_s = {}
        for k, v in s.items():
            new_s[k.replace('module.', '')] = v
        model.load_state_dict(new_s)

        model = model.to(self.device)
        return model.eval()

    def get_mel_chunks(self):
        if self.sound_start_frame > 0:
            self.audio = self.add_delay(self.audio)
        wav = audio.load_wav(self.audio, 16000)
        #augmenter le volume
        wav = wav * self.volume_amplifier
        mel = audio.melspectrogram(wav)
        self.logger.info("{0}{1}", 'Mel shape: ', mel.shape)

        if np.isnan(mel.reshape(-1)).sum() > 0:
            raise ValueError(
                'Mel contains nan! Using a TTS voice? Add a small epsilon noise to the wav file and try again')

        mel_chunks = []
        mel_idx_multiplier = 80. / self.video_properties['fps']
        i = 0
        while 1:
            start_idx = int(i * mel_idx_multiplier)
            if start_idx + self.mel_step_size > len(mel[0]):
                mel_chunks.append(mel[:, len(mel[0]) - self.mel_step_size:])
                break
            mel_chunks.append(mel[:, start_idx: start_idx + self.mel_step_size])
            i += 1
        return mel_chunks

    def get_face_info(self, rect):
        jaw = rect.landmark_3d_68[self.jaw[0]:self.jaw[1]][1:-1]
        jaw = [[int(sublist[0]), int(sublist[1])] for sublist in jaw]
        nose = rect.landmark_3d_68[self.nose[0]:self.nose[1]][2][:2]
        nose = [int(nose[0]), int(nose[1])]

        mouth = rect.landmark_3d_68[self.mouth[0]:self.mouth[1]][:-8]
        mouth = [[int(sublist[0]), int(sublist[1])] for sublist in mouth]
        mouth = np.delete(mouth, [3], axis=0)
        return jaw, nose, mouth

    @staticmethod
    def dilate_mouth(mouth, w, h, mouth_mask_dilatation):
        mask = np.zeros((w, h), dtype=np.uint8)
        cv2.fillPoly(mask, [mouth], 255)
        kernel = np.ones((mouth_mask_dilatation, mouth_mask_dilatation), np.uint8)
        dilated_mask = cv2.dilate(mask, kernel, iterations=1)
        contours, _ = cv2.findContours(dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        dilated_points = contours[0].squeeze()
        return dilated_points

    def create_mask(self, original_gray, image_restored_gray, face_property, jaw, nose, mouth):
        mask = np.zeros_like(original_gray)
        if not self.only_mouth:
            external_shape = np.append(jaw, [nose], axis=0)
            hull = ConvexHull(external_shape)
            hull_points = external_shape[hull.vertices]
            external_shape_pts = hull_points.reshape((-1, 1, 2))
            mask = cv2.fillPoly(mask, [external_shape_pts], 255)
            if face_property["erode_face_mask"] > 0:
                kernel = np.ones((face_property["erode_face_mask"], face_property["erode_face_mask"]), np.uint8)
                mask = cv2.erode(mask, kernel, iterations=1)
            # Calculate diff between frames and apply threshold
            diff = np.abs(original_gray - image_restored_gray)
            diff[diff > 10] = 255
            diff[diff <= 10] = 0
            masked_diff = cv2.bitwise_and(diff, diff, mask=mask)
        else:
            masked_diff = np.zeros_like(original_gray)
        # WARNINGGGGGGGGGGGGGGGGGGG
        if face_property["mouth_mask_dilatation"] > 0:
            mouth_mask_dilatation = face_property["mouth_mask_dilatation"]
            mouth = self.dilate_mouth(mouth, original_gray.shape[0], original_gray.shape[1], mouth_mask_dilatation)
        masked_diff = cv2.fillConvexPoly(masked_diff, mouth, 255)

        # Save mask
        if face_property["mask_blur"] > 0:
            blur = face_property["mask_blur"] if face_property["mask_blur"] % 2 == 1 else face_property[
                                                                                              "mask_blur"] - 1
            masked_save = cv2.GaussianBlur(masked_diff, (blur, blur), 0)
        else:
            masked_save = masked_diff
        return masked_save

    @staticmethod
    def restore_face(cropped_face, opts):
        cropped_face = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)
        cropped_face_restored = fr.restore_faces(cropped_face, opts.face_restoration_model)
        image_restored = cv2.cvtColor(cropped_face_restored, cv2.COLOR_RGB2BGR)
        return image_restored

    @staticmethod
    def load_image(chemin):
        return cv2.imread(chemin)

    def execute(self):
        if not os.path.isfile(self.face):
            raise ValueError('--face argument must be a valid path to video/image file')

        elif self.face.split('.')[1] in ['jpg', 'png', 'jpeg']:
            full_frames = [cv2.imread(self.face)]
        else:
            self.logger.info("{0}{1}", 'Reading video frames...', "")
            image_pattern = "frame_*.png"
            if self.face_swap_img is not None:
                self.logger.info("{0}{1}", 'Reading face swap frames...', "")
                image_pattern = "face_swap_*.png"

            frames = sorted([f"{self.analyse_folder}/{f.name}" for f in Path(f"{self.analyse_folder}/").glob(image_pattern)])
            with concurrent.futures.ThreadPoolExecutor() as executor:
                all_frames = list(executor.map(self.load_image, frames))
            full_frames = copy.deepcopy(all_frames)

        self.logger.info("{0}{1}", "Number of frames available for inference: ", str(len(full_frames)))
        if self.audio_type == "Input Video":
            if self.translated_audio != '':
                self.audio = self.translated_audio
            else:
                self.logger.info("{0}{1}", "Extracting raw audio from video...", "")
                command = [self.ffmpeg_binary, "-i", self.face, "-vn", "-acodec",
                           "pcm_s16le", "-ar", "44100", "-y", "-ac", "2", self.audio_folder + "/audio_video.wav"]

                self.execute_command(command)
                self.audio = self.audio_folder + "/audio_video.wav"
        elif self.audio_type == "File":
            if not self.audio_file.endswith('.wav'):
                self.logger.info("{0}{1}", "Extracting raw audio...", "")
                command = [self.ffmpeg_binary, "-y", "-i", self.audio_file, "-strict", "-2",
                           self.audio_folder + "/audio_file.wav"]
                self.execute_command(command)
            else:
                shutil.copyfile(f"{self.audio_file}", self.audio_folder + "/audio_file.wav")
            self.audio = self.audio_folder + "/audio_file.wav"
        else:
            self.audio = self.audio_generated

        mel_chunks = self.get_mel_chunks()
        self.logger.info("{0}{1}", "Number of mel chunks available for inference: ", str(len(mel_chunks)))

        frame_h, frame_w = full_frames[0].shape[:-1]
        video_output = cv2.VideoWriter(f"{self.video_no_sound}",
                                       cv2.VideoWriter_fourcc(*'DIVX'), self.video_properties['fps'],
                                       (frame_w, frame_h))

        batch_size = self.wav2lip_batch_size
        gen, no_face_frames, last_mel_frame = self.datagen(full_frames, mel_chunks)

        preds = None
        pred_coords = []
        pred_frame = []
        pred_targets = []
        for i, (img_batch, mel_batch, frames, coords, targets) in enumerate(tqdm(gen,
                                                                                 total=int(
                                                                                     np.ceil(
                                                                                         float(
                                                                                             len(mel_chunks)) / batch_size)))):
            if i == 0:
                model = self.load_model(self.checkpoint_path)
                self.logger.info("{0}{1}", "Model loaded", "")
            img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(self.device)
            mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(self.device)
            with torch.no_grad():
                pred = model(mel_batch, img_batch)

            pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.
            pred_coords += coords
            pred_frame += frames
            pred_targets += targets
            if preds is None:
                preds = pred
            else:
                preds = np.concatenate((preds, pred), axis=0)

        nb_frames = len(mel_chunks) if len(mel_chunks) > len(all_frames) else len(all_frames)
        if self.stop_after_audio and self.audio_type in ["File", "Generate"]:
            nb_frames = last_mel_frame
        current_frame = 0
        opts.face_restoration_model = self.face_restore_model
        def _process_frame(state):
            i, last_keyframe, speaker_id, needs_processing, c_frame = state
            frame = all_frames[i % len(all_frames)].copy()
            
            if not needs_processing:
                return frame
                
            p, f, c, target = preds[c_frame], pred_frame[c_frame], pred_coords[c_frame], pred_targets[c_frame]
            y1, y2, x1, x2 = c
            p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))
            
            if self.video_quality in ["Medium", "High"]:
                image_restored = frame.copy()
                if self.video_quality == "High":
                    p = self.restore_face(p, opts)

                if self.align_face and "center" in target.__dict__:
                    center = (int(target.center[0]-x1), int(target.center[1]-y1))
                    M = cv2.getRotationMatrix2D(center, -target.angle, 1)
                    p = cv2.warpAffine(p, M, (p.shape[1], p.shape[0]))

                    masque = np.ones_like(p, dtype=np.uint8) * 255
                    rotated_masque = cv2.warpAffine(masque, M, (p.shape[1], p.shape[0]))
                    masque_binaire = cv2.cvtColor(rotated_masque, cv2.COLOR_BGR2GRAY)
                    p = np.where(masque_binaire[..., None] == 255, p, image_restored[y1:y2, x1:x2])
                
                image_restored[y1:y2, x1:x2] = p
                image_restored_gray = cv2.cvtColor(image_restored, cv2.COLOR_RGB2GRAY)
                jaw, nose, mouth = self.get_face_info(target)
                original_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                
                face_property = last_keyframe["faces_properties"][speaker_id]
                masked_save = self.create_mask(original_gray, image_restored_gray, face_property, jaw, nose, mouth)

                extended_mask = np.stack([masked_save] * 3, axis=-1)
                normalized_mask = extended_mask / 255.0
                dst = image_restored * normalized_mask
                original_frame = frame * (1 - normalized_mask) + dst
                original_frame = original_frame.astype(np.uint8)
                frame = original_frame
            else:
                if self.align_face and "angle" in target.__dict__:
                    center = (int(target.center[0] - x1), int(target.center[1] - y1))
                    M = cv2.getRotationMatrix2D(center, -target.angle, 1)
                    p = cv2.warpAffine(p, M, (p.shape[1], p.shape[0]))

                    masque = np.ones_like(p, dtype=np.uint8) * 255
                    rotated_masque = cv2.warpAffine(masque, M, (p.shape[1], p.shape[0]))
                    masque_binaire = cv2.cvtColor(rotated_masque, cv2.COLOR_BGR2GRAY)
                    p = np.where(masque_binaire[..., None] == 255, p, frame[y1:y2, x1:x2])

                frame[y1:y2, x1:x2] = p
                
            return frame

        frame_states = []
        last_keyframe = None
        speaker_id = None
        current_frame = 0
        
        for i in range(nb_frames):
            keyframe = [x for x in self.keyframes if x["frame"] == str(i + 1)]
            if keyframe:
                last_keyframe = keyframe[0]
                if last_keyframe["faces"] > 0:
                    speaker = any([True for x in last_keyframe["faces_properties"] if x["speaker"] == True])
                    if speaker:
                        speaker_id = [x for x, key in enumerate(last_keyframe["faces_properties"]) if key["speaker"] == True][0]
            
            needs_processing = (i not in no_face_frames) and (i < len(mel_chunks) + self.sound_start_frame)
            frame_states.append((i, last_keyframe, speaker_id, needs_processing, current_frame))
            if needs_processing:
                current_frame += 1

        for i, state in enumerate(frame_states):
            result_frame = _process_frame(state)
            print(f"[INFO] saving: {i} of {nb_frames} - ", end="\r")
            video_output.write(result_frame)
        # release memory
        video_output.release()
        model.cpu()
        del model
        torch.cuda.empty_cache()
        gc.collect()

        self.add_audio_to_video(self.audio)
        self.save_generation_data()

        return self.video_output, nb_frames

    def save_generation_data(self):
        shutil.copyfile(f"{self.project_folder}/keyframes.json", self.temp_folder + "/keyframes.json")
        shutil.copyfile(f"{self.project_folder}/project.json", self.temp_folder + "/project.json")

    @staticmethod
    def convert_to_object(faces):
        targets = []
        for i, target in enumerate(faces):
            new_target = {}
            new_target["bbox"] = np.array(target["bbox"]).astype(np.float64)
            new_target["center"] = target["center"]
            new_target["angle"] = np.float64(target["angle"])
            new_target["det_score"] = np.float32(target["det_score"])
            new_target["embedding"] = np.array(target["embedding"]).astype(np.float32)
            new_target["embedding_norm"] = np.float32(target["embedding_norm"])
            new_target["gender"] = np.int64(target["gender"])
            new_target["kps"] = np.array(target["kps"]).astype(np.float32)
            new_target["landmark_2d_106"] = np.array(target["landmark_2d_106"]).astype(np.float32)
            new_target["landmark_3d_68"] = np.array(target["landmark_3d_68"]).astype(np.float32)
            new_target["pose"] = np.array(target["pose"]).astype(np.float32)
            new_target["sex"] = target["sex"]
            targets.append(SimpleNamespace(**new_target))
        return targets

    def load(self):
        full_frames = []
        if os.path.exists(self.project_folder):
            project_properties = {}

            if os.path.exists(f"{self.project_folder}/keyframes.json"):
                with open(f"{self.project_folder}/keyframes.json", "r") as f:
                    self.keyframes = json.load(f)
            if os.path.exists(f"{self.project_folder}/full_frames.json"):
                with open(f"{self.project_folder}/full_frames.json", "r") as f:
                    full_frames = json.load(f)
            if os.path.exists(f"{self.project_folder}/frames.json"):
                with open(f"{self.project_folder}/frames.json", "r") as f:
                    self.frames = json.load(f)
            if os.path.exists(f"{self.project_folder}/project.json"):
                with open(f"{self.project_folder}/project.json", "r") as f:
                    project_properties = json.load(f)

            if "source_faces" in project_properties:
                targets = self.convert_to_object(project_properties["source_faces"])
                project_properties["source_faces"] = targets

            frames = {}
            for x, frame in enumerate(self.frames.keys()):
                new_frame = {"frame": self.frames[frame]["frame"], "faces": self.frames[frame]["faces"]}
                if "target_face" in self.frames[frame]:
                    new_frame["target_face"] = self.convert_to_object(self.frames[frame]["target_face"])
                frames[frame] = new_frame
            self.frames = frames
        return full_frames, len(full_frames), self.keyframes, self.frames, project_properties
