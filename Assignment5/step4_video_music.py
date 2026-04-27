"""
STEP 4 : Video Slideshow + Algorithmic Music Integration
=========================================================
For each cluster (or a selected subset) this script:
  1. Analyses the cluster's visual properties (brightness, colour temperature,
     saturation, edge density) to score it on a warm/cool and calm/energetic axis.
  2. Maps those scores algorithmically to a musical profile
     (tempo BPM, base frequency, scale type, note duration).
  3. Generates a unique background audio track using pure NumPy/SciPy sine
     synthesis – no external music files needed, fully algorithmic.
  4. Builds a slideshow video from the cluster's images using MoviePy.
  5. Mixes the generated audio into the video and saves the final .mp4.

"""

import os
import glob
import numpy as np
from PIL import Image
from tqdm import tqdm

# Audio / video
import scipy.io.wavfile as wavfile
from moviepy import ImageClip, concatenate_videoclips, AudioFileClip

import config

# Constants 

SR   = config.AUDIO_SAMPLE_RATE  
CLIP = config.SECONDS_PER_IMAGE   
FPS  = config.VIDEO_FPS


# Visual feature analysis

def analyse_cluster_visuals(filepaths: list, sample_n: int = 50) -> dict:
    rng    = np.random.default_rng(42)
    sample = rng.choice(filepaths,
                        size=min(sample_n, len(filepaths)),
                        replace=False).tolist()

    brightness_list, warmth_list, sat_list, edge_list = [], [], [], []

    for fp in sample:
        try:
            img_rgb = np.array(Image.open(fp).convert("RGB").resize((64, 64)),
                               dtype=np.float32)
            r, g, b = img_rgb[:,:,0], img_rgb[:,:,1], img_rgb[:,:,2]

            brightness_list.append(img_rgb.mean())
            warmth_list.append(r.mean() - b.mean())          # positive = warm

            # HSV-based saturation (simplified)
            mx   = img_rgb.max(axis=2)
            mn   = img_rgb.min(axis=2)
            diff = mx - mn
            sat  = np.where(mx > 0, diff / mx, 0).mean()
            sat_list.append(float(sat))

            # Edge density via simple gradient
            gray  = img_rgb.mean(axis=2)
            gx    = np.abs(np.diff(gray, axis=1)).mean()
            gy    = np.abs(np.diff(gray, axis=0)).mean()
            edge_list.append((gx + gy) / 2.0)

        except Exception:
            pass

    return {
        "brightness"  : float(np.mean(brightness_list)) if brightness_list else 128.0,
        "warmth"      : float(np.mean(warmth_list))     if warmth_list     else 0.0,
        "saturation"  : float(np.mean(sat_list))        if sat_list        else 0.5,
        "edge_density": float(np.mean(edge_list))       if edge_list       else 5.0,
    }


# Music profile derivation.

# Musical scales as semitone intervals above root
SCALES = {
    "major"            : [0, 2, 4, 5, 7, 9, 11],
    "minor"            : [0, 2, 3, 5, 7, 8, 10],
    "pentatonic_major" : [0, 2, 4, 7, 9],
    "pentatonic_minor" : [0, 3, 5, 7, 10],
    "dorian"           : [0, 2, 3, 5, 7, 9, 10],
}


def derive_music_profile(visuals: dict) -> dict:
    """
    Map visual properties to a musical profile:
    """
    warmth       = visuals["warmth"]           # −255 … +255
    edge         = visuals["edge_density"]     # 0 … ~30
    brightness   = visuals["brightness"]       # 0 … 255
    saturation   = visuals["saturation"]       # 0 … 1

    # Normalise warmth & edge to [0, 1]
    warmth_norm = np.clip((warmth + 50) / 100.0, 0, 1)
    edge_norm   = np.clip(edge / 20.0, 0, 1)

    # BPM: 50 (slow, glacial) → 120 (energetic, urban)
    bpm   = int(50 + warmth_norm * 30 + edge_norm * 40)

    # Scale selection
    if warmth_norm > 0.6 and edge_norm > 0.5:
        scale_name = "major"
        mood_label = "Energetic / Urban"
    elif warmth_norm < 0.4 and edge_norm < 0.4:
        scale_name = "pentatonic_minor"
        mood_label = "Calm / Glacial"
    elif warmth_norm > 0.5:
        scale_name = "pentatonic_major"
        mood_label = "Warm / Natural"
    elif edge_norm < 0.3:
        scale_name = "dorian"
        mood_label = "Meditative / Forest"
    else:
        scale_name = "minor"
        mood_label = "Dramatic / Mountain"

    # Root frequency: A3 (220 Hz) to A4 (440 Hz) mapped to brightness
    root_freq = 220.0 + (brightness / 255.0) * 220.0

    # Harmonic richness (number of overtones): more saturated → richer
    n_harmonics = int(2 + saturation * 5)

    return {
        "bpm"         : bpm,
        "scale_name"  : scale_name,
        "root_freq"   : root_freq,
        "n_harmonics" : n_harmonics,
        "mood_label"  : mood_label,
    }


# Audio synthesis

def semitone_to_freq(root_hz: float, semitone: int) -> float:
    return root_hz * (2 ** (semitone / 12.0))


def sine_wave(freq: float, duration_s: float, amplitude: float = 0.3,
              n_harmonics: int = 3) -> np.ndarray:
    """
    Generate a sine tone with overtones (harmonic synthesis).
    The fundamental has amplitude A, each harmonic decays by 1/n.
    """
    t      = np.linspace(0, duration_s, int(SR * duration_s), endpoint=False)
    signal = np.zeros_like(t)
    for h in range(1, n_harmonics + 1):
        signal += (amplitude / h) * np.sin(2 * np.pi * freq * h * t)
    return signal


def apply_envelope(signal: np.ndarray, attack_s: float = 0.05,
                   release_s: float = 0.1) -> np.ndarray:
    """Simple linear attack / release envelope to avoid clicks."""
    if len(signal) == 0:
        return signal
    atk  = min(int(attack_s  * SR), len(signal) // 2)
    rel  = min(int(release_s * SR), len(signal) // 2)
    env  = np.ones(len(signal))
    if atk > 0:
        env[:atk]  = np.linspace(0, 1, atk)
    if rel > 0:
        env[-rel:] = np.linspace(1, 0, rel)
    return signal * env


def generate_audio(profile: dict, total_duration_s: float) -> np.ndarray:
    """
    Generate a melodic audio track matching the given musical profile.
    Notes are picked from the chosen scale, sequenced at the given BPM.
    """
    bpm         = profile["bpm"]
    scale_name  = profile["scale_name"]
    root_freq   = profile["root_freq"]
    n_harmonics = profile["n_harmonics"]

    beat_s      = 60.0 / bpm          # seconds per beat
    note_dur    = beat_s * 1.5        # slightly longer than one beat (overlap)

    intervals   = SCALES[scale_name]
    # Build 2 octaves of the scale for melodic variety
    two_oct     = intervals + [i + 12 for i in intervals]

    rng         = np.random.default_rng(42)
    # Shuffle to create a non-trivial melodic sequence (deterministic)
    melody_seq  = rng.permutation(two_oct).tolist()
    melody_seq  = (melody_seq * 20)[:int(total_duration_s / beat_s) + 4]

    track       = np.zeros(int(SR * total_duration_s))
    t_cursor    = 0.0

    for semitone in melody_seq:
        if t_cursor >= total_duration_s:
            break
        freq   = semitone_to_freq(root_freq, semitone)
        dur    = min(note_dur, total_duration_s - t_cursor)
        note   = apply_envelope(
                    sine_wave(freq, dur, amplitude=0.25, n_harmonics=n_harmonics))
        start  = int(t_cursor * SR)
        end    = start + len(note)
        if end <= len(track):
            track[start:end] += note
        t_cursor += beat_s

    # Add a quiet bass drone on the root
    drone = sine_wave(root_freq / 2, total_duration_s, amplitude=0.08, n_harmonics=2)
    track += drone

    # Normalise to prevent clipping
    peak = np.abs(track).max()
    if peak > 0:
        track = track / peak * 0.85

    return (track * 32767).astype(np.int16)


# Video creation
def create_slideshow_video(filepaths: list, audio_path: str,
                           output_path: str, cluster_id: int,
                           profile: dict):
    print(f"  Building video for cluster {cluster_id} ({len(filepaths)} images) …")

    clips = []
    for fp in tqdm(filepaths, desc="  Loading frames", leave=False):
        try:
            img  = Image.open(fp).convert("RGB").resize(
                       (config.VIDEO_SIZE[0], config.VIDEO_SIZE[1]))
            arr  = np.array(img)
            clip = ImageClip(arr, duration=config.SECONDS_PER_IMAGE)
            clips.append(clip)
        except Exception as e:
            print(f"    [SKIP] {fp}: {e}")

    if not clips:
        print(f"  [ERROR] No valid images for cluster {cluster_id}")
        return

    slideshow = concatenate_videoclips(clips, method="compose")

    # Load generated audio
    audio     = AudioFileClip(audio_path)

    # Loop audio if video is longer, trim if shorter
    total_dur = slideshow.duration
    if audio.duration < total_dur:
        loops = int(np.ceil(total_dur / audio.duration))
        from moviepy import concatenate_audioclips
        audio = concatenate_audioclips([audio] * loops)
    audio = audio.subclipped(0, total_dur)

    final = slideshow.with_audio(audio)
    final.write_videofile(
        output_path,
        fps=FPS,
        codec="libx264",
        audio_codec="aac",
        logger=None,
    )
    print(f"  [Saved] {output_path}")


# Main execution

def main():
    os.makedirs(config.VIDEO_DIR,  exist_ok=True)
    os.makedirs(config.PLOTS_DIR,  exist_ok=True)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    print("\n========== STEP 4: Video + Music Generation ==========\n")

    # Load data ──────────────────────────────────────────────────────────────
    for f in [config.FEATURES_FILE, config.FILEPATHS_FILE, config.CLUSTERS_FILE]:
        if not os.path.exists(f):
            raise FileNotFoundError(f"Missing {f} – run previous steps first.")

    filepaths = list(np.load(config.FILEPATHS_FILE, allow_pickle=True))
    labels    = np.load(config.CLUSTERS_FILE)
    n_clusters = config.N_CLUSTERS

    # Deciding the clusters to generate videos for
    cluster_sizes = {c: int((labels == c).sum()) for c in range(n_clusters)}
    largest       = max(cluster_sizes, key=cluster_sizes.get)
    # Select up to 3 clusters: largest + next two biggest
    sorted_clusters = sorted(cluster_sizes, key=cluster_sizes.get, reverse=True)
    video_clusters  = sorted_clusters[:min(3, n_clusters)]

    print(f"Cluster sizes     : {cluster_sizes}")
    print(f"Generating videos for clusters: {video_clusters}\n")

    music_summary = []

    for c in video_clusters:
        print(f"── Cluster {c} ({cluster_sizes[c]} images) ─────────────────────────────────")

        cluster_paths = [filepaths[i] for i, lbl in enumerate(labels) if lbl == c]

        # Limit to 50 images for the video (keeps file size manageable)
        rng    = np.random.default_rng(config.RANDOM_SEED)
        subset = rng.choice(cluster_paths,
                            size=min(50, len(cluster_paths)),
                            replace=False).tolist()

        # 1. Analyse visuals
        visuals = analyse_cluster_visuals(cluster_paths)
        print(f"  Visual properties: {visuals}")

        # 2. Derive music profile
        profile = derive_music_profile(visuals)
        print(f"  Music profile    : BPM={profile['bpm']}, "
              f"Scale={profile['scale_name']}, "
              f"Mood={profile['mood_label']}, "
              f"Root={profile['root_freq']:.1f}Hz, "
              f"Harmonics={profile['n_harmonics']}")
        music_summary.append({"cluster": c, **profile, **visuals})

        # 3. Generate audio
        total_dur = len(subset) * config.SECONDS_PER_IMAGE
        print(f"  Generating {total_dur}s audio …")
        audio_data = generate_audio(profile, total_dur)

        wav_path = os.path.join(config.OUTPUT_DIR, f"cluster_{c}_audio.wav")
        wavfile.write(wav_path, SR, audio_data)
        print(f"  [Saved] {wav_path}")

        # 4. Create video
        video_path = os.path.join(config.VIDEO_DIR, f"cluster_{c}_slideshow.mp4")
        create_slideshow_video(subset, wav_path, video_path, c, profile)

    # Print music selection summary
    print("\n── Music Selection Summary (Algorithmic Mapping) ───────────────────────────")
    print(f"{'Cluster':>8}  {'Mood':>22}  {'Scale':>20}  {'BPM':>5}  {'Root Hz':>8}")
    print("-" * 75)
    for s in music_summary:
        print(f"  C{s['cluster']:>4}    {s['mood_label']:>22}  "
              f"{s['scale_name']:>20}  {s['bpm']:>5}  {s['root_freq']:>8.1f}")
    print()

    print("\n========== Step 4 Complete ==========\n")
    print(f"  Videos saved to: {config.VIDEO_DIR}/")
    print()


if __name__ == "__main__":
    main()
