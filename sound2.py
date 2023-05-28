import whisper_timestamped as whisper
import moviepy.editor as mp
import pprint
import torch
from pydub import AudioSegment
from pydub.silence import split_on_silence
import gc
gc.collect()
from tqdm import tqdm

def match_target_amplitude(aChunk, target_dBFS):
    ''' Normalize given audio chunk '''
    change_in_dBFS = target_dBFS - aChunk.dBFS
    return aChunk.apply_gain(change_in_dBFS)

torch.cuda.empty_cache()

# Load video and convert to audio
video = mp.VideoFileClip(r"input2.webm")
video.audio.write_audiofile(r"output.mp3")

# Load audio for transcription
audio = whisper.load_audio("output.mp3")
model = whisper.load_model("medium", device="cuda")
output_dir = "/mnt/sd/adobe/shuton"
result = whisper.transcribe(model, audio, language="ja")

song = AudioSegment.from_mp3("output.mp3")

# Process each chunk with your parameters
for i, seg in enumerate(result["segments"]):
    # Create a silence chunk that's 0.5 seconds (or 500 ms) long for padding.
    #silence_chunk = AudioSegment.silent(duration=500)

    # Add the padding chunk to beginning and end of the entire chunk.
    #audio_chunk = silence_chunk + chunk + silence_chunk

    # Normalize the entire chunk.
    #normalized_chunk = audio_chunk # match_target_amplitude(audio_chunk, -20.0)

    # Extract the start and end times in milliseconds.
    start_time = result["segments"][i]["start"] * 1000
    end_time = result["segments"][i]["end"] * 1000

    # Slice the audio segment for this word.
    word_audio = song[start_time:end_time]

    # Export the audio chunk with new bitrate and transcribed text as filename.
    print("Exporting {0}.mp3.".format(result['segments'][i]['text']))
    word_audio.export(
            output_dir+"/{0}.mp3".format(result['segments'][i]['text'][:10]),
        bitrate = "192k",
        format = "mp3"
    )
