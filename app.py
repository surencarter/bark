import numpy as np
import gradio as gr
from bark import SAMPLE_RATE, generate_audio, preload_models

DEBUG_MODE = False

if not DEBUG_MODE:
    _ = preload_models()

default_text = "Hello, my name is Suno. And, uh — and I like pizza. [laughs]\nBut I also have other interests such as playing tic tac toe."

def gen_tts(text, history_prompt, temp_semantic, temp_waveform):
    if history_prompt == "Unconditional":
        history_prompt = None
    else:
        history_prompt = history_prompt.lower().replace(" ", "_")
    if DEBUG_MODE:
        audio_arr = np.zeros(SAMPLE_RATE)
    else:
        audio_arr = generate_audio(text, history_prompt=history_prompt, text_temp=temp_semantic, waveform_temp=temp_waveform)
    return (SAMPLE_RATE, audio_arr)

iface = gr.Interface(
    title="<div style='text-align:left'>🐶 Bark</div>",
    description="Bark is a universal text-to-audio model created by [Suno](www.suno.ai), with code publicly available [here](https://github.com/suno-ai/bark). Bark can generate highly realistic, multilingual speech as well as other audio - including music, background noise and simple sound effects. This demo should be used for research purposes only. Commercial use is strictly prohibited. The model output is not censored and the authors do not endorse the opinions in the generated content. Use at your own risk.",
    fn=gen_tts, 
    inputs=[
        gr.Textbox(label="Input Text", lines=3, value=default_text), 
        gr.Dropdown(
            ["Unconditional"] + [f"Speech {n}" for n in range(8)] + [f"Music {n}" for n in range(6)], 
            value="None", label="Acoustic Prompt", info="This choice primes the model on how to condition the generated audio."
        ),
        gr.Slider(minimum=0, maximum=1, step=0.01, value=0.7, label="Temp 1", info="Gen. temperature of semantic tokens. (lower is more conservative, higher is more diverse)"),
        gr.Slider(minimum=0, maximum=1, step=0.01, value=0.7, label="Temp 2", info="Gen. temperature of waveform tokens. (lower is more conservative, higher is more diverse)"),
    ], 
    outputs="audio",
    )
iface.launch()
