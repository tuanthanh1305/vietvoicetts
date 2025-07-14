import gradio as gr
from vietvoicetts import synthesize

def run_tts(text, gender, area, emotion, style, speed):
    output_path = "output.wav"
    try:
        synthesize(
            text,
            output_path,
            gender=gender,
            area=area,
            emotion=emotion,
            group=style,
            speed=speed
        )
        return output_path
    except Exception as e:
        return f"Lỗi: {str(e)}"

demo = gr.Interface(
    fn=run_tts,
    inputs=[
        gr.Textbox(label="Nhập văn bản tiếng Việt"),
        gr.Radio(["female", "male"], label="Giới tính", value="female"),
        gr.Radio(["northern", "central", "southern"], label="Vùng miền", value="northern"),
        gr.Radio(["neutral", "happy", "sad", "angry", "surprised", "monotone", "serious"], label="Cảm xúc", value="neutral"),
        gr.Radio(["story", "news", "audiobook", "interview", "review"], label="Kiểu giọng", value="story"),
        gr.Slider(0.5, 1.5, value=1.0, label="Tốc độ nói")
    ],
    outputs=gr.Audio(label="Kết quả TTS", type="filepath"),
    title="VietVoice-TTS Demo",
    description="Tổng hợp giọng nói tiếng Việt với lựa chọn giới tính, vùng miền, cảm xúc, và phong cách."
)

demo.launch()
