# from IPython.display import display, Image, Audio

# import cv2  # We're using OpenCV to read video, to install !pip install opencv-python
# import base64
# import time
# from openai import OpenAI
# import os
# import requests

# client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "<your OpenAI API key if not set as env var>"))
import numpy as np
import cv2
import base64
import time
from openai import OpenAI
import os
import requests
from IPython.display import display, Image, Audio

# Initialize OpenAI client
client = OpenAI()

video = cv2.VideoCapture("/Users/sandeep/Documents/ZooAnalysis/demo1.mp4")

base64Frames = []
frame_interval = 100 
while video.isOpened():
    success, frame = video.read()
    if not success:
        break
    current_frame = int(video.get(cv2.CAP_PROP_POS_FRAMES))
    # if current_frame % frame_interval == 0:
    #     _, buffer = cv2.imencode(".jpg", frame)
    #     base64_frame = base64.b64encode(buffer).decode("utf-8")
    #     base64Frames.append(base64_frame)
    _, buffer = cv2.imencode(".jpg", frame)
    base64Frames.append(base64.b64encode(buffer).decode("utf-8"))

video.release()
print(len(base64Frames), "frames read.")

# original_fps = video.get(cv2.CAP_PROP_FPS)
# frame_delay = 1 / original_fps

# display_handle = display(None, display_id=True)
# for img in base64Frames:
#     display_handle.update(Image(data=base64.b64decode(img.encode("utf-8"))))
#     time.sleep(0.1)


PROMPT_MESSAGES = [
    {
        "role": "user",
        "content": [
            "You are an animal behaviorist observing the highlighted animal from the zoo in video frames. Provide a comprehensive analysis with a focus on: 1. **Activity Observation**: Describe specific actions and interactions of the animal. Include any unique or interesting behaviors. 2. **Signs of Illness or Health Issues**: Note any subtle signs that might indicate health issues or discomfort. 3. **Health Assessment**: Provide an assessment of the animal's physical and mental health, covering movement, social interaction, physical form, and coat condition. 4. **Recommendations**: Offer recommendations for maintaining the animal's health and well-being. 5. **Narrative Style**: Present the observations in a narrative style to make the report engaging and informative, also don't mention frames and remember to give recommendations for a zoo animal. Here are the selected frames:",
            *map(lambda x: {"image": x, "resize": 768}, base64Frames[0::50]),
        ],
    },
]
params = {
    "model": "gpt-4o",
    "messages": PROMPT_MESSAGES,
    "max_tokens": 2000,
}

result = client.chat.completions.create(**params)
print(result.choices[0].message.content)

# PROMPT_MESSAGES = [
#     {
#         "role": "user",
#         "content": [
#             "These are frames of a video. Create a short voiceover script in the style of David Attenborough. Only include the narration.",
#             *map(lambda x: {"image": x, "resize": 768}, base64Frames[0::60]),
#         ],
#     },
# ]
# params = {
#     "model": "gpt-4o",
#     "messages": PROMPT_MESSAGES,
#     "max_tokens": 500,
# }

# result = client.chat.completions.create(**params)
# print(result.choices[0].message.content)

# response = requests.post(
#     "https://api.openai.com/v1/audio/speech",
#     headers={
#         "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
#     },
#     json={
#         "model": "tts-1-1106",
#         "input": result.choices[0].message.content,
#         "voice": "onyx",
#     },
# )

# audio = b""
# for chunk in response.iter_content(chunk_size=1024 * 1024):
#     audio += chunk
# Audio(audio)

# def extract_frames(video_path, interval=50):
#     video = cv2.VideoCapture(video_path)
#     base64Frames = []
#     frame_count = 0

#     while video.isOpened():
#         success, frame = video.read()
#         if not success:
#             break
#         if frame_count % interval == 0:
#             _, buffer = cv2.imencode(".jpg", frame)
#             base64Frames.append(base64.b64encode(buffer).decode("utf-8"))
#         frame_count += 1

#     video.release()
#     print(f"{len(base64Frames)} frames extracted.")
#     return base64Frames

# def get_video_description(frames):
#     PROMPT_MESSAGES = [
#         {
#             "role": "user",
#             "content": [
#                 "These are frames from a video that I want to upload. Generate a compelling description that I can upload along with the video.",
#                 *map(lambda x: {"image": x, "resize": 768}, frames),
#             ],
#         },
#     ]
#     params = {
#         "model": "gpt-4-vision-preview",
#         "messages": PROMPT_MESSAGES,
#         "max_tokens": 200,
#     }

#     result = client.chat.completions.create(**params)
#     return result.choices[0].message.content

# def generate_voiceover_script(frames):
#     PROMPT_MESSAGES = [
#         {
#             "role": "user",
#             "content": [
#                 "These are frames of a video. Create a short voiceover script in the style of David Attenborough. Only include the narration.",
#                 *map(lambda x: {"image": x, "resize": 768}, frames),
#             ],
#         },
#     ]
#     params = {
#         "model": "gpt-4-vision-preview",
#         "messages": PROMPT_MESSAGES,
#         "max_tokens": 500,
#     }

#     result = client.chat.completions.create(**params)
#     return result.choices[0].message.content

# def generate_voiceover_audio(script):
#     response = requests.post(
#         "https://api.openai.com/v1/audio/speech",
#         headers={
#             "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
#         },
#         json={
#             "model": "tts-1",
#             "input": script,
#             "voice": "onyx",
#         },
#     )

#     if response.status_code == 200:
#         with open("voiceover.mp3", "wb") as f:
#             f.write(response.content)
#         print("Voiceover audio generated and saved as 'voiceover.mp3'")
#     else:
#         print("Failed to generate voiceover audio")

# def main():
#     video_path = "path/to/your/video.mp4"  # Replace with your video path
#     frames = extract_frames(video_path)

#     description = get_video_description(frames)
#     print("Video Description:")
#     print(description)

#     voiceover_script = generate_voiceover_script(frames)
#     print("\nVoiceover Script:")
#     print(voiceover_script)

#     generate_voiceover_audio(voiceover_script)

# if __name__ == "__main__":
#     main()