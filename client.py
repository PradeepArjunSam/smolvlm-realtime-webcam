import cv2
import requests
import json
import base64
import time
import argparse
import threading
import sys

# Argument parsing
parser = argparse.ArgumentParser(description='SmolVLM Client for IP Camera/Webcam')
parser.add_argument('--url', type=str, default='http://localhost:8080', help='Llama server base URL')
parser.add_argument('--camera', type=str, default='0', help='Camera index (0) or URL (http://.../video)')
parser.add_argument('--interval', type=float, default=0.5, help='Interval between requests in seconds')
parser.add_argument('--prompt', type=str, default='What do you see?', help='Instruction for the model')
args = parser.parse_args()

# Global variables
current_response = "Waiting for server..."
running = True

def send_request_thread(frame, current_prompt):
    global current_response
    
    # Encode frame to base64
    _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    img_str = base64.b64encode(buffer).decode('utf-8')
    image_url = f"data:image/jpeg;base64,{img_str}"

    payload = {
        "max_tokens": 100,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": current_prompt},
                    {"type": "image_url", "image_url": {"url": image_url}}
                ]
            }
        ]
    }

    try:
        response = requests.post(
            f"{args.url}/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload),
            timeout=10 # Avoid hanging forever
        )
        
        if response.status_code == 200:
            data = response.json()
            if 'choices' in data and len(data['choices']) > 0:
                current_response = data['choices'][0]['message']['content']
            else:
                current_response = "No content in response"
        else:
            current_response = f"Error: {response.status_code} - {response.text[:50]}"
            
    except Exception as e:
        current_response = f"Request failed: {str(e)}"

def main():
    global running, current_response
    
    # Handle camera input
    camera_source = args.camera
    if camera_source.isdigit():
        camera_source = int(camera_source)
        
    print(f"Opening camera: {camera_source}")
    cap = cv2.VideoCapture(camera_source)

    if not cap.isOpened():
        print("Error: Could not open video source.")
        print("If using phone, make sure 'IP Webcam' is running and enter the URL (e.g., http://192.168.1.X:8080/video)")
        sys.exit(1)

    print(f"Connected to Llama server at {args.url}")
    print("Press 'q' to quit.")

    last_req_time = 0
    
    while running:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            time.sleep(1)
            continue

        # Resize for performance/display (optional, but good for speed)
        # frame = cv2.resize(frame, (640, 480))

        cur_time = time.time()
        if cur_time - last_req_time > args.interval:
            # Run request in a separate thread to avoid blocking the video feed
            threading.Thread(target=send_request_thread, args=(frame.copy(), args.prompt)).start()
            last_req_time = cur_time

        # Overlay text
        # Add background rectangle for text
        text = f"AI: {current_response}"
        (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(frame, (0, 0), (frame.shape[1], h + 20), (0, 0, 0), -1)
        cv2.putText(frame, text, (10, h + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow('SmolVLM Client', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            running = False
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
