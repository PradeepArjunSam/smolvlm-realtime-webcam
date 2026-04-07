import cv2
import requests
import json
import base64
import time
import argparse
import threading
import sys

# ── Argument parsing ───────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description='SmolVLM Client — supports llama.cpp, Ollama, and vLLM backends',
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Examples:
  # llama.cpp (default)
  python client.py --url http://localhost:8080

  # Ollama
  python client.py --backend ollama --url http://localhost:11434 --model smolvlm

  # vLLM
  python client.py --backend vllm --url http://localhost:8000 --model HuggingFaceTB/SmolVLM-500M-Instruct

  # IP Camera / phone camera (e.g. IP Webcam app)
  python client.py --camera http://192.168.1.100:8080/video
"""
)
parser.add_argument('--url',      type=str, default='http://localhost:8080',
                    help='Llama server / Ollama / vLLM base URL')
parser.add_argument('--backend',  type=str, default='llamacpp',
                    choices=['llamacpp', 'ollama', 'vllm', 'custom'],
                    help='Backend type (default: llamacpp)')
parser.add_argument('--model',    type=str, default='',
                    help='Model name (required for Ollama/vLLM)')
parser.add_argument('--camera',   type=str, default='0',
                    help='Camera index (0) or RTSP/HTTP URL')
parser.add_argument('--interval', type=float, default=0.5,
                    help='Interval between requests in seconds (default: 0.5)')
parser.add_argument('--prompt',   type=str,
                    default='<image>\nDescribe what you see in this image in detail. Focus on objects, people, and actions.',
                    help='Instruction for the model')
parser.add_argument('--max-tokens', type=int, default=150,
                    help='Max tokens for model response (default: 150)')
parser.add_argument('--timeout',  type=float, default=10.0,
                    help='Request timeout in seconds (default: 10)')
parser.add_argument('--no-mirror', action='store_true',
                    help='Disable horizontal mirror flip of the video feed')
args = parser.parse_args()

# ── Global state ───────────────────────────────────────────────────────────────
current_response = "Waiting for server..."
connection_ok    = False
running          = True
request_count    = 0
lock             = threading.Lock()

# ── Backend URL builder (FIX: Ollama / vLLM support) ──────────────────────────
def get_endpoint_url():
    base = args.url.rstrip('/')
    return f"{base}/v1/chat/completions"

# ── Request thread ─────────────────────────────────────────────────────────────
def send_request_thread(frame, prompt):
    global current_response, connection_ok, request_count

    try:
        # Encode frame to JPEG base64
        _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        img_b64   = base64.b64encode(buffer).decode('utf-8')
        image_url = f"data:image/jpeg;base64,{img_b64}"

        # FIX: Detect JSON mode
        wants_json = '{' in prompt or 'json' in prompt.lower()

        payload = {
            "max_tokens": args.max_tokens,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text",      "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_url}}
                    ]
                }
            ]
        }

        # Add model field for Ollama / vLLM
        if args.backend in ('ollama', 'vllm', 'custom') and args.model:
            payload["model"] = args.model

        # JSON response format
        if wants_json:
            payload["response_format"] = {"type": "json_object"}

        response = requests.post(
            get_endpoint_url(),
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload),
            timeout=args.timeout
        )

        if response.status_code == 200:
            data = response.json()
            choices = data.get('choices', [])
            if choices:
                with lock:
                    current_response = choices[0]['message']['content']
                    connection_ok    = True
                    request_count   += 1
            else:
                with lock:
                    current_response = "[No content in response]"
        else:
            with lock:
                current_response = f"[HTTP {response.status_code}] {response.text[:80]}"
                connection_ok = False

    except requests.exceptions.ConnectionError:
        with lock:
            current_response = f"[Connection refused] Is the server running at {get_endpoint_url()}?"
            connection_ok = False
    except requests.exceptions.Timeout:
        with lock:
            current_response = f"[Timeout after {args.timeout}s] Server too slow — try longer interval"
            connection_ok = False
    except Exception as e:
        with lock:
            current_response = f"[Error] {type(e).__name__}: {str(e)[:80]}"
            connection_ok = False


# ── Video overlay helpers ──────────────────────────────────────────────────────
def draw_text_with_bg(frame, text, org, font_scale=0.55, thickness=1,
                      text_color=(0, 255, 100), bg_color=(0, 0, 0)):
    """Draw text with a filled background rectangle."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = org
    cv2.rectangle(frame, (x - 4, y - th - 6), (x + tw + 4, y + baseline), bg_color, -1)
    cv2.putText(frame, text, (x, y), font, font_scale, text_color, thickness, cv2.LINE_AA)


def wrap_text(text, max_chars=80):
    """Wrap long text into multiple lines."""
    words = text.split()
    lines, current = [], ""
    for word in words:
        if len(current) + len(word) + 1 <= max_chars:
            current = f"{current} {word}".strip()
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines[:6]   # max 6 lines on screen


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    global running

    # Open camera
    camera_source = args.camera
    if camera_source.isdigit():
        camera_source = int(camera_source)

    print(f"[SmolVLM] Opening camera: {camera_source}")
    cap = cv2.VideoCapture(camera_source)

    if not cap.isOpened():
        print("[SmolVLM] ERROR: Could not open video source.")
        print("  → If using a phone, start 'IP Webcam' app and pass the URL:")
        print("     python client.py --camera http://192.168.x.x:8080/video")
        sys.exit(1)

    print(f"[SmolVLM] Backend : {args.backend}")
    print(f"[SmolVLM] Endpoint: {get_endpoint_url()}")
    if args.model:
        print(f"[SmolVLM] Model   : {args.model}")
    print("[SmolVLM] Press 'q' to quit, 'p' to pause/resume requests.")

    last_req_time  = 0
    request_active = False
    paused         = False

    while running:
        ret, frame = cap.read()
        if not ret:
            print("[SmolVLM] Failed to grab frame — retrying in 1s…")
            time.sleep(1)
            continue

        # Mirror display (natural selfie view)
        if not args.no_mirror:
            frame = cv2.flip(frame, 1)

        cur_time = time.time()

        # FIX: Non-blocking threaded requests with proper interval
        if not paused and not request_active and (cur_time - last_req_time > args.interval):
            request_active = True
            last_req_time  = cur_time
            t = threading.Thread(
                target=lambda: (send_request_thread(frame.copy(), args.prompt),
                                setattr(sys.modules[__name__], 'request_active', False)),
                daemon=True
            )
            t.start()

        # ── Overlay ────────────────────────────────────────────────────────────
        h, w = frame.shape[:2]

        # Status bar at top
        with lock:
            resp = current_response
            ok   = connection_ok
            rc   = request_count

        status_color = (0, 200, 80) if ok else (0, 80, 220)
        status_text  = f"{'[PAUSED] ' if paused else ''}Backend: {args.backend}  |  Requests: {rc}"
        draw_text_with_bg(frame, status_text, (8, 22),
                          text_color=status_color, bg_color=(0, 0, 0, 180))

        # Response text at bottom
        lines = wrap_text(f"AI: {resp}", max_chars=max(40, w // 12))
        y_start = h - (len(lines) * 24) - 12
        for i, line in enumerate(lines):
            draw_text_with_bg(frame, line, (8, y_start + i * 24),
                              font_scale=0.58, text_color=(200, 255, 200))

        # FPS counter
        elapsed = time.time() - cur_time
        fps_val = f"{1.0/max(elapsed, 0.001):.0f}" if elapsed > 0 else "—"
        draw_text_with_bg(frame, f"FPS: {fps_val}", (w - 90, 22),
                          text_color=(200, 200, 200))

        cv2.imshow('SmolVLM Client  [q=quit, p=pause]', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            running = False
            break
        elif key == ord('p'):
            paused = not paused
            print(f"[SmolVLM] {'Paused' if paused else 'Resumed'}")

    cap.release()
    cv2.destroyAllWindows()
    print("[SmolVLM] Exited cleanly.")


if __name__ == "__main__":
    main()
