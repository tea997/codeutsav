from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64
import mediapipe as mp  # Google's MediaPipe
import time
import os

app = Flask(__name__)

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

@app.route('/')
def index():
    # INLINE HTML - No templates needed!
    html = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MediaPipe Person Detection: Approve/Decline</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Segoe UI', sans-serif; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; display: flex; justify-content: center; align-items: center; padding: 20px; }
        .container { background: white; border-radius: 20px; padding: 40px; box-shadow: 0 20px 60px rgba(0,0,0,0.3); max-width: 600px; width: 100%; }
        h1 { color: #667eea; text-align: center; margin-bottom: 30px; font-size: 28px; }
        .video-container { position: relative; background: #000; border-radius: 15px; overflow: hidden; margin-bottom: 20px; }
        video { width: 100%; display: block; border-radius: 15px; }
        .challenge-overlay { position: absolute; top: 0; left: 0; right: 0; background: rgba(102, 126, 234, 0.95); color: white; padding: 30px 20px; text-align: center; font-size: 32px; font-weight: bold; transform: translateY(-100%); transition: transform 0.3s ease; z-index: 10; }
        .challenge-overlay.active { transform: translateY(0); }
        button { width: 100%; padding: 15px; font-size: 18px; font-weight: bold; color: white; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border: none; border-radius: 10px; cursor: pointer; transition: transform 0.2s; }
        button:hover:not(:disabled) { transform: translateY(-2px); }
        button:disabled { background: #ccc; cursor: not-allowed; }
        .result { margin-top: 20px; padding: 20px; border-radius: 10px; text-align: center; font-weight: bold; display: none; }
        .result.success { background: #d4edda; color: #155724; border: 2px solid #c3e6cb; display: block; }
        .result.fail { background: #f8d7da; color: #721c24; border: 2px solid #f5c6cb; display: block; }
        .instructions { background: #f8f9fa; padding: 15px; border-radius: 10px; margin-bottom: 20px; font-size: 14px; color: #666; }
        .instructions ul { margin-left: 20px; margin-top: 10px; }
        .instructions li { margin-bottom: 5px; }
        .status { text-align: center; margin-bottom: 15px; padding: 10px; background: #e3f2fd; border-radius: 8px; color: #1976d2; font-weight: 500; }
        .warning-text { color: #dc3545; font-weight: bold; font-size: 12px; margin-top: 5px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔐 Person Detection: Approve if Present, Decline if Not</h1>
        
        <div class="instructions">
            <strong>📋 Instructions:</strong>
            <ul>
                <li><strong>Person in frame?</strong> Stay centered (≥50% frames) or decline</li>
                <li><strong>Approve:</strong> Person detected consistently → "Approve"</li>
                <li><strong>Decline:</strong> No person or inconsistent → "Decline"</li>
            </ul>
            <div class="warning-text">Pre-trained MediaPipe model - no blink check, just presence!</div>
        </div>

        <div class="status" id="status">Camera initializing...</div>
        
        <div class="video-container">
            <video id="video" autoplay playsinline></video>
            <div id="challengeOverlay" class="challenge-overlay"></div>
        </div>
        
        <button id="startBtn" disabled>Start Person Check</button>
        
        <div id="result" class="result"></div>
    </div>

    <script>
        const video = document.getElementById('video');
        const startBtn = document.getElementById('startBtn');
        const challengeOverlay = document.getElementById('challengeOverlay');
        const resultDiv = document.getElementById('result');
        const statusDiv = document.getElementById('status');
        
        let stream = null;
        let capturedFrames = [];
        
        // Initialize camera
        async function initCamera() {
            try {
                stream = await navigator.mediaDevices.getUserMedia({ 
                    video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' } 
                });
                video.srcObject = stream;
                video.onloadedmetadata = () => {
                    video.play();
                    startBtn.disabled = false;
                    statusDiv.textContent = '✅ Ready! Click to check presence.';
                    statusDiv.style.background = '#d4edda';
                    statusDiv.style.color = '#155724';
                };
            } catch (err) {
                statusDiv.textContent = '❌ Camera denied: ' + err.message;
                statusDiv.style.background = '#f8d7da';
                statusDiv.style.color = '#721c24';
            }
        }
        
        // Capture frame
        function captureFrame() {
            try {
                const canvas = document.createElement('canvas');
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
                if (canvas.width === 0 || canvas.height === 0) return null;
                const ctx = canvas.getContext('2d');
                ctx.drawImage(video, 0, 0);
                return canvas.toDataURL('image/jpeg', 0.8);
            } catch (err) {
                console.error('Capture error:', err);
                return null;
            }
        }
        
        function hideChallenge() {
            challengeOverlay.classList.remove('active');
        }
        
        // Start verification
        startBtn.addEventListener('click', async () => {
            resultDiv.className = 'result'; resultDiv.innerHTML = '';
            startBtn.disabled = true;
            capturedFrames = [];
            
            try {
                statusDiv.textContent = '⏳ Capturing 5s... Stay in frame!';
                statusDiv.style.background = '#fff3cd';
                statusDiv.style.color = '#856404';
                
                challengeOverlay.textContent = '👤 Stay in frame!';
                challengeOverlay.classList.add('active');
                
                const captureDuration = 5000;  // 10s
                const frameInterval = 100;  // 10 fps
                let startTime = Date.now();
                
                while (Date.now() - startTime < captureDuration) {
                    await new Promise(r => setTimeout(r, frameInterval));
                    const frame = captureFrame();
                    if (frame && frame !== 'data:,') capturedFrames.push(frame);
                }
                
                hideChallenge();
                
                if (capturedFrames.length === 0) throw new Error('No frames!');
                
                statusDiv.textContent = '🔍 Analyzing presence...';
                challengeOverlay.textContent = '🔍 Processing...';
                challengeOverlay.classList.add('active');
                
                const verifyRes = await fetch('/verify_liveness', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ frames: capturedFrames })
                });
                
                if (!verifyRes.ok) throw new Error(`Server: ${verifyRes.status}`);
                
                const result = await verifyRes.json();
                hideChallenge();
                
                const isApprove = result.decision === 'Approve';
                resultDiv.className = isApprove ? 'result success' : 'result fail';
                
                statusDiv.textContent = isApprove ? '✅ Approved!' : '❌ Declined';
                statusDiv.style.background = isApprove ? '#d4edda' : '#f8d7da';
                statusDiv.style.color = isApprove ? '#155724' : '#721c24';
                
                let html = `<div style="font-size: 24px;">${result.decision}</div>`;
                if (result.message) html += `<div style="font-size: 14px; color: #666; margin-top: 10px;">${result.message}</div>`;
                if (result.presence_percent !== undefined) html += `<div style="font-size: 18px; margin-top: 10px;">Presence: ${result.presence_percent}%</div>`;
                if (result.frames_analyzed) html += `<div style="font-size: 14px; color: #666; margin-top: 5px;">Frames: ${result.frames_analyzed}</div>`;
                resultDiv.innerHTML = html;
                
            } catch (err) {
                console.error('Error:', err);
                resultDiv.className = 'result fail';
                resultDiv.innerHTML = `<div>❌ Error: ${err.message}</div>`;
                statusDiv.textContent = '❌ Error occurred';
                statusDiv.style.background = '#f8d7da';
                statusDiv.style.color = '#721c24';
            } finally {
                startBtn.disabled = false;
            }
        });
        
        initCamera();
    </script>
</body>
</html>
    '''
    return html

@app.route('/verify_liveness', methods=['POST'])
def verify_liveness():
    try:
        data = request.json
        frames_data = data.get('frames', [])
        
        print(f"\n{'='*60}")
        print(f"Analyzing {len(frames_data)} frames for person presence")
        
        if not frames_data:
            return jsonify({"decision": "Decline", "message": "No frames received", "presence_percent": 0, "frames_analyzed": 0})
        
        frames_with_face = 0
        total_frames = len(frames_data)
        
        for idx, image_data in enumerate(frames_data):
            # Decode frame
            try:
                if ',' in image_data: image_data = image_data.split(',')[1]
                image_bytes = base64.b64decode(image_data)
                np_arr = np.frombuffer(image_bytes, np.uint8)
                frame_rgb = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)  # MediaPipe uses RGB
                if frame_rgb is None: continue
                frame_rgb = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2RGB)
            except Exception as e:
                print(f"Frame {idx} decode error: {e}")
                continue
            
            # Detect person (face presence)
            detection_results = face_detection.process(frame_rgb)
            if not detection_results.detections:
                print(f"Frame {idx}: No person detected")
                continue
            frames_with_face += 1
            print(f"Frame {idx}: Person detected")
        
        presence_percent = (frames_with_face / total_frames * 100) if total_frames > 0 else 0
        
        print(f"\nSummary: {frames_with_face}/{total_frames} frames with person ({presence_percent:.1f}%)")
        
        # Simple Logic: Approve if person present ≥50%, else decline
        if presence_percent >= 50:
            decision = "Approve"
            message = f"Person consistently present ({presence_percent:.1f}% frames)."
        else:
            decision = "Decline"
            message = f"Person not consistently present ({presence_percent:.1f}% frames)."
        
        print(f"Decision: {decision}\n{'='*60}")
        
        return jsonify({
            "decision": decision,
            "message": message,
            "presence_percent": round(presence_percent, 1),
            "frames_analyzed": total_frames
        })
    
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback; traceback.print_exc()
        return jsonify({"decision": "Decline", "message": str(e), "presence_percent": 0, "frames_analyzed": 0})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', debug=False, port=port)
