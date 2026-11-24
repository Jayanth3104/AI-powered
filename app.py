import os
import torch
import cv2
import uuid
import time
import threading
from flask import Flask, render_template, request, redirect, url_for, Response, jsonify
from werkzeug.utils import secure_filename
import glob

# Initialize Flask app
app = Flask(__name__)

# Configuration
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['RESULTS_FOLDER'] = os.path.join(app.config['UPLOAD_FOLDER'], "results")
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'mp4', 'mov', 'avi', 'mkv'}

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

# Load YOLO model (with error handling)
try:
    model = torch.hub.load('yolov5', 'custom',
                           path='yolov5/runs/train/ambulance-detection3/weights/best.pt',
                           source='local')
    # Set model to evaluation mode for faster inference
    model.eval()
    # Use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f"✅ YOLOv5 model loaded successfully on {device}")
except Exception as e:
    print(f"❌ Failed to load YOLOv5 model: {e}")
    model = None

# Progress tracking
progress_data = {}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# ✅ Landing Page
@app.route('/')
def landing():
    return render_template('landing.html')

# ✅ Image Page
@app.route('/image')
def image_page():
    return render_template('image.html')

@app.route('/upload_image', methods=['POST'])
def upload_image():
    try:
        if 'image' not in request.files:
            return redirect('/image')

        file = request.files['image']
        if file.filename == '' or not allowed_file(file.filename):
            return redirect('/image')

        # ✅ unique filename
        ext = file.filename.rsplit('.', 1)[-1].lower()
        unique_filename = f"{uuid.uuid4().hex}.{ext}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)

        # Check if model is loaded
        if model is None:
            return "Model not loaded. Please check the server configuration.", 500

        # ✅ Run YOLO detection
        results = model(filepath)
        
        # ✅ Render results directly on the image
        rendered_imgs = results.render()  # returns list of images with detections drawn
        rendered_img = rendered_imgs[0] if rendered_imgs else None
        
        if rendered_img is None:
            return "Detection failed. Please try another image.", 500
            
        # ✅ Save rendered image directly to results folder
        result_filename = f"result_{unique_filename}"
        result_path = os.path.join(app.config['RESULTS_FOLDER'], result_filename)
        
        # Convert RGB to BGR for OpenCV (YOLOv5 render returns RGB)
        rendered_img_bgr = cv2.cvtColor(rendered_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(result_path, rendered_img_bgr)
        
        # ✅ Create correct web path for Flask static serving
        web_path = f"/static/uploads/results/{result_filename}"
        
        print(f"✅ Saved result to: {result_path}")
        print(f"✅ Web path: {web_path}")
        
        # Verify file exists
        if os.path.exists(result_path):
            return render_template('image.html', result_img=web_path)
        else:
            print(f"❌ Result file not found at: {result_path}")
            return "Error: Failed to save detection result", 500

    except Exception as e:
        import traceback
        print("❌ ERROR in /upload_image:", e)
        traceback.print_exc()
        return "Internal Server Error: " + str(e), 500

# ✅ Video Page
@app.route('/video')
def video_page():
    return render_template('video.html')

@app.route('/upload_video', methods=['POST'])
def upload_video():
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400
            
        file = request.files['video']
        if file.filename == '' or not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file'}), 400
            
        # Generate unique ID for this processing job
        process_id = str(uuid.uuid4())
        
        # Save the uploaded file with unique name
        ext = file.filename.rsplit('.', 1)[-1].lower()
        unique_filename = f"{process_id}.{ext}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)
        
        # Start processing in a separate thread
        thread = threading.Thread(
            target=process_video, 
            args=(filepath, process_id)
        )
        thread.daemon = True
        thread.start()
        
        # Return the process ID to the client
        return jsonify({'process_id': process_id})
        
    except Exception as e:
        print(f"❌ ERROR in /upload_video: {e}")
        return jsonify({'error': 'Internal Server Error'}), 500

def process_video(filepath, process_id):
    try:
        # Initialize progress
        progress_data[process_id] = {
            'status': 'processing',
            'progress': 0,
            'message': 'Initializing video processing'
        }
        
        # Check if model is loaded
        if model is None:
            progress_data[process_id] = {
                'status': 'error',
                'progress': 0,
                'message': 'Model not loaded'
            }
            return
            
        # Open the video file
        cap = cv2.VideoCapture(filepath)
        if not cap.isOpened():
            progress_data[process_id] = {
                'status': 'error',
                'progress': 0,
                'message': 'Failed to open video file'
            }
            return
            
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Create output video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_path = os.path.join(app.config['RESULTS_FOLDER'], f"result_{process_id}.mp4")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not out.isOpened():
            progress_data[process_id] = {
                'status': 'error',
                'progress': 0,
                'message': 'Failed to create output video'
            }
            cap.release()
            return
            
        # Process each frame
        frame_count = 0
        batch_size = 4  # Process frames in batches for efficiency
        
        while True:
            frames = []
            frame_positions = []
            
            # Read a batch of frames
            for _ in range(batch_size):
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
                frame_count += 1
                
            if not frames:
                break
                
            # Process the batch
            try:
                results = model(frames) if len(frames) > 1 else model(frames[0])
                
                # Handle results (single frame or batch)
                if len(frames) > 1:
                    rendered_frames = results.render()
                else:
                    rendered_frames = [results.render()[0]]
                
                # Write processed frames to output
                for rendered_frame in rendered_frames:
                    # Convert RGB to BGR for OpenCV
                    frame_bgr = cv2.cvtColor(rendered_frame, cv2.COLOR_RGB2BGR)
                    out.write(frame_bgr)
            except Exception as e:
                print(f"Error processing frame batch: {e}")
                continue
            
            # Update progress
            progress = min(100, int((frame_count / total_frames) * 100))
            progress_data[process_id] = {
                'status': 'processing',
                'progress': progress,
                'message': f'Processing frame {frame_count} of {total_frames}'
            }
            
            # Add small delay to prevent overwhelming the system
            time.sleep(0.01)
            
        # Clean up
        cap.release()
        out.release()
        
        # Mark as complete
        progress_data[process_id] = {
            'status': 'complete',
            'progress': 100,
            'message': 'Processing complete',
            'result_path': f"uploads/results/result_{process_id}.mp4"
        }
        
        # Clean up uploaded file
        try:
            os.remove(filepath)
        except:
            pass
            
    except Exception as e:
        print(f"❌ ERROR in process_video: {e}")
        progress_data[process_id] = {
            'status': 'error',
            'progress': 0,
            'message': f'Error: {str(e)}'
        }

@app.route('/progress/<process_id>')
def get_progress(process_id):
    if process_id in progress_data:
        return jsonify(progress_data[process_id])
    else:
        return jsonify({
            'status': 'error',
            'progress': 0,
            'message': 'Process ID not found'
        })

# ✅ Live Streaming Page
@app.route('/live')
def live_page():
    return render_template('live.html')

def gen_frames():
    cap = cv2.VideoCapture(0)  # webcam
    
    if not cap.isOpened():
        print("❌ Cannot open webcam")
        return
        
    while True:
        success, frame = cap.read()
        if not success:
            break
            
        # Check if model is loaded
        if model is not None:
            try:
                results = model(frame)
                if results.render():
                    frame = results.render()[0]
            except Exception as e:
                print(f"Error in live detection: {e}")
            
        # Encode frame for streaming
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
               
    cap.release()

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Clean up old progress data periodically
def cleanup_old_progress():
    while True:
        time.sleep(300)  # Clean up every 5 minutes
        current_time = time.time()
        keys_to_remove = []
        
        for process_id, data in progress_data.items():
            # Remove entries older than 1 hour
            if 'timestamp' in data and current_time - data['timestamp'] > 3600:
                keys_to_remove.append(process_id)
        
        for key in keys_to_remove:
            progress_data.pop(key, None)

# Start cleanup thread
cleanup_thread = threading.Thread(target=cleanup_old_progress)
cleanup_thread.daemon = True
cleanup_thread.start()

if __name__ == "__main__":
    app.run(debug=True, threaded=True) 