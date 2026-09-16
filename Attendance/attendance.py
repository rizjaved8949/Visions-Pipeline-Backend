import cv2
import time
import numpy as np
import os
import re
import io
import shutil
import tempfile
import insightface
import pandas as pd
from datetime import datetime
import threading
import keyboard

# === Configuration === #
MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_SOURCE = 0
SIMILARITY_THRESHOLD = 0.4
CSV_FILENAME = os.path.join(MODULE_DIR, "attendance_sessions.csv")
FACES_DIR = os.path.join(MODULE_DIR, "Faces")
os.makedirs(FACES_DIR, exist_ok=True)

# Global variables
known_embeddings = []
known_names = []
all_required_persons = set()
sessions = []  # List of session records
current_session = None
is_capturing = False
session_counter = 0

# Add a lock for thread safety
session_lock = threading.Lock()

# Shared state for server/web usage (camera loop -> API endpoints)
frame_lock = threading.Lock()
latest_frame = None  # most recent annotated BGR frame (numpy array)
face_model = None
capture_thread_running = False

# Guards concurrent access to face_model.get() (camera loop, media processing, registration)
model_lock = threading.Lock()

video_source_lock = threading.Lock()


def get_video_source():
    with video_source_lock:
        return VIDEO_SOURCE


def set_video_source(source):
    """Set the camera source used by the NEXT camera_loop() start - changing
    this while a session is already running has no effect until it's
    stopped and started again. `source` may be a local device index (int,
    or a numeric string like "1") or a network stream URL (e.g. rtsp://...
    or http://... from a phone/laptop acting as an IP camera)."""
    global VIDEO_SOURCE
    source = str(source).strip()
    if not source:
        raise ValueError("Camera source is required.")
    parsed = int(source) if source.isdigit() else source
    with video_source_lock:
        VIDEO_SOURCE = parsed
    return parsed


def list_available_cameras(max_index=5):
    """Probe local device indices 0..max_index-1 and return the ones that
    actually open. Best-effort and a bit slow (opens/closes each device) -
    call it on demand (e.g. when the user opens a camera-source picker),
    not on a hot path."""
    available = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available.append(i)
        cap.release()
    return available
# Guards known_embeddings / known_names / all_required_persons mutation & reads
faces_db_lock = threading.Lock()

# === Known faces loader === #
def load_known_faces(known_faces_dir=FACES_DIR):
    model = insightface.app.FaceAnalysis(allowed_modules=['detection', 'recognition'])
    model.prepare(ctx_id=0)
    known_embeddings, known_names = [], []
    
    for person_name in os.listdir(known_faces_dir):
        person_dir = os.path.join(known_faces_dir, person_name)
        if not os.path.isdir(person_dir):
            continue
        
        all_required_persons.add(person_name)
            
        for img_name in os.listdir(person_dir):
            if not img_name.lower().endswith(('.jpg', '.png', '.jpeg')):
                continue
                
            img_path = os.path.join(person_dir, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            faces = model.get(img)
            if faces:
                known_embeddings.append(faces[0].embedding)
                known_names.append(person_name)
                
    print(f"[INFO] Loaded {len(known_embeddings)} known faces from {len(all_required_persons)} persons.")
    return known_embeddings, known_names

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def recognize_faces_in_frame(frame, face_model):
    """Recognize faces in a single frame.

    Returns (recognized_names, recognized_matches, unknown_faces, all_faces) where:
      - recognized_names: set of known names seen in this frame
      - recognized_matches: list of (name, confidence_pct, face) for known matches
      - unknown_faces: list of (confidence_pct, face) for faces below the threshold
    confidence_pct is the best cosine-similarity match, scaled to 0-100.
    """
    with model_lock:
        faces = face_model.get(frame)

    recognized_names = set()
    recognized_matches = []
    unknown_faces = []

    with faces_db_lock:
        embeddings_snapshot = list(known_embeddings)
        names_snapshot = list(known_names)

    if faces:
        for face in faces:
            embedding = face.embedding
            best_name = None
            best_sim = 0.0

            for known_emb, known_name in zip(embeddings_snapshot, names_snapshot):
                sim = cosine_similarity(embedding, known_emb)
                if sim > best_sim:
                    best_sim = sim
                    best_name = known_name

            confidence_pct = round(max(best_sim, 0.0) * 100)

            if best_sim > SIMILARITY_THRESHOLD:
                recognized_names.add(best_name)
                recognized_matches.append((best_name, confidence_pct, face))
            else:
                unknown_faces.append((confidence_pct, face))

    return recognized_names, recognized_matches, unknown_faces, faces

def _log_recognitions(session, recognized_matches, unknown_faces):
    """Append check-in log entries for this frame's detections.
    Must be called while holding session_lock."""
    now = datetime.now()

    for name, confidence_pct, _face in recognized_matches:
        if name not in session['present_persons']:
            session['check_in_log'].append({
                'name': name,
                'status': 'recognized',
                'confidence': confidence_pct,
                'time': now.strftime('%H:%M:%S'),
            })

    if unknown_faces:
        last = session['last_unknown_log_time']
        if last is None or (now - last).total_seconds() >= 5:
            best_confidence = max(conf for conf, _face in unknown_faces)
            session['check_in_log'].append({
                'name': 'Unregistered visitor',
                'status': 'unknown',
                'confidence': best_confidence,
                'time': now.strftime('%H:%M:%S'),
            })
            session['last_unknown_log_time'] = now


def draw_detections(display_frame, recognized_matches, unknown_faces):
    """Draw bounding boxes with name + confidence labels directly onto the frame."""
    for name, confidence_pct, face in recognized_matches:
        x1, y1, x2, y2 = face.bbox.astype(int)
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{name.upper()} - {confidence_pct}%"
        label_y = max(y1 - 10, 15)
        cv2.putText(display_frame, label, (x1, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    for confidence_pct, face in unknown_faces:
        x1, y1, x2, y2 = face.bbox.astype(int)
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        label = f"UNKNOWN - {confidence_pct}%"
        label_y = max(y1 - 10, 15)
        cv2.putText(display_frame, label, (x1, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


def start_session():
    """Start a new attendance session"""
    global current_session, session_counter, is_capturing
    
    with session_lock:
        if is_capturing:
            print("[INFO] Session is already running. Stop current session first.")
            return
        
        session_counter += 1
        current_session = {
            'session_id': session_counter,
            'start_time': datetime.now(),
            'end_time': None,
            'present_persons': set(),
            'frame_count': 0,
            'unknown_count': 0,
            'check_in_log': [],
            'last_unknown_log_time': None
        }
        is_capturing = True
        print(f"\n[SESSION {session_counter}] Started at {current_session['start_time'].strftime('%H:%M:%S')}")
        print(f"[INFO] Session {session_counter} capturing... Press 's' to stop.")

def stop_session():
    """Stop the current attendance session and record results"""
    global current_session, is_capturing

    with faces_db_lock:
        persons_snapshot = sorted(all_required_persons)

    with session_lock:
        if not is_capturing or current_session is None:
            print("[INFO] No active session to stop.")
            return
        
        current_session['end_time'] = datetime.now()
        current_session['duration'] = (current_session['end_time'] - current_session['start_time']).total_seconds()
        
        # Create session record
        session_record = {
            'Session_ID': current_session['session_id'],
            'Date': current_session['start_time'].date().isoformat(),
            'Start_Time': current_session['start_time'].strftime('%H:%M:%S'),
            'End_Time': current_session['end_time'].strftime('%H:%M:%S'),
            'Duration_Seconds': current_session['duration'],
            'Duration_Formatted': f"{int(current_session['duration'] // 3600)}h {int((current_session['duration'] % 3600) // 60)}m {int(current_session['duration'] % 60)}s",
            'Total_Frames': current_session['frame_count'],
            'Unknown_Detections': current_session['unknown_count']
        }
        
        # Add attendance for each person
        for person in persons_snapshot:
            if person in current_session['present_persons']:
                session_record[person] = 'Present'
            else:
                session_record[person] = 'Absent'

        sessions.append(session_record)

        # Print session summary
        print(f"\n=== SESSION {session_counter} SUMMARY ===")
        print(f"Duration: {session_record['Duration_Formatted']}")
        print(f"Total Frames: {current_session['frame_count']}")

        present_count = len(current_session['present_persons'])
        absent_count = len(persons_snapshot) - present_count
        print(f"Present: {present_count}/{len(persons_snapshot)}")
        print(f"Absent: {absent_count}/{len(persons_snapshot)}")
        print(f"Unknown face detections: {current_session['unknown_count']}")

        if present_count > 0:
            print("\nPresent persons:")
            for person in sorted(current_session['present_persons']):
                print(f"  ✓ {person}")

        if absent_count > 0:
            print("\nAbsent persons:")
            for person in sorted(set(persons_snapshot) - current_session['present_persons']):
                print(f"  ✗ {person}")
        
        # Reset for next session
        is_capturing = False
        current_session = None
        print(f"\n[INFO] Session {session_counter} ended. Ready for next session.")

def save_sessions_to_csv():
    """Save all recorded sessions to CSV file with proper structure (Names as rows)"""
    if not sessions:
        print("[INFO] No sessions recorded.")
        return
    
    # Create DataFrame in the desired format
    # Each person is a row, each session is a column
    rows = []
    
    for person in sorted(all_required_persons):
        row = {'Name': person}
        for session in sessions:
            session_id = session['Session_ID']
            row[f'Session_{session_id}'] = session.get(person, 'Absent')
        rows.append(row)
    
    # Create summary row for each session
    summary_row = {'Name': 'SUMMARY'}
    for session in sessions:
        session_id = session['Session_ID']
        present_count = sum(1 for person in all_required_persons if session.get(person) == 'Present')
        summary_row[f'Session_{session_id}'] = f"{present_count}/{len(all_required_persons)} Present"
    rows.append(summary_row)
    
    # Create metadata rows
    for session in sessions:
        session_id = session['Session_ID']
        # Date row
        date_row = {'Name': f'Session_{session_id}_Date'}
        date_row[f'Session_{session_id}'] = session['Date']
        rows.append(date_row)
        
        # Time row
        time_row = {'Name': f'Session_{session_id}_Time'}
        time_row[f'Session_{session_id}'] = f"{session['Start_Time']} to {session['End_Time']}"
        rows.append(time_row)
        
        # Duration row
        duration_row = {'Name': f'Session_{session_id}_Duration'}
        duration_row[f'Session_{session_id}'] = session['Duration_Formatted']
        rows.append(duration_row)
    
    df = pd.DataFrame(rows)
    
    # Save to CSV
    df.to_csv(CSV_FILENAME, index=False)
    
    print(f"\n=== ALL SESSIONS SAVED ===")
    print(f"Total Sessions: {len(sessions)}")
    print(f"File: {CSV_FILENAME}")
    
    # Print summary table
    print("\nSession Summary:")
    for session in sessions:
        session_id = session['Session_ID']
        present_count = sum(1 for person in all_required_persons if session.get(person) == 'Present')
        print(f"Session {session_id}: {session['Start_Time']} to {session['End_Time']} - {present_count}/{len(all_required_persons)} present")

def keyboard_listener():
    """Listen for keyboard commands in a separate thread"""
    print("\n=== KEYBOARD CONTROLS ===")
    print("'s' - Start/Stop session (toggle)")
    print("'q' - Quit and save all sessions")
    print("==========================\n")
    
    while True:
        try:
            if keyboard.is_pressed('s'):
                if not is_capturing:
                    start_session()
                else:
                    stop_session()
                time.sleep(0.5)  # Debounce
            
            if keyboard.is_pressed('q'):
                print("\n[INFO] Quit requested...")
                if is_capturing:
                    print("[INFO] Stopping active session before quitting...")
                    stop_session()
                break
                
            time.sleep(0.1)
        except Exception as e:
            print(f"[ERROR] Keyboard listener: {e}")
            break

def init_system():
    """Load known faces and the face recognition model. Used by the CLI (run())."""
    global known_embeddings, known_names, face_model, _disk_scanned

    ensure_face_model()

    print("[INFO] Loading known faces...")
    embeddings, names = load_known_faces()

    with faces_db_lock:
        known_embeddings = embeddings
        known_names = names
    _disk_scanned = True

    if not all_required_persons:
        raise RuntimeError("No known persons found in Faces directory.")

    print(f"\n[INFO] System ready for {len(all_required_persons)} persons:")
    for person in sorted(all_required_persons):
        print(f"  - {person}")


_disk_scanned = False


def load_existing_faces_from_disk():
    """Scan the Faces directory once and merge whatever's there into the
    in-memory face database. Unlike init_system(), tolerates an empty
    directory - used for lazily bootstrapping the API without requiring
    at least one person to already exist on disk."""
    global known_embeddings, known_names, _disk_scanned
    if _disk_scanned:
        return
    ensure_face_model()
    print("[INFO] Loading known faces...")
    embeddings, names = load_known_faces()
    with faces_db_lock:
        known_embeddings.extend(embeddings)
        known_names.extend(names)
    _disk_scanned = True


def ensure_face_model():
    """Load the face recognition model if it hasn't been loaded yet.
    Safe to call before any persons are registered."""
    global face_model
    if face_model is None:
        with model_lock:
            if face_model is None:
                print("\n[INFO] Loading face recognition model...")
                face_model = insightface.app.FaceAnalysis(allowed_modules=['detection', 'recognition'])
                face_model.prepare(ctx_id=0)


def _safe_person_name(name):
    """Sanitize a person's name into something safe to use as a folder/file name."""
    name = (name or "").strip()
    if not name:
        raise ValueError("Name is required.")
    safe = re.sub(r'[^A-Za-z0-9 _-]', '', name).strip()
    if not safe:
        raise ValueError("Name contains no valid characters.")
    return safe


def register_person(name, image_bytes):
    """Register a new person from a name and a photo's raw bytes.
    Saves the photo under Faces/<name>/<name>.jpg and adds their embedding
    to the in-memory face database so recognition picks them up immediately."""
    safe_name = _safe_person_name(name)
    ensure_face_model()

    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not read the uploaded image.")

    with model_lock:
        faces = face_model.get(img)
    if not faces:
        raise ValueError("No face detected in the photo. Use a clear, front-facing photo.")

    person_dir = os.path.join(FACES_DIR, safe_name)
    os.makedirs(person_dir, exist_ok=True)
    file_path = os.path.join(person_dir, f"{safe_name}.jpg")
    cv2.imwrite(file_path, img)

    with faces_db_lock:
        known_embeddings.append(faces[0].embedding)
        known_names.append(safe_name)
        all_required_persons.add(safe_name)

    print(f"[INFO] Registered new person: {safe_name}")
    return safe_name


def process_image_bytes(image_bytes):
    """Run recognition on an uploaded image.
    Returns (jpeg_bytes, media_type, summary) where summary mirrors get_status()'s
    shape closely enough for the frontend to reuse the same rendering as live results."""
    ensure_face_model()
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not read the uploaded image.")

    _names, recognized_matches, unknown_faces, _faces = recognize_faces_in_frame(img, face_model)

    display_frame = img.copy()
    draw_detections(display_frame, recognized_matches, unknown_faces)
    ok, buf = cv2.imencode('.jpg', display_frame)
    if not ok:
        raise ValueError("Could not encode the processed image.")

    detections = (
        [{'name': name, 'status': 'recognized', 'confidence': conf, 'time': '-'}
         for name, conf, _face in recognized_matches]
        + [{'name': 'Unregistered visitor', 'status': 'unknown', 'confidence': conf, 'time': '-'}
           for conf, _face in unknown_faces]
    )
    summary = {
        'recognized': sorted({name for name, _conf, _face in recognized_matches}),
        'unknown_count': len(unknown_faces),
        'frame_count': 1,
        'detections': detections,
    }
    return buf.tobytes(), 'image/jpeg', summary


def process_video_bytes(video_bytes):
    """Run recognition on every frame of an uploaded video.
    Returns (mp4_bytes, media_type, summary), boxes + labels drawn into the video."""
    ensure_face_model()

    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_in:
        tmp_in.write(video_bytes)
        in_path = tmp_in.name
    out_path = in_path + '_out.mp4'

    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        cap.release()
        os.remove(in_path)
        raise ValueError("Could not read the uploaded video.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    present_names = set()
    unknown_count = 0
    frame_count = 0
    detections = []
    last_unknown_log_frame = None
    unknown_log_interval = max(int(fps * 2), 1)  # avoid one log line per frame

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1

            _names, recognized_matches, unknown_faces, _faces = recognize_faces_in_frame(frame, face_model)

            for name, conf, _face in recognized_matches:
                if name not in present_names:
                    detections.append({
                        'name': name, 'status': 'recognized', 'confidence': conf,
                        'time': f"frame {frame_count}",
                    })
                present_names.add(name)

            if unknown_faces:
                unknown_count += len(unknown_faces)
                if last_unknown_log_frame is None or frame_count - last_unknown_log_frame >= unknown_log_interval:
                    best_conf = max(conf for conf, _face in unknown_faces)
                    detections.append({
                        'name': 'Unregistered visitor', 'status': 'unknown', 'confidence': best_conf,
                        'time': f"frame {frame_count}",
                    })
                    last_unknown_log_frame = frame_count

            display_frame = frame.copy()
            draw_detections(display_frame, recognized_matches, unknown_faces)
            writer.write(display_frame)
    finally:
        cap.release()
        writer.release()

    try:
        with open(out_path, 'rb') as f:
            result_bytes = f.read()
    finally:
        os.remove(in_path)
        os.remove(out_path)

    summary = {
        'recognized': sorted(present_names),
        'unknown_count': unknown_count,
        'frame_count': frame_count,
        'detections': list(reversed(detections)),
    }
    return result_bytes, 'video/mp4', summary


def get_status():
    """Return a JSON-serializable snapshot of the current session state."""
    with faces_db_lock:
        persons_snapshot = set(all_required_persons)

    with session_lock:
        if is_capturing and current_session:
            present = sorted(current_session['present_persons'])
            return {
                'is_capturing': True,
                'session_id': current_session['session_id'],
                'start_time': current_session['start_time'].strftime('%H:%M:%S'),
                'frame_count': current_session['frame_count'],
                'total_persons': len(persons_snapshot),
                'present_persons': present,
                'absent_persons': sorted(persons_snapshot - current_session['present_persons']),
                'unknown_count': current_session['unknown_count'],
                'check_in_log': list(reversed(current_session['check_in_log'])),
            }
        return {
            'is_capturing': False,
            'total_persons': len(persons_snapshot),
            'total_sessions_recorded': len(sessions),
        }


def get_registered_people():
    """Return the list of enrolled persons (from the Faces directory), with
    their last-seen time. Completed sessions are wiped right after they end
    (see wipe_all_data), so historical session records can't be relied on -
    instead this reflects whether each person is present in the *current*
    active session, which is the only recognition data that still exists."""
    with faces_db_lock:
        persons_snapshot = sorted(all_required_persons)

    with session_lock:
        present_now = set()
        if is_capturing and current_session:
            present_now = set(current_session['present_persons'])

    return [
        {
            'name': person,
            'status': 'active',
            'last_seen': 'Just now' if person in present_now else 'Not seen yet',
        }
        for person in persons_snapshot
    ]


def _session_report_rows():
    """Build the (summary_rows, records_rows) pair shared by both the xlsx
    and pdf session reports. Returns None if no sessions have been recorded."""
    with faces_db_lock:
        persons_snapshot = sorted(all_required_persons)

    with session_lock:
        sessions_snapshot = list(sessions)

    if not sessions_snapshot:
        return None

    summary_rows = []
    for session in sessions_snapshot:
        present_count = sum(1 for p in persons_snapshot if session.get(p) == 'Present')
        summary_rows.append({
            'Session ID': session['Session_ID'],
            'Date': session['Date'],
            'Start Time': session['Start_Time'],
            'End Time': session['End_Time'],
            'Duration': session['Duration_Formatted'],
            'Total Frames': session['Total_Frames'],
            'Total Enrolled': len(persons_snapshot),
            'Present Count': present_count,
            'Absent Count': len(persons_snapshot) - present_count,
            'Unknown Detections': session['Unknown_Detections'],
        })

    records_rows = []
    for person in persons_snapshot:
        row = {'Name': person}
        for session in sessions_snapshot:
            row[f"Session {session['Session_ID']}"] = session.get(person, 'Absent')
        records_rows.append(row)

    return summary_rows, records_rows


def _media_report_rows(filename, summary):
    """Build the (summary_rows, records_rows) pair shared by both the xlsx
    and pdf media reports."""
    with faces_db_lock:
        persons_snapshot = sorted(all_required_persons)

    recognized = set(summary.get('recognized', []))
    present_count = sum(1 for p in persons_snapshot if p in recognized)

    summary_rows = [{
        'Source File': filename,
        'Frames Processed': summary.get('frame_count', 0),
        'Total Enrolled': len(persons_snapshot),
        'Present Count': present_count,
        'Absent Count': len(persons_snapshot) - present_count,
        'Unknown Detections': summary.get('unknown_count', 0),
    }]

    records_rows = [
        {'Name': person, 'Result': 'Present' if person in recognized else 'Absent'}
        for person in persons_snapshot
    ]

    return summary_rows, records_rows


def build_session_report_xlsx():
    """Two-sheet Excel report (bytes) for all sessions currently recorded:
    'Summary' (one row per session) and 'Records' (every person x every
    session). Returns None if no sessions have been recorded."""
    rows = _session_report_rows()
    if rows is None:
        return None
    summary_rows, records_rows = rows
    return _build_xlsx({'Summary': summary_rows, 'Records': records_rows})


def build_session_report_pdf():
    """Formatted PDF report (bytes) for all sessions currently recorded.
    Returns None if no sessions have been recorded."""
    rows = _session_report_rows()
    if rows is None:
        return None
    summary_rows, records_rows = rows
    return _build_pdf('Attendance Session Report', summary_rows, records_rows)


def build_media_report_xlsx(filename, summary):
    """Two-sheet Excel report (bytes) for a single processed upload."""
    summary_rows, records_rows = _media_report_rows(filename, summary)
    return _build_xlsx({'Summary': summary_rows, 'Records': records_rows})


def build_media_report_pdf(filename, summary):
    """Formatted PDF report (bytes) for a single processed upload."""
    summary_rows, records_rows = _media_report_rows(filename, summary)
    return _build_pdf('Media Recognition Report', summary_rows, records_rows)


def _build_xlsx(sheets):
    """sheets: dict of sheet_name -> list of row dicts. Returns .xlsx bytes."""
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        for sheet_name, rows in sheets.items():
            pd.DataFrame(rows).to_excel(writer, sheet_name=sheet_name, index=False)
    return buffer.getvalue()


def _build_pdf(title, summary_rows, records_rows):
    """title: report heading. summary_rows/records_rows: list of row dicts
    (same shape as for xlsx). Returns formatted .pdf bytes."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

    styles = getSampleStyleSheet()
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=letter,
        topMargin=0.6 * inch, bottomMargin=0.6 * inch,
        leftMargin=0.6 * inch, rightMargin=0.6 * inch,
    )

    page_width = letter[0] - 1.2 * inch  # minus left+right margins

    def styled_table(data, col_widths=None):
        table = Table(data, hAlign='LEFT', repeatRows=1, colWidths=col_widths)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f2430')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cccccc')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f5f5f5')]),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        return table

    def as_table(rows):
        """Wide table: one column per field. Fine when there are few fields
        or many rows (e.g. one row per person)."""
        if not rows:
            return Paragraph("No data.", styles['Normal'])
        columns = list(rows[0].keys())
        data = [columns] + [[str(row.get(c, '')) for c in columns] for row in rows]
        return styled_table(data, col_widths=page_width / len(columns))

    def as_vertical_table(rows):
        """Field/Value table: one row per field instead of one column per
        field. Used for summaries with many fields but few rows, which would
        otherwise run off the page width as a wide horizontal table.
        Always returns a list of flowables."""
        if not rows:
            return [Paragraph("No data.", styles['Normal'])]
        blocks = []
        for i, row in enumerate(rows):
            if len(rows) > 1:
                blocks.append(Paragraph(f"Session {i + 1}", styles['Heading3']))
            data = [['Field', 'Value']] + [[str(k), str(v)] for k, v in row.items()]
            blocks.append(styled_table(data, col_widths=[page_width * 0.4, page_width * 0.6]))
            if i < len(rows) - 1:
                blocks.append(Spacer(1, 0.15 * inch))
        return blocks

    story = [
        Paragraph(title, styles['Title']),
        Spacer(1, 0.15 * inch),
        Paragraph('Summary', styles['Heading2']),
        *as_vertical_table(summary_rows),
        Spacer(1, 0.3 * inch),
        Paragraph('Records', styles['Heading2']),
        as_table(records_rows),
    ]

    doc.build(story)
    return buffer.getvalue()


def wipe_all_data():
    """Delete all registered people (photos on disk + in-memory face data)
    and all recorded session history, so the app doesn't accumulate data
    across uses. Called after a session's report has been generated, or
    after a processed upload's result/report has been downloaded."""
    global known_embeddings, known_names, sessions, current_session, is_capturing, _disk_scanned

    with faces_db_lock:
        # Scan the disk directly rather than relying on all_required_persons -
        # that set can be empty at a fresh boot even though old folders are
        # still sitting on disk from a previous run that never got wiped.
        if os.path.isdir(FACES_DIR):
            for entry in os.listdir(FACES_DIR):
                entry_path = os.path.join(FACES_DIR, entry)
                if os.path.isdir(entry_path):
                    shutil.rmtree(entry_path, ignore_errors=True)
        all_required_persons.clear()
        known_embeddings = []
        known_names = []
        _disk_scanned = False

    with session_lock:
        sessions = []
        current_session = None
        is_capturing = False

    if os.path.exists(CSV_FILENAME):
        os.remove(CSV_FILENAME)

    print("[INFO] Wiped all registered people and session data.")


def get_latest_jpeg():
    """Return the most recent annotated frame encoded as JPEG bytes, or None."""
    with frame_lock:
        frame = None if latest_frame is None else latest_frame.copy()
    if frame is None:
        return None
    ok, buf = cv2.imencode('.jpg', frame)
    if not ok:
        return None
    return buf.tobytes()


def camera_loop():
    """Continuously read frames, run recognition when a session is active,
    and store the annotated frame for consumers (e.g. an MJPEG endpoint).
    Runs headless - no cv2.imshow/waitKey, no keyboard module."""
    global latest_frame, capture_thread_running

    source = get_video_source()
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video source: {source!r}")
        return

    capture_thread_running = True
    print("[INFO] Camera loop started (headless).")

    # Recognition (face detection + embedding match) is CPU-heavy and can
    # easily take well over one frame interval - running it on every single
    # frame makes the loop fall behind real-time, which looks like stutter/
    # freezing. Only run it every RECOGNITION_EVERY_N_FRAMES frames; the
    # frame is still read and streamed every iteration regardless, reusing
    # the last detection boxes in between so the feed itself stays smooth.
    RECOGNITION_EVERY_N_FRAMES = 3
    frame_counter = 0
    recognized_matches_to_draw = []
    unknown_faces_to_draw = []

    try:
        while capture_thread_running:
            ret, frame = cap.read()
            if not ret:
                print("[WARN] Frame read failed.")
                continue

            display_frame = frame.copy()
            frame_counter += 1

            with session_lock:
                active = is_capturing and current_session is not None

            if active and frame_counter % RECOGNITION_EVERY_N_FRAMES == 0:
                # The heavy work happens with no lock held at all, so it
                # never blocks /api/status polling or session start/stop.
                recognized_names, recognized_matches, unknown_faces, _faces = recognize_faces_in_frame(frame, face_model)
                recognized_matches_to_draw = recognized_matches
                unknown_faces_to_draw = unknown_faces

                with session_lock:
                    if is_capturing and current_session:
                        current_session['frame_count'] += 1
                        _log_recognitions(current_session, recognized_matches, unknown_faces)
                        current_session['present_persons'].update(recognized_names)
                        current_session['unknown_count'] += len(unknown_faces)
            elif not active:
                recognized_matches_to_draw = []
                unknown_faces_to_draw = []

            draw_detections(display_frame, recognized_matches_to_draw, unknown_faces_to_draw)

            with frame_lock:
                latest_frame = display_frame

            time.sleep(0.03)
    finally:
        cap.release()
        with frame_lock:
            latest_frame = None
        print("[INFO] Camera loop stopped.")


def stop_camera_loop():
    global capture_thread_running
    capture_thread_running = False


def run():
    global known_embeddings, known_names

    # Load known faces
    print("[INFO] Loading known faces...")
    known_embeddings, known_names = load_known_faces()

    if not all_required_persons:
        print("[ERROR] No known persons found in KnownFaces directory.")
        return

    print(f"\n[INFO] System ready for {len(all_required_persons)} persons:")
    for person in sorted(all_required_persons):
        print(f"  - {person}")

    # Initialize face model
    print("\n[INFO] Loading face recognition model...")
    face_model = insightface.app.FaceAnalysis(allowed_modules=['detection', 'recognition'])
    face_model.prepare(ctx_id=0)

    # Start keyboard listener in separate thread
    keyboard_thread = threading.Thread(target=keyboard_listener, daemon=True)
    keyboard_thread.start()
    
    # Open camera
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print("[ERROR] Could not open video source.")
        return
    
    print("\n[INFO] Camera started. Waiting for session commands...")
    print("[INFO] Press 's' to start first session.")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARN] Frame read failed.")
                continue
            
            display_frame = frame.copy()
            recognized_matches_to_draw = []
            unknown_faces_to_draw = []

            # If session is active, process frame for attendance
            with session_lock:
                if is_capturing and current_session:
                    current_session['frame_count'] += 1

                    # Recognize faces in current frame using InsightFace
                    recognized_names, recognized_matches, unknown_faces, _faces = recognize_faces_in_frame(frame, face_model)
                    _log_recognitions(current_session, recognized_matches, unknown_faces)

                    # Update present persons
                    current_session['present_persons'].update(recognized_names)
                    current_session['unknown_count'] += len(unknown_faces)

                    # Store detections for drawing
                    recognized_matches_to_draw = recognized_matches
                    unknown_faces_to_draw = unknown_faces

            # Draw recognized faces with name + confidence labels
            draw_detections(display_frame, recognized_matches_to_draw, unknown_faces_to_draw)

            # Display session status
            with session_lock:
                if is_capturing and current_session:
                    status_text = f"SESSION {current_session['session_id']} - ACTIVE"
                    status_color = (0, 255, 0)
                    
                    # Show recognized count
                    present_count = len(current_session['present_persons'])
                    cv2.putText(display_frame, f"Present: {present_count}/{len(all_required_persons)}", 
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    # Show recognized persons
                    y_offset = 90
                    for person in sorted(current_session['present_persons']):
                        cv2.putText(display_frame, f"✓ {person}", 
                                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        y_offset += 25
                else:
                    status_text = "SESSION INACTIVE - Press 's' to start"
                    status_color = (0, 0, 255)
            
            # Show session status
            cv2.putText(display_frame, status_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            
            # Show instructions
            cv2.putText(display_frame, "Press 's': Start/Stop Session", 
                       (10, display_frame.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(display_frame, "Press 'q': Quit & Save", 
                       (10, display_frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow("Attendance System - Session Based", display_frame)
            
            # Check for quit from main thread
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n[INFO] Quit requested from main thread...")
                if is_capturing:
                    print("[INFO] Stopping active session before quitting...")
                    stop_session()
                break
            
            # Small delay to prevent high CPU usage
            time.sleep(0.03)
            
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
    
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        # Save all sessions to CSV
        if sessions:
            save_sessions_to_csv()
        else:
            print("[INFO] No sessions were recorded.")
        
        print("\n[INFO] System shutdown complete.")

if __name__ == "__main__":
    run()