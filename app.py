from flask import Flask, render_template, request, redirect, url_for, Response, jsonify, session
import cv2
import os
import time
import threading
from detector import detect_vehicles

app = Flask(__name__)
app.secret_key = "itms_secret_key_123"

USERNAME = "traffic-admin"
PASSWORD = "admin123"

latest_lane_stats = {
    1: {"density": 0, "total": 0, "ambulance": "No"},
    2: {"density": 0, "total": 0, "ambulance": "No"},
    3: {"density": 0, "total": 0, "ambulance": "No"},
    4: {"density": 0, "total": 0, "ambulance": "No"},
}

cumulative_stats = {
    1: {"car": 0, "bus": 0, "truck": 0, "motorcycle": 0, "ambulance": 0},
    2: {"car": 0, "bus": 0, "truck": 0, "motorcycle": 0, "ambulance": 0},
    3: {"car": 0, "bus": 0, "truck": 0, "motorcycle": 0, "ambulance": 0},
    4: {"car": 0, "bus": 0, "truck": 0, "motorcycle": 0, "ambulance": 0},
}

cycle_peaks = {
    1: {"car": 0, "bus": 0, "truck": 0, "motorcycle": 0},
    2: {"car": 0, "bus": 0, "truck": 0, "motorcycle": 0},
    3: {"car": 0, "bus": 0, "truck": 0, "motorcycle": 0},
    4: {"car": 0, "bus": 0, "truck": 0, "motorcycle": 0},
}

signal_state = {
    1: {"state": "red", "timer": "--"},
    2: {"state": "red", "timer": "--"},
    3: {"state": "red", "timer": "--"},
    4: {"state": "red", "timer": "--"},
}

controller = {
    "green_lane": 1,
    "green_remaining": 20,
    "yellow_lane": None,
    "yellow_remaining": 0,
}

lock = threading.Lock()
controller_started = False


def is_logged_in():
    return session.get("logged_in", False)


def choose_next_lane(exclude_lane=None):
    densities = {
        i: latest_lane_stats[i]["density"]
        for i in range(1, 5)
        if i != exclude_lane
    }
    if not densities:
        return 1
    return max(densities, key=densities.get)


def add_cycle_peak_to_cumulative(lane):
    cumulative_stats[lane]["car"] += cycle_peaks[lane]["car"]
    cumulative_stats[lane]["bus"] += cycle_peaks[lane]["bus"]
    cumulative_stats[lane]["truck"] += cycle_peaks[lane]["truck"]
    cumulative_stats[lane]["motorcycle"] += cycle_peaks[lane]["motorcycle"]

    cycle_peaks[lane]["car"] = 0
    cycle_peaks[lane]["bus"] = 0
    cycle_peaks[lane]["truck"] = 0
    cycle_peaks[lane]["motorcycle"] = 0


def update_cycle_peaks(lane_number, counts):
    cycle_peaks[lane_number]["car"] = max(cycle_peaks[lane_number]["car"], counts["car"])
    cycle_peaks[lane_number]["bus"] = max(cycle_peaks[lane_number]["bus"], counts["bus"])
    cycle_peaks[lane_number]["truck"] = max(cycle_peaks[lane_number]["truck"], counts["truck"])
    cycle_peaks[lane_number]["motorcycle"] = max(cycle_peaks[lane_number]["motorcycle"], counts["motorcycle"])


def compute_signal_display():
    for i in range(1, 5):
        signal_state[i]["state"] = "red"
        signal_state[i]["timer"] = "--"

    if controller["yellow_lane"] is not None:
        lane = controller["yellow_lane"]
        signal_state[lane]["state"] = "yellow"
        signal_state[lane]["timer"] = controller["yellow_remaining"]
    else:
        lane = controller["green_lane"]
        signal_state[lane]["state"] = "green"
        signal_state[lane]["timer"] = controller["green_remaining"]


def signal_controller_loop():
    while True:
        time.sleep(1)

        with lock:
            if controller["yellow_lane"] is not None:
                controller["yellow_remaining"] -= 1

                if controller["yellow_remaining"] <= 0:
                    next_lane = choose_next_lane(exclude_lane=controller["yellow_lane"])
                    controller["green_lane"] = next_lane
                    controller["green_remaining"] = 20
                    controller["yellow_lane"] = None
                    controller["yellow_remaining"] = 0
            else:
                controller["green_remaining"] -= 1

                if controller["green_remaining"] <= 0:
                    old_green = controller["green_lane"]
                    add_cycle_peak_to_cumulative(old_green)
                    controller["yellow_lane"] = old_green
                    controller["yellow_remaining"] = 3

            compute_signal_display()


def start_controller_once():
    global controller_started
    if not controller_started:
        controller_started = True
        compute_signal_display()
        threading.Thread(target=signal_controller_loop, daemon=True).start()


def generate_video(video_name, lane_number):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    video_path = os.path.join(base_dir, "videos", video_name)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"ERROR: Could not open video: {video_path}")
        return

    while True:
        success, frame = cap.read()

        if not success:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        frame = cv2.resize(frame, (620, 320))
        detected_frame, counts, density = detect_vehicles(frame)
        total = sum(counts.values())

        with lock:
            latest_lane_stats[lane_number] = {
                "density": density,
                "total": total,
                "ambulance": "No"
            }
            update_cycle_peaks(lane_number, counts)

            sig = signal_state[lane_number]["state"]
            timer = signal_state[lane_number]["timer"]

        # Top lane label
        cv2.putText(
            detected_frame,
            f"Lane {lane_number}",
            (245, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2
        )

    
        # Bottom info bar - taller for full visibility
        cv2.rectangle(detected_frame, (0, 265), (620, 320), (0, 0, 0), -1)

        timer_text = str(timer) if timer != "--" else "--"

        cv2.putText(
            detected_frame,
            f"Density: {density} | Ambulance: No | Time: {timer_text}",
            (10, 300),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.82,
            (255, 255, 255),
            2
        )

        ret, buffer = cv2.imencode(".jpg", detected_frame)
        if not ret:
            continue

        frame_bytes = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )


@app.before_request
def initialize_controller():
    start_controller_once()


@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        user = request.form.get("username", "").strip()
        pwd = request.form.get("password", "").strip()

        if user == USERNAME and pwd == PASSWORD:
            session["logged_in"] = True
            return redirect(url_for("home"))

        return render_template("login.html", error="Invalid username or password")

    return render_template("login.html", error="")


@app.route("/home")
def home():
    if not is_logged_in():
        return redirect(url_for("login"))
    return render_template("home.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/dashboard")
def dashboard():
    if not is_logged_in():
        return redirect(url_for("login"))
    return render_template("dashboard.html")


@app.route("/analysis")
def analysis():
    if not is_logged_in():
        return redirect(url_for("login"))

    with lock:
        stats = {k: v.copy() for k, v in cumulative_stats.items()}
        current_density = {k: latest_lane_stats[k]["density"] for k in range(1, 5)}

    return render_template(
        "analysis.html",
        stats=stats,
        current_density=current_density
    )


@app.route("/video1")
def video1():
    if not is_logged_in():
        return redirect(url_for("login"))
    return Response(generate_video("lane1.mp4", 1), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/video2")
def video2():
    if not is_logged_in():
        return redirect(url_for("login"))
    return Response(generate_video("lane2.mp4", 2), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/video3")
def video3():
    if not is_logged_in():
        return redirect(url_for("login"))
    return Response(generate_video("lane3.mp4", 3), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/video4")
def video4():
    if not is_logged_in():
        return redirect(url_for("login"))
    return Response(generate_video("lane4.mp4", 4), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/lane_data")
def lane_data():
    if not is_logged_in():
        return jsonify({"error": "unauthorized"}), 401

    with lock:
        data = {}
        for i in range(1, 5):
            data[f"lane{i}"] = {
                "signal": signal_state[i]["state"],
                "timer": signal_state[i]["timer"],
                "density": latest_lane_stats[i]["density"],
                "ambulance": latest_lane_stats[i]["ambulance"],
                "total": latest_lane_stats[i]["total"]
            }

    return jsonify(data)


if __name__ == "__main__":
    app.run(debug=True, threaded=True)