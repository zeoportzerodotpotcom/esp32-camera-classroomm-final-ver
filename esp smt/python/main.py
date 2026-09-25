from flask import Flask, render_template, Response, request, jsonify
import cv2
import numpy as np
import requests
import threading
import time
import socket


# ============================================================
# ESP32
# ============================================================

ESP_IP = "espip"

CAPTURE_URL = f"http://{ESP_IP}/capture"
LIGHT_URL = f"http://{ESP_IP}/light"


# ============================================================
# FLASK
# ============================================================

app = Flask(__name__)


# ============================================================
# SYSTEM
# ============================================================

MODE = "AUTO"

NUM_LED = 6

LIGHTING_MODES = {

    "standard_study": {
        "name": "Học bình thường",
        "pwms": [180, 220, 220, 220, 180, 120],
        "auto_lock": False
    },

    "slide_presentation": {
        "name": "Trình chiếu slide",
        "pwms": [60, 60, 60, 60, 40, 0],
        "auto_lock": False
    },

    "lecturing": {
        "name": "Thuyết trình",
        "pwms": [80, 80, 80, 80, 255, 30],
        "auto_lock": False
    },

    "exam": {
        "name": "Thi kiểm tra",
        "pwms": [255, 255, 255, 255, 255, 255],
        "auto_lock": True
    }
}


# ============================================================
# DETECTION WORKING SIZE
# ============================================================
#
# ROI của bạn được thiết kế theo hệ tọa độ 640x480.
#
# Camera OV3660 có thể chụp ở độ phân giải cao hơn.
# Ta giữ nguyên frame gốc để hiển thị, nhưng dùng bản 640x480
# để xử lý detection.
#
# ============================================================

DETECTION_WIDTH = 640
DETECTION_HEIGHT = 480


# ============================================================
# CAMERA
# ============================================================

CAPTURE_TIMEOUT = 5

ESP32_REQUEST_HEADERS = {
    "Connection": "close"
}

CAMERA_INTERVAL = 0.15

STREAM_INTERVAL = 0.03


# ============================================================
# ROI
# ============================================================

ROIS = [

    {
        "name": "Cuối phòng 1",
        "x": 68,
        "y": 411,
        "w": 467,
        "h": 68,
        "led": 0
    },

    {
        "name": "Trái 42",
        "x": 6,
        "y": 127,
        "w": 150,
        "h": 267,
        "led": 1
    },

    {
        "name": "Giữa 41",
        "x": 224,
        "y": 128,
        "w": 171,
        "h": 268,
        "led": 2
    },

    {
        "name": "Phải 39",
        "x": 462,
        "y": 128,
        "w": 171,
        "h": 263,
        "led": 3
    },

    {
        "name": "Bục giảng 40",
        "x": 67,
        "y": 36,
        "w": 474,
        "h": 74,
        "led": 4
    },

    {
        "name": "Máy chiếu 2",
        "x": 90,
        "y": 0,
        "w": 243,
        "h": 65,
        "led": 5
    }

]


# ============================================================
# MOTION DETECTION
# ============================================================

# Ngưỡng khác biệt giữa frame hiện tại và frame trước.
#
# Giá trị nhỏ:
#   nhạy hơn
#
# Giá trị lớn:
#   ít nhiễu hơn
#
MOTION_THRESHOLD = 14


# Số pixel motion tối thiểu trong ROI
MIN_MOTION_PIXELS = 180


# Tỷ lệ motion tối thiểu trong ROI
MIN_MOTION_RATIO = 0.004


# Phải xuất hiện liên tiếp bao nhiêu frame
# trước khi xác nhận object.
MOTION_CONFIRM_FRAMES = 3


# ============================================================
# GLOBAL LIGHT CHANGE
# ============================================================
#
# Nếu phần lớn khung hình thay đổi cùng lúc,
# coi đó là thay đổi ánh sáng/camera thay vì vật thể.
#
# ============================================================

GLOBAL_CHANGE_RATIO = 0.33

GLOBAL_CHANGE_MEAN = 8.0


# ============================================================
# MORPHOLOGY
# ============================================================

MOTION_OPEN_SIZE = 3

MOTION_CLOSE_SIZE = 7

MIN_COMPONENT_AREA = 80


# ============================================================
# BACKGROUND STRUCTURE
# ============================================================
#
# Background chỉ dùng để xác nhận object đã biến mất.
#
# Nó KHÔNG được dùng làm trigger chính.
#
# ============================================================

CANNY_LOW = 50

CANNY_HIGH = 120

EDGE_DILATE_SIZE = 7


# Nếu cấu trúc hiện tại gần background trong số frame
# nhất định thì coi object đã biến mất.
BACKGROUND_EDGE_THRESHOLD_RATIO = 0.020

BACKGROUND_EDGE_MIN_PIXELS = 250

RELEASE_CONFIRM_FRAMES = 5


# ============================================================
# BACKGROUND ADAPTATION
# ============================================================

BACKGROUND_ALPHA = 0.005


# ============================================================
# BACKGROUND BUILDING
# ============================================================

BACKGROUND_WARMUP_FRAMES = 8

BACKGROUND_FRAMES = 20

BACKGROUND_CAPTURE_INTERVAL = 0.10


# Sau khi LED thay đổi thì chờ camera ổn định
LED_SETTLE_TIME = 0.80


# ============================================================
# GLOBAL DATA
# ============================================================

background_feature = None

background_edges = None

previous_feature = None

previous_gray = None

latest_frame = None

latest_jpeg = None

esp32_online = False


# ============================================================
# BACKGROUND STATUS
# ============================================================

background_building = False

background_progress = 0

background_status_text = "NOT READY"

display_status_text = "NORMAL"


# ============================================================
# ROI STATUS
# ============================================================

latest_roi_data = [

    {
        "name": roi["name"],
        "led": roi["led"],
        "detected": False,
        "pixels": 0,
        "motion_ratio": 0.0,
        "background_pixels": 0
    }

    for roi in ROIS

]


# ============================================================
# ROI STATE
# ============================================================

roi_detect_state = [False] * len(ROIS)

roi_positive_count = [0] * len(ROIS)

roi_negative_count = [0] * len(ROIS)

roi_last_motion_time = [0.0] * len(ROIS)

roi_release_count = [0] * len(ROIS)


# ============================================================
# LED CACHE
# ============================================================

last_led_pwm = [0] * NUM_LED

last_led_change_time = 0.0

lighting_mode_enabled = False

lighting_mode_name = None


# ============================================================
# LOCKS
# ============================================================

background_lock = threading.Lock()

frame_lock = threading.Lock()

mode_lock = threading.Lock()

led_lock = threading.Lock()

status_lock = threading.Lock()

roi_lock = threading.Lock()

capture_lock = threading.Lock()

esp_request_lock = threading.Lock()

lighting_mode_lock = threading.Lock()


# ============================================================
# REQUEST SESSION
# ============================================================

session = requests.Session()


# ============================================================
# CAPTURE
# ============================================================

def capture():

    global esp32_online

    with capture_lock:

        try:

            with esp_request_lock:

                response = session.get(

                    CAPTURE_URL,

                    headers=ESP32_REQUEST_HEADERS,

                    timeout=(2, CAPTURE_TIMEOUT)

                )


            if response.status_code != 200:

                with status_lock:

                    esp32_online = False


                print(

                    "Capture HTTP error:",

                    response.status_code

                )

                return None


            if not response.content:

                with status_lock:

                    esp32_online = False


                print(

                    "ESP32 returned empty image"

                )

                return None


            data = np.frombuffer(

                response.content,

                dtype=np.uint8

            )


            frame = cv2.imdecode(

                data,

                cv2.IMREAD_COLOR

            )


            if frame is None:

                with status_lock:

                    esp32_online = False


                print(

                    "Không decode được ảnh"

                )

                return None


            with status_lock:

                esp32_online = True


            return frame


        except requests.exceptions.Timeout:

            session.close()

            with status_lock:

                esp32_online = False


            print(

                "ESP32 capture timeout"

            )

            return None


        except requests.exceptions.ConnectionError as e:

            session.close()

            with status_lock:

                esp32_online = False


            print(

                "ESP32 connection error:",

                e

            )

            return None


        except Exception as e:

            session.close()

            with status_lock:

                esp32_online = False


            print(

                "ESP32 camera error:",

                e

            )

            return None


# ============================================================
# SET LED
# ============================================================

def set_led(
    led_id,
    pwm
):

    global esp32_online

    global last_led_change_time


    if led_id < 0 or led_id >= NUM_LED:

        return False


    pwm = max(

        0,

        min(
            255,
            int(pwm)
        )

    )


    try:

        with esp_request_lock:

            response = session.get(

                LIGHT_URL,

                params={

                    "id": led_id,

                    "pwm": pwm

                },

                headers=ESP32_REQUEST_HEADERS,

                timeout=(2, 4)

            )


        if response.status_code == 200:

            with led_lock:

                old_pwm = last_led_pwm[led_id]

                last_led_pwm[led_id] = pwm


            if old_pwm != pwm:

                last_led_change_time = time.time()


            with status_lock:

                esp32_online = True


            return True


        with status_lock:

            esp32_online = False


        print(

            f"LED {led_id} HTTP error:",

            response.status_code

        )


        return False


    except requests.exceptions.Timeout:

        session.close()

        with status_lock:

            esp32_online = False


        print(

            f"LED {led_id} timeout"

        )


        return False


    except requests.exceptions.ConnectionError as e:

        session.close()

        with status_lock:

            esp32_online = False


        print(

            f"LED {led_id} connection error:",

            e

        )


        return False


    except Exception as e:

        session.close()

        with status_lock:

            esp32_online = False


        print(

            f"LED {led_id} error:",

            e

        )


        return False


# ============================================================
# SET LED ONLY IF PWM CHANGED
# ============================================================

def set_led_if_changed(
    led_id,
    pwm
):

    if led_id < 0 or led_id >= NUM_LED:

        return False


    pwm = max(

        0,

        min(
            255,
            int(pwm)
        )

    )


    with led_lock:

        current_pwm = last_led_pwm[led_id]


    if current_pwm == pwm:

        return True


    return set_led(

        led_id,

        pwm

    )


# ============================================================
# LIGHTING MODE
# ============================================================

def apply_lighting_mode(
    mode_name
):

    global lighting_mode_enabled
    global lighting_mode_name


    mode = LIGHTING_MODES.get(
        mode_name
    )


    if mode is None:

        return False


    for led_id, pwm in enumerate(
        mode["pwms"]
    ):

        if not set_led(
            led_id,
            pwm
        ):

            return False


    with lighting_mode_lock:

        lighting_mode_enabled = True

        lighting_mode_name = mode_name


    return True


def disable_lighting_mode():

    global lighting_mode_enabled
    global lighting_mode_name


    with lighting_mode_lock:

        lighting_mode_enabled = False

        lighting_mode_name = None


# ============================================================
# TURN OFF ALL LED
# ============================================================

def turn_off_all():

    for i in range(NUM_LED):

        set_led(

            i,

            0

        )


# ============================================================
# RESET DETECTION STATES
# ============================================================

def reset_detection_state():

    global roi_detect_state
    global roi_positive_count
    global roi_negative_count
    global roi_last_motion_time
    global roi_release_count


    with roi_lock:

        roi_detect_state = [

            False

            for _ in ROIS

        ]


        roi_positive_count = [

            0

            for _ in ROIS

        ]


        roi_negative_count = [

            0

            for _ in ROIS

        ]


        roi_last_motion_time = [

            0.0

            for _ in ROIS

        ]


        roi_release_count = [

            0

            for _ in ROIS

        ]


# ============================================================
# RESET STATUS
# ============================================================

def reset_status():

    global latest_roi_data

    latest_roi_data = [

        {

            "name": roi["name"],

            "led": roi["led"],

            "detected": False,

            "pixels": 0,

            "motion_ratio": 0.0,

            "background_pixels": 0

        }

        for roi in ROIS

    ]


    reset_detection_state()


# ============================================================
# RESIZE FOR DETECTION
# ============================================================

def resize_for_detection(
    frame
):

    h, w = frame.shape[:2]


    if (

        w == DETECTION_WIDTH

        and

        h == DETECTION_HEIGHT

    ):

        return frame.copy()


    return cv2.resize(

        frame,

        (
            DETECTION_WIDTH,
            DETECTION_HEIGHT
        ),

        interpolation=cv2.INTER_AREA

    )


# ============================================================
# FEATURE FRAME
# ============================================================

def make_feature_frame(
    gray
):

    """
    Tạo ảnh high-pass.

    Mục tiêu:
    giảm ảnh hưởng thay đổi sáng/tối chậm,
    giữ lại hình dạng/cạnh chuyển động.
    """

    gray_blur = cv2.GaussianBlur(

        gray,

        (5, 5),

        0

    )


    local_light = cv2.GaussianBlur(

        gray_blur.astype(
            np.float32
        ),

        (51, 51),

        0

    )


    feature = (

        gray_blur.astype(
            np.float32
        )

        -

        local_light

        +

        128.0

    )


    feature = np.clip(

        feature,

        0,

        255

    ).astype(

        np.uint8

    )


    return feature


# ============================================================
# EDGE FRAME
# ============================================================

def make_edges(
    gray
):

    gray = cv2.GaussianBlur(

        gray,

        (5, 5),

        0

    )


    edges = cv2.Canny(

        gray,

        CANNY_LOW,

        CANNY_HIGH

    )


    kernel = np.ones(

        (3, 3),

        np.uint8

    )


    edges = cv2.morphologyEx(

        edges,

        cv2.MORPH_CLOSE,

        kernel

    )


    return edges


# ============================================================
# GLOBAL CHANGE
# ============================================================

def calculate_global_change(
    gray,
    previous_gray
):

    if previous_gray is None:

        return 0.0, 0.0


    delta = (

        gray.astype(
            np.float32
        )

        -

        previous_gray.astype(
            np.float32
        )

    )


    # Loại sự thay đổi độ sáng đồng đều
    median_shift = float(

        np.median(delta)

    )


    corrected = np.abs(

        delta
        -
        median_shift

    )


    mean_change = float(

        np.mean(
            corrected
        )

    )


    changed_ratio = float(

        np.mean(

            corrected
            >
            MOTION_THRESHOLD

        )

    )


    return (

        mean_change,

        changed_ratio

    )


# ============================================================
# CREATE MOTION MASK
# ============================================================

def make_motion_mask(
    current_feature,
    previous_feature
):

    if previous_feature is None:

        return np.zeros(

            current_feature.shape,

            dtype=np.uint8

        )


    delta = cv2.absdiff(

        current_feature,

        previous_feature

    )


    _, mask = cv2.threshold(

        delta,

        MOTION_THRESHOLD,

        255,

        cv2.THRESH_BINARY

    )


    open_kernel = np.ones(

        (
            MOTION_OPEN_SIZE,
            MOTION_OPEN_SIZE
        ),

        np.uint8

    )


    close_kernel = np.ones(

        (
            MOTION_CLOSE_SIZE,
            MOTION_CLOSE_SIZE
        ),

        np.uint8

    )


    mask = cv2.morphologyEx(

        mask,

        cv2.MORPH_OPEN,

        open_kernel

    )


    mask = cv2.morphologyEx(

        mask,

        cv2.MORPH_CLOSE,

        close_kernel

    )


    # --------------------------------------------------------
    # Loại các component rất nhỏ
    # --------------------------------------------------------

    num_labels, labels, stats, _ = (

        cv2.connectedComponentsWithStats(

            mask,

            connectivity=8

        )

    )


    cleaned = np.zeros_like(

        mask

    )


    for label in range(

        1,

        num_labels

    ):

        area = stats[

            label,

            cv2.CC_STAT_AREA

        ]


        if area >= MIN_COMPONENT_AREA:

            cleaned[

                labels == label

            ] = 255


    return cleaned


# ============================================================
# BACKGROUND STRUCTURE DIFFERENCE
# ============================================================

def background_structure_difference(
    current_edges,
    bg_edges,
    x1,
    y1,
    x2,
    y2
):

    if bg_edges is None:

        return 0, 0


    current_roi_edges = current_edges[

        y1:y2,

        x1:x2

    ]


    bg_roi_edges = bg_edges[

        y1:y2,

        x1:x2

    ]


    size = EDGE_DILATE_SIZE


    if size % 2 == 0:

        size += 1


    kernel = np.ones(

        (
            size,
            size
        ),

        np.uint8

    )


    bg_dilated = cv2.dilate(

        bg_roi_edges,

        kernel

    )


    current_dilated = cv2.dilate(

        current_roi_edges,

        kernel

    )


    # --------------------------------------------------------
    # Cạnh mới
    # --------------------------------------------------------

    current_new = cv2.bitwise_and(

        current_roi_edges,

        cv2.bitwise_not(
            bg_dilated
        )

    )


    # --------------------------------------------------------
    # Cạnh bị mất
    # --------------------------------------------------------

    current_missing = cv2.bitwise_and(

        bg_roi_edges,

        cv2.bitwise_not(
            current_dilated
        )

    )


    structural_change = cv2.bitwise_or(

        current_new,

        current_missing

    )


    changed_pixels = int(

        cv2.countNonZero(

            structural_change

        )

    )


    roi_area = int(

        structural_change.shape[0]

        *

        structural_change.shape[1]

    )


    return (

        changed_pixels,

        roi_area

    )


# ============================================================
# BUILD BACKGROUND
# ============================================================

def build_background():

    global background_feature

    global background_edges

    global previous_feature

    global previous_gray

    global background_building

    global background_progress

    global background_status_text


    print()

    print(
        "=========================================="
    )

    print(
        "BUILD EMPTY SCENE BACKGROUND"
    )

    print(
        "ALL LEDs OFF"
    )

    print(
        "=========================================="
    )


    background_building = True

    background_progress = 0

    background_status_text = "TURNING LED OFF"


    reset_detection_state()


    # --------------------------------------------------------
    # LED OFF
    # --------------------------------------------------------

    turn_off_all()


    # --------------------------------------------------------
    # Đợi camera/ánh sáng ổn định
    # --------------------------------------------------------

    time.sleep(

        LED_SETTLE_TIME

    )


    # --------------------------------------------------------
    # Warm-up
    # --------------------------------------------------------

    background_status_text = "CAMERA WARMUP"


    warmup_count = 0


    while (

        warmup_count

        <

        BACKGROUND_WARMUP_FRAMES

    ):

        frame = capture()


        if frame is not None:

            warmup_count += 1


        time.sleep(

            BACKGROUND_CAPTURE_INTERVAL

        )


    # --------------------------------------------------------
    # Capture background sequence
    # --------------------------------------------------------

    background_status_text = "CAPTURING BG"


    frames = []


    while len(frames) < BACKGROUND_FRAMES:

        frame = capture()


        if frame is None:

            time.sleep(

                BACKGROUND_CAPTURE_INTERVAL

            )

            continue


        work = resize_for_detection(

            frame

        )


        gray = cv2.cvtColor(

            work,

            cv2.COLOR_BGR2GRAY

        )


        feature = make_feature_frame(

            gray

        )


        frames.append(

            feature

        )


        background_progress = int(

            (

                len(frames)

                /

                BACKGROUND_FRAMES

            )

            *

            100

        )


        print(

            f"Background "
            f"{len(frames)}/"
            f"{BACKGROUND_FRAMES}"

        )


        time.sleep(

            BACKGROUND_CAPTURE_INTERVAL

        )


    # --------------------------------------------------------
    # Median background
    # --------------------------------------------------------

    stack = np.stack(

        frames,

        axis=0

    )


    median_feature = np.median(

        stack,

        axis=0

    ).astype(

        np.uint8

    )


    median_edges = make_edges(

        median_feature

    )


    with background_lock:

        background_feature = (

            median_feature.astype(

                np.float32

            )

        )

        background_edges = (

            median_edges.copy()

        )


    # --------------------------------------------------------
    # Reset temporal reference
    # --------------------------------------------------------

    previous_feature = (

        median_feature.copy()

    )


    previous_gray = cv2.GaussianBlur(

        median_feature,

        (5, 5),

        0

    )


    reset_detection_state()


    background_building = False

    background_progress = 100

    background_status_text = "READY"


    print()

    print(
        "BACKGROUND READY"
    )

    print(
        "Motion is the PRIMARY detection signal."
    )

    print(
        "Static furniture will not trigger by BG difference."
    )

    print()


    return True


# ============================================================
# PROCESS FRAME
# ============================================================

def process_frame(
    frame
):

    global previous_feature

    global previous_gray

    global background_feature

    global background_edges

    global latest_roi_data

    global display_status_text


    # ========================================================
    # BACKGROUND BUILD
    # ========================================================

    if background_building:

        work = resize_for_detection(

            frame

        )


        return work


    # ========================================================
    # WORKING FRAME
    # ========================================================

    work = resize_for_detection(

        frame

    )


    gray = cv2.cvtColor(

        work,

        cv2.COLOR_BGR2GRAY

    )


    gray = cv2.GaussianBlur(

        gray,

        (5, 5),

        0

    )


    current_feature = make_feature_frame(

        gray

    )


    current_edges = make_edges(

        gray

    )


    # ========================================================
    # BACKGROUND CHECK
    # ========================================================

    with background_lock:

        if background_feature is None:

            previous_feature = (

                current_feature.copy()

            )

            previous_gray = gray.copy()


            return work


        bg_edges = (

            background_edges.copy()

            if background_edges is not None

            else None

        )


    # ========================================================
    # FRAME-TO-FRAME MOTION
    # ========================================================

    motion_mask = make_motion_mask(

        current_feature,

        previous_feature

    )


    # ========================================================
    # GLOBAL CHANGE
    # ========================================================

    global_mean_change, global_change_ratio = (

        calculate_global_change(

            gray,

            previous_gray

        )

    )


    global_light_event = (

        global_change_ratio
        >=
        GLOBAL_CHANGE_RATIO

        and

        global_mean_change
        >=
        GLOBAL_CHANGE_MEAN

    )


    # ========================================================
    # LED SETTLING
    # ========================================================

    elapsed = (

        time.time()

        -

        last_led_change_time

    )


    led_is_stabilizing = (

        elapsed < LED_SETTLE_TIME

    )


    # ========================================================
    # ROI DATA
    # ========================================================

    new_roi_data = []

    pending_led_commands = {}

    detected_any = False


    # ========================================================
    # PROCESS EACH ROI
    # ========================================================

    for index, r in enumerate(ROIS):

        x = r["x"]

        y = r["y"]

        w = r["w"]

        h = r["h"]

        led = r["led"]


        x1 = max(

            0,

            x

        )


        y1 = max(

            0,

            y

        )


        x2 = min(

            DETECTION_WIDTH,

            x + w

        )


        y2 = min(

            DETECTION_HEIGHT,

            y + h

        )


        if x1 >= x2 or y1 >= y2:

            new_roi_data.append({

                "name":
                    r["name"],

                "led":
                    led,

                "detected":
                    False,

                "pixels":
                    0,

                "motion_ratio":
                    0.0,

                "background_pixels":
                    0

            })

            continue


        # ----------------------------------------------------
        # Motion ROI
        # ----------------------------------------------------

        motion_roi = motion_mask[

            y1:y2,

            x1:x2

        ]


        motion_pixels = int(

            cv2.countNonZero(

                motion_roi

            )

        )


        roi_area = int(

            motion_roi.shape[0]

            *

            motion_roi.shape[1]

        )


        motion_ratio = (

            motion_pixels

            /

            max(

                roi_area,

                1

            )

        )


        # ----------------------------------------------------
        # Find meaningful component
        # ----------------------------------------------------

        meaningful_component = False


        num_labels, labels, stats, _ = (

            cv2.connectedComponentsWithStats(

                motion_roi,

                connectivity=8

            )

        )


        largest_component = 0


        for label in range(

            1,

            num_labels

        ):

            area = stats[

                label,

                cv2.CC_STAT_AREA

            ]


            if area > largest_component:

                largest_component = area


            if area >= MIN_COMPONENT_AREA:

                meaningful_component = True


        # ----------------------------------------------------
        # Raw motion candidate
        # ----------------------------------------------------

        raw_motion = (

            meaningful_component

            and

            (

                motion_pixels
                >=
                MIN_MOTION_PIXELS

                or

                motion_ratio
                >=
                MIN_MOTION_RATIO

            )

        )


        # ----------------------------------------------------
        # Reject global light change
        # ----------------------------------------------------

        if global_light_event:

            raw_motion = False


        # ----------------------------------------------------
        # Reject LED settling
        # ----------------------------------------------------

        if led_is_stabilizing:

            raw_motion = False


        # ====================================================
        # STRUCTURAL BACKGROUND DIFFERENCE
        # ====================================================

        structural_pixels, structural_area = (

            background_structure_difference(

                current_edges,

                bg_edges,

                x1,

                y1,

                x2,

                y2

            )

        )


        structural_ratio = (

            structural_pixels

            /

            max(

                structural_area,

                1

            )

        )


        # ====================================================
        # ROI STATE MACHINE
        # ====================================================

        with roi_lock:

            state = (

                roi_detect_state[index]

            )


            # =================================================
            # RAW MOTION
            # =================================================

            if raw_motion:

                roi_last_motion_time[index] = (

                    time.time()

                )


                roi_release_count[index] = 0

                roi_negative_count[index] = 0


                if not state:

                    roi_positive_count[index] += 1


                    if (

                        roi_positive_count[index]

                        >=

                        MOTION_CONFIRM_FRAMES

                    ):

                        roi_detect_state[index] = True

                        roi_positive_count[index] = 0


                else:

                    roi_positive_count[index] = 0


            # =================================================
            # NO MOTION
            # =================================================

            else:

                roi_positive_count[index] = 0


                if state:

                    # ----------------------------------------
                    # Object currently active.
                    # Do not turn it off immediately when
                    # object stops moving.
                    # ----------------------------------------

                    time_since_motion = (

                        time.time()

                        -

                        roi_last_motion_time[index]

                    )


                    if (

                        time_since_motion

                        >=

                        LED_SETTLE_TIME

                    ):

                        structure_is_background = (

                            (

                                structural_ratio
                                <=
                                BACKGROUND_EDGE_THRESHOLD_RATIO

                            )

                            or

                            (

                                structural_pixels
                                <=
                                BACKGROUND_EDGE_MIN_PIXELS

                            )

                        )


                        if structure_is_background:

                            roi_release_count[index] += 1

                        else:

                            roi_release_count[index] = 0


                        if (

                            roi_release_count[index]

                            >=

                            RELEASE_CONFIRM_FRAMES

                        ):

                            roi_detect_state[index] = False

                            roi_release_count[index] = 0


                    else:

                        roi_release_count[index] = 0


                else:

                    roi_negative_count[index] += 1


            state = (

                roi_detect_state[index]

            )


        # ====================================================
        # FINAL DETECTION
        # ====================================================

        detected = bool(

            state

        )


        if detected:

            detected_any = True

            pending_led_commands[led] = 255

        else:

            # Only set default if another ROI has not already
            # requested the same LED ON.
            pending_led_commands.setdefault(

                led,

                0

            )


        # ====================================================
        # ROI STATUS
        # ====================================================

        new_roi_data.append({

            "name":
                r["name"],

            "led":
                led,

            "detected":
                detected,

            "pixels":
                int(
                    motion_pixels
                ),

            "motion_ratio":
                round(
                    motion_ratio,
                    5
                ),

            "background_pixels":
                int(
                    structural_pixels
                )

        })


        # ====================================================
        # ROI COLOR
        # ====================================================

        if detected:

            color = (

                0,
                0,
                255

            )

        else:

            color = (

                0,
                255,
                0

            )


        # ====================================================
        # DRAW ROI
        # ====================================================

        cv2.rectangle(

            work,

            (x1, y1),

            (x2, y2),

            color,

            2

        )


        # ====================================================
        # NAME
        # ====================================================

        cv2.putText(

            work,

            r["name"],

            (
                x1,
                max(
                    18,
                    y1 - 5
                )
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.55,

            color,

            2,

            cv2.LINE_AA

        )


        # ====================================================
        # MOTION
        # ====================================================

        cv2.putText(

            work,

            f"M:{motion_pixels}",

            (
                x1 + 5,
                min(
                    DETECTION_HEIGHT - 28,
                    y1 + 38
                )
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.48,

            color,

            1,

            cv2.LINE_AA

        )


        # ====================================================
        # BACKGROUND STRUCTURE
        # ====================================================

        cv2.putText(

            work,

            f"BG:{structural_pixels}",

            (
                x1 + 5,
                min(
                    DETECTION_HEIGHT - 8,
                    y1 + 56
                )
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.43,

            color,

            1,

            cv2.LINE_AA

        )


    # ========================================================
    # UPDATE STATUS
    # ========================================================

    with roi_lock:

        latest_roi_data = new_roi_data


    # ========================================================
    # LED COMMANDS
    # ========================================================
    #
    # IMPORTANT:
    #
    # Tất cả ROI được xử lý xong trước khi LED thay đổi.
    #
    # Không được để ROI 1 bật đèn rồi ánh sáng đó ảnh hưởng
    # ngay lập tức tới ROI 2 trong cùng một frame.
    #
    # ========================================================

    with lighting_mode_lock:

        lighting_mode_active = lighting_mode_enabled

        lighting_mode = LIGHTING_MODES.get(
            lighting_mode_name
        )

        auto_locked = (
            lighting_mode_active
            and
            lighting_mode is not None
            and
            lighting_mode["auto_lock"]
        )


    if not led_is_stabilizing and not auto_locked:

        for led_id in range(

            NUM_LED

        ):

            pwm = pending_led_commands.get(

                led_id,

                0

            )


            set_led_if_changed(

                led_id,

                pwm

            )


    # ========================================================
    # BACKGROUND ADAPTATION
    # ========================================================
    #
    # Chỉ cập nhật nền khi:
    #
    # - không có object
    # - LED không đang thay đổi
    # - không có thay đổi ánh sáng toàn cục
    #
    # ========================================================

    if (

        not detected_any

        and

        not led_is_stabilizing

        and

        not global_light_event

    ):

        with background_lock:

            if background_feature is not None:

                cv2.accumulateWeighted(

                    current_feature,

                    background_feature,

                    BACKGROUND_ALPHA

                )


                background_edges = make_edges(

                    cv2.convertScaleAbs(

                        background_feature

                    )

                )


    # ========================================================
    # UPDATE TEMPORAL REFERENCE
    # ========================================================

    previous_feature = (

        current_feature.copy()

    )


    previous_gray = (

        gray.copy()

    )


    # ========================================================
    # GLOBAL STATUS
    # ========================================================

    if global_light_event:

        status_text = "LIGHT CHANGE"

        status_color = (

            0,
            255,
            255

        )

    elif led_is_stabilizing:

        status_text = "LIGHT STABILIZING"

        status_color = (

            0,
            255,
            255

        )

    elif detected_any:

        status_text = "OBJECT"

        status_color = (

            0,
            0,
            255

        )

    else:

        status_text = "NORMAL"

        status_color = (

            0,
            255,
            0

        )


    display_status_text = status_text


    return work


# ============================================================
# MANUAL MODE
# ============================================================

def draw_manual_frame(
    frame
):

    global latest_roi_data

    global display_status_text


    work = resize_for_detection(

        frame

    )


    new_roi_data = []


    for r in ROIS:

        x = r["x"]

        y = r["y"]

        w = r["w"]

        h = r["h"]


        x1 = max(

            0,

            x

        )


        y1 = max(

            0,

            y

        )


        x2 = min(

            DETECTION_WIDTH,

            x + w

        )


        y2 = min(

            DETECTION_HEIGHT,

            y + h

        )


        color = (

            255,
            255,
            0

        )


        cv2.rectangle(

            work,

            (x1, y1),

            (x2, y2),

            color,

            2

        )


        cv2.putText(

            work,

            r["name"],

            (
                x1,
                max(
                    18,
                    y1 - 5
                )
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.55,

            color,

            2,

            cv2.LINE_AA

        )


        new_roi_data.append({

            "name":
                r["name"],

            "led":
                r["led"],

            "detected":
                False,

            "pixels":
                0,

            "motion_ratio":
                0.0,

            "background_pixels":
                0

        })


    with roi_lock:

        latest_roi_data = new_roi_data


    display_status_text = "MANUAL MODE"


    return work


# ============================================================
# OFFLINE FRAME
# ============================================================

def create_offline_frame():

    frame = np.zeros(

        (
            DETECTION_HEIGHT,
            DETECTION_WIDTH,
            3
        ),

        dtype=np.uint8

    )


    cv2.putText(

        frame,

        "ESP32 OFFLINE",

        (
            165,
            225
        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        1.2,

        (
            0,
            0,
            255
        ),

        3,

        cv2.LINE_AA

    )


    cv2.putText(

        frame,

        ESP_IP,

        (
            245,
            265
        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.7,

        (
            255,
            255,
            255
        ),

        2,

        cv2.LINE_AA

    )


    return frame


# ============================================================
# RESTORE DISPLAY RESOLUTION
# ============================================================

def restore_display_resolution(

    annotated,

    original

):

    h, w = original.shape[:2]


    if (

        w == DETECTION_WIDTH

        and

        h == DETECTION_HEIGHT

    ):

        return annotated


    return cv2.resize(

        annotated,

        (
            w,
            h
        ),

        interpolation=cv2.INTER_LINEAR

    )


# ============================================================
# CAMERA WORKER
# ============================================================

def camera_worker():

    global latest_frame

    global latest_jpeg


    print(

        "Camera worker started"

    )


    # --------------------------------------------------------
    # Build initial background
    # --------------------------------------------------------

    success = build_background()


    if not success:

        print(

            "Initial background failed"

        )


    while True:

        # ====================================================
        # CAPTURE
        # ====================================================

        frame = capture()


        if frame is None:

            offline = create_offline_frame()


            ret, jpeg = cv2.imencode(

                ".jpg",

                offline,

                [

                    cv2.IMWRITE_JPEG_QUALITY,

                    85

                ]

            )


            if ret:

                with frame_lock:

                    latest_frame = (

                        offline.copy()

                    )

                    latest_jpeg = (

                        jpeg.tobytes()

                    )


            reset_status()


            time.sleep(

                1

            )


            continue


        # ====================================================
        # MODE
        # ====================================================

        with mode_lock:

            current_mode = MODE


        # ====================================================
        # PROCESS
        # ====================================================

        if current_mode == "AUTO":

            annotated = process_frame(

                frame

            )

        else:

            annotated = draw_manual_frame(

                frame

            )


        # ====================================================
        # RETURN ORIGINAL CAMERA SIZE
        # ====================================================

        output_frame = restore_display_resolution(

            annotated,

            frame

        )


        # ====================================================
        # JPEG
        # ====================================================

        ret, jpeg = cv2.imencode(

            ".jpg",

            output_frame,

            [

                cv2.IMWRITE_JPEG_QUALITY,

                85

            ]

        )


        if ret:

            jpeg_bytes = jpeg.tobytes()


            with frame_lock:

                latest_frame = (

                    output_frame.copy()

                )

                latest_jpeg = (

                    jpeg_bytes

                )


        time.sleep(

            CAMERA_INTERVAL

        )


# ============================================================
# VIDEO STREAM
# ============================================================

def video_stream():

    while True:

        with frame_lock:

            jpeg = latest_jpeg


        if jpeg is not None:

            yield (

                b"--frame\r\n"

                b"Content-Type: image/jpeg\r\n"

                b"Content-Length: "

                +

                str(

                    len(jpeg)

                ).encode()

                +

                b"\r\n\r\n"

                +

                jpeg

                +

                b"\r\n"

            )


        time.sleep(

            STREAM_INTERVAL

        )


# ============================================================
# HOME
# ============================================================

@app.route("/")

def index():

    return render_template(

        "index.html"

    )


# ============================================================
# VIDEO
# ============================================================

@app.route("/video")

def video_feed():

    return Response(

        video_stream(),

        mimetype=
        "multipart/x-mixed-replace; boundary=frame"

    )


# ============================================================
# READ MODE
# ============================================================

def read_mode():

    data = request.get_json(

        silent=True

    )


    if (

        data

        and

        "mode" in data

    ):

        return str(

            data["mode"]

        ).upper()


    return request.form.get(

        "mode",

        "AUTO"

    ).upper()


# ============================================================
# MODE
# ============================================================

@app.route(

    "/mode",

    methods=["POST"]

)

def change_mode():

    global MODE


    new_mode = read_mode()

    if new_mode not in [

        "AUTO",

        "MANUAL"

    ]:

        return jsonify({

            "ok":
                False,

            "error":
                "Invalid mode"

        }), 400


    disable_lighting_mode()


    with mode_lock:

        MODE = new_mode


    # ========================================================
    # AUTO
    # ========================================================

    if new_mode == "AUTO":

        print(

            "Switching to AUTO..."

        )


        reset_status()


        success = build_background()


        if not success:

            return jsonify({

                "ok":
                    False,

                "error":
                    "Background build failed"

            }), 500


        print(

            "MODE: AUTO"

        )


    # ========================================================
    # MANUAL
    # ========================================================

    else:

        print(

            "MODE: MANUAL"

        )


    return jsonify({

        "ok":
            True,

        "mode":
            new_mode

    })


# ============================================================
# LIGHTING MODE API
# ============================================================

@app.route(

    "/lighting_mode",

    methods=["POST"]

)

def lighting_mode():

    data = request.get_json(

        silent=True

    ) or {}


    if not data.get(
        "enabled",
        True
    ):

        disable_lighting_mode()

        return jsonify({

            "ok": True,

            "enabled": False,

            "mode": None

        })


    mode_name = str(

        data.get(
            "mode",
            ""
        )

    ).lower()


    if mode_name not in LIGHTING_MODES:

        return jsonify({

            "ok": False,

            "error": "Invalid lighting mode"

        }), 400


    if not apply_lighting_mode(
        mode_name
    ):

        return jsonify({

            "ok": False,

            "error": "ESP32 error while applying lighting mode"

        }), 502


    mode = LIGHTING_MODES[mode_name]


    return jsonify({

        "ok": True,

        "enabled": True,

        "mode": mode_name,

        "name": mode["name"],

        "auto_lock": mode["auto_lock"],

        "pwms": mode["pwms"]

    })


# ============================================================
# MANUAL ALL LED
# ============================================================

@app.route(

    "/manual",

    methods=["POST"]

)

def manual():

    global MODE

    disable_lighting_mode()


    with mode_lock:

        MODE = "MANUAL"


    data = request.get_json(

        silent=True

    )


    for i in range(

        NUM_LED

    ):

        key = f"led{i}"


        if data is not None:

            value = data.get(

                key,

                0

            )

        else:

            value = request.form.get(

                key,

                0

            )


        try:

            pwm = int(

                value

            )

        except Exception:

            pwm = 0


        pwm = max(

            0,

            min(
                255,
                pwm
            )

        )


        set_led(

            i,

            pwm

        )


    print(

        "Manual PWM updated"

    )


    return jsonify({

        "ok":
            True,

        "mode":
            "MANUAL",

        "led_pwm":
            list(last_led_pwm)

    })


# ============================================================
# MANUAL SINGLE LED
# ============================================================

@app.route(

    "/manual_led",

    methods=["POST"]

)

def manual_led():

    global MODE

    disable_lighting_mode()


    with mode_lock:

        MODE = "MANUAL"


    data = request.get_json(

        silent=True

    )


    try:

        if data is not None:

            led_id = int(

                data.get(

                    "id",

                    0

                )

            )

            pwm = int(

                data.get(

                    "pwm",

                    0

                )

            )

        else:

            led_id = int(

                request.form.get(

                    "id",

                    0

                )

            )

            pwm = int(

                request.form.get(

                    "pwm",

                    0

                )

            )


    except Exception:

        return jsonify({

            "ok":
                False,

            "error":
                "Invalid parameters"

        }), 400


    if (

        led_id < 0

        or

        led_id >= NUM_LED

    ):

        return jsonify({

            "ok":
                False,

            "error":
                "Invalid LED"

        }), 400


    pwm = max(

        0,

        min(
            255,
            pwm
        )

    )


    success = set_led(

        led_id,

        pwm

    )


    if not success:

        return jsonify({

            "ok":
                False,

            "error":
                "ESP32 error"

        }), 500


    return jsonify({

        "ok":
            True,

        "mode":
            "MANUAL",

        "id":
            led_id,

        "pwm":
            pwm

    })


# ============================================================
# RESET BACKGROUND
# ============================================================

@app.route(

    "/reset",

    methods=["GET", "POST"]

)

def reset():

    disable_lighting_mode()

    print()

    print(

        "=========================================="

    )

    print(

        "RESET BACKGROUND"

    )

    print(

        "=========================================="

    )


    reset_status()


    success = build_background()


    if success:

        return jsonify({

            "ok":
                True,

            "message":
                "Background rebuilt",

            "frames":
                BACKGROUND_FRAMES

        })


    return jsonify({

        "ok":
            False,

        "message":
            "Background build failed"

    }), 500


# ============================================================
# ALL OFF
# ============================================================

@app.route(

    "/all_off",

    methods=["POST"]

)

def all_off():

    disable_lighting_mode()

    turn_off_all()


    reset_status()


    return jsonify({

        "ok":
            True,

        "led_pwm":
            list(last_led_pwm)

    })


# ============================================================
# STATUS
# ============================================================

@app.route("/status")

def status():

    with mode_lock:

        current_mode = MODE


    with led_lock:

        pwm_values = list(

            last_led_pwm

        )


    with status_lock:

        online = bool(

            esp32_online

        )


    with roi_lock:

        roi_values = list(

            latest_roi_data

        )


    with lighting_mode_lock:

        mode_enabled = lighting_mode_enabled

        mode_name = lighting_mode_name


    return jsonify({

        "ok":
            True,

        "mode":
            current_mode,

        "rois":
            roi_values,

        "roi":
            roi_values,

        "led_pwm":
            pwm_values,

        "esp32_online":
            online,

        "esp_online":
            online,

        "esp_ip":
            ESP_IP,

        "background_ready":
            background_feature is not None,

        "background_building":
            background_building,

        "background_progress":
            background_progress,

        "background_status":
            background_status_text,

        "display_status":
            display_status_text,

        "lighting_mode_enabled":
            mode_enabled,

        "lighting_mode":
            mode_name

    })


# ============================================================
# TEST ESP32
# ============================================================

@app.route("/test_esp")

def test_esp():

    try:

        response = session.get(

            CAPTURE_URL,

            timeout=5

        )


        if response.status_code == 200:

            with status_lock:

                globals()[

                    "esp32_online"

                ] = True


            return jsonify({

                "ok":
                    True,

                "status_code":
                    response.status_code,

                "bytes":
                    len(

                        response.content

                    )

            })


        with status_lock:

            globals()[

                "esp32_online"

            ] = False


        return jsonify({

            "ok":
                False,

            "status_code":
                response.status_code

        }), 500


    except Exception as e:

        with status_lock:

            globals()[

                "esp32_online"

            ] = False


        return jsonify({

            "ok":
                False,

            "error":
                str(e)

        }), 500


# ============================================================
# GET PC LOCAL IP
# ============================================================

def get_local_ip():

    sock = None


    try:

        sock = socket.socket(

            socket.AF_INET,

            socket.SOCK_DGRAM

        )


        sock.connect(

            (
                "8.8.8.8",
                80
            )

        )


        ip = sock.getsockname()[0]


        return ip


    except Exception:

        return "127.0.0.1"


    finally:

        if sock is not None:

            sock.close()


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    print()

    print(

        "=========================================="

    )

    print(

        "ESP32 SMART LIGHT CONTROLLER"

    )

    print(

        "MOTION-FIRST DETECTION"

    )

    print(

        "=========================================="

    )

    print()

    print(

        f"ESP32 IP: {ESP_IP}"

    )

    print()


    # --------------------------------------------------------
    # Camera worker
    # --------------------------------------------------------

    worker = threading.Thread(

        target=camera_worker,

        daemon=True

    )


    worker.start()


    # --------------------------------------------------------
    # PC IP
    # --------------------------------------------------------

    local_ip = get_local_ip()


    print(

        "WEB CONTROL:"

    )


    print(

        f"http://{local_ip}:5000"

    )


    print()


    print(

        "Local:"

    )


    print(

        "http://127.0.0.1:5000"

    )


    print()


    print(

        "ESP32 camera:"

    )


    print(

        f"http://{ESP_IP}/capture"

    )


    print()


    print(

        "=========================================="

    )


    # --------------------------------------------------------
    # Flask
    # --------------------------------------------------------

    app.run(

        host="0.0.0.0",

        port=5000,

        threaded=True,

        debug=False,

        use_reloader=False

    )
