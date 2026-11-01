# Standalone OV5640 + ContourGate. Deploy this file as E:/main.py.
# Relative camera coordinates by default. No servo output or host PC required.
import sensor
import time
import math
import gc
try:
    import ujson as json
except ImportError:
    import json
from contour_gate_runtime import TemporalDepthFilter, MODEL_ID

# Nominal lens HFOV from WT manual p3; NOT a calibrated camera matrix.
HFOV_DEG = 70.8
OBJECT_HEIGHT_M = 0.0  # Keep 0 for relative coordinates until calibrated.
DIFFERENCE_THRESHOLD = 24
TELEMETRY_INTERVAL_MS = 100
MAX_LOST = 8
WIDTH, HEIGHT = 320, 240

sensor.reset()
sensor.set_pixformat(sensor.GRAYSCALE)
sensor.set_framesize(sensor.QVGA)
sensor.skip_frames(time=1500)
sensor.set_auto_gain(False)
sensor.set_auto_whitebal(False)
try:
    sensor.set_auto_exposure(False)
except Exception:
    pass
background = sensor.alloc_extra_fb(WIDTH, HEIGHT, sensor.GRAYSCALE)
work = sensor.alloc_extra_fb(WIDTH, HEIGHT, sensor.GRAYSCALE)
previous = sensor.alloc_extra_fb(WIDTH, HEIGHT, sensor.GRAYSCALE)
background.replace(sensor.snapshot())
previous.replace(background)
background_mean = background.get_statistics().mean()
focal = WIDTH / (2 * math.tan(HFOV_DEG * math.pi / 360))
filt = TemporalDepthFilter()
clock = time.clock()
last_ticks = time.ticks_ms()
last_print = last_ticks
current = None
lost = 0
track_id = 0
reference_height = None
vx, vy = 0.0, 0.0
frame_number = 0
idle_frames = 0
gc.collect()
print('CG3D_READY ' + json.dumps({'model': MODEL_ID, 'parameters': 161,
    'sensor_id': sensor.get_id(), 'expected_ov5640_id': sensor.OV5640,
    'width': WIDTH, 'height': HEIGHT, 'unit': 'm' if OBJECT_HEIGHT_M > 0 else 'relative',
    'heap_free': gc.mem_free(), 'hfov_deg_nominal': HFOV_DEG}))

while True:
    clock.tick()
    img = sensor.snapshot()
    started_us = time.ticks_us()
    now = time.ticks_ms()
    dt = max(.01, min(.3, time.ticks_diff(now, last_ticks) / 1000.0))
    last_ticks = now
    frame_number += 1
    status = 'SEARCH'
    xyz = None
    raw_depth = None
    gate = None
    quality = None
    box = None
    inference_us = 0

    # Independent whole-frame motion guard. Check texture patches away from
    # the central target; invalidate trajectories when background motion agrees.
    moved = False
    if frame_number % 5 == 0:
        shifts = []
        for roi in ((8, 8, 32, 32), (280, 8, 32, 32), (8, 200, 32, 32)):
            displacement = img.find_displacement(previous, roi=roi, template_roi=roi)
            if displacement.response() > .15:
                shifts.append((displacement.x_translation(), displacement.y_translation()))
        for i in range(len(shifts)):
            for j in range(i + 1, len(shifts)):
                dx, dy = shifts[i]
                ex, ey = shifts[j]
                if dx * dx + dy * dy > 9 and (dx-ex)**2 + (dy-ey)**2 < 4:
                    moved = True
    previous.replace(img)
    illumination_jump = abs(img.get_statistics().mean() - background_mean) > 30
    if moved or illumination_jump:
        background.replace(img)
        background_mean = img.get_statistics().mean()
        current = None
        reference_height = None
        lost = 0
        vx, vy = 0.0, 0.0
        filt.reset()
        status = 'CAMERA_MOVED' if moved else 'LIGHT_RESET'
    else:
        work.replace(img)
        work.difference(background)
        work.binary([(DIFFERENCE_THRESHOLD, 255)])
        work.erode(1)
        work.dilate(1)
        candidates = []
        for b in work.find_blobs([(1, 255)], pixels_threshold=115,
                                  area_threshold=115, merge=True, margin=3):
            if min(b.w(), b.h()) >= 4 and b.pixels() <= WIDTH * HEIGHT * .45:
                candidates.append(b)
        chosen = None
        changed = False
        if current is not None and lost < MAX_LOST:
            px, py, pw, ph, pa = current
            expected_x = px + vx * dt * (lost + 1)
            expected_y = py + vy * dt * (lost + 1)
            best_score = 1e12
            for b in candidates:
                distance = math.sqrt((b.cx()-expected_x)**2 + (b.cy()-expected_y)**2)
                ratio = b.pixels() / max(1, pa)
                if distance < max(WIDTH*.12, pw, ph) + lost*4 and .2 < ratio < 5:
                    score = distance + abs(math.log(ratio))*12
                    if score < best_score:
                        chosen = b
                        best_score = score
        if chosen is None and (current is None or lost >= MAX_LOST):
            central = [b for b in candidates if abs(b.cx()/WIDTH-.5)<.4 and abs(b.cy()/HEIGHT-.5)<.4]
            for b in (central or candidates):
                if chosen is None or b.pixels() > chosen.pixels():
                    chosen = b
            if chosen is not None:
                changed = True
                track_id += 1
                vx, vy = 0.0, 0.0
                reference_height = chosen.h()
                filt.reset()
        if chosen is not None:
            b = chosen
            if current is not None and not changed:
                vx = .65*vx + .35*(b.cx()-current[0])/dt
                vy = .65*vy + .35*(b.cy()-current[1])/dt
            current = (b.cx(), b.cy(), b.w(), b.h(), b.pixels())
            lost = 0
            idle_frames = 0
            box = b.rect()
            clipped = b.x() <= 1 or b.y() <= 1 or b.x()+b.w() >= WIDTH-1 or b.y()+b.h() >= HEIGHT-1
            quality = max(.05, min(1.0, b.solidity()*min(1.0, b.density()/.5)))
            if clipped:
                quality *= .25
                filt.reset()
                status = 'CLIPPED'
            else:
                raw_depth = focal*OBJECT_HEIGHT_M/b.h() if OBJECT_HEIGHT_M > 0 else reference_height/b.h()
                tick = time.ticks_us()
                depth, gate = filt.update(raw_depth, dt, quality, b.h())
                inference_us = time.ticks_diff(time.ticks_us(), tick)
                xyz = ((b.cx()-WIDTH*.5)/focal*depth, -(b.cy()-HEIGHT*.5)/focal*depth, depth)
                status = 'TRACKING'
            # Draw only after all detection/background operations have finished.
            img.draw_rectangle(box, color=255)
            img.draw_cross(b.cx(), b.cy(), color=255)
        else:
            lost += 1
            idle_frames += 1
            if lost == 1:
                filt.reset()
            if lost > MAX_LOST:
                current = None
            if not candidates and idle_frames >= 30:
                background.replace(img)
                background_mean = img.get_statistics().mean()
                idle_frames = 0
    processing_us = time.ticks_diff(time.ticks_us(), started_us)
    if time.ticks_diff(now, last_print) >= TELEMETRY_INTERVAL_MS:
        last_print = now
        print('CG3D ' + json.dumps({'frame': frame_number, 'ticks_ms': now,
            'status': status, 'track_id': track_id, 'bbox': box, 'xyz': xyz,
            'raw_depth': raw_depth, 'gate': gate, 'quality': quality,
            'unit': 'm' if OBJECT_HEIGHT_M > 0 else 'relative',
            'fps': clock.fps(), 'processing_us': processing_us,
            'inference_us': inference_us, 'heap_free': gc.mem_free()}))
    if frame_number % 100 == 0:
        gc.collect()
