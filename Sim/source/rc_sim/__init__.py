"""LituanicaX car simulator: test driving algorithms on simulated detections, IMU and encoder.

Module map
    interface.py        the car <-> algorithm contract (pure Python; also valid on the Pi)
    algorithms/         driving algorithms (pure Python + numpy)
    config.py           every tunable number
    track.py            track.json: objects, centre line, lap progress
    detector_model.py   synthetic YOLO (occlusion / size / truncation / blur / jitter / FP / NMS / latency)
    imu_model.py        BNO055/BNO08x model        encoder_model.py   AS5600 on the motor shaft
    car.py, sim.py      Isaac Sim scene, vehicle dynamics, multi-rate sensor loop  (need Isaac Sim)
    viewer.py, run.py   live view and the entry point                               (need Isaac Sim)
"""
