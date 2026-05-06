📖 Project Description
This project features a high-performance, multi-task visual neural network model designed for resource-constrained environments. It integrates Video Matting and Arm Motion Prediction into a single architecture, optimized for ultra-lightweight deployment on embedded hardware to achieve precise target identification.

Key Technical Highlights
Dual-Task Integration: The model simultaneously handles pixel-level semantic analysis for video matting and temporal trajectory forecasting for arm movements, providing a versatile solution for complex visual tasks.

Optimized for ESP32: Specifically re-engineered for microcontrollers like the ESP32-S3. Through operator fusion, depthwise separable convolutions, and model pruning, the parameter count and computational overhead have been minimized.

Edge-Ready Deployment: Fully compatible with deployment frameworks such as ESP-DL, TFLite Micro, or ONNX. It supports INT8 quantization, achieving millisecond-level inference latency while maintaining high recognition accuracy.

Robust Target Identification: By incorporating a lightweight attention mechanism, the model effectively suppresses background noise, ensuring stable tracking and precise segmentation of targets even in dynamic or cluttered environments.

Application Scenarios
Real-time Video Processing: Enabling high-quality background removal and matting on low-power mobile or IoT devices.

Human-Machine Interaction (HMI): Powering ESP32-based wearables or small-scale robotics to recognize and predict arm gestures for intuitive control.

Smart Surveillance & Industrial AI: Providing precise motion recognition and target monitoring in scenarios where high-performance computing is unavailable.

Technical Evolution
Architecture: Based on an Encoder-Decoder structure with a lightweight backbone to maximize feature extraction speed.

Hardware Acceleration: Tailored to leverage the hardware acceleration instructions (such as the PIE set) of the ESP32 series to ensure smooth real-time FPS performance.
