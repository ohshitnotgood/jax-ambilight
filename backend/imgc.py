import cv2
import dxcam
import torch

print(dxcam.device_info())
camera = dxcam.create()

# Start capturing frames
camera.start()

while True:
    # Get the latest captured frame from the camera
    frame = camera.get_latest_frame()
    g = torch.tensor(frame)
    print(frame.shape)
    print(g.shape)

# Stop capturing when done
camera.stop()

# Close OpenCV windows
cv2.destroyAllWindows()