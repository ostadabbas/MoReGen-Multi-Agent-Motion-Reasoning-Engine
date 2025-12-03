import win32gui
import win32ui
import win32con
import win32api
import cv2
import numpy as np
import win32process
import psutil
import argparse
import time


def list_open_windows(wdnw_list):
    def callback(hwnd, extra):
        if win32gui.IsWindowVisible(hwnd):
            title = win32gui.GetWindowText(hwnd)
            _, pid = win32process.GetWindowThreadProcessId(hwnd)
            # Optional: Get process name using psutil
            proc = psutil.Process(pid)

            if title:  # Filter out empty titles
                wdnw_list.append(f"{hex(hwnd)}: {title}")
                print(f"Process name: {proc.name()}, PID: {pid}, {hex(hwnd)}: {title}")
    win32gui.EnumWindows(callback, None)
    return wdnw_list

# Get PID for the given window title
def get_pid_by_title(title):
    def callback(hwnd, result):
        if win32gui.IsWindowVisible(hwnd):
            if win32gui.GetWindowText(hwnd) == title:
                _, pid = win32process.GetWindowThreadProcessId(hwnd)
                result.append(pid)
    result = []
    win32gui.EnumWindows(callback, result)
    return result[0] if result else None

def get_hwnd_by_pid(pid):
    def callback(hwnd, result):
        if win32gui.IsWindowVisible(hwnd):
            _, win_pid = win32process.GetWindowThreadProcessId(hwnd)
            if win_pid == pid:
                result.append(hwnd)
    result = []
    win32gui.EnumWindows(callback, result)
    return result[0] if result else None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record a specific window.")
    parser.add_argument("--pid", type=int, required=True, help="PID of the window to record.")
    parser.add_argument("--output", type=str, default="window_capture.mp4", help="Output video file name.")
    parser.add_argument("--duration", type=int, default=10, help="Duration to record in seconds.")

    args = parser.parse_args()

    OUTPUT_VIDEO = args.output
    RECORD_SECONDS = args.duration
    # Get current monitor's refresh rate (FPS)
    def get_monitor_fps():
        # Get the primary monitor's device context
        hdc = win32gui.GetDC(0)
        fps = win32ui.GetDeviceCaps(hdc, win32con.VREFRESH)
        win32gui.ReleaseDC(0, hdc)
        return fps if fps > 1 else 30  # fallback to 30 if not available

    # FPS = get_monitor_fps()
    FPS = 120
    print(f"Detected monitor refresh rate: {FPS} FPS")

    # (Re-run the above code with updated parameters)
    # Locate window, setup video writer, and capture loop as above...
    hwnd = False
    start_time = time.time()
    while not hwnd and (time.time() - start_time < 2):
        hwnd = get_hwnd_by_pid(args.pid)
    if not hwnd:
        raise Exception(f"Window '{args.pid}' not found.")
    left, top, right, bottom = win32gui.GetWindowRect(hwnd)
    width = right - left
    height = bottom - top
    # --- Setup Video Writer ---
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, FPS, (width, height))

    # --- Capture Loop ---
    for _ in range(RECORD_SECONDS * FPS):
        hwindc = win32gui.GetWindowDC(hwnd)
        srcdc = win32ui.CreateDCFromHandle(hwindc)
        memdc = srcdc.CreateCompatibleDC()

        bmp = win32ui.CreateBitmap()
        bmp.CreateCompatibleBitmap(srcdc, width, height)
        memdc.SelectObject(bmp)

        memdc.BitBlt((0, 0), (width, height), srcdc, (0, 0), win32con.SRCCOPY)

        bmpinfo = bmp.GetInfo()
        bmpstr = bmp.GetBitmapBits(True)
        img = np.frombuffer(bmpstr, dtype='uint8').reshape((height, width, 4))
        frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        out.write(frame)

        # Cleanup
        win32gui.DeleteObject(bmp.GetHandle())
        memdc.DeleteDC()
        srcdc.DeleteDC()
        win32gui.ReleaseDC(hwnd, hwindc)

    out.release()
    print(f">*&SUCCESS>*&Recording saved to {OUTPUT_VIDEO}")