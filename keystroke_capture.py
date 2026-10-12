
from pynput import keyboard
import time
import csv
import os

user_id = input(
    "Enter your User ID (User1/User2/User3/User4): "
).strip()

if user_id not in {"User1", "User2", "User3", "User4"}:
    print("Invalid User ID.")
    raise SystemExit(1)

filename = f"{user_id}.csv"
file_exists = os.path.exists(filename) and os.path.getsize(filename) > 0

press_times = {}
press_flight_times = {}
last_release_time = None

file = open(filename, "a", newline="", encoding="utf-8")
writer = csv.writer(file)

if not file_exists:
    writer.writerow([
        "key", "press_time", "release_time",
        "dwell_time_ms", "flight_time_ms"
    ])
    file.flush()


def key_name_of(key):
    try:
        return key.char
    except AttributeError:
        return str(key)


def on_press(key):
    global last_release_time

    if key == keyboard.Key.esc:
        print("\nESC pressed. Stopping recording...")
        return False

    current_time = time.perf_counter()

    if key not in press_times:
        press_times[key] = current_time

        if last_release_time is None:
            press_flight_times[key] = None
        else:
            press_flight_times[key] = (
                current_time - last_release_time
            ) * 1000

    flight = press_flight_times.get(key)

    print(
        f"Pressed: {key_name_of(key)} | "
        f"Flight: {flight if flight is not None else 0:.2f} ms"
    )


def on_release(key):
    global last_release_time

    release_time = time.perf_counter()

    if key in press_times:
        press_time = press_times.pop(key)
        flight_time = press_flight_times.pop(key, None)

        dwell_time = (release_time - press_time) * 1000

        writer.writerow([
            key_name_of(key),
            round(press_time, 6),
            round(release_time, 6),
            round(dwell_time, 2),
            round(flight_time, 2)
            if flight_time is not None else ""
        ])

        file.flush()

        print(
            f"Released: {key_name_of(key)} | "
            f"Dwell: {dwell_time:.2f} ms"
        )

    last_release_time = release_time


print("\nStart typing!")
print("Press ESC to stop recording.")
print(f"Saving data to: {os.path.abspath(filename)}")

try:
    with keyboard.Listener(
        on_press=on_press,
        on_release=on_release
    ) as listener:
        listener.join()

except KeyboardInterrupt:
    print("\nRecording interrupted by Ctrl+C.")

except Exception as error:
    print(f"\nKeyboard listener error: {error}")

finally:
    file.close()
    print(f"\nData saved in {filename}")
