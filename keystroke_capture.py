from pynput import keyboard
import time
import csv

file = open("keystroke_data.csv", "w", newline="")
writer = csv.writer(file)

# CSV header
writer.writerow([
    "key",
    "press_time",
    "release_time",
    "dwell_time_ms",
    "flight_time_ms"
])

press_times = {}
last_release_time = None


def on_press(key):
    global last_release_time

    current_time = time.perf_counter()

    try:
        key_name = key.char
    except AttributeError:
        key_name = str(key)

    # Calculate flight time
    if last_release_time is not None:
        flight_time = (current_time - last_release_time) * 1000
    else:
        flight_time = None

    press_times[key] = current_time

    print(
        f"Pressed: {key_name} | "
        f"Flight: {flight_time if flight_time else 0:.2f} ms"
    )


def on_release(key):
    global last_release_time

    release_time = time.perf_counter()

    try:
        key_name = key.char
    except AttributeError:
        key_name = str(key)

    if key in press_times:

        dwell_time = (
            release_time - press_times[key]
        ) * 1000

        flight_time = None

        if last_release_time is not None:
            # Flight time was already calculated during press
            flight_time = (
                press_times[key] - last_release_time
            ) * 1000

        print(
            f"Released: {key_name} | "
            f"Dwell: {dwell_time:.2f} ms"
        )

        writer.writerow([
            key_name,
            press_times[key],
            release_time,
            round(dwell_time, 2),
            round(flight_time, 2) if flight_time is not None else ""
        ])

        file.flush()

        del press_times[key]

    last_release_time = release_time

    # ESC stops the program
    if key == keyboard.Key.esc:
        file.close()
        return False


print("Start typing...")
print("Press ESC to stop.")

with keyboard.Listener(
    on_press=on_press,
    on_release=on_release
) as listener:
    listener.join()