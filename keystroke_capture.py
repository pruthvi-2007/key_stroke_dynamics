from pynput import keyboard
import time
import csv
import os

PHRASE = "the quick brown fox jumps over the lazy dog"
VALID_USERS = {"User1", "User2", "User3", "User4"}

user_id = input(
    "Enter your User ID (User1/User2/User3/User4): "
).strip()

if user_id not in VALID_USERS:
    print("Invalid User ID.")
    raise SystemExit(1)

# A new file, separate from your old per-key CSV files
filename = "keystroke_attempts.csv"
file_exists = os.path.exists(filename) and os.path.getsize(filename) > 0

# Continue attempt numbering when restarting the program
attempt_id = 0

if file_exists:
    with open(filename, "r", newline="", encoding="utf-8") as existing_file:
        reader = csv.DictReader(existing_file)

        if reader.fieldnames and "attempt_id" in reader.fieldnames:
            for row in reader:
                try:
                    attempt_id = max(attempt_id, int(row["attempt_id"]))
                except (ValueError, TypeError):
                    pass

# Stop if an old file has an incompatible structure
expected_columns = [
    "user_id", "attempt_id", "key",
    "press_time", "release_time",
    "dwell_time_ms", "flight_time_ms"
]

if file_exists:
    with open(filename, "r", newline="", encoding="utf-8") as check_file:
        header = next(csv.reader(check_file), [])

    if header != expected_columns:
        print("CSV format mismatch.")
        print("Rename or remove the old keystroke_attempts.csv file.")
        raise SystemExit(1)

file = open(filename, "a", newline="", encoding="utf-8")
writer = csv.writer(file)

if not file_exists:
    writer.writerow(expected_columns)
    file.flush()

# Current attempt data
typed_text = ""
press_times = {}
attempt_events = []
last_release_time = None
complete_pending = False


def key_name_of(key):
    try:
        return key.char if key.char is not None else str(key)
    except AttributeError:
        return str(key)


def update_typed_text(key):
    """Update the phrase as the user types or corrects mistakes."""
    global typed_text

    if key == keyboard.Key.backspace:
        typed_text = typed_text[:-1]

    elif key == keyboard.Key.space:
        typed_text += " "

    else:
        try:
            char = key.char
            if char is not None and char.isprintable():
                typed_text += char.lower()
        except AttributeError:
            pass


def on_press(key):
    global complete_pending, typed_text

    current_time = time.perf_counter()

    if key == keyboard.Key.esc:
        print("\nESC pressed. Stopping recording...")
        return False

    # Enter discards an incomplete attempt
    if key == keyboard.Key.enter:
        typed_text = ""
        attempt_events.clear()
        complete_pending = False
        print("Incomplete attempt discarded. Start again.")
        return

    # Ignore keyboard auto-repeat
    if key in press_times:
        return

    if last_release_time is not None:
        flight_time = (
            current_time - last_release_time
        ) * 1000
    else:
        flight_time = None

    press_times[key] = {
        "press_time": current_time,
        "flight_time": flight_time
    }

    update_typed_text(key)

    # The full phrase, not just the final letter, is the boundary
    if typed_text == PHRASE:
        complete_pending = True
        print("\nFull phrase detected! Finishing this attempt...")


def on_release(key):
    global last_release_time
    global attempt_id, typed_text, complete_pending

    release_time = time.perf_counter()
    key_name = key_name_of(key)

    if key in press_times:
        data = press_times.pop(key)

        press_time = data["press_time"]
        flight_time = data["flight_time"]
        dwell_time = (release_time - press_time) * 1000

        # Keep the timings in memory until the whole phrase is complete
        attempt_events.append([
            key_name,
            round(press_time, 6),
            round(release_time, 6),
            round(dwell_time, 2),
            round(flight_time, 2)
            if flight_time is not None else ""
        ])

        print(
            f"Released: {key_name} | "
            f"Dwell: {dwell_time:.2f} ms"
        )

    last_release_time = release_time

    # Save only after the final character has been released
    if complete_pending and key_name.lower() == "g":
        attempt_id += 1

        for event in attempt_events:
            writer.writerow([
                user_id,
                attempt_id,
                *event
            ])

        file.flush()

        print(
            f"\nAttempt {attempt_id} saved successfully!"
        )
        print(f"Phrase: {PHRASE}")
        print("Type the phrase again for another attempt.\n")

        # Reset for the next attempt
        typed_text = ""
        attempt_events.clear()
        complete_pending = False


print("\nKESTROKE DYNAMICS DATA COLLECTION")
print("--------------------------------")
print(f"User: {user_id}")
print(f"Required phrase: {PHRASE}")
print("Type the exact phrase to save an attempt.")
print("Press ENTER to discard an incomplete attempt.")
print("Press ESC to stop recording.")
print(f"CSV file: {os.path.abspath(filename)}\n")

try:
    with keyboard.Listener(
        on_press=on_press,
        on_release=on_release
    ) as listener:
        listener.join()

except KeyboardInterrupt:
    print("\nRecording interrupted.")

except Exception as error:
    print(f"\nKeyboard listener error: {error}")

finally:
    file.close()
    print(f"\nData saved in {filename}")