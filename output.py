import ast
import string
import cv2 as cv
import easyocr
from ultralytics import YOLO

video = cv.VideoCapture(r"C:\Users\bhanu\Downloads\pythonProject7-20241112T155347Z-001\pythonProject7\Automatic Number Plate Recognition (ANPR) _ Vehicle Number Plate Recognition (1).mp4")
model = YOLO(r"C:\Users\bhanu\Downloads\pythonProject7-20241112T155347Z-001\pythonProject7\best.pt")
ret = True
frame_number = -1
results = {}
reader = easyocr.Reader(['en'], gpu=False)

# Mapping dictionaries for character conversion
# characters that can easily be confused can be
# verified by their location - an `O` in a place
# where a number is expected is probably a `0`
dict_char_to_int = {'O': '0',
                    'I': '1',
                    'J': '3',
                    'A': '4',
                    'G': '6',
                    'S': '5'}

dict_int_to_char = {'0': 'O',
                    '1': 'I',
                    '3': 'J',
                    '4': 'A',
                    '6': 'G',
                    '5': 'S'}


def license_complies_format(text):
    # True if the license plate complies with the format, False otherwise.
    if len(text) <= 8:
        return False
    if len(text) == 9:
        if (text[0] in string.ascii_uppercase or text[0] in dict_int_to_char.keys()) and \
                (text[1] in string.ascii_uppercase or text[1] in dict_int_to_char.keys()) and \
                (text[2] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[
                    2] in dict_char_to_int.keys()) and \
                (text[3] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[
                    3] in dict_char_to_int.keys()) and \
                (text[4] in string.ascii_uppercase or text[4] in dict_int_to_char.keys()) and \
                (text[5] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[
                    5] in dict_char_to_int.keys()) and \
                (text[6] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[
                    6] in dict_char_to_int.keys()) and \
                (text[7] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[
                    7] in dict_char_to_int.keys()) and \
                (text[8] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[8] in dict_char_to_int.keys()):
            return True

    elif len(text) == 10:
        if (text[0] in string.ascii_uppercase or text[0] in dict_int_to_char.keys()) and \
                (text[1] in string.ascii_uppercase or text[1] in dict_int_to_char.keys()) and \
                (text[2] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[
                    2] in dict_char_to_int.keys()) and \
                (text[3] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[
                    3] in dict_char_to_int.keys()) and \
                (text[4] in string.ascii_uppercase or text[4] in dict_int_to_char.keys()) and \
                (text[5] in string.ascii_uppercase or text[5] in dict_int_to_char.keys()) and \
                (text[6] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[
                    6] in dict_char_to_int.keys()) and \
                (text[7] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[
                    7] in dict_char_to_int.keys()) and \
                (text[8] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[
                    8] in dict_char_to_int.keys()) and \
                (text[9] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[9] in dict_char_to_int.keys()):
            return True
    else:
        return False



def format_license(text):
    license_plate_ = ''
    if len(text) == 10:
        mapping = {0: dict_int_to_char, 1: dict_int_to_char, 9: dict_char_to_int, 4: dict_int_to_char,
                   5: dict_int_to_char, 6: dict_char_to_int, 7: dict_char_to_int, 8: dict_char_to_int,
                   2: dict_char_to_int, 3: dict_char_to_int}
        for j in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
            if text[j] in mapping[j].keys():
                license_plate_ += mapping[j][text[j]]
            else:
                license_plate_ += text[j]


    else:
        mapping1 = {0: dict_int_to_char, 1: dict_int_to_char, 4: dict_int_to_char, 5: dict_int_to_char,
                    6: dict_char_to_int, 7: dict_char_to_int, 8: dict_char_to_int,
                    2: dict_char_to_int, 3: dict_char_to_int}
        for j in [0, 1, 2, 3, 4, 5, 6, 7, 8]:
            if text[j] in mapping1[j].keys():
                license_plate_ += mapping1[j][text[j]]
            else:
                license_plate_ += text[j]

    return license_plate_


def read_license_plate(license_plate_crop):
    detections = reader.readtext(license_plate_crop)

    for detection in detections:
        bbox, text, score = detection

        text = text.upper().replace(' ', '')

        # verify that text is conform to a standard license plate
        if license_complies_format(text):
            # bring text into the default license plate format
            return format_license(text), score

    return None, None


def write_csv(results, output_path):
    with open(output_path, 'w') as f:
        f.write('{},{},{},{},{},{},{},{}\n'.format(
            'frame_number', 'track_id', 'car_bbox', 'car_bbox_score',
            'license_plate_bbox', 'license_plate_bbox_score', 'license_plate_number',
            'license_text_score'))

        for frame_number in results.keys():
            for track_id in results[frame_number].keys():
                print(results[frame_number][track_id])
                if 'car' in results[frame_number][track_id].keys() and \
                        'license_plate' in results[frame_number][track_id].keys() and \
                        'number' in results[frame_number][track_id]['license_plate'].keys():
                    f.write('{},{},{},{},{},{},{},{}\n'.format(
                        frame_number,
                        track_id,
                        '[{} {} {} {}]'.format(
                            results[frame_number][track_id]['car']['bbox'][0],
                            results[frame_number][track_id]['car']['bbox'][1],
                            results[frame_number][track_id]['car']['bbox'][2],
                            results[frame_number][track_id]['car']['bbox'][3]
                        ),
                        results[frame_number][track_id]['car']['bbox_score'],
                        '[{} {} {} {}]'.format(
                            results[frame_number][track_id]['license_plate']['bbox'][0],
                            results[frame_number][track_id]['license_plate']['bbox'][1],
                            results[frame_number][track_id]['license_plate']['bbox'][2],
                            results[frame_number][track_id]['license_plate']['bbox'][3]
                        ),
                        results[frame_number][track_id]['license_plate']['bbox_score'],
                        results[frame_number][track_id]['license_plate']['number'],
                        results[frame_number][track_id]['license_plate']['text_score'])
                    )
        f.close()


# read the 10 first frames
while ret:
    frame_number += 1
    ret, frame = video.read()

    if ret and frame_number < 500:
        results[frame_number] = {}
        # vehicle detector
        detections = model.predict(frame)[0]

        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, track_id, score = detection
            plates = [[x1, y1, x2, y2, track_id, score]]
            for bbox in plates:
                print(bbox)
                roi = frame[int(y1):int(y2), int(x1):int(x2)]

                plate_gray = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)


                # posterize
                _, plate_treshold = cv.threshold(plate_gray, 64, 255, cv.THRESH_BINARY_INV)

                np_text, np_score = read_license_plate(plate_gray)

                if np_text is not None:
                    results[frame_number][track_id] = {
                        'car': {
                            'bbox': [x1, y1, x2, y2],
                            'bbox_score': score
                        },
                        'license_plate': {
                            'bbox': [x1, y1, x2, y2],
                            'bbox_score': score,
                            'number': np_text,
                            'text_score': np_score
                        }
                    }

write_csv(results, './results.csv')

video.release()

print("-------------------------------------")
print(results)

print(results[130])


def draw_border(img, top_left, bottom_right, color=(0, 255, 0), thickness=6, line_length_x=200, line_length_y=200):
    x1, y1 = top_left
    x2, y2 = bottom_right

    cv.line(img, (x1, y1), (x1, y1 + line_length_y), color, thickness)  # -- top-left
    cv.line(img, (x1, y1), (x1 + line_length_x, y1), color, thickness)

    cv.line(img, (x1, y2), (x1, y2 - line_length_y), color, thickness)  # -- bottom-left
    cv.line(img, (x1, y2), (x1 + line_length_x, y2), color, thickness)

    cv.line(img, (x2, y1), (x2 - line_length_x, y1), color, thickness)  # -- top-right
    cv.line(img, (x2, y1), (x2, y1 + line_length_y), color, thickness)

    cv.line(img, (x2, y2), (x2, y2 - line_length_y), color, thickness)  # -- bottom-right
    cv.line(img, (x2, y2), (x2 - line_length_x, y2), color, thickness)

    return img


video = cv.VideoCapture(r"C:\Users\bhanu\Downloads\pythonProject7-20241112T155347Z-001\pythonProject7\Automatic Number Plate Recognition (ANPR) _ Vehicle Number Plate Recognition (1).mp4")

# get video dims
frame_width = int(video.get(3))
frame_height = int(video.get(4))
size = (frame_width, frame_height)

# Define the codec and create VideoWriter object
fourcc = cv.VideoWriter_fourcc(*'XVID')
out = cv.VideoWriter(r"C:\Users\bhanu\Downloads\pythonProject7-20241112T155347Z-001\pythonProject7\output.avi", fourcc, 20.0, (frame_width, frame_height))

# reset video before you re-run cell below
frame_number = -1
video.set(cv.CAP_PROP_POS_FRAMES, 0)

ret = True

while ret:
    ret, frame = video.read()
    frame_number += 1
    if ret:
        # Check if the current frame number has data in the results dictionary
        if frame_number in results:
            # Retrieve frame-specific data
            frame_data = results[frame_number]

            for track_id, data in frame_data.items():
                # Draw car bounding box
                x1, y1, x2, y2 = data['car']['bbox']
                draw_border(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 12, line_length_x=200,
                            line_length_y=200)

                # Draw license plate bounding box
                x1, y1, x2, y2 = data['license_plate']['bbox']
                cv.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 6)

                # Display detected number on the frame
                license_number = data['license_plate']['number']
                (text_width, text_height), _ = cv.getTextSize(license_number, cv.FONT_HERSHEY_SIMPLEX, 2, 6)
                cv.putText(frame, license_number, (int((x2 + x1 - text_width) / 2), int(y1 - text_height)),
                           cv.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 6)

            # Write the processed frame to the output video
            out.write(frame)
        import cv2 as cv

        # Inside the frame processing loop
        while ret:
            ret, frame = video.read()
            frame_number += 1
            if ret:
                # Your existing code for vehicle and license plate detection
                detections = model.predict(frame)[0]  # Example for vehicle detection

                # Example: Draw bounding boxes and other processing here
                for detection in detections.boxes.data.tolist():
                    x1, y1, x2, y2, track_id, score = detection
                    plates = [[x1, y1, x2, y2, track_id, score]]
                    for bbox in plates:
                        # Example: Draw car bounding box
                        cv.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

                        # Example: Draw license plate bounding box (if applicable)
                        cv.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

                        # Example: Add license plate text (if detected)
                        cv.putText(frame, "License Plate", (int(x1), int(y1) - 10),
                                   cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Display the frame with bounding boxes and other visualizations
                cv.imshow("Frame", frame)  # Display the frame with detection results

                # Add waitKey() to make sure the window updates and stays open
                if cv.waitKey(1) & 0xFF == ord('q'):  # Close the window if 'q' is pressed
                    break

        # Close all OpenCV windows after the loop
        cv.destroyAllWindows()

        # After the loop, you can continue writing the output to video or CSV, etc.

        frame = cv.resize(frame, (1280, 720))

out.release()
video.release()
print('hi')
