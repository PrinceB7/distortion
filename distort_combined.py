import os
import cv2
import csv
import math
import numpy as np
from tqdm import tqdm
from datetime import datetime

import mediapipe as mp
mp_face_mesh = mp.solutions.face_mesh

def angle_between_three_points(pointA, pointB, pointC):
    x1x2s = math.pow((pointA[0] - pointB[0]),2)
    x1x3s = math.pow((pointA[0] - pointC[0]),2)
    x2x3s = math.pow((pointB[0] - pointC[0]),2)
    
    y1y2s = math.pow((pointA[1] - pointB[1]),2)
    y1y3s = math.pow((pointA[1] - pointC[1]),2)
    y2y3s = math.pow((pointB[1] - pointC[1]),2)

    cosine_angle = np.arccos((x1x2s + y1y2s + x2x3s + y2y3s - x1x3s - y1y3s)/(2*math.sqrt(x1x2s + y1y2s)*math.sqrt(x2x3s + y2y3s)))
    return np.degrees(cosine_angle)


def calculate_angles(landmark):
    angles = {}
    point_triples = (
        (468, 4, 473), # a1
        (468, 5, 473), # a2
        (468, 10, 473), # a3
        (468, 151, 473), # a4
        (468, 9, 473), # a5
        (6, 50, 0), # a6
        (253, 123, 436), # a7
        (187, 5, 35), # a8
        (206, 5, 230),
        (138, 5, 71),
        (202, 199, 335),
        (34, 199, 372),
        (229, 4, 449),
        (229, 5, 449),
        (105, 4, 334),
        (105, 5, 334),

    )

    for i, (p1_idx, p_center_idx, p2_idx) in enumerate(point_triples):
        p_center_2d = [landmark[p_center_idx].x, landmark[p_center_idx].y]
        p1_2d = [landmark[p1_idx].x, landmark[p1_idx].y]
        p2_2d = [landmark[p2_idx].x, landmark[p2_idx].y]
        angle_2d = angle_between_three_points(p1_2d, p_center_2d, p2_2d)
        angles[f"a{i+1}"] = angle_2d

    return angles

def calculate_angle_variance(angles):
    angle_variances = {}
    for i in range(len(angles[0])):
        angls = [a[f"a{i+1}"] for a in angles]
        plot_var = round(np.var(angls), 4)
        angle_variances[f'a{i+1}'] = plot_var
    return angle_variances

def webcam(path=None, video=None):
    cap = cv2.VideoCapture(path+video)
    w, h = 480, 640
    cap.set(3, w)
    cap.set(4, h)

    times = []
    angles = []
    with mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as face_mesh:
        
        success, image = cap.read()
        while success:
            image = cv2.flip(image, 1)
            
            if not success:
                print("Failed to detect face mesh")
                continue
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = face_mesh.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:    
                    now = datetime.now()
                    times.append(now.strftime("%H:%M:%S"))
                    angle = calculate_angles(face_landmarks.landmark)
                    angles.append(angle)      
            success, image = cap.read()
        
    return calculate_angle_variance(angles)
    
if __name__ == '__main__':
    type_ = "real"
    path = f"data/{type_}/"
    csv_file = f"angle_variance_{type_}.csv"

    data = {'a1': 9.2157, 'a2': 11.6405, 'a3': 2.3516, 'a4': 2.8826, 'a5': 4.3642,
    'a6': 8.397, 'a7': 1.6447, 'a8': 4.0336, 'a9': 6.1826, 'a10': 12.359,
    'a11': 1.9375, 'a12': 6.8716, 'a13': 8.6, 'a14': 9.2862, 'a15': 2.2931,
    'a16': 2.8749
    }   

    with open(csv_file, 'w', newline='') as file:
        fieldnames = data.keys()
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for video in tqdm(os.listdir(path), desc="Processing videos"):
            angle_variance = webcam(path, video)
            writer.writerow(angle_variance)
