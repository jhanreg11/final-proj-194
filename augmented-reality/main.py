import cv2
import matplotlib.pyplot as plt
import numpy as np


def cube_in_video(video, pts_by_frame, pts3d):
    """
    Place 3d cube into video at origin.
    :param video: np.array, shape - (frames, h, w, c)
    :param pts_by_frame: np.array, 2d points, shape - (frames, n_points, 2)
    :param pts3d: np.array, 3d world coords, shape - (n_points, 3)
    :return: np.array, shape - (frames, h, w, c)
    """
    cube_pts = np.float32([[0, 0, 0, 1], [0, 1, 0, 1], [1, 1, 0, 1], [1, 0, 0, 1],
                           [0, 0, 1, 1], [0, 1, 1, 1], [1, 1, 1, 1], [1, 0, 1, 1]])
    frames, _, _, _ = video.shape
    video = video.copy()

    for i in range(frames):
        proj = calc_proj(pts_by_frame[i], pts3d)
        cube2d = proj.dot(cube_pts.T)
        cube2d = (cube2d[:2] / cube2d[-1]).T
        video[i] = draw_cube(video[i], cube2d)

    return video


def show_detected_pts(video, pts):
    """
    Show detected points for each frame in the video.
    :param video: np.array, shape - (frames, h, w, c)
    :param pts: np.array, shape - (frames, n_points, 2)
    :return: np.array, shape - (frames, h, w, c)
    """
    video = video.copy()
    for i in range(video.shape[0]):
        for j in range(pts[i].shape[0]):
            video[i] = cv2.circle(
                video[i], (int(pts[i][j][0]), int(pts[i][j][1])),
                2, (0, 255, 0), -1)
    return video


def track_pts(vid, initial_pts):
    """
    :param vid: np.array, video to track pts for
        shape - (frames, h, w, c)
    :param initial_pts: list, hand-selected 2d keypoints for 1st frame [(x, y), ....]
    :return: np.array, 2d coords of each point for each frame, if unable to find for a certain frame then the point has x = -1000, y = -1000
        shape - (frames, n_points, 2)
    """
    frames, _, _, _ = vid.shape
    trackers = [cv2.TrackerCSRT_create()
                for _ in range(len(initial_pts))]
    [t.init(vid[0], (int(x - 4), int(y - 4), 8, 8))
     for t, (x, y) in zip(trackers, initial_pts)]
    pts_by_frame = [initial_pts]

    for i in range(1, frames):
        new_pts = []
        for t in trackers:
            ok, bbox = t.update(vid[i])
            if ok:
                new_pts.append((bbox[0] + 4, bbox[1] + 4))
            else:
                new_pts.append((-1000, -1000))
        pts_by_frame.append(new_pts)

    return np.array(pts_by_frame)


def calc_proj(pts_2d, pts_3d):
    """
    calculate projection matrix for 3d -> 2d using homogenous coords.
    """
    # pts_2d = np.roll(pts_2d, 1, axis=1)
    n_points = pts_3d.shape[0]
    last3_cols = np.zeros((n_points * 2, 3))
    last3_cols[::2] = pts_3d
    last3_cols[1::2] = pts_3d
    last3_cols *= -pts_2d.reshape((-1, 1))

    pts_3d = np.concatenate((pts_3d, np.ones((n_points, 1))), 1)
    a = np.zeros((n_points * 2, 8))
    a[::2, :4] = pts_3d
    a[1::2, 4:8] = pts_3d

    a = np.concatenate((a, last3_cols), 1)

    b = pts_2d.flatten()

    proj = np.linalg.inv(a.T.dot(a)).dot(a.T).dot(b)

    return np.concatenate((proj, [1])).reshape((3, 4))


def draw_cube(img, pts2d):
    """Draw cube for a single frame given the 2d coordinates necessary"""
    imgpts = np.int32(pts2d).reshape(-1, 2)

    img = cv2.drawContours(img, [imgpts[:4]], -1, (0, 255, 0), -3)

    for i, j in zip(range(4), range(4, 8)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), (255), 3)

    img = cv2.drawContours(img, [imgpts[4:]], -1, (0, 0, 255), 3)

    return img


def select_initial_pts(video):
    plt.imshow(video[0])
    pts2d = []
    pts3d = []
    pt3d = input('Enter 3d coords (-1 to quit x,y,z otherwise): ')
    while pt3d != 'q':
        pt2d = plt.ginput(1)
        pts3d.append([int(a) for a in pt3d.split(',')])
        pts2d.append(pt2d[0])
        pt3d = input('Enter 3d coords (q to quit; x,y,z otherwise): ')

    with open('initial_pts.csv', 'w') as file:
        [file.write(f'{x},{y},{x1},{y1},{z}\n')
         for (x, y), (x1, y1, z) in zip(pts2d, pts3d)]
    plt.clf()
    return np.array(pts2d), np.array(pts3d).astype(np.float32)


def get_frames(fp):
    cap = cv2.VideoCapture(fp)
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = 480
    frameHeight = 640

    buf = np.empty((frameCount, frameHeight, frameWidth, 3), np.dtype('uint8'))

    fc = 0
    ret = True

    while (fc < frameCount and ret):
        ret, frame = cap.read()
        buf[fc] = cv2.resize(frame, (frameWidth, frameHeight))
        fc += 1

    cap.release()

    return buf


def write_video(video, fp):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(fp, fourcc, 20.0, (video.shape[2], video.shape[1]))
    for i in range(video.shape[0]):
        out.write(video[i])
    out.release()


if __name__ == '__main__':
    video = get_frames('IMG_6424.mp4')

    if input('Enter initial pts & 3d locations (y/n)? ') == 'y':
        pts2d, pts3d = select_initial_pts(video)
    else:
        with open('initial_pts.csv', 'r') as f:
            pts = [l.split(',') for l in f.readlines()]
            pts2d = np.array([(float(l[0]), float(l[1])) for l in pts])
            pts3d = np.array([(int(l[2]), int(l[3]), int(l[4]))
                             for l in pts]).astype(np.float32)

    all_pts = track_pts(video, pts2d)
    write_video(show_detected_pts(video, all_pts), 'detected_pts.avi')

    write_video(cube_in_video(video, all_pts, pts3d), 'ar.avi')
