import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
from mpl_toolkits.mplot3d import axes3d, Axes3D
from scipy.spatial import distance
from pyquaternion import Quaternion

import argparse

# authors: Pieter Wolfert, Taras Kucherenko

class ModelSkeletons:
    def __init__(self, data_path , start_t, end_t):
        skeletons = np.loadtxt(data_path)

        # Just for now
        skeletons = skeletons[start_t*60:end_t*60] # sec -> min
        skeletons = np.reshape(skeletons, (-1,46, 3))

        self.skeletons = skeletons
        # [self.preprocessSkeletonT(x) for x in skeletons]

    def getSkeletons(self):
        return self.skeletons

    def preprocessSkeletonT(self, skelet):

        """Normalize and zero orient skeleton file by Taras"""
        x, y, z = skelet[:, 0], skelet[:, 1], skelet[:, 2]
        mid_x, mid_y, mid_z = (x[5] + x[2]) / 2, (y[5] + y[2]) / 2,\
                              (z[5] + z[2]) / 2
        # make neck the middle point of the shoulders
        skelet[1] = np.array([mid_x, mid_y, mid_z])
        # first zero orient for the midpoint
        x, y, z = x - mid_x, y - mid_y, z - mid_z

        # distance between the shoulders for normalization
        resize = distance.euclidean([x[2], y[2], z[2]], [x[5], y[5], z[5]])
        x, y, z = x / resize, y / resize, z / resize

        # calculate angle betweeen two shoulder positions
        shoulder_right = [x[2], y[2], z[2]]
        shoulder_left = [x[5], y[5], z[5]]
        angle_between = np.array(shoulder_left) - np.array(shoulder_right)

        angle = self.angle_between_3d(angle_between,  [-1.0, 0.0, 0.0])
        quaternion = Quaternion(axis=[0, 0, 1], angle=angle)  # radian

        skelet = np.zeros((8, 3))
        skelet[:, 0] = x
        skelet[:, 1] = y
        skelet[:, 2] = z

        """

        for i, item in enumerate(skelet):
            skelet[i] = quaternion.rotate(skelet[i])

        # horizontal alignment
        x = skelet[5][0] - skelet[2][0]
        y = skelet[5][1] - skelet[2][1]
        z = skelet[5][2] - skelet[2][2]
        horizontal_angle = self.angle_between_3d([x, y, z],  [-1.0, 0.0, 0.0])
        quaternion_hz = Quaternion(axis=[0, -1, 0], angle=horizontal_angle)

        for i, item in enumerate(skelet):
            skelet[i] = quaternion_hz.rotate(skelet[i])
            
        """

        skelet = np.zeros((8, 3))
        skelet[:, 0] = x
        skelet[:, 1] = y
        skelet[:, 2] = z

        # new head
        skelet[0] = np.array([0.0, 0.0, 0.4])

        return skelet

    def angle_between_3d(self, v1, v2):
        """ Returns the angle in radians between vectors 'v1' and 'v2'::
                >>> angle_between((1, 0, 0), (0, 1, 0))
                1.5707963267948966
                >>> angle_between((1, 0, 0), (1, 0, 0))
                0.0
                >>> angle_between((1, 0, 0), (-1, 0, 0))
                3.141592653589793
        """
        v1_u = self.unit_vector(v1)
        v2_u = self.unit_vector(v2)
        return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

    def unit_vector(self, vector):
        """ Returns the unit vector of the vector.  """
        return vector / np.linalg.norm(vector)

    def plotSkeletonT(self, skelet):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.plot(skelet[:, 0][0:2], skelet[:, 1][0:2], skelet[:, 2][0:2])
        ax.plot(skelet[:, 0][1:5], skelet[:, 1][1:5], skelet[:, 2][1:5])
        ax.plot(skelet[:, 0][5:8], skelet[:, 1][5:8], skelet[:, 2][5:8])
        ax.plot([skelet[:, 0][1], skelet[:, 0][5]],
                [skelet[:, 1][1], skelet[:, 1][5]],
                [skelet[:, 2][1], skelet[:, 2][5]])
        ax.scatter(skelet[:, 0], skelet[:, 1], skelet[:, 2])

class AnimateSkeletons:
    """Animate plots for drawing Taras' skeleton sequences in 2D."""

    def __init__(self):
        """Instantiate an object to visualize the generated poses."""
        self.fig = plt.figure()
        self.ax = plt.axes(xlim=(-60, 60), ylim=(-20, 100))  # plt.axes(xlim=(-1.5, 1.5), ylim=(-1.2, 1.2))
        #
        self.ax.axis('off')
        self.line_one = self.ax.plot([], [], lw=5, c='b', marker="s")[0]
        self.line_two = self.ax.plot([], [], lw=5, c='b', marker="s")[0]
        self.line_three = self.ax.plot([], [], lw=5, c='b', marker="s")[0]

        self.r_fing_1 = self.ax.plot([], [], lw=2, markersize=2, c='b', marker="s")[0]
        self.r_fing_2 = self.ax.plot([], [], lw=2, markersize=2, c='b', marker="s")[0]
        self.r_fing_3 = self.ax.plot([], [], lw=2, markersize=2, c='b', marker="s")[0]
        self.r_fing_4 = self.ax.plot([], [], lw=2, markersize=2, c='b', marker="s")[0]
        self.r_fing_5 = self.ax.plot([], [], lw=2, markersize=2, c='b', marker="s")[0]

        self.l_fing_1 = self.ax.plot([], [], lw=2, markersize=2, c='b', marker="s")[0]
        self.l_fing_2 = self.ax.plot([], [], lw=2, markersize=2, c='b', marker="s")[0]
        self.l_fing_3 = self.ax.plot([], [], lw=2, markersize=2, c='b', marker="s")[0]
        self.l_fing_4 = self.ax.plot([], [], lw=2, markersize=2, c='b', marker="s")[0]
        self.l_fing_5 = self.ax.plot([], [], lw=2, markersize=2, c='b', marker="s")[0]

    def initLines(self):
        """Initialize the lines for plotting the limbs."""
        self.line_one.set_data([], [])
        self.line_two.set_data([], [])
        self.line_three.set_data([], [])

        self.r_fing_1.set_data([], [])
        self.r_fing_2.set_data([], [])
        self.r_fing_3.set_data([], [])
        self.r_fing_4.set_data([], [])
        self.r_fing_5.set_data([], [])

        self.l_fing_1.set_data([], [])
        self.l_fing_2.set_data([], [])
        self.l_fing_3.set_data([], [])
        self.l_fing_4.set_data([], [])
        self.l_fing_5.set_data([], [])

        return self.line_one, self.line_two, self.line_three, self.r_fing_1, self.r_fing_2, self.r_fing_3

    def animateframe(self, skeleton):
        """Animate frame plot with two arms."""

        """
        ['Hips', 'Spine', 'Spine1', 'Spine2', 'Spine3', 'Neck', 'Neck1', 'Head',  # Head and spine
            'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand', 'RightHandThumb1',
            'RightHandThumb2', 'RightHandThumb3', 'RightHandIndex1', 'RightHandIndex2', 'RightHandIndex3',
            'RightHandMiddle1', 'RightHandMiddle2', 'RightHandMiddle3', 'RightHandRing1', 'RightHandRing2',
            'RightHandRing3', 'RightHandPinky1', 'RightHandPinky2', 'RightHandPinky3',  # Right hand
            'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand', 'LeftHandThumb1',
            'LeftHandThumb2', 'LeftHandThumb3', 'LeftHandIndex1', 'LeftHandIndex2', 'LeftHandIndex3',
            'LeftHandMiddle1', 'LeftHandMiddle2', 'LeftHandMiddle3', 'LeftHandRing1', 'LeftHandRing2',
            'LeftHandRing3', 'LeftHandPinky1', 'LeftHandPinky2', 'LeftHandPinky3',  # left hand
        ]
        """
        # Torso
        self.line_one.set_data(skeleton[:, 0][0:8], skeleton[:, 2][0:8])
        # Right arm
        x = np.concatenate(([skeleton[:, 0][4]], skeleton[:, 0][9:12]))
        z = np.concatenate(([skeleton[:, 2][4]], skeleton[:, 2][9:12]))
        self.line_two.set_data(x,z)
        # Left arm
        x = np.concatenate(([skeleton[:, 0][4]], skeleton[:, 0][28:31]))
        z = np.concatenate(([skeleton[:, 2][4]], skeleton[:, 2][28:31]))
        self.line_three.set_data(x,z)

        # Right finger 1
        self.r_fing_1.set_data(skeleton[:, 0][11:15], skeleton[:, 2][11:15])
        # Right finger 2
        x = np.concatenate(([skeleton[:, 0][11]], skeleton[:, 0][15:18]))
        z = np.concatenate(([skeleton[:, 2][11]], skeleton[:, 2][15:18]))
        self.r_fing_2.set_data(x, z)
        # Right finger 3
        x = np.concatenate(([skeleton[:, 0][11]], skeleton[:, 0][18:21]))
        z = np.concatenate(([skeleton[:, 2][11]], skeleton[:, 2][18:21]))
        self.r_fing_3.set_data(x, z)
        # Right finger 4
        x = np.concatenate(([skeleton[:, 0][11]], skeleton[:, 0][21:24]))
        z = np.concatenate(([skeleton[:, 2][11]], skeleton[:, 2][21:24]))
        self.r_fing_4.set_data(x, z)
        # Right finger 5
        x = np.concatenate(([skeleton[:, 0][11]], skeleton[:, 0][24:27]))
        z = np.concatenate(([skeleton[:, 2][11]], skeleton[:, 2][24:27]))
        self.r_fing_5.set_data(x, z)

        # Left finger 1
        self.l_fing_1.set_data(skeleton[:, 0][30:34], skeleton[:, 2][30:34])
        # Right finger 2
        x = np.concatenate(([skeleton[:, 0][30]], skeleton[:, 0][34:37]))
        z = np.concatenate(([skeleton[:, 2][30]], skeleton[:, 2][34:37]))
        self.l_fing_2.set_data(x, z)
        # Right finger 3
        x = np.concatenate(([skeleton[:, 0][30]], skeleton[:, 0][37:40]))
        z = np.concatenate(([skeleton[:, 2][30]], skeleton[:, 2][37:40]))
        self.l_fing_3.set_data(x, z)
        # Right finger 4
        x = np.concatenate(([skeleton[:, 0][30]], skeleton[:, 0][40:43]))
        z = np.concatenate(([skeleton[:, 2][30]], skeleton[:, 2][40:43]))
        self.l_fing_4.set_data(x, z)
        # Right finger 5
        x = np.concatenate(([skeleton[:, 0][30]], skeleton[:, 0][43:46]))
        z = np.concatenate(([skeleton[:, 2][30]], skeleton[:, 2][43:46]))
        self.l_fing_5.set_data(x, z)

        return self.line_one, self.line_two, self.line_three, self.r_fing_1, self.r_fing_2,\
               self.r_fing_3, self.r_fing_4, self.r_fing_5, self.l_fing_1, self.l_fing_2,\
               self.l_fing_3, self.l_fing_4, self.l_fing_5

    def animate(self, frames_to_play):
        """Return an animation object that can be saved as a video."""
        anim = animation.FuncAnimation(self.fig, self.animateframe,
                                       init_func=self.initLines,
                                       frames=frames_to_play, blit=True)
        return anim


def main():

    # Parse command line params

    parser = argparse.ArgumentParser(
        description='Visualize 3d coords. seq. into a video file')
    parser.add_argument('--input', '-d', default='../evaluation/data/original/gesture1.txt',
                        help='Path to the input file with the motion')
    parser.add_argument('--out', '-o', default='Gesture1_original.mp4',
                        help='Path to the output file with the video')
    parser.add_argument('--start', '-s', default=10, type=int,
                        help='Start time (in sec)')
    parser.add_argument('--end', '-e', default=20, type=int,
                        help='End time (in sec)')
    args = parser.parse_args()

    ml = ModelSkeletons(args.input, args.start, args.end)
    skeletons = ml.getSkeletons()
    #ml.plotSkeletonT(skeletons[0])

    am = AnimateSkeletons()
    am.initLines()

    anim = am.animate(frames_to_play=skeletons)
    anim.save(args.out, writer='ffmpeg', fps=60)


if __name__ == '__main__':
    main()
