import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
from mpl_toolkits.mplot3d import axes3d, Axes3D
from scipy.spatial import distance
from pyquaternion import Quaternion
from ikpy import geometry_utils

# author Pieter Wolfert

class ModelSkeletons:
    def __init__(self, data_path):
        skeletons = np.load(data_path)

        # Just for now
        skeletons = skeletons[:2000]
        skeletons = np.reshape(skeletons, (-1,8, 3))

        self.skeletons = skeletons# [self.preprocessSkeletonT(x) for x in skeletons]

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
        self.ax = plt.axes(xlim=(-150, 150), ylim=(-20, 220))  # plt.axes(xlim=(-1.5, 1.5), ylim=(-1.2, 1.2))
        #
        self.ax.axis('off')
        self.line_one = self.ax.plot([], [], lw=2, c='b', marker="s")[0]
        self.line_two = self.ax.plot([], [], lw=2, c='b', marker="s")[0]
        self.line_three = self.ax.plot([], [], lw=2, c='b', marker="s")[0]

    def initLines(self):
        """Initialize the lines for plotting the limbs."""
        self.line_one.set_data([], [])
        self.line_two.set_data([], [])
        self.line_three.set_data([], [])
        return self.line_one, self.line_two, self.line_three

    def animateframe(self, skeleton):
        """Animate frame plot with two arms."""
        self.line_one.set_data(skeleton[:, 0][0:5], skeleton[:, 2][0:5])
        x = [skeleton[:, 0][1], skeleton[:, 0][5],
             skeleton[:, 0][6], skeleton[:, 0][7]]
        z = [skeleton[:, 2][1], skeleton[:, 2][5],
             skeleton[:, 2][6], skeleton[:, 2][7]]
        self.line_two.set_data(x, z)
        self.line_three.set_data(skeleton[:, 0][0:2], skeleton[:, 2][0:2])
        return self.line_one, self.line_two, self.line_three

    def animate(self, frames_to_play, interval):
        """Return an animation object that can be saved as a video."""
        anim = animation.FuncAnimation(self.fig, self.animateframe,
                                       init_func=self.initLines,
                                       frames=frames_to_play,
                                       interval=interval, blit=True)
        return anim


def main():
    ml = ModelSkeletons("./bvh_read/test_data/Pros2/gesture2.npy") #Motion_30.npy")
    skeletons = ml.getSkeletons()
    ml.plotSkeletonT(skeletons[0])

    am = AnimateSkeletons()
    am.initLines()

    anim = am.animate(frames_to_play=skeletons, interval=17)
    anim.save('Gesture2_centered.mp4', writer='ffmpeg', fps=60)


if __name__ == '__main__':
    main()
