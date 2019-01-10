import matplotlib.pyplot as plt
import csv
import numpy as np
from scipy import interpolate

def read_joint_names(filename):
    with open(filename, 'r') as f:
        org = f.read()
        joint_names = org.split(',')

    return joint_names

def read_csv(filename):

    joint_names = read_joint_names("joints.txt")

    r_shoulder_index = joint_names.index("RightShoulder") + 1
    l_shoulder_index = joint_names.index("LeftShoulder") + 1

    r_hand_index = joint_names.index("RightHand") + 1
    l_hand_index = joint_names.index("LeftHand") + 1

    r_forearm_index = joint_names.index("LeftForeArm") + 1
    l_forearm_index = joint_names.index("LeftForeArm") + 1

    x=[]
    y=[]
    total_sum = 0
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader, None)  # skip the headers
        for row in reader:
            x.append(float(row[0]) * 20)  # Scale the velocity
            next_val =  float(row[l_hand_index])   #row[r_shoulder_index]) + float(row[l_shoulder_index]) # float(row[-1]) #
            y.append(next_val)
            total_sum+=next_val

    return np.array(x), np.array(y) / total_sum

def barplot_annotate_brackets(num1, num2, data, center, height, yerr=None, dh=.05, barh=.05, fs=None, maxasterix=None):
    """
    Annotate barplot with p-values.

    :param num1: number of left bar to put bracket over
    :param num2: number of right bar to put bracket over
    :param data: string to write or number for generating asterixes
    :param center: centers of all bars (like plt.bar() input)
    :param height: heights of all bars (like plt.bar() input)
    :param yerr: yerrs of all bars (like plt.bar() input)
    :param dh: height offset over bar / bar + yerr in axes coordinates (0 to 1)
    :param barh: bar height in axes coordinates (0 to 1)
    :param fs: font size
    :param maxasterix: maximum number of asterixes to write (for very small p-values)
    """

    text = data

    lx, ly = center[num1], height[num1]
    rx, ry = center[num2], height[num2]

    if yerr:
        ly += yerr[num1]
        ry += yerr[num2]

    ax_y0, ax_y1 = plt.gca().get_ylim()
    dh *= (ax_y1 - ax_y0)
    barh *= (ax_y1 - ax_y0)

    y = max(ly, ry) + dh

    barx = [lx, lx, rx, rx]
    bary = [y, y+barh, y+barh, y]
    mid = ((lx+rx)/2, y+barh)

    plt.plot(barx, bary, c='black')

    kwargs = dict(ha='center', va='bottom')
    if fs is not None:
        kwargs['fontsize'] = fs

    plt.text(*mid, text, **kwargs)

# Custom function to draw the diff bars


plt.rcParams.update({'font.size': 35})


type = "vel"

original_filename = 'result/original/hmd_'+type+'_0.05.csv'

x,original = read_csv(original_filename)


mfcc_filename = 'result/MFCC/hmd_'+type+'_0.05.csv'
_, mfcc = read_csv(mfcc_filename)

spectr_filename = 'result/Spectr/hmd_'+type+'_0.05.csv'
_, spectr = read_csv(spectr_filename)


"""baseline = [4.160, 4.940, 4.319]
encoder = np.array([4.798, 4.830, 4.151])
x = np.arange(3)

errorB = [0.93, 1, 1.43]
errorE = [0.89, 0.98, 1.43]

plt.bar(x, baseline, yerr=errorB, label='Baseline' ,width = 0.25, hatch='/')
plt.bar(x+0.25, encoder, label = 'Proposed' ,width = 0.25)

special_x = np.array([0, 0.25, 0.5, 0.75])

barplot_annotate_brackets(0, 1, "p < 0.002", special_x, encoder)
barplot_annotate_brackets(1, 2, "p = 0.32", special_x+0.75, encoder)
barplot_annotate_brackets(1, 2, "p = 0.13", special_x+1.75, encoder)

plt.xticks(np.arange(3),('Naturalness', 'Time-consistency', 'Semantic-consistency'))

plt.legend(bbox_to_anchor=(0.2, 0.99), ncol=2)

plt.ylim(top=6)"""





#plt.plot(x,original, label='Ground Truth',linewidth=7.0)#,width = 0.25)
plt.plot(x,original,linewidth=7.0, label='Ground Truth')
plt.plot(x,mfcc , label='MFCC',linewidth=7.0)
plt.plot(x,spectr , label='Spectrogram',linewidth=7.0)


plt.xlabel("Velocity (cm/$s$)")
plt.ylabel('Relative frequency')
#plt.title('Average Velocity Histogram')



plt.xticks(np.arange(20))#, ('Tom', 'Dick', 'Harry', 'Sally', 'Sue'))


leg = plt.legend()



plt.show()