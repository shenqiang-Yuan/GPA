import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import seaborn
from sklearn.manifold import TSNE
import pickle
import math
import pandas as pd


extend_cnames = seaborn.xkcd_rgb
cnames = {
    'aliceblue':            '#F0F8FF',
    'antiquewhite':         '#FAEBD7',
    'aqua':                 '#00FFFF',
    'aquamarine':           '#7FFFD4',
    'azure':                '#F0FFFF',
    'beige':                '#F5F5DC',
    'bisque':               '#FFE4C4',
    'black':                '#000000',
    'blanchedalmond':       '#FFEBCD',
    'blue':                 '#0000FF',
    'blueviolet':           '#8A2BE2',
    'brown':                '#A52A2A',
    'burlywood':            '#DEB887',
    'cadetblue':            '#5F9EA0',
    'chartreuse':           '#7FFF00',
    'chocolate':            '#D2691E',
    'coral':                '#FF7F50',
    'cornflowerblue':       '#6495ED',
    'cornsilk':             '#FFF8DC',
    'crimson':              '#DC143C',
    'cyan':                 '#00FFFF',
    'darkblue':             '#00008B',
    'darkcyan':             '#008B8B',
    'darkgoldenrod':        '#B8860B',
    'darkgray':             '#A9A9A9',
    'darkgreen':            '#006400',
    'darkkhaki':            '#BDB76B',
    'darkmagenta':          '#8B008B',
    'darkolivegreen':       '#556B2F',
    'darkorange':           '#FF8C00',
    'darkorchid':           '#9932CC',
    'darkred':              '#8B0000',
    'darksalmon':           '#E9967A',
    'darkseagreen':         '#8FBC8F',
    'darkslateblue':        '#483D8B',
    'darkslategray':        '#2F4F4F',
    'darkturquoise':        '#00CED1',
    'darkviolet':           '#9400D3',
    'deeppink':             '#FF1493',
    'deepskyblue':          '#00BFFF',
    'dimgray':              '#696969',
    'dodgerblue':           '#1E90FF',
    'firebrick':            '#B22222',
    'floralwhite':          '#FFFAF0',
    'forestgreen':          '#228B22',
    'fuchsia':              '#FF00FF',
    'gainsboro':            '#DCDCDC',
    'ghostwhite':           '#F8F8FF',
    'gold':                 '#FFD700',
    'goldenrod':            '#DAA520',
    'gray':                 '#808080',
    'green':                '#008000',
    'greenyellow':          '#ADFF2F',
    'honeydew':             '#F0FFF0',
    'hotpink':              '#FF69B4',
    'indianred':            '#CD5C5C',
    'indigo':               '#4B0082',
    'ivory':                '#FFFFF0',
    'khaki':                '#F0E68C',
    'lavender':             '#E6E6FA',
    'lavenderblush':        '#FFF0F5',
    'lawngreen':            '#7CFC00',
    'lemonchiffon':         '#FFFACD',
    'lightblue':            '#ADD8E6',
    'lightcoral':           '#F08080',
    'lightcyan':            '#E0FFFF',
    'lightgoldenrodyellow': '#FAFAD2',
    'lightgreen':           '#90EE90',
    'lightgray':            '#D3D3D3',
    'lightpink':            '#FFB6C1',
    'lightsalmon':          '#FFA07A',
    'lightseagreen':        '#20B2AA',
    'lightskyblue':         '#87CEFA',
    'lightslategray':       '#778899',
    'lightsteelblue':       '#B0C4DE',
    'lightyellow':          '#FFFFE0',
    'lime':                 '#00FF00',
    'limegreen':            '#32CD32',
    'linen':                '#FAF0E6',
    'magenta':              '#FF00FF',
    'maroon':               '#800000',
    'mediumaquamarine':     '#66CDAA',
    'mediumblue':           '#0000CD',
    'mediumorchid':         '#BA55D3',
    'mediumpurple':         '#9370DB',
    'mediumseagreen':       '#3CB371',
    'mediumslateblue':      '#7B68EE',
    'mediumspringgreen':    '#00FA9A',
    'mediumturquoise':      '#48D1CC',
    'mediumvioletred':      '#C71585',
    'midnightblue':         '#191970',
    'mintcream':            '#F5FFFA',
    'mistyrose':            '#FFE4E1',
    'moccasin':             '#FFE4B5',
    'navajowhite':          '#FFDEAD',
    'navy':                 '#000080',
    'oldlace':              '#FDF5E6',
    'olive':                '#808000',
    'olivedrab':            '#6B8E23',
    'orange':               '#FFA500',
    'orangered':            '#FF4500',
    'orchid':               '#DA70D6',
    'palegoldenrod':        '#EEE8AA',
    'palegreen':            '#98FB98',
    'paleturquoise':        '#AFEEEE',
    'palevioletred':        '#DB7093',
    'papayawhip':           '#FFEFD5',
    'peachpuff':            '#FFDAB9',
    'peru':                 '#CD853F',
    'pink':                 '#FFC0CB',
    'plum':                 '#DDA0DD',
    'powderblue':           '#B0E0E6',
    'purple':               '#800080',
    'red':                  '#FF0000',
    'rosybrown':            '#BC8F8F',
    'royalblue':            '#4169E1',
    'saddlebrown':          '#8B4513',
    'salmon':               '#FA8072',
    'sandybrown':           '#FAA460',
    'seagreen':             '#2E8B57',
    'seashell':             '#FFF5EE',
    'sienna':               '#A0522D',
    'silver':               '#C0C0C0',
    'skyblue':              '#87CEEB',
    'slateblue':            '#6A5ACD',
    'slategray':            '#708090',
    'snow':                 '#FFFAFA',
    'springgreen':          '#00FF7F',
    'steelblue':            '#4682B4',
    'tan':                  '#D2B48C',
    'teal':                 '#008080',
    'thistle':              '#D8BFD8',
    'tomato':               '#FF6347',
    'turquoise':            '#40E0D0',
    'violet':               '#EE82EE',
    'wheat':                '#F5DEB3',
    'white':                '#FFFFFF',
    'whitesmoke':           '#F5F5F5',
    'yellow':               '#FFFF00',
    'yellowgreen':          '#9ACD32'
}

ucf_class = ['applyeyemakeup',
             'applylipstick',
             'archery',
             'babycrawling',
             'balancebeam',
             'bandmarching',
             'baseballpitch',
             'basketball',
             'basketballdunk',
             'benchpress',
             'biking',
             'billiards',
             'blowdryhair',
             'blowingcandles',
             'bodyweightsquats',
             'bowling',
             'boxingpunchingbag',
             'boxingspeedbag',
             'breaststroke',
             'brushingteeth',
             'cleanandjerk',
             'cliffdiving',
             'cricketbowling',
             'cricketshot',
             'cuttinginkitchen',
             'diving',
             'drumming',
             'fencing',
             'fieldhockeypenalty',
             'floorgymnastics',
             'frisbeecatch',
             'frontcrawl',
             'golfswing',
             'haircut',
             'hammering',
             'hammerthrow',
             'handstandpushups',
             'handstandwalking',
             'headmassage',
             'highjump',
             'horserace',
             'horseriding',
             'hulahoop',
             'icedancing',
             'javelinthrow',
             'jugglingballs',
             'jumpingjack',
             'jumprope',
             'kayaking',
             'knitting',
             'longjump',
             'lunges',
             'militaryparade',
             'mixing',
             'moppingfloor',
             'nunchucks',
             'parallelbars',
             'pizzatossing',
             'playingcello',
             'playingdaf',
             'playingdhol',
             'playingflute',
             'playingguitar',
             'playingpiano',
             'playingsitar',
             'playingtabla',
             'playingviolin',
             'polevault',
             'pommelhorse',
             'pullups',
             'punch',
             'pushups',
             'rafting',
             'rockclimbingindoor',
             'ropeclimbing',
             'rowing',
             'salsaspin',
             'shavingbeard',
             'shotput',
             'skateboarding',
             'skiing',
             'skijet',
             'skydiving',
             'soccerjuggling',
             'soccerpenalty',
             'stillrings',
             'sumowrestling',
             'surfing',
             'swing',
             'tabletennisshot',
             'taichi',
             'tennisswing',
             'throwdiscus',
             'trampolinejumping',
             'typing',
             'unevenbars',
             'volleyballspiking',
             'walkingwithdog',
             'wallpushups',
             'writingonboard',
             'yoyo']
'''
path = '/home/pr606/python_vir/yuan/caffe2-R2plus1D/data/users/trandu/datasets/ucf101_test_01_video_id_dense_l32_1/'
with open(path+'features.pickle','rb') as handle:
    features = pickle.load(handle)
feat=features['final_avg']
labels = pd.read_csv(path+'info_in_lmdb.csv',delimiter=',',header=0)['label']
print(feat.shape)
print(labels.shape)
data = np.squeeze(feat[0:5000])
class_info=labels[0:5000]
'''

path = '/home/pr606/YUAN/history/tf-R2plus1D/resnet-GPA-10-features.pickle'
with open(path,'rb') as handle:
    features = pickle.load(handle)

feat = features['final'][:9000]
map = features['mapping'][:9000]
labels = []
for lab in map:
    labels.append(lab.split('_')[1].lower())
# labels = pd.read_csv('/home/pr606/python_vir/yuan/EXTRA_DATA/kinstics_part_val.csv',delimiter=',',header=0)['label']
data = np.squeeze(feat)
num_class = len(ucf_class)
color_list = list(extend_cnames.values())[:num_class]
class_info = labels

tsne = TSNE(n_components=2,n_iter=2000,perplexity=45)
embedding = tsne.fit_transform(data)
print(embedding.shape)

fig3 = plt.figure()
ax3 = fig3.add_subplot(111)
#ax4 = fig3.add_subplot(111)

for index in range(embedding.shape[0]):
    ax3.scatter(embedding[index,0],embedding[index,1],c=color_list[ucf_class.index(class_info[index])],s=5,marker='o')
"""
for index in range(result.shape[0]):
    ax4.scatter(result[index,0],result[index,1],c=color_list[class_info[index]],s=10)
"""

ax3.grid(False)
ax3.axis('off')
ax3.set_xlabel('resnet-gpa-18')
"""
ax4.grid(False)
ax4.axis('off')
ax4.set_xlabel('after training')
"""
#plt.legend(bbox_to_anchor=(1.05, 0), loc='lower left', borderaxespad=0,ncol=5)
plt.show()
fig3.savefig('resnet10-gpa.png',dpi=600,bbox_inches='tight')


