import os
from PIL import Image

"""Convert all pdf images in docs/paper to eps format"""

figs = os.listdir('docs/paper')

for fig in figs:
    im = Image.open('docs/paper/' + fig)
    im.save('docs/paper/' + fig[:-4] + '.eps')