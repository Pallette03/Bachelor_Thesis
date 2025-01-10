import bpy # type: ignore
import bpy_extras # type: ignore
import os
import sys

dir = os.path.dirname(bpy.data.filepath)
dir = os.path.join(dir, 'Scripts')
if not dir in sys.path:
    sys.path.append(dir )

import util_functions

uf = util_functions.Util_functions()

render_images_folder = '//dataset/images/rgb'
annotations_folder = '//dataset/annotations'
hdri_folder = '//hdri'

uf.load_hdri_image(os.path.join(bpy.path.abspath(hdri_folder), 'blue_photo_studio_4k.hdr'))