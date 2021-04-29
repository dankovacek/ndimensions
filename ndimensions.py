import numpy as np
import pandas as pd

from PIL import Image, ImageOps

from functools import reduce

from skimage.util import view_as_blocks

from bokeh.plotting import figure, show
from bokeh.layouts import row, column
from bokeh.models import Select, Slider, Div
from bokeh.models import ColumnDataSource
from bokeh.io import curdoc

from time import time

raw_img = Image.open('img/ada_lovelace.jpg')

raw_img = ImageOps.flip(raw_img)

img_gs = raw_img.convert('L')
img_col = raw_img.convert('RGBA')

original_dims = img_gs.size
n_pixels = original_dims[0] * original_dims[1]

pixelated_dims = (16, 16)
colour_dims = 4


def factors(n):    
    return set(reduce(list.__add__,
                ([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0)))

# find the factors of the original image dimensions
fx = sorted([e for e in factors(original_dims[1]) if (e != 1) and (e < original_dims[1] / 2)])
fy = sorted([e for e in factors(original_dims[0]) if e != 1 and (e < original_dims[0] / 2)])



def digitized_array(img_array, new_dims, colour_depth):
    """[summary]

    Args:
        img_array (np array): hxw array of greyscale values in [0,255] 
        new_dims (tuple): 2x2 tuple describing the "pixellated" dimensions
                        to reshape the image into.

    Returns:
        array: new array of greyscale values in original image resolution 
    """
    original_dims = img_array.shape
    nx, ny = new_dims
    dh = int(img_array.shape[1] / nx) # new number of rows
    dw = int(img_array.shape[0] / ny) # new number of cols
    print(f'Original image array shape {original_dims}')
    print(f'window shape = {dw} wide, {dh} high')

    blocks = view_as_blocks(img_array, (dw, dh))

    colours = np.array(set_color_array(colour_depth))

    for i in range(blocks.shape[0]):
        for j in range(blocks.shape[1]):
            mean_gs_value = int(np.mean(blocks[i, j]))
            nearest_value_idx = (np.abs(colours - mean_gs_value)).argmin()
            blocks[i, j] = nearest_value_idx

    return np.swapaxes(blocks, 1, 2).reshape(original_dims)


def set_color_array(colour_depth):
    spacing = int(255 / colour_depth)
    return [i * spacing for i in range(colour_depth)]



def update_block_vals(blocks, colours):
    for i in range(blocks.shape[0]):
        for j in range(blocks.shape[1]):
            rando_colour = np.random.choice(colours)
            blocks[i, j] = rando_colour
    return blocks


def rando_scrambo_array(img_array, new_dims, colour_depth):
    """[summary]

    Args:
        img_array (np array): hxw array of greyscale values in [0,255] 
        new_dims (tuple): 2x2 tuple describing the "pixellated" dimensions
                        to reshape the image into.

    Returns:
        array: new array of greyscale values in original image resolution 
    """
    t0 = time()
    original_dims = img_array.shape
    nx, ny = new_dims
    dh = int(img_array.shape[1] / nx) # new number of rows
    dw = int(img_array.shape[0] / ny) # new number of cols

    blocks = view_as_blocks(img_array, (dw, dh))

    colours = set_color_array(colour_depth)

    blocks_updated = update_block_vals(blocks, colours)

    out = np.swapaxes(blocks_updated, 1, 2).reshape(original_dims)
    t1 = time()
    tt = t1 - t0
    # print(f'rando time = {tt:.1f}')
    return out
      

def create_fig(title, fig_source, dims=None):    
    aspect = 1
    if dims:
        aspect = dims[1] / dims[0]
    
    h = 450
    w = int(h / aspect)
    p = figure(title=title,
        toolbar_location='above', match_aspect=True,
        width=w, height=h)
    p.x_range.range_padding = p.y_range.range_padding = 0

    p.image(image='d', source=fig_source, 
            x=0, y=0, dw=1, dh=aspect)

    p.xaxis.major_tick_line_color = None
    p.xaxis.minor_tick_line_color = None
    p.yaxis.major_tick_line_color = None
    p.yaxis.minor_tick_line_color = None
    p.xaxis.major_label_text_font_size = '0pt'
    p.yaxis.major_label_text_font_size = '0pt'
    p.outline_line_color = None
    p.axis.visible = False

    p.toolbar_location = None

    return p


# Image arrays
pixelated_array = digitized_array(np.array(img_gs), pixelated_dims, colour_dims)
rando_array = rando_scrambo_array(np.array(img_gs), pixelated_dims, colour_dims)

# Declare Sources for dynamic content
original_source = ColumnDataSource(data=dict(d=[np.array(img_gs)]))
pixelated_source = ColumnDataSource(data=dict(d=[pixelated_array]))
rando_source = ColumnDataSource(data=dict(d=[rando_array]))

# Figures 
original_fig = create_fig(f'Original {original_dims}', original_source, original_dims)
pixelated_fig = create_fig(f'"Correct" Downsampled Image', pixelated_source, original_dims)
rando_fig = create_fig(f'Random Scramble', rando_source, original_dims)


height_dropdown = Select(title="Height Sample: (px)", 
                        value=str(pixelated_dims[0]), options=[str(e) for e in fy])
width_dropdown = Select(title="Width Sample: (px)", 
                        value=str(pixelated_dims[1]), options=[str(e) for e in fx])

colour_depth_selector = Slider(title="Number of Colours",
                        value=colour_dims, step=1, start=2, end=8)

init_size = int(height_dropdown.value) * int(width_dropdown.value)

# Update UI

def update_system_info_text():
    image_frequency = 5 
    h = int(height_dropdown.value)
    w = int(width_dropdown.value)
    c = colour_depth_selector.value
    n_images = (int(h) * int(w))**c
    p_image = 1 / n_images
    ev_image = (1/image_frequency) / p_image
    if ev_image < 600:
        time_text = f'{ev_image:.0f} seconds'
    elif ev_image < 3600:
        time_text = f'{ev_image/60:.1f} minutes'
    elif ev_image < 86400:
        time_text = f'{ev_image/3600:.1f} hours'
    elif ev_image < 31536000:
        time_text = f'{ev_image/86400:.1f} days'
    else:
        time_text = f'{ev_image/31536000:.1e} years'

    updated_size = h * w

    system_description_div.text = f"""<p>Reducing the image resolution to <strong>{h} x {w}</strong> pixels, with <strong>{c}</strong> possible colours, the system can have <strong>{updated_size}<sup>{c}</sup></strong> possible states.  
    </p>
    <p>The "random scramble" at far right generates <strong>{image_frequency} random images per second</strong> at this reduced resolution.  At this rate, you can expect to see the "correct" image <strong>once every {time_text}.</strong>  
    </p>
    """
    if ev_image > 31536000:
        universe_age = 13.7E9
        age_factor = (ev_image/31536000) / universe_age
        if age_factor > 0.1:
            system_description_div.text += f"""
            <p style="color: red">{time_text} is <strong>~{age_factor:.1f}x the age of the universe.</strong><p>
            """

def redraw_pixels(attr, old, new):

    dh = int(height_dropdown.value)
    dw = int(width_dropdown.value)
    colour_depth = int(colour_depth_selector.value)

    img_array = np.array(img_gs) 

    # update pixelated image and gen new image
    pixelated_array = digitized_array(img_array, (dh, dw), colour_depth)

    pixelated_source.data = {'d': [pixelated_array]}
    update_system_info_text()


def update_rando_fig():

    t0 = time()

    dh = int(height_dropdown.value)
    dw = int(width_dropdown.value)
    colour_depth = int(colour_depth_selector.value)

    # update rando image resolution and gen new image
    rando_array = rando_scrambo_array(np.array(img_gs), (dh, dw), colour_depth)

    rando_source.data = {'d': [rando_array]}
    update_system_info_text()
    


height_dropdown.on_change('value', redraw_pixels)
width_dropdown.on_change('value', redraw_pixels)
colour_depth_selector.on_change('value', redraw_pixels)

## Content sections

intro_div = Div(width=600, height=300,
                style={'font-size': '1.2em', 'font-family': 'Helvetica'})

intro_div.text = f"""
<h2>N-Dimensional Space</h2>
<h3>The Curse of Dimensionality and the Density of Structure of Nature</h3>

<p>This interactive application was inspired by <a href="https://en.wikipedia.org/wiki/Richard_Hamming">Richard Hamming's</a> <em>"Art of Doing Science and Engineering"</em></p>
<p>The <a href="https://blogs.scientificamerican.com/observations/ada-lovelace-day-honors-the-first-computer-programmer/">iconic image</a> below at left is {original_dims[0]} pixels wide by {original_dims[1]} pixels high in the original file.  Since the image is in greyscale, each pixel is represented by an integer value in the range [0, 255].</p> 

<p>If the original image were represented as a system, there are ({original_dims[0]}x{original_dims[1]})<sup>256</sup> possible configurations (states).  In other words, If each pixel can take one of 256 values, and there are {original_dims[0]} x {original_dims[1]} = {n_pixels:,} pixels, the number of possible <em>unique</em> images is {n_pixels:,}<sup>256</sup>, or roughly <strong>10<sup>262</sup></strong>.  As a hopelessly inadequate basis of comparison, the <a href="https://physics.stackexchange.com/questions/47941/dumbed-down-explanation-how-scientists-know-the-number-of-atoms-in-the-universe">cosmological estimate of the number of <strong>atoms in the universe</a> is 10<sup>80</sup></strong>.  The set of all images of this size that humans can recognize is <strong>effectively zero</strong>.  Virtually every arrangement is unrecognizable to the human eye.  </p>

<p>So what is it about organisms that make pattern recognition not just possible, but common?</p>

"""

system_description_div = Div(width=350, height=400,
                    style={'font-size': '1.1em', 
                    'font-family': 'Helvetica',
                    'padding': '5px'})


update_system_info_text()


references_div = Div(width=600, height=400,
                    style={'font-size': '1.2em', 'font-family': 'Helvetica'})

references_div.text = """
<br>
<h2>References</h2>
<ol>
<li>Hamming, Richard R. <em>Art of Doing Science and Engineering: Learning to Learn.</em> CRC Press LLC, Lisse, 1997;1996;.</li>
</ol>

"""

selections = column(height_dropdown, width_dropdown, 
                    colour_depth_selector, system_description_div)

content_row = row(original_fig, selections, pixelated_fig, rando_fig)

layout = column(intro_div, content_row, references_div)

curdoc().add_periodic_callback(update_rando_fig, 200)

curdoc().add_root(layout)
curdoc().title = "N-Dimensional Space"

