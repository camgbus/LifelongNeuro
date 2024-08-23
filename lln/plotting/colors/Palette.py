import spectra
from IPython.display import HTML

swatch_template = """
<div style="float: left;">
    <div style="width: 50px; height: 50px; background: {0};"></div>
    <div>{0}</div>
</div>
"""
swatch_outer = """
<div style='width: 500px; overflow: auto; font-size: 10px; 
    font-weight: bold; text-align: center; line-height: 1.5;'>{0}</div>
"""

def swatches(colors):
    hexes = (c.hexcode.upper() for c in colors)
    html =  swatch_outer.format("".join(map(swatch_template.format, hexes)))
    return HTML(html)

class Palette:
    def __init__(self, inflection_points, space="rgb"):
        """
        Initialize the Palette with a list of colors.

        Args:
        colors (list): A list of (float, color values).
        space (str): The colorspace to use for the palette. Default is "rgb". 
            Valid options are: ['cmy', 'cmyk', 'hsl', 'hsv', 'lab', 'lch', 'rgb', 'xyz']
        """
        self.inflection_points = inflection_points
        self.scale = spectra.scale([c for p, c in inflection_points]).domain([p for p, c in inflection_points])
        self.scale.colorspace(space)

    def swatch(self, n):
        return self.scale(n)
    
    def swatch_to_plt(self, n):
        c = self.scale(n)
        c = c.rgb + (0,)
        return c
    
    def visualize_palette(self, nr):
        return swatches(self.scale.range(nr))
    
    def visualize_color(self, n):
        return swatches([self.scale(n)])